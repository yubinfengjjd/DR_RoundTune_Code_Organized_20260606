"""Compute calibration-locked clinical deployment metrics for uncertainty triage."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

import uncertainty_compare as uncertainty_lib
from revision_utils import accuracy, entropy, macro_f1, qwk, read_json, read_rows, write_json, write_rows


DEFAULT_COVERAGES = (0.95, 0.90, 0.85, 0.80, 0.70)
DEFAULT_FIXED_POINTS = (0.90, 0.95)
DEFAULT_MISSED_STDR_TARGETS = (0.05, 0.025, 0.01, 0.0)
PRIMARY_METHOD = "Ensemble PredEntropy"


def _percentile_ci(values: Iterable[float], alpha: float = 0.05) -> Tuple[float, float]:
    vals = np.asarray(list(values), dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2.0)), float(np.quantile(vals, 1.0 - alpha / 2.0))


def _bootstrap_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, int(n), size=(int(n_boot), int(n)), endpoint=False, dtype=np.int64)


def _metric_ci(
    n: int,
    metric: Callable[[np.ndarray], float],
    *,
    n_boot: int,
    seed: int,
) -> Tuple[float, float]:
    values = [metric(index) for index in _bootstrap_indices(n, n_boot, seed)]
    return _percentile_ci(values)


def risk_coverage_curve(errors: np.ndarray, uncertainty: np.ndarray) -> Tuple[List[Dict[str, float]], float]:
    """Return cumulative selective risk from lowest to highest uncertainty and discrete AURC."""
    err = np.asarray(errors, dtype=np.float64).reshape(-1)
    unc = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    valid = np.isfinite(err) & np.isfinite(unc)
    err = err[valid]
    unc = unc[valid]
    if err.size == 0:
        return [], float("nan")
    order = np.argsort(unc, kind="mergesort")
    sorted_err = err[order]
    cumulative_risk = np.cumsum(sorted_err) / np.arange(1, len(sorted_err) + 1)
    coverages = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    rows = [
        {
            "n_kept": int(index + 1),
            "coverage": float(coverages[index]),
            "risk": float(cumulative_risk[index]),
            "uncertainty_threshold": float(unc[order[index]]),
        }
        for index in range(len(sorted_err))
    ]
    return rows, float(cumulative_risk.mean())


def _binary_metrics(y_true01: np.ndarray, score: np.ndarray, threshold: float) -> Dict[str, float]:
    y = np.asarray(y_true01, dtype=np.int64).reshape(-1)
    values = np.asarray(score, dtype=np.float64).reshape(-1)
    valid = np.isfinite(values) & ((y == 0) | (y == 1))
    y = y[valid]
    values = values[valid]
    pred = (values >= float(threshold)).astype(np.int64)
    tp = int(((y == 1) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    return {
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
    }


def threshold_at_fixed_operating_point(
    y_true01: np.ndarray,
    score: np.ndarray,
    constraint: str,
    target: float,
) -> Tuple[float, Dict[str, float]]:
    """Choose an STDR probability threshold on calibration data only."""
    y = np.asarray(y_true01, dtype=np.int64).reshape(-1)
    values = np.asarray(score, dtype=np.float64).reshape(-1)
    valid = np.isfinite(values) & ((y == 0) | (y == 1))
    y = y[valid]
    values = values[valid]
    candidates = np.unique(np.concatenate([values, np.array([-np.inf, np.inf])]))
    evaluated = [(float(tau), _binary_metrics(y, values, float(tau))) for tau in candidates]
    if constraint == "specificity":
        feasible = [(tau, metrics) for tau, metrics in evaluated if metrics["specificity"] >= float(target)]
        if not feasible:
            feasible = evaluated
        return max(feasible, key=lambda item: (item[1]["sensitivity"], item[1]["specificity"], -item[0]))
    if constraint == "sensitivity":
        feasible = [(tau, metrics) for tau, metrics in evaluated if metrics["sensitivity"] >= float(target)]
        if not feasible:
            feasible = evaluated
        return max(feasible, key=lambda item: (item[1]["specificity"], item[1]["sensitivity"], item[0]))
    raise ValueError("constraint must be 'specificity' or 'sensitivity'")


def _workload_metrics(
    y_true: np.ndarray,
    pred_grade: np.ndarray,
    uncertainty: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    pred = np.asarray(pred_grade, dtype=np.int64).reshape(-1)
    unc = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    valid = (y >= 0) & (pred >= 0) & np.isfinite(unc)
    y = y[valid]
    pred = pred[valid]
    unc = unc[valid]
    keep = unc <= float(threshold)
    stdr = y >= 3
    missed = keep & stdr & (pred < 3)
    n_stdr = int(stdr.sum())
    return {
        "auto_handled_fraction": float(keep.mean()) if keep.size else float("nan"),
        "manual_review_fraction": float(1.0 - keep.mean()) if keep.size else float("nan"),
        "missed_stdr_rate": float(missed.sum() / n_stdr) if n_stdr else float("nan"),
        "n": int(len(y)),
        "n_stdr": n_stdr,
        "n_auto_handled": int(keep.sum()),
        "n_manual_review": int((~keep).sum()),
        "n_missed_stdr": int(missed.sum()),
    }


def select_workload_threshold(
    y_true: np.ndarray,
    pred_grade: np.ndarray,
    uncertainty: np.ndarray,
    max_missed_stdr_rate: float,
) -> Tuple[float, Dict[str, float]]:
    """Maximize calibration auto-handled fraction under a missed-STDR constraint."""
    unc = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    valid_unc = unc[np.isfinite(unc)]
    candidates = np.unique(np.concatenate([np.array([-np.inf]), valid_unc, np.array([np.inf])]))
    feasible: List[Tuple[float, Dict[str, float]]] = []
    for threshold in candidates:
        metrics = _workload_metrics(y_true, pred_grade, uncertainty, float(threshold))
        if metrics["missed_stdr_rate"] <= float(max_missed_stdr_rate):
            feasible.append((float(threshold), metrics))
    if not feasible:
        threshold = float("-inf")
        return threshold, _workload_metrics(y_true, pred_grade, uncertainty, threshold)
    return max(feasible, key=lambda item: (item[1]["auto_handled_fraction"], item[0]))


def selective_rows(
    y_cal: np.ndarray,
    pred_cal: np.ndarray,
    uncertainty_cal: np.ndarray,
    y_test: np.ndarray,
    pred_test: np.ndarray,
    uncertainty_test: np.ndarray,
    *,
    coverages: Sequence[float] = DEFAULT_COVERAGES,
    n_boot: int = 2000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Compute locked-test selective metrics using calibration quantile thresholds."""
    y_cal = np.asarray(y_cal, dtype=np.int64)
    pred_cal = np.asarray(pred_cal, dtype=np.int64)
    uncertainty_cal = np.asarray(uncertainty_cal, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.int64)
    pred_test = np.asarray(pred_test, dtype=np.int64)
    uncertainty_test = np.asarray(uncertainty_test, dtype=np.float64)
    valid_cal = (y_cal >= 0) & (pred_cal >= 0) & np.isfinite(uncertainty_cal)
    valid_test = (y_test >= 0) & (pred_test >= 0) & np.isfinite(uncertainty_test)
    y_test = y_test[valid_test]
    pred_test = pred_test[valid_test]
    uncertainty_test = uncertainty_test[valid_test]
    boot = _bootstrap_indices(len(y_test), n_boot, seed)
    rows: List[Dict[str, Any]] = []
    for target_coverage in coverages:
        tau = float(np.quantile(uncertainty_cal[valid_cal], float(target_coverage)))
        keep = uncertainty_test <= tau

        def sampled(index: np.ndarray) -> Tuple[float, float, float, float]:
            sampled_keep = keep[index]
            if not int(sampled_keep.sum()):
                return float("nan"), float("nan"), float("nan"), float("nan")
            sampled_y = y_test[index][sampled_keep]
            sampled_pred = pred_test[index][sampled_keep]
            return (
                float(sampled_keep.mean()),
                qwk(sampled_y, sampled_pred),
                accuracy(sampled_y, sampled_pred),
                float((sampled_y != sampled_pred).mean()),
            )

        sampled_values = [sampled(index) for index in boot]
        coverage_ci = _percentile_ci(item[0] for item in sampled_values)
        qwk_ci = _percentile_ci(item[1] for item in sampled_values)
        acc_ci = _percentile_ci(item[2] for item in sampled_values)
        risk_ci = _percentile_ci(item[3] for item in sampled_values)
        kept_y = y_test[keep]
        kept_pred = pred_test[keep]
        rows.append(
            {
                "metric": PRIMARY_METHOD,
                "mode": "global",
                "target_coverage": float(target_coverage),
                "tau_from_calibration": tau,
                "test_coverage": float(keep.mean()),
                "test_coverage_ci_low": coverage_ci[0],
                "test_coverage_ci_high": coverage_ci[1],
                "n_test": int(len(y_test)),
                "n_kept": int(keep.sum()),
                "selective_qwk": qwk(kept_y, kept_pred),
                "selective_qwk_ci_low": qwk_ci[0],
                "selective_qwk_ci_high": qwk_ci[1],
                "selective_accuracy": accuracy(kept_y, kept_pred),
                "selective_accuracy_ci_low": acc_ci[0],
                "selective_accuracy_ci_high": acc_ci[1],
                "selective_risk": float((kept_y != kept_pred).mean()),
                "selective_risk_ci_low": risk_ci[0],
                "selective_risk_ci_high": risk_ci[1],
                "threshold_source": "calibration_split_only",
            }
        )
    return rows


def _take_preds(preds: uncertainty_lib.Preds, ordered_ids: Sequence[str]) -> uncertainty_lib.Preds:
    index = {str(sample_id): idx for idx, sample_id in enumerate(preds.ids.tolist())}
    positions = np.array([index[str(sample_id)] for sample_id in ordered_ids], dtype=np.int64)
    return uncertainty_lib.Preds(
        ids=preds.ids[positions],
        y_true=preds.y_true[positions],
        pred_grade=preds.pred_grade[positions],
        pred_argmax=preds.pred_argmax[positions],
        probs=preds.probs[positions],
        logits=preds.logits[positions] if preds.logits is not None else None,
        edl_u=preds.edl_u[positions] if preds.edl_u is not None else None,
    )


def _align_preds(reference: uncertainty_lib.Preds, others: Sequence[uncertainty_lib.Preds]) -> Tuple[uncertainty_lib.Preds, List[uncertainty_lib.Preds]]:
    common = set(str(sample_id) for sample_id in reference.ids.tolist())
    for preds in others:
        common &= set(str(sample_id) for sample_id in preds.ids.tolist())
    ordered = [str(sample_id) for sample_id in reference.ids.tolist() if str(sample_id) in common]
    if not ordered:
        raise RuntimeError("No common sample ids across prediction artifacts.")
    return _take_preds(reference, ordered), [_take_preds(preds, ordered) for preds in others]


def _prediction_grade(preds: uncertainty_lib.Preds) -> np.ndarray:
    return np.where(preds.pred_grade >= 0, preds.pred_grade, preds.pred_argmax).astype(np.int64)


def load_uncertainty_arrays(project_root: Path, tta: int) -> Dict[str, Dict[str, Any]]:
    """Load raw uncertainty signals without regenerating or modifying prediction files."""
    summary = read_json(project_root / "checkpoints" / "uncertainty_compare" / f"uncertainty_compare_tta{int(tta)}.json")
    paths = summary["paths"]
    ensemble_dir = Path(paths["ensemble_posthoc_dir"])
    seed_dirs = [Path(path) for path in paths["seed_run_dirs"]]
    single_dir = Path(paths["single_run_dir"])
    edl_dir = Path(paths["edl_run_dir"])

    ens_cal = uncertainty_lib.read_preds_csv(ensemble_dir / f"preds_image_ens_tta{int(tta)}_calib_thresholded.csv")
    ens_test = uncertainty_lib.read_preds_csv(ensemble_dir / f"preds_image_ens_tta{int(tta)}_test_thresholded.csv")
    seed_cal = [uncertainty_lib.read_preds_csv(path / f"preds_image_ce_tta{int(tta)}_calib.csv") for path in seed_dirs]
    seed_test = [uncertainty_lib.read_preds_csv(path / f"preds_image_ce_tta{int(tta)}_test.csv") for path in seed_dirs]
    ens_cal, seed_cal = _align_preds(ens_cal, seed_cal)
    ens_test, seed_test = _align_preds(ens_test, seed_test)
    u_mi_cal, u_entropy_cal, _ = uncertainty_lib.ensemble_mi(seed_cal)
    u_mi_test, u_entropy_test, _ = uncertainty_lib.ensemble_mi(seed_test)

    single_cal = uncertainty_lib.read_preds_csv(single_dir / f"preds_image_ce_tta{int(tta)}_calib.csv")
    single_test = uncertainty_lib.read_preds_csv(single_dir / f"preds_image_ce_tta{int(tta)}_test.csv")
    single_cal, _ = _align_preds(single_cal, [])
    single_test, _ = _align_preds(single_test, [])

    edl_cal = uncertainty_lib.read_preds_csv(edl_dir / f"preds_image_edl_tta{int(tta)}_calib.csv")
    edl_test = uncertainty_lib.read_preds_csv(edl_dir / f"preds_image_edl_tta{int(tta)}_test.csv")
    u_edl_cal = edl_cal.edl_u if edl_cal.edl_u is not None else uncertainty_lib.edl_u_from_logits(edl_cal.logits)
    u_edl_test = edl_test.edl_u if edl_test.edl_u is not None else uncertainty_lib.edl_u_from_logits(edl_test.logits)

    return {
        "ensemble_pred_entropy": {
            "label": PRIMARY_METHOD,
            "role": "primary_deployment_metric",
            "calib": ens_cal,
            "test": ens_test,
            "u_calib": u_entropy_cal,
            "u_test": u_entropy_test,
        },
        "ensemble_mi": {
            "label": "Ensemble MI",
            "role": "ensemble_comparator",
            "calib": ens_cal,
            "test": ens_test,
            "u_calib": u_mi_cal,
            "u_test": u_mi_test,
        },
        "softmax_entropy": {
            "label": "Softmax entropy",
            "role": "single_model_comparator",
            "calib": single_cal,
            "test": single_test,
            "u_calib": uncertainty_lib.entropy_from_probs(single_cal.probs),
            "u_test": uncertainty_lib.entropy_from_probs(single_test.probs),
        },
        "edl_u": {
            "label": "EDL-u (K/S)",
            "role": "edl_model_comparator",
            "calib": edl_cal,
            "test": edl_test,
            "u_calib": np.asarray(u_edl_cal, dtype=np.float64),
            "u_test": np.asarray(u_edl_test, dtype=np.float64),
        },
    }


def error_detection_rows(methods: Dict[str, Dict[str, Any]], *, n_boot: int, seed: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for offset, (key, item) in enumerate(methods.items()):
        test = item["test"]
        uncertainty = np.asarray(item["u_test"], dtype=np.float64)
        pred = _prediction_grade(test)
        errors = (test.y_true != pred).astype(np.int64)
        valid = (test.y_true >= 0) & (pred >= 0) & np.isfinite(uncertainty)
        errors = errors[valid]
        uncertainty = uncertainty[valid]
        boot = _bootstrap_indices(len(errors), n_boot, seed + offset)
        boot_auroc = [uncertainty_lib.auroc_binary(errors[index], uncertainty[index]) for index in boot]
        boot_auprc = [uncertainty_lib.auprc_binary(errors[index], uncertainty[index]) for index in boot]
        auroc_ci = _percentile_ci(boot_auroc)
        auprc_ci = _percentile_ci(boot_auprc)
        rows.append(
            {
                "method_key": key,
                "method": item["label"],
                "model_path_role": item["role"],
                "n_test": int(len(errors)),
                "error_rate": float(errors.mean()),
                "error_detection_auroc": uncertainty_lib.auroc_binary(errors, uncertainty),
                "error_detection_auroc_ci_low": auroc_ci[0],
                "error_detection_auroc_ci_high": auroc_ci[1],
                "error_detection_auprc": uncertainty_lib.auprc_binary(errors, uncertainty),
                "error_detection_auprc_ci_low": auprc_ci[0],
                "error_detection_auprc_ci_high": auprc_ci[1],
                "evaluation_split": "locked_test",
            }
        )
    return rows


def risk_coverage_outputs(
    methods: Dict[str, Dict[str, Any]],
    *,
    n_boot: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Compute risk-coverage curves and bootstrap AURC for each uncertainty method."""
    curve_rows: List[Dict[str, Any]] = []
    aurc_rows: List[Dict[str, Any]] = []
    for offset, (key, item) in enumerate(methods.items()):
        test = item["test"]
        pred = _prediction_grade(test)
        uncertainty = np.asarray(item["u_test"], dtype=np.float64)
        valid = (test.y_true >= 0) & (pred >= 0) & np.isfinite(uncertainty)
        errors = (test.y_true[valid] != pred[valid]).astype(np.int64)
        uncertainty = uncertainty[valid]
        rows, aurc = risk_coverage_curve(errors, uncertainty)
        for row in rows:
            row.update({"method_key": key, "method": item["label"], "model_path_role": item["role"]})
        boot_aurc = []
        for index in _bootstrap_indices(len(errors), n_boot, seed + offset):
            _, value = risk_coverage_curve(errors[index], uncertainty[index])
            boot_aurc.append(value)
        aurc_ci = _percentile_ci(boot_aurc)
        curve_rows.extend(rows)
        aurc_rows.append(
            {
                "method_key": key,
                "method": item["label"],
                "model_path_role": item["role"],
                "aurc": aurc,
                "aurc_ci_low": aurc_ci[0],
                "aurc_ci_high": aurc_ci[1],
                "n_test": int(len(errors)),
                "evaluation_split": "locked_test",
            }
        )
    return curve_rows, aurc_rows


def stdr_operating_rows(
    y_cal: np.ndarray,
    probs_cal: np.ndarray,
    y_test: np.ndarray,
    probs_test: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> List[Dict[str, Any]]:
    y_cal_binary = (np.asarray(y_cal, dtype=np.int64) >= 3).astype(np.int64)
    y_test_binary = (np.asarray(y_test, dtype=np.int64) >= 3).astype(np.int64)
    score_cal = np.asarray(probs_cal, dtype=np.float64)[:, 3:].sum(axis=1)
    score_test = np.asarray(probs_test, dtype=np.float64)[:, 3:].sum(axis=1)
    boot = _bootstrap_indices(len(y_test_binary), n_boot, seed)
    rows: List[Dict[str, Any]] = []
    for constraint in ("specificity", "sensitivity"):
        for target in DEFAULT_FIXED_POINTS:
            threshold, calib_metrics = threshold_at_fixed_operating_point(y_cal_binary, score_cal, constraint, target)
            test_metrics = _binary_metrics(y_test_binary, score_test, threshold)
            sampled = [_binary_metrics(y_test_binary[index], score_test[index], threshold) for index in boot]
            sensitivity_ci = _percentile_ci(item["sensitivity"] for item in sampled)
            specificity_ci = _percentile_ci(item["specificity"] for item in sampled)
            rows.append(
                {
                    "endpoint": "STDR (grade>=3)",
                    "constraint": f"fixed_{constraint}",
                    "target": float(target),
                    "threshold_from_calibration": threshold,
                    "calibration_sensitivity": calib_metrics["sensitivity"],
                    "calibration_specificity": calib_metrics["specificity"],
                    "test_sensitivity": test_metrics["sensitivity"],
                    "test_sensitivity_ci_low": sensitivity_ci[0],
                    "test_sensitivity_ci_high": sensitivity_ci[1],
                    "test_specificity": test_metrics["specificity"],
                    "test_specificity_ci_low": specificity_ci[0],
                    "test_specificity_ci_high": specificity_ci[1],
                    "threshold_source": "calibration_split_only",
                    "evaluation_split": "locked_test",
                }
            )
    return rows


def workload_rows(
    y_cal: np.ndarray,
    pred_cal: np.ndarray,
    uncertainty_cal: np.ndarray,
    y_test: np.ndarray,
    pred_test: np.ndarray,
    uncertainty_test: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> List[Dict[str, Any]]:
    boot = _bootstrap_indices(len(y_test), n_boot, seed)
    rows: List[Dict[str, Any]] = []
    for target in DEFAULT_MISSED_STDR_TARGETS:
        threshold, calib_metrics = select_workload_threshold(y_cal, pred_cal, uncertainty_cal, target)
        test_metrics = _workload_metrics(y_test, pred_test, uncertainty_test, threshold)
        sampled = [_workload_metrics(y_test[index], pred_test[index], uncertainty_test[index], threshold) for index in boot]
        auto_ci = _percentile_ci(item["auto_handled_fraction"] for item in sampled)
        missed_ci = _percentile_ci(item["missed_stdr_rate"] for item in sampled)
        rows.append(
            {
                "metric": PRIMARY_METHOD,
                "fixed_calibration_missed_stdr_rate": float(target),
                "tau_from_calibration": threshold,
                "calibration_auto_handled_fraction": calib_metrics["auto_handled_fraction"],
                "calibration_missed_stdr_rate": calib_metrics["missed_stdr_rate"],
                "test_auto_handled_fraction": test_metrics["auto_handled_fraction"],
                "test_auto_handled_fraction_ci_low": auto_ci[0],
                "test_auto_handled_fraction_ci_high": auto_ci[1],
                "test_manual_review_fraction": test_metrics["manual_review_fraction"],
                "test_missed_stdr_rate": test_metrics["missed_stdr_rate"],
                "test_missed_stdr_rate_ci_low": missed_ci[0],
                "test_missed_stdr_rate_ci_high": missed_ci[1],
                "n_test": test_metrics["n"],
                "n_test_stdr": test_metrics["n_stdr"],
                "n_test_auto_handled": test_metrics["n_auto_handled"],
                "n_test_manual_review": test_metrics["n_manual_review"],
                "n_test_missed_stdr": test_metrics["n_missed_stdr"],
                "threshold_source": "calibration_split_only",
                "evaluation_split": "locked_test",
            }
        )
    return rows


def _plot_risk_coverage(rows: Sequence[Dict[str, Any]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "ensemble_pred_entropy": "#2C7FB8",
        "ensemble_mi": "#41AB5D",
        "softmax_entropy": "#F28E2B",
        "edl_u": "#D95F0E",
    }
    fig, ax = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
    for method_key in dict.fromkeys(str(row["method_key"]) for row in rows):
        method_rows = [row for row in rows if row["method_key"] == method_key]
        ax.plot(
            [row["coverage"] for row in method_rows],
            [row["risk"] for row in method_rows],
            color=colors.get(method_key, "#666666"),
            linewidth=1.7,
            label=method_rows[0]["method"],
        )
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk")
    ax.set_title("Risk-coverage curves (TTA=4)")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    for suffix in (".png", ".pdf"):
        fig.savefig(output_dir / f"clinical_risk_coverage_curve{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_summary(
    output_dir: Path,
    selective: Sequence[Dict[str, Any]],
    aurc_rows: Sequence[Dict[str, Any]],
    error_rows: Sequence[Dict[str, Any]],
    stdr_rows: Sequence[Dict[str, Any]],
    workload: Sequence[Dict[str, Any]],
) -> None:
    best_error = max(error_rows, key=lambda row: float(row["error_detection_auroc"]))
    primary_aurc = next(row for row in aurc_rows if row["method_key"] == "ensemble_pred_entropy")
    lowest_aurc = min(aurc_rows, key=lambda row: float(row["aurc"]))
    lowest_comparator_aurc = min(
        (row for row in aurc_rows if row["method_key"] != "ensemble_pred_entropy"),
        key=lambda row: float(row["aurc"]),
    )
    selected_90 = next(row for row in selective if float(row["target_coverage"]) == 0.90)
    workload_5 = next(row for row in workload if float(row["fixed_calibration_missed_stdr_rate"]) == 0.05)
    text = [
        "# Clinical Deployment Metrics Summary",
        "",
        "Protocol: all thresholds were selected on the calibration split and applied once to the locked test split.",
        "",
        f"- Primary uncertainty metric: {PRIMARY_METHOD}.",
        f"- Primary-metric AURC: {primary_aurc['aurc']} (95% CI {primary_aurc['aurc_ci_low']} to {primary_aurc['aurc_ci_high']}).",
        f"- Lowest overall AURC: {lowest_aurc['method']} ({lowest_aurc['aurc']}, 95% CI {lowest_aurc['aurc_ci_low']} to {lowest_aurc['aurc_ci_high']}).",
        f"- Best non-primary comparator AURC: {lowest_comparator_aurc['method']} ({lowest_comparator_aurc['aurc']}, 95% CI {lowest_comparator_aurc['aurc_ci_low']} to {lowest_comparator_aurc['aurc_ci_high']}).",
        f"- At 90% target coverage: locked-test coverage {selected_90['test_coverage']}, selective QWK {selected_90['selective_qwk']}, selective accuracy {selected_90['selective_accuracy']}.",
        f"- Strongest error detector by AUROC: {best_error['method']} ({best_error['error_detection_auroc']}, 95% CI {best_error['error_detection_auroc_ci_low']} to {best_error['error_detection_auroc_ci_high']}).",
        f"- At calibration missed-STDR target 5%: locked-test automated workload reduction {workload_5['test_auto_handled_fraction']}, manual-review fraction {workload_5['test_manual_review_fraction']}, missed-STDR rate {workload_5['test_missed_stdr_rate']}.",
        "",
        "Interpretation guardrail: workload reduction assumes that withheld high-uncertainty cases receive human grading. It is a retrospective locked-test estimate, not a prospective clinical utility claim.",
    ]
    (output_dir / "clinical_deployment_metrics_summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out_dir", type=Path, default=Path("checkpoints/paper_assets/revision_round2/clinical_deployment_tta4"))
    parser.add_argument("--tta", type=int, default=4)
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = args.project_root.resolve()
    output_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = load_uncertainty_arrays(root, args.tta)
    primary = methods["ensemble_pred_entropy"]
    calib = primary["calib"]
    test = primary["test"]
    pred_cal = _prediction_grade(calib)
    pred_test = _prediction_grade(test)
    u_cal = np.asarray(primary["u_calib"], dtype=np.float64)
    u_test = np.asarray(primary["u_test"], dtype=np.float64)

    selective = selective_rows(
        calib.y_true,
        pred_cal,
        u_cal,
        test.y_true,
        pred_test,
        u_test,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    risk_rows, aurc_rows = risk_coverage_outputs(methods, n_boot=args.n_boot, seed=args.seed)
    error_rows = error_detection_rows(methods, n_boot=args.n_boot, seed=args.seed)
    stdr_rows = stdr_operating_rows(calib.y_true, calib.probs, test.y_true, test.probs, n_boot=args.n_boot, seed=args.seed)
    workload = workload_rows(
        calib.y_true,
        pred_cal,
        u_cal,
        test.y_true,
        pred_test,
        u_test,
        n_boot=args.n_boot,
        seed=args.seed,
    )

    for row in risk_rows:
        row.update({"tta": int(args.tta), "evaluation_split": "locked_test"})
    for row in aurc_rows:
        row.update({"tta": int(args.tta)})
    write_rows(output_dir / "clinical_selective_metrics_ci.csv", selective, list(selective[0]))
    write_rows(output_dir / "clinical_risk_coverage_curve.csv", risk_rows, list(risk_rows[0]))
    write_rows(output_dir / "clinical_aurc_ci.csv", aurc_rows, list(aurc_rows[0]))
    write_rows(output_dir / "clinical_error_detection_metrics_ci.csv", error_rows, list(error_rows[0]))
    write_rows(output_dir / "clinical_stdr_operating_points_ci.csv", stdr_rows, list(stdr_rows[0]))
    write_rows(output_dir / "clinical_workload_reduction_ci.csv", workload, list(workload[0]))
    write_json(
        output_dir / "clinical_deployment_metrics_metadata.json",
        {
            "tta": int(args.tta),
            "bootstrap_replicates": int(args.n_boot),
            "bootstrap_seed": int(args.seed),
            "threshold_fit_split": "calibration",
            "evaluation_split": "locked_test",
            "primary_uncertainty_metric": PRIMARY_METHOD,
            "stdr_definition": "grade >= 3",
            "workload_definition": "fraction auto-handled after withholding high-uncertainty cases for human grading",
            "input_summary": str(root / "checkpoints" / "uncertainty_compare" / f"uncertainty_compare_tta{int(args.tta)}.json"),
        },
    )
    _plot_risk_coverage(risk_rows, output_dir)
    _write_summary(output_dir, selective, aurc_rows, error_rows, stdr_rows, workload)
    print(f"[OK] wrote clinical deployment metrics to {output_dir}")


if __name__ == "__main__":
    main()
