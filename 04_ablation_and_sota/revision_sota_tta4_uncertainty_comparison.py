"""Compare TTA4 uncertainty triage for the submitted pipeline and SOTA baselines."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

import uncertainty_compare as uncertainty_lib
from paper_metrics_ci import auprc_binary as referral_auprc_binary
from revision_clinical_deployment_metrics import (
    DEFAULT_MISSED_STDR_TARGETS,
    _bootstrap_indices,
    _percentile_ci,
    _workload_metrics,
    risk_coverage_curve,
    select_workload_threshold,
    selective_rows,
    stdr_operating_rows,
    workload_rows,
)
from revision_utils import accuracy, entropy, macro_f1, qwk, write_json, write_rows


def default_sources(project_root: Path) -> List[Dict[str, Any]]:
    root = Path(project_root)
    sota = root / "checkpoints" / "SOTA baselines" / "revision_round2" / "posthoc_internal_tta4"
    return [
        {
            "model_key": "deployed_ensemble_tta4",
            "model": "Deployed quality-aware ordinal ensemble TTA4",
            "role": "submitted_deployment_pipeline",
            "uncertainty_type": "ensemble_predictive_entropy",
            "tta": 4,
            "calib_csv": root / "checkpoints" / "Multi-seed integration" / "ensemble_posthoc" / "preds_image_ens_tta4_calib_thresholded.csv",
            "test_csv": root / "checkpoints" / "Multi-seed integration" / "ensemble_posthoc" / "preds_image_ens_tta4_test_thresholded.csv",
        },
        {
            "model_key": "submitted_single_tta4",
            "model": "Submitted quality-aware MAE ViT-L LoRA16 single TTA4",
            "role": "submitted_single_model",
            "uncertainty_type": "softmax_predictive_entropy",
            "tta": 4,
            "calib_csv": root / "checkpoints" / "Fine-tuning ablation" / "runs_grade_tune_lora_r16" / "grade" / "preds_image_ce_tta4_calib.csv",
            "test_csv": root / "checkpoints" / "Fine-tuning ablation" / "runs_grade_tune_lora_r16" / "grade" / "preds_image_ce_tta4_test.csv",
        },
        {
            "model_key": "resnet50_tta4",
            "model": "ResNet50 ImageNet full fine-tuning TTA4",
            "role": "mainstream_baseline",
            "uncertainty_type": "softmax_predictive_entropy",
            "tta": 4,
            "calib_csv": sota / "resnet50_imagenet" / "preds_image_ce_tta4_calib.csv",
            "test_csv": sota / "resnet50_imagenet" / "preds_image_ce_tta4_test.csv",
        },
        {
            "model_key": "efficientnet_b4_tta4",
            "model": "EfficientNet-B4 ImageNet full fine-tuning TTA4",
            "role": "mainstream_baseline",
            "uncertainty_type": "softmax_predictive_entropy",
            "tta": 4,
            "calib_csv": sota / "efficientnet_b4_imagenet" / "preds_image_ce_tta4_calib.csv",
            "test_csv": sota / "efficientnet_b4_imagenet" / "preds_image_ce_tta4_test.csv",
        },
        {
            "model_key": "vit_base_lora16_tta4",
            "model": "ViT-B/16 ImageNet LoRA16 TTA4",
            "role": "mainstream_baseline",
            "uncertainty_type": "softmax_predictive_entropy",
            "tta": 4,
            "calib_csv": sota / "vit_base_imagenet_lora16" / "preds_image_ce_tta4_calib.csv",
            "test_csv": sota / "vit_base_imagenet_lora16" / "preds_image_ce_tta4_test.csv",
        },
        {
            "model_key": "swin_t_tta4",
            "model": "Swin-T ImageNet full fine-tuning TTA4",
            "role": "mainstream_baseline",
            "uncertainty_type": "softmax_predictive_entropy",
            "tta": 4,
            "calib_csv": sota / "swin_tiny_imagenet_full_lr5e5" / "preds_image_ce_tta4_calib.csv",
            "test_csv": sota / "swin_tiny_imagenet_full_lr5e5" / "preds_image_ce_tta4_test.csv",
        },
    ]


def _prediction_grade(preds: uncertainty_lib.Preds) -> np.ndarray:
    return np.where(preds.pred_grade >= 0, preds.pred_grade, preds.pred_argmax).astype(np.int64)


def _add_source_fields(rows: Sequence[Dict[str, Any]], source_fields: Dict[str, Any]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for row in rows:
        output.append({**row, **source_fields})
    return output


def evaluate_arrays(
    *,
    model_key: str,
    model: str,
    role: str,
    uncertainty_type: str,
    y_cal: np.ndarray,
    pred_cal: np.ndarray,
    probs_cal: np.ndarray,
    y_test: np.ndarray,
    pred_test: np.ndarray,
    probs_test: np.ndarray,
    prediction_csv: str,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    """Compute one model's locked-test deployment metrics with calibration-only thresholds."""
    probs_cal = np.asarray(probs_cal, dtype=np.float64)
    probs_test = np.asarray(probs_test, dtype=np.float64)
    y_cal = np.asarray(y_cal, dtype=np.int64)
    pred_cal = np.asarray(pred_cal, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)
    pred_test = np.asarray(pred_test, dtype=np.int64)
    uncertainty_cal = entropy(probs_cal)
    uncertainty_test = entropy(probs_test)
    source = {
        "model_key": model_key,
        "model": model,
        "metric": f"{model} | {uncertainty_type}",
        "role": role,
        "uncertainty_type": uncertainty_type,
        "tta": 4,
        "prediction_csv": prediction_csv,
    }

    selective = _add_source_fields(
        selective_rows(
            y_cal,
            pred_cal,
            uncertainty_cal,
            y_test,
            pred_test,
            uncertainty_test,
            n_boot=n_boot,
            seed=seed,
        ),
        source,
    )
    curve, aurc = risk_coverage_curve((y_test != pred_test).astype(np.int64), uncertainty_test)
    curve = _add_source_fields(curve, source)
    boot = _bootstrap_indices(len(y_test), n_boot, seed)
    aurc_values = [
        risk_coverage_curve((y_test[index] != pred_test[index]).astype(np.int64), uncertainty_test[index])[1]
        for index in boot
    ]
    aurc_ci = _percentile_ci(aurc_values)
    errors = (y_test != pred_test).astype(np.int64)
    auroc_values = [uncertainty_lib.auroc_binary(errors[index], uncertainty_test[index]) for index in boot]
    auprc_values = [uncertainty_lib.auprc_binary(errors[index], uncertainty_test[index]) for index in boot]
    auroc_ci = _percentile_ci(auroc_values)
    auprc_ci = _percentile_ci(auprc_values)
    stdr_true = (y_test >= 3).astype(np.int64)
    stdr_score = probs_test[:, 3:].sum(axis=1)
    stdr_auroc = uncertainty_lib.auroc_binary(stdr_true, stdr_score)
    stdr_auprc = referral_auprc_binary(stdr_true, stdr_score)
    stdr_auroc_ci = _percentile_ci(
        uncertainty_lib.auroc_binary(stdr_true[index], stdr_score[index]) for index in boot
    )
    stdr_auprc_ci = _percentile_ci(
        referral_auprc_binary(stdr_true[index], stdr_score[index]) for index in boot
    )
    qwk_values = [qwk(y_test[index], pred_test[index]) for index in boot]
    acc_values = [accuracy(y_test[index], pred_test[index]) for index in boot]
    summary = {
        **source,
        "n_test": int(len(y_test)),
        "qwk": qwk(y_test, pred_test),
        "qwk_ci_low": _percentile_ci(qwk_values)[0],
        "qwk_ci_high": _percentile_ci(qwk_values)[1],
        "accuracy": accuracy(y_test, pred_test),
        "accuracy_ci_low": _percentile_ci(acc_values)[0],
        "accuracy_ci_high": _percentile_ci(acc_values)[1],
        "macro_f1": macro_f1(y_test, pred_test),
        "aurc": aurc,
        "aurc_ci_low": aurc_ci[0],
        "aurc_ci_high": aurc_ci[1],
        "error_rate": float(errors.mean()),
        "error_detection_auroc": uncertainty_lib.auroc_binary(errors, uncertainty_test),
        "error_detection_auroc_ci_low": auroc_ci[0],
        "error_detection_auroc_ci_high": auroc_ci[1],
        "error_detection_auprc": uncertainty_lib.auprc_binary(errors, uncertainty_test),
        "error_detection_auprc_ci_low": auprc_ci[0],
        "error_detection_auprc_ci_high": auprc_ci[1],
        "stdr_auroc": stdr_auroc,
        "stdr_auroc_ci_low": stdr_auroc_ci[0],
        "stdr_auroc_ci_high": stdr_auroc_ci[1],
        "stdr_auprc": stdr_auprc,
        "stdr_auprc_ci_low": stdr_auprc_ci[0],
        "stdr_auprc_ci_high": stdr_auprc_ci[1],
        "threshold_source": "calibration_split_only",
        "evaluation_split": "locked_test",
    }
    workload = _add_source_fields(
        workload_rows(
            y_cal,
            pred_cal,
            uncertainty_cal,
            y_test,
            pred_test,
            uncertainty_test,
            n_boot=n_boot,
            seed=seed,
        ),
        source,
    )
    stdr_operating = _add_source_fields(
        stdr_operating_rows(y_cal, probs_cal, y_test, probs_test, n_boot=n_boot, seed=seed),
        source,
    )
    return {
        "summary": summary,
        "selective_rows": selective,
        "risk_curve_rows": curve,
        "workload_rows": workload,
        "stdr_operating_rows": stdr_operating,
    }


def evaluate_source(source: Dict[str, Any], *, n_boot: int, seed: int) -> Dict[str, Any]:
    calib = uncertainty_lib.read_preds_csv(Path(source["calib_csv"]))
    test = uncertainty_lib.read_preds_csv(Path(source["test_csv"]))
    return evaluate_arrays(
        model_key=source["model_key"],
        model=source["model"],
        role=source["role"],
        uncertainty_type=source["uncertainty_type"],
        y_cal=calib.y_true,
        pred_cal=_prediction_grade(calib),
        probs_cal=calib.probs,
        y_test=test.y_true,
        pred_test=_prediction_grade(test),
        probs_test=test.probs,
        prediction_csv=str(source["test_csv"]),
        n_boot=n_boot,
        seed=seed,
    )


def paired_metric_delta_rows(
    *,
    y_test: np.ndarray,
    deployed_pred: np.ndarray,
    deployed_probs: np.ndarray,
    comparator_pred: np.ndarray,
    comparator_probs: np.ndarray,
    n_boot: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Return paired locked-test deltas with positive values favoring deployed."""
    y_test = np.asarray(y_test, dtype=np.int64)
    deployed_pred = np.asarray(deployed_pred, dtype=np.int64)
    comparator_pred = np.asarray(comparator_pred, dtype=np.int64)
    deployed_probs = np.asarray(deployed_probs, dtype=np.float64)
    comparator_probs = np.asarray(comparator_probs, dtype=np.float64)
    if not (
        len(y_test)
        == len(deployed_pred)
        == len(comparator_pred)
        == len(deployed_probs)
        == len(comparator_probs)
    ):
        raise ValueError("Paired comparison arrays must have identical lengths")
    deployed_uncertainty = entropy(deployed_probs)
    comparator_uncertainty = entropy(comparator_probs)
    deployed_errors = (y_test != deployed_pred).astype(np.int64)
    comparator_errors = (y_test != comparator_pred).astype(np.int64)
    stdr_true = (y_test >= 3).astype(np.int64)
    deployed_stdr_score = deployed_probs[:, 3:].sum(axis=1)
    comparator_stdr_score = comparator_probs[:, 3:].sum(axis=1)

    def aurc(errors: np.ndarray, uncertainty: np.ndarray) -> float:
        return risk_coverage_curve(errors, uncertainty)[1]

    metric_specs = [
        ("qwk", lambda index: qwk(y_test[index], deployed_pred[index]), lambda index: qwk(y_test[index], comparator_pred[index]), "deployed_minus_comparator"),
        ("accuracy", lambda index: accuracy(y_test[index], deployed_pred[index]), lambda index: accuracy(y_test[index], comparator_pred[index]), "deployed_minus_comparator"),
        ("aurc", lambda index: aurc(deployed_errors[index], deployed_uncertainty[index]), lambda index: aurc(comparator_errors[index], comparator_uncertainty[index]), "comparator_minus_deployed"),
        ("error_detection_auroc", lambda index: uncertainty_lib.auroc_binary(deployed_errors[index], deployed_uncertainty[index]), lambda index: uncertainty_lib.auroc_binary(comparator_errors[index], comparator_uncertainty[index]), "deployed_minus_comparator"),
        ("error_detection_auprc", lambda index: uncertainty_lib.auprc_binary(deployed_errors[index], deployed_uncertainty[index]), lambda index: uncertainty_lib.auprc_binary(comparator_errors[index], comparator_uncertainty[index]), "deployed_minus_comparator"),
        ("stdr_auroc", lambda index: uncertainty_lib.auroc_binary(stdr_true[index], deployed_stdr_score[index]), lambda index: uncertainty_lib.auroc_binary(stdr_true[index], comparator_stdr_score[index]), "deployed_minus_comparator"),
        ("stdr_auprc", lambda index: referral_auprc_binary(stdr_true[index], deployed_stdr_score[index]), lambda index: referral_auprc_binary(stdr_true[index], comparator_stdr_score[index]), "deployed_minus_comparator"),
    ]
    all_index = np.arange(len(y_test), dtype=np.int64)
    boot = _bootstrap_indices(len(y_test), n_boot, seed)
    rows: List[Dict[str, Any]] = []
    for name, deployed_metric, comparator_metric, delta_definition in metric_specs:
        deployed_value = deployed_metric(all_index)
        comparator_value = comparator_metric(all_index)
        if delta_definition == "comparator_minus_deployed":
            delta = comparator_value - deployed_value
            sampled = [comparator_metric(index) - deployed_metric(index) for index in boot]
        else:
            delta = deployed_value - comparator_value
            sampled = [deployed_metric(index) - comparator_metric(index) for index in boot]
        ci = _percentile_ci(sampled)
        rows.append(
            {
                "metric": name,
                "deployed_value": deployed_value,
                "comparator_value": comparator_value,
                "delta_favors_deployed": delta,
                "delta_ci_low": ci[0],
                "delta_ci_high": ci[1],
                "delta_definition": delta_definition,
                "delta_interpretation": "positive_values_favor_deployed",
                "evaluation_split": "locked_test",
            }
        )
    return rows


def paired_workload_delta_rows(
    *,
    y_cal: np.ndarray,
    deployed_pred_cal: np.ndarray,
    deployed_probs_cal: np.ndarray,
    comparator_pred_cal: np.ndarray,
    comparator_probs_cal: np.ndarray,
    y_test: np.ndarray,
    deployed_pred_test: np.ndarray,
    deployed_probs_test: np.ndarray,
    comparator_pred_test: np.ndarray,
    comparator_probs_test: np.ndarray,
    n_boot: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Compare calibration-locked workload reduction with paired test bootstraps."""
    deployed_unc_cal = entropy(deployed_probs_cal)
    comparator_unc_cal = entropy(comparator_probs_cal)
    deployed_unc_test = entropy(deployed_probs_test)
    comparator_unc_test = entropy(comparator_probs_test)
    boot = _bootstrap_indices(len(y_test), n_boot, seed)
    rows: List[Dict[str, Any]] = []
    for target in DEFAULT_MISSED_STDR_TARGETS:
        deployed_threshold, _ = select_workload_threshold(y_cal, deployed_pred_cal, deployed_unc_cal, target)
        comparator_threshold, _ = select_workload_threshold(y_cal, comparator_pred_cal, comparator_unc_cal, target)
        deployed = _workload_metrics(y_test, deployed_pred_test, deployed_unc_test, deployed_threshold)
        comparator = _workload_metrics(y_test, comparator_pred_test, comparator_unc_test, comparator_threshold)
        sampled = [
            (
                _workload_metrics(y_test[index], deployed_pred_test[index], deployed_unc_test[index], deployed_threshold),
                _workload_metrics(y_test[index], comparator_pred_test[index], comparator_unc_test[index], comparator_threshold),
            )
            for index in boot
        ]
        auto_ci = _percentile_ci(dep["auto_handled_fraction"] - comp["auto_handled_fraction"] for dep, comp in sampled)
        missed_ci = _percentile_ci(comp["missed_stdr_rate"] - dep["missed_stdr_rate"] for dep, comp in sampled)
        rows.append(
            {
                "fixed_calibration_missed_stdr_rate": float(target),
                "deployed_threshold_from_calibration": deployed_threshold,
                "comparator_threshold_from_calibration": comparator_threshold,
                "deployed_test_auto_handled_fraction": deployed["auto_handled_fraction"],
                "comparator_test_auto_handled_fraction": comparator["auto_handled_fraction"],
                "auto_handled_delta_favors_deployed": deployed["auto_handled_fraction"] - comparator["auto_handled_fraction"],
                "auto_handled_delta_ci_low": auto_ci[0],
                "auto_handled_delta_ci_high": auto_ci[1],
                "deployed_test_missed_stdr_rate": deployed["missed_stdr_rate"],
                "comparator_test_missed_stdr_rate": comparator["missed_stdr_rate"],
                "missed_stdr_rate_delta_favors_deployed": comparator["missed_stdr_rate"] - deployed["missed_stdr_rate"],
                "missed_stdr_rate_delta_ci_low": missed_ci[0],
                "missed_stdr_rate_delta_ci_high": missed_ci[1],
                "delta_interpretation": "positive_values_favor_deployed",
                "threshold_source": "calibration_split_only",
                "evaluation_split": "locked_test",
            }
        )
    return rows


def paired_deployed_vs_swin_rows(project_root: Path, *, n_boot: int, seed: int) -> Dict[str, List[Dict[str, Any]]]:
    sources = default_sources(project_root)
    deployed_source = next(source for source in sources if source["model_key"] == "deployed_ensemble_tta4")
    swin_source = next(source for source in sources if source["model_key"] == "swin_t_tta4")
    deployed_cal, swin_cal = uncertainty_lib.align_to(
        uncertainty_lib.read_preds_csv(Path(deployed_source["calib_csv"])),
        uncertainty_lib.read_preds_csv(Path(swin_source["calib_csv"])),
    )
    deployed_test, swin_test = uncertainty_lib.align_to(
        uncertainty_lib.read_preds_csv(Path(deployed_source["test_csv"])),
        uncertainty_lib.read_preds_csv(Path(swin_source["test_csv"])),
    )
    if not np.array_equal(deployed_cal.y_true, swin_cal.y_true):
        raise ValueError("Calibration labels differ after deployed/Swin-T id alignment")
    if not np.array_equal(deployed_test.y_true, swin_test.y_true):
        raise ValueError("Test labels differ after deployed/Swin-T id alignment")
    source = {
        "deployed_model_key": deployed_source["model_key"],
        "deployed_model": deployed_source["model"],
        "comparator_model_key": swin_source["model_key"],
        "comparator_model": swin_source["model"],
        "tta": 4,
    }
    metrics = paired_metric_delta_rows(
        y_test=deployed_test.y_true,
        deployed_pred=_prediction_grade(deployed_test),
        deployed_probs=deployed_test.probs,
        comparator_pred=_prediction_grade(swin_test),
        comparator_probs=swin_test.probs,
        n_boot=n_boot,
        seed=seed,
    )
    workload = paired_workload_delta_rows(
        y_cal=deployed_cal.y_true,
        deployed_pred_cal=_prediction_grade(deployed_cal),
        deployed_probs_cal=deployed_cal.probs,
        comparator_pred_cal=_prediction_grade(swin_cal),
        comparator_probs_cal=swin_cal.probs,
        y_test=deployed_test.y_true,
        deployed_pred_test=_prediction_grade(deployed_test),
        deployed_probs_test=deployed_test.probs,
        comparator_pred_test=_prediction_grade(swin_test),
        comparator_probs_test=swin_test.probs,
        n_boot=n_boot,
        seed=seed,
    )
    return {
        "metric_rows": [{**source, **row} for row in metrics],
        "workload_rows": [{**source, **row} for row in workload],
    }


def _plot_risk_curves(rows: Sequence[Dict[str, Any]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#1F77B4", "#4C78A8", "#59A14F", "#F28E2B", "#B07AA1", "#E15759"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for color, model_key in zip(colors, dict.fromkeys(row["model_key"] for row in rows)):
        selected = [row for row in rows if row["model_key"] == model_key]
        ax.plot(
            [row["coverage"] for row in selected],
            [row["risk"] for row in selected],
            color=color,
            linewidth=1.5,
            label=selected[0]["model"],
        )
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk")
    ax.set_title("TTA4 risk-coverage comparison")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, fontsize=7)
    for suffix in (".png", ".pdf"):
        fig.savefig(output_dir / f"sota_tta4_risk_coverage_comparison{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_summary(
    output_dir: Path,
    summaries: Sequence[Dict[str, Any]],
    selective: Sequence[Dict[str, Any]],
    workload: Sequence[Dict[str, Any]],
    paired_workload: Sequence[Dict[str, Any]],
) -> None:
    deployed = next(row for row in summaries if row["model_key"] == "deployed_ensemble_tta4")
    swin = next(row for row in summaries if row["model_key"] == "swin_t_tta4")
    deployed_90 = next(row for row in selective if row["model_key"] == "deployed_ensemble_tta4" and float(row["target_coverage"]) == 0.90)
    swin_90 = next(row for row in selective if row["model_key"] == "swin_t_tta4" and float(row["target_coverage"]) == 0.90)
    deployed_work = next(row for row in workload if row["model_key"] == "deployed_ensemble_tta4" and float(row["fixed_calibration_missed_stdr_rate"]) == 0.05)
    swin_work = next(row for row in workload if row["model_key"] == "swin_t_tta4" and float(row["fixed_calibration_missed_stdr_rate"]) == 0.05)
    paired_work = next(row for row in paired_workload if float(row["fixed_calibration_missed_stdr_rate"]) == 0.05)
    lines = [
        "# SOTA TTA4 Uncertainty Comparison",
        "",
        "All uncertainty thresholds are derived from each model's calibration split and applied once to the locked internal test split.",
        "",
        f"- Full-coverage QWK: deployed ensemble {deployed['qwk']} vs Swin-T {swin['qwk']}.",
        f"- AURC: deployed ensemble {deployed['aurc']} vs Swin-T {swin['aurc']}.",
        f"- Error-detection AUROC: deployed ensemble {deployed['error_detection_auroc']} vs Swin-T {swin['error_detection_auroc']}.",
        f"- At 90% target coverage, selective QWK: deployed ensemble {deployed_90['selective_qwk']} vs Swin-T {swin_90['selective_qwk']}.",
        f"- At calibration missed-STDR target 5%, automated workload reduction: deployed ensemble {deployed_work['test_auto_handled_fraction']} vs Swin-T {swin_work['test_auto_handled_fraction']}.",
        f"- Paired bootstrap automated workload-reduction delta favoring deployed: {paired_work['auto_handled_delta_favors_deployed']} (95% CI {paired_work['auto_handled_delta_ci_low']} to {paired_work['auto_handled_delta_ci_high']}).",
        "",
        "Interpretation guardrail: this is a retrospective locked-test deployment comparison. It must be reported together with full-coverage classification metrics and must not be described as prospective clinical utility.",
    ]
    (output_dir / "sota_tta4_uncertainty_comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out_dir", type=Path, default=Path("checkpoints/paper_assets/revision_round2/sota_tta4_uncertainty"))
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = args.project_root.resolve()
    output_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = default_sources(root)
    for source in sources:
        for key in ("calib_csv", "test_csv"):
            if not Path(source[key]).exists():
                raise FileNotFoundError(f"Missing {key}: {source[key]}")

    summaries: List[Dict[str, Any]] = []
    selective: List[Dict[str, Any]] = []
    risk_curves: List[Dict[str, Any]] = []
    workload: List[Dict[str, Any]] = []
    stdr_operating: List[Dict[str, Any]] = []
    for offset, source in enumerate(sources):
        print(f"[INFO] evaluating {source['model']}", flush=True)
        result = evaluate_source(source, n_boot=args.n_boot, seed=args.seed + offset)
        summaries.append(result["summary"])
        selective.extend(result["selective_rows"])
        risk_curves.extend(result["risk_curve_rows"])
        workload.extend(result["workload_rows"])
        stdr_operating.extend(result["stdr_operating_rows"])
    paired = paired_deployed_vs_swin_rows(root, n_boot=args.n_boot, seed=args.seed)

    write_rows(output_dir / "sota_tta4_uncertainty_summary_ci.csv", summaries, list(summaries[0]))
    write_rows(output_dir / "sota_tta4_selective_metrics_ci.csv", selective, list(selective[0]))
    write_rows(output_dir / "sota_tta4_risk_coverage_curve.csv", risk_curves, list(risk_curves[0]))
    write_rows(output_dir / "sota_tta4_workload_reduction_ci.csv", workload, list(workload[0]))
    write_rows(output_dir / "sota_tta4_stdr_operating_points_ci.csv", stdr_operating, list(stdr_operating[0]))
    write_rows(output_dir / "sota_tta4_paired_deployed_vs_swin_metric_deltas_ci.csv", paired["metric_rows"], list(paired["metric_rows"][0]))
    write_rows(output_dir / "sota_tta4_paired_deployed_vs_swin_workload_deltas_ci.csv", paired["workload_rows"], list(paired["workload_rows"][0]))
    write_json(
        output_dir / "sota_tta4_uncertainty_metadata.json",
        {
            "tta": 4,
            "bootstrap_replicates": int(args.n_boot),
            "bootstrap_seed": int(args.seed),
            "threshold_fit_split": "calibration",
            "evaluation_split": "locked_test",
            "stdr_definition": "grade >= 3",
            "stdr_pr_auc_definition": "trapezoidal_precision_recall_auc",
            "error_detection_auprc_definition": "average_precision",
            "sources": [{**source, "calib_csv": str(source["calib_csv"]), "test_csv": str(source["test_csv"])} for source in sources],
        },
    )
    _plot_risk_curves(risk_curves, output_dir)
    _write_summary(output_dir, summaries, selective, workload, paired["workload_rows"])
    print(f"[OK] wrote SOTA TTA4 uncertainty comparison to {output_dir}")


if __name__ == "__main__":
    main()
