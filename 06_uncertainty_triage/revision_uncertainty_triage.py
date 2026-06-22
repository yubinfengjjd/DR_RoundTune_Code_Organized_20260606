from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from revision_utils import accuracy, entropy, macro_f1, qwk, read_json, read_rows, write_rows


METHOD_LABELS = {
    "edl_u": "EDL-u (K/S)",
    "softmax_entropy": "Softmax entropy",
    "ensemble_mi": "Ensemble MI",
    "ensemble_pred_entropy": "Ensemble PredEntropy",
}


def norm_method(name: str) -> str:
    s = name.strip().lower().replace("_", " ")
    if "predentropy" in s or "pred entropy" in s:
        return "ensemble_pred_entropy"
    if "ensemble mi" in s:
        return "ensemble_mi"
    if "softmax" in s:
        return "softmax_entropy"
    if "edl" in s:
        return "edl_u"
    return name.strip()


def read_selective(path: Path, mode: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for row in read_rows(path):
        method_key = norm_method(row.get("method", ""))
        rows.append(
            {
                "method_key": method_key,
                "method": METHOD_LABELS.get(method_key, row.get("method", "")),
                "mode": row.get("mode", mode),
                "target_coverage": float(row.get("target_coverage", "nan")),
                "test_coverage": float(row.get("test_coverage", "nan")),
                "n_test": int(float(row.get("n_test", 0))),
                "n_kept": int(float(row.get("n_kept", 0))),
                "acc": float(row.get("acc", "nan")),
                "macro_f1": float(row.get("macro_f1", "nan")),
                "qwk": float(row.get("qwk", "nan")),
            }
        )
    return rows


def summarize_metrics(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    auc = summary.get("auc", {})
    for key, label in METHOD_LABELS.items():
        vals = auc.get(key, {})
        rows.append(
            {
                "method_key": key,
                "method": label,
                "error_detection_auroc": vals.get("auroc", ""),
                "error_detection_auprc": vals.get("auprc", ""),
                "error_rate": vals.get("err_rate", ""),
                "model_path_role": "main ensemble" if key.startswith("ensemble") else "auxiliary/single-model comparator",
            }
        )
    return rows


def choose_primary(selective_rows: List[Dict[str, Any]]) -> Tuple[str, str]:
    candidates = [r for r in selective_rows if r["mode"] == "stratified_pred" and r["target_coverage"] in (0.9, 0.8)]
    scores: Dict[str, List[float]] = {}
    for row in candidates:
        if not row["method_key"].startswith("ensemble"):
            continue
        scores.setdefault(row["method_key"], []).append(float(row["qwk"]))
    if not scores:
        return "ensemble_pred_entropy", "Fallback to main ensemble predictive entropy because no selective rows were available."
    ranked = sorted(scores.items(), key=lambda kv: (sum(kv[1]) / len(kv[1]), kv[0] == "ensemble_pred_entropy"), reverse=True)
    primary = ranked[0][0]
    reason = (
        "Chosen from main ensemble uncertainty metrics by mean selective QWK at 90% and 80% target coverage "
        "under predicted-grade-stratified calibration thresholds."
    )
    return primary, reason


def threshold_rows_from_json(summary: Dict[str, Any], primary_key: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in summary.get("selective", []):
        method_key = norm_method(str(item.get("method", "")))
        if method_key != primary_key:
            continue
        target = item.get("target_coverage", "")
        tau = item.get("tau_from_calib", "")
        rows.append(
            {
                "recommended_metric": METHOD_LABELS.get(primary_key, primary_key),
                "mode": item.get("mode", ""),
                "target_coverage": target,
                "tau_from_calibration": tau,
                "test_coverage": item.get("test_coverage", ""),
                "test_qwk": item.get("qwk", ""),
                "test_acc": item.get("acc", ""),
                "test_macro_f1": item.get("macro_f1", ""),
                "threshold_source": "calibration_split_only",
            }
        )
    return rows


def load_ensemble_uncertainty(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = read_rows(path)
    y = np.array([int(float(r.get("true_grade", r.get("y_true", -1)))) for r in rows], dtype=int)
    pred = np.array(
        [
            int(float(r.get("pred_grade", r.get("pred_thresholded", r.get("pred_argmax", -1)))))
            for r in rows
        ],
        dtype=int,
    )
    probs = np.array([[float(r[f"p{i}"]) for i in range(5)] for r in rows], dtype=np.float64)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    return y, pred, entropy(probs)


def calibration_threshold_tables(
    project_root: Path, primary_key: str, tta: int = 4
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if primary_key != "ensemble_pred_entropy":
        return [], []
    base = project_root / "checkpoints" / "Multi-seed integration" / "ensemble_posthoc"
    y_cal, pred_cal, u_cal = load_ensemble_uncertainty(base / f"preds_image_ens_tta{int(tta)}_calib_thresholded.csv")
    y_test, pred_test, u_test = load_ensemble_uncertainty(base / f"preds_image_ens_tta{int(tta)}_test_thresholded.csv")
    coverages = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7]
    threshold_rows: List[Dict[str, Any]] = []
    fixed_rows: List[Dict[str, Any]] = []
    valid_cal = (y_cal >= 0) & (pred_cal >= 0) & np.isfinite(u_cal)
    valid_test = (y_test >= 0) & (pred_test >= 0) & np.isfinite(u_test)

    for cov in coverages:
        tau_global = float("inf") if cov >= 1.0 else float(np.quantile(u_cal[valid_cal], cov))
        tau_by_class: Dict[int, float] = {}
        for c in range(5):
            mask_c = valid_cal & (pred_cal == c)
            tau_by_class[c] = tau_global if int(mask_c.sum()) < 30 else float(np.quantile(u_cal[mask_c], cov))

        for mode in ("global", "stratified_pred"):
            if mode == "global":
                keep = valid_test & (u_test <= tau_global)
                tau_repr = tau_global
                tau_by_repr = ""
            else:
                tau_vec = np.full_like(u_test, fill_value=float("inf"), dtype=np.float64)
                for c, tau_c in tau_by_class.items():
                    tau_vec[pred_test == c] = tau_c
                keep = valid_test & (u_test <= tau_vec)
                tau_repr = ""
                tau_by_repr = ";".join(f"{c}:{tau_by_class[c]:.6f}" for c in range(5))
            threshold_rows.append(
                {
                    "tta": int(tta),
                    "recommended_metric": METHOD_LABELS[primary_key],
                    "mode": mode,
                    "target_coverage": cov,
                    "tau_from_calibration": tau_repr,
                    "tau_by_pred_grade_from_calibration": tau_by_repr,
                    "test_coverage": float(keep.sum() / max(1, valid_test.sum())),
                    "test_qwk": qwk(y_test[keep], pred_test[keep]),
                    "test_acc": accuracy(y_test[keep], pred_test[keep]),
                    "test_macro_f1": macro_f1(y_test[keep], pred_test[keep]),
                    "threshold_source": "calibration_split_only",
                }
            )

    order = np.argsort(u_cal[valid_cal])
    u_sorted = u_cal[valid_cal][order]
    y_sorted = y_cal[valid_cal][order]
    p_sorted = pred_cal[valid_cal][order]
    err_sorted = (y_sorted != p_sorted).astype(np.float64)
    for target_risk in (0.30, 0.25, 0.20, 0.15):
        cum_err = np.cumsum(err_sorted) / np.arange(1, len(err_sorted) + 1)
        ok = np.where(cum_err <= target_risk)[0]
        if ok.size == 0:
            tau = float(u_sorted[0])
        else:
            tau = float(u_sorted[int(ok[-1])])
        keep_test = valid_test & (u_test <= tau)
        fixed_rows.append(
            {
                "tta": int(tta),
                "metric": METHOD_LABELS[primary_key],
                "fixed_calibration_retained_error_target": target_risk,
                "tau_from_calibration": tau,
                "calibration_coverage": float((u_cal[valid_cal] <= tau).mean()),
                "test_coverage": float(keep_test.sum() / max(1, valid_test.sum())),
                "test_retained_error_rate": float((y_test[keep_test] != pred_test[keep_test]).mean()) if int(keep_test.sum()) else "",
                "test_qwk": qwk(y_test[keep_test], pred_test[keep_test]),
                "test_acc": accuracy(y_test[keep_test], pred_test[keep_test]),
                "threshold_source": "calibration_split_only",
            }
        )
    return threshold_rows, fixed_rows


def write_recommendation(path: Path, primary_key: str, reason: str, metric_rows: List[Dict[str, Any]], tta: int) -> None:
    edl = next((r for r in metric_rows if r["method_key"] == "edl_u"), {})
    primary = next((r for r in metric_rows if r["method_key"] == primary_key), {})
    text = [
        "# Uncertainty Triage Recommendation",
        "",
        f"Primary deployment metric: **{METHOD_LABELS.get(primary_key, primary_key)}**.",
        "",
        f"Recommended inference protocol: **TTA={int(tta)}**.",
        "",
        f"Rationale: {reason}",
        "",
        "Clinical handling rule: cases above the calibration-derived uncertainty threshold should be withheld from automatic reporting and routed to human grading. If image quality is inadequate, recapture should be requested before final grading. Clinically ambiguous retained cases should be escalated to senior adjudication.",
        "",
        "Endpoint interpretation:",
        "",
        f"- Primary selective-prediction metric AUROC/AUPRC for error detection: {primary.get('error_detection_auroc', '')} / {primary.get('error_detection_auprc', '')}.",
        f"- EDL-u comparator AUROC/AUPRC for error detection: {edl.get('error_detection_auroc', '')} / {edl.get('error_detection_auprc', '')}.",
        "- EDL-u is retained as an auxiliary error-detection comparator because it comes from the EDL model path, whereas the primary triage metric belongs to the final ensemble grading path.",
        "",
        "All triage thresholds are derived on the calibration split and applied once to locked test or external evaluation files.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default=".")
    ap.add_argument("--in_dir", type=str, default="checkpoints/uncertainty_compare")
    ap.add_argument("--out_dir", type=str, default="checkpoints/paper_assets/revision")
    ap.add_argument("--tta", type=int, default=4, help="Recommended ensemble TTA count used for triage thresholds.")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    in_dir = root / args.in_dir
    out_dir = root / args.out_dir
    summary = read_json(in_dir / f"uncertainty_compare_tta{int(args.tta)}.json")
    metric_rows = summarize_metrics(summary)
    selective_rows = (
        read_selective(in_dir / f"selective_points_tta{int(args.tta)}_global.csv", "global")
        + read_selective(in_dir / f"selective_points_tta{int(args.tta)}_stratified_pred.csv", "stratified_pred")
    )
    primary_key, reason = choose_primary(selective_rows)
    for row in metric_rows:
        row["deployment_recommendation"] = "primary" if row["method_key"] == primary_key else "secondary/comparator"
    write_rows(
        out_dir / "revision_uncertainty_metric_summary.csv",
        metric_rows,
        ["method_key", "method", "deployment_recommendation", "model_path_role", "error_detection_auroc", "error_detection_auprc", "error_rate"],
    )
    write_rows(
        out_dir / "revision_selective_sensitivity.csv",
        selective_rows,
        ["method_key", "method", "mode", "target_coverage", "test_coverage", "n_test", "n_kept", "acc", "macro_f1", "qwk"],
    )
    triage, fixed_risk = calibration_threshold_tables(root, primary_key, tta=int(args.tta))
    if not triage:
        triage = threshold_rows_from_json(summary, primary_key)
    write_rows(
        out_dir / "revision_triage_thresholds.csv",
        triage,
        [
            "tta",
            "recommended_metric",
            "mode",
            "target_coverage",
            "tau_from_calibration",
            "tau_by_pred_grade_from_calibration",
            "test_coverage",
            "test_qwk",
            "test_acc",
            "test_macro_f1",
            "threshold_source",
        ],
    )
    write_rows(
        out_dir / "revision_fixed_risk_thresholds.csv",
        fixed_risk,
        [
            "tta",
            "metric",
            "fixed_calibration_retained_error_target",
            "tau_from_calibration",
            "calibration_coverage",
            "test_coverage",
            "test_retained_error_rate",
            "test_qwk",
            "test_acc",
            "threshold_source",
        ],
    )
    write_recommendation(out_dir / "revision_uncertainty_recommendation.md", primary_key, reason, metric_rows, tta=int(args.tta))
    print(f"[OK] primary uncertainty metric: {METHOD_LABELS.get(primary_key, primary_key)}")


if __name__ == "__main__":
    main()
