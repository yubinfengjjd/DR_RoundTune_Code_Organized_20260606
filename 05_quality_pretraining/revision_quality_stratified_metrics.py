"""Build post-hoc grading metrics stratified by observed EyeQ quality labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

try:
    from revision_round2_package import compute_multiclass_calibration_metrics
    from revision_utils import accuracy, macro_f1, normalize_probs, qwk
    from paper_metrics_ci import auroc_binary
except ImportError:  # pragma: no cover
    from src.revision_round2_package import compute_multiclass_calibration_metrics
    from src.revision_utils import accuracy, macro_f1, normalize_probs, qwk
    from src.paper_metrics_ci import auroc_binary


QUALITY_NAMES = {0: "good", 1: "usable", 2: "reject"}
PROB_COLS = [f"p{i}" for i in range(5)]


def read_rows(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def normalized_key(row: Mapping[str, str]) -> str:
    return str(row.get("key", "")).strip().lower()


def join_quality_records(
    prediction_rows: Sequence[Mapping[str, str]],
    quality_rows: Sequence[Mapping[str, str]],
) -> List[Dict[str, Any]]:
    quality = {
        normalized_key(row): int(float(row["quality"]))
        for row in quality_rows
        if normalized_key(row) and row.get("quality", "") != ""
    }
    records: List[Dict[str, Any]] = []
    for row in prediction_rows:
        key = normalized_key(row)
        if key not in quality:
            continue
        probs = normalize_probs(np.asarray([[float(row[col]) for col in PROB_COLS]], dtype=np.float64))[0]
        pred_value = row.get("pred_grade") or row.get("pred_thresholded") or row.get("y_pred") or row.get("pred_argmax")
        records.append(
            {
                "key": key,
                "quality_label": quality[key],
                "quality_source_type": "observed_quality_label",
                "true_grade": int(float(row.get("true_grade") or row.get("y_true", ""))),
                "pred_grade": int(float(pred_value)) if pred_value not in ("", None) else int(np.argmax(probs)),
                "probs": probs,
            }
        )
    return records


def compute_metrics(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"n": 0}
    y_true = np.asarray([row["true_grade"] for row in records], dtype=np.int64)
    y_pred = np.asarray([row["pred_grade"] for row in records], dtype=np.int64)
    probs = np.asarray([row["probs"] for row in records], dtype=np.float64)
    calibration = compute_multiclass_calibration_metrics(y_true.tolist(), probs.tolist(), n_bins=15)
    return {
        "n": int(y_true.size),
        "acc": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, k=5),
        "qwk": qwk(y_true, y_pred, k=5),
        "rdr_auroc": auroc_binary((y_true >= 2).astype(np.int64), probs[:, 2:].sum(axis=1)),
        "stdr_auroc": auroc_binary((y_true >= 3).astype(np.int64), probs[:, 3:].sum(axis=1)),
        "ece": calibration["ece"],
        "brier": calibration["brier"],
    }


def build_table(project_root: Path, config_path: Path, out_dir: Path) -> List[Dict[str, Any]]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    quality_path = Path(config["quality_csv"])
    if not quality_path.is_absolute():
        quality_path = project_root / quality_path
    quality_rows = read_rows(quality_path)
    output: List[Dict[str, Any]] = []
    for source in config["sources"]:
        prediction_path = Path(source["prediction_csv"])
        if not prediction_path.is_absolute():
            prediction_path = project_root / prediction_path
        records = join_quality_records(read_rows(prediction_path), quality_rows)
        for label in (None, 0, 1, 2):
            selected = records if label is None else [row for row in records if row["quality_label"] == label]
            output.append(
                {
                    "model": source["model"],
                    "scope": "internal_test_posthoc_observed_quality",
                    "quality_label": "all_matched" if label is None else label,
                    "quality_name": "all_matched" if label is None else QUALITY_NAMES[label],
                    "quality_source_type": "observed_quality_label",
                    **compute_metrics(selected),
                    "prediction_csv": str(prediction_path),
                    "quality_csv": str(quality_path),
                }
            )

    fields = ["model", "scope", "quality_label", "quality_name", "quality_source_type", "n", "acc", "macro_f1", "qwk", "rdr_auroc", "stdr_auroc", "ece", "brier", "prediction_csv", "quality_csv"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "quality_stratified_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    lines = [
        "# Post-hoc Quality-Stratified Metrics",
        "",
        "Observed EyeQ labels are used only for locked-test post-hoc stratification. They are never used for training, calibration thresholds, or decision-making.",
        "",
        "| Model | Quality | n | QWK | Accuracy | Macro-F1 | RDR AUROC | STDR AUROC | ECE | Brier |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in output:
        lines.append(
            f"| {row['model']} | {row['quality_name']} | {row['n']} | {row.get('qwk', '')} | {row.get('acc', '')} | "
            f"{row.get('macro_f1', '')} | {row.get('rdr_auroc', '')} | {row.get('stdr_auroc', '')} | {row.get('ece', '')} | {row.get('brier', '')} |"
        )
    (out_dir / "quality_stratified_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--config", default="configs/revision_quality_stratified_sources.json")
    parser.add_argument("--out_dir", default="checkpoints/paper_assets/revision_round2/quality_pretraining_metrics")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    config = Path(args.config)
    if not config.is_absolute():
        config = root / config
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    rows = build_table(root, config, out_dir)
    print(f"[OK] wrote quality-stratified rows: {len(rows)}")


if __name__ == "__main__":
    main()
