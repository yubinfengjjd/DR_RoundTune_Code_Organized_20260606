"""Build reviewer-facing SOTA comparison metrics from frozen prediction CSVs.

The script never trains models and never fits thresholds. It reads prediction
artifacts produced by the fixed-split evaluation notebooks and reports a
consistent metric set for internal and external cohorts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

try:
    from revision_round2_package import compute_multiclass_calibration_metrics
    from revision_utils import accuracy, macro_f1, normalize_probs, qwk
    from paper_metrics_ci import auprc_binary, auroc_binary
except ImportError:  # pragma: no cover - package-style import for unit tests
    from src.revision_round2_package import compute_multiclass_calibration_metrics
    from src.revision_utils import accuracy, macro_f1, normalize_probs, qwk
    from src.paper_metrics_ci import auprc_binary, auroc_binary


PROB_COLS = [f"p{i}" for i in range(5)]
METRIC_COLUMNS = [
    "n",
    "trainable_params",
    "total_params_per_model",
    "inference_models",
    "prediction_column",
    "acc",
    "macro_f1",
    "qwk",
    "argmax_acc",
    "argmax_macro_f1",
    "argmax_qwk",
    "rdr_auroc",
    "rdr_auprc",
    "stdr_auroc",
    "stdr_auprc",
    "ece",
    "mce",
    "nll",
    "brier",
    "mean_confidence",
]
EXTERNAL_MEAN_COLUMNS = [
    "acc",
    "macro_f1",
    "qwk",
    "argmax_acc",
    "argmax_macro_f1",
    "argmax_qwk",
    "rdr_auroc",
    "rdr_auprc",
    "stdr_auroc",
    "stdr_auprc",
    "ece",
    "mce",
    "nll",
    "brier",
    "mean_confidence",
]


def read_rows(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def first_present(row: Mapping[str, str], names: Iterable[str]) -> str:
    for name in names:
        value = row.get(name, "")
        if value not in ("", None):
            return str(value)
    return ""


def select_prediction_column(rows: Sequence[Mapping[str, str]]) -> str:
    if not rows:
        raise ValueError("Prediction CSV contains no rows")
    for name in ("pred_grade", "pred_thresholded", "y_pred", "pred_argmax"):
        if all(row.get(name, "") not in ("", None) for row in rows):
            return name
    return "argmax_probs"


def compute_prediction_metrics(path: Path) -> Dict[str, Any]:
    rows = read_rows(Path(path))
    if not rows:
        raise ValueError(f"Prediction CSV contains no rows: {path}")
    missing = [name for name in PROB_COLS if name not in rows[0]]
    if missing:
        raise ValueError(f"Prediction CSV is missing probability columns {missing}: {path}")

    y_true = np.asarray(
        [int(float(first_present(row, ("true_grade", "y_true")))) for row in rows],
        dtype=np.int64,
    )
    probs = normalize_probs(
        np.asarray([[float(row[name]) for name in PROB_COLS] for row in rows], dtype=np.float64)
    )
    prediction_column = select_prediction_column(rows)
    argmax_pred = np.argmax(probs, axis=1).astype(np.int64)
    if prediction_column == "argmax_probs":
        y_pred = argmax_pred
    else:
        y_pred = np.asarray([int(float(row[prediction_column])) for row in rows], dtype=np.int64)

    rdr_true = (y_true >= 2).astype(np.int64)
    rdr_score = probs[:, 2:].sum(axis=1)
    stdr_true = (y_true >= 3).astype(np.int64)
    stdr_score = probs[:, 3:].sum(axis=1)
    calibration = compute_multiclass_calibration_metrics(y_true.tolist(), probs.tolist(), n_bins=15)
    return {
        "n": int(y_true.size),
        "prediction_column": prediction_column,
        "acc": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, k=5),
        "qwk": qwk(y_true, y_pred, k=5),
        "argmax_acc": accuracy(y_true, argmax_pred),
        "argmax_macro_f1": macro_f1(y_true, argmax_pred, k=5),
        "argmax_qwk": qwk(y_true, argmax_pred, k=5),
        "rdr_auroc": auroc_binary(rdr_true, rdr_score),
        "rdr_auprc": auprc_binary(rdr_true, rdr_score),
        "stdr_auroc": auroc_binary(stdr_true, stdr_score),
        "stdr_auprc": auprc_binary(stdr_true, stdr_score),
        **calibration,
    }


def load_source_config(path: Path) -> Dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def load_sources(path: Path) -> List[Dict[str, Any]]:
    data = load_source_config(path)
    sources = data.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError(f"'sources' must be a list: {path}")
    return [dict(source) for source in sources]


def load_model_efficiency(project_root: Path, config: Mapping[str, Any], model: str) -> Dict[str, Any]:
    metadata = config.get("model_metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("'model_metadata' must be an object when provided")
    item = metadata.get(model, {})
    if not isinstance(item, dict):
        raise ValueError(f"Model metadata must be an object: {model}")
    result: Dict[str, Any] = {
        "trainable_params": "",
        "total_params_per_model": "",
        "inference_models": item.get("inference_models", ""),
    }
    value = item.get("summary_json", "")
    if value:
        path = Path(str(value))
        if not path.is_absolute():
            path = project_root / path
        if path.exists():
            with path.open(encoding="utf-8") as f:
                summary = json.load(f)
            finetune = summary.get("init", {}).get("finetune", {})
            result["trainable_params"] = finetune.get("trainable_params", "")
            result["total_params_per_model"] = finetune.get("total_params", "")
    return result


def build_external_mean_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate available external cohorts without mixing internal results or imputing gaps."""
    grouped: Dict[tuple[str, str, Any], List[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("scope") != "external" or row.get("status") != "available":
            continue
        key = (str(row.get("model", "")), str(row.get("role", "")), row.get("tta", ""))
        grouped.setdefault(key, []).append(row)
    output: List[Dict[str, Any]] = []
    for (model, role, tta), items in grouped.items():
        summary: Dict[str, Any] = {
            "model": model,
            "role": role,
            "scope": "external_mean",
            "tta": tta,
            "external_cohorts": len(items),
        }
        for name in EXTERNAL_MEAN_COLUMNS:
            values = [float(item[name]) for item in items if item.get(name, "") not in ("", None)]
            summary[name] = float(np.mean(values)) if values else ""
        output.append(summary)
    return output


def build_comparison_table(project_root: Path, sources_json: Path, out_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    config = load_source_config(sources_json)
    for source in load_sources(sources_json):
        rel = Path(str(source["prediction_csv"]))
        path = rel if rel.is_absolute() else project_root / rel
        row: Dict[str, Any] = {
            "model": source["model"],
            "role": source["role"],
            "scope": source["scope"],
            "cohort": source["cohort"],
            "tta": source.get("tta", ""),
            "status": "available" if path.exists() else "pending_prediction_csv",
            "prediction_csv": str(path),
        }
        if path.exists():
            row.update(compute_prediction_metrics(path))
        row.update(load_model_efficiency(project_root, config, str(source["model"])))
        rows.append(row)

    fields = ["model", "role", "scope", "cohort", "tta", "status", *METRIC_COLUMNS, "prediction_csv"]
    write_rows(out_dir / "sota_prediction_metrics.csv", rows, fields)
    write_rows(
        out_dir / "sota_missing_prediction_artifacts.csv",
        [row for row in rows if row["status"] != "available"],
        ["model", "role", "scope", "cohort", "status", "prediction_csv"],
    )
    external_mean = build_external_mean_rows(rows)
    write_rows(
        out_dir / "sota_external_mean_metrics.csv",
        external_mean,
        ["model", "role", "scope", "tta", "external_cohorts", *EXTERNAL_MEAN_COLUMNS],
    )
    write_markdown(out_dir / "sota_prediction_metrics.md", rows)
    return rows


def format_value(value: Any) -> str:
    if value in ("", None):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = ["model", "scope", "cohort", "tta", "status", "trainable_params", "inference_models", "qwk", "argmax_qwk", "acc", "macro_f1", "rdr_auroc", "stdr_auroc", "ece", "brier"]
    text = [
        "# SOTA Prediction Metrics",
        "",
        "RDR is fixed as grade >= 2 and STDR is fixed as grade >= 3. Missing prediction artifacts remain pending and are never imputed from unrelated runs.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        text.append("| " + " | ".join(format_value(row.get(col, "")) for col in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--sources_json", default="configs/revision_sota_metric_sources.json")
    parser.add_argument("--out_dir", default="checkpoints/paper_assets/revision_round2/sota_metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    sources_json = Path(args.sources_json)
    if not sources_json.is_absolute():
        sources_json = root / sources_json
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    rows = build_comparison_table(root, sources_json, out_dir)
    available = sum(row["status"] == "available" for row in rows)
    print(f"[OK] wrote SOTA metric table: available={available} pending={len(rows) - available}")
    print(out_dir / "sota_prediction_metrics.csv")


if __name__ == "__main__":
    main()
