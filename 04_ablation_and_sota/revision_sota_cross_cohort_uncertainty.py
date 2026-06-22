"""Summarize matched predictive-entropy metrics across internal and external cohorts."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import uncertainty_compare as uncertainty_lib
    from revision_clinical_deployment_metrics import risk_coverage_curve
    from revision_utils import entropy, normalize_probs
except ImportError:  # pragma: no cover
    from src import uncertainty_compare as uncertainty_lib
    from src.revision_clinical_deployment_metrics import risk_coverage_curve
    from src.revision_utils import entropy, normalize_probs


PROB_COLS = [f"p{i}" for i in range(5)]


def default_sources(project_root: Path) -> List[Dict[str, Any]]:
    root = Path(project_root)
    sota = root / "checkpoints" / "SOTA baselines" / "revision_round2"
    models = (
        (
            "Deployed quality-aware ordinal ensemble TTA4",
            "deployment_reference",
            root / "checkpoints" / "Multi-seed integration" / "ensemble_posthoc" / "preds_image_ens_tta4_test_thresholded.csv",
            root / "checkpoints" / "external_eval_ensemble_tta4" / "ensemble_preds_{cohort}.csv",
        ),
        (
            "ResNet50 ImageNet full fine-tuning TTA4",
            "mainstream_baseline",
            sota / "posthoc_internal_tta4" / "resnet50_imagenet" / "preds_image_ce_tta4_test.csv",
            sota / "posthoc_external_tta4" / "resnet50_imagenet" / "pred_grade_{cohort}.csv",
        ),
        (
            "EfficientNet-B4 ImageNet full fine-tuning TTA4",
            "mainstream_baseline",
            sota / "posthoc_internal_tta4" / "efficientnet_b4_imagenet" / "preds_image_ce_tta4_test.csv",
            sota / "posthoc_external_tta4" / "efficientnet_b4_imagenet" / "pred_grade_{cohort}.csv",
        ),
        (
            "ViT-B/16 ImageNet LoRA16 TTA4",
            "mainstream_baseline",
            sota / "posthoc_internal_tta4" / "vit_base_imagenet_lora16" / "preds_image_ce_tta4_test.csv",
            sota / "posthoc_external_tta4" / "vit_base_imagenet_lora16" / "pred_grade_{cohort}.csv",
        ),
        (
            "Swin-T ImageNet full fine-tuning TTA4",
            "mainstream_baseline",
            sota / "posthoc_internal_tta4" / "swin_tiny_imagenet_full_lr5e5" / "preds_image_ce_tta4_test.csv",
            sota / "posthoc_external_tta4" / "swin_tiny_imagenet_full_lr5e5" / "pred_grade_{cohort}.csv",
        ),
    )
    cohorts = (
        ("eyepacs_internal_test", "internal_locked_test"),
        ("aptos2019", "external_descriptive"),
        ("messidor2_dr_grades", "external_descriptive"),
        ("ddr_dr_grading", "external_descriptive"),
    )
    sources: List[Dict[str, Any]] = []
    for model, role, internal_path, external_template in models:
        for cohort, scope in cohorts:
            prediction_csv = internal_path if cohort == "eyepacs_internal_test" else Path(str(external_template).format(cohort=cohort))
            sources.append(
                {
                    "model": model,
                    "role": role,
                    "scope": scope,
                    "cohort": cohort,
                    "prediction_csv": prediction_csv,
                }
            )
    return sources


def _first_value(row: Mapping[str, str], keys: Sequence[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value not in ("", None):
            return str(value)
    raise ValueError(f"Missing required columns: expected one of {list(keys)}")


def read_prediction_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Prediction CSV is empty: {path}")
    missing = set(PROB_COLS).difference(rows[0])
    if missing:
        raise ValueError(f"Missing probability columns {sorted(missing)} in {path}")
    probabilities = normalize_probs(
        np.asarray([[float(row[col]) for col in PROB_COLS] for row in rows], dtype=np.float64)
    )
    y_true = np.asarray([int(float(_first_value(row, ("y_true", "true_grade")))) for row in rows], dtype=np.int64)
    y_pred = np.asarray(
        [
            int(float(_first_value(row, ("pred_thresholded", "pred_grade", "y_pred", "pred_argmax"))))
            for row in rows
        ],
        dtype=np.int64,
    )
    return y_true, y_pred, probabilities


def summarize_prediction_arrays(
    *,
    model: str,
    cohort: str,
    scope: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    prediction_csv: str,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    probabilities = normalize_probs(np.asarray(probabilities, dtype=np.float64))
    if not (len(y_true) == len(y_pred) == len(probabilities)):
        raise ValueError("Prediction arrays must have identical lengths")
    uncertainty = entropy(probabilities)
    errors = (y_true != y_pred).astype(np.int64)
    _, aurc = risk_coverage_curve(errors, uncertainty)
    return {
        "model": model,
        "scope": scope,
        "cohort": cohort,
        "n": int(len(y_true)),
        "uncertainty_metric": "predictive_entropy",
        "error_rate": float(errors.mean()),
        "aurc": aurc,
        "error_detection_auroc": uncertainty_lib.auroc_binary(errors, uncertainty),
        "error_detection_auprc": uncertainty_lib.auprc_binary(errors, uncertainty),
        "prediction_csv": prediction_csv,
    }


def build_table(project_root: Path, out_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for source in default_sources(project_root):
        prediction_csv = Path(source["prediction_csv"])
        if not prediction_csv.is_file():
            raise FileNotFoundError(f"Missing matched-protocol prediction CSV: {prediction_csv}")
        y_true, y_pred, probabilities = read_prediction_arrays(prediction_csv)
        row = summarize_prediction_arrays(
            model=source["model"],
            cohort=source["cohort"],
            scope=source["scope"],
            y_true=y_true,
            y_pred=y_pred,
            probabilities=probabilities,
            prediction_csv=str(prediction_csv),
        )
        rows.append({**row, "role": source["role"]})

    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "role",
        "scope",
        "cohort",
        "n",
        "uncertainty_metric",
        "error_rate",
        "aurc",
        "error_detection_auroc",
        "error_detection_auprc",
        "prediction_csv",
    ]
    output = out_dir / "sota_cross_cohort_predictive_entropy_metrics.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("checkpoints/paper_assets/revision_round2/sota_cross_cohort_uncertainty"),
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else project_root / args.out_dir
    rows = build_table(project_root, out_dir)
    print(f"[OK] wrote cross-cohort predictive-entropy rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
