"""Select post-hoc interpretability cases from frozen prediction CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


PROB_COLS = [f"p{i}" for i in range(5)]


def read_rows(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def first_present(row: Mapping[str, str], names: Sequence[str], default: str = "") -> str:
    for name in names:
        value = row.get(name, "")
        if value not in ("", None):
            return str(value)
    return default


def sample_id(row: Mapping[str, str]) -> str:
    return first_present(row, ("key", "path", "image_path")).lower().replace("\\", "/")


def normalize_probs(row: Mapping[str, str]) -> List[float]:
    values = [max(0.0, float(row[name])) for name in PROB_COLS]
    denom = sum(values)
    if denom <= 0.0:
        raise ValueError("Probability row sums to zero")
    return [value / denom for value in values]


def read_quality_map(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    rows = read_rows(path)
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = sample_id(row)
        if not key:
            continue
        value = first_present(row, ("p_reject", "quality_reject_prob", "p_ungradable"))
        if value != "":
            result[key] = {
                "quality_score": float(value),
                "quality_source_type": "quality_head_probability",
                "p_reject": float(value),
            }
            continue
        value = first_present(row, ("quality",))
        if value != "":
            result[key] = {
                "quality_score": float(value),
                "quality_source_type": "observed_quality_label",
                "p_reject": "",
            }
    return result


def prepare_records(prediction_csv: Path, quality_csv: Optional[Path]) -> List[Dict[str, Any]]:
    quality = read_quality_map(quality_csv)
    records: List[Dict[str, Any]] = []
    for row in read_rows(prediction_csv):
        probs = normalize_probs(row)
        truth = int(float(first_present(row, ("true_grade", "y_true"))))
        pred = int(float(first_present(row, ("pred_grade", "pred_thresholded", "pred_argmax"), default=str(max(range(5), key=lambda idx: probs[idx])))))
        confidence = max(probs)
        entropy = -sum(prob * math.log(max(prob, 1e-12)) for prob in probs)
        key = sample_id(row)
        quality_row = quality.get(key, {})
        records.append(
            {
                "key": first_present(row, ("key",), default=key),
                "path": first_present(row, ("path", "image_path")),
                "true_grade": truth,
                "pred_grade": pred,
                "correct": int(truth == pred),
                "confidence": confidence,
                "entropy": entropy,
                "quality_score": quality_row.get("quality_score", ""),
                "quality_source_type": quality_row.get("quality_source_type", ""),
                "p_reject": quality_row.get("p_reject", ""),
            }
        )
    return records


def take(records: Sequence[Mapping[str, Any]], category: str, n: int, sort_key: str, reverse: bool = True) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for rank, row in enumerate(sorted(records, key=lambda item: float(item[sort_key]), reverse=reverse)[: int(n)], 1):
        selected.append({"category": category, "rank": rank, **dict(row)})
    return selected


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "category",
        "rank",
        "key",
        "path",
        "true_grade",
        "pred_grade",
        "correct",
        "confidence",
        "entropy",
        "quality_score",
        "quality_source_type",
        "p_reject",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fields})


def build_case_metadata(
    prediction_csv: Path,
    out_dir: Path,
    *,
    quality_csv: Optional[Path] = None,
    n_per_category: int = 12,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records = prepare_records(Path(prediction_csv), Path(quality_csv) if quality_csv else None)
    correct = [row for row in records if row["correct"] == 1]
    wrong = [row for row in records if row["correct"] == 0]
    rows: List[Dict[str, Any]] = []
    rows.extend(take(correct, "confident_correct", n_per_category, "confidence"))
    rows.extend(take(wrong, "confident_wrong", n_per_category, "confidence"))
    rows.extend(take(records, "high_uncertainty", n_per_category, "entropy"))
    quality_records = [row for row in records if row["quality_score"] != ""]
    quality_source_types = sorted({str(row["quality_source_type"]) for row in quality_records})
    categories = {
        "confident_correct": "available" if correct else "no_matching_cases",
        "confident_wrong": "available" if wrong else "no_matching_cases",
        "high_uncertainty": "available" if records else "no_matching_cases",
        "low_quality": "pending_quality_predictions",
    }
    if quality_records:
        rows.extend(take(quality_records, "low_quality", n_per_category, "quality_score"))
        categories["low_quality"] = "available"
    if quality_source_types == ["observed_quality_label"]:
        quality_source_mode = "posthoc_observed_label"
    elif quality_source_types == ["quality_head_probability"]:
        quality_source_mode = "posthoc_quality_head_probability"
    elif quality_source_types:
        quality_source_mode = "mixed_posthoc_quality_sources"
    else:
        quality_source_mode = "unavailable"

    status = {
        "prediction_csv": str(prediction_csv),
        "quality_csv": str(quality_csv) if quality_csv else "",
        "n_prediction_rows": len(records),
        "n_quality_rows_matched": len(quality_records),
        "quality_source_mode": quality_source_mode,
        "n_per_category": int(n_per_category),
        "categories": categories,
        "guardrail": "posthoc_case_selection_only",
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "interpretability_cases.csv", rows)
    with (out_dir / "interpretability_cases_status.json").open("w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)
    return rows, status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", required=True)
    parser.add_argument("--quality_csv", default="")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_per_category", type=int, default=12)
    args = parser.parse_args()
    quality = Path(args.quality_csv) if args.quality_csv else None
    rows, status = build_case_metadata(Path(args.pred_csv), Path(args.out_dir), quality_csv=quality, n_per_category=args.n_per_category)
    print(f"[OK] interpretability case rows: {len(rows)}")
    print(json.dumps(status["categories"], ensure_ascii=False))


if __name__ == "__main__":
    main()
