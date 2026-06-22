"""Export post-hoc predictive entropy stratified by observed EyeQ quality labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

try:
    from revision_quality_stratified_metrics import QUALITY_NAMES, join_quality_records, read_rows
    from revision_utils import entropy
except ImportError:  # pragma: no cover
    from src.revision_quality_stratified_metrics import QUALITY_NAMES, join_quality_records, read_rows
    from src.revision_utils import entropy


def build_rows(
    model: str,
    prediction_rows: Sequence[Mapping[str, str]],
    quality_rows: Sequence[Mapping[str, str]],
) -> List[Dict[str, Any]]:
    records = join_quality_records(prediction_rows, quality_rows)
    output: List[Dict[str, Any]] = []
    for record in records:
        uncertainty = float(entropy(np.asarray([record["probs"]], dtype=np.float64))[0])
        output.append(
            {
                "model": model,
                "key": record["key"],
                "quality_label": record["quality_label"],
                "quality_name": QUALITY_NAMES[record["quality_label"]],
                "quality_source_type": record["quality_source_type"],
                "predictive_entropy": uncertainty,
            }
        )
    return output


def write_table(out_dir: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "quality_stratified_predictive_entropy.csv"
    fields = ["model", "key", "quality_label", "quality_name", "quality_source_type", "predictive_entropy"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output


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
        output.extend(build_rows(source["model"], read_rows(prediction_path), quality_rows))
    write_table(out_dir, output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/revision_quality_stratified_sources.json"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("checkpoints/paper_assets/revision_round2/quality_pretraining_metrics"),
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    out_dir = args.out_dir if args.out_dir.is_absolute() else project_root / args.out_dir
    rows = build_table(project_root, config_path, out_dir)
    print(f"[OK] wrote quality-stratified predictive-entropy rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
