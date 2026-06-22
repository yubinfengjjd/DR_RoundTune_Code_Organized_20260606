"""Audit quality-label manifests against the locked DR split."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence


def read_rows(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def keyed(rows: Sequence[Mapping[str, str]]) -> set[str]:
    return {str(row.get("key", "")).strip().lower() for row in rows if str(row.get("key", "")).strip()}


def audit_row(metric: str, value: int, allowed_use: str, interpretation: str) -> Dict[str, object]:
    return {
        "metric": metric,
        "value": int(value),
        "allowed_use": allowed_use,
        "interpretation": interpretation,
    }


def build_audit_rows(
    master_rows: Sequence[Mapping[str, str]],
    clean_quality_rows: Sequence[Mapping[str, str]],
    eyeq_rows: Sequence[Mapping[str, str]],
) -> List[Dict[str, object]]:
    master_by_split = {
        split: keyed([row for row in master_rows if row.get("split") == split])
        for split in ("train", "val_train", "calib", "test")
    }
    clean_keys = keyed(clean_quality_rows)
    eyeq_keys = keyed(eyeq_rows)
    rows = [
        audit_row("clean_quality_manifest_rows", len(clean_quality_rows), "training_only", "Rows used to train the quality encoder."),
        audit_row("clean_quality_manifest_vs_train_overlap", len(clean_keys & master_by_split["train"]), "training_only", "Quality-training rows mapped to locked DR train split."),
        audit_row("clean_quality_manifest_vs_val_train_overlap", len(clean_keys & master_by_split["val_train"]), "training_only", "Must remain zero."),
        audit_row("clean_quality_manifest_vs_calib_overlap", len(clean_keys & master_by_split["calib"]), "training_only", "Must remain zero."),
        audit_row("clean_quality_manifest_vs_test_overlap", len(clean_keys & master_by_split["test"]), "training_only", "Must remain zero."),
        audit_row("eyeq_posthoc_labels_rows", len(eyeq_rows), "posthoc_visualization_only", "Full matched EyeQ labels; not a training manifest."),
        audit_row("eyeq_posthoc_labels_vs_train_overlap", len(eyeq_keys & master_by_split["train"]), "posthoc_visualization_only", "Train overlap is represented separately by clean quality manifest."),
        audit_row("eyeq_posthoc_labels_vs_val_train_overlap", len(eyeq_keys & master_by_split["val_train"]), "posthoc_visualization_only", "Observed labels only."),
        audit_row("eyeq_posthoc_labels_vs_calib_overlap", len(eyeq_keys & master_by_split["calib"]), "posthoc_visualization_only", "Observed labels only; forbidden for calibration thresholds."),
        audit_row("eyeq_posthoc_labels_vs_test_overlap", len(eyeq_keys & master_by_split["test"]), "posthoc_visualization_only", "Observed labels only; permitted for post-hoc case panels."),
    ]
    clean_quality_counts = Counter(str(row.get("quality", "")) for row in clean_quality_rows)
    eyeq_test_quality_counts = Counter(
        str(row.get("quality", ""))
        for row in eyeq_rows
        if str(row.get("key", "")).strip().lower() in master_by_split["test"]
    )
    for label in ("0", "1", "2"):
        rows.append(audit_row(f"clean_quality_manifest_label_{label}", clean_quality_counts[label], "training_only", "Quality-label class count."))
        rows.append(audit_row(f"eyeq_test_posthoc_label_{label}", eyeq_test_quality_counts[label], "posthoc_visualization_only", "Locked-test observed quality-label count."))
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = ["metric", "value", "allowed_use", "interpretation"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    lines = [
        "# Quality Manifest Audit",
        "",
        "| Metric | Value | Allowed use | Interpretation |",
        "|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['metric']} | {row['value']} | {row['allowed_use']} | {row['interpretation']} |")
    lines.extend(
        [
            "",
            "Guardrail: EyeQ labels outside the clean training manifest are restricted to post-hoc visualization. They are never used for grade-model training, fine-tuning, calibration thresholds, or decision-making.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_manifest", required=True)
    parser.add_argument("--clean_quality_manifest", required=True)
    parser.add_argument("--eyeq_manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    rows = build_audit_rows(
        read_rows(Path(args.master_manifest)),
        read_rows(Path(args.clean_quality_manifest)),
        read_rows(Path(args.eyeq_manifest)),
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "quality_manifest_audit.csv", rows)
    write_markdown(out_dir / "quality_manifest_audit.md", rows)
    (out_dir / "quality_manifest_audit.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[OK] audit rows: {len(rows)}")
    for row in rows:
        print(f"{row['metric']}={row['value']} allowed_use={row['allowed_use']}")


if __name__ == "__main__":
    main()
