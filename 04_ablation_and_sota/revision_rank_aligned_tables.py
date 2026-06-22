"""Rebuild synchronized rank-aligned ablation tables from explicit sources."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


def read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def load_sources(path: Path) -> List[Dict[str, str]]:
    data = read_json(path)
    sources = data.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError(f"'sources' must be a list: {path}")
    return [dict(row) for row in sources]


def read_history(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    parsed: List[Dict[str, Any]] = []
    for row in rows:
        item: Dict[str, Any] = dict(row)
        for key in ("epoch", "loss", "acc", "macro_f1", "qwk", "time_sec"):
            if item.get(key, "") != "":
                item[key] = int(item[key]) if key == "epoch" else float(item[key])
        parsed.append(item)
    return parsed


def history_row_at(rows: Sequence[Mapping[str, Any]], epoch: int, split: str) -> Dict[str, Any]:
    for row in rows:
        if row.get("epoch") == epoch and row.get("split") == split:
            return dict(row)
    return {}


def resolve_run(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def summarize_source(project_root: Path, source: Mapping[str, str]) -> Dict[str, Any]:
    quality_run = resolve_run(project_root, source["quality_run_dir"])
    grade_run = resolve_run(project_root, source["grade_run_dir"])
    quality_summary_path = quality_run / "logs" / "summary.json"
    grade_summary_path = grade_run / "logs" / "summary.json"
    for path in (quality_summary_path, grade_summary_path, quality_run / "logs" / "history.csv", grade_run / "logs" / "history.csv"):
        if not path.exists():
            raise FileNotFoundError(f"Missing synchronized rank-table input: {path}")

    qsum = read_json(quality_summary_path)
    gsum = read_json(grade_summary_path)
    q_epoch = int(qsum["best_epoch"])
    g_epoch = int(gsum["best_epoch"])
    qhist = read_history(quality_run / "logs" / "history.csv")
    ghist = read_history(grade_run / "logs" / "history.csv")
    qtr = history_row_at(qhist, q_epoch, "train")
    qval = history_row_at(qhist, q_epoch, "val")
    gtr = history_row_at(ghist, g_epoch, "train")
    gval = history_row_at(ghist, g_epoch, "val")
    qtest = qsum.get("internal_test", {})
    gtest = gsum.get("internal_test", {})
    qft = qsum.get("init", {}).get("finetune", {})
    gft = gsum.get("init", {}).get("finetune", {})
    return {
        "setting": source["setting"],
        "name": source["name"],
        "source_role": source["source_role"],
        "quality_run_dir": str(quality_run),
        "grade_run_dir": str(grade_run),
        "quality_best_epoch": q_epoch,
        "quality_train_loss": qtr.get("loss", ""),
        "quality_train_acc": qtr.get("acc", ""),
        "quality_train_macro_f1": qtr.get("macro_f1", ""),
        "quality_val_loss": qval.get("loss", ""),
        "quality_val_acc": qval.get("acc", ""),
        "quality_val_macro_f1": qval.get("macro_f1", ""),
        "quality_test_acc": qtest.get("acc", ""),
        "quality_test_macro_f1": qtest.get("macro_f1", ""),
        "grade_best_epoch": g_epoch,
        "grade_train_loss": gtr.get("loss", ""),
        "grade_train_acc": gtr.get("acc", ""),
        "grade_train_macro_f1": gtr.get("macro_f1", ""),
        "grade_train_qwk": gtr.get("qwk", ""),
        "grade_val_loss": gval.get("loss", ""),
        "grade_val_acc": gval.get("acc", ""),
        "grade_val_macro_f1": gval.get("macro_f1", ""),
        "grade_val_qwk": gval.get("qwk", ""),
        "grade_test_acc": gtest.get("acc", ""),
        "grade_test_macro_f1": gtest.get("macro_f1", ""),
        "grade_test_qwk": gtest.get("qwk", ""),
        "grade_train_val_qwk_gap": (gtr["qwk"] - gval["qwk"]) if "qwk" in gtr and "qwk" in gval else "",
        "grade_trainable_params": gft.get("trainable_params", ""),
        "quality_trainable_params": qft.get("trainable_params", ""),
        "grade_missing": len(gsum.get("init", {}).get("missing", [])),
        "grade_unexpected": len(gsum.get("init", {}).get("unexpected", [])),
        "grade_shape_mismatch": len(gsum.get("init", {}).get("shape_mismatch", [])),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value in ("", None):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    cols = [
        ("setting", "Setting"),
        ("source_role", "Source Role"),
        ("quality_test_acc", "Qual Test Acc"),
        ("grade_train_qwk", "Grade Train QWK"),
        ("grade_val_qwk", "Grade Val QWK"),
        ("grade_test_qwk", "Grade Test QWK"),
        ("grade_train_val_qwk_gap", "Train-Val QWK Gap"),
        ("grade_test_acc", "Grade Test Acc"),
        ("grade_test_macro_f1", "Grade Test Macro-F1"),
        ("grade_trainable_params", "Grade Trainable Params"),
    ]
    lines = [
        "# Full-Pipeline Rank-Aligned Fit Table (Synchronized)",
        "",
        "The LoRA r=16 row explicitly uses mirrored submitted quality and grade artifacts. Source paths are recorded in the CSV and JSON outputs.",
        "",
        "| " + " | ".join(label for _, label in cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key, "")) for key, _ in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_tables(project_root: Path, sources_json: Path, out_dir: Path) -> List[Dict[str, Any]]:
    rows = [summarize_source(project_root, source) for source in load_sources(sources_json)]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "full_pipeline_rank_aligned_fit_table_synced.csv", rows)
    write_csv(out_dir / "full_pipeline_rank_aligned_results_synced.csv", rows)
    with (out_dir / "full_pipeline_rank_aligned_results_synced.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    write_markdown(out_dir / "full_pipeline_rank_aligned_fit_table_synced.md", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--sources_json", default="configs/revision_rank_aligned_sources.json")
    parser.add_argument("--out_dir", default="checkpoints/Fine-tuning ablation/full_pipeline_rank_aligned")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    sources = Path(args.sources_json)
    if not sources.is_absolute():
        sources = root / sources
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    rows = build_tables(root, sources, out_dir)
    print(f"[OK] synchronized rank-aligned rows: {len(rows)}")
    print(out_dir / "full_pipeline_rank_aligned_fit_table_synced.csv")


if __name__ == "__main__":
    main()
