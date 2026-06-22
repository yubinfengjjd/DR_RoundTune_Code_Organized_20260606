"""Generate publication-ready ablation figures from verified summary tables."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RANK_TABLE = (
    PROJECT_ROOT
    / "checkpoints"
    / "Fine-tuning ablation"
    / "full_pipeline_rank_aligned"
    / "full_pipeline_rank_aligned_fit_table_synced.csv"
)
DEFAULT_QUALITY_TABLE = (
    PROJECT_ROOT
    / "checkpoints"
    / "Quality encoder ablation"
    / "lora16_quality_vs_no_quality"
    / "quality_pretraining_lora16_fit_table.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "checkpoints" / "paper_assets" / "revision_round2" / "figures"
)
RANK_PATTERN = re.compile(r"^LoRA r=(\d+)$")


def read_rows(path: Path) -> List[Dict[str, str]]:
    """Read a CSV summary table without mutating the source artifact."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"Missing required numeric field {key!r} in row: {row}")
    return float(value)


def rank_points(rows: Iterable[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Extract numeric LoRA rank rows in ascending-rank order."""
    points: List[Dict[str, Any]] = []
    for row in rows:
        match = RANK_PATTERN.fullmatch(row.get("setting", ""))
        if not match:
            continue
        points.append(
            {
                "rank": int(match.group(1)),
                "test_qwk": _as_float(row, "grade_test_qwk"),
                "trainable_params": int(float(row["grade_trainable_params"])),
            }
        )
    if not points:
        raise ValueError("No LoRA rank rows were found in the rank-aligned table.")
    return sorted(points, key=lambda point: point["rank"])


def quality_points(rows: Iterable[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Extract with-quality and no-quality control metrics in display order."""
    points: List[Dict[str, Any]] = []
    setting_aliases = {
        "With quality-aware pretraining": "With quality pretraining",
        "With quality pretraining (LoRA r16)": "With quality pretraining",
        "Without quality-aware pretraining": "Without quality pretraining",
        "No quality pretraining": "Without quality pretraining",
    }
    for row in rows:
        setting = setting_aliases.get(row.get("setting", ""))
        if setting is None:
            continue
        points.append(
            {
                "setting": setting,
                "test_qwk": _as_float(row, "test_qwk"),
                "test_acc": _as_float(row, "test_acc"),
                "test_macro_f1": _as_float(row, "test_macro_f1"),
            }
        )
    order = {
        "With quality pretraining": 0,
        "Without quality pretraining": 1,
    }
    points.sort(key=lambda point: order[point["setting"]])
    if len(points) != 2:
        raise ValueError("Expected exactly two quality-pretraining control rows.")
    return points


def _configure_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def plot_rank_ablation(points: List[Dict[str, Any]], output_dir: Path) -> List[Path]:
    """Plot LoRA rank performance and parameter-efficiency panels."""
    import matplotlib.pyplot as plt

    ranks = [point["rank"] for point in points]
    qwks = [point["test_qwk"] for point in points]
    params_m = [point["trainable_params"] / 1_000_000 for point in points]
    colors = ["#2C7FB8" if rank == 16 else "#7FCDBB" for rank in ranks]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), constrained_layout=True)
    axes[0].plot(ranks, qwks, color="#355C7D", marker="o", linewidth=1.8)
    axes[0].scatter(ranks, qwks, c=colors, edgecolor="#1F2933", linewidth=0.7, s=58, zorder=3)
    axes[0].set_xlabel("LoRA rank")
    axes[0].set_ylabel("Internal test QWK")
    axes[0].set_xticks(ranks)
    axes[0].set_title("Performance")
    for rank, qwk in zip(ranks, qwks):
        axes[0].annotate(f"{qwk:.4f}", (rank, qwk), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)

    axes[1].bar([str(rank) for rank in ranks], params_m, color=colors, edgecolor="#1F2933", linewidth=0.6)
    axes[1].set_xlabel("LoRA rank")
    axes[1].set_ylabel("Trainable parameters (M)")
    axes[1].set_title("Parameter cost")
    for index, value in enumerate(params_m):
        axes[1].text(index, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Full-pipeline rank-aligned LoRA ablation", fontsize=11)
    output_base = output_dir / "ablation_lora_rank_qwk_params"
    paths = [output_base.with_suffix(".png"), output_base.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_quality_control(points: List[Dict[str, Any]], output_dir: Path) -> List[Path]:
    """Plot the native grade metrics for the quality-pretraining control."""
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ["With quality-aware\npretraining", "Without quality-aware\npretraining"]
    metric_specs = [
        ("test_qwk", "QWK"),
        ("test_acc", "Accuracy"),
        ("test_macro_f1", "Macro-F1"),
    ]
    x = np.arange(len(metric_specs))
    width = 0.34
    colors = ["#2C7FB8", "#F28E2B"]

    fig, ax = plt.subplots(figsize=(6.8, 3.5), constrained_layout=True)
    for index, point in enumerate(points):
        values = [point[key] for key, _ in metric_specs]
        positions = x + (index - 0.5) * width
        bars = ax.bar(
            positions,
            values,
            width,
            label=labels[index],
            color=colors[index],
            edgecolor="#1F2933",
            linewidth=0.5,
        )
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(x, [label for _, label in metric_specs])
    ax.set_ylabel("Internal test metric")
    ax.set_ylim(0, 0.85)
    ax.set_title("Quality-aware pretraining control (LoRA r=16)")
    ax.legend(frameon=False, loc="upper right", fontsize=8)

    output_base = output_dir / "ablation_quality_pretraining_control"
    paths = [output_base.with_suffix(".png"), output_base.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-table", type=Path, default=DEFAULT_RANK_TABLE)
    parser.add_argument("--quality-table", type=Path, default=DEFAULT_QUALITY_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        *plot_rank_ablation(rank_points(read_rows(args.rank_table)), args.output_dir),
        *plot_quality_control(quality_points(read_rows(args.quality_table)), args.output_dir),
    ]
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
