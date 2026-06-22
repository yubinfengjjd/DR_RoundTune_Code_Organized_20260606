"""Render a post-hoc montage for frozen interpretability case selections."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Mapping, Sequence


CATEGORY_ORDER = (
    "confident_correct",
    "confident_wrong",
    "high_uncertainty",
    "low_quality",
)

CATEGORY_LABELS = {
    "confident_correct": "Confident correct",
    "confident_wrong": "Confident wrong",
    "high_uncertainty": "High uncertainty",
    "low_quality": "Low quality",
}


def read_rows(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def group_case_rows(rows: Sequence[Mapping[str, str]], n_per_category: int) -> Dict[str, List[Mapping[str, str]]]:
    grouped: Dict[str, List[Mapping[str, str]]] = {}
    for category in CATEGORY_ORDER:
        matching = [row for row in rows if row.get("category") == category]
        grouped[category] = sorted(matching, key=lambda row: int(float(row.get("rank", "0"))))[: int(n_per_category)]
    return grouped


def resolve_image_path(project_root: Path, row: Mapping[str, str]) -> Path:
    value = row.get("path") or row.get("image_path") or ""
    path = Path(value)
    return path if path.is_absolute() else (Path(project_root) / path).resolve()


def case_panel_title(item: Mapping[str, str], category: str) -> str:
    truth = item.get("true_grade", "")
    pred = item.get("pred_grade", "")
    if category == "low_quality":
        quality = float(item.get("quality_score", 0.0))
        detail = f"Quality grade {quality:g}"
    else:
        entropy = float(item.get("entropy", 0.0))
        detail = f"Entropy {entropy:.2f}"
    return f"True grade {truth} | Predicted grade {pred}\n{detail}"


def render_montage(case_csv: Path, out_dir: Path, *, project_root: Path, n_per_category: int = 4, dpi: int = 300) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from PIL import Image

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    grouped = group_case_rows(read_rows(case_csv), n_per_category=n_per_category)
    missing = [category for category, rows in grouped.items() if not rows]
    if missing:
        raise RuntimeError(f"Missing interpretability case categories: {missing}")

    fig, axes = plt.subplots(len(CATEGORY_ORDER), int(n_per_category), figsize=(3.1 * n_per_category, 3.0 * len(CATEGORY_ORDER)), squeeze=False)
    for row_idx, category in enumerate(CATEGORY_ORDER):
        for col_idx in range(int(n_per_category)):
            ax = axes[row_idx][col_idx]
            ax.axis("off")
            if col_idx >= len(grouped[category]):
                continue
            item = grouped[category][col_idx]
            path = resolve_image_path(project_root, item)
            if not path.exists():
                raise FileNotFoundError(str(path))
            with Image.open(path) as image:
                ax.imshow(image.convert("RGB"))
            ax.set_title(case_panel_title(item, category), fontsize=8)
            if col_idx == 0:
                ax.text(
                    -0.12,
                    0.5,
                    CATEGORY_LABELS[category],
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                )

    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [out_dir / "interpretability_case_montage.png", out_dir / "interpretability_case_montage.pdf"]
    for path in outputs:
        fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--case_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_per_category", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    outputs = render_montage(
        Path(args.case_csv),
        Path(args.out_dir),
        project_root=Path(args.project_root),
        n_per_category=args.n_per_category,
        dpi=args.dpi,
    )
    for path in outputs:
        print(f"[OK] wrote: {path}")


if __name__ == "__main__":
    main()
