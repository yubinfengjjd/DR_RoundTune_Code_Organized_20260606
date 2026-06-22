"""DCA Combined Plot: Multiple datasets in one figure with shared Treat-all/Treat-none.

Generates a single PDF with all datasets overlaid, sharing the same Treat-all and Treat-none lines.

Usage:
    python src/dca_combined.py \
        --csvs "Internal:path1.csv" "APTOS:path2.csv" ... \
        --endpoint 2 \
        --out_dir checkpoints/paper_assets/figures/dca_combined \
        --tag yge2
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_y_true(row: Dict[str, str]) -> int:
    for k in ("true_grade", "y_true"):
        if k in row and row[k] not in ("", None):
            return int(float(row[k]))
    raise KeyError("Missing y_true/true_grade column")


def get_probs(row: Dict[str, str]) -> np.ndarray:
    ps = []
    for i in range(5):
        v = row.get(f"p{i}", "")
        if v in ("", None):
            raise KeyError("Missing p0..p4 columns")
        ps.append(float(v))
    p = np.array(ps, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p


def net_benefit(y_true01: np.ndarray, p_hat: np.ndarray, pt: float) -> float:
    y = np.asarray(y_true01).astype(int)
    p = np.asarray(p_hat).astype(float)
    m = np.isfinite(p) & ((y == 0) | (y == 1))
    y = y[m]
    p = p[m]
    N = int(y.size)
    if N == 0:
        return float("nan")
    if pt <= 0.0:
        return float((y == 1).sum() / N)
    if pt >= 1.0:
        return 0.0
    pred = (p >= pt).astype(int)
    TP = int(((pred == 1) & (y == 1)).sum())
    FP = int(((pred == 1) & (y == 0)).sum())
    w = pt / (1.0 - pt)
    return float(TP / N - FP / N * w)


def treat_all_nb(y_true01: np.ndarray, pt: float) -> float:
    y = np.asarray(y_true01).astype(int)
    m = (y == 0) | (y == 1)
    y = y[m]
    N = int(y.size)
    if N == 0:
        return float("nan")
    if pt <= 0.0:
        return float((y == 1).sum() / N)
    if pt >= 1.0:
        return float("nan")
    TP = int((y == 1).sum())
    FP = int((y == 0).sum())
    w = pt / (1.0 - pt)
    return float(TP / N - FP / N * w)


def set_style() -> None:
    try:
        matplotlib.use("Agg", force=True)
    except Exception:
        pass
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,  # TrueType fonts (editable in Illustrator)
        "ps.fonttype": 42,   # TrueType fonts for PS/EPS
    })


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csvs", nargs="+", required=True, help="label:path pairs")
    p.add_argument("--endpoint", type=int, default=2, help="k for y>=k endpoint")
    p.add_argument("--out_dir", type=str, default="checkpoints/paper_assets/figures/dca_combined")
    p.add_argument("--tag", type=str, default="combined")
    p.add_argument("--pt_min", type=float, default=0.0)
    p.add_argument("--pt_max", type=float, default=1.0)  # Full range 0-1
    p.add_argument("--pt_step", type=float, default=0.01)
    p.add_argument("--dpi", type=int, default=600)
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    k = int(args.endpoint)
    tag = str(args.tag).strip()

    # Parse CSV inputs
    datasets: List[Tuple[str, Path]] = []
    for s in args.csvs:
        if ":" in s:
            label, path = s.split(":", 1)
            datasets.append((label.strip(), Path(path.strip()).resolve()))
        else:
            datasets.append((Path(s).stem, Path(s).resolve()))

    set_style()

    # Threshold range
    eps = 1e-6
    pts = np.arange(max(eps, args.pt_min), min(args.pt_max, 1.0 - eps) + 1e-12, args.pt_step, dtype=np.float64)

    # Color palette for datasets
    colors = ["#3498DB", "#E74C3C", "#2ECC71", "#9B59B6", "#F39C12", "#1ABC9C"]
    
    # Endpoint name mapping
    endpoint_names = {2: "RDR (y≥2)", 3: "STDR (y≥3)"}

    fig, ax = plt.subplots(figsize=(9.0, 6.5))

    # Store all net benefits to determine shared treat-all
    all_y01 = []
    dataset_results = []

    for idx, (label, csv_path) in enumerate(datasets):
        if not csv_path.exists():
            print(f"[SKIP] {label}: {csv_path} not found")
            continue

        rows = read_rows(csv_path)
        y = np.array([get_y_true(r) for r in rows], dtype=np.int64)
        probs = np.stack([get_probs(r) for r in rows], axis=0).astype(np.float64)
        
        y01 = (y >= k).astype(int)
        p_hat = probs[:, k:].sum(axis=1)
        
        all_y01.append(y01)
        
        # Calculate model net benefit
        nb_model = np.array([net_benefit(y01, p_hat, float(pt)) for pt in pts], dtype=np.float64)
        dataset_results.append((label, nb_model, colors[idx % len(colors)]))

    if not dataset_results:
        print("[ERROR] No valid datasets found")
        return

    # Calculate shared Treat-all using pooled data (or first dataset as reference)
    # Using first dataset's event rate as reference for Treat-all
    ref_y01 = all_y01[0]
    nb_treat_all = np.array([treat_all_nb(ref_y01, float(pt)) for pt in pts], dtype=np.float64)

    # ===== Plot Treat-none (baseline at y=0) =====
    ax.axhline(y=0, color='#7F8C8D', linestyle=':', linewidth=2.0, label='Treat None', zorder=1)

    # ===== Plot Treat-all (shared reference) =====
    ax.plot(pts, nb_treat_all, color='#2C3E50', linestyle='--', linewidth=2.0, 
            label='Treat All', zorder=2)

    # ===== Find the best performing model (highest average net benefit) =====
    best_idx = 0
    best_avg_nb = -np.inf
    for i, (label, nb_model, color) in enumerate(dataset_results):
        avg_nb = np.nanmean(nb_model)
        if avg_nb > best_avg_nb:
            best_avg_nb = avg_nb
            best_idx = i

    # ===== Plot each dataset's model curve =====
    for i, (label, nb_model, color) in enumerate(dataset_results):
        # Plot the curve
        ax.plot(pts, nb_model, color=color, linewidth=2.5, label=label, zorder=3)
        
        # Fill area for the BEST model only (between model curve and max(treat_all, 0))
        if i == best_idx:
            baseline = np.maximum(nb_treat_all, 0.0)
            benefit_area = np.maximum(nb_model - baseline, 0)
            ax.fill_between(pts, baseline, nb_model, 
                           where=(nb_model > baseline),
                           alpha=0.25, color=color, 
                           label=f'{label} Net Benefit', zorder=1)

    # Y-axis limits - STRICT: show ALL 4 MODEL curves completely
    # Only use MODEL curves to determine Y range (not Treat-all which goes very negative)
    model_curves_only = [nb for _, nb, _ in dataset_results]
    model_values = np.concatenate(model_curves_only)
    valid_model_values = model_values[np.isfinite(model_values)]
    
    if len(valid_model_values) > 0:
        # STRICT: use actual min/max of all MODEL curves
        y_data_min = np.min(valid_model_values)
        y_data_max = np.max(valid_model_values)
        
        # Minimal padding (5% of range)
        y_range = y_data_max - y_data_min
        pad = max(0.05 * y_range, 0.01)
        
        y_min = y_data_min - pad
        y_max = y_data_max + pad
        
        # Ensure y=0 (Treat None) is always visible
        y_min = min(y_min, -0.01)
        y_max = max(y_max, 0.01)
        
        ax.set_ylim(y_min, y_max)

    ax.set_xlabel("Threshold Probability", fontsize=13, fontweight='bold')
    ax.set_ylabel("Net Benefit", fontsize=13, fontweight='bold')
    ax.set_title(f"Decision Curve Analysis: {endpoint_names.get(k, f'y≥{k}')}", 
                fontsize=15, fontweight='bold', pad=12)
    ax.set_xlim(0.0, 1.0)  # ALWAYS show full X-axis range 0-1
    ax.set_xticks(np.arange(0, 1.1, 0.1))  # Tick marks at 0.0, 0.1, ..., 1.0

    # Legend
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=False,
             fontsize=10, edgecolor='#CCCCCC', facecolor='white')

    # Grid and spines
    ax.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    fig.tight_layout()

    out_pdf = out_dir / f"dca_combined_{tag}_yge{k}.pdf"
    out_png = out_dir / f"dca_combined_{tag}_yge{k}.png"
    fig.savefig(out_pdf, dpi=args.dpi, bbox_inches="tight", facecolor='white')
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight", facecolor='white')
    plt.close(fig)

    print(f"[OK] wrote: {out_pdf}")
    print(f"[OK] wrote: {out_png}")


if __name__ == "__main__":
    main()
