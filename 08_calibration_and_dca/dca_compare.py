"""Compare DCA curves across multiple validation datasets in ONE figure.

Use when you want fewer figures and direct visual comparison.

Input CSV requirements (each file):
- true label column: true_grade OR y_true
- probabilities: p0..p4

Output:
- One PNG + one PDF with subplots for endpoints (k values)
- Curves CSV + summary JSON
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _get_y_true(row: Dict[str, str]) -> int:
    for k in ("true_grade", "y_true"):
        v = row.get(k, "")
        if v not in ("", None):
            return int(float(v))
    raise KeyError("Missing y_true/true_grade column")


def _get_probs(row: Dict[str, str]) -> np.ndarray:
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
    if N == 0 or pt <= 0.0 or pt >= 1.0:
        return float("nan")
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
    if N == 0 or pt <= 0.0 or pt >= 1.0:
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
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,  # TrueType fonts (editable in Illustrator)
            "ps.fonttype": 42,   # TrueType fonts for PS/EPS
        }
    )


def _parse_list_arg(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in_csvs", type=str, required=True, help="Comma-separated list of CSVs")
    p.add_argument("--labels", type=str, default="", help="Comma-separated labels (same count as in_csvs). Defaults to file stems.")
    p.add_argument("--out_dir", type=str, default=r"checkpoints/dca")
    p.add_argument("--tag", type=str, default="compare")
    p.add_argument("--endpoints", type=str, default="2,3")
    p.add_argument("--pt_min", type=float, default=0.05)
    p.add_argument("--pt_max", type=float, default=0.50)
    p.add_argument("--pt_step", type=float, default=0.01)
    p.add_argument("--ytrim_q", type=float, default=0.0, help="0 disables trimming (show full curves)")
    p.add_argument("--dpi", type=int, default=600)
    args = p.parse_args()

    in_csvs = [Path(x).resolve() for x in _parse_list_arg(args.in_csvs)]
    if not in_csvs:
        raise ValueError("--in_csvs is empty")
    labels = _parse_list_arg(args.labels)
    if labels and len(labels) != len(in_csvs):
        raise ValueError("--labels count must match --in_csvs")
    if not labels:
        labels = [p.stem for p in in_csvs]

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = str(args.tag).strip() or "compare"
    ks = [int(x) for x in str(args.endpoints).split(",") if x.strip()]

    pt_min = float(args.pt_min)
    pt_max = float(args.pt_max)
    pt_step = float(args.pt_step)
    ytrim_q = float(args.ytrim_q)
    dpi = int(args.dpi)
    if not (0.0 < pt_min < pt_max < 1.0):
        raise ValueError("Require 0 < pt_min < pt_max < 1")
    if pt_step <= 0:
        raise ValueError("pt_step must be > 0")
    if not (0.0 <= ytrim_q < 0.5):
        raise ValueError("Require 0 <= ytrim_q < 0.5")
    if dpi <= 0:
        raise ValueError("dpi must be > 0")

    # Load all datasets
    datasets: List[Dict[str, Any]] = []
    for path, name in zip(in_csvs, labels):
        rows = _read_rows(path)
        y = np.array([_get_y_true(r) for r in rows], dtype=np.int64)
        probs = np.stack([_get_probs(r) for r in rows], axis=0).astype(np.float64)
        datasets.append({"name": name, "path": str(path), "y": y, "probs": probs})

    set_style()

    pts = np.arange(pt_min, pt_max + 1e-12, pt_step, dtype=np.float64)
    curves_rows: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []

    ncols = len(ks)
    fig, axes = plt.subplots(1, ncols, figsize=(8.4 * max(1, ncols), 6.2), squeeze=False)
    axes = axes[0]

    color_cycle = list(plt.rcParams.get("axes.prop_cycle").by_key().get("color", []))
    if not color_cycle:
        color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for ax, k in zip(axes, ks):
        # Baseline for each dataset (treat all differs due to prevalence)
        ax.plot((pt_min, pt_max), (0, 0), linestyle=":", color="black", label="Treat none")

        all_vals: List[float] = [0.0]
        for i, ds in enumerate(datasets):
            y01 = (ds["y"] >= int(k)).astype(int)
            p_hat = ds["probs"][:, int(k) :].sum(axis=1)
            nb_model = np.array([net_benefit(y01, p_hat, float(pt)) for pt in pts], dtype=np.float64)
            nb_all = np.array([treat_all_nb(y01, float(pt)) for pt in pts], dtype=np.float64)

            c = color_cycle[i % len(color_cycle)]
            ax.plot(pts, nb_model, color=c, linewidth=2.0, label=f"{ds['name']} (model)")
            ax.plot(pts, nb_all, color=c, linewidth=1.2, linestyle="--", alpha=0.8, label=f"{ds['name']} (treat all)")

            for j, pt in enumerate(pts.tolist()):
                curves_rows.append(
                    {
                        "dataset": ds["name"],
                        "endpoint": f"y>= {k}",
                        "pt": float(pt),
                        "nb_model": float(nb_model[j]),
                        "nb_treat_all": float(nb_all[j]),
                        "nb_treat_none": 0.0,
                    }
                )
            all_vals.extend([float(x) for x in nb_model[np.isfinite(nb_model)].tolist()])
            all_vals.extend([float(x) for x in nb_all[np.isfinite(nb_all)].tolist()])

            jmax = int(np.nanargmax(nb_model)) if np.isfinite(nb_model).any() else 0
            summary.append(
                {
                    "dataset": ds["name"],
                    "endpoint": f"y>= {k}",
                    "pt_range": f"[{pt_min:.2f}, {pt_max:.2f}]",
                    "max_nb_pt": float(pts[jmax]),
                    "max_nb": float(nb_model[jmax]),
                    "event_rate": float(y01.mean()),
                }
            )

        # y-limits
        av = np.asarray(all_vals, dtype=np.float64)
        av = av[np.isfinite(av)]
        if av.size == 0:
            y_min, y_max = -0.05, 0.05
        else:
            y_min = float(np.quantile(av, ytrim_q)) if ytrim_q > 0 else float(av.min())
            y_min = min(y_min, 0.0)
            y_max = max(float(av.max()), 0.0)
        pad = 0.02 * (y_max - y_min if y_max > y_min else 1.0)
        ax.set_ylim(y_min - pad, y_max + pad)

        ax.set_xlim(pt_min, pt_max)
        ax.set_title(f"DCA – endpoint: y>= {k}")
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit")
        ax.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    out_png = out_dir / f"dca_compare_{tag}.png"
    out_pdf = out_dir / f"dca_compare_{tag}.pdf"
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    curves_csv = out_dir / f"dca_compare_curves_{tag}.csv"
    with curves_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "endpoint", "pt", "nb_model", "nb_treat_all", "nb_treat_none"])
        w.writeheader()
        for r in curves_rows:
            w.writerow(r)

    out_json = out_dir / f"dca_compare_summary_{tag}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "inputs": [{"label": d["name"], "csv": d["path"]} for d in datasets],
                "endpoints": ks,
                "pt_min": pt_min,
                "pt_max": pt_max,
                "pt_step": pt_step,
                "ytrim_q": ytrim_q,
                "summary": summary,
                "out_png": str(out_png),
                "out_pdf": str(out_pdf),
                "curves_csv": str(curves_csv),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("[OK] wrote:", out_png)
    print("[OK] wrote:", out_pdf)
    print("[OK] wrote:", curves_csv)
    print("[OK] wrote:", out_json)


if __name__ == "__main__":
    main()
