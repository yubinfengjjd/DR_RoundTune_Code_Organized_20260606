"""Calibration analysis for referral endpoints (y>=k).

Outputs:
- reliability curves (PDF) for each endpoint, overlaying multiple datasets/variants
- metrics table (ECE/Brier/mean_pred/event_rate)

Input:
--csv name:path (repeatable)
CSV must include y_true or true_grade and p0..p4.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def publication_dataset_label(name: str) -> str:
    labels = {
        "Internal_TTA1": "TTA = 1",
        "Internal_TTA4": "TTA = 4",
        "Internal_TTA8": "TTA = 8",
    }
    return labels.get(str(name), str(name))


def calibration_plot_title(endpoint: int) -> str:
    labels = {
        2: "RDR",
        3: "STDR",
    }
    endpoint = int(endpoint)
    name = labels.get(endpoint, f"grade >= {endpoint}")
    if endpoint in labels:
        return f"Calibration reliability: {name} (grade >= {endpoint})"
    return f"Calibration reliability: {name}"


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _try_int(v: Any) -> Optional[int]:
    if v in (None, ""):
        return None
    try:
        return int(float(v))
    except Exception:
        return None


def _try_float(v: Any) -> Optional[float]:
    if v in (None, ""):
        return None
    try:
        return float(v)
    except Exception:
        return None


def _row_probs(r: Dict[str, str], k: int = 5) -> Optional[np.ndarray]:
    vals: List[float] = []
    for i in range(int(k)):
        vv = _try_float(r.get(f"p{i}"))
        if vv is None:
            return None
        vals.append(float(vv))
    p = np.asarray(vals, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p


def brier_binary(y_true01: np.ndarray, y_score: np.ndarray) -> float:
    y = np.asarray(y_true01).astype(np.float64).reshape(-1)
    s = np.asarray(y_score).astype(np.float64).reshape(-1)
    m = np.isfinite(s) & ((y == 0) | (y == 1))
    y = y[m]
    s = s[m]
    if y.size == 0:
        return float("nan")
    return float(np.mean((s - y) ** 2))


def ece_binary(y_true01: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y_true01).astype(np.int64).reshape(-1)
    s = np.asarray(y_score).astype(np.float64).reshape(-1)
    m = np.isfinite(s) & ((y == 0) | (y == 1))
    y = y[m]
    s = s[m]
    if y.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    N = float(y.size)
    for i in range(int(n_bins)):
        lo, hi = float(bins[i]), float(bins[i + 1])
        if i == int(n_bins) - 1:
            mm = (s >= lo) & (s <= hi)
        else:
            mm = (s >= lo) & (s < hi)
        n = int(mm.sum())
        if n == 0:
            continue
        acc = float(y[mm].mean())
        conf = float(s[mm].mean())
        ece += (n / N) * abs(acc - conf)
    return float(ece)


def reliability_curve(y_true01: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y_true01).astype(np.int64).reshape(-1)
    s = np.asarray(y_score).astype(np.float64).reshape(-1)
    m = np.isfinite(s) & ((y == 0) | (y == 1))
    y = y[m]
    s = s[m]
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    xs: List[float] = []
    ys: List[float] = []
    ns: List[int] = []
    for i in range(int(n_bins)):
        lo, hi = float(bins[i]), float(bins[i + 1])
        if i == int(n_bins) - 1:
            mm = (s >= lo) & (s <= hi)
        else:
            mm = (s >= lo) & (s < hi)
        n = int(mm.sum())
        if n == 0:
            xs.append((lo + hi) / 2.0)
            ys.append(float("nan"))
            ns.append(0)
            continue
        xs.append(float(s[mm].mean()))
        ys.append(float(y[mm].mean()))
        ns.append(int(n))
    return np.asarray(xs), np.asarray(ys), np.asarray(ns)


@dataclass
class Dataset:
    name: str
    csv_path: str
    y_true: np.ndarray
    probs: np.ndarray


def load_dataset(name: str, csv_path: Path) -> Dataset:
    rows = _read_rows(csv_path)
    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")
    y_true: List[int] = []
    probs: List[np.ndarray] = []
    for r in rows:
        yt = None
        for k in ("true_grade", "y_true"):
            yt = _try_int(r.get(k))
            if yt is not None:
                break
        if yt is None:
            continue
        p = _row_probs(r, k=5)
        if p is None:
            continue
        y_true.append(int(yt))
        probs.append(p)
    y = np.asarray(y_true, dtype=np.int64)
    P = np.stack(probs, axis=0).astype(np.float64)
    return Dataset(name=str(name), csv_path=str(csv_path), y_true=y, probs=P)


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
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,  # TrueType fonts (editable in Illustrator)
            "ps.fonttype": 42,   # TrueType fonts for PS/EPS
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", default=[], help="Repeatable name:path")
    ap.add_argument("--out_dir", type=str, default=r"checkpoints/paper_assets/calibration")
    ap.add_argument("--tag", type=str, default="phase4")
    ap.add_argument("--endpoints", type=str, default="2,3")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    items: List[Tuple[str, Path]] = []
    for s in list(args.csv):
        if ":" not in str(s):
            raise ValueError("--csv must be name:path")
        name, p = str(s).split(":", 1)
        name = name.strip()
        pp = Path(p.strip()).resolve()
        if not pp.exists():
            raise FileNotFoundError(str(pp))
        items.append((name, pp))
    if not items:
        raise ValueError("Provide at least one --csv name:path")

    ks = [int(x) for x in str(args.endpoints).split(",") if x.strip()]
    n_bins = int(args.bins)
    if n_bins <= 1:
        raise ValueError("--bins must be >= 2")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = str(args.tag).strip() or "phase4"

    sets = [load_dataset(n, p) for n, p in items]
    set_style()

    metrics_rows: List[Dict[str, Any]] = []
    curve_rows: List[Dict[str, Any]] = []

    for k in ks:
        fig = plt.figure(figsize=(8.4, 6.2))
        ax = plt.gca()
        ax.plot([0, 1], [0, 1], linestyle=":", color="black", label="Perfect")

        for ds in sets:
            y01 = (ds.y_true >= int(k)).astype(np.int64)
            p_hat = ds.probs[:, int(k) :].sum(axis=1)
            x, y, n = reliability_curve(y01, p_hat, n_bins=n_bins)
            ax.plot(x, y, marker="o", linewidth=2.0, label=publication_dataset_label(ds.name))

            e = ece_binary(y01, p_hat, n_bins=n_bins)
            b = brier_binary(y01, p_hat)
            metrics_rows.append(
                {
                    "dataset": ds.name,
                    "endpoint": f"y>= {k}",
                    "n": int(ds.y_true.size),
                    "event_rate": float(y01.mean()) if y01.size else float("nan"),
                    "mean_pred": float(np.mean(p_hat)) if p_hat.size else float("nan"),
                    "ece": float(e),
                    "brier": float(b),
                    "csv": ds.csv_path,
                }
            )
            for i in range(int(n_bins)):
                curve_rows.append(
                    {
                        "dataset": ds.name,
                        "endpoint": f"y>= {k}",
                        "bin": int(i),
                        "mean_pred": float(x[i]),
                        "mean_obs": float(y[i]) if np.isfinite(y[i]) else "",
                        "n_bin": int(n[i]),
                    }
                )

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed fraction")
        ax.set_title(calibration_plot_title(k))
        ax.legend(loc="lower right", frameon=False)
        fig.tight_layout()

        out_pdf = out_dir / f"calibration_{tag}_yge{k}.pdf"
        fig.savefig(out_pdf, dpi=int(args.dpi), bbox_inches="tight")
        plt.close(fig)
        print("[OK] wrote:", out_pdf)

    out_metrics = out_dir / f"calibration_metrics_{tag}.csv"
    out_bins = out_dir / f"calibration_bins_{tag}.csv"
    out_json = out_dir / f"calibration_{tag}.json"

    with out_metrics.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["dataset", "endpoint", "n", "event_rate", "mean_pred", "ece", "brier", "csv"],
        )
        w.writeheader()
        for r in metrics_rows:
            w.writerow(r)

    with out_bins.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["dataset", "endpoint", "bin", "mean_pred", "mean_obs", "n_bin"],
        )
        w.writeheader()
        for r in curve_rows:
            w.writerow(r)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "tag": tag,
                "endpoints": ks,
                "bins": int(n_bins),
                "inputs": [{"name": n, "csv": str(p)} for n, p in items],
                "outputs": {"metrics_csv": str(out_metrics), "bins_csv": str(out_bins)},
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("[OK] wrote:", out_metrics)
    print("[OK] wrote:", out_bins)
    print("[OK] wrote:", out_json)


if __name__ == "__main__":
    main()
