"""Paper figure: failure case montage (top confident wrong predictions).

Input CSV must include:
- y_true: true_grade or y_true
- pred_argmax
- path (absolute path preferred) or image_path
- probabilities p0..p4

Outputs (default):
- checkpoints/paper_assets/figures/failure_cases_<tag>.pdf (+ .png)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
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


def _get_y_true(r: Dict[str, str]) -> Optional[int]:
    for k in ("true_grade", "y_true"):
        y = _try_int(r.get(k))
        if y is not None:
            return int(y)
    return None


def _row_probs(r: Dict[str, str]) -> Optional[np.ndarray]:
    ps: List[float] = []
    for i in range(5):
        v = _try_float(r.get(f"p{i}"))
        if v is None:
            return None
        ps.append(float(v))
    p = np.asarray(ps, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p


def _set_style() -> None:
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
            "pdf.fonttype": 42,  # TrueType fonts (editable in Illustrator)
            "ps.fonttype": 42,   # TrueType fonts for PS/EPS
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, required=True)
    ap.add_argument("--in_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default=r"checkpoints/paper_assets/figures")
    ap.add_argument("--tag", type=str, default="phase4")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    in_csv = Path(args.in_csv).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (project_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = str(args.tag).strip() or "phase4"

    rows = _read_rows(in_csv)
    items: List[Dict[str, Any]] = []
    for r in rows:
        yt = _get_y_true(r)
        pa = _try_int(r.get("pred_argmax"))
        if yt is None or pa is None:
            continue
        if not (0 <= int(yt) <= 4 and 0 <= int(pa) <= 4):
            continue
        if int(pa) == int(yt):
            continue
        p = _row_probs(r)
        if p is None:
            continue
        conf = float(p.max())
        path = str(r.get("path") or r.get("image_path") or "")
        if not path:
            continue
        pth = Path(path)
        if not pth.is_absolute():
            pth = (project_root / pth).resolve()
        if not pth.exists():
            continue
        items.append({"path": str(pth), "y_true": int(yt), "pred": int(pa), "conf": conf})

    if not items:
        raise RuntimeError("No wrong predictions found (or image paths missing).")

    items = sorted(items, key=lambda x: float(x["conf"]), reverse=True)[: int(args.n)]
    cols = int(args.cols)
    cols = max(1, cols)
    rows_n = int(np.ceil(len(items) / cols))

    _set_style()
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 3.2, rows_n * 3.2), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    import PIL.Image

    for i, it in enumerate(items):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        img = PIL.Image.open(it["path"]).convert("RGB")
        ax.imshow(img)
        ax.set_title(f"T={it['y_true']}  P={it['pred']}  conf={it['conf']:.2f}", fontsize=9)
        ax.axis("off")

    fig.tight_layout()
    out_pdf = out_dir / f"failure_cases_{tag}.pdf"
    out_png = out_dir / f"failure_cases_{tag}.png"
    fig.savefig(out_pdf, dpi=int(args.dpi), bbox_inches="tight")
    fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    print("[OK] wrote:", out_pdf)
    print("[OK] wrote:", out_png)


if __name__ == "__main__":
    main()
