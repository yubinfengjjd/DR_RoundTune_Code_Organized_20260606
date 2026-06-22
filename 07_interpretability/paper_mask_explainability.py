"""Paper-ready quantitative explainability using GT lesion masks.

Runs IG-vs-mask overlap (IoU/Hit/Recall/Precision) on:
- IDRiD lesion masks (data/manifests/seg_idrid.csv)
- DDR lesion masks (data/manifests/seg_ddr_lesions.csv)

Outputs:
- per-run attr_overlap.csv (per image)
- summary table with bootstrap 95% CI
- LaTeX table for supplement
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np



def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pct_ci(x: np.ndarray, alpha: float) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    lo = float(np.quantile(x, alpha / 2.0))
    hi = float(np.quantile(x, 1.0 - alpha / 2.0))
    return lo, hi


def _write_latex_table(path: Path, caption: str, label: str, header: List[str], rows: List[List[str]]) -> None:
    cols = "l" + "c" * (len(header) - 1)
    lines: List[str] = []
    lines.append("% requires: \\usepackage{booktabs,makecell}")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{cols}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(header) + r" \\")
    lines.append("\\midrule")
    for r in rows:
        rr2: List[str] = []
        for x in r:
            x = str(x)
            if "\n" in x:
                rr2.append("\\makecell{" + x.replace("\n", r"\\") + "}")
            else:
                rr2.append(x)
        lines.append(" & ".join(rr2) + r" \\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ci_cell(est: float, lo: float, hi: float, decimals: int = 3) -> str:
    if not np.isfinite(est) or not np.isfinite(lo) or not np.isfinite(hi):
        return ""
    return f"{est:.{decimals}f}\n[{lo:.{decimals}f}, {hi:.{decimals}f}]"


def _run_attr_overlap(project_root: Path, *, out_dir: Path, grade_run_dir: Path, seg_manifest: Path, ge_thr: int, device: str, limit: int, top_pct: float, ig_steps: int) -> Path:
    # run src/explainability_quant.py as a subprocess to keep logic identical
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "-u",
        str(project_root / "src" / "explainability_quant.py"),
        "--project_root",
        str(project_root),
        "--device",
        str(device),
        "attr_overlap",
        "--out_dir",
        str(out_dir),
        "--grade_run_dir",
        str(grade_run_dir),
        "--seg_manifest",
        str(seg_manifest),
        "--split",
        "test",
        "--img_size",
        "512",
        "--mask_source",
        "gt",
        "--ig_target",
        "p_ge",
        "--ig_steps",
        str(int(ig_steps)),
        "--top_pct",
        str(float(top_pct)),
        "--grade_ge_thr",
        str(int(ge_thr)),
        "--reg_score_tau",
        "0.5",
    ]
    if int(limit) > 0:
        cmd += ["--limit", str(int(limit))]
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(project_root))
    return Path(out_dir) / "attr_overlap.csv"


def _bootstrap_mean_ci(x: np.ndarray, n_boot: int, seed: int, alpha: float) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    est = float(x.mean())
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, x.size, size=(int(n_boot), int(x.size)), endpoint=False)
    boot = x[idx].mean(axis=1)
    lo, hi = _pct_ci(boot, alpha=alpha)
    return est, lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, required=True)
    ap.add_argument("--grade_run_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default=r"checkpoints/paper_assets/explain_mask")
    ap.add_argument("--tag", type=str, default="phase4")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--top_pct", type=float, default=0.1)
    ap.add_argument("--ig_steps", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (project_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = str(args.tag).strip() or "phase4"
    grade_run_dir = Path(args.grade_run_dir)
    if not grade_run_dir.is_absolute():
        grade_run_dir = (project_root / grade_run_dir).resolve()

    specs = [
        ("idrid", project_root / "data" / "manifests" / "seg_idrid.csv"),
        ("ddr", project_root / "data" / "manifests" / "seg_ddr_lesions.csv"),
    ]

    table_rows: List[Dict[str, Any]] = []
    for ds, seg_manifest in specs:
        for k in (2, 3):
            run_dir = out_dir / f"{ds}_yge{k}"
            run_dir.mkdir(parents=True, exist_ok=True)
            csv_path = _run_attr_overlap(
                project_root,
                out_dir=run_dir,
                grade_run_dir=grade_run_dir,
                seg_manifest=seg_manifest,
                ge_thr=int(k),
                device=str(args.device),
                limit=int(args.limit),
                top_pct=float(args.top_pct),
                ig_steps=int(args.ig_steps),
            )

            rows = _read_rows(csv_path)
            iou = np.array([float(r.get("iou", "nan")) for r in rows], dtype=np.float64)
            hit = np.array([float(r.get("hit", "nan")) for r in rows], dtype=np.float64)
            rec = np.array([float(r.get("recall", "nan")) for r in rows], dtype=np.float64)
            prec = np.array([float(r.get("precision", "nan")) for r in rows], dtype=np.float64)

            est_iou, lo_iou, hi_iou = _bootstrap_mean_ci(iou, n_boot=int(args.n_boot), seed=int(args.seed), alpha=float(args.alpha))
            est_hit, lo_hit, hi_hit = _bootstrap_mean_ci(hit, n_boot=int(args.n_boot), seed=int(args.seed), alpha=float(args.alpha))
            est_rec, lo_rec, hi_rec = _bootstrap_mean_ci(rec, n_boot=int(args.n_boot), seed=int(args.seed), alpha=float(args.alpha))
            est_prec, lo_prec, hi_prec = _bootstrap_mean_ci(prec, n_boot=int(args.n_boot), seed=int(args.seed), alpha=float(args.alpha))

            table_rows.append(
                {
                    "dataset": ds,
                    "endpoint": f"y>= {k}",
                    "n": int(len(rows)),
                    "mean_iou": est_iou,
                    "ci_iou_lo": lo_iou,
                    "ci_iou_hi": hi_iou,
                    "mean_hit": est_hit,
                    "ci_hit_lo": lo_hit,
                    "ci_hit_hi": hi_hit,
                    "mean_recall": est_rec,
                    "ci_recall_lo": lo_rec,
                    "ci_recall_hi": hi_rec,
                    "mean_precision": est_prec,
                    "ci_precision_lo": lo_prec,
                    "ci_precision_hi": hi_prec,
                    "attr_overlap_csv": str(csv_path),
                }
            )

    out_csv = out_dir / f"mask_explainability_summary_{tag}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        cols = [
            "dataset",
            "endpoint",
            "n",
            "mean_iou",
            "ci_iou_lo",
            "ci_iou_hi",
            "mean_hit",
            "ci_hit_lo",
            "ci_hit_hi",
            "mean_recall",
            "ci_recall_lo",
            "ci_recall_hi",
            "mean_precision",
            "ci_precision_lo",
            "ci_precision_hi",
            "attr_overlap_csv",
        ]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in table_rows:
            w.writerow(r)

    # LaTeX
    header = ["Dataset", "Endpoint", "N", "IoU", "Hit", "Recall", "Precision"]
    rows_tex: List[List[str]] = []
    for r in table_rows:
        rows_tex.append(
            [
                str(r["dataset"]).upper(),
                str(r["endpoint"]),
                str(r["n"]),
                _ci_cell(r["mean_iou"], r["ci_iou_lo"], r["ci_iou_hi"]),
                _ci_cell(r["mean_hit"], r["ci_hit_lo"], r["ci_hit_hi"]),
                _ci_cell(r["mean_recall"], r["ci_recall_lo"], r["ci_recall_hi"]),
                _ci_cell(r["mean_precision"], r["ci_precision_lo"], r["ci_precision_hi"]),
            ]
        )
    out_tex = out_dir / f"mask_explainability_{tag}.tex"
    _write_latex_table(out_tex, "Quantitative explainability: IG heatmap overlap with GT lesion masks.", f"tab:mask_explain_{tag}", header, rows_tex)

    out_json = out_dir / f"mask_explainability_{tag}.json"
    out_json.write_text(json.dumps({"rows": table_rows}, indent=2), encoding="utf-8")
    print("[OK] wrote:", out_csv)
    print("[OK] wrote:", out_tex)
    print("[OK] wrote:", out_json)


if __name__ == "__main__":
    main()



