"""External validation with multi-seed ensemble (no leakage).

This script uses src/eval_external.py to run per-seed inference on external datasets
and save per-image probabilities. Then it builds an ensemble by averaging probs
across seeds on the intersection of samples.

Optionally applies internal-calibrated ordinal thresholds (expected score -> grade)
using a thresholds JSON (e.g. ensemble_posthoc/calib_thresholds_ens.json).

Design goals:
- No tuning on external labels (no leakage).
- Reproducible: writes merged CSV + metrics JSON per dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np



def _norm_path(p: str) -> str:
    return os.path.normcase(os.path.abspath(p))


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _qwk_from_confusion(cm: np.ndarray) -> float:
    cm = cm.astype(np.float64, copy=False)
    n = float(cm.sum())
    if n <= 0:
        return float("nan")
    k = int(cm.shape[0])
    row = cm.sum(axis=1)
    col = cm.sum(axis=0)
    expected = np.outer(row, col) / n
    w = (np.subtract.outer(np.arange(k), np.arange(k)) ** 2) / float((k - 1) ** 2)
    den = float((w * expected).sum())
    if den <= 0:
        return float("nan")
    num = float((w * cm).sum())
    return float(1.0 - (num / den))


def qwk(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    m = (y_true >= 0) & (y_pred >= 0)
    y_true = y_true[m]
    y_pred = y_pred[m]
    if y_true.size == 0:
        return float("nan")
    if np.unique(y_true).size <= 1 and np.unique(y_pred).size <= 1:
        return float("nan")
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y_true.clip(0, k - 1), y_pred.clip(0, k - 1)), 1)
    return _qwk_from_confusion(cm)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    m = (y_true >= 0) & (y_pred >= 0)
    y_true = y_true[m]
    y_pred = y_pred[m]
    if y_true.size == 0:
        return float("nan")
    f1s = []
    for c in range(k):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        den = (2 * tp + fp + fn)
        f1s.append((2 * tp / den) if den > 0 else 0.0)
    return float(np.mean(f1s))


def acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    m = (y_true >= 0) & (y_pred >= 0)
    if m.sum() == 0:
        return float("nan")
    return float((y_true[m] == y_pred[m]).mean())


def apply_thresholds(score: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    score = np.asarray(score, dtype=np.float32).reshape(-1)
    th = [float(x) for x in thresholds]
    if len(th) != 4:
        raise ValueError(f"thresholds must have length 4, got {len(th)}")
    pred = np.zeros(score.shape[0], dtype=np.int64)
    pred[score >= th[0]] = 1
    pred[score >= th[1]] = 2
    pred[score >= th[2]] = 3
    pred[score >= th[3]] = 4
    return pred


def expected_score(probs: np.ndarray) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return (p * np.arange(5, dtype=np.float64)[None, :]).sum(axis=1)


def read_pred_grade_csv(path: Path) -> Dict[str, Any]:
    """Read src/eval_external.py saved file pred_grade_<dataset>.csv.

    Expected header: path,y_true,y_pred,p0..p4,(p_ge1,p_ge2,p_ge3)
    """
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")

    paths = np.array([_norm_path(str(r.get("path", ""))) for r in rows], dtype=object)
    y_true = np.array([int(float(r.get("y_true", "-1"))) for r in rows], dtype=np.int64)
    y_pred = np.array([int(float(r.get("y_pred", "-1"))) for r in rows], dtype=np.int64)
    probs = np.stack([np.array([float(r.get(f"p{i}", "nan")) for i in range(5)], dtype=np.float32) for r in rows], axis=0)
    return {"paths": paths, "y_true": y_true, "y_pred": y_pred, "probs": probs}


def run_eval_external(
    project_root: Path,
    out_dir: Path,
    grade_run_dir: Path,
    *,
    prefer_processed: bool,
    device: str,
    batch_size: int,
    num_workers: int,
    amp: bool,
    grade_tta: int,
    grade_use_temperature: int,
) -> Path:
    script = (project_root / "src" / "eval_external.py").resolve()
    if not script.exists():
        raise FileNotFoundError(f"Missing: {script}")
    cmd = [
        sys.executable,
        "-u",
        str(script),
        "--project_root",
        str(project_root),
        "--out_dir",
        str(out_dir),
        "--device",
        str(device),
        "--batch_size",
        str(int(batch_size)),
        "--num_workers",
        str(int(num_workers)),
        "--grade_run_dir",
        str(grade_run_dir),
        "--grade_tta",
        str(int(grade_tta)),
        "--grade_use_temperature",
        str(int(grade_use_temperature)),
        "--save_preds",
    ]
    if prefer_processed:
        cmd.append("--prefer_processed")
    if amp:
        cmd.append("--amp")
    subprocess.check_call(cmd, cwd=str(project_root))
    return out_dir / "external_eval_summary.json"


def merge_ensemble(per_seed: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (paths_common, y_true_common, probs_avg, probs_stack).

    probs_stack: (S,N,5)
    """
    # Intersection on normalized absolute paths
    common = set(str(p) for p in per_seed[0]["paths"].tolist())
    for d in per_seed[1:]:
        common &= set(str(p) for p in d["paths"].tolist())
    common_paths = np.array(sorted(common), dtype=object)
    if common_paths.size == 0:
        raise RuntimeError("No common samples across seeds for this dataset")

    # index each seed
    idxs = []
    for d in per_seed:
        mp = {str(p): i for i, p in enumerate(d["paths"].tolist())}
        idx = np.array([mp[str(p)] for p in common_paths], dtype=int)
        idxs.append(idx)

    y_true0 = per_seed[0]["y_true"][idxs[0]]
    for s in range(1, len(per_seed)):
        yt = per_seed[s]["y_true"][idxs[s]]
        if not np.array_equal(yt, y_true0):
            # external labels should match by sample; if not, something is wrong with mapping.
            raise RuntimeError("Label mismatch across seeds after alignment; check dataset mapping")

    probs_stack = np.stack([d["probs"][idx] for d, idx in zip(per_seed, idxs)], axis=0).astype(np.float64)
    probs_stack = np.clip(probs_stack, 1e-12, 1.0)
    probs_stack = probs_stack / probs_stack.sum(axis=2, keepdims=True)
    probs_avg = probs_stack.mean(axis=0)
    return common_paths, y_true0, probs_avg.astype(np.float32), probs_stack.astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project_root", type=str, required=True)
    p.add_argument("--seed_grade_run_dirs", type=str, required=True, help="Comma-separated list of seed run_dir ending with /grade")
    p.add_argument("--out_dir", type=str, default="checkpoints/external_eval_ensemble")
    p.add_argument("--prefer_processed", type=int, default=1, choices=[0, 1])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", type=int, default=1, choices=[0, 1])
    p.add_argument("--grade_tta", type=int, default=8, help="Passed to src/eval_external.py --grade_tta")
    p.add_argument(
        "--grade_use_temperature",
        type=int,
        default=1,
        choices=[0, 1],
        help="Passed to src/eval_external.py --grade_use_temperature",
    )
    p.add_argument("--run_missing", type=int, default=1, choices=[0, 1], help="Run eval_external per seed if missing pred CSVs")
    p.add_argument("--internal_thresholds_json", type=str, default="", help="Optional thresholds json with key 'thresholds' (len=4)")
    p.add_argument(
        "--datasets",
        type=str,
        default="aptos2019,messidor2_dr_grades,ddr_dr_grading",
        help="Comma-separated dataset names to ensemble (subset of: aptos2019,messidor2_dr_grades,ddr_dr_grading)",
    )
    args = p.parse_args()

    project_root = Path(args.project_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (project_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_dirs = [Path(s.strip()).resolve() for s in str(args.seed_grade_run_dirs).split(",") if s.strip()]
    if len(seed_dirs) < 2:
        raise ValueError("Need >=2 seed run dirs for ensemble")

    thresholds: Optional[List[float]] = None
    if str(args.internal_thresholds_json).strip():
        tj = Path(str(args.internal_thresholds_json).strip())
        if not tj.is_absolute():
            tj = (project_root / tj).resolve()
        obj = _read_json(tj)
        th = obj.get("thresholds")
        if not (isinstance(th, list) and len(th) == 4):
            raise ValueError(f"Invalid thresholds json: {tj}")
        thresholds = [float(x) for x in th]

    ds_to_fname = {
        "aptos2019": "pred_grade_aptos2019.csv",
        "messidor2_dr_grades": "pred_grade_messidor2_dr_grades.csv",
        "ddr_dr_grading": "pred_grade_ddr_dr_grading.csv",
    }
    wanted = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    if not wanted:
        raise ValueError("--datasets is empty")
    for d in wanted:
        if d not in ds_to_fname:
            raise ValueError(f"Unknown dataset in --datasets: {d}")

    # Ensure per-seed external preds exist.
    per_seed_outs: List[Path] = []
    for sd in seed_dirs:
        seed_tag = sd.parent.name  # runs_grade_ordinal_seedXXX
        seed_out = out_dir / "per_seed" / seed_tag
        seed_out.mkdir(parents=True, exist_ok=True)

        req = [seed_out / ds_to_fname[d] for d in wanted]
        if int(args.run_missing) == 1 and any((not p.exists()) for p in req):
            run_eval_external(
                project_root=project_root,
                out_dir=seed_out,
                grade_run_dir=sd,
                prefer_processed=bool(int(args.prefer_processed)),
                device=str(args.device),
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                amp=bool(int(args.amp)),
                grade_tta=int(args.grade_tta),
                grade_use_temperature=int(args.grade_use_temperature),
            )
        per_seed_outs.append(seed_out)

    # Datasets to aggregate
    datasets = [(d, ds_to_fname[d]) for d in wanted]

    results: Dict[str, Any] = {"project_root": str(project_root), "seeds": [str(s) for s in seed_dirs], "thresholds": thresholds}

    for name, fname in datasets:
        seed_data = []
        missing = []
        for seed_out in per_seed_outs:
            p_csv = seed_out / fname
            if not p_csv.exists():
                missing.append(str(p_csv))
                continue
            seed_data.append(read_pred_grade_csv(p_csv))
        if missing:
            print(f"[WARN] missing {name} preds in {len(missing)} seeds; will ensemble on available intersection.", flush=True)
        if len(seed_data) < 2:
            print(f"[SKIP] {name}: not enough seed preds", flush=True)
            continue

        paths, y_true, probs_avg, probs_stack = merge_ensemble(seed_data)
        pred_argmax = probs_avg.argmax(axis=1).astype(np.int64)
        met = {"n": int(y_true.size), "acc": acc(y_true, pred_argmax), "macro_f1": macro_f1(y_true, pred_argmax), "qwk": qwk(y_true, pred_argmax)}

        out_rows = []
        score = expected_score(probs_avg)
        pred_thr = None
        met_thr = None
        if thresholds is not None:
            pred_thr = apply_thresholds(score, thresholds)
            met_thr = {"acc": acc(y_true, pred_thr), "macro_f1": macro_f1(y_true, pred_thr), "qwk": qwk(y_true, pred_thr)}

        for i in range(paths.size):
            r = {
                "path": str(paths[i]),
                "y_true": int(y_true[i]),
                "pred_argmax": int(pred_argmax[i]),
                "score": float(score[i]),
                "p0": float(probs_avg[i, 0]),
                "p1": float(probs_avg[i, 1]),
                "p2": float(probs_avg[i, 2]),
                "p3": float(probs_avg[i, 3]),
                "p4": float(probs_avg[i, 4]),
            }
            if pred_thr is not None:
                r["pred_thresholded"] = int(pred_thr[i])
            out_rows.append(r)

        out_csv = out_dir / f"ensemble_preds_{name}.csv"
        fields = ["path", "y_true", "pred_argmax", "score", "pred_thresholded", "p0", "p1", "p2", "p3", "p4"]
        _write_csv(out_csv, out_rows, fields)

        results[f"ensemble_{name}"] = {"metrics_argmax": met, "metrics_thresholded": met_thr, "csv": str(out_csv)}
        print(f"[EXT][ENSEMBLE][{name}] argmax_qwk={met['qwk']:.4f} thr_qwk={(met_thr or {}).get('qwk', float('nan')):.4f} n={met['n']}", flush=True)

    out_json = out_dir / "external_ensemble_summary.json"
    _write_json(out_json, results)
    print(f"[OK] wrote: {out_json}", flush=True)


if __name__ == "__main__":
    main()





