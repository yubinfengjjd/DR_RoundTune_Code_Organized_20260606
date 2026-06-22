"""Uncertainty comparison + selective prediction (no leakage).

Compares 3 uncertainty signals under the same evaluation protocol:
- EDL-u: u = K / S where S = sum(alpha), alpha = evidence(logits) + 1
- Softmax entropy: H(p) from a single model
- Deep ensemble disagreement: MI = H(mean p) - mean H(p_m) from multi-seed models

Protocol (strict):
- All post-hoc thresholds for DR grade prediction are tuned on calib only.
- Selective-prediction reject threshold tau is tuned on calib only.
- Test is evaluated once per configured tau (reported as coverage points).

Inputs are CSV artifacts produced by src/eval_grade_ce_posthoc.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
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
    mask = (y_true >= 0) & (y_pred >= 0)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return float("nan")
    if np.unique(y_true).size <= 1 and np.unique(y_pred).size <= 1:
        return float("nan")
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y_true.clip(0, k - 1), y_pred.clip(0, k - 1)), 1)
    return _qwk_from_confusion(cm)


def acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    m = (y_true >= 0) & (y_pred >= 0)
    if m.sum() == 0:
        return float("nan")
    return float((y_true[m] == y_pred[m]).mean())


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


def auroc_binary(y_true01: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true01).astype(int)
    s = np.asarray(score).astype(float)
    mask = np.isfinite(s) & ((y == 0) | (y == 1))
    y = y[mask]
    s = s[mask]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)

    # average ranks for ties
    uniq, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    for i, cnt in enumerate(counts):
        if cnt <= 1:
            continue
        idx = np.where(inv == i)[0]
        ranks[idx] = ranks[idx].mean()

    sum_ranks_pos = float(ranks[y == 1].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def auprc_binary(y_true01: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true01).astype(int)
    s = np.asarray(score).astype(float)
    mask = np.isfinite(s) & ((y == 0) | (y == 1))
    y = y[mask]
    s = s[mask]
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-s)
    y = y[order]
    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0
    for i in range(len(y)):
        if y[i] == 1:
            tp += 1
        else:
            fp += 1
        prec = tp / max(1, tp + fp)
        rec = tp / n_pos
        ap += prec * (rec - prev_recall)
        prev_recall = rec
    return float(ap)


def entropy_from_probs(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return (-np.sum(p * np.log(p), axis=1)).astype(np.float32)


def edl_u_from_logits(logits: np.ndarray, evidence: str = "relu") -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    m = str(evidence).lower().strip()
    if m == "relu":
        e = np.maximum(z, 0.0)
    elif m == "softplus":
        e = np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)
    else:
        raise ValueError("--edl_evidence must be relu/softplus")
    alpha = e + 1.0
    S = alpha.sum(axis=1, keepdims=True)
    K = alpha.shape[1]
    u = (K / np.maximum(S, 1e-9)).reshape(-1)
    return u.astype(np.float32)


@dataclass
class Preds:
    ids: np.ndarray  # (N,) unique key for alignment
    y_true: np.ndarray  # (N,)
    pred_grade: np.ndarray  # (N,) thresholded pred if present else -1
    pred_argmax: np.ndarray  # (N,)
    probs: np.ndarray  # (N,5)
    logits: Optional[np.ndarray]  # (N,5)
    edl_u: Optional[np.ndarray]  # (N,) if present


def read_preds_csv(path: Path) -> Preds:
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")

    def get_int(col: str, default: int = -1) -> np.ndarray:
        out = np.full(len(rows), default, dtype=int)
        for i, row in enumerate(rows):
            v = row.get(col, "")
            if v is None or v == "":
                continue
            out[i] = int(float(v))
        return out

    def get_float(col: str, default: float = np.nan) -> np.ndarray:
        out = np.full(len(rows), default, dtype=np.float32)
        for i, row in enumerate(rows):
            v = row.get(col, "")
            if v is None or v == "":
                continue
            out[i] = float(v)
        return out

    keys = np.array([str(row.get("key", "")).strip() for row in rows], dtype=object)
    paths = np.array([_norm_path(str(row.get("path", ""))) for row in rows], dtype=object)
    ids = np.where(keys != "", keys, paths)

    y_true = get_int("true_grade", -1)
    pred_grade = get_int("pred_grade", -1)
    pred_argmax = get_int("pred_argmax", -1)

    probs = np.stack([get_float(f"p{i}") for i in range(5)], axis=1)
    logits: Optional[np.ndarray] = None
    if "logit0" in rows[0]:
        logits = np.stack([get_float(f"logit{i}") for i in range(5)], axis=1)

    edl_u: Optional[np.ndarray] = None
    if "edl_u" in rows[0]:
        edl_u = get_float("edl_u", default=np.nan)

    return Preds(ids=ids, y_true=y_true, pred_grade=pred_grade, pred_argmax=pred_argmax, probs=probs, logits=logits, edl_u=edl_u)


def align_to(ref: Preds, other: Preds) -> Tuple[Preds, Preds]:
    idx = {str(k): i for i, k in enumerate(ref.ids)}
    pairs: List[Tuple[int, int]] = []
    for j, k in enumerate(other.ids):
        kk = str(k)
        if kk in idx:
            pairs.append((idx[kk], j))
    if not pairs:
        raise ValueError("No overlapping ids for alignment")
    pairs.sort(key=lambda x: x[0])
    ri = np.array([a for a, _ in pairs], dtype=int)
    oi = np.array([b for _, b in pairs], dtype=int)

    def take(p: Preds, sel: np.ndarray) -> Preds:
        return Preds(
            ids=p.ids[sel],
            y_true=p.y_true[sel],
            pred_grade=p.pred_grade[sel],
            pred_argmax=p.pred_argmax[sel],
            probs=p.probs[sel],
            logits=(p.logits[sel] if p.logits is not None else None),
            edl_u=(p.edl_u[sel] if p.edl_u is not None else None),
        )

    return take(ref, ri), take(other, oi)


def error_labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    m = (y_true >= 0) & (y_pred >= 0)
    err = np.zeros_like(y_true, dtype=int)
    err[m] = (y_true[m] != y_pred[m]).astype(int)
    return err


def ensemble_mi(probs_list: List[Preds]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    P = np.stack([p.probs for p in probs_list], axis=0).astype(np.float64)  # (M,N,K)
    P = np.clip(P, 1e-12, 1.0)
    P = P / P.sum(axis=2, keepdims=True)
    P_bar = P.mean(axis=0)
    H_bar = entropy_from_probs(P_bar.astype(np.float32)).astype(np.float64)
    H_m = np.stack([entropy_from_probs(P[m].astype(np.float32)) for m in range(P.shape[0])], axis=0).mean(axis=0).astype(np.float64)
    MI = (H_bar - H_m).astype(np.float32)
    return MI, H_bar.astype(np.float32), H_m.astype(np.float32)


def filter_to_ids(p: Preds, keep_ids: np.ndarray) -> Preds:
    keep_set = {str(x) for x in keep_ids.tolist()}
    m = np.array([str(x) in keep_set for x in p.ids], dtype=bool)
    return Preds(
        ids=p.ids[m],
        y_true=p.y_true[m],
        pred_grade=p.pred_grade[m],
        pred_argmax=p.pred_argmax[m],
        probs=p.probs[m],
        logits=(p.logits[m] if p.logits is not None else None),
        edl_u=(p.edl_u[m] if p.edl_u is not None else None),
    )


def run_eval_posthoc(
    project_root: Path,
    run_dir: Path,
    tta: int,
    *,
    device: str,
    grid_step: float,
    search_iters: int,
    eval_script: Path,
) -> None:
    # Disable patient-level for speed by passing empty string.
    cmd_calib = [
        sys.executable,
        "-u",
        str(eval_script),
        "--project_root",
        str(project_root),
        "--run_dir",
        str(run_dir),
        "--do_calib",
        "--tta",
        str(int(tta)),
        "--threshold_grid_step",
        str(float(grid_step)),
        "--threshold_search_iters",
        str(int(search_iters)),
        "--device",
        str(device),
    ]
    cmd_test = [
        sys.executable,
        "-u",
        str(eval_script),
        "--project_root",
        str(project_root),
        "--run_dir",
        str(run_dir),
        "--do_test",
        "--tta",
        str(int(tta)),
        "--patient_aggs",
        "",
        "--device",
        str(device),
    ]
    subprocess.check_call(cmd_calib, cwd=str(project_root))
    subprocess.check_call(cmd_test, cwd=str(project_root))


def selective_points(
    *,
    name: str,
    calib: Preds,
    test: Preds,
    u_calib: np.ndarray,
    u_test: np.ndarray,
    coverages: List[float],
    mode: str = "global",
    stratified_min_bin: int = 30,
    k: int = 5,
) -> List[Dict[str, Any]]:
    # Use thresholded pred if present; else fallback to argmax.
    ypc = np.where(calib.pred_grade >= 0, calib.pred_grade, calib.pred_argmax)
    ypt = np.where(test.pred_grade >= 0, test.pred_grade, test.pred_argmax)
    mcal = (calib.y_true >= 0) & (ypc >= 0) & np.isfinite(u_calib)
    mtes = (test.y_true >= 0) & (ypt >= 0) & np.isfinite(u_test)

    if int(mcal.sum()) == 0:
        raise RuntimeError(
            f"No valid calib samples for tau selection in '{name}'. "
            f"This usually means calib split is empty after alignment/filtering, or u_calib is all-NaN."
        )
    if int(mtes.sum()) == 0:
        raise RuntimeError(
            f"No valid test samples for evaluation in '{name}'. "
            f"This usually means test split is empty after alignment/filtering, or predictions are missing."
        )

    mode_n = str(mode).lower().strip()
    if mode_n not in ("global", "stratified_pred"):
        raise ValueError("--selective_mode must be one of: global, stratified_pred")

    # Helper: compute tau(s) from calib only.
    def _tau_global(cov_target: float) -> float:
        if cov_target >= 1.0:
            return float("inf")
        return float(np.quantile(u_calib[mcal], cov_target))

    def _tau_by_pred_class(cov_target: float) -> Dict[int, float]:
        # Stratify by predicted grade on calib (uses the same decision rule as evaluation).
        # This prevents rejection from collapsing the class distribution (which can destroy QWK).
        tau_g = _tau_global(cov_target)
        out_t: Dict[int, float] = {}
        for c in range(int(k)):
            mc = mcal & (ypc == c)
            n = int(mc.sum())
            if n < int(stratified_min_bin):
                out_t[c] = tau_g
            else:
                out_t[c] = float(np.quantile(u_calib[mc], cov_target))
        return out_t

    out: List[Dict[str, Any]] = []
    for cov_target in coverages:
        cov_target = float(cov_target)

        if mode_n == "global":
            tau = _tau_global(cov_target)
            keep = mtes & (u_test <= tau)
        else:
            tau_by = _tau_by_pred_class(cov_target)
            # Apply per-sample based on its predicted grade on test.
            tau_vec = np.full_like(u_test, fill_value=float("inf"), dtype=np.float64)
            for c, t in tau_by.items():
                tau_vec[ypt == int(c)] = float(t)
            keep = mtes & (u_test <= tau_vec)
            tau = float("nan")

        yt = test.y_true[keep]
        yp = ypt[keep]

        out.append(
            {
                "method": name,
                "mode": mode_n,
                "target_coverage": cov_target,
                "tau_from_calib": tau,
                "test_coverage": float(keep.sum() / max(1, int(mtes.sum()))),
                "n_test": int(mtes.sum()),
                "n_kept": int(keep.sum()),
                "acc": acc(yt, yp),
                "macro_f1": macro_f1(yt, yp, k=k),
                "qwk": qwk(yt, yp, k=k),
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project_root", type=str, default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--tta", type=int, default=8)
    p.add_argument("--coverages", type=str, default="1.0,0.95,0.90,0.80,0.70")
    p.add_argument("--out_dir", type=str, default="checkpoints/uncertainty_compare")

    p.add_argument(
        "--selective_mode",
        type=str,
        default="global",
        choices=["global", "stratified_pred"],
        help="How to choose reject threshold tau on calib. global=single tau; stratified_pred=per predicted class tau.",
    )
    p.add_argument(
        "--stratified_min_bin",
        type=int,
        default=30,
        help="For stratified_pred: minimum calib samples per predicted class to use per-class quantile; otherwise fallback to global tau.",
    )

    p.add_argument("--edl_run_dir", type=str, required=True)
    p.add_argument("--edl_evidence", type=str, default="relu", choices=["relu", "softplus"])

    p.add_argument("--single_run_dir", type=str, required=True)
    p.add_argument("--seed_run_dirs", type=str, required=True, help="Comma-separated list of seed run_dir (each ending with /grade)")
    p.add_argument("--ensemble_posthoc_dir", type=str, required=True)

    p.add_argument("--generate_missing", type=int, default=1, choices=[0, 1])
    p.add_argument("--threshold_grid_step", type=float, default=0.01)
    p.add_argument("--threshold_search_iters", type=int, default=15)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    project_root = Path(args.project_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (project_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tta = int(args.tta)
    coverages = [float(x) for x in str(args.coverages).split(",") if x.strip()]

    eval_script = (project_root / "src" / "eval_grade_ce_posthoc.py").resolve()
    if not eval_script.exists():
        raise FileNotFoundError(f"Missing eval script: {eval_script}")

    edl_run = Path(args.edl_run_dir).resolve()
    single_run = Path(args.single_run_dir).resolve()
    seed_runs = [Path(s.strip()).resolve() for s in str(args.seed_run_dirs).split(",") if s.strip()]
    ens_dir = Path(args.ensemble_posthoc_dir).resolve()

    # Paths: EDL
    edl_calib_csv = edl_run / f"preds_image_edl_tta{tta}_calib.csv"
    edl_test_csv = edl_run / f"preds_image_edl_tta{tta}_test.csv"

    # Paths: single
    single_calib_csv = single_run / f"preds_image_ce_tta{tta}_calib.csv"
    single_test_csv = single_run / f"preds_image_ce_tta{tta}_test.csv"

    # Paths: ensemble (already thresholded)
    ens_calib_csv = ens_dir / f"preds_image_ens_tta{tta}_calib_thresholded.csv"
    ens_test_csv = ens_dir / f"preds_image_ens_tta{tta}_test_thresholded.csv"

    if not ens_calib_csv.exists() or not ens_test_csv.exists():
        raise FileNotFoundError(f"Missing ensemble_posthoc CSVs for tta={tta}: {ens_calib_csv} / {ens_test_csv}")

    # Generate missing single/seed predictions if requested.
    if int(args.generate_missing) == 1:
        for rd in [single_run] + seed_runs:
            sc = rd / f"preds_image_ce_tta{tta}_calib.csv"
            st = rd / f"preds_image_ce_tta{tta}_test.csv"
            if sc.exists() and st.exists():
                continue
            run_eval_posthoc(
                project_root=project_root,
                run_dir=rd,
                tta=tta,
                device=str(args.device),
                grid_step=float(args.threshold_grid_step),
                search_iters=int(args.threshold_search_iters),
                eval_script=eval_script,
            )

    # Load predictions
    ens_calib = read_preds_csv(ens_calib_csv)
    ens_test = read_preds_csv(ens_test_csv)

    edl_calib = read_preds_csv(edl_calib_csv)
    edl_test = read_preds_csv(edl_test_csv)

    single_calib = read_preds_csv(single_calib_csv)
    single_test = read_preds_csv(single_test_csv)

    # Align all to ensemble index
    ens_calib, edl_calib = align_to(ens_calib, edl_calib)
    ens_test, edl_test = align_to(ens_test, edl_test)
    _, single_calib = align_to(ens_calib, single_calib)
    _, single_test = align_to(ens_test, single_test)

    seed_calib_list: List[Preds] = []
    seed_test_list: List[Preds] = []
    for rd in seed_runs:
        sc = read_preds_csv(rd / f"preds_image_ce_tta{tta}_calib.csv")
        st = read_preds_csv(rd / f"preds_image_ce_tta{tta}_test.csv")
        _, sc = align_to(ens_calib, sc)
        _, st = align_to(ens_test, st)
        seed_calib_list.append(sc)
        seed_test_list.append(st)

    # Ensure all seeds share the exact same ids within each split (robust MI stacking).
    common_cal = set(str(x) for x in ens_calib.ids.tolist())
    for s in seed_calib_list:
        common_cal &= set(str(x) for x in s.ids.tolist())
    common_cal_ids = np.array(sorted(common_cal), dtype=object)
    if common_cal_ids.size == 0:
        raise RuntimeError("No common calib ids across ensemble and seed runs")

    common_te = set(str(x) for x in ens_test.ids.tolist())
    for s in seed_test_list:
        common_te &= set(str(x) for x in s.ids.tolist())
    common_te_ids = np.array(sorted(common_te), dtype=object)
    if common_te_ids.size == 0:
        raise RuntimeError("No common test ids across ensemble and seed runs")

    # Filter per-split separately (do NOT use test ids to filter calib).
    ens_calib = filter_to_ids(ens_calib, common_cal_ids)
    edl_calib = filter_to_ids(edl_calib, common_cal_ids)
    single_calib = filter_to_ids(single_calib, common_cal_ids)
    seed_calib_list = [filter_to_ids(x, common_cal_ids) for x in seed_calib_list]

    ens_test = filter_to_ids(ens_test, common_te_ids)
    edl_test = filter_to_ids(edl_test, common_te_ids)
    single_test = filter_to_ids(single_test, common_te_ids)
    seed_test_list = [filter_to_ids(x, common_te_ids) for x in seed_test_list]

    # Uncertainty signals
    # Prefer the exact EDL u written by eval script (u averaged across TTA), fallback to recompute from logits.
    if edl_calib.edl_u is not None and np.isfinite(edl_calib.edl_u).any():
        u_edl_cal = edl_calib.edl_u.astype(np.float32)
    else:
        if edl_calib.logits is None:
            raise ValueError("EDL CSV missing edl_u and logits; cannot compute EDL-u")
        u_edl_cal = edl_u_from_logits(edl_calib.logits, evidence=str(args.edl_evidence))

    if edl_test.edl_u is not None and np.isfinite(edl_test.edl_u).any():
        u_edl_te = edl_test.edl_u.astype(np.float32)
    else:
        if edl_test.logits is None:
            raise ValueError("EDL CSV missing edl_u and logits; cannot compute EDL-u")
        u_edl_te = edl_u_from_logits(edl_test.logits, evidence=str(args.edl_evidence))

    u_ent_cal = entropy_from_probs(single_calib.probs)
    u_ent_te = entropy_from_probs(single_test.probs)

    u_mi_cal, u_entbar_cal, _ = ensemble_mi(seed_calib_list)
    u_mi_te, u_entbar_te, _ = ensemble_mi(seed_test_list)

    # Error detection on test (no leakage)
    y_pred_edl = np.where(edl_test.pred_grade >= 0, edl_test.pred_grade, edl_test.pred_argmax)
    y_pred_ent = np.where(single_test.pred_grade >= 0, single_test.pred_grade, single_test.pred_argmax)
    y_pred_ens = np.where(ens_test.pred_grade >= 0, ens_test.pred_grade, ens_test.pred_argmax)

    err_edl = error_labels(ens_test.y_true, y_pred_edl)
    err_ent = error_labels(ens_test.y_true, y_pred_ent)
    err_mi = error_labels(ens_test.y_true, y_pred_ens)
    err_entbar = error_labels(ens_test.y_true, y_pred_ens)

    auc = {
        "edl_u": {"auroc": auroc_binary(err_edl, u_edl_te), "auprc": auprc_binary(err_edl, u_edl_te), "err_rate": float(err_edl.mean())},
        "softmax_entropy": {"auroc": auroc_binary(err_ent, u_ent_te), "auprc": auprc_binary(err_ent, u_ent_te), "err_rate": float(err_ent.mean())},
        "ensemble_mi": {"auroc": auroc_binary(err_mi, u_mi_te), "auprc": auprc_binary(err_mi, u_mi_te), "err_rate": float(err_mi.mean())},
        "ensemble_pred_entropy": {"auroc": auroc_binary(err_entbar, u_entbar_te), "auprc": auprc_binary(err_entbar, u_entbar_te), "err_rate": float(err_entbar.mean())},
    }

    # Selective prediction points (tau from calib; test reported once)
    points: List[Dict[str, Any]] = []
    sel_mode = str(args.selective_mode)
    min_bin = int(args.stratified_min_bin)
    points += selective_points(name="EDL-u (K/S)", calib=edl_calib, test=edl_test, u_calib=u_edl_cal, u_test=u_edl_te, coverages=coverages, mode=sel_mode, stratified_min_bin=min_bin)
    points += selective_points(name="Softmax entropy", calib=single_calib, test=single_test, u_calib=u_ent_cal, u_test=u_ent_te, coverages=coverages, mode=sel_mode, stratified_min_bin=min_bin)
    points += selective_points(name="Ensemble MI", calib=ens_calib, test=ens_test, u_calib=u_mi_cal, u_test=u_mi_te, coverages=coverages, mode=sel_mode, stratified_min_bin=min_bin)
    points += selective_points(name="Ensemble PredEntropy", calib=ens_calib, test=ens_test, u_calib=u_entbar_cal, u_test=u_entbar_te, coverages=coverages, mode=sel_mode, stratified_min_bin=min_bin)

    # Save
    out = {
        "tta": tta,
        "coverages": coverages,
        "paths": {
            "edl_run_dir": str(edl_run),
            "single_run_dir": str(single_run),
            "seed_run_dirs": [str(x) for x in seed_runs],
            "ensemble_posthoc_dir": str(ens_dir),
        },
        "auc": auc,
        "selective": points,
    }
    out_json = out_dir / f"uncertainty_compare_tta{tta}.json"
    _write_json(out_json, out)

    # Also write a flat CSV for quick paper tables
    out_csv = out_dir / f"selective_points_tta{tta}_{sel_mode}.csv"
    _write_csv(
        out_csv,
        points,
        fieldnames=["method", "mode", "target_coverage", "tau_from_calib", "test_coverage", "n_test", "n_kept", "qwk", "acc", "macro_f1"],
    )

    print("[OK] wrote:", out_json)
    print("[OK] wrote:", out_csv)
    print("\nError detection (test):")
    for k, v in auc.items():
        print(f"- {k}: AUROC={v['auroc']:.4f} AUPRC={v['auprc']:.4f} err_rate={v['err_rate']:.3f}")


if __name__ == "__main__":
    main()



