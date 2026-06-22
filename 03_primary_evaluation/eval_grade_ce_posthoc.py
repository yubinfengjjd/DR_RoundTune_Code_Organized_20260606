"""
Post-hoc evaluation for DR grading (CE) runs.

This script is intentionally separate from the training loop. It reuses the model and dataset
definitions from `src/train_roundtune_cpf.py` to avoid duplicating preprocessing and model code.

Typical workflow (with master_split.csv that contains split={train,val_train,calib,test}):

1) Train (CE + LoRA, sampler=none)
   `python src/train_roundtune_cpf.py --task grade ... --grade_loss ce --grade_sampler none --tune_mode lora ...`

2) Calibrate thresholds on split=calib (TTA optional)
   `python src/eval_grade_ce_posthoc.py --project_root D:\\DR_RoundTune_Project --run_dir checkpoints\\runs_grade\\grade --do_calib --tta 4`

3) Evaluate split=test with the saved thresholds + patient-level aggregation
   `python src/eval_grade_ce_posthoc.py --project_root D:\\DR_RoundTune_Project --run_dir checkpoints\\runs_grade\\grade --do_test --tta 8 --patient_aggs max_score,mean_score,eye_max`

Note:
  - CE-route "grade_blend" (RMS feature + MLP) is not implemented in this script yet; the existing `--grade_blend`
    flag in `train_roundtune_cpf.py` currently supports only the regression (MSE) route.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import train_roundtune_cpf as tr


def resolve_out_dir(run_dir: Path, value: str) -> Path:
    """Keep historical in-place behavior unless an isolated output dir is requested."""
    return Path(value).resolve() if str(value).strip() else Path(run_dir).resolve()


def _edl_evidence(logits: torch.Tensor, mode: str = "relu") -> torch.Tensor:
    m = str(mode).lower().strip()
    if m == "relu":
        return F.relu(logits)
    if m == "softplus":
        return F.softplus(logits)
    raise ValueError(f"Unsupported edl_evidence='{mode}'. Choose from: relu, softplus")


def _edl_probs_from_logits(logits: torch.Tensor, evidence: str = "relu") -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (probs, S) where probs=alpha/S, alpha=evidence+1."""
    logits = logits.to(torch.float32)
    e = _edl_evidence(logits, mode=evidence)
    alpha = e + 1.0
    S = alpha.sum(dim=1, keepdim=True).clamp(min=1e-6)
    probs = alpha / S
    return probs, S


def _norm_path(p: str) -> str:
    return os.path.normcase(os.path.abspath(p))


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    tr.ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    tr.ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def eval_artifact_paths(out_dir: Path, tag: str, tta: int) -> Dict[str, Path]:
    suffix = "" if int(tta) == 8 else f"_tta{int(tta)}"
    return {
        "thresholds": out_dir / f"calib_thresholds_{tag}{suffix}.json",
        "summary": out_dir / f"eval_summary_{tag}{suffix}.json",
    }


def apply_eval_overrides(cfg: Dict[str, Any], *, num_workers: Optional[int]) -> Dict[str, Any]:
    out = dict(cfg)
    if num_workers is not None:
        out["num_workers"] = int(num_workers)
    return out


def _qwk_from_confusion(cm: np.ndarray) -> float:
    """
    Quadratic weighted kappa from a confusion matrix (KxK), numpy-only.
    """
    cm = cm.astype(np.float64, copy=False)
    n = float(cm.sum())
    if n <= 0:
        return float("nan")
    k = cm.shape[0]
    row = cm.sum(axis=1)
    col = cm.sum(axis=0)
    expected = np.outer(row, col) / n
    w = (np.subtract.outer(np.arange(k), np.arange(k)) ** 2) / float((k - 1) ** 2)
    den = float((w * expected).sum())
    if den <= 0:
        return float("nan")
    num = float((w * cm).sum())
    return float(1.0 - (num / den))


def qwk_numpy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    mask = (y_true >= 0) & (y_pred >= 0)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return float("nan")
    if np.unique(y_true).size <= 1 and np.unique(y_pred).size <= 1:
        return float("nan")
    k = int(num_classes)
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y_true.clip(0, k - 1), y_pred.clip(0, k - 1)), 1)
    return _qwk_from_confusion(cm)


def acc_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    mask = (y_true >= 0) & (y_pred >= 0)
    if mask.sum() == 0:
        return float("nan")
    return float((y_true[mask] == y_pred[mask]).mean())


def apply_thresholds(scores: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    t1, t2, t3, t4 = [float(x) for x in thresholds]
    s = np.asarray(scores, dtype=np.float32)
    pred = np.zeros_like(s, dtype=np.int64)
    pred[s >= t1] = 1
    pred[s >= t2] = 2
    pred[s >= t3] = 3
    pred[s >= t4] = 4
    return pred


@dataclass
class ThresholdSearchResult:
    thresholds: List[float]
    qwk: float
    iters: int


def search_thresholds_coord_descent(
    *,
    scores: np.ndarray,
    y_true: np.ndarray,
    grid: np.ndarray,
    init: Sequence[float] = (0.5, 1.5, 2.5, 3.5),
    max_iters: int = 15,
) -> ThresholdSearchResult:
    """
    Coordinate-descent grid search for 4 ordered thresholds that maximize QWK on (scores -> bins).
    """
    grid = np.asarray(grid, dtype=np.float32)
    thr = np.array([float(x) for x in init], dtype=np.float32)
    thr.sort()
    best_qwk = qwk_numpy(y_true, apply_thresholds(scores, thr))

    for it in range(1, int(max_iters) + 1):
        improved = False
        for i in range(4):
            lo = float(grid.min()) if i == 0 else float(thr[i - 1] + 1e-6)
            hi = float(grid.max()) if i == 3 else float(thr[i + 1] - 1e-6)
            cand = grid[(grid > lo) & (grid < hi)]
            if cand.size == 0:
                continue
            best_t = float(thr[i])
            best_local = float(best_qwk)
            for t in cand:
                tmp = thr.copy()
                tmp[i] = float(t)
                q = qwk_numpy(y_true, apply_thresholds(scores, tmp))
                if q > (best_local + 1e-12):
                    best_local = float(q)
                    best_t = float(t)
            if abs(best_t - float(thr[i])) > 1e-12:
                thr[i] = float(best_t)
                best_qwk = float(best_local)
                improved = True
        if not improved:
            return ThresholdSearchResult(thresholds=[float(x) for x in thr.tolist()], qwk=float(best_qwk), iters=it)
    return ThresholdSearchResult(thresholds=[float(x) for x in thr.tolist()], qwk=float(best_qwk), iters=int(max_iters))


def build_tta_ops(n: int) -> List[Tuple[str, Callable[[torch.Tensor], torch.Tensor]]]:
    ops: List[Tuple[str, Callable[[torch.Tensor], torch.Tensor]]] = [
        ("id", lambda x: x),
        ("hflip", lambda x: torch.flip(x, dims=[-1])),
        ("vflip", lambda x: torch.flip(x, dims=[-2])),
        ("hvflip", lambda x: torch.flip(x, dims=[-2, -1])),
        ("rot90", lambda x: torch.rot90(x, k=1, dims=[-2, -1])),
        ("rot90_hflip", lambda x: torch.flip(torch.rot90(x, k=1, dims=[-2, -1]), dims=[-1])),
        ("rot90_vflip", lambda x: torch.flip(torch.rot90(x, k=1, dims=[-2, -1]), dims=[-2])),
        ("rot90_hvflip", lambda x: torch.flip(torch.rot90(x, k=1, dims=[-2, -1]), dims=[-2, -1])),
    ]
    n = int(n)
    if n <= 1:
        return [ops[0]]
    if n <= len(ops):
        return ops[:n]
    # Repeat deterministically if user asks for more than we have unique ops.
    out: List[Tuple[str, Callable[[torch.Tensor], torch.Tensor]]] = []
    for i in range(n):
        out.append(ops[i % len(ops)])
    return out


@torch.no_grad()
def infer_logits_tta(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    tta: int,
    x_rmf: Optional[torch.Tensor] = None,
    has_rmf: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    ops = build_tta_ops(int(tta))
    logits_sum: Optional[torch.Tensor] = None
    for _, op in ops:
        out = model(op(x), x_rmf, has_rmf) if x_rmf is not None else model(op(x))
        if isinstance(out, dict):
            logits = out["grade_logits"]
        elif isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out
        logits_sum = logits if logits_sum is None else (logits_sum + logits)
    assert logits_sum is not None
    return logits_sum / float(len(ops))


@torch.no_grad()
def infer_probs_tta_edl(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    tta: int,
    evidence: str,
    x_rmf: Optional[torch.Tensor] = None,
    has_rmf: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """EDL TTA: average probabilities (not logits). Returns (probs_avg, logits_avg, S_avg, u_avg)."""
    ops = build_tta_ops(int(tta))
    probs_sum: Optional[torch.Tensor] = None
    logits_sum: Optional[torch.Tensor] = None
    S_sum: Optional[torch.Tensor] = None
    u_sum: Optional[torch.Tensor] = None
    K = 5
    for _, op in ops:
        out = model(op(x), x_rmf, has_rmf) if x_rmf is not None else model(op(x))
        if isinstance(out, dict):
            logits = out["grade_logits"]
        elif isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out
        logits = logits.to(torch.float32)
        probs, S = _edl_probs_from_logits(logits, evidence=evidence)
        u = (float(K) / S).clamp(min=0.0, max=1e6)
        probs_sum = probs if probs_sum is None else (probs_sum + probs)
        logits_sum = logits if logits_sum is None else (logits_sum + logits)
        S_sum = S if S_sum is None else (S_sum + S)
        u_sum = u if u_sum is None else (u_sum + u)
    assert probs_sum is not None and logits_sum is not None and S_sum is not None and u_sum is not None
    denom = float(len(ops))
    return probs_sum / denom, logits_sum / denom, S_sum / denom, u_sum / denom


def resolve_grade_aug(cfg: Dict[str, Any]) -> str:
    aug = str(cfg.get("grade_aug", "auto")).lower().strip()
    if aug == "auto":
        init_encoder = str(cfg.get("init_encoder_ckpt") or "").strip()
        has_init = bool(init_encoder) and init_encoder.lower() != "none"
        aug = "imagenet" if has_init else "dr"
    if aug not in ("dr", "imagenet"):
        raise ValueError(f"Unsupported grade_aug='{aug}'. Expected dr/imagenet/auto.")
    return aug


def build_grade_eval_dataset(
    *,
    rows: List[Dict[str, str]],
    project_root: Path,
    img_size: int,
    grade_aug: str,
) -> torch.utils.data.Dataset:
    if grade_aug == "imagenet":
        return tr.GradeAlbDataset(rows, project_root, img_size, is_train=False)
    return tr.GradeDataset(rows, project_root, img_size, is_train=False)


def build_grade_model_for_eval(cfg: Dict[str, Any], device: str) -> torch.nn.Module:
    backbone_name = str(cfg.get("backbone", "mae_vit_large_patch16_224")).lower().strip()
    fusion = str(cfg.get("grade_fusion", "image_only")).lower().strip()
    if fusion == "cross_attn":
        kv_img = cfg.get("crossattn_kv_img", cfg.get("kv_include_img", 0))
        model = tr.GradeCrossAttnModel(
            backbone=tr.SharedBackbone(backbone_name, pretrained=False),
            num_classes_grade=5,
            num_classes_quality=int(cfg.get("num_classes_quality", 3)),
            fusion_dim=int(cfg.get("fusion_dim", 0)),
            rmf_dim=int(cfg.get("rmf_dim", 1024)),
            num_heads=int(cfg.get("num_heads", 8)),
            dropout=float(cfg.get("dropout", 0.2)),
            residual=bool(int(cfg.get("crossattn_residual", 1))),
            kv_include_img=bool(int(kv_img)),
            rmf_missing_mode=str(cfg.get("rmf_missing_mode", "mask_token")),
            attn_scale=float(cfg.get("crossattn_scale", 1.0)),
            attn_scale_learnable=bool(int(cfg.get("crossattn_scale_learnable", 0))),
            kv_img_drop_prob=float(cfg.get("kv_img_drop_prob", 0.0)),
            kv_quality_drop_prob=float(cfg.get("kv_quality_drop_prob", 0.0)),
            kv_rmf_drop_prob=float(cfg.get("kv_rmf_drop_prob", 0.0)),
        ).to(device)
    else:
        model = tr.GradeModel(tr.SharedBackbone(backbone_name, pretrained=False), num_classes=5, dropout=0.2).to(device)
    ft_cfg = SimpleNamespace(
        tune_mode=str(cfg.get("tune_mode", "full")),
        unfreeze_last_n=int(cfg.get("unfreeze_last_n", 0) or 0),
        lora_r=int(cfg.get("lora_r", 0) or 0),
        lora_alpha=float(cfg.get("lora_alpha", 1.0) or 1.0),
        lora_dropout=float(cfg.get("lora_dropout", 0.0) or 0.0),
        lora_targets=str(cfg.get("lora_targets", "qkv,proj") or "qkv,proj"),
    )
    tr.apply_finetune_strategy(model, ft_cfg)
    return model


def load_model_ckpt_strict(model: torch.nn.Module, ckpt_path: Path, device: str) -> None:
    sd = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if not isinstance(sd, dict):
        raise TypeError(f"Unsupported checkpoint format: {ckpt_path}")
    sd = tr.strip_module_prefix(sd)
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()


def load_temperature_if_present(run_dir: Path) -> Optional[float]:
    tj = run_dir / "logs" / "temperature.json"
    if not tj.exists():
        return None
    try:
        obj = _load_json(tj)
        T = float(obj.get("T", 1.0))
        if not np.isfinite(T) or T <= 0:
            return None
        return T
    except Exception:
        return None


def _select_split_rows(rows: List[Dict[str, str]], split: str) -> List[Dict[str, str]]:
    s = str(split).lower().strip()
    out = [r for r in rows if str(r.get("split", "")).lower().strip() == s]
    if not out:
        uniq = sorted({str(r.get("split", "")).lower().strip() for r in rows if "split" in r})
        raise ValueError(f"No rows for split='{split}'. Found splits={uniq}")
    return out


@torch.no_grad()
def predict_split(
    *,
    model: torch.nn.Module,
    cfg: Dict[str, Any],
    project_root: Path,
    rows_all: List[Dict[str, str]],
    split: str,
    device: str,
    tta: int,
    temperature: Optional[float],
    out_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    rows = _select_split_rows(rows_all, split)
    img_size = int(cfg.get("img_size", 224))
    fusion = str(cfg.get("grade_fusion", "image_only")).lower().strip()
    grade_aug = resolve_grade_aug(cfg)
    rmf_dim = int(cfg.get("rmf_dim", 0) or 0)
    rmf_norm: Optional[Dict[str, Any]] = None
    if fusion == "cross_attn":
        p = out_dir / "rmf_norm.json"
        if p.exists():
            try:
                rmf_norm = _load_json(p)
                if not isinstance(rmf_norm, dict):
                    rmf_norm = None
            except Exception:
                rmf_norm = None

    # Build a stable mapping from resolved image path -> manifest metadata.
    meta: Dict[str, Dict[str, str]] = {}
    for r in rows:
        p = tr.resolve_path(project_root, r["image_path"])
        meta[_norm_path(p)] = r

    if fusion == "cross_attn":
        ds = tr.GradeRmfDataset(rows, project_root, img_size, is_train=False, rmf_norm=rmf_norm)
    else:
        ds = build_grade_eval_dataset(rows=rows, project_root=project_root, img_size=img_size, grade_aug=grade_aug)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=int(cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=True,
        drop_last=False,
        persistent_workers=(int(cfg.get("num_workers", 0)) > 0),
    )

    probs_all: List[np.ndarray] = []
    y_all: List[int] = []
    preds_argmax_all: List[int] = []
    keys_all: List[str] = []

    out_rows: List[Dict[str, Any]] = []
    for batch in dl:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].view(-1).cpu().numpy().astype(int)
        has_label = batch["has_label"].view(-1).cpu().numpy().astype(int)
        paths = list(batch["path"])

        x_rmf: Optional[torch.Tensor] = None
        has_rmf: Optional[torch.Tensor] = None
        if fusion == "cross_attn":
            x_rmf = batch["rmf"].to(device, non_blocking=True)
            has_rmf = batch.get("has_rmf", None)
            has_rmf = has_rmf.to(device, non_blocking=True).view(-1) if has_rmf is not None else None
            if x_rmf.ndim == 2 and rmf_dim > 0 and x_rmf.shape[1] != rmf_dim:
                if x_rmf.shape[1] == 1:
                    x_rmf = x_rmf.repeat(1, rmf_dim)
                else:
                    raise ValueError(f"RMF dim mismatch in eval: got {x_rmf.shape[1]} expected {rmf_dim}")

        grade_loss = str(cfg.get("grade_loss", "ce")).lower().strip()
        S_np: Optional[np.ndarray] = None
        u_np: Optional[np.ndarray] = None
        if grade_loss == "edl":
            # EDL: do NOT apply temperature scaling; average probabilities across TTA.
            evidence = str(cfg.get("edl_evidence", "relu"))
            probs_t, logits_t, S_t, u_t = infer_probs_tta_edl(model, x, tta=int(tta), evidence=evidence, x_rmf=x_rmf, has_rmf=has_rmf)
            logits_raw = logits_t.cpu()
            probs = probs_t.cpu().numpy()
            preds_argmax = probs.argmax(axis=1).astype(int)
            score = (probs * np.arange(5, dtype=np.float32)[None, :]).sum(axis=1).astype(np.float32)
            S_np = S_t.view(-1).cpu().numpy().astype(np.float32)
            u_np = u_t.view(-1).cpu().numpy().astype(np.float32)
        else:
            logits_raw = infer_logits_tta(model, x, tta=int(tta), x_rmf=x_rmf, has_rmf=has_rmf).float().cpu()
            logits_scaled = logits_raw
            if temperature is not None:
                logits_scaled = logits_raw / float(temperature)
            probs = torch.softmax(logits_scaled, dim=1).numpy()
            preds_argmax = probs.argmax(axis=1).astype(int)
            score = (probs * np.arange(5, dtype=np.float32)[None, :]).sum(axis=1).astype(np.float32)

        for i, p in enumerate(paths):
            r = meta.get(_norm_path(p), {})
            image_path = r.get("image_path", "")
            key = r.get("key", "")
            pid = r.get("patient_id", "") or tr.extract_patient_id(p)
            eye = r.get("eye", "")
            split_name = r.get("split", split)
            true_y = int(y[i])
            has_lab = int(has_label[i])

            row_out: Dict[str, Any] = {
                "image_path": image_path,
                "path": p,
                "key": key,
                "patient_id": pid,
                "eye": eye,
                "split": split_name,
                "true_grade": true_y if has_lab > 0 else -1,
                "pred_argmax": int(preds_argmax[i]),
                "score": float(score[i]),
                "logit0": float(logits_raw[i, 0]),
                "logit1": float(logits_raw[i, 1]),
                "logit2": float(logits_raw[i, 2]),
                "logit3": float(logits_raw[i, 3]),
                "logit4": float(logits_raw[i, 4]),
                "p0": float(probs[i, 0]),
                "p1": float(probs[i, 1]),
                "p2": float(probs[i, 2]),
                "p3": float(probs[i, 3]),
                "p4": float(probs[i, 4]),
            }
            if grade_loss == "edl":
                if S_np is None or u_np is None:
                    raise RuntimeError("EDL outputs missing S/u")
                row_out["edl_S"] = float(S_np[i])
                row_out["edl_u"] = float(u_np[i])
            out_rows.append(row_out)

        probs_all.append(probs)
        y_all.extend([int(v) if int(has_label[j]) > 0 else -1 for j, v in enumerate(y.tolist())])
        preds_argmax_all.extend([int(v) for v in preds_argmax.tolist()])
        keys_all.extend(paths)

    probs_cat = np.concatenate(probs_all, axis=0) if probs_all else np.zeros((0, 5), dtype=np.float32)
    y_arr = np.asarray(y_all, dtype=int)
    pred_argmax_arr = np.asarray(preds_argmax_all, dtype=int)

    metrics: Dict[str, float] = {
        "acc_argmax": acc_numpy(y_arr, pred_argmax_arr),
        "qwk_argmax": qwk_numpy(y_arr, pred_argmax_arr, num_classes=5),
    }

    # Save raw predictions (no thresholding yet).
    # Keep legacy naming unless we're explicitly in EDL mode.
    tag = "edl" if str(cfg.get("grade_loss", "")).lower().strip() == "edl" else "ce"
    preds_path = out_dir / f"preds_image_{tag}_tta{int(tta)}_{split}.csv"
    _write_csv(
        preds_path,
        out_rows,
        fieldnames=[
            "image_path",
            "path",
            "key",
            "patient_id",
            "eye",
            "split",
            "true_grade",
            "pred_argmax",
            "score",
            "logit0",
            "logit1",
            "logit2",
            "logit3",
            "logit4",
            "p0",
            "p1",
            "p2",
            "p3",
            "p4",
            "edl_S",
            "edl_u",
        ],
    )
    print(f"[OK] wrote: {preds_path}", flush=True)

    return out_rows, metrics


def _add_thresholded_preds(rows: List[Dict[str, Any]], thresholds: Sequence[float]) -> None:
    for r in rows:
        r["pred_grade"] = int(apply_thresholds(np.array([float(r["score"])], dtype=np.float32), thresholds)[0])


def _patient_agg_scores(rows: List[Dict[str, Any]], agg: str) -> List[Dict[str, Any]]:
    agg = str(agg).lower().strip()
    by_pid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        pid = str(r.get("patient_id", "")).strip() or tr.extract_patient_id(str(r.get("path", "")))
        by_pid[pid].append(r)

    out: List[Dict[str, Any]] = []
    for pid, items in by_pid.items():
        y_list = [int(x.get("true_grade", -1)) for x in items if int(x.get("true_grade", -1)) >= 0]
        true_grade = max(y_list) if y_list else -1

        scores = [float(x.get("score", 0.0)) for x in items]
        if not scores:
            continue

        if agg == "max_score":
            agg_score = float(np.max(scores))
        elif agg == "mean_score":
            agg_score = float(np.mean(scores))
        elif agg == "eye_max":
            by_eye: Dict[str, List[float]] = defaultdict(list)
            for x in items:
                eye = str(x.get("eye", "")).strip().lower()
                if not eye:
                    eye = "right" if int(tr.infer_eye_side(str(x.get("path", "")))) == 1 else "left"
                by_eye[eye].append(float(x.get("score", 0.0)))
            eye_scores = [max(v) for v in by_eye.values() if v]
            agg_score = float(np.max(eye_scores)) if eye_scores else float(np.max(scores))
        else:
            raise ValueError(f"Unsupported patient_agg='{agg}'. Choose from: max_score, mean_score, eye_max")

        out.append({"patient_id": pid, "true_grade": true_grade, "agg_score": agg_score})
    return out


def _patient_metrics(
    patient_rows: List[Dict[str, Any]],
    thresholds: Sequence[float],
) -> Dict[str, float]:
    y_true = np.array([int(r.get("true_grade", -1)) for r in patient_rows], dtype=int)
    score = np.array([float(r.get("agg_score", 0.0)) for r in patient_rows], dtype=np.float32)
    pred = apply_thresholds(score, thresholds)
    return {"acc": acc_numpy(y_true, pred), "qwk": qwk_numpy(y_true, pred, num_classes=5)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project_root", type=str, required=True)
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="", help="Optional isolated artifact directory. Defaults to run_dir.")
    default_manifest = r"data\manifests\master_split.csv"
    p.add_argument("--manifest", type=str, default=default_manifest)
    p.add_argument("--ckpt", type=str, default="best_grade.pth")

    p.add_argument("--tta", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Optional eval-only DataLoader worker override. Use 0 in restricted Windows/sandbox environments.",
    )

    p.add_argument("--do_calib", action="store_true")
    p.add_argument("--do_test", action="store_true")
    p.add_argument("--threshold_grid_step", type=float, default=0.01)
    p.add_argument("--threshold_search_iters", type=int, default=15)

    p.add_argument("--patient_aggs", type=str, default="max_score,mean_score,eye_max")

    args = p.parse_args()

    project_root = Path(args.project_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = (project_root / manifest).resolve()

    # Default remains backward-compatible. Revision post-hoc notebooks use an
    # isolated directory so frozen training folders are not modified.
    out_dir = resolve_out_dir(run_dir, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_json = run_dir / "logs" / "summary.json"
    if not summary_json.exists():
        raise FileNotFoundError(f"Missing run summary: {summary_json}")
    summary = _load_json(summary_json)
    cfg = summary.get("config", {})
    if not isinstance(cfg, dict):
        raise TypeError(f"Invalid config in {summary_json}")
    cfg = apply_eval_overrides(cfg, num_workers=args.num_workers)
    if args.num_workers is not None:
        print(f"[INFO] eval num_workers override: {cfg['num_workers']}", flush=True)

    # If user keeps default --manifest, prefer the manifest recorded in the run summary (if present).
    man_arg_norm = str(args.manifest).replace("/", "\\").strip()
    if os.path.normcase(man_arg_norm) == os.path.normcase(default_manifest):
        man_from_run = summary.get("grade_manifest") or summary.get("grade_rmf_manifest")
        if isinstance(man_from_run, str) and man_from_run.strip():
            cand = Path(man_from_run.strip())
            if not cand.is_absolute():
                cand = (project_root / cand).resolve()
            if cand.exists():
                manifest = cand
                print(f"[INFO] using manifest from run summary: {manifest}", flush=True)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = (run_dir / ckpt_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    temperature = load_temperature_if_present(run_dir)
    grade_loss = str(cfg.get("grade_loss", "ce")).lower().strip()
    if grade_loss == "edl":
        if temperature is not None:
            print("[INFO] temperature scaling ignored for grade_loss=edl.", flush=True)
        temperature = None
        print("[INFO] temperature scaling: disabled (edl)", flush=True)
    else:
        if temperature is not None:
            print(f"[INFO] temperature scaling enabled: T={temperature:.4f} (from {run_dir / 'logs' / 'temperature.json'})", flush=True)
        else:
            print("[INFO] temperature scaling: disabled (temperature.json not found)", flush=True)

    device = str(args.device)
    model = build_grade_model_for_eval(cfg, device=device)
    load_model_ckpt_strict(model, ckpt_path, device=device)
    print(f"[OK] loaded: {ckpt_path}", flush=True)

    rows_all = tr.read_csv_rows(manifest)

    calib_preds: Optional[List[Dict[str, Any]]] = None
    tag = "edl" if grade_loss == "edl" else "ce"
    artifacts = eval_artifact_paths(out_dir, tag, int(args.tta))
    thresholds_path = artifacts["thresholds"]
    thresholds: Optional[List[float]] = None

    summary_out: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "manifest": str(manifest),
        "tta": int(args.tta),
        "temperature": float(temperature) if temperature is not None else None,
        "image_level": {},
        "patient_level": {},
    }

    if args.do_calib:
        preds, mets = predict_split(
            model=model,
            cfg=cfg,
            project_root=project_root,
            rows_all=rows_all,
            split="calib",
            device=device,
            tta=int(args.tta),
            temperature=temperature,
            out_dir=out_dir,
        )
        calib_preds = preds
        summary_out["image_level"]["calib_argmax"] = mets

        # Threshold search on calib.
        y_true = np.array([int(r["true_grade"]) for r in preds], dtype=int)
        scores = np.array([float(r["score"]) for r in preds], dtype=np.float32)
        step = float(args.threshold_grid_step)
        if step <= 0:
            raise ValueError("--threshold_grid_step must be > 0")
        grid = np.arange(0.0, 4.0 + 1e-6, step, dtype=np.float32)
        res = search_thresholds_coord_descent(
            scores=scores,
            y_true=y_true,
            grid=grid,
            max_iters=int(args.threshold_search_iters),
        )
        thresholds = res.thresholds
        _add_thresholded_preds(preds, thresholds)
        y_pred = np.array([int(r["pred_grade"]) for r in preds], dtype=int)
        calib_mets_thr = {"acc": acc_numpy(y_true, y_pred), "qwk": qwk_numpy(y_true, y_pred, num_classes=5)}
        summary_out["image_level"]["calib_thresholded"] = {**calib_mets_thr, "search_qwk": float(res.qwk), "search_iters": int(res.iters)}

        _save_json(
            thresholds_path,
            {
                "thresholds": thresholds,
                "calib_qwk": float(calib_mets_thr["qwk"]),
                "tta": int(args.tta),
                "temperature": float(temperature) if temperature is not None else None,
                "grid_step": float(step),
                "search_iters": int(res.iters),
                "method": "expected_score_thresholds_max_qwk",
            },
        )
        print(f"[OK] wrote: {thresholds_path}", flush=True)

        # Re-write calib csv with pred_grade column (append).
        calib_csv = out_dir / f"preds_image_{tag}_tta{int(args.tta)}_calib.csv"
        _write_csv(
            calib_csv,
            preds,
            fieldnames=[
                "image_path",
                "path",
                "key",
                "patient_id",
                "eye",
                "split",
                "true_grade",
                "pred_argmax",
                "pred_grade",
                "score",
                "logit0",
                "logit1",
                "logit2",
                "logit3",
                "logit4",
                "p0",
                "p1",
                "p2",
                "p3",
                "p4",
            ],
        )
        print(f"[OK] wrote: {calib_csv}", flush=True)

    if args.do_test:
        if thresholds is None:
            if thresholds_path.exists():
                thresholds = [float(x) for x in _load_json(thresholds_path).get("thresholds", [])]
            else:
                raise FileNotFoundError(
                    f"Missing thresholds file: {thresholds_path}. Run with --do_calib first (or provide it in {thresholds_path})."
                )
        if thresholds is None or len(thresholds) != 4:
            raise ValueError(f"Invalid thresholds: {thresholds}")

        preds, mets = predict_split(
            model=model,
            cfg=cfg,
            project_root=project_root,
            rows_all=rows_all,
            split="test",
            device=device,
            tta=int(args.tta),
            temperature=temperature,
            out_dir=out_dir,
        )
        _add_thresholded_preds(preds, thresholds)

        y_true = np.array([int(r["true_grade"]) for r in preds], dtype=int)
        y_pred = np.array([int(r["pred_grade"]) for r in preds], dtype=int)
        test_mets_thr = {"acc": acc_numpy(y_true, y_pred), "qwk": qwk_numpy(y_true, y_pred, num_classes=5)}
        summary_out["image_level"]["test_argmax"] = mets
        summary_out["image_level"]["test_thresholded"] = test_mets_thr

        # Re-write test csv with pred_grade column (append).
        test_csv = out_dir / f"preds_image_{tag}_tta{int(args.tta)}_test.csv"
        _write_csv(
            test_csv,
            preds,
            fieldnames=[
                "image_path",
                "path",
                "key",
                "patient_id",
                "eye",
                "split",
                "true_grade",
                "pred_argmax",
                "pred_grade",
                "score",
                "logit0",
                "logit1",
                "logit2",
                "logit3",
                "logit4",
                "p0",
                "p1",
                "p2",
                "p3",
                "p4",
            ],
        )
        print(f"[OK] wrote: {test_csv}", flush=True)

        # Patient-level aggregation.
        aggs = [a.strip() for a in str(args.patient_aggs).split(",") if a.strip()]
        for agg in aggs:
            patient_rows = _patient_agg_scores(preds, agg=agg)
            for r in patient_rows:
                r["pred_grade"] = int(apply_thresholds(np.array([r["agg_score"]], dtype=np.float32), thresholds)[0])
            mets_p = _patient_metrics(patient_rows, thresholds)
            summary_out["patient_level"][agg] = mets_p

            out_patient_csv = out_dir / f"preds_patient_{tag}_agg-{agg}_tta{int(args.tta)}_test.csv"
            _write_csv(out_patient_csv, patient_rows, fieldnames=["patient_id", "true_grade", "agg_score", "pred_grade"])
            print(f"[OK] wrote: {out_patient_csv}", flush=True)

    summary_path = artifacts["summary"]

    # Preserve existing calibration keys when running test-only.
    # This prevents accidental overwrites that would blank downstream paper tables.
    if summary_path.exists():
        try:
            prev = _load_json(summary_path)
            if isinstance(prev, dict):
                prev_il = prev.get("image_level", {}) if isinstance(prev.get("image_level"), dict) else {}
                prev_pl = prev.get("patient_level", {}) if isinstance(prev.get("patient_level"), dict) else {}
                il = summary_out.get("image_level", {}) if isinstance(summary_out.get("image_level"), dict) else {}
                pl = summary_out.get("patient_level", {}) if isinstance(summary_out.get("patient_level"), dict) else {}

                # Keep calib_* if missing in current output.
                for k in ("calib_argmax", "calib_thresholded"):
                    if k not in il and k in prev_il:
                        il[k] = prev_il[k]

                # Keep any patient-level aggregates not computed this run.
                for k, v in prev_pl.items():
                    if k not in pl:
                        pl[k] = v

                summary_out["image_level"] = il
                summary_out["patient_level"] = pl
        except Exception:
            pass

    _save_json(summary_path, summary_out)
    print(f"[OK] wrote: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
