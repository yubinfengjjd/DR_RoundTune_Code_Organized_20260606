from __future__ import annotations

# This file mirrors tools/explainability_quant.py so that all .py live under src/ for release.
# Analysis-only; no training.

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Allow running from repo root:
#   python src/explainability_quant.py ...
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import train_roundtune_cpf as rt  # noqa: E402


LESION_TYPES = ["MA", "HE", "EX", "SE"]  # order used by this repo's multilabel masks


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    return rt.read_csv_rows(path)


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _grade_pred_from_score(score: np.ndarray) -> np.ndarray:
    score = score.reshape(-1)
    pred = np.zeros_like(score, dtype=np.int64)
    pred[score >= 0.5] = 1
    pred[score >= 1.5] = 2
    pred[score >= 2.5] = 3
    pred[score >= 3.5] = 4
    return pred


def _p_ge_from_reg_score(score: np.ndarray, *, ge_thr: int, tau: float) -> np.ndarray:
    thr = float(int(ge_thr) - 0.5)
    z = (score.reshape(-1) - thr) / max(1e-6, float(tau))
    return _sigmoid_np(z)


def _infer_backbone_name(state: Dict[str, torch.Tensor]) -> str:
    inferred = rt.infer_backbone_from_state_dict_keys(list(state.keys()), state)
    if inferred:
        return inferred
    raise ValueError("Cannot infer backbone from checkpoint keys.")


def _infer_seg_decoder_from_state(state: Dict[str, torch.Tensor]) -> str:
    keys = list(state.keys())
    if any(k.startswith("decoder.aspp.") or k.startswith("decoder.low_proj.") for k in keys):
        return "deeplabv3p"
    if any(k.startswith("decoder.up") for k in keys):
        return "unet"
    if any(k.startswith("decoder.lateral.") for k in keys):
        if any(k.startswith("head.net.") for k in keys):
            return "fpn_gn2"
        return "fpn"
    if any(k.startswith("head.net.") for k in keys):
        return "fpn_gn2"
    return "fpn"


def _infer_seg_num_classes_from_state(state: Dict[str, torch.Tensor]) -> int:
    if "head.weight" in state and torch.is_tensor(state["head.weight"]):
        return int(state["head.weight"].shape[0])
    if "head.net.7.weight" in state and torch.is_tensor(state["head.net.7.weight"]):
        return int(state["head.net.7.weight"].shape[0])
    raise ValueError("Cannot infer num_classes_seg from checkpoint (missing head weights).")


def _infer_cls_num_outputs_from_state(state: Dict[str, torch.Tensor]) -> int:
    if "head.weight" in state and torch.is_tensor(state["head.weight"]):
        return int(state["head.weight"].shape[0])
    raise ValueError("Cannot infer num_outputs from checkpoint (missing head.weight).")


def _infer_lora_rank_from_state(state: Dict[str, torch.Tensor]) -> int:
    for k, v in state.items():
        if k.endswith(".lora_A") and torch.is_tensor(v) and v.ndim == 2:
            return int(v.shape[0])
    return 0


def _infer_lora_targets_from_state(state: Dict[str, torch.Tensor]) -> List[str]:
    targets = set()
    for k in state.keys():
        if ".qkv.lora_" in k:
            targets.add("qkv")
        if ".proj.lora_" in k:
            targets.add("proj")
    return sorted(list(targets))


def _maybe_apply_lora(backbone: rt.SharedBackbone, state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    r = _infer_lora_rank_from_state(state)
    if r <= 0:
        return {"enabled": False}
    targets = set(_infer_lora_targets_from_state(state)) or {"qkv", "proj"}
    n_wrapped = rt.apply_lora_to_mae_vit_backbone(backbone, r=int(r), alpha=1.0, dropout=0.0, targets=set(targets))
    return {"enabled": True, "r": int(r), "targets": sorted(list(targets)), "wrapped": int(n_wrapped)}


@dataclass
class ModelBundle:
    grade_model: rt.GradeModel
    grade_is_regression: bool
    grade_num_outputs: int
    seg_model: Optional[rt.SegModel]
    seg_num_classes: int
    seg_decoder: str
    lora: Dict[str, Any]


def _load_grade_model(project_root: Path, grade_run_dir: Path, device: str) -> Tuple[rt.GradeModel, bool, int, Dict[str, Any]]:
    ckpt = grade_run_dir / "best_grade.pth"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    state = rt.load_checkpoint_state(ckpt)
    bb = _infer_backbone_name(state)
    n_out = _infer_cls_num_outputs_from_state(state)
    is_reg = (n_out == 1)
    backbone = rt.SharedBackbone(bb)
    lora_info = _maybe_apply_lora(backbone, state)
    model = rt.GradeModel(backbone, num_classes=int(n_out), dropout=0.0).to(device)
    rt.safe_load(model, state, strict=False)
    model.eval()
    return model, is_reg, int(n_out), lora_info


def _load_seg_model(project_root: Path, seg_run_dir: Path, device: str, decoder_override: str = "") -> Tuple[rt.SegModel, int, str, Dict[str, Any]]:
    ckpt = seg_run_dir / "best_seg.pth"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    state = rt.load_checkpoint_state(ckpt)
    bb = _infer_backbone_name(state)
    dec = decoder_override.strip() or _infer_seg_decoder_from_state(state)
    num_classes = _infer_seg_num_classes_from_state(state)
    backbone = rt.SharedBackbone(bb)
    lora_info = _maybe_apply_lora(backbone, state)
    model = rt.SegModel(backbone, num_classes=int(num_classes), mask_mode="multilabel", decoder_name=dec).to(device)
    rt.safe_load(model, state, strict=False)
    model.eval()
    return model, int(num_classes), dec, lora_info


def _ig_heatmap(
    model: rt.GradeModel,
    x: torch.Tensor,
    *,
    target: str,
    ge_thr: int,
    reg_tau: float,
    steps: int,
) -> torch.Tensor:
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError(f"IG expects x shape (1,3,H,W). Got {tuple(x.shape)}")
    steps = max(1, int(steps))
    baseline = torch.zeros_like(x)
    diff = x - baseline
    grads_sum = torch.zeros_like(x)
    for i in range(1, steps + 1):
        alpha = float(i) / float(steps)
        xi = (baseline + alpha * diff).detach().requires_grad_(True)
        out, _ = model(xi)
        if out.shape[1] == 1:
            score = out[:, 0]
            if target == "p_ge":
                thr = float(int(ge_thr) - 0.5)
                p = torch.sigmoid((score - thr) / max(1e-6, float(reg_tau)))
                scalar = p.sum()
            else:
                scalar = score.sum()
        else:
            logits = out
            if target == "p_ge":
                probs = torch.softmax(logits, dim=1)
                p = probs[:, int(ge_thr) :].sum(dim=1)
                scalar = p.sum()
            elif target == "pred":
                pred = torch.argmax(logits, dim=1)
                scalar = logits[torch.arange(logits.shape[0], device=logits.device), pred].sum()
            else:
                scalar = logits.max(dim=1).values.sum()
        scalar.backward()
        assert xi.grad is not None
        grads_sum += xi.grad.detach()
    avg_grads = grads_sum / float(steps)
    ig = diff * avg_grads
    heat = ig.detach().abs().sum(dim=1, keepdim=False)[0]
    heat = heat - heat.min()
    heat = heat / (heat.max() + 1e-8)
    return heat


def _topk_mask(heat: np.ndarray, top_pct: float) -> np.ndarray:
    h = heat.reshape(-1)
    k = int(max(1, round(float(top_pct) * float(h.size))))
    thr = np.partition(h, -k)[-k]
    return (heat >= thr).astype(np.uint8)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    if union <= 0:
        return float("nan")
    return float(inter / union)


def _resolve_run_dir(project_root: Path, s: str) -> Path:
    p = Path(rt.resolve_path(project_root, s))
    return p.resolve()


def cmd_attr_overlap(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    out_dir = Path(rt.resolve_path(project_root, args.out_dir)).resolve()
    _ensure_dir(out_dir)

    grade_run_dir = _resolve_run_dir(project_root, args.grade_run_dir)
    grade_model, grade_is_reg, grade_n_out, grade_lora = _load_grade_model(project_root, grade_run_dir, args.device)

    seg_run_dir = _resolve_run_dir(project_root, args.seg_run_dir) if args.seg_run_dir else None
    seg_model = None
    seg_num_classes = 0
    seg_decoder = ""
    seg_lora: Dict[str, Any] = {"enabled": False}
    if str(args.mask_source).lower().strip() == "pred":
        if seg_run_dir is None:
            raise ValueError("--mask_source pred requires --seg_run_dir")
        seg_model, seg_num_classes, seg_decoder, seg_lora = _load_seg_model(project_root, seg_run_dir, args.device, decoder_override=str(args.seg_decoder or ""))

    seg_manifest = Path(rt.resolve_path(project_root, args.seg_manifest))
    seg_rows = _read_csv_rows(seg_manifest)
    split = str(args.split).lower().strip()
    if split and split != "all" and "split" in seg_rows[0]:
        seg_rows = [r for r in seg_rows if str(r.get("split", "")).lower() in (split, "valid" if split == "val" else split)]

    if args.limit > 0:
        seg_rows = seg_rows[: int(args.limit)]

    img_size = int(args.img_size)
    num_classes = int(args.num_classes_seg)
    if num_classes <= 0 and seg_rows:
        mp = seg_rows[0].get("mask_path", "")
        if mp:
            m = np.load(rt.resolve_path(project_root, mp))
            if m.ndim == 3:
                num_classes = int(m.shape[0] if m.shape[0] <= 16 else m.shape[-1])
    if num_classes <= 0:
        num_classes = 4

    ds = rt.SegDataset(seg_rows, project_root, img_size, False, "multilabel", num_classes)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=int(args.num_workers), pin_memory=True)

    ge_thr = int(args.grade_ge_thr)
    reg_tau = float(args.reg_score_tau)
    top_pct = float(args.top_pct)
    ig_steps = int(args.ig_steps)
    target = str(args.ig_target).lower().strip()

    rows_out: List[Dict[str, Any]] = []
    t0 = time.time()

    for i, batch in enumerate(dl, 1):
        x = batch["image"].to(args.device, non_blocking=True)
        y_mask = batch["mask"].detach().cpu().numpy()[0]
        path = str(batch.get("path", [""])[0])

        with torch.no_grad():
            out, _ = grade_model(x)
        if out.shape[1] == 1:
            score = float(out[0, 0].detach().float().cpu().item())
            pred = int(_grade_pred_from_score(np.array([score], dtype=np.float32))[0])
            p_ge = float(_p_ge_from_reg_score(np.array([score], dtype=np.float32), ge_thr=ge_thr, tau=reg_tau)[0])
        else:
            logits = out[0].detach().float().cpu()
            probs = torch.softmax(logits, dim=0).numpy()
            pred = int(np.argmax(probs))
            score = float(pred)
            p_ge = float(float(probs[int(ge_thr) :].sum()))

        if str(args.mask_source).lower().strip() == "gt":
            lesion_union = (y_mask.sum(axis=0) > 0).astype(np.uint8)
        else:
            assert seg_model is not None
            with torch.no_grad():
                slogits, _ = seg_model(x)
            prob = torch.sigmoid(slogits[0]).detach().float().cpu().numpy()
            lesion_union = (prob.max(axis=0) >= float(args.seg_thr)).astype(np.uint8)

        heat = _ig_heatmap(grade_model, x, target=target, ge_thr=ge_thr, reg_tau=reg_tau, steps=ig_steps)
        heat_np = heat.detach().float().cpu().numpy()
        top = _topk_mask(heat_np, top_pct=top_pct)

        iou = _iou(top, lesion_union)
        hit = float(lesion_union[np.unravel_index(int(np.argmax(heat_np)), heat_np.shape)] > 0)
        inter = float(np.logical_and(top > 0, lesion_union > 0).sum())
        denom_r = float(lesion_union.sum())
        denom_p = float(top.sum())
        recall = float(inter / denom_r) if denom_r > 0 else float("nan")
        precision = float(inter / denom_p) if denom_p > 0 else float("nan")

        rows_out.append(
            {
                "path": path,
                "grade_pred": int(pred),
                "grade_score": float(score),
                f"p_ge{ge_thr}": float(p_ge),
                "heat_top_pct": float(top_pct),
                "ig_steps": int(ig_steps),
                "iou": float(iou),
                "hit": float(hit),
                "recall": float(recall),
                "precision": float(precision),
            }
        )

        if (i % 25) == 0:
            print(f"[ATTR] processed {i}/{len(ds)}", flush=True)

    out_csv = out_dir / "attr_overlap.csv"
    if rows_out:
        keys = sorted({k for rr in rows_out for k in rr.keys()})
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for rr in rows_out:
                w.writerow(rr)

    summary: Dict[str, Any] = {
        "n": int(len(rows_out)),
        "elapsed_sec": float(time.time() - t0),
        "grade": {"run_dir": str(grade_run_dir), "regression": bool(grade_is_reg), "num_outputs": int(grade_n_out), "lora": grade_lora},
        "mask_source": str(args.mask_source),
        "seg": {
            "enabled": bool(seg_model is not None),
            "run_dir": str(seg_run_dir) if seg_run_dir is not None else "",
            "num_classes": int(seg_num_classes),
            "decoder": str(seg_decoder),
            "thr": float(args.seg_thr),
            "lora": seg_lora,
        },
        "ig": {"target": str(args.ig_target), "steps": int(args.ig_steps)},
        "top_pct": float(args.top_pct),
    }
    if rows_out:
        arr_iou = np.array([r["iou"] for r in rows_out if not np.isnan(r["iou"])], dtype=np.float64)
        summary["mean_iou"] = float(arr_iou.mean()) if arr_iou.size else float("nan")
        summary["mean_hit"] = float(np.mean([r["hit"] for r in rows_out]))
        arr_rec = np.array([r["recall"] for r in rows_out if not np.isnan(r["recall"])], dtype=np.float64)
        summary["mean_recall"] = float(arr_rec.mean()) if arr_rec.size else float("nan")

    _save_json(out_dir / "attr_overlap_summary.json", summary)
    print(f"[ATTR] Saved: {out_csv}", flush=True)
    print(f"[ATTR] Saved: {out_dir / 'attr_overlap_summary.json'}", flush=True)
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    p.add_argument("--project_root", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=0)

    sp2 = sub.add_parser("attr_overlap", help="IG heatmap vs lesion mask overlap (IoU/hit/recall/precision).")
    sp2.add_argument("--out_dir", type=str, required=True)
    sp2.add_argument("--grade_run_dir", type=str, required=True)
    sp2.add_argument("--seg_manifest", type=str, default="data/manifests/seg_idrid.csv")
    sp2.add_argument("--split", type=str, default="test", choices=["all", "train", "val", "valid", "test"])
    sp2.add_argument("--limit", type=int, default=0)
    sp2.add_argument("--img_size", type=int, default=512)
    sp2.add_argument("--num_classes_seg", type=int, default=0)
    sp2.add_argument("--mask_source", type=str, default="gt", choices=["gt", "pred"])
    sp2.add_argument("--seg_run_dir", type=str, default="")
    sp2.add_argument("--seg_decoder", type=str, default="")
    sp2.add_argument("--seg_thr", type=float, default=0.5)
    sp2.add_argument("--ig_target", type=str, default="p_ge", choices=["p_ge", "score", "pred"])
    sp2.add_argument("--ig_steps", type=int, default=16)
    sp2.add_argument("--top_pct", type=float, default=0.1)
    sp2.add_argument("--grade_ge_thr", type=int, default=2)
    sp2.add_argument("--reg_score_tau", type=float, default=0.5)

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.cmd == "attr_overlap":
        return cmd_attr_overlap(args)
    raise SystemExit(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
