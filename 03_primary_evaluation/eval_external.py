from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Make sure we can import the training script as a module when running:
#   python src/eval_external.py ...
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import train_roundtune_cpf as rt  # noqa: E402

# Reuse the exact same TTA ops and logits aggregation as internal post-hoc eval.
import eval_grade_ce_posthoc as egp  # noqa: E402


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _try_load_run_cfg(run_dir: Path) -> Dict[str, Any]:
    summary = run_dir / "logs" / "summary.json"
    if summary.exists():
        try:
            return _load_json(summary).get("config", {}) or {}
        except Exception:
            return {}
    return {}


def _as_relpath(project_root: Path, p: str) -> str:
    pp = Path(p)
    if not pp.is_absolute():
        return p
    try:
        rel = pp.resolve().relative_to(project_root.resolve())
        return str(rel)
    except Exception:
        return str(pp)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()})
    return rows


def _maybe_offline_proc_map(project_root: Path) -> Dict[Tuple[str, str], str]:
    """
    Parse data/meta/manifests/manifest_offline.csv for mapping:
      - ('aptos2019', id_code) -> proc_path (relative if possible)
      - ('messidor2', image_stem) -> proc_path
    """
    mp: Dict[Tuple[str, str], str] = {}
    p = project_root / "data" / "meta" / "manifests" / "manifest_offline.csv"
    if not p.exists():
        return mp
    rows = _read_csv(p)
    for r in rows:
        if (r.get("status", "") or "").lower() != "ok":
            continue
        ds = (r.get("dataset", "") or "").strip().lower()
        sid = (r.get("sample_id", "") or "").strip()
        proc = (r.get("proc_path", "") or "").strip()
        if not (ds and sid and proc):
            continue
        parts = sid.split("__")
        if len(parts) < 3:
            continue
        # sample_id: <dataset>__<modality>__<key>__<hash>
        key = parts[2]
        if ds in ("aptos2019", "messidor2"):
            mp[(ds, key)] = _as_relpath(project_root, proc)
    return mp


def _read_ddr_txt_labels(txt_path: Path) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            fn = parts[0].strip()
            y = parts[1].strip()
            try:
                yy = int(float(y))
            except Exception:
                continue
            # DDR grading uses 0..5 in some splits; map 5 -> 4 to match ICDR 0..4.
            if yy == 5:
                yy = 4
            items.append((fn, int(yy)))
    return items


def _qwk_numpy(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5, eps: float = 1e-12) -> float:
    y_true = y_true.astype(np.int64, copy=False).reshape(-1)
    y_pred = y_pred.astype(np.int64, copy=False).reshape(-1)
    if y_true.size == 0:
        return float("nan")
    # Defensive: ignore out-of-range labels.
    m = (y_true >= 0) & (y_true < int(k)) & (y_pred >= 0) & (y_pred < int(k))
    if int(m.sum()) == 0:
        return float("nan")
    y_true = y_true[m]
    y_pred = y_pred[m]
    if int(np.count_nonzero(np.bincount(y_true, minlength=k))) <= 1:
        return float("nan")
    cm = np.zeros((k, k), dtype=np.float64)
    np.add.at(cm, (y_true, y_pred), 1)
    n = float(cm.sum())
    w = (np.subtract.outer(np.arange(k), np.arange(k)) ** 2) / float((k - 1) ** 2)
    hist_true = cm.sum(axis=1)
    hist_pred = cm.sum(axis=0)
    expected = np.outer(hist_true, hist_pred) / max(eps, n)
    den = float((w * expected).sum())
    num = float((w * cm).sum())
    if den <= 0:
        return float("nan")
    return float(1.0 - (num / den))


def _macro_f1_numpy(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    y_true = y_true.astype(np.int64, copy=False).reshape(-1)
    y_pred = y_pred.astype(np.int64, copy=False).reshape(-1)
    if y_true.size == 0:
        return float("nan")
    m = (y_true >= 0) & (y_pred >= 0)
    if int(m.sum()) == 0:
        return float("nan")
    y_true = y_true[m]
    y_pred = y_pred[m]
    k = int(k)
    f1s: List[float] = []
    for c in range(k):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        den = (2 * tp + fp + fn)
        f1s.append((2 * tp / den) if den > 0 else 0.0)
    return float(np.mean(np.array(f1s, dtype=np.float64)))


def _infer_backbone_name(state: Dict[str, torch.Tensor]) -> str:
    inferred = rt.infer_backbone_from_state_dict_keys(list(state.keys()), state)
    if inferred:
        return inferred
    raise ValueError("Cannot infer backbone from checkpoint keys. Pass --*_backbone explicitly.")


def _infer_lora_spec_from_state(state: Dict[str, torch.Tensor]) -> Tuple[int, Set[str]]:
    r = 0
    targets: Set[str] = set()
    for k, v in state.items():
        if not torch.is_tensor(v):
            continue
        if k.endswith(".lora_A") and v.ndim == 2:
            r = int(v.shape[0])
        if ".qkv.lora_" in k:
            targets.add("qkv")
        if ".proj.lora_" in k:
            targets.add("proj")
    return int(r), targets


def _maybe_apply_lora(backbone: rt.SharedBackbone, state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    r, targets = _infer_lora_spec_from_state(state)
    if r <= 0:
        return {"enabled": False}
    if not targets:
        targets = {"qkv", "proj"}
    n_wrapped = rt.apply_lora_to_mae_vit_backbone(backbone, r=r, alpha=1.0, dropout=0.0, targets=set(targets))
    return {"enabled": True, "r": int(r), "targets": sorted(list(targets)), "wrapped": int(n_wrapped)}


def _infer_seg_decoder_from_state(state: Dict[str, torch.Tensor]) -> str:
    keys = list(state.keys())
    if any(k.startswith("decoder.aspp.") or k.startswith("decoder.low_proj.") for k in keys):
        return "deeplabv3p"
    if any(k.startswith("decoder.up") for k in keys):
        return "unet"
    if any(k.startswith("decoder.lateral.") for k in keys):
        # distinguish head type
        if any(k.startswith("head.net.") for k in keys):
            return "fpn_gn2"
        return "fpn"
    if any(k.startswith("head.net.") for k in keys):
        return "fpn_gn2"
    return "fpn"


def _infer_seg_num_classes_from_state(state: Dict[str, torch.Tensor]) -> int:
    if "head.weight" in state:
        return int(state["head.weight"].shape[0])
    if "head.net.7.weight" in state:
        return int(state["head.net.7.weight"].shape[0])
    raise ValueError("Cannot infer --num_classes_seg from checkpoint (missing head weights).")


def _infer_cls_num_classes_from_state(state: Dict[str, torch.Tensor]) -> int:
    if "head.weight" in state:
        return int(state["head.weight"].shape[0])
    raise ValueError("Cannot infer num_classes from checkpoint (missing head.weight).")


@dataclass
class EvalIO:
    project_root: Path
    device: str
    batch_size: int
    num_workers: int
    amp: bool
    out_dir: Path
    save_preds: bool
    limit: int


def _make_loader(ds: torch.utils.data.Dataset, io: EvalIO, *, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=io.batch_size,
        shuffle=shuffle,
        num_workers=io.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(io.num_workers > 0),
    )


def _build_grade_dataset(rows: List[Dict[str, str]], io: EvalIO, img_size: int, grade_aug: str):
    aug = str(grade_aug).lower().strip()
    if aug == "auto":
        aug = "imagenet"
    if aug == "imagenet":
        if getattr(rt, "A", None) is None:
            raise ImportError("albumentations required for grade_aug=imagenet (install albumentations or use --grade_aug dr).")
        return rt.GradeAlbDataset(rows, io.project_root, int(img_size), False)
    if aug == "dr":
        return rt.GradeDataset(rows, io.project_root, int(img_size), False)
    raise ValueError(f"Unsupported grade_aug='{grade_aug}'. Choose from: auto, imagenet, dr.")


def _eval_grade_dataset(
    *,
    name: str,
    ckpt_path: Path,
    backbone_name: Optional[str],
    img_size: int,
    grade_aug: str,
    rows: List[Dict[str, str]],
    io: EvalIO,
    tta: int,
    temperature: Optional[float],
) -> Dict[str, Any]:
    state = rt.load_checkpoint_state(ckpt_path)
    bb = backbone_name or _infer_backbone_name(state)
    n_out = _infer_cls_num_classes_from_state(state)
    use_regression = (n_out == 1)

    backbone = rt.SharedBackbone(bb)
    lora_info = _maybe_apply_lora(backbone, state)
    model = rt.GradeModel(backbone, num_classes=n_out, dropout=0.0).to(io.device)
    info = rt.safe_load(model, state, strict=False)
    if info["missing"] or info["unexpected"]:
        print(f"[EXT][GRADE] load: missing={len(info['missing'])} unexpected={len(info['unexpected'])}", flush=True)

    ds = _build_grade_dataset(rows, io, img_size, grade_aug)
    dl = _make_loader(ds, io, shuffle=False)

    model.eval()
    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    score_all: List[float] = []
    score_all: List[float] = []
    prob_all: List[List[float]] = []
    path_all: List[str] = []
    t0 = time.time()
    with torch.no_grad():
        for batch in dl:
            x = batch["image"].to(io.device, non_blocking=True)
            y = batch["label"].to(io.device, non_blocking=True).view(-1)
            paths = batch.get("path", [""] * int(y.shape[0]))
            with rt.amp_autocast(io.device, io.amp):
                if int(tta) > 1:
                    logits = egp.infer_logits_tta(model, x, tta=int(tta))
                else:
                    out, _ = model(x)
                    logits = out
            if use_regression:
                scores = logits.view(-1).detach().float().cpu().numpy()
                pred = np.zeros_like(scores, dtype=np.int64)
                pred[scores >= 0.5] = 1
                pred[scores >= 1.5] = 2
                pred[scores >= 2.5] = 3
                pred[scores >= 3.5] = 4
                score_all.extend([float(s) for s in scores.tolist()])
                y_pred_all.extend([int(v) for v in pred.tolist()])
            else:
                logits_cpu = logits.detach().float().cpu()
                if temperature is not None and float(temperature) > 0:
                    logits_cpu = logits_cpu / float(temperature)
                probs = torch.softmax(logits_cpu, dim=1).numpy().astype(np.float32)
                pred = np.argmax(probs, axis=1).astype(np.int64)
                y_pred_all.extend([int(v) for v in pred.tolist()])
                # Expected score for ordinal thresholding downstream.
                score = (probs * np.arange(int(n_out), dtype=np.float32)[None, :]).sum(axis=1)
                score_all.extend([float(s) for s in score.tolist()])
                if io.save_preds:
                    prob_all.extend([[float(x) for x in row] for row in probs.tolist()])
            y_true_all.extend([int(v) for v in y.detach().cpu().numpy().astype(np.int64).tolist()])
            path_all.extend([str(p) for p in paths])

            if io.limit > 0 and len(y_true_all) >= io.limit:
                break

    y_true = np.array(y_true_all, dtype=np.int64)
    y_pred = np.array(y_pred_all, dtype=np.int64)
    out: Dict[str, Any] = {}
    out["name"] = name
    out["n"] = int(y_true.size)
    out["acc"] = float((y_true == y_pred).mean()) if y_true.size else 0.0
    out["macro_f1"] = _macro_f1_numpy(y_true, y_pred, 5) if y_true.size else float("nan")
    out["qwk"] = _qwk_numpy(y_true, y_pred, k=5) if y_true.size else float("nan")
    out["time_sec"] = float(time.time() - t0)
    out["mode"] = "regression" if use_regression else "classification"
    out["backbone"] = bb
    out["lora"] = lora_info
    out["ckpt"] = str(ckpt_path)
    out["img_size"] = int(img_size)
    out["grade_aug"] = str(grade_aug)
    out["tta"] = int(tta)
    out["temperature"] = float(temperature) if temperature is not None else None

    print(
        f"[EXT][GRADE][{name}] n={out['n']} acc={out['acc']:.4f} macro_f1={out['macro_f1']:.4f} qwk={out['qwk']:.4f} "
        f"mode={out['mode']} time={out['time_sec']:.1f}s",
        flush=True,
    )

    if io.save_preds and y_true.size:
        _ensure_dir(io.out_dir)
        pred_path = io.out_dir / f"pred_grade_{name}.csv"
        with pred_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            header = ["path", "y_true", "y_pred"]
            if use_regression:
                header.append("score")
            else:
                header.append("score")
                header.extend([f"p{i}" for i in range(int(n_out))])
                if int(n_out) == 5:
                    header.extend(["p_ge1", "p_ge2", "p_ge3"])
            w.writerow(header)
            for i in range(y_true.size):
                row = [path_all[i], int(y_true[i]), int(y_pred[i])]
                if use_regression:
                    row.append(float(score_all[i]))
                else:
                    row.append(float(score_all[i]))
                    p = prob_all[i]
                    row.extend([float(x) for x in p])
                    if int(n_out) == 5:
                        row.extend([float(sum(p[1:])), float(sum(p[2:])), float(sum(p[3:]))])
                w.writerow(row)
        out["pred_csv"] = str(pred_path)

    return out


def _eval_quality_messidor2_gradability(
    *,
    name: str,
    ckpt_path: Path,
    backbone_name: Optional[str],
    img_size: int,
    gradable_csv: Path,
    messidor2_images_dir: Path,
    prefer_processed: bool,
    offline_map: Dict[Tuple[str, str], str],
    gradable_index_2cls: int,
    gradable_classes_3cls: Sequence[int],
    io: EvalIO,
) -> Dict[str, Any]:
    rows_raw = _read_csv(gradable_csv)
    rows: List[Dict[str, str]] = []
    missing_img = 0
    for r in rows_raw:
        img_id = (r.get("image_id", "") or "").strip()
        if not img_id:
            continue
        stem = Path(img_id).stem
        img_path: Optional[Path] = None
        if prefer_processed:
            proc = offline_map.get(("messidor2", stem))
            if proc:
                img_path = Path(rt.resolve_path(io.project_root, proc))
        if img_path is None:
            img_path = Path(rt.resolve_path(io.project_root, str(messidor2_images_dir / img_id)))
        if not img_path.exists():
            missing_img += 1
            continue
        y = (r.get("adjudicated_gradable", "") or "").strip()
        if y == "":
            continue
        rows.append({"image_path": _as_relpath(io.project_root, str(img_path)), "label": str(int(float(y)))})
        if io.limit > 0 and len(rows) >= io.limit:
            break

    state = rt.load_checkpoint_state(ckpt_path)
    bb = backbone_name or _infer_backbone_name(state)
    n_out = _infer_cls_num_classes_from_state(state)

    backbone = rt.SharedBackbone(bb)
    lora_info = _maybe_apply_lora(backbone, state)
    model = rt.QualityModel(backbone, num_classes_quality=n_out, dropout=0.0).to(io.device)
    info = rt.safe_load(model, state, strict=False)
    if info["missing"] or info["unexpected"]:
        print(f"[EXT][QUAL] load: missing={len(info['missing'])} unexpected={len(info['unexpected'])}", flush=True)

    ds = rt.QualityDataset(rows, io.project_root, int(img_size), False, collapse_to_binary=False)
    dl = _make_loader(ds, io, shuffle=False)

    model.eval()
    y_true_all: List[int] = []
    p_grad_all: List[float] = []
    path_all: List[str] = []
    t0 = time.time()
    with torch.no_grad():
        for batch in dl:
            x = batch["image"].to(io.device, non_blocking=True)
            y = batch["label"].to(io.device, non_blocking=True).view(-1)
            paths = batch.get("path", [""] * int(y.shape[0]))
            with rt.amp_autocast(io.device, io.amp):
                logits, _ = model(x)
            probs = torch.softmax(logits.float(), dim=1).detach().cpu().numpy()
            if n_out == 2:
                gi = int(gradable_index_2cls)
                if gi not in (0, 1):
                    raise ValueError("--qual_gradable_index_2cls must be 0 or 1")
                p_grad = probs[:, gi]
            else:
                idxs = [int(i) for i in gradable_classes_3cls]
                if not idxs:
                    idxs = [0, 1]
                p_grad = probs[:, idxs].sum(axis=1)
            y_true_all.extend([int(v) for v in y.detach().cpu().numpy().astype(np.int64).tolist()])
            p_grad_all.extend([float(v) for v in p_grad.tolist()])
            path_all.extend([str(p) for p in paths])
            if io.limit > 0 and len(y_true_all) >= io.limit:
                break

    y_true = np.array(y_true_all, dtype=np.int64)
    p_grad = np.array(p_grad_all, dtype=np.float32)
    y_pred = (p_grad >= 0.5).astype(np.int64)

    out: Dict[str, Any] = {}
    out["name"] = name
    out["n"] = int(y_true.size)
    out["acc"] = float((y_true == y_pred).mean()) if y_true.size else 0.0
    out["macro_f1"] = _macro_f1_numpy(y_true, y_pred, 2) if y_true.size else float("nan")
    out["time_sec"] = float(time.time() - t0)
    out["backbone"] = bb
    out["lora"] = lora_info
    out["ckpt"] = str(ckpt_path)
    out["img_size"] = int(img_size)
    out["missing_images"] = int(missing_img)
    out["label_csv"] = str(gradable_csv)

    print(
        f"[EXT][QUAL][{name}] n={out['n']} acc={out['acc']:.4f} macro_f1={out['macro_f1']:.4f} "
        f"p_grad_mean={float(p_grad.mean()) if y_true.size else float('nan'):.4f} time={out['time_sec']:.1f}s",
        flush=True,
    )

    if io.save_preds and y_true.size:
        _ensure_dir(io.out_dir)
        pred_path = io.out_dir / f"pred_qual_{name}.csv"
        with pred_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "y_true_gradable", "p_gradable", "y_pred_gradable"])
            for i in range(y_true.size):
                w.writerow([path_all[i], int(y_true[i]), float(p_grad[i]), int(y_pred[i])])
        out["pred_csv"] = str(pred_path)

    return out


def _eval_seg_idrid(
    *,
    name: str,
    ckpt_path: Path,
    backbone_name: Optional[str],
    decoder_name: Optional[str],
    img_size: int,
    seg_manifest: Path,
    seg_split: str,
    io: EvalIO,
) -> Dict[str, Any]:
    rows = rt.read_csv_rows(seg_manifest)
    split = str(seg_split).lower().strip()
    if split and split != "all" and "split" in rows[0]:
        rows = [r for r in rows if (r.get("split", "").lower() == split)]

    if io.limit > 0:
        rows = rows[: io.limit]

    state = rt.load_checkpoint_state(ckpt_path)
    bb = backbone_name or _infer_backbone_name(state)
    dec = decoder_name or _infer_seg_decoder_from_state(state)
    num_classes = _infer_seg_num_classes_from_state(state)
    mask_mode = "multilabel"  # IDRiD masks in this repo are multilabel .npy (C,H,W)

    backbone = rt.SharedBackbone(bb)
    lora_info = _maybe_apply_lora(backbone, state)
    model = rt.SegModel(backbone, num_classes=num_classes, mask_mode=mask_mode, decoder_name=dec).to(io.device)
    info = rt.safe_load(model, state, strict=False)
    if info["missing"] or info["unexpected"]:
        print(f"[EXT][SEG] load: missing={len(info['missing'])} unexpected={len(info['unexpected'])}", flush=True)

    ds = rt.SegDataset(rows, io.project_root, int(img_size), False, mask_mode, num_classes)
    dl = _make_loader(ds, io, shuffle=False)

    model.eval()
    dice_hards: List[float] = []
    dice_softs: List[float] = []
    dice_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    dices_by_thr: Dict[float, List[float]] = {t: [] for t in dice_thresholds}

    inter = np.zeros((num_classes,), dtype=np.float64)
    pred_sum = np.zeros((num_classes,), dtype=np.float64)
    tgt_sum = np.zeros((num_classes,), dtype=np.float64)

    t0 = time.time()
    with torch.no_grad():
        for batch in dl:
            x = batch["image"].to(io.device, non_blocking=True)
            y = batch["mask"].to(io.device, non_blocking=True)
            with rt.amp_autocast(io.device, io.amp):
                logits, _ = model(x)

            dice_hards.append(rt.seg_dice_score(logits, y, mask_mode, threshold=0.5))
            dice_softs.append(rt.seg_soft_dice_score(logits, y, mask_mode))
            for t in dice_thresholds:
                dices_by_thr[t].append(rt.seg_dice_score(logits, y, mask_mode, threshold=t))

            probs = (torch.sigmoid(logits) > 0.5).float()
            inter += (probs * y).sum(dim=(0, 2, 3)).detach().cpu().numpy()
            pred_sum += probs.sum(dim=(0, 2, 3)).detach().cpu().numpy()
            tgt_sum += y.sum(dim=(0, 2, 3)).detach().cpu().numpy()

    val_dice_hard = float(np.mean(dice_hards)) if dice_hards else 0.0
    val_dice_soft = float(np.mean(dice_softs)) if dice_softs else 0.0
    means = {t: float(np.mean(v)) if v else 0.0 for t, v in dices_by_thr.items()}
    best_thr = max(means, key=means.get) if means else 0.5
    val_dice_best = float(means.get(best_thr, val_dice_hard))

    dice_pc = (2.0 * inter + 1e-6) / (pred_sum + tgt_sum + 1e-6)
    dice_pc = dice_pc.astype(np.float32)
    dice_global_mean = float(dice_pc.mean()) if dice_pc.size else 0.0

    out: Dict[str, Any] = {}
    out["name"] = name
    out["n"] = int(len(rows))
    out["val_dice@0.5"] = float(val_dice_hard)
    out["val_dice_soft"] = float(val_dice_soft)
    out["val_dice_best"] = float(val_dice_best)
    out["val_best_thr"] = float(best_thr)
    out["dice_global_mean@0.5"] = float(dice_global_mean)
    out["dice_global_per_class@0.5"] = [float(x) for x in dice_pc.tolist()]
    out["time_sec"] = float(time.time() - t0)
    out["backbone"] = bb
    out["decoder"] = dec
    out["num_classes_seg"] = int(num_classes)
    out["lora"] = lora_info
    out["ckpt"] = str(ckpt_path)
    out["img_size"] = int(img_size)
    out["manifest"] = str(seg_manifest)
    out["split"] = split

    print(
        f"[EXT][SEG][{name}] n={out['n']} dice@0.5={out['val_dice@0.5']:.4f} dice_soft={out['val_dice_soft']:.4f} "
        f"dice_best={out['val_dice_best']:.4f}@{out['val_best_thr']:.2f} dice_global_mean={out['dice_global_mean@0.5']:.4f} "
        f"time={out['time_sec']:.1f}s",
        flush=True,
    )

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="checkpoints/external_eval")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--save_preds", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="0 means no limit")
    p.add_argument("--prefer_processed", action="store_true", help="Prefer data/processed/cfp_512 images if mapping exists")

    # Grade: external test 1/2
    p.add_argument("--grade_run_dir", type=str, default="")
    p.add_argument("--grade_ckpt", type=str, default="")
    p.add_argument("--grade_backbone", type=str, default="")
    p.add_argument("--grade_img_size", type=int, default=0)
    p.add_argument("--grade_aug", type=str, default="auto", choices=["auto", "imagenet", "dr"])
    p.add_argument("--grade_tta", type=int, default=1, help="TTA count for grade external eval (1 disables).")
    p.add_argument(
        "--grade_use_temperature",
        type=int,
        default=1,
        choices=[0, 1],
        help="If grade_run_dir has logs/temperature.json, apply it to logits before softmax.",
    )

    p.add_argument("--aptos_train_csv", type=str, default=r"data/raw/aptos2019/aptos2019-blindness-detection/train.csv")
    p.add_argument("--aptos_images_dir", type=str, default=r"data/raw/aptos2019/aptos2019-blindness-detection/train_images")

    p.add_argument("--messidor2_dr_grades_csv", type=str, default=r"data/raw/messidor2_labels/messidor2_dr_grades.csv")
    p.add_argument("--messidor2_images_dir", type=str, default=r"data/raw/messidor2/official/images")

    # DDR DR grading (external)
    p.add_argument("--ddr_train_txt", type=str, default=r"data/raw/ddr/DDR-dataset/DR_grading/train.txt")
    p.add_argument("--ddr_valid_txt", type=str, default=r"data/raw/ddr/DDR-dataset/DR_grading/valid.txt")
    p.add_argument("--ddr_train_images_dir", type=str, default=r"data/raw/ddr/DDR-dataset/DR_grading/train")
    p.add_argument("--ddr_valid_images_dir", type=str, default=r"data/raw/ddr/DDR-dataset/DR_grading/valid")
    p.add_argument("--ddr_split", type=str, default="valid", choices=["valid", "train"], help="Which DDR split to evaluate")

    # Quality: messidor2_dr_grades gradability
    p.add_argument("--qual_run_dir", type=str, default="")
    p.add_argument("--qual_ckpt", type=str, default="")
    p.add_argument("--qual_backbone", type=str, default="")
    p.add_argument("--qual_img_size", type=int, default=0)
    p.add_argument("--qual_gradable_index_2cls", type=int, default=1, choices=[0, 1])
    p.add_argument("--qual_gradable_classes_3cls", type=str, default="0,1", help="Comma-separated indices treated as gradable for 3-class quality")

    # Seg: external IDRiD
    p.add_argument("--seg_run_dir", type=str, default="")
    p.add_argument("--seg_ckpt", type=str, default="")
    p.add_argument("--seg_backbone", type=str, default="")
    p.add_argument("--seg_decoder", type=str, default="", help="If empty, infer from ckpt keys")
    p.add_argument("--seg_img_size", type=int, default=0)
    p.add_argument("--seg_manifest", type=str, default=r"data/manifests/seg_idrid.csv")
    p.add_argument("--seg_split", type=str, default="test", choices=["all", "train", "valid", "val", "test"])

    return p.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    out_dir = Path(rt.resolve_path(project_root, args.out_dir)).resolve()

    io = EvalIO(
        project_root=project_root,
        device=args.device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        amp=bool(args.amp),
        out_dir=out_dir,
        save_preds=bool(args.save_preds),
        limit=max(0, int(args.limit)),
    )

    prefer_processed = bool(args.prefer_processed)
    offline_map = _maybe_offline_proc_map(project_root) if prefer_processed else {}

    results: Dict[str, Any] = {"project_root": str(project_root), "device": io.device, "time": time.strftime("%Y-%m-%d %H:%M:%S")}

    # -----------------------------
    # Grade head: APTOS + Messidor2
    # -----------------------------
    grade_run_dir = Path(rt.resolve_path(project_root, args.grade_run_dir)).resolve() if args.grade_run_dir else None
    grade_cfg = _try_load_run_cfg(grade_run_dir) if grade_run_dir else {}
    grade_ckpt = (
        Path(rt.resolve_path(project_root, args.grade_ckpt)).resolve()
        if args.grade_ckpt
        else ((grade_run_dir / "best_grade.pth") if grade_run_dir else None)
    )
    if grade_ckpt is not None and grade_ckpt.exists():
        grade_backbone = (args.grade_backbone or grade_cfg.get("backbone", "") or "").strip() or None
        grade_img_size = int(args.grade_img_size or grade_cfg.get("img_size", 0) or 0) or 224
        grade_aug = str(args.grade_aug if args.grade_aug != "auto" else (grade_cfg.get("grade_aug", "auto") or "auto"))

        # Temperature scaling (learned on internal calib). Safe to *apply* externally without tuning.
        temperature: Optional[float] = None
        if bool(int(args.grade_use_temperature)) and grade_run_dir is not None:
            tj = grade_run_dir / "logs" / "temperature.json"
            if tj.exists():
                try:
                    temperature = float(_load_json(tj).get("T", 1.0))
                    if not np.isfinite(temperature) or temperature <= 0:
                        temperature = None
                except Exception:
                    temperature = None

        # APTOS2019
        aptos_csv = Path(rt.resolve_path(project_root, args.aptos_train_csv))
        aptos_img_dir = Path(rt.resolve_path(project_root, args.aptos_images_dir))
        if aptos_csv.exists():
            apt_rows = _read_csv(aptos_csv)
            rows: List[Dict[str, str]] = []
            miss = 0
            for r in apt_rows:
                id_code = (r.get("id_code", "") or "").strip()
                if not id_code:
                    continue
                y = (r.get("diagnosis", "") or "").strip()
                if y == "":
                    continue
                img_path: Optional[Path] = None
                if prefer_processed:
                    proc = offline_map.get(("aptos2019", id_code))
                    if proc:
                        img_path = Path(rt.resolve_path(project_root, proc))
                if img_path is None:
                    img_path = aptos_img_dir / f"{id_code}.png"
                if not img_path.exists():
                    miss += 1
                    continue
                rows.append({"image_path": _as_relpath(project_root, str(img_path)), "label": str(int(float(y)))})
                if io.limit > 0 and len(rows) >= io.limit:
                    break
            if miss:
                print(f"[EXT][GRADE][aptos2019] missing_images={miss}", flush=True)
            results["grade_aptos2019"] = _eval_grade_dataset(
                name="aptos2019",
                ckpt_path=grade_ckpt,
                backbone_name=grade_backbone,
                img_size=grade_img_size,
                grade_aug=grade_aug,
                rows=rows,
                io=io,
                tta=int(args.grade_tta),
                temperature=temperature,
            )
        else:
            print(f"[EXT][GRADE] APTOS train.csv not found: {aptos_csv}. Skipped.", flush=True)

        # Messidor2 adjudicated DR grade
        m2_csv = Path(rt.resolve_path(project_root, args.messidor2_dr_grades_csv))
        m2_img_dir = Path(rt.resolve_path(project_root, args.messidor2_images_dir))
        if m2_csv.exists():
            m2_rows = _read_csv(m2_csv)
            rows2: List[Dict[str, str]] = []
            miss2 = 0
            for r in m2_rows:
                img_id = (r.get("image_id", "") or "").strip()
                if not img_id:
                    continue
                stem = Path(img_id).stem
                y = (r.get("adjudicated_dr_grade", "") or "").strip()
                if y == "":
                    continue
                img_path: Optional[Path] = None
                if prefer_processed:
                    proc = offline_map.get(("messidor2", stem))
                    if proc:
                        img_path = Path(rt.resolve_path(project_root, proc))
                if img_path is None:
                    img_path = m2_img_dir / img_id
                if not img_path.exists():
                    miss2 += 1
                    continue
                rows2.append({"image_path": _as_relpath(project_root, str(img_path)), "label": str(int(float(y)))})
                if io.limit > 0 and len(rows2) >= io.limit:
                    break
            if miss2:
                print(f"[EXT][GRADE][messidor2] missing_images={miss2}", flush=True)
            results["grade_messidor2_dr_grades"] = _eval_grade_dataset(
                name="messidor2_dr_grades",
                ckpt_path=grade_ckpt,
                backbone_name=grade_backbone,
                img_size=grade_img_size,
                grade_aug=grade_aug,
                rows=rows2,
                io=io,
                tta=int(args.grade_tta),
                temperature=temperature,
            )
        else:
            print(f"[EXT][GRADE] Messidor2 dr_grades not found: {m2_csv}. Skipped.", flush=True)

        # DDR DR grading
        ddr_split = str(args.ddr_split).lower().strip()
        ddr_txt = Path(rt.resolve_path(project_root, args.ddr_valid_txt if ddr_split == "valid" else args.ddr_train_txt))
        ddr_img_dir = Path(rt.resolve_path(project_root, args.ddr_valid_images_dir if ddr_split == "valid" else args.ddr_train_images_dir))
        if ddr_txt.exists() and ddr_img_dir.exists():
            items = _read_ddr_txt_labels(ddr_txt)
            rows3: List[Dict[str, str]] = []
            miss3 = 0
            for fn, y in items:
                img_path = ddr_img_dir / str(fn)
                if not img_path.exists():
                    miss3 += 1
                    continue
                rows3.append({"image_path": _as_relpath(project_root, str(img_path)), "label": str(int(y))})
                if io.limit > 0 and len(rows3) >= io.limit:
                    break
            if miss3:
                print(f"[EXT][GRADE][ddr] missing_images={miss3}", flush=True)
            results["grade_ddr_dr_grading"] = _eval_grade_dataset(
                name="ddr_dr_grading",
                ckpt_path=grade_ckpt,
                backbone_name=grade_backbone,
                img_size=grade_img_size,
                grade_aug=grade_aug,
                rows=rows3,
                io=io,
                tta=int(args.grade_tta),
                temperature=temperature,
            )
        else:
            print(f"[EXT][GRADE] DDR not found (txt={ddr_txt}, dir={ddr_img_dir}). Skipped.", flush=True)
    else:
        print("[EXT][GRADE] grade_ckpt not provided/found. Skipped grade external eval.", flush=True)

    # -----------------------------
    # Quality head: Messidor2 gradability
    # -----------------------------
    qual_run_dir = Path(rt.resolve_path(project_root, args.qual_run_dir)).resolve() if args.qual_run_dir else None
    qual_cfg = _try_load_run_cfg(qual_run_dir) if qual_run_dir else {}
    qual_ckpt = (
        Path(rt.resolve_path(project_root, args.qual_ckpt)).resolve()
        if args.qual_ckpt
        else ((qual_run_dir / "best_quality.pth") if qual_run_dir else None)
    )
    if qual_ckpt is not None and qual_ckpt.exists():
        qual_backbone = (args.qual_backbone or qual_cfg.get("backbone", "") or "").strip() or None
        qual_img_size = int(args.qual_img_size or qual_cfg.get("img_size", 0) or 0) or 224
        m2_csv = Path(rt.resolve_path(project_root, args.messidor2_dr_grades_csv))
        m2_img_dir = Path(rt.resolve_path(project_root, args.messidor2_images_dir))
        if m2_csv.exists():
            idxs = [int(x) for x in str(args.qual_gradable_classes_3cls).split(",") if x.strip() != ""]
            results["qual_messidor2_gradability"] = _eval_quality_messidor2_gradability(
                name="messidor2_gradability",
                ckpt_path=qual_ckpt,
                backbone_name=qual_backbone,
                img_size=qual_img_size,
                gradable_csv=m2_csv,
                messidor2_images_dir=m2_img_dir,
                prefer_processed=prefer_processed,
                offline_map=offline_map,
                gradable_index_2cls=int(args.qual_gradable_index_2cls),
                gradable_classes_3cls=idxs,
                io=io,
            )
        else:
            print(f"[EXT][QUAL] Messidor2 dr_grades not found: {m2_csv}. Skipped.", flush=True)
    else:
        print("[EXT][QUAL] qual_ckpt not provided/found. Skipped quality external eval.", flush=True)

    # -----------------------------
    # Seg head: external IDRiD
    # -----------------------------
    seg_run_dir = Path(rt.resolve_path(project_root, args.seg_run_dir)).resolve() if args.seg_run_dir else None
    seg_cfg = _try_load_run_cfg(seg_run_dir) if seg_run_dir else {}
    seg_ckpt = (
        Path(rt.resolve_path(project_root, args.seg_ckpt)).resolve()
        if args.seg_ckpt
        else ((seg_run_dir / "best_seg.pth") if seg_run_dir else None)
    )
    if seg_ckpt is not None and seg_ckpt.exists():
        seg_backbone = (args.seg_backbone or seg_cfg.get("backbone", "") or "").strip() or None
        seg_img_size = int(args.seg_img_size or seg_cfg.get("img_size", 0) or 0) or 512
        seg_decoder = (args.seg_decoder or seg_cfg.get("seg_decoder", "") or "").strip() or None
        seg_manifest = Path(rt.resolve_path(project_root, args.seg_manifest))
        if seg_manifest.exists():
            results["seg_idrid"] = _eval_seg_idrid(
                name="idrid",
                ckpt_path=seg_ckpt,
                backbone_name=seg_backbone,
                decoder_name=seg_decoder,
                img_size=seg_img_size,
                seg_manifest=seg_manifest,
                seg_split=str(args.seg_split),
                io=io,
            )
        else:
            print(f"[EXT][SEG] seg_manifest not found: {seg_manifest}. Skipped.", flush=True)
    else:
        print("[EXT][SEG] seg_ckpt not provided/found. Skipped seg external eval.", flush=True)

    _ensure_dir(out_dir)
    out_json = out_dir / "external_eval_summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[EXT] Saved: {out_json}", flush=True)


if __name__ == "__main__":
    main()
