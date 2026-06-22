"""Create a 6-panel figure per dataset: raw + grade0..4 heatmaps.

Layout (1x6, all square):
- Raw (uses the grade0-selected sample's background)
- Grade 0 heatmap (on grade0-selected sample)
- Grade 1 heatmap (on grade1-selected sample)
- Grade 2 heatmap (on grade2-selected sample)
- Grade 3 heatmap (on grade3-selected sample)
- Grade 4 heatmap (on grade4-selected sample)

"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _require(pkg: str):
    import importlib

    try:
        return importlib.import_module(pkg)
    except Exception as e:
        raise RuntimeError(f"Missing dependency '{pkg}'. Install it in your environment. Original error: {e}")


torch = _require("torch")
plt = _require("matplotlib.pyplot")


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _try_float(v: Any) -> Optional[float]:
    if v in (None, ""):
        return None
    try:
        return float(v)
    except Exception:
        return None


def _try_int(v: Any) -> Optional[int]:
    if v in (None, ""):
        return None
    try:
        return int(float(v))
    except Exception:
        return None


def _row_probs(r: Dict[str, str]) -> Optional[np.ndarray]:
    vals: List[float] = []
    for i in range(5):
        if f"p{i}" not in r:
            return None
        fv = _try_float(r.get(f"p{i}"))
        if fv is None:
            return None
        vals.append(float(fv))
    p = np.asarray(vals, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p


def _sharpness(abs_path: str) -> float:
    cv2 = _require("cv2")
    img = cv2.imread(str(abs_path), cv2.IMREAD_COLOR)
    if img is None:
        return float("-inf")
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def _try_load_offline_raw_map(project_root: Path) -> Dict[str, str]:
    """Map normalized proc_path -> raw_path using data/meta/manifests/manifest_offline.csv."""
    mp: Dict[str, str] = {}
    p = project_root / "data" / "meta" / "manifests" / "manifest_offline.csv"
    if not p.exists():
        return mp
    rows = _read_csv_rows(p)
    for r in rows:
        if str(r.get("status", "")).lower().strip() != "ok":
            continue
        raw_path = str(r.get("raw_path", "") or "").strip()
        proc_path = str(r.get("proc_path", "") or "").strip()
        if not raw_path or not proc_path:
            continue
        mp[os.path.normcase(os.path.abspath(proc_path))] = raw_path
    return mp


def select_examples_by_true_grade(csv_path: Path, *, prefer_correct: bool, max_candidates: int) -> Dict[int, Dict[str, Any]]:
    rows = _read_csv_rows(csv_path)
    by: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(5)}
    for r in rows:
        ap = str(r.get("path", "") or "").strip()
        if not ap or not Path(ap).exists():
            continue
        y_true = _try_int(r.get("y_true"))
        if y_true is None:
            y_true = _try_int(r.get("true_grade"))
        if y_true is None or not (0 <= int(y_true) <= 4):
            continue
        pred = _try_int(r.get("pred_argmax"))
        p = _row_probs(r)
        is_correct = (pred is not None) and (int(pred) == int(y_true))
        conf = float(p[int(y_true)]) if (p is not None) else -1.0
        by[int(y_true)].append({"row": r, "path": ap, "y_true": int(y_true), "pred": pred, "conf": conf, "is_correct": bool(is_correct)})

    picked: Dict[int, Dict[str, Any]] = {}
    for g in range(5):
        cand = by[g]
        if not cand:
            continue
        if prefer_correct:
            good = [c for c in cand if c["is_correct"]]
            if good:
                cand = good
        cand = sorted(cand, key=lambda c: float(c.get("conf", -1.0)), reverse=True)[: int(max_candidates)]
        for c in cand:
            c["sharpness"] = _sharpness(str(c["path"]))
        cand = sorted(cand, key=lambda c: (float(c.get("sharpness", -1.0)), float(c.get("conf", -1.0))), reverse=True)
        picked[g] = cand[0]
    return picked


def patch_attention_capture(tr_mod) -> None:
    if getattr(tr_mod.Attention, "_omo_patched", False):
        return
    orig = tr_mod.Attention.forward

    def new_forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        if torch.is_grad_enabled():
            attn.retain_grad()
        self.last_attn = attn
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out

    tr_mod.Attention.forward = new_forward
    tr_mod.Attention._omo_patched = True
    tr_mod.Attention._omo_orig_forward = orig


def grad_rollout(attn_list: List["torch.Tensor"]) -> "torch.Tensor":
    if not attn_list:
        raise ValueError("No attention tensors captured")
    B = int(attn_list[0].shape[0])
    if B != 1:
        raise ValueError("This script expects batch_size=1")
    N = int(attn_list[0].shape[-1])
    eye = torch.eye(N, device=attn_list[0].device, dtype=attn_list[0].dtype).unsqueeze(0)
    joint = eye
    for attn in attn_list:
        if attn.grad is None:
            raise RuntimeError("Missing attention gradients; run backward() first")
        a = (attn * attn.grad).mean(dim=1)
        a = torch.clamp(a, min=0.0)
        a = (a + eye) / 2.0
        a = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        joint = a @ joint
    return joint[0, 0, 1:]


def mask_to_grid(mask: "torch.Tensor", *, grid_h: int, grid_w: int) -> np.ndarray:
    m = mask.detach().float().cpu().numpy()
    if m.size != grid_h * grid_w:
        raise ValueError(f"Token count mismatch: got {m.size} expected {grid_h*grid_w}")
    m = m.reshape(grid_h, grid_w)
    m = m - m.min()
    if m.max() > 0:
        m = m / m.max()
    return m.astype(np.float32)


def overlay_heatmap(rgb: np.ndarray, mask: np.ndarray, *, alpha: float) -> np.ndarray:
    cv2 = _require("cv2")
    if mask.dtype != np.float32:
        mask = mask.astype(np.float32)
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    if mask.shape[0] != h or mask.shape[1] != w:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LANCZOS4)
    mask_u8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(mask_u8, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out = (rgb.astype(np.float32) * (1.0 - alpha) + heat.astype(np.float32) * alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def load_image_rgb(abs_path: str, *, size: int) -> np.ndarray:
    cv2 = _require("cv2")
    img = cv2.imread(str(abs_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(str(abs_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (int(size), int(size)), interpolation=cv2.INTER_AREA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, required=True)
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default="best_grade.pth")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--in_csv", type=str, required=True)
    ap.add_argument("--tag", type=str, default="dataset")
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--vis_size", type=int, default=1024)
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--prefer_correct", type=int, default=1, choices=[0, 1])
    ap.add_argument("--max_candidates_per_grade", type=int, default=200)
    ap.add_argument("--target", type=str, default="true", choices=["true", "pred"], help="Grad target: true grade or predicted argmax")
    ap.add_argument("--vis_source", type=str, default="auto", choices=["auto", "input"], help="Use raw background if available")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = (run_dir / ckpt_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(str(ckpt_path))

    in_csv = Path(args.in_csv).resolve()
    picked = select_examples_by_true_grade(in_csv, prefer_correct=bool(int(args.prefer_correct)), max_candidates=int(args.max_candidates_per_grade))
    missing = [g for g in range(5) if g not in picked]
    if missing:
        raise RuntimeError(f"Missing grades in CSV selection: {missing}. Try --prefer_correct 0.")

    offline_map = _try_load_offline_raw_map(project_root) if str(args.vis_source).lower().strip() == "auto" else {}

    # Import repo modules
    src_dir = (project_root / "src").resolve()
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    import train_roundtune_cpf as tr
    import eval_grade_ce_posthoc as egp

    patch_attention_capture(tr)

    summary = {}
    sj = run_dir / "logs" / "summary.json"
    if sj.exists():
        with sj.open("r", encoding="utf-8") as f:
            summary = json.load(f)
    cfg = summary.get("config", {}) if isinstance(summary.get("config", {}), dict) else {}
    img_size = int(cfg.get("img_size", 224) or 224)
    grid_h = img_size // 16
    grid_w = img_size // 16

    device = str(args.device)
    model = egp.build_grade_model_for_eval(cfg, device=device)
    egp.load_model_ckpt_strict(model, ckpt_path, device=device)
    model.eval()

    overlays: List[np.ndarray] = []
    titles: List[str] = []

    for g in range(5):
        abs_path = str(picked[g]["path"])
        vis_abs = offline_map.get(os.path.normcase(os.path.abspath(abs_path)), abs_path) if offline_map else abs_path
        rgb = load_image_rgb(vis_abs, size=int(args.vis_size))

        rel = os.path.relpath(abs_path, str(project_root)) if Path(abs_path).is_absolute() else abs_path
        rows = [{"image_path": rel, "label": "0"}]
        aug = egp.resolve_grade_aug(cfg)
        ds = egp.build_grade_eval_dataset(rows=rows, project_root=project_root, img_size=img_size, grade_aug=aug)
        batch = ds[0]
        x = batch["image"].unsqueeze(0).to(device)

        # target class
        pred_argmax = int(torch.argmax(model(x)["grade_logits"], dim=1).item()) if isinstance(model(x), dict) else None
        target_cls = int(g) if str(args.target) == "true" else int(pred_argmax if pred_argmax is not None else g)

        model.zero_grad(set_to_none=True)
        for blk in model.backbone.backbone.blocks:
            if hasattr(blk.attn, "last_attn"):
                blk.attn.last_attn = None
        out = model(x)
        logits = out["grade_logits"] if isinstance(out, dict) else (out[0] if isinstance(out, (tuple, list)) else out)
        loss = logits[0, int(target_cls)]
        loss.backward()

        attn_list: List[torch.Tensor] = []
        for blk in model.backbone.backbone.blocks:
            a = getattr(blk.attn, "last_attn", None)
            if a is None:
                raise RuntimeError("Attention not captured")
            attn_list.append(a)
        m = grad_rollout(attn_list)
        m = mask_to_grid(m, grid_h=grid_h, grid_w=grid_w)
        ov = overlay_heatmap(rgb, m, alpha=float(args.alpha))
        overlays.append(ov)
        titles.append(f"Grade {g}")

    out_dir = Path(args.out_dir).resolve() if str(args.out_dir).strip() else (project_root / "checkpoints" / "cam_panels")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = str(args.tag).strip() or "dataset"
    out_png = out_dir / f"panel_by_grade_{tag}.png"
    out_pdf = out_dir / f"panel_by_grade_{tag}.pdf"

    fig, axes = plt.subplots(1, 5, figsize=(5 * 3.0, 3.0), squeeze=False)
    for ax, img, t in zip(axes[0], overlays, titles):
        ax.imshow(img)
        ax.set_title(t)
        ax.set_axis_off()
        ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
    fig.savefig(out_pdf, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    print("[OK] wrote:", out_png)
    print("[OK] wrote:", out_pdf)


if __name__ == "__main__":
    main()
