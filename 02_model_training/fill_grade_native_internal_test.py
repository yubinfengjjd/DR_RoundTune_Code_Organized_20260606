"""Fill missing native DR-grade internal_test metrics from an existing best checkpoint.

This utility reproduces the no-TTA, no-temperature internal-test evaluation path
used at the end of `train_roundtune_cpf.py::train_grade`. It is intended for
interrupted long-running training jobs that already saved `best_grade.pth` but
ended before the final summary write.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import eval_grade_ce_posthoc as posthoc  # noqa: E402
import train_roundtune_cpf as tr  # noqa: E402


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def write_json(path: Path, data: Dict[str, Any]) -> None:
    tr.ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def resolve_manifest(project_root: Path, summary: Dict[str, Any], manifest_arg: str) -> Path:
    manifest = Path(manifest_arg)
    if not manifest.is_absolute():
        manifest = (project_root / manifest).resolve()
    default_norm = os.path.normcase(str(Path(r"data\manifests\master_split.csv")))
    arg_norm = os.path.normcase(str(Path(manifest_arg)))
    if arg_norm == default_norm:
        recorded = summary.get("grade_manifest") or summary.get("grade_rmf_manifest")
        if isinstance(recorded, str) and recorded.strip():
            candidate = Path(recorded.strip())
            if not candidate.is_absolute():
                candidate = (project_root / candidate).resolve()
            if candidate.exists():
                return candidate
    return manifest


def update_summary_with_native_metrics(
    summary: Dict[str, Any],
    *,
    metrics: Dict[str, float],
    source: Dict[str, Any],
) -> Dict[str, Any]:
    required = ("acc", "macro_f1", "qwk")
    missing = [k for k in required if k not in metrics]
    if missing:
        raise ValueError(f"Missing native metric(s): {missing}")

    out = dict(summary)
    old_posthoc = out.pop("posthoc_eval_tta1", None)
    if old_posthoc is not None:
        out["archived_posthoc_eval_tta1"] = old_posthoc
    out["internal_test"] = {k: float(metrics[k]) for k in required}
    out["internal_test_source"] = dict(source)
    return out


@torch.no_grad()
def evaluate_native_internal_test(
    *,
    model: torch.nn.Module,
    cfg: Dict[str, Any],
    project_root: Path,
    manifest: Path,
    device: str,
) -> Dict[str, float]:
    rows_all = tr.read_csv_rows(manifest)
    rows = posthoc._select_split_rows(rows_all, "test")
    img_size = int(cfg.get("img_size", 224))
    grade_aug = posthoc.resolve_grade_aug(cfg)
    ds = posthoc.build_grade_eval_dataset(rows=rows, project_root=project_root, img_size=img_size, grade_aug=grade_aug)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=int(cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=str(device).startswith("cuda"),
        persistent_workers=(int(cfg.get("num_workers", 0)) > 0),
    )

    model.eval()
    all_logits: List[torch.Tensor] = []
    all_y: List[torch.Tensor] = []
    for batch in dl:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        has_label = batch["has_label"].to(device, non_blocking=True).bool()
        out = model(x)
        if isinstance(out, dict):
            logits = out["grade_logits"]
        elif isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out
        mask = has_label.view(-1)
        if not mask.any():
            continue
        all_logits.append(logits[mask].detach().cpu())
        all_y.append(y.view(-1)[mask].detach().cpu())

    if not all_y:
        raise RuntimeError("No labeled samples found in test split.")

    logits_cat = torch.cat(all_logits, dim=0)
    y_cat = torch.cat(all_y, dim=0).long()
    loss_name = str(cfg.get("grade_loss", "ce")).lower().strip()
    if loss_name == "corn":
        mets = tr.cls_metrics_corn(logits_cat, y_cat, num_classes=5)
    elif loss_name == "edl":
        probs_cat, _, _ = tr.edl_dirichlet_probs(
            logits_cat.to(torch.float32),
            evidence=str(cfg.get("edl_evidence", "relu")),
        )
        mets = tr.cls_metrics_from_probs(probs_cat, y_cat)
    else:
        mets = tr.cls_metrics(logits_cat, y_cat)

    return {
        "acc": float(mets.get("acc", 0.0)),
        "macro_f1": float(mets.get("macro_f1", np.nan)),
        "qwk": float(mets.get("qwk", np.nan)),
    }


def fill_native_internal_test(
    *,
    project_root: Path,
    run_dir: Path,
    manifest_arg: str,
    ckpt_name: str,
    device: str,
    num_workers: Optional[int],
) -> Dict[str, Any]:
    summary_path = run_dir / "logs" / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")
    summary = read_json(summary_path)
    cfg = dict(summary.get("config") or {})
    if num_workers is not None:
        cfg["num_workers"] = int(num_workers)
    manifest = resolve_manifest(project_root, summary, manifest_arg)
    if not manifest.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest}")
    ckpt = Path(ckpt_name)
    if not ckpt.is_absolute():
        ckpt = run_dir / ckpt
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    model = posthoc.build_grade_model_for_eval(cfg, device=device)
    posthoc.load_model_ckpt_strict(model, ckpt, device=device)
    metrics = evaluate_native_internal_test(
        model=model,
        cfg=cfg,
        project_root=project_root,
        manifest=manifest,
        device=device,
    )
    source = {
        "method": "native_no_tta_best_checkpoint_eval",
        "checkpoint": str(ckpt),
        "manifest": str(manifest),
        "script": str(Path(__file__).resolve()),
        "temperature": None,
        "tta": 1,
        "note": "No TTA transforms, no temperature scaling, no threshold calibration; matches train_grade internal_test argmax path.",
    }
    updated = update_summary_with_native_metrics(summary, metrics=metrics, source=source)
    write_json(summary_path, updated)
    return {"summary": str(summary_path), "metrics": metrics, "source": source}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project_root", type=str, required=True)
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--manifest", type=str, default=r"data\manifests\master_split.csv")
    p.add_argument("--ckpt", type=str, default="best_grade.pth")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = fill_native_internal_test(
        project_root=Path(args.project_root).resolve(),
        run_dir=Path(args.run_dir).resolve(),
        manifest_arg=args.manifest,
        ckpt_name=args.ckpt,
        device=str(args.device),
        num_workers=int(args.num_workers),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
