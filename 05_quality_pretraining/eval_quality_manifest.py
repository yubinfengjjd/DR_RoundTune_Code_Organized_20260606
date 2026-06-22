"""Run a frozen quality head on a locked manifest split for post-hoc analysis."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


def read_rows(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def filter_manifest_rows(rows: Sequence[Mapping[str, str]], *, split: str) -> List[Dict[str, str]]:
    selected = [dict(row) for row in rows if str(row.get("split", "")).strip().lower() == str(split).strip().lower()]
    if not selected:
        raise ValueError(f"No rows found for locked split='{split}'")
    return selected


def quality_prediction_row(
    manifest_row: Mapping[str, str],
    resolved_path: str,
    probs: Sequence[float],
    *,
    reject_index: int,
) -> Dict[str, Any]:
    values = [float(value) for value in probs]
    if not 0 <= int(reject_index) < len(values):
        raise ValueError(f"reject_index={reject_index} outside probability vector length={len(values)}")
    return {
        "key": manifest_row.get("key", ""),
        "image_path": manifest_row.get("image_path", ""),
        "path": str(resolved_path),
        "split": manifest_row.get("split", ""),
        "pred_quality": max(range(len(values)), key=lambda idx: values[idx]),
        "p_reject": values[int(reject_index)],
        **{f"p{i}": value for i, value in enumerate(values)},
    }


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("No quality prediction rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_inference(args: argparse.Namespace) -> Path:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    project_root = Path(args.project_root).resolve()
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    import eval_external as ext
    import train_roundtune_cpf as rt

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = project_root / run_dir
    summary_path = run_dir / "logs" / "summary.json"
    ckpt_path = run_dir / "best_quality.pth"
    with summary_path.open(encoding="utf-8") as f:
        summary = json.load(f)
    cfg = summary.get("config", {})
    state = rt.load_checkpoint_state(ckpt_path)
    backbone_name = str(cfg.get("backbone", "") or ext._infer_backbone_name(state))
    num_classes = ext._infer_cls_num_classes_from_state(state)
    backbone = rt.SharedBackbone(backbone_name)
    ext._maybe_apply_lora(backbone, state)
    model = rt.QualityModel(backbone, num_classes_quality=num_classes, dropout=0.0).to(args.device)
    info = rt.safe_load(model, state, strict=False)
    if info["missing"] or info["unexpected"]:
        raise RuntimeError(f"Quality checkpoint load mismatch: missing={len(info['missing'])} unexpected={len(info['unexpected'])}")

    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = project_root / manifest
    rows = filter_manifest_rows(read_rows(manifest), split=args.split)
    dataset = rt.QualityDataset(rows, project_root, int(cfg.get("img_size", 224)), False, collapse_to_binary=False)
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers), pin_memory=str(args.device).startswith("cuda"))

    outputs: List[Dict[str, Any]] = []
    offset = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(args.device, non_blocking=True)
            with rt.amp_autocast(args.device, bool(args.amp)):
                logits, _ = model(images)
            probs = torch.softmax(logits.float(), dim=1).detach().cpu().numpy().astype(np.float64)
            paths = batch.get("path", [""] * len(probs))
            for index, prob_row in enumerate(probs):
                outputs.append(
                    quality_prediction_row(
                        rows[offset + index],
                        str(paths[index]),
                        prob_row.tolist(),
                        reject_index=int(args.reject_index),
                    )
                )
            offset += len(probs)
            print(f"[QUAL-POSTHOC] processed={offset}/{len(rows)}", flush=True)

    out_csv = Path(args.out_csv)
    if not out_csv.is_absolute():
        out_csv = project_root / out_csv
    write_rows(out_csv, outputs)
    print(f"[OK] wrote: {out_csv}")
    return out_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--manifest", default="data/manifests/master_split.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--reject_index", type=int, default=2, help="Three-class quality convention: 0=Good, 1=Usable, 2=Reject.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_inference(parse_args())


if __name__ == "__main__":
    main()
