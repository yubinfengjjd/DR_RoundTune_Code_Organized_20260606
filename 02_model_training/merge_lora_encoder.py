"""Merge LoRA-wrapped encoder checkpoints into dense encoder checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch


def load_state(path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(str(path), map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"Checkpoint is not a state dict: {path}")
    return obj


def merge_lora_state_dict(state: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    merged: Dict[str, torch.Tensor] = {}
    consumed = set()
    merged_count = 0

    for key, value in state.items():
        if not key.endswith(".base.weight"):
            continue

        prefix = key[: -len(".base.weight")]
        dense_key = f"{prefix}.weight"
        bias_key = f"{prefix}.base.bias"
        dense_bias_key = f"{prefix}.bias"
        a_key = f"{prefix}.lora_A"
        b_key = f"{prefix}.lora_B"
        alpha_key = f"{prefix}.lora_alpha"

        if a_key in state and b_key in state:
            base = value
            a = state[a_key]
            b = state[b_key]
            r = int(a.shape[0])
            alpha = float(state.get(alpha_key, torch.tensor(float(r))).detach().cpu().item())
            delta = torch.matmul(b.float(), a.float()) * (alpha / float(r))
            merged[dense_key] = (base.float() + delta).to(dtype=base.dtype)
            consumed.update({key, a_key, b_key, alpha_key})
            if bias_key in state:
                merged[dense_bias_key] = state[bias_key]
                consumed.add(bias_key)
            merged_count += 1
        else:
            merged[dense_key] = value
            consumed.add(key)
            if bias_key in state:
                merged[dense_bias_key] = state[bias_key]
                consumed.add(bias_key)

    for key, value in state.items():
        if key in consumed:
            continue
        if ".lora_" in key or ".base." in key:
            continue
        merged[key] = value

    report = {
        "input_key_count": len(state),
        "output_key_count": len(merged),
        "merged_linear_count": merged_count,
        "copied_key_count": len(merged) - merged_count,
        "dropped_lora_key_count": sum(1 for key in state if ".lora_" in key),
    }
    return merged, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--report", default="", type=str)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve() if args.report else output_path.with_suffix(".merge_report.json")

    state = load_state(input_path)
    merged, report = merge_lora_state_dict(state)
    report = {**report, "input": str(input_path), "output": str(output_path)}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, str(output_path))
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[OK] merged LoRA encoder")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
