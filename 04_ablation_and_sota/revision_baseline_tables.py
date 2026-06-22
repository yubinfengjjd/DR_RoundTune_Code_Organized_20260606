import argparse
import csv
import json
from pathlib import Path


FIELDS = ["method", "group", "status", "tta", "calib_acc", "calib_qwk", "test_acc", "test_qwk", "summary_path"]


def read_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def extract_eval_metrics(summary):
    image = summary.get("image_level", summary)
    calib = image["calib_thresholded"]
    test = image["test_thresholded"]
    return {
        "tta": int(summary["tta"]),
        "calib_acc": float(calib["acc"]),
        "calib_qwk": float(calib["qwk"]),
        "test_acc": float(test["acc"]),
        "test_qwk": float(test["qwk"]),
    }


def row_from_summary_path(method, summary_path, group):
    path = Path(summary_path)
    if not path.exists():
        return {
            "method": method,
            "group": group,
            "status": "missing",
            "tta": "",
            "calib_acc": "",
            "calib_qwk": "",
            "test_acc": "",
            "test_qwk": "",
            "summary_path": str(path),
        }
    metrics = extract_eval_metrics(read_json(path))
    return {
        "method": method,
        "group": group,
        "status": "available",
        "summary_path": str(path),
        **metrics,
    }


def default_specs(project_root):
    root = Path(project_root)
    return [
        ("CE", "loss", root / "checkpoints/Loss Fine-Tuning/runs_grade_ce/grade/eval_summary_ce_tta4.json"),
        ("Focal", "loss", root / "checkpoints/Loss Fine-Tuning/runs_grade_focal/grade/eval_summary_ce_tta4.json"),
        ("CORN", "loss", root / "checkpoints/Loss Fine-Tuning/runs_grade_corn/grade/eval_summary_ce_tta4.json"),
        ("CE-QWK", "loss", root / "checkpoints/Loss Fine-Tuning/runs_grade_ce_qwk/grade/eval_summary_ce_tta4.json"),
        ("Ordinal", "loss", root / "checkpoints/Loss Fine-Tuning/runs_grade_ordinal/grade/eval_summary_ce_tta4.json"),
        ("EDL", "uncertainty-loss", root / "checkpoints/EDL/runs_grade_edl_seed42_20260129_023209/grade/eval_summary_edl_tta4.json"),
        ("Freeze/head-only", "tuning", root / "checkpoints/Fine-tuning ablation/runs_grade_tune_freeze/grade/eval_summary_ce_tta4.json"),
        ("Full fine-tuning", "tuning", root / "checkpoints/Fine-tuning ablation/runs_grade_tune_full/grade/eval_summary_ce_tta4.json"),
        ("LoRA-r16", "tuning", root / "checkpoints/Fine-tuning ablation/runs_grade_tune_lora_r16/grade/eval_summary_ce_tta4.json"),
        ("Quality-aware init LoRA-r16", "quality-pretraining", root / "checkpoints/Fine-tuning ablation/runs_grade_tune_lora_r16/grade/eval_summary_ce_tta4.json"),
        ("No-quality init LoRA-r16", "quality-pretraining", root / "checkpoints/Quality pretraining ablation/runs_grade_no_quality_init/grade/eval_summary_ce_tta4.json"),
        ("LoRA-r4", "lora-rank", root / "checkpoints/Fine-tuning ablation/runs_grade_tune_lora_r4/grade/eval_summary_ce_tta4.json"),
        ("LoRA-r8", "lora-rank", root / "checkpoints/Fine-tuning ablation/runs_grade_tune_lora_r8/grade/eval_summary_ce_tta4.json"),
        ("LoRA-r32", "lora-rank", root / "checkpoints/Fine-tuning ablation/runs_grade_tune_lora_r32/grade/eval_summary_ce_tta4.json"),
        ("5-seed ensemble", "ensemble", root / "checkpoints/Multi-seed integration/ensemble_posthoc/eval_summary_ens_tta4.json"),
    ]


def build_baseline_table(project_root, out_dir):
    rows = [row_from_summary_path(method, path, group) for method, group, path in default_specs(project_root)]
    out = Path(out_dir) / "revision_final_baseline_comparison_tta4.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    out = build_baseline_table(args.project_root, args.out_dir)
    print(out)


if __name__ == "__main__":
    main()
