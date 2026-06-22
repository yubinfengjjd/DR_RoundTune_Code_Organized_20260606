import argparse
import csv
import itertools
import json
import math
from pathlib import Path

from revision_final_tables import compute_ece_from_csv
from revision_tdd_ensemble import build_ensemble_posthoc


PROB_COLS = ["p0", "p1", "p2", "p3", "p4"]


def seed_subsets(seed_run_dirs, sizes):
    seed_run_dirs = list(seed_run_dirs)
    out = {}
    for size in sizes:
        out[int(size)] = [tuple(combo) for combo in itertools.combinations(seed_run_dirs, int(size))]
    return out


def mean(values):
    return sum(values) / len(values) if values else 0.0


def population_sd(values):
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def summarize_records(records):
    rows = []
    for n_seeds in sorted({int(row["n_seeds"]) for row in records}):
        group = [row for row in records if int(row["n_seeds"]) == n_seeds]
        out = {"n_seeds": n_seeds, "n_subsets": len(group)}
        for metric in ["test_qwk", "test_acc", "test_ece", "mean_pred_entropy"]:
            values = [float(row[metric]) for row in group]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_sd"] = population_sd(values)
        rows.append(out)
    return rows


def prediction_entropy_summary(path):
    entropies = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            probs = [float(row[col]) for col in PROB_COLS]
            entropy = -sum(prob * math.log(prob + 1e-12) for prob in probs)
            entropies.append(entropy)
    return mean(entropies)


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def subset_name(subset):
    return "__".join(Path(item).parents[0].name for item in subset)


def run_seed_ablation(seed_run_dirs, out_dir, tta=4, sizes=(1, 3, 5), grid_step=0.01, search_iters=2):
    out_dir = Path(out_dir)
    records = []
    subsets = seed_subsets(seed_run_dirs, sizes)
    for n_seeds, combos in subsets.items():
        for index, combo in enumerate(combos, start=1):
            subset_dir = out_dir / f"n{n_seeds}_subset{index:02d}"
            summary = build_ensemble_posthoc(
                combo,
                subset_dir,
                tta=tta,
                grid_step=grid_step,
                search_iters=search_iters,
            )
            test_csv = subset_dir / f"preds_image_ens_tta{int(tta)}_test_thresholded.csv"
            records.append({
                "n_seeds": n_seeds,
                "subset_index": index,
                "subset": subset_name(combo),
                "tta": int(tta),
                "test_qwk": summary["test_thresholded"]["qwk"],
                "test_acc": summary["test_thresholded"]["acc"],
                "test_ece": compute_ece_from_csv(test_csv),
                "mean_pred_entropy": prediction_entropy_summary(test_csv),
                "subset_dir": str(subset_dir),
            })

    detail_path = out_dir / f"revision_final_seed_ablation_detail_tta{int(tta)}.csv"
    summary_path = out_dir / f"revision_final_seed_ablation_summary_tta{int(tta)}.csv"
    write_csv(
        detail_path,
        records,
        ["n_seeds", "subset_index", "subset", "tta", "test_qwk", "test_acc", "test_ece", "mean_pred_entropy", "subset_dir"],
    )
    summary_rows = summarize_records(records)
    write_csv(
        summary_path,
        summary_rows,
        [
            "n_seeds",
            "n_subsets",
            "test_qwk_mean",
            "test_qwk_sd",
            "test_acc_mean",
            "test_acc_sd",
            "test_ece_mean",
            "test_ece_sd",
            "mean_pred_entropy_mean",
            "mean_pred_entropy_sd",
        ],
    )
    metadata_path = out_dir / f"revision_final_seed_ablation_tta{int(tta)}.json"
    metadata_path.write_text(json.dumps({
        "tta": int(tta),
        "sizes": [int(size) for size in sizes],
        "n_records": len(records),
        "detail_csv": str(detail_path),
        "summary_csv": str(summary_path),
    }, indent=2), encoding="utf-8")
    return {"detail": detail_path, "summary": summary_path, "metadata": metadata_path}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_run_dirs", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tta", type=int, default=4)
    parser.add_argument("--sizes", default="1,3,5")
    parser.add_argument("--threshold_grid_step", type=float, default=0.01)
    parser.add_argument("--threshold_search_iters", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_run_dirs = [item.strip() for item in args.seed_run_dirs.split(",") if item.strip()]
    sizes = [int(item) for item in args.sizes.split(",") if item.strip()]
    outputs = run_seed_ablation(
        seed_run_dirs,
        args.out_dir,
        tta=args.tta,
        sizes=sizes,
        grid_step=args.threshold_grid_step,
        search_iters=args.threshold_search_iters,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
