import argparse
import csv
import json
from pathlib import Path


PROB_COLS = ["p0", "p1", "p2", "p3", "p4"]


def external_calibration_metrics(path, n_bins=15):
    records = []
    brier_sum = 0.0
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            truth = int(row["y_true"])
            pred = int(row["pred_thresholded"])
            probs = [float(row[col]) for col in PROB_COLS]
            confidence = max(probs)
            correct = pred == truth
            records.append((confidence, correct))
            for idx, prob in enumerate(probs):
                target = 1.0 if idx == truth else 0.0
                brier_sum += (prob - target) ** 2

    if not records:
        return {"n": 0, "accuracy": 0.0, "ece": 0.0, "brier": 0.0, "mean_confidence": 0.0}

    n = len(records)
    ece = 0.0
    for bin_index in range(n_bins):
        lo = bin_index / n_bins
        hi = (bin_index + 1) / n_bins
        if bin_index == n_bins - 1:
            in_bin = [(conf, corr) for conf, corr in records if lo <= conf <= hi]
        else:
            in_bin = [(conf, corr) for conf, corr in records if lo <= conf < hi]
        if not in_bin:
            continue
        avg_conf = sum(conf for conf, _ in in_bin) / len(in_bin)
        avg_acc = sum(1.0 if corr else 0.0 for _, corr in in_bin) / len(in_bin)
        ece += (len(in_bin) / n) * abs(avg_acc - avg_conf)

    return {
        "n": n,
        "accuracy": sum(1.0 if corr else 0.0 for _, corr in records) / n,
        "ece": ece,
        "brier": brier_sum / n,
        "mean_confidence": sum(conf for conf, _ in records) / n,
    }


def read_external_summary(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def build_external_calibration_table(summary_json, out_dir, n_bins=15):
    summary = read_external_summary(summary_json)
    rows = []
    for key, value in summary.items():
        if not key.startswith("ensemble_") or not isinstance(value, dict) or "csv" not in value:
            continue
        metrics = external_calibration_metrics(value["csv"], n_bins=n_bins)
        thresholded = value["metrics_thresholded"]
        rows.append({
            "cohort": key.replace("ensemble_", ""),
            "n": metrics["n"],
            "acc": thresholded["acc"],
            "macro_f1": thresholded["macro_f1"],
            "qwk": thresholded["qwk"],
            "ece": metrics["ece"],
            "brier": metrics["brier"],
            "mean_confidence": metrics["mean_confidence"],
            "csv": value["csv"],
        })

    out = Path(out_dir) / "revision_final_external_calibration.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["cohort", "n", "acc", "macro_f1", "qwk", "ece", "brier", "mean_confidence", "csv"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_bins", type=int, default=15)
    return parser.parse_args()


def main():
    args = parse_args()
    out = build_external_calibration_table(args.summary_json, args.out_dir, n_bins=args.n_bins)
    print(out)


if __name__ == "__main__":
    main()
