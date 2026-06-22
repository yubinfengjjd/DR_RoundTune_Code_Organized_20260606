import argparse
import csv
import json
from itertools import product
from pathlib import Path


PROB_COLS = ["p0", "p1", "p2", "p3", "p4"]
META_COLS = ["image_path", "path", "key", "patient_id", "eye", "split", "true_grade"]
OUT_COLS = META_COLS + ["pred_argmax", "pred_grade", "score"] + PROB_COLS


def apply_thresholds(scores, thresholds):
    grades = []
    for score in scores:
        grade = 0
        for threshold in thresholds:
            if float(score) >= float(threshold):
                grade += 1
        grades.append(grade)
    return grades


def quadratic_weighted_kappa(y_true, y_pred, num_classes=5):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        return 0.0

    observed = [[0.0 for _ in range(num_classes)] for _ in range(num_classes)]
    hist_true = [0.0 for _ in range(num_classes)]
    hist_pred = [0.0 for _ in range(num_classes)]
    for truth, pred in zip(y_true, y_pred):
        truth_i = int(truth)
        pred_i = int(pred)
        observed[truth_i][pred_i] += 1.0
        hist_true[truth_i] += 1.0
        hist_pred[pred_i] += 1.0

    n = float(len(y_true))
    weighted_observed = 0.0
    weighted_expected = 0.0
    denom = float((num_classes - 1) ** 2)
    for i in range(num_classes):
        for j in range(num_classes):
            weight = ((i - j) ** 2) / denom
            expected = hist_true[i] * hist_pred[j] / n
            weighted_observed += weight * observed[i][j]
            weighted_expected += weight * expected

    if weighted_expected == 0.0:
        return 1.0 if weighted_observed == 0.0 else 0.0
    return 1.0 - weighted_observed / weighted_expected


def fit_thresholds(scores, y_true, grid_step=0.05, search_iters=2):
    thresholds = [0.5, 1.5, 2.5, 3.5]
    grid = [round(i * grid_step, 6) for i in range(int(4.0 / grid_step) + 1)]

    def valid(candidate):
        return all(candidate[i] < candidate[i + 1] for i in range(3))

    best_qwk = quadratic_weighted_kappa(y_true, apply_thresholds(scores, thresholds))
    for _ in range(int(search_iters)):
        improved = False
        for idx in range(4):
            best_local = thresholds[idx]
            for value in grid:
                candidate = list(thresholds)
                candidate[idx] = value
                if not valid(candidate):
                    continue
                qwk = quadratic_weighted_kappa(y_true, apply_thresholds(scores, candidate))
                if qwk > best_qwk:
                    best_qwk = qwk
                    best_local = value
                    improved = True
            thresholds[idx] = best_local
        if not improved:
            break
    return thresholds, best_qwk


def read_seed_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty prediction CSV: {path}")
    missing = [col for col in META_COLS + PROB_COLS if col not in rows[0]]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return rows


def average_seed_rows(seed_csvs):
    per_seed = [read_seed_csv(path) for path in seed_csvs]
    maps = [{row["key"]: row for row in rows} for rows in per_seed]
    keys = sorted(maps[0])
    for mapping in maps[1:]:
        if sorted(mapping) != keys:
            raise ValueError("Seed prediction CSVs must contain the same keys")

    out_rows = []
    for key in keys:
        seed_rows = [mapping[key] for mapping in maps]
        first = seed_rows[0]
        out = {col: first[col] for col in META_COLS}
        probs = []
        for col in PROB_COLS:
            probs.append(sum(float(row[col]) for row in seed_rows) / len(seed_rows))
        score = sum(index * prob for index, prob in enumerate(probs))
        pred_argmax = max(range(len(probs)), key=lambda index: probs[index])
        out.update({
            "pred_argmax": pred_argmax,
            "score": score,
        })
        for col, value in zip(PROB_COLS, probs):
            out[col] = value
        out_rows.append(out)
    return out_rows


def finalize_rows(rows, thresholds):
    scores = [float(row["score"]) for row in rows]
    pred_grades = apply_thresholds(scores, thresholds)
    for row, grade in zip(rows, pred_grades):
        row["pred_grade"] = grade
    return rows


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_COLS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row[col] for col in OUT_COLS})


def accuracy(y_true, y_pred):
    if not y_true:
        return 0.0
    return sum(int(a) == int(b) for a, b in zip(y_true, y_pred)) / len(y_true)


def metric_block(rows, pred_col):
    y_true = [int(row["true_grade"]) for row in rows]
    y_pred = [int(row[pred_col]) for row in rows]
    return {"acc": accuracy(y_true, y_pred), "qwk": quadratic_weighted_kappa(y_true, y_pred)}


def summary_name(prefix, tta):
    return f"{prefix}.json" if int(tta) == 8 else f"{prefix}_tta{int(tta)}.json"


def build_ensemble_posthoc(seed_run_dirs, out_dir, tta, thresholds=None, grid_step=0.05, search_iters=2):
    seed_dirs = [Path(path) for path in seed_run_dirs]
    out_dir = Path(out_dir)
    tta = int(tta)

    calib_csvs = [run_dir / f"preds_image_ce_tta{tta}_calib.csv" for run_dir in seed_dirs]
    test_csvs = [run_dir / f"preds_image_ce_tta{tta}_test.csv" for run_dir in seed_dirs]
    for path in calib_csvs + test_csvs:
        if not path.exists():
            raise FileNotFoundError(path)

    calib_rows = average_seed_rows(calib_csvs)
    test_rows = average_seed_rows(test_csvs)

    calib_scores = [float(row["score"]) for row in calib_rows]
    calib_truth = [int(row["true_grade"]) for row in calib_rows]
    if thresholds is None:
        thresholds, search_qwk = fit_thresholds(calib_scores, calib_truth, grid_step=grid_step, search_iters=search_iters)
    else:
        thresholds = [float(value) for value in thresholds]
        search_qwk = quadratic_weighted_kappa(calib_truth, apply_thresholds(calib_scores, thresholds))

    finalize_rows(calib_rows, thresholds)
    finalize_rows(test_rows, thresholds)

    calib_out = out_dir / f"preds_image_ens_tta{tta}_calib_thresholded.csv"
    test_out = out_dir / f"preds_image_ens_tta{tta}_test_thresholded.csv"
    write_rows(calib_out, calib_rows)
    write_rows(test_out, test_rows)

    thresholds_payload = {
        "thresholds": thresholds,
        "tta": tta,
        "grid_step": grid_step,
        "search_iters": search_iters,
        "search_qwk": search_qwk,
        "method": "expected_score_thresholds_max_qwk",
        "runs": [str(path) for path in seed_dirs],
    }
    thresholds_path = out_dir / summary_name("calib_thresholds_ens", tta)
    thresholds_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds_path.write_text(json.dumps(thresholds_payload, indent=2), encoding="utf-8")

    summary = {
        "tta": tta,
        "n_seed_runs": len(seed_dirs),
        "n_calib": len(calib_rows),
        "n_test": len(test_rows),
        "thresholds": thresholds,
        "calib_argmax": metric_block(calib_rows, "pred_argmax"),
        "calib_thresholded": metric_block(calib_rows, "pred_grade"),
        "test_argmax": metric_block(test_rows, "pred_argmax"),
        "test_thresholded": metric_block(test_rows, "pred_grade"),
        "outputs": {
            "calib_csv": str(calib_out),
            "test_csv": str(test_out),
            "thresholds_json": str(thresholds_path),
        },
    }
    summary_path = out_dir / summary_name("eval_summary_ens", tta)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_run_dirs", required=True, help="Comma-separated run dirs ending with grade")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tta", type=int, required=True)
    parser.add_argument("--thresholds", default="", help="Optional comma-separated thresholds")
    parser.add_argument("--threshold_grid_step", type=float, default=0.05)
    parser.add_argument("--threshold_search_iters", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_run_dirs = [item.strip() for item in args.seed_run_dirs.split(",") if item.strip()]
    thresholds = None
    if args.thresholds.strip():
        thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    summary = build_ensemble_posthoc(
        seed_run_dirs,
        args.out_dir,
        args.tta,
        thresholds=thresholds,
        grid_step=args.threshold_grid_step,
        search_iters=args.threshold_search_iters,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
