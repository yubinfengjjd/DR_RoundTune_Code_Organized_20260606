"""Paper metrics + bootstrap confidence intervals.

Non-invasive analysis script. Reads prediction CSV(s) and computes:
- Multiclass: QWK, accuracy, macro-F1 (argmax and, if present, posthoc/thresholded).
- Referral endpoints (binary): y>=k (k in {2,3} by default)
  AUROC, AUPRC, sensitivity/specificity at selected probability thresholds.

Also computes bootstrap 95% CIs and (optionally) paired bootstrap difference between
posthoc and argmax metrics.

Input CSV formats supported:
- Internal ensemble posthoc: true_grade, pred_argmax, pred_grade, p0..p4
- External ensemble: y_true, pred_argmax, pred_thresholded, p0..p4
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _try_int(v: Any) -> Optional[int]:
    if v in (None, ""):
        return None
    try:
        return int(float(v))
    except Exception:
        return None


def _try_float(v: Any) -> Optional[float]:
    if v in (None, ""):
        return None
    try:
        return float(v)
    except Exception:
        return None


def _row_probs(r: Dict[str, str], k: int = 5) -> Optional[np.ndarray]:
    vals: List[float] = []
    for i in range(int(k)):
        vv = _try_float(r.get(f"p{i}"))
        if vv is None:
            return None
        vals.append(float(vv))
    p = np.asarray(vals, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p


def _qwk_numpy(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
    y_pred = np.asarray(y_pred).astype(np.int64).reshape(-1)
    if y_true.size == 0:
        return float("nan")
    k = int(k)
    m = (y_true >= 0) & (y_true < k) & (y_pred >= 0) & (y_pred < k)
    if int(m.sum()) == 0:
        return float("nan")
    y_true = y_true[m]
    y_pred = y_pred[m]
    if int(np.count_nonzero(np.bincount(y_true, minlength=k))) <= 1:
        return float("nan")
    cm = np.zeros((k, k), dtype=np.float64)
    np.add.at(cm, (y_true, y_pred), 1)
    n = float(cm.sum())
    if n <= 0:
        return float("nan")
    w = (np.subtract.outer(np.arange(k), np.arange(k)) ** 2) / float((k - 1) ** 2)
    hist_true = cm.sum(axis=1)
    hist_pred = cm.sum(axis=0)
    expected = np.outer(hist_true, hist_pred) / max(eps, n)
    den = float((w * expected).sum())
    num = float((w * cm).sum())
    if den <= 0:
        return float("nan")
    return float(1.0 - (num / den))


def _acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    m = (y_true >= 0) & (y_pred >= 0)
    if int(m.sum()) == 0:
        return float("nan")
    return float((y_true[m] == y_pred[m]).mean())


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5) -> float:
    y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
    y_pred = np.asarray(y_pred).astype(np.int64).reshape(-1)
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
    return float(np.mean(np.asarray(f1s, dtype=np.float64)))


def _roc_curve_binary(y_true01: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true01).astype(np.int64).reshape(-1)
    s = np.asarray(y_score).astype(np.float64).reshape(-1)
    m = np.isfinite(s) & ((y == 0) | (y == 1))
    y = y[m]
    s = s[m]
    if y.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    P = int((y == 1).sum())
    N = int((y == 0).sum())
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    tpr = tp / max(1, P)
    fpr = fp / max(1, N)
    # prepend origin
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    return fpr, tpr


def auroc_binary(y_true01: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr = _roc_curve_binary(y_true01, y_score)
    if fpr.size < 2:
        return float("nan")
    return float(np.trapezoid(tpr, fpr))


def _pr_curve_binary(y_true01: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true01).astype(np.int64).reshape(-1)
    s = np.asarray(y_score).astype(np.float64).reshape(-1)
    m = np.isfinite(s) & ((y == 0) | (y == 1))
    y = y[m]
    s = s[m]
    if y.size == 0:
        return np.array([0.0, 1.0]), np.array([1.0, 0.0])
    P = int((y == 1).sum())
    if P == 0:
        return np.array([0.0, 1.0]), np.array([1.0, 0.0])
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(1, (tp + fp))
    recall = tp / max(1, P)
    # prepend start
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return recall, precision


def auprc_binary(y_true01: np.ndarray, y_score: np.ndarray) -> float:
    recall, precision = _pr_curve_binary(y_true01, y_score)
    if recall.size < 2:
        return float("nan")
    return float(np.trapezoid(precision, recall))


def sens_spec_at_pt(y_true01: np.ndarray, y_score: np.ndarray, pt: float) -> Dict[str, float]:
    y = np.asarray(y_true01).astype(np.int64).reshape(-1)
    s = np.asarray(y_score).astype(np.float64).reshape(-1)
    m = np.isfinite(s) & ((y == 0) | (y == 1))
    y = y[m]
    s = s[m]
    if y.size == 0:
        return {"sens": float("nan"), "spec": float("nan"), "ppv": float("nan"), "npv": float("nan")}
    pred = (s >= float(pt)).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    return {"sens": float(sens), "spec": float(spec), "ppv": float(ppv), "npv": float(npv)}


def brier_binary(y_true01: np.ndarray, y_score: np.ndarray) -> float:
    y = np.asarray(y_true01).astype(np.float64).reshape(-1)
    s = np.asarray(y_score).astype(np.float64).reshape(-1)
    m = np.isfinite(s) & ((y == 0) | (y == 1))
    y = y[m]
    s = s[m]
    if y.size == 0:
        return float("nan")
    return float(np.mean((s - y) ** 2))


def ece_binary(y_true01: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y_true01).astype(np.int64).reshape(-1)
    s = np.asarray(y_score).astype(np.float64).reshape(-1)
    m = np.isfinite(s) & ((y == 0) | (y == 1))
    y = y[m]
    s = s[m]
    if y.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    N = float(y.size)
    for i in range(int(n_bins)):
        lo, hi = float(bins[i]), float(bins[i + 1])
        if i == int(n_bins) - 1:
            mm = (s >= lo) & (s <= hi)
        else:
            mm = (s >= lo) & (s < hi)
        n = int(mm.sum())
        if n == 0:
            continue
        acc = float(y[mm].mean())
        conf = float(s[mm].mean())
        ece += (n / N) * abs(acc - conf)
    return float(ece)


@dataclass
class DatasetPreds:
    name: str
    csv_path: str
    y_true: np.ndarray
    probs: np.ndarray
    pred_argmax: np.ndarray
    pred_posthoc: Optional[np.ndarray]


def load_dataset(name: str, csv_path: Path) -> DatasetPreds:
    rows = _read_rows(csv_path)
    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")

    y_true: List[int] = []
    probs: List[np.ndarray] = []
    pred_argmax: List[int] = []
    pred_post: List[int] = []
    has_post = False

    for r in rows:
        yt = None
        for k in ("true_grade", "y_true"):
            yt = _try_int(r.get(k))
            if yt is not None:
                break
        if yt is None:
            continue

        p = _row_probs(r, k=5)
        if p is None:
            continue

        pa = _try_int(r.get("pred_argmax"))
        if pa is None:
            pa = int(np.argmax(p))

        pp = None
        for k in ("pred_grade", "pred_thresholded"):
            pp = _try_int(r.get(k))
            if pp is not None:
                break
        if pp is not None:
            has_post = True
            pred_post.append(int(pp))
        else:
            pred_post.append(-1)

        y_true.append(int(yt))
        probs.append(p)
        pred_argmax.append(int(pa))

    y = np.asarray(y_true, dtype=np.int64)
    P = np.stack(probs, axis=0).astype(np.float64)
    pa = np.asarray(pred_argmax, dtype=np.int64)
    pp_arr = np.asarray(pred_post, dtype=np.int64)
    return DatasetPreds(
        name=str(name),
        csv_path=str(csv_path),
        y_true=y,
        probs=P,
        pred_argmax=pa,
        pred_posthoc=(pp_arr if has_post else None),
    )


def _pct_ci(x: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    lo = float(np.quantile(x, alpha / 2.0))
    hi = float(np.quantile(x, 1.0 - alpha / 2.0))
    return lo, hi


def _bootstrap_indices(n: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, int(n), size=(int(n_boot), int(n)), endpoint=False, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        action="append",
        default=[],
        help="Repeatable. Format: name:path_to_csv",
    )
    ap.add_argument("--out_dir", type=str, default=r"checkpoints/paper_assets/metrics_ci")
    ap.add_argument("--tag", type=str, default="phase4")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--endpoints", type=str, default="2,3", help="Comma-separated k values for y>=k")
    ap.add_argument("--op_pts", type=str, default="0.1,0.2", help="Comma-separated probability thresholds for sens/spec")
    ap.add_argument("--bins", type=int, default=10, help="Bins for ECE (binary)")
    ap.add_argument("--no_diff", type=int, default=0, choices=[0, 1], help="If 1, skip paired diff bootstrap")
    args = ap.parse_args()

    items: List[Tuple[str, Path]] = []
    for s in list(args.csv):
        if ":" not in str(s):
            raise ValueError("--csv must be name:path")
        name, p = str(s).split(":", 1)
        name = name.strip()
        pp = Path(p.strip()).resolve()
        if not pp.exists():
            raise FileNotFoundError(str(pp))
        items.append((name, pp))
    if not items:
        raise ValueError("Provide at least one --csv name:path")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = str(args.tag).strip() or "phase4"

    ks = [int(x) for x in str(args.endpoints).split(",") if x.strip()]
    op_pts = [float(x) for x in str(args.op_pts).split(",") if x.strip()]
    n_boot = int(args.n_boot)
    if n_boot <= 0:
        raise ValueError("--n_boot must be > 0")
    rng = np.random.default_rng(int(args.seed))
    alpha = float(args.alpha)
    n_bins = int(args.bins)
    if n_bins <= 1:
        raise ValueError("--bins must be >= 2")

    multiclass_rows: List[Dict[str, Any]] = []
    referral_rows: List[Dict[str, Any]] = []
    diff_rows: List[Dict[str, Any]] = []

    for name, csv_path in items:
        ds = load_dataset(name, csv_path)
        n = int(ds.y_true.size)
        idx_boot = _bootstrap_indices(n, n_boot=n_boot, rng=rng)

        # Multiclass: argmax
        def mc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
            return {
                "qwk": _qwk_numpy(y_true, y_pred, k=5),
                "acc": _acc(y_true, y_pred),
                "macro_f1": _macro_f1(y_true, y_pred, k=5),
            }

        mc0 = mc_metrics(ds.y_true, ds.pred_argmax)
        boot_qwk = np.array([mc_metrics(ds.y_true[i], ds.pred_argmax[i])["qwk"] for i in idx_boot], dtype=np.float64)
        boot_acc = np.array([mc_metrics(ds.y_true[i], ds.pred_argmax[i])["acc"] for i in idx_boot], dtype=np.float64)
        boot_f1 = np.array([mc_metrics(ds.y_true[i], ds.pred_argmax[i])["macro_f1"] for i in idx_boot], dtype=np.float64)

        for metric, est, boot in [
            ("qwk", mc0["qwk"], boot_qwk),
            ("acc", mc0["acc"], boot_acc),
            ("macro_f1", mc0["macro_f1"], boot_f1),
        ]:
            lo, hi = _pct_ci(boot, alpha=alpha)
            multiclass_rows.append(
                {
                    "dataset": ds.name,
                    "variant": "argmax",
                    "metric": metric,
                    "estimate": float(est),
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                    "n": n,
                    "csv": ds.csv_path,
                }
            )

        # Multiclass: posthoc (if present)
        if ds.pred_posthoc is not None:
            mc1 = mc_metrics(ds.y_true, ds.pred_posthoc)
            boot_qwk2 = np.array([mc_metrics(ds.y_true[i], ds.pred_posthoc[i])["qwk"] for i in idx_boot], dtype=np.float64)
            boot_acc2 = np.array([mc_metrics(ds.y_true[i], ds.pred_posthoc[i])["acc"] for i in idx_boot], dtype=np.float64)
            boot_f12 = np.array([mc_metrics(ds.y_true[i], ds.pred_posthoc[i])["macro_f1"] for i in idx_boot], dtype=np.float64)

            for metric, est, boot in [
                ("qwk", mc1["qwk"], boot_qwk2),
                ("acc", mc1["acc"], boot_acc2),
                ("macro_f1", mc1["macro_f1"], boot_f12),
            ]:
                lo, hi = _pct_ci(boot, alpha=alpha)
                multiclass_rows.append(
                    {
                        "dataset": ds.name,
                        "variant": "posthoc",
                        "metric": metric,
                        "estimate": float(est),
                        "ci_low": float(lo),
                        "ci_high": float(hi),
                        "n": n,
                        "csv": ds.csv_path,
                    }
                )

            if int(args.no_diff) == 0:
                for metric, boot_a, boot_b in [
                    ("qwk", boot_qwk, boot_qwk2),
                    ("acc", boot_acc, boot_acc2),
                    ("macro_f1", boot_f1, boot_f12),
                ]:
                    diff = boot_b - boot_a
                    lo, hi = _pct_ci(diff, alpha=alpha)
                    p_two = 2.0 * min(float(np.mean(diff <= 0.0)), float(np.mean(diff >= 0.0)))
                    diff_rows.append(
                        {
                            "dataset": ds.name,
                            "metric": metric,
                            "delta_posthoc_minus_argmax": float(np.nanmean(diff)),
                            "ci_low": float(lo),
                            "ci_high": float(hi),
                            "p_two_sided": float(min(1.0, max(0.0, p_two))),
                            "n": n,
                        }
                    )

        # Referral endpoints from probabilities (common across variants)
        for k in ks:
            if k < 1 or k > 4:
                raise ValueError("endpoints must be in [1..4]")
            y01 = (ds.y_true >= int(k)).astype(np.int64)
            p_hat = ds.probs[:, int(k) :].sum(axis=1)
            auroc = auroc_binary(y01, p_hat)
            auprc = auprc_binary(y01, p_hat)
            br = brier_binary(y01, p_hat)
            ece = ece_binary(y01, p_hat, n_bins=n_bins)

            # Bootstrap
            boot_auroc = np.array([auroc_binary(y01[i], p_hat[i]) for i in idx_boot], dtype=np.float64)
            boot_auprc = np.array([auprc_binary(y01[i], p_hat[i]) for i in idx_boot], dtype=np.float64)
            boot_brier = np.array([brier_binary(y01[i], p_hat[i]) for i in idx_boot], dtype=np.float64)
            boot_ece = np.array([ece_binary(y01[i], p_hat[i], n_bins=n_bins) for i in idx_boot], dtype=np.float64)

            for metric, est, boot in [
                ("auroc", auroc, boot_auroc),
                ("auprc", auprc, boot_auprc),
                ("brier", br, boot_brier),
                ("ece", ece, boot_ece),
            ]:
                lo, hi = _pct_ci(boot, alpha=alpha)
                referral_rows.append(
                    {
                        "dataset": ds.name,
                        "endpoint": f"y>= {k}",
                        "metric": metric,
                        "estimate": float(est),
                        "ci_low": float(lo),
                        "ci_high": float(hi),
                        "n": n,
                    }
                )

            for pt in op_pts:
                ss = sens_spec_at_pt(y01, p_hat, pt=float(pt))
                referral_rows.append(
                    {
                        "dataset": ds.name,
                        "endpoint": f"y>= {k}",
                        "metric": f"sens@{pt}",
                        "estimate": float(ss["sens"]),
                        "ci_low": "",
                        "ci_high": "",
                        "n": n,
                    }
                )
                referral_rows.append(
                    {
                        "dataset": ds.name,
                        "endpoint": f"y>= {k}",
                        "metric": f"spec@{pt}",
                        "estimate": float(ss["spec"]),
                        "ci_low": "",
                        "ci_high": "",
                        "n": n,
                    }
                )

    # Write outputs
    out_mc = out_dir / f"paper_multiclass_ci_{tag}.csv"
    out_ref = out_dir / f"paper_referral_ci_{tag}.csv"
    out_diff = out_dir / f"paper_posthoc_deltas_{tag}.csv"
    out_json = out_dir / f"paper_metrics_ci_{tag}.json"

    def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(fields))
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})

    write_csv(
        out_mc,
        multiclass_rows,
        ["dataset", "variant", "metric", "estimate", "ci_low", "ci_high", "n", "csv"],
    )
    write_csv(
        out_ref,
        referral_rows,
        ["dataset", "endpoint", "metric", "estimate", "ci_low", "ci_high", "n"],
    )
    write_csv(
        out_diff,
        diff_rows,
        ["dataset", "metric", "delta_posthoc_minus_argmax", "ci_low", "ci_high", "p_two_sided", "n"],
    )
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "tag": tag,
                "n_boot": int(n_boot),
                "seed": int(args.seed),
                "alpha": float(alpha),
                "endpoints": ks,
                "op_pts": op_pts,
                "bins": int(n_bins),
                "outputs": {
                    "multiclass_csv": str(out_mc),
                    "referral_csv": str(out_ref),
                    "deltas_csv": str(out_diff),
                },
                "inputs": [{"name": n, "csv": str(p)} for n, p in items],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("[OK] wrote:", out_mc)
    print("[OK] wrote:", out_ref)
    print("[OK] wrote:", out_diff)
    print("[OK] wrote:", out_json)


if __name__ == "__main__":
    main()
