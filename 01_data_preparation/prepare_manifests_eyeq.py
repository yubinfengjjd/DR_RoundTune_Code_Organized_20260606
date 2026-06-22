# prepare_manifests_eyeq.py
# ------------------------------------------------------------
# Generate training manifests for DR_CPF_RoundTune project.
#
# Outputs (under <project_root>/data/manifests/):
#   - grade_eyepacs.csv         (image_path,label)
#   - qual_deepdrid.csv         (image_path,label)
#   - qual_eyeq.csv             (image_path,label,quality + optional meta)
#   - seg_idrid.csv             (image_path,mask_path,dataset,split)
#   - seg_ddr_lesions.csv       (image_path,mask_path,dataset,split)
#
# Multilabel masks (.npy, shape=(4,H,W), order=[MA,HE,EX,SE]):
#   <project_root>/data/processed/cfp_<img_size>/multilabel_masks/{idrid,ddr}/<sample_id>.npy
#
# Notes:
#   - data/meta/manifests/ is the offline index created by preprocess_offline.py
#   - data/manifests/ are task manifests created by this script
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

try:
    import cv2  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore


# --- Console encoding safety (Windows GBK) ---
try:  # pragma: no cover - best effort
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ---------------------------
# Small utilities
# ---------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sanitize_stem(stem: str) -> str:
    """Sanitize filename stem into a safe sample key."""
    stem = stem.replace(" ", "_")
    stem = re.sub(r"[^0-9a-zA-Z_\\-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem or "img"


def to_rel(project_root: Path, path_str: str) -> str:
    """Convert an absolute path to a path relative to project_root if possible."""
    p = Path(path_str)
    try:
        # NOTE: `project_root` is expected to be already resolved by callers (main uses `.resolve()`).
        # Avoid repeated `.resolve()` calls for speed (offline manifests can have ~50k+ rows).
        if p.is_absolute():
            return str(p.relative_to(project_root))
        return str(p)
    except Exception:
        # Fallback: resolve, then try again.
        try:
            return str(p.resolve().relative_to(project_root))
        except Exception:
            return str(p)

_EYE_TOKEN_RE = re.compile(r"^(?P<pid>\d+)[_-](?P<eye>left|right|l\d*|r\d*)$", flags=re.IGNORECASE)


def _norm_eye(eye: str) -> str:
    e = (eye or "").strip().lower()
    if e.startswith("l"):
        return "left"
    if e.startswith("r"):
        return "right"
    if e in ("left", "right"):
        return e
    return ""


def infer_eyepacs_key_from_any_id(s: str) -> str:
    """
    Infer a stable per-image key like: "<patient_id>_<left|right>" from:
      - processed sample_id stem: eyepacs__fundus__10003_left__xxxxxx
      - raw filename stem: 10003_left
      - any string containing a similar token
    """
    if s is None:
        return ""
    s2 = str(s).strip()
    if not s2:
        return ""

    # If a path is provided, use stem (drops extension).
    try:
        s2 = Path(s2).stem
    except Exception:
        pass

    parts = [p for p in s2.split("__") if p]
    # Search tokens from the end (the patient eye token is usually near the end).
    for tok in reversed(parts if parts else [s2]):
        m = _EYE_TOKEN_RE.match(tok)
        if m:
            pid = m.group("pid")
            eye = _norm_eye(m.group("eye"))
            if pid and eye:
                return f"{pid}_{eye}"

    # Fallback: find a token inside the full string.
    m = re.search(r"(\d+)[_-](left|right|l\d*|r\d*)", s2, flags=re.IGNORECASE)
    if m:
        pid = m.group(1)
        eye = _norm_eye(m.group(2))
        if pid and eye:
            return f"{pid}_{eye}"
    return ""


def key_to_patient_id(key: str) -> str:
    k = (key or "").strip()
    if "_" not in k:
        return ""
    return k.split("_", 1)[0]


def key_to_eye(key: str) -> str:
    k = (key or "").strip()
    if "_" not in k:
        return ""
    return _norm_eye(k.split("_", 1)[1])


def _split_patients(
    patient_ids: List[str],
    *,
    seed: int,
    ratios: Dict[str, float],
) -> Dict[str, str]:
    """
    Split patient_ids into named groups by ratios; returns mapping patient_id -> split.
    """
    pids = [p for p in patient_ids if str(p).strip() != ""]
    if not pids:
        return {}

    rng = np.random.RandomState(int(seed))
    rng.shuffle(pids)

    names = ["train", "val_train", "calib", "test"]
    r = [float(ratios.get(n, 0.0)) for n in names]
    s = sum(r)
    if s <= 0:
        raise ValueError(f"Invalid split ratios (sum<=0): {ratios}")
    r = [x / s for x in r]

    n = len(pids)
    # Allocate by floor + remainder to keep sum exact.
    raw = [x * n for x in r]
    counts = [int(np.floor(x)) for x in raw]
    rem = n - sum(counts)
    remainders = sorted([(raw[i] - counts[i], i) for i in range(len(names))], reverse=True)
    for j in range(rem):
        counts[remainders[j % len(names)][1]] += 1

    # Ensure non-empty splits when possible (n>=4). Borrow from train if needed.
    if n >= len(names):
        for i in range(1, len(names)):  # keep train as donor
            if counts[i] == 0 and counts[0] > 1:
                counts[i] = 1
                counts[0] -= 1

    mapping: Dict[str, str] = {}
    start = 0
    for name, c in zip(names, counts):
        for pid in pids[start : start + c]:
            mapping[str(pid)] = name
        start += c
    return mapping


def split_patients_stratified_by_label(
    patient_ids: List[str],
    patient_labels: Dict[str, int],
    seed: int,
    ratios: Dict[str, float],
) -> Dict[str, List[str]]:
    """
    Stratified split of patient_ids into multiple splits using patient-level labels.

    patient_ids: list of unique patient_id strings
    patient_labels: mapping patient_id -> int label (e.g., worst DR grade of that patient)
    ratios: e.g., {"train":0.7, "val_train":0.1, "calib":0.1, "test":0.1}

    Returns: dict split_name -> list of patient_ids
    """
    import math

    rng = np.random.RandomState(int(seed))

    # 1) collect patients per label
    label_to_pids: Dict[int, List[str]] = {}
    for pid in patient_ids:
        lab = int(patient_labels.get(pid, -1))
        label_to_pids.setdefault(lab, []).append(pid)

    splits: Dict[str, List[str]] = {k: [] for k in ratios.keys()}

    # 2) for each label, stratify patient_ids into splits according to ratios
    for lab, pids in label_to_pids.items():
        if lab < 0:
            # unknown label, just shuffle and assign later
            pids = list(pids)
            rng.shuffle(pids)
            # assign all to train for simplicity
            if "train" not in splits:
                raise ValueError("ratios must contain split 'train'")
            splits["train"].extend(pids)
            continue

        pids = list(pids)
        rng.shuffle(pids)
        n = len(pids)
        if n == 0:
            continue

        # compute counts per split for this label
        # base = floor(ratio * n), leftover 按小数部分排序分配
        keys = list(ratios.keys())
        base_counts = {k: int(math.floor(float(ratios[k]) * n)) for k in keys}
        used = sum(base_counts.values())
        leftover = n - used
        if leftover > 0:
            # 分配剩余：按 ratios 从大到小分配，避免都挤到 train
            keys_sorted = sorted(keys, key=lambda k: float(ratios[k]), reverse=True)
            for k in keys_sorted:
                if leftover <= 0:
                    break
                base_counts[k] += 1
                leftover -= 1

        # 根据 base_counts 切片
        idx = 0
        for k in keys:
            c = int(base_counts.get(k, 0))
            if c <= 0:
                continue
            splits[k].extend(pids[idx : idx + c])
            idx += c

    # 3) 可选：保证每个 split 至少有一点 patient，如果有 split 为空，从 train 借一点
    keys = list(ratios.keys())
    for k in keys:
        if len(splits[k]) == 0 and len(splits.get("train", [])) > 1:
            # 从 train 随机借一个 patient 过来
            pid = splits["train"].pop()
            splits[k].append(pid)

    return splits


def build_master_split_csv(
    *,
    grade_manifest_csv: Path,
    out_csv: Path,
    seed: int,
    ratios: Dict[str, float],
) -> Path:
    """
    Global patient-level split for EyePACS, based on grade_eyepacs.csv (processed image paths).
    Output columns: image_path,label,key,patient_id,eye,split
    """
    if not grade_manifest_csv.exists():
        raise FileNotFoundError(f"Grade manifest not found: {grade_manifest_csv}")

    df = pd.read_csv(grade_manifest_csv)
    if "image_path" not in df.columns:
        raise ValueError(f"{grade_manifest_csv} missing 'image_path' column. Columns={list(df.columns)}")

    if "label" not in df.columns:
        df["label"] = ""

    df["key"] = df["image_path"].astype(str).apply(infer_eyepacs_key_from_any_id)
    df["patient_id"] = df["key"].astype(str).apply(key_to_patient_id)
    df["eye"] = df["key"].astype(str).apply(key_to_eye)

    bad = df[df["patient_id"] == ""]
    if len(bad) > 0:
        print(f"[master_split] WARNING: {len(bad)} rows cannot infer patient_id; they will be dropped from split.")
        df = df[df["patient_id"] != ""].copy()

    patient_ids = sorted([p for p in df["patient_id"].unique().tolist() if str(p).strip() != ""])

    # ==== New: build patient-level labels for stratification ====
    #   使用该 patient 下所有图像的 "label" 列的最大值作为患者级 DR label
    #   假设 label 列已经是 0-4 的整数；如果有缺失/空字符串，先填成 -1 再处理
    df_labels = df[["patient_id", "label"]].copy()

    # 确保 label 是整数，无法转换的置为 -1
    def _safe_label(x):
        try:
            return int(x)
        except Exception:
            return -1

    df_labels["label"] = df_labels["label"].apply(_safe_label)

    patient_labels: Dict[str, int] = {}
    for pid, grp in df_labels.groupby("patient_id"):
        # worst-grade strategy: 该 patient 所有图像中 label 的最大值
        lab = int(grp["label"].max())
        patient_labels[str(pid)] = lab

    # ==== Replace _split_patients with stratified by label ====
    split_to_pids = split_patients_stratified_by_label(
        patient_ids=patient_ids,
        patient_labels=patient_labels,
        seed=int(seed),
        ratios=ratios,
    )

    pid_to_split: Dict[str, str] = {}
    for split_name, pids in split_to_pids.items():
        for pid in pids:
            pid_to_split[str(pid)] = str(split_name)

    df["split"] = df["patient_id"].map(pid_to_split).fillna("")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out = df[["image_path", "label", "key", "patient_id", "eye", "split"]].copy()
    df_out.to_csv(out_csv, index=False, encoding="utf-8")
    split_counts = {k: len(v) for k, v in split_to_pids.items()}
    print(f"[master_split] patients={len(patient_ids)} splits={split_counts}")
    print(f"[master_split] wrote: {out_csv} (patients={len(patient_ids)} images={len(df_out)})")
    return out_csv


def build_clean_eyeq_manifest_csv(
    *,
    master_split_csv: Path,
    eyeq_manifest_csv: Path,
    out_csv: Path,
) -> Path:
    """
    De-leak EyeQ (EyePACS subset):
      EyeQ_Train = EyeQ intersect Master_Train
    Output is intended for quality-head training only (no images from val/calib/test).
    """
    if not master_split_csv.exists():
        raise FileNotFoundError(f"master_split.csv not found: {master_split_csv}")
    if not eyeq_manifest_csv.exists():
        raise FileNotFoundError(f"EyeQ manifest not found: {eyeq_manifest_csv}")

    ms = pd.read_csv(master_split_csv)
    eq = pd.read_csv(eyeq_manifest_csv)

    if "split" not in ms.columns:
        raise ValueError(f"{master_split_csv} missing 'split' column. Columns={list(ms.columns)}")

    if "key" not in ms.columns:
        if "image_path" not in ms.columns:
            raise ValueError(f"{master_split_csv} missing 'key' and 'image_path' columns.")
        ms["key"] = ms["image_path"].astype(str).apply(infer_eyepacs_key_from_any_id)

    if "key" not in eq.columns:
        if "image_path" not in eq.columns:
            raise ValueError(f"{eyeq_manifest_csv} missing 'key' and 'image_path' columns.")
        eq["key"] = eq["image_path"].astype(str).apply(infer_eyepacs_key_from_any_id)

    key_to_split = dict(zip(ms["key"].astype(str), ms["split"].astype(str)))
    eq["master_split"] = eq["key"].astype(str).map(key_to_split)

    before = len(eq)
    eq_clean = eq[eq["master_split"].astype(str) == "train"].copy()
    after = len(eq_clean)
    dropped = before - after

    # Also drop rows that cannot be matched to EyePACS master split.
    unmatched = int(eq["master_split"].isna().sum())
    print(f"[clean_eyeq] matched={before - unmatched}/{before} unmatched={unmatched} dropped_non_train={dropped - unmatched}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    eq_clean.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[clean_eyeq] wrote: {out_csv} rows={after}")
    return out_csv


def build_rmf_features_processed_csv(
    *,
    master_split_csv: Path,
    rmf_csv: Path,
    out_csv: Path,
) -> Path:
    """
    Leak-proof RMF feature processing:
      - Left-join RMF onto EyePACS master split
      - Imputer fitted on Master_Train, then transform All
      - Scaler fitted on Master_Train, then transform All
      - Add rmf_is_missing indicator (1=missing RMF row / all-NaN features)
    """
    if not master_split_csv.exists():
        raise FileNotFoundError(f"master_split.csv not found: {master_split_csv}")
    if not rmf_csv.exists():
        raise FileNotFoundError(f"RMF csv not found: {rmf_csv}")

    ms = pd.read_csv(master_split_csv)
    rmf = pd.read_csv(rmf_csv)

    if "key" not in ms.columns:
        if "image_path" not in ms.columns:
            raise ValueError(f"{master_split_csv} missing 'key' and 'image_path' columns.")
        ms["key"] = ms["image_path"].astype(str).apply(infer_eyepacs_key_from_any_id)

    # Infer key from RMF table
    if "key" in rmf.columns:
        rmf["key"] = rmf["key"].astype(str).apply(infer_eyepacs_key_from_any_id)
    else:
        id_src = None
        for c in ["raw_path", "sample_id", "staged_name", "image", "image_name", "filename", "file", "name"]:
            if c in rmf.columns:
                id_src = c
                break
        if id_src is None:
            raise ValueError(f"{rmf_csv} cannot infer join key. Columns={list(rmf.columns)[:30]}")
        rmf["key"] = rmf[id_src].astype(str).apply(infer_eyepacs_key_from_any_id)

    rmf = rmf[rmf["key"].astype(str) != ""].copy()
    rmf = rmf.drop_duplicates(subset=["key"], keep="first")

    # Select numeric feature columns (exclude IDs / metadata)
    ignore_cols = {
        "key",
        "dataset",
        "group",
        "raw_path",
        "proc_path",
        "image_path",
        "mask_path",
        "sample_id",
        "staged_name",
        "paired_sample_id",
        "status",
        "aligned",
        "split",
    }
    cand_cols = [c for c in rmf.columns if c not in ignore_cols]
    if not cand_cols:
        raise ValueError(f"{rmf_csv} has no candidate RMF feature columns after excluding ids/meta.")

    # Keep existing rmf_is_missing if provided by RMF table; otherwise we'll generate it after merge.
    keep_rmf_is_missing = "rmf_is_missing" in cand_cols
    feat_cand_cols = [c for c in cand_cols if c != "rmf_is_missing"]

    rmf_num = rmf[feat_cand_cols].apply(pd.to_numeric, errors="coerce")
    feature_cols = [c for c in feat_cand_cols if rmf_num[c].notna().any()]
    if not feature_cols:
        raise ValueError(f"{rmf_csv} has no numeric RMF feature columns.")

    rmf_feat = pd.concat([rmf[["key"]].reset_index(drop=True), rmf_num[feature_cols].reset_index(drop=True)], axis=1)
    if keep_rmf_is_missing:
        rmf_feat["rmf_is_missing"] = pd.to_numeric(rmf["rmf_is_missing"], errors="coerce")

    merged = ms.merge(rmf_feat, on="key", how="left", indicator=True)
    # Whether this sample has a corresponding RMF row in the source table (strict multimodal intersection key).
    merged["rmf_has_row"] = (merged["_merge"] == "both").astype(int)
    merged = merged.drop(columns=["_merge"])

    if "split" not in merged.columns:
        raise ValueError(f"{master_split_csv} missing 'split' column; cannot fit leak-proof transforms.")

    # ---- Meta cols vs feature cols (post-merge) ----
    meta_cols = {
        "image_path",
        "label",
        "grade",
        "dr_grade",
        "icdr",
        "key",
        "patient_id",
        "eye",
        "split",
        "rmf_has_row",
        "rmf_path",
        "rmf_vec",
        "lesion_path",
        "lesion_vec",
    }
    feat_cols = [c for c in merged.columns if (c not in meta_cols and c != "rmf_is_missing")]
    if not feat_cols:
        raise ValueError(f"{rmf_csv} has no usable numeric RMF feature columns after merge.")

    # ---- rmf_is_missing indicator (before imputation/standardization) ----
    if "rmf_is_missing" in merged.columns:
        # Use existing; fill missing (no RMF row) as missing=1 and binarize.
        x = pd.to_numeric(merged["rmf_is_missing"], errors="coerce")
        merged["rmf_is_missing"] = (x.fillna(1.0) > 0).astype(int)
    else:
        raw_vals = merged[feat_cols].to_numpy(dtype=float)
        # Non-finite treated as missing
        mask_missing_any = (~np.isfinite(raw_vals) | np.isnan(raw_vals)).any(axis=1)
        merged["rmf_is_missing"] = mask_missing_any.astype(int)

    tr_mask = merged["split"].astype(str) == "train"
    if int(tr_mask.sum()) == 0:
        raise ValueError("No 'train' rows found in master_split; cannot fit RMF imputer/scaler.")

    # --- Fit statistics on TRAIN only: medians -> fill -> means/stds ---
    X_train = merged.loc[tr_mask, feat_cols].to_numpy(dtype=float)
    X_train[~np.isfinite(X_train)] = np.nan
    with np.errstate(all="ignore"):
        medians = np.nanmedian(X_train, axis=0)
    medians[~np.isfinite(medians)] = 0.0

    X_train_filled = np.where(np.isnan(X_train), medians, X_train)
    means = X_train_filled.mean(axis=0)
    stds = X_train_filled.std(axis=0)
    means[~np.isfinite(means)] = 0.0
    bad_std = (~np.isfinite(stds)) | (stds <= 0)
    stds[bad_std] = 1.0

    # --- Apply to ALL: inf->nan -> median fill -> z-score ---
    X_all = merged[feat_cols].to_numpy(dtype=float)
    X_all[~np.isfinite(X_all)] = np.nan
    X_all_filled = np.where(np.isnan(X_all), medians, X_all)
    X_all_scaled = (X_all_filled - means) / stds
    X_all_scaled[~np.isfinite(X_all_scaled)] = 0.0
    merged.loc[:, feat_cols] = X_all_scaled.astype(np.float32)

    # Output columns: meta + rmf_is_missing + standardized features
    out_cols = ["image_path", "label", "key", "patient_id", "eye", "split", "rmf_has_row", "rmf_is_missing"] + feat_cols
    out_cols = [c for c in out_cols if c in merged.columns]
    out_df = merged[out_cols].copy()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[rmf] wrote: {out_csv} rows={len(out_df)} features={len(feat_cols)}")
    return out_csv


def _norm_col(c: str) -> str:
    c = c.strip().lower().replace(" ", "_")
    c = re.sub(r"_+", "_", c)
    return c


def _col_map(df: pd.DataFrame) -> dict:
    """Map normalized column names -> original names."""
    return {_norm_col(c): c for c in df.columns}


def _numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def find_tabular_candidates(root: Path, keywords: Iterable[str]) -> List[Path]:
    """Find CSV / Excel files under root, scored by simple keyword matching."""
    kws = tuple(k.lower() for k in keywords)
    cands: List[Tuple[int, Path]] = []
    if not root.exists():
        return []
    for p in root.rglob("*"):
        if p.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
            continue
        name = p.name.lower()
        score = sum(1 for k in kws if k in name)
        cands.append((score, p))
    cands.sort(key=lambda x: (-x[0], str(x[1])))
    return [p for _, p in cands]


def load_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(path, engine="openpyxl")
        except Exception:
            return pd.read_excel(path)
    raise ValueError(f"Unsupported table file: {path}")


# ---------------------------
# Offline preprocessed images
# ---------------------------
def load_offline_images(project_root: Path) -> pd.DataFrame:
    """Load offline processed image index created by preprocess_offline.py."""
    offline = project_root / "data" / "meta" / "manifests" / "manifest_offline_images.csv"
    if not offline.exists():
        raise FileNotFoundError(
            f"Missing offline manifest: {offline}\n"
            f"Run preprocess_offline.py first."
        )
    df = pd.read_csv(offline)

    # only keep status == ok if present
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    else:
        df = df.copy()

    # resolve image path
    if "proc_path" in df.columns:
        df["image_path"] = df["proc_path"].map(lambda s: to_rel(project_root, s))
    elif "image_path" in df.columns:
        df["image_path"] = df["image_path"].map(lambda s: to_rel(project_root, s))
    else:
        raise ValueError(f"offline manifest missing proc_path/image_path columns: {list(df.columns)}")

    # derive key/sample_id if absent
    if "key" not in df.columns:
        if "raw_path" in df.columns:
            df["key"] = df["raw_path"].astype(str).map(lambda x: sanitize_stem(Path(x).stem))
        else:
            df["key"] = df["image_path"].astype(str).map(lambda x: sanitize_stem(Path(x).stem))
    if "sample_id" not in df.columns:
        df["sample_id"] = df["key"]

    if "dataset" in df.columns:
        df["dataset"] = df["dataset"].astype(str)
    else:
        df["dataset"] = ""

    if "group" in df.columns:
        df["group"] = df["group"].astype(str)
    else:
        df["group"] = ""

    return df


# ---------------------------
# Grade manifest: EyePACS
# ---------------------------
def build_grade_eyepacs(
    project_root: Path,
    df_img: pd.DataFrame,
    out_csv: Path,
    labels_file: Optional[Path] = None,
) -> int:
    """Build grade_eyepacs.csv manifest from EyePACS DR labels."""
    if labels_file is not None:
        label_path = labels_file
        if not label_path.exists():
            raise FileNotFoundError(f"EyePACS labels file not found: {label_path}")
    else:
        raw_root = project_root / "data" / "raw" / "eyepacs" / "diabetic-retinopathy-detection"
        cands = find_tabular_candidates(raw_root, keywords=("trainlabels", "label", "train"))
        if not cands:
            raise FileNotFoundError(
                f"No labels found under {raw_root}\n"
                f"Tip: put trainLabels.csv there, or pass --eyepacs_labels <path_to_csv>."
            )
        label_path = cands[0]

    print(f"[grade] Using labels: {label_path}")
    lab = load_table(label_path)

    # image id column
    img_col = None
    for c in ["image", "image_id", "img", "id_code", "filename"]:
        if c in lab.columns:
            img_col = c
            break
    if img_col is None:
        img_col = lab.columns[0]

    # label column
    label_col = None
    for c in ["level", "diagnosis", "label", "grade"]:
        if c in lab.columns:
            label_col = c
            break
    if label_col is None:
        raise ValueError(f"EyePACS label column not found in {label_path}. cols={list(lab.columns)}")

    lab["key"] = lab[img_col].astype(str).map(lambda x: sanitize_stem(Path(x).stem))
    lab["label"] = _numeric_series(lab[label_col]).astype(int)

    img_eyepacs = df_img[df_img["dataset"].astype(str).str.lower() == "eyepacs"].copy()
    if img_eyepacs.empty:
        print("[grade] WARNING: no EyePACS rows found in offline manifest; writing empty grade_eyepacs.csv")
        ensure_dir(out_csv.parent)
        pd.DataFrame(columns=["image_path", "label"]).to_csv(out_csv, index=False, encoding="utf-8")
        return 0

    if "group" in img_eyepacs.columns:
        pref = img_eyepacs[img_eyepacs["group"].astype(str).str.contains("fundus", case=False, na=False)].copy()
        # Backward compatibility: if group naming differs, fall back to all EyePACS rows.
        if not pref.empty:
            img_eyepacs = pref

    img_eyepacs = img_eyepacs[["key", "image_path"]].copy()

    merged = img_eyepacs.merge(lab[["key", "label"]], on="key", how="inner")

    ensure_dir(out_csv.parent)
    merged[["image_path", "label"]].to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[grade] wrote: {out_csv} rows={len(merged)}")
    return len(merged)


# ---------------------------
# DeepDRiD quality label selection
# ---------------------------
def pick_best_deepdrid_quality_table(raw_root: Path) -> Path:
    """Heuristic search for the best DeepDRiD quality labels file."""
    cands = find_tabular_candidates(
        raw_root,
        keywords=("challenge", "quality", "grad", "label", "labels", "overall", "artifact", "clarity", "field"),
    )
    if not cands:
        raise FileNotFoundError(f"No tabular files (.csv/.xlsx) found under {raw_root}")

    best: Tuple[int, Path] | None = None

    for p in cands[:300]:
        name = p.name.lower()
        bonus = 0
        if "labels" in name or name.endswith("_labels.csv") or name.endswith("_labels.xlsx"):
            bonus += 20000
        if "upload" in name or "template" in name or "submission" in name:
            bonus -= 20000

        try:
            df = load_table(p)
        except Exception:
            continue

        cm = _col_map(df)

        # direct label columns
        direct_cols = []
        for k in ["overall", "overall_quality", "quality", "gradable", "gradability"]:
            if k in cm:
                direct_cols.append(cm[k])

        direct_score = -1
        if direct_cols:
            for dc in direct_cols:
                s = _numeric_series(df[dc])
                nn = int(s.notna().sum())
                direct_score = max(direct_score, nn)

        # component columns
        comp_keys = ["artifact", "clarity", "field", "field_definition", "fielddef", "fielddefinition"]
        comps = [cm[k] for k in comp_keys if k in cm]
        comp_score = -1
        if len(comps) >= 2:
            tmp = df[comps].apply(_numeric_series)
            mean = tmp.mean(axis=1, skipna=True)
            comp_score = int(mean.notna().sum())

        score = 0
        if direct_score > 0:
            score = 200000 + direct_score
        elif comp_score > 0:
            score = 100000 + comp_score

        score += bonus

        if score > 0:
            if best is None or score > best[0]:
                best = (score, p)

    if best is None:
        top = "\n".join([str(x) for x in cands[:20]])
        raise FileNotFoundError(
            "Could not find DeepDRiD quality label file that contains non-empty labels.\n"
            "Please pass --deepdrid_quality_labels <path_to_(Challenge*_labels.xlsx or csv)>.\n"
            f"Candidates:\n{top}"
        )

    return best[1]


# ---------------------------
# Quality manifest: DeepDRiD
# ---------------------------
def build_qual_deepdrid(
    project_root: Path,
    df_img: pd.DataFrame,
    out_csv: Path,
    labels_file: Optional[Path] = None,
) -> Tuple[int, int]:
    """Build qual_deepdrid.csv manifest from DeepDRiD quality labels."""
    if labels_file is not None:
        quality_path = labels_file
        if not quality_path.exists():
            raise FileNotFoundError(f"DeepDRiD quality labels file not found: {quality_path}")
    else:
        raw_root = project_root / "data" / "raw" / "deepdrid"
        quality_path = pick_best_deepdrid_quality_table(raw_root)

    print(f"[qual] Using labels: {quality_path}")
    qd = load_table(quality_path)

    # ID column
    qid_col = None
    for c in ["image", "image_id", "img_id", "img", "filename", "file", "name", "id_code", "image_name"]:
        if c in qd.columns:
            qid_col = c
            break
    if qid_col is None:
        for c in qd.columns:
            lc = _norm_col(c)
            if "image" in lc or "file" in lc or "name" in lc:
                qid_col = c
                break
    if qid_col is None:
        qid_col = qd.columns[0]

    qd["key"] = qd[qid_col].astype(str).map(lambda x: sanitize_stem(Path(x).stem))
    cm = _col_map(qd)

    # direct label column
    direct_key_order = ["overall", "overall_quality", "quality", "gradable", "gradability"]
    direct_col = None
    for k in direct_key_order:
        if k in cm:
            direct_col = cm[k]
            break

    if direct_col is not None:
        vals = qd[direct_col]
        if vals.dtype == object:
            mapping = {"gradable": 1, "ungradable": 0, "yes": 1, "no": 0, "true": 1, "false": 0}
            qd["label"] = vals.astype(str).str.strip().str.lower().map(mapping)
        else:
            qd["label"] = _numeric_series(vals)

        before = len(qd)
        qd = qd[qd["label"].notna()].copy()
        dropped = before - len(qd)
        if dropped > 0:
            print(f"[qual] dropped NaN labels: {dropped}")

        qd["label"] = qd["label"].astype(int)

    else:
        # component-based label
        comp_keys = ["artifact", "clarity", "field", "field_definition", "fielddef", "fielddefinition"]
        comps = [cm[k] for k in comp_keys if k in cm]
        if len(comps) < 2:
            raise ValueError(f"DeepDRiD quality label columns not found in {quality_path}. cols={list(qd.columns)}")

        tmp = qd[comps].apply(_numeric_series)
        mean = tmp.mean(axis=1, skipna=True)

        qd["label_float"] = mean
        before = len(qd)
        qd = qd[qd["label_float"].notna()].copy()
        dropped = before - len(qd)
        if dropped > 0:
            print(f"[qual] dropped rows with all-NaN components: {dropped}")

        qd["label"] = np.round(qd["label_float"]).astype(int)
        qd.drop(columns=["label_float"], inplace=True)

    # join to processed DeepDRiD images
    img_deep = df_img[
        (df_img["dataset"] == "deepdrid") & (df_img["group"].str.contains("fundus", case=False, na=False))
    ][["key", "image_path"]].copy()

    merged = img_deep.merge(qd[["key", "label"]], on="key", how="inner")

    if len(merged) == 0:
        ex_img = img_deep["key"].head(5).tolist()
        ex_lab = qd["key"].head(5).tolist()
        raise RuntimeError(
            "[qual] merged result is empty. Image IDs in labels file do not match filenames.\n"
            f"Examples image keys: {ex_img}\nExamples label keys: {ex_lab}\n"
            "Fix: pass correct --deepdrid_quality_labels and ensure the ID column contains image filename stems."
        )

    ensure_dir(out_csv.parent)
    merged[["image_path", "label"]].to_csv(out_csv, index=False, encoding="utf-8")

    num_rows = len(merged)
    num_classes = int(merged["label"].max() + 1) if num_rows else 0
    uniq = sorted(merged["label"].unique().tolist())
    print(f"[qual] wrote: {out_csv} rows={num_rows} classes={num_classes} unique={uniq}")
    return num_rows, num_classes


# ---------------------------
# Quality manifest: EyeQ
# ---------------------------
def build_qual_eyeq(
    project_root: Path,
    df_img: pd.DataFrame,
    out_csv: Path,
    train_labels_file: Path,
    test_labels_file: Optional[Path] = None,
) -> Tuple[int, List[int]]:
    """
    Build quality manifest from EyeQ labels.

    Expected columns in label files (train/test):
      - image   : filename (e.g., "10009_left.jpeg") or any column whose name contains "image"/"file"
      - quality : integer quality score (EyeQ 3-class: 0, 1, 2), or any column whose normalized
                  name contains "quality" or "label".

    The produced manifest keeps both:
      - label   : integer class (same as quality)
      - quality : original integer score

    During training you can request binary collapse by passing
    `--num_classes_quality 2 --qual_collapse_to_binary` to train_roundtune_cpf.py.
    """
    if not train_labels_file.exists():
        raise FileNotFoundError(f"EyeQ train labels not found: {train_labels_file}")
    print(f"[qual_eyeq] Using train labels: {train_labels_file}")
    df_tr = load_table(train_labels_file)
    df_tr["_source"] = "train"

    if test_labels_file is not None:
        if not test_labels_file.exists():
            raise FileNotFoundError(f"EyeQ test labels not found: {test_labels_file}")
        print(f"[qual_eyeq] Using test labels: {test_labels_file}")
        df_te = load_table(test_labels_file)
        df_te["_source"] = "test"
        eyeq = pd.concat([df_tr, df_te], axis=0, ignore_index=True)
    else:
        eyeq = df_tr.copy()

    cm = _col_map(eyeq)

    # image filename column
    img_col = None
    for k in ["image", "img", "image_name", "filename", "file", "name", "id_code"]:
        if k in cm:
            img_col = cm[k]
            break
    if img_col is None:
        for c in eyeq.columns:
            lc = _norm_col(c)
            if "image" in lc or "file" in lc or "name" in lc:
                img_col = c
                break
    if img_col is None:
        raise ValueError(f"EyeQ labels missing image column. Columns={list(eyeq.columns)}")

    # quality label column
    q_col = None
    for k in ["quality", "quality_label", "qualityscore", "quality_score", "label", "grade"]:
        if k in cm:
            q_col = cm[k]
            break
    if q_col is None:
        # best-effort: take the second column as label if exists
        if len(eyeq.columns) >= 2:
            q_col = eyeq.columns[1]
        else:
            raise ValueError(f"EyeQ labels missing quality column. Columns={list(eyeq.columns)}")

    eyeq["key"] = eyeq[img_col].astype(str).map(lambda x: sanitize_stem(Path(x).stem))
    eyeq["quality"] = _numeric_series(eyeq[q_col])

    before = len(eyeq)
    eyeq = eyeq[eyeq["quality"].notna()].copy()
    dropped = before - len(eyeq)
    if dropped > 0:
        print(f"[qual_eyeq] dropped rows with NaN quality: {dropped}")

    eyeq["quality"] = eyeq["quality"].astype(int)

    # Report coverage against offline preprocessed images.
    off_keys = set(df_img["key"].astype(str).tolist()) if "key" in df_img.columns else set()
    uniq_total = int(eyeq["key"].nunique())
    matched_keys = set(eyeq["key"].unique().tolist()) & off_keys
    unmatched_n = uniq_total - len(matched_keys)

    n_tr = int((eyeq["_source"] == "train").sum()) if "_source" in eyeq.columns else 0
    n_te = int((eyeq["_source"] == "test").sum()) if "_source" in eyeq.columns else 0
    print(
        f"[qual_eyeq] labels rows: train={n_tr} test={n_te} total={len(eyeq)} unique_keys={uniq_total} "
        f"matched_unique={len(matched_keys)} unmatched_unique={unmatched_n}"
    )

    if unmatched_n > 0 and "_source" in eyeq.columns:
        # Save the dropped labels for debugging (usually missing offline preprocessing for those images).
        unmatched_csv = out_csv.with_name(out_csv.stem + "_unmatched.csv")
        um = eyeq[~eyeq["key"].isin(matched_keys)][[img_col, "key", "quality", "_source"]].copy()
        um = um.rename(columns={img_col: "image", "_source": "source"})
        ensure_dir(unmatched_csv.parent)
        um.to_csv(unmatched_csv, index=False, encoding="utf-8")
        print(
            f"[qual_eyeq] WARN: {len(um)} labels cannot be matched to offline images and were excluded. "
            f"Wrote: {unmatched_csv}"
        )

    # Join EyeQ labels to ANY offline image with matching key (e.g. EyeQ is a subset of EyePACS)
    merged = df_img.merge(eyeq[["key", "quality"]], on="key", how="inner")
    if merged.empty:
        ex_img = df_img["key"].head(5).tolist()
        ex_lab = eyeq["key"].head(5).tolist()
        raise RuntimeError(
            "[qual_eyeq] merged result is empty. Check that EyeQ label filenames "
            "match the filenames used in offline preprocessing.\n"
            f"Example offline keys: {ex_img}\nExample label keys: {ex_lab}"
        )

    merged["label"] = merged["quality"]

    ensure_dir(out_csv.parent)
    cols = ["image_path", "label", "quality"]
    extra_cols = [c for c in ["dataset", "split", "sample_id", "key"] if c in merged.columns]
    merged[cols + extra_cols].to_csv(out_csv, index=False, encoding="utf-8")

    uniq = sorted(merged["quality"].unique().tolist())
    print(f"[qual_eyeq] wrote: {out_csv} rows={len(merged)} unique_quality={uniq}")
    return len(merged), uniq


# ---------------------------
# Segmentation manifest: IDRiD + DDR (multilabel .npy)
# ---------------------------
def _infer_split_from_path(path: str) -> str:
    lp = str(path).lower()
    if "train" in lp or "training" in lp:
        return "train"
    if "valid" in lp or "val" in lp:
        return "val"
    if "test" in lp:
        return "test"
    return "train"


def build_seg_idrid_and_optional_ddr(
    project_root: Path,
    df_img: pd.DataFrame,
    out_idrid_csv: Path,
    out_ddr_csv: Path,
    img_size: int,
    seg_source: str = "both",
) -> None:
    """
    Build segmentation manifests for IDRiD and optionally DDR lesions.

    We assume multilabel masks are stored as:
      <project_root>/data/processed/cfp_<img_size>/multilabel_masks/{idrid,ddr}/<sample_id>.npy
    """
    masks_root = project_root / "data" / "processed" / f"cfp_{img_size}" / "multilabel_masks"

    def _write_empty(csv_path: Path, tag: str) -> None:
        ensure_dir(csv_path.parent)
        pd.DataFrame(columns=["image_path", "mask_path", "dataset", "split"]).to_csv(csv_path, index=False, encoding="utf-8")
        print(f"[seg][{tag}] wrote: {csv_path} rows=0 (empty)")

    # --- IDRiD ---
    img_idrid = df_img[df_img["dataset"].str.lower() == "idrid"].copy()
    if not img_idrid.empty and "group" in img_idrid.columns:
        # Prefer the segmentation group produced by preprocess_offline.py
        pref = img_idrid[img_idrid["group"].astype(str).str.lower() == "seg"].copy()
        if pref.empty:
            # Backward compatibility: older offline manifests might tag IDRiD fundus images differently.
            pref = img_idrid[img_idrid["group"].astype(str).str.contains("fundus", case=False, na=False)].copy()
        img_idrid = pref

    if not img_idrid.empty:
        img_idrid["mask_path"] = img_idrid["sample_id"].astype(str).map(
            lambda sid: to_rel(project_root, masks_root / "idrid" / f"{sid}.npy")
        )

        if "split" in img_idrid.columns:
            img_idrid["split"] = img_idrid["split"].fillna("train")
        else:
            raw_col = "raw_path" if "raw_path" in img_idrid.columns else None
            if raw_col:
                img_idrid["split"] = img_idrid[raw_col].map(_infer_split_from_path)
            else:
                img_idrid["split"] = "train"

        ensure_dir(out_idrid_csv.parent)
        img_idrid[["image_path", "mask_path", "dataset", "split"]].to_csv(
            out_idrid_csv, index=False, encoding="utf-8"
        )
        print(f"[seg][IDRiD] wrote: {out_idrid_csv} rows={len(img_idrid)}")
    else:
        print("[seg][IDRiD] no samples found in offline manifest.")
        _write_empty(out_idrid_csv, "IDRiD")

    # --- DDR lesions ---
    if seg_source in ("both", "ddr"):
        img_ddr = df_img[df_img["dataset"].str.lower() == "ddr"].copy()
        if not img_ddr.empty and "group" in img_ddr.columns:
            # Prefer the lesion segmentation group produced by preprocess_offline.py
            pref = img_ddr[img_ddr["group"].astype(str).str.lower() == "lesion_seg"].copy()
            if pref.empty:
                # Backward compatibility: older offline manifests might tag DDR images differently.
                pref = img_ddr[img_ddr["group"].astype(str).str.contains("fundus", case=False, na=False)].copy()
            img_ddr = pref

        if not img_ddr.empty:
            img_ddr["mask_path"] = img_ddr["sample_id"].astype(str).map(
                lambda sid: to_rel(project_root, masks_root / "ddr" / f"{sid}.npy")
            )

            if "split" in img_ddr.columns:
                img_ddr["split"] = img_ddr["split"].fillna("train")
            else:
                raw_col = "raw_path" if "raw_path" in img_ddr.columns else None
                if raw_col:
                    img_ddr["split"] = img_ddr[raw_col].map(_infer_split_from_path)
                else:
                    img_ddr["split"] = "train"

            ensure_dir(out_ddr_csv.parent)
            img_ddr[["image_path", "mask_path", "dataset", "split"]].to_csv(
                out_ddr_csv, index=False, encoding="utf-8"
            )
            print(f"[seg][DDR] wrote: {out_ddr_csv} rows={len(img_ddr)}")
        else:
            print("[seg][DDR] no samples found in offline manifest.")
            _write_empty(out_ddr_csv, "DDR")
    else:
        print("[seg][DDR] skipped by seg_source=", seg_source)


# ---------------------------
# Main
# ---------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, required=True)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)

    # Global EyePACS patient split (leak-proof split used across tasks)
    ap.add_argument(
        "--skip_master_split",
        action="store_true",
        help="Skip building global EyePACS patient split (master_split.csv).",
    )
    ap.add_argument(
        "--master_split_out",
        type=str,
        default=r"data\manifests\master_split.csv",
        help=r"Output path for master_split.csv (relative to project_root by default).",
    )
    ap.add_argument("--split_train", type=float, default=0.7)
    ap.add_argument("--split_val_train", type=float, default=0.1)
    ap.add_argument("--split_calib", type=float, default=0.1)
    ap.add_argument("--split_test", type=float, default=0.1)

    # backward compatible switch: if --skip_ddr set and seg_source==both, we treat as idrid only.
    ap.add_argument(
        "--skip_ddr",
        action="store_true",
        help="Backward-compatible flag; prefer --seg_source.",
    )
    ap.add_argument(
        "--seg_source",
        type=str,
        default="both",
        choices=["both", "ddr", "idrid"],
        help="Segmentation source: build DDR/IDRiD/both lesion manifests.",
    )

    ap.add_argument("--skip_grade", action="store_true")
    ap.add_argument("--skip_qual", action="store_true")
    ap.add_argument("--skip_seg", action="store_true")
    ap.add_argument("--force", action="store_true")

    # EyeQ de-leaking (optional): filter EyeQ to Master_Train only
    ap.add_argument(
        "--build_clean_eyeq",
        action="store_true",
        help="Build clean_qual_manifest.csv = EyeQ intersect Master_Train (requires master_split + qual_eyeq.csv).",
    )
    ap.add_argument(
        "--clean_eyeq_out",
        type=str,
        default=r"data\manifests\clean_qual_manifest.csv",
        help=r"Output path for clean EyeQ manifest (relative to project_root by default).",
    )

    # RMF processing (optional): leak-proof imputer/scaler fitted on Master_Train
    ap.add_argument(
        "--rmf_csv",
        type=str,
        default="",
        help=r"Path to AutoMorphalyzer rmf.csv (e.g., data\processed\rmf_automorphalyzer\runs\<run>\rmf.csv).",
    )
    ap.add_argument(
        "--rmf_out",
        type=str,
        default=r"data\manifests\rmf_features_processed.csv",
        help=r"Output path for rmf_features_processed.csv (relative to project_root by default).",
    )

    ap.add_argument(
        "--eyepacs_labels",
        type=str,
        default="",
        help="Optional EyePACS labels file path",
    )
    ap.add_argument(
        "--deepdrid_quality_labels",
        type=str,
        default="",
        help="Optional DeepDRiD quality labels file path (.xlsx/.csv)",
    )
    ap.add_argument(
        "--eyeq_train_labels",
        type=str,
        default="",
        help="Optional EyeQ train labels file (Label_EyeQ_train.csv)",
    )
    ap.add_argument(
        "--eyeq_test_labels",
        type=str,
        default="",
        help="Optional EyeQ test labels file (Label_EyeQ_test.csv)",
    )

    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    if args.skip_ddr and args.seg_source == "both":
        args.seg_source = "idrid"

    print("[info] meta/manifests = offline image index; data/manifests = task manifests with labels/masks.")
    df_img = load_offline_images(project_root)

    mani_dir = project_root / "data" / "manifests"
    ensure_dir(mani_dir)

    grade_manifest = mani_dir / "grade_eyepacs.csv"
    qual_manifest = mani_dir / "qual_deepdrid.csv"
    qual_eyeq_manifest = mani_dir / "qual_eyeq.csv"
    master_split_csv = Path(args.master_split_out)
    if not master_split_csv.is_absolute():
        master_split_csv = project_root / master_split_csv
    clean_eyeq_csv = Path(args.clean_eyeq_out)
    if not clean_eyeq_csv.is_absolute():
        clean_eyeq_csv = project_root / clean_eyeq_csv
    rmf_in_csv = Path(args.rmf_csv) if str(args.rmf_csv).strip() else None
    if rmf_in_csv is not None and (not rmf_in_csv.is_absolute()):
        rmf_in_csv = project_root / rmf_in_csv
    rmf_out_csv = Path(args.rmf_out)
    if not rmf_out_csv.is_absolute():
        rmf_out_csv = project_root / rmf_out_csv
    seg_manifest = mani_dir / "seg_idrid.csv"
    seg_extra_manifest = mani_dir / "seg_ddr_lesions.csv"

    # grade
    if args.skip_grade:
        print("[grade] skipped")
    else:
        if args.force or not grade_manifest.exists():
            labels_path = Path(args.eyepacs_labels) if args.eyepacs_labels.strip() else None
            build_grade_eyepacs(project_root, df_img, grade_manifest, labels_file=labels_path)
        else:
            print(f"[grade] exists: {grade_manifest}")

    # Global patient split for EyePACS (master_split.csv)
    if args.skip_master_split:
        print("[master_split] skipped")
    else:
        if not grade_manifest.exists():
            print("[master_split] skipped (missing grade_eyepacs.csv)")
        else:
            ratios = {
                "train": float(args.split_train),
                "val_train": float(args.split_val_train),
                "calib": float(args.split_calib),
                "test": float(args.split_test),
            }
            build_master_split_csv(
                grade_manifest_csv=grade_manifest,
                out_csv=master_split_csv,
                seed=int(args.seed),
                ratios=ratios,
            )

    # qual (DeepDRiD)
    num_classes_quality: Optional[int] = None
    if args.skip_qual:
        print("[qual] skipped")
    else:
        if args.force or not qual_manifest.exists():
            deep_labels = Path(args.deepdrid_quality_labels) if args.deepdrid_quality_labels.strip() else None
            _, num_classes_quality = build_qual_deepdrid(project_root, df_img, qual_manifest, labels_file=deep_labels)
        else:
            print(f"[qual] exists: {qual_manifest}")
            dfq = pd.read_csv(qual_manifest)
            num_classes_quality = int(dfq["label"].max() + 1) if len(dfq) else 2

    # EyeQ quality manifest (optional, built only if label paths are provided)
    built_eyeq = False
    if args.eyeq_train_labels.strip():
        eyeq_train = Path(args.eyeq_train_labels)
        if not eyeq_train.is_absolute():
            eyeq_train = project_root / eyeq_train

        eyeq_test: Optional[Path]
        if args.eyeq_test_labels.strip():
            eyeq_test = Path(args.eyeq_test_labels)
            if not eyeq_test.is_absolute():
                eyeq_test = project_root / eyeq_test
        else:
            # Auto-detect EyeQ test labels next to the train labels (common layout).
            eyeq_test = None
            parent = eyeq_train.parent
            for name in ("Label_EyeQ_test.csv", "Label_EyeQ_test.xlsx", "Label_EyeQ_test.xls"):
                cand = parent / name
                if cand.exists():
                    eyeq_test = cand
                    break
            if eyeq_test is None:
                # Heuristic: replace "train" with "test" in filename.
                n = eyeq_train.name
                for a, b in (("train", "test"), ("Train", "Test"), ("TRAIN", "TEST")):
                    if a in n:
                        cand = parent / n.replace(a, b)
                        if cand.exists():
                            eyeq_test = cand
                            break
            if eyeq_test is not None:
                print(f"[qual_eyeq] Auto-detected test labels: {eyeq_test}")

        n_eyeq, uniq_eyeq = build_qual_eyeq(project_root, df_img, qual_eyeq_manifest, eyeq_train, eyeq_test)
        built_eyeq = True
        print(f"[qual_eyeq] hint: {n_eyeq} samples, unique quality labels={uniq_eyeq}")

    # De-leak EyeQ against the global master split:
    # - explicit: --build_clean_eyeq
    # - automatic: if EyeQ labels are provided (built_eyeq), we also build the clean manifest by default
    if bool(getattr(args, "build_clean_eyeq", False)) or built_eyeq:
        if not master_split_csv.exists():
            print("[clean_eyeq] skipped (missing master_split.csv)")
        elif not qual_eyeq_manifest.exists():
            print("[clean_eyeq] skipped (missing qual_eyeq.csv)")
        else:
            build_clean_eyeq_manifest_csv(
                master_split_csv=master_split_csv,
                eyeq_manifest_csv=qual_eyeq_manifest,
                out_csv=clean_eyeq_csv,
            )

    # Leak-proof RMF preprocessing (optional)
    if rmf_in_csv is not None:
        if not master_split_csv.exists():
            print("[rmf] skipped (missing master_split.csv)")
        elif not rmf_in_csv.exists():
            print(f"[rmf] skipped (missing rmf_csv: {rmf_in_csv})")
        else:
            build_rmf_features_processed_csv(
                master_split_csv=master_split_csv,
                rmf_csv=rmf_in_csv,
                out_csv=rmf_out_csv,
            )

    # seg
    if args.skip_seg:
        print("[seg] skipped")
    else:
        need = args.force or (not seg_manifest.exists()) or (
            (args.seg_source in ("both", "ddr")) and (not seg_extra_manifest.exists())
        )
        if need:
            build_seg_idrid_and_optional_ddr(
                project_root=project_root,
                df_img=df_img,
                out_idrid_csv=seg_manifest,
                out_ddr_csv=seg_extra_manifest,
                img_size=args.img_size,
                seg_source=args.seg_source,
            )
        else:
            print(f"[seg] exists: {seg_manifest}")
            if args.seg_source in ("both", "ddr"):
                print(f"[seg] exists: {seg_extra_manifest}")

    print("\n[OK] DONE. Outputs:")
    if not args.skip_master_split and master_split_csv.exists():
        print(" -", master_split_csv)
    if not args.skip_grade:
        print(" -", grade_manifest)
    if not args.skip_qual:
        print(" -", qual_manifest)
    if built_eyeq and qual_eyeq_manifest.exists():
        print(" -", qual_eyeq_manifest)
    if (bool(getattr(args, "build_clean_eyeq", False)) or built_eyeq) and clean_eyeq_csv.exists():
        print(" -", clean_eyeq_csv)
    if rmf_in_csv is not None and rmf_out_csv.exists():
        print(" -", rmf_out_csv)
    if not args.skip_seg:
        print(" -", seg_manifest)
        if args.seg_source in ("both", "ddr"):
            print(" -", seg_extra_manifest)

    if num_classes_quality is not None:
        print("\n[hint] quality num_classes for training:", num_classes_quality)
    masks_root = project_root / "data" / "processed" / f"cfp_{args.img_size}" / "multilabel_masks"
    print("[hint] multilabel masks root:", masks_root)


if __name__ == "__main__":
    main()
