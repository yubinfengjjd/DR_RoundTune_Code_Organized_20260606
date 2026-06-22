# save as: src/extract_rmf_automorphalyzer.py
import argparse
import os
import re
import shutil
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from tqdm import tqdm

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# 常见“非CFP”的关键字（兜底过滤：掩膜/标注/GT等）
EXCLUDE_KEYWORDS = [
    "mask", "masks", "lesion", "segmentation", "groundtruth", "ground_truth",
    "gt", "annotation", "annotations", "label", "labels"
]

# AutoMorphalyzer 输出里可能用于标识图片名的列（不同版本可能不同）
ID_COL_CANDIDATES = [
    "image", "image_name", "img", "filename", "file",
    "Image", "ImageName", "FileName"
]

def sanitize_stem(stem: str) -> str:
    stem = stem.replace(" ", "_")
    stem = re.sub(r"[^0-9a-zA-Z_\-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem or "img"

def sha1_short(s: str, n: int = 10) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]

def hardlink_or_copy(src: Path, dst: Path, prefer_hardlink: bool = True):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if prefer_hardlink:
        try:
            os.link(src, dst)  # same NTFS volume => fast
            return
        except Exception:
            pass
    shutil.copy2(src, dst)

def write_config(cfg_path: Path, input_dir: Path, output_dir: Path):
    """
    AutoMorphalyzer 的 config.txt 解析一般是按 ': ' split
    必须严格写成 'key: value' 且包含 ': '。
    """
    lines = [
        f"input_directory: {str(input_dir)}",
        f"output_directory: {str(output_dir)}",
        "robust_run: 1",
    ]
    cfg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def find_feature_csv(automorph_out: Path) -> Path:
    """
    宽松寻找 feature measurement csv
    """
    m3 = automorph_out / "M3"
    if m3.exists():
        cands = []
        cands += list(m3.glob("feature_measurements.csv"))
        cands += list(m3.glob("*feature*measure*.csv"))
        cands += list(m3.glob("*feature*.csv"))
        cands = [p for p in cands if p.exists()]
        if cands:
            for p in cands:
                if p.name.lower() == "feature_measurements.csv":
                    return p
            return cands[0]

    cands = list(automorph_out.rglob("feature_measurements.csv"))
    if cands:
        return cands[0]
    cands = list(automorph_out.rglob("*feature*measure*.csv"))
    if cands:
        return cands[0]
    cands = list(automorph_out.rglob("*feature*.csv"))
    if cands:
        return cands[0]

    raise FileNotFoundError(f"Cannot find feature measurement csv under: {automorph_out}")

def run_with_heartbeat(cmd, cwd: Path, out_dir: Path, interval_sec: int = 30):
    print(f"[INFO] AutoMorphalyzer running... (heartbeat every {interval_sec}s)")
    proc = subprocess.Popen(cmd, cwd=str(cwd))
    while True:
        ret = proc.poll()
        try:
            n_files = len(list(out_dir.rglob("*")))
        except Exception:
            n_files = -1
        print(f"[HB] out_files={n_files}")
        if ret is not None:
            if ret != 0:
                raise subprocess.CalledProcessError(ret, cmd)
            break
        time.sleep(interval_sec)

def detect_id_col(df: pd.DataFrame) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in ID_COL_CANDIDATES:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    # 兜底：找包含 image/file/name 的列
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["image", "file", "name"]):
            return c
    return ""

def looks_like_fundus_image(p: Path, min_kb: int = 10) -> bool:
    if not p.is_file():
        return False
    if p.suffix.lower() not in IMG_EXT:
        return False
    low = p.as_posix().lower()
    # 路径里出现这些词，基本是mask/gt
    if any(k in low for k in EXCLUDE_KEYWORDS):
        return False
    try:
        return p.stat().st_size >= min_kb * 1024
    except Exception:
        return False

# ==========================================================
# ✅ RAW 白名单（只把“原图目录”列出来，避免 mask/label 混入）
# 说明：
# - group 用于区分 split / 子集（例如 grading_train）
# - 你要的是“分类头相关”，所以默认只放“分级原图”
# - APTOS test 通常无标签：默认不纳入（可用 --include_aptos_test 打开）
# - DDR 默认只纳入 DR_grading（如你确实要 lesion_seg 的 image，可用参数打开）
# ==========================================================
def build_raw_cfg(include_aptos_test: bool, include_ddr_lesion: bool) -> Dict[str, List[Tuple[str, str]]]:
    cfg = {
        "eyepacs": [
            ("train", r"diabetic-retinopathy-detection/extracted/train"),
            ("test",  r"diabetic-retinopathy-detection/extracted/test"),  # 若不存在会跳过
        ],
        "aptos2019": [
            ("train", r"aptos2019-blindness-detection/train_images"),
        ],
        "ddr": [
            ("grading_train", r"DDR-dataset/DR_grading/train"),
            ("grading_valid", r"DDR-dataset/DR_grading/valid"),
            ("grading_test",  r"DDR-dataset/DR_grading/test"),
        ],
        "messidor2": [
            ("official", r"official/images"),
        ],
        "deepdrid": [
            ("train", r"regular_fundus_images/regular-fundus-training/Images"),
            ("valid", r"regular_fundus_images/regular-fundus-validation/Images"),
            ("eval",  r"regular_fundus_images/Online-Challenge1&2-Evaluation/Images"),
        ],
        "idrid": [
            ("grading_train", r"B. Disease Grading/1. Original Images/a. Training Set"),
            ("grading_test",  r"B. Disease Grading/1. Original Images/b. Testing Set"),
        ],
    }

    if include_aptos_test:
        cfg["aptos2019"].append(("test", r"aptos2019-blindness-detection/test_images"))

    if include_ddr_lesion:
        cfg["ddr"] += [
            ("lesion_train", r"DDR-dataset/lesion_segmentation/train/image"),
            ("lesion_valid", r"DDR-dataset/lesion_segmentation/valid/image"),
            ("lesion_test",  r"DDR-dataset/lesion_segmentation/test/image"),
        ]
    return cfg

def collect_raw_images(project_root: Path, dataset: str,
                       include_aptos_test: bool, include_ddr_lesion: bool,
                       min_kb: int) -> List[Tuple[str, Path]]:
    """
    返回 [(group, raw_path), ...]
    """
    raw_root = project_root / "data" / "raw"
    ds_dir = raw_root / dataset
    if not ds_dir.exists():
        return []

    cfg = build_raw_cfg(include_aptos_test, include_ddr_lesion)
    if dataset not in cfg:
        # 未配置的数据集：兜底扫描，但仍排除关键词
        items = []
        for p in ds_dir.rglob("*"):
            if looks_like_fundus_image(p, min_kb=min_kb):
                items.append(("unknown", p))
        return sorted(items, key=lambda x: str(x[1]))

    items: List[Tuple[str, Path]] = []
    for group, rel in cfg[dataset]:
        root = ds_dir / rel
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if looks_like_fundus_image(p, min_kb=min_kb):
                items.append((group, p))
    # 去重
    seen = set()
    uniq = []
    for g, p in items:
        k = (g, str(p))
        if k in seen:
            continue
        seen.add(k)
        uniq.append((g, p))
    return sorted(uniq, key=lambda x: (x[0], str(x[1])))

def make_sample_id(dataset: str, group: str, raw_path: Path, ds_root: Path) -> str:
    """
    ✅稳定ID：dataset__group__<stem>__<hash10>
    hash 来自相对路径，保证同名不同路径不冲突。
    """
    try:
        rel = raw_path.relative_to(ds_root).as_posix()
    except Exception:
        rel = raw_path.as_posix()
    stem = sanitize_stem(raw_path.stem)
    h = sha1_short(rel, 10)
    return f"{dataset}__{group}__{stem}__{h}"

def prepare_flat_input(run_root: Path, dataset: str, img_items: List[Tuple[str, Path]],
                       prefer_hardlink: bool, limit: int) -> Path:
    """
    AutoMorphalyzer 常见限制：不递归读子目录
    所以 stage 到 input_flat，文件名用 sample_id + ext
    同时写 stage_map.csv：staged_name/sample_id/dataset/group/raw_path
    """
    stage_dir = run_root / "input_flat"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    ds_root = (run_root.parent.parent.parent / "raw" / dataset)  # 这里不用，后面传真实 ds_root 更稳
    # 实际 ds_root 由 main 里传入
    stage_rows = []
    used = set()

    if limit and limit > 0:
        img_items = img_items[:limit]

    for group, p in tqdm(img_items, desc=f"Stage {dataset}", unit="img"):
        # ds_root 这里用 p 的上级推断不准，所以 sample_id 在外面传入更好
        # 我们在 main 里会传真实 ds_dir，这里用 p 直接生成
        # staged 文件名：
        # sample_id + 原扩展名（统一小写）
        # sample_id 由 main 计算，这里暂时占位，main 会覆盖 stage_rows 的 sample_id
        pass

    # 上面占位：我们在 main 里会调用一个“带ds_dir”的版本
    raise RuntimeError("Internal: prepare_flat_input should be called via prepare_flat_input_with_dsdir().")

def prepare_flat_input_with_dsdir(run_root: Path, dataset: str, ds_dir: Path,
                                  img_items: List[Tuple[str, Path]],
                                  prefer_hardlink: bool, limit: int) -> Path:
    stage_dir = run_root / "input_flat"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    stage_rows = []
    used = set()

    if limit and limit > 0:
        img_items = img_items[:limit]

    for group, p in tqdm(img_items, desc=f"Stage {dataset}", unit="img"):
        sample_id = make_sample_id(dataset, group, p, ds_dir)
        ext = p.suffix.lower()
        staged_name = f"{sample_id}{ext}"
        # 避免极端冲突（理论上不会）
        if staged_name in used:
            staged_name = f"{sample_id}__dup__{len(used)}{ext}"
        used.add(staged_name)

        dst = stage_dir / staged_name
        hardlink_or_copy(p, dst, prefer_hardlink=prefer_hardlink)

        stage_rows.append({
            "staged_name": staged_name,
            "sample_id": sample_id,
            "dataset": dataset,
            "group": group,
            "raw_path": str(p),
        })

    stage_map_path = run_root / "stage_map.csv"
    pd.DataFrame(stage_rows).to_csv(stage_map_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] stage_map -> {stage_map_path}")
    print(f"[INFO] Prepared flat input: {len(stage_rows)} images -> {stage_dir}")
    return stage_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--automorphalyzer_dir", required=True)
    ap.add_argument("--run_name", default="run_rmf")

    # ✅raw模式：指定 dataset
    ap.add_argument("--dataset", default="", help="raw模式：数据集名，如 eyepacs/aptos2019/ddr/messidor2")
    ap.add_argument("--min_kb", type=int, default=10, help="过滤过小文件（KB），避免mask/缩略图")

    # 可选白名单开关
    ap.add_argument("--include_aptos_test", action="store_true", help="包含 aptos2019 test_images（通常无标签）")
    ap.add_argument("--include_ddr_lesion", action="store_true", help="包含 DDR lesion_segmentation/image")

    # 兼容旧模式：直接指定输入目录
    ap.add_argument("--input_images_dir", default="", help="(可选) 直接指定输入目录（不推荐raw时用）")

    ap.add_argument("--limit", type=int, default=0, help="0=全跑，>0=限制张数")
    ap.add_argument("--heartbeat", type=int, default=30, help="心跳间隔秒，0=关闭")
    ap.add_argument("--prefer_hardlink", action="store_true", help="stage时优先硬链接（同盘更快更省空间）")

    ap.add_argument("--split_by_dataset", action="store_true", help="额外按 dataset 导出（raw单dataset时可不需要）")

    args = ap.parse_args()

    project = Path(args.project_root)
    repo_dir = Path(args.automorphalyzer_dir)

    main_py = repo_dir / "automorph" / "main.py"
    if not main_py.exists():
        raise FileNotFoundError(f"Cannot find AutoMorphalyzer entry: {main_py}")

    run_root = project / "data" / "processed" / "rmf_automorphalyzer" / "runs" / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)

    automorph_out = run_root / "automorph_out"
    automorph_out.mkdir(parents=True, exist_ok=True)

    # ------------------------
    # 1) collect + stage inputs
    # ------------------------
    if args.input_images_dir:
        # 旧模式：输入目录扫描（仍会过滤关键词）
        input_dir = Path(args.input_images_dir)
        img_paths = [( "unknown", p) for p in sorted(input_dir.rglob("*")) if looks_like_fundus_image(p, min_kb=args.min_kb)]
        if args.limit and args.limit > 0:
            img_paths = img_paths[:args.limit]

        # dataset 名用于 sample_id 前缀
        dataset = args.dataset if args.dataset else "inputdir"
        ds_dir = input_dir  # 用input_dir做相对hash根
        input_for_run = prepare_flat_input_with_dsdir(
            run_root, dataset, ds_dir, img_paths,
            prefer_hardlink=args.prefer_hardlink, limit=0
        )
    else:
        # ✅raw模式：dataset 必须给
        if not args.dataset:
            raise ValueError("raw模式必须提供 --dataset，或提供 --input_images_dir")
        dataset = args.dataset

        raw_root = project / "data" / "raw"
        ds_dir = raw_root / dataset
        if not ds_dir.exists():
            raise FileNotFoundError(f"Dataset raw dir not found: {ds_dir}")

        img_items = collect_raw_images(
            project_root=project,
            dataset=dataset,
            include_aptos_test=args.include_aptos_test,
            include_ddr_lesion=args.include_ddr_lesion,
            min_kb=args.min_kb
        )
        print(f"[INFO] {dataset}: collected raw fundus images = {len(img_items)}")
        if len(img_items) == 0:
            raise RuntimeError(f"No images collected for dataset={dataset}. Check whitelist paths.")

        input_for_run = prepare_flat_input_with_dsdir(
            run_root, dataset, ds_dir, img_items,
            prefer_hardlink=args.prefer_hardlink, limit=args.limit
        )

    # ------------------------
    # 2) write config & run AutoMorphalyzer
    # ------------------------
    cfg_path = repo_dir / "config.txt"
    write_config(cfg_path, input_for_run, automorph_out)
    print(f"[INFO] Wrote config: {cfg_path}")
    print("[INFO] ===== config.txt =====")
    print(cfg_path.read_text(encoding="utf-8", errors="ignore"))
    print("[INFO] ======================")

    cmd = [sys.executable, str(main_py)]
    print("[INFO] Running:", " ".join(cmd))

    if args.heartbeat and args.heartbeat > 0:
        run_with_heartbeat(cmd, cwd=repo_dir, out_dir=automorph_out, interval_sec=args.heartbeat)
    else:
        subprocess.run(cmd, cwd=str(repo_dir), check=True)

    # ------------------------
    # 3) collect rmf + merge stage map
    # ------------------------
    feat_csv = find_feature_csv(automorph_out)

    rmf_raw = run_root / "rmf_raw.csv"
    shutil.copy2(feat_csv, rmf_raw)
    print(f"[DONE] RMF raw -> {rmf_raw}")

    stage_map = pd.read_csv(run_root / "stage_map.csv")
    df = pd.read_csv(rmf_raw)

    id_col = detect_id_col(df)
    if not id_col:
        raise RuntimeError(f"Cannot detect image id column in RMF table. Columns={list(df.columns)[:30]}")

    df[id_col] = df[id_col].astype(str).apply(lambda x: Path(x).name)
    merged = df.merge(stage_map, left_on=id_col, right_on="staged_name", how="left")

    rmf_merged = run_root / "rmf_merged.csv"
    merged.to_csv(rmf_merged, index=False, encoding="utf-8-sig")
    print(f"[DONE] RMF merged (with sample_id/raw_path) -> {rmf_merged}")

    # 你想要“每个数据集一份 RMF 特征表”：这里输出标准 rmf.csv（优先用 merged）
    rmf_final = run_root / "rmf.csv"
    merged.to_csv(rmf_final, index=False, encoding="utf-8-sig")
    print(f"[DONE] RMF final -> {rmf_final}")

    # QC：NaN 或 -1
    bad = merged[merged.isna().any(axis=1) | (merged == -1).any(axis=1)]
    qc_out = run_root / "qc_fail.csv"
    bad.to_csv(qc_out, index=False, encoding="utf-8-sig")
    print(f"[DONE] QC fail -> {qc_out} (rows={len(bad)})")

    if args.split_by_dataset and "dataset" in merged.columns:
        out_dir = run_root / "by_dataset"
        out_dir.mkdir(parents=True, exist_ok=True)
        for ds, ddf in merged.groupby("dataset"):
            (out_dir / ds).mkdir(parents=True, exist_ok=True)
            ddf.to_csv(out_dir / ds / "rmf.csv", index=False, encoding="utf-8-sig")
        print(f"[DONE] Split by dataset -> {out_dir}")


if __name__ == "__main__":
    main()
