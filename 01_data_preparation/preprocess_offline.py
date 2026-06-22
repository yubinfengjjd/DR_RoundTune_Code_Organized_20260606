# DR_RoundTune_Project/src/preprocess_offline.py
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ==========================================================
# ✅ White-list config (Only scan "original image directories" to avoid mixing masks/labels with original images)
# Your raw directory structure is based on the provided tree.
# Each image_roots / mask_roots supports two formats:
#   1) ("group", r"rel/path")
#   2) r"rel/path"  -> group automatically uses default values (image defaults to "fundus", mask defaults to "seg")
# ==========================================================
DATASET_CFG: Dict[str, Dict[str, List[Union[str, Tuple[str, str]]]]] = {
    "aptos2019": {
        "image_roots": [
            ("fundus", r"aptos2019-blindness-detection/train_images"),
            ("fundus", r"aptos2019-blindness-detection/test_images"),  # If not present, it will be skipped automatically
        ],
        "mask_roots": [],
    },
    "ddr": {
        "image_roots": [
            ("grading", r"DDR-dataset/DR_grading/train"),
            ("grading", r"DDR-dataset/DR_grading/valid"),
            ("grading", r"DDR-dataset/DR_grading/test"),
            ("lesion_seg", r"DDR-dataset/lesion_segmentation/train/image"),
            ("lesion_seg", r"DDR-dataset/lesion_segmentation/valid/image"),
            ("lesion_seg", r"DDR-dataset/lesion_segmentation/test/image"),
        ],
        "mask_roots": [
            # DDR: label/EX HE MA SE ...
            ("lesion_seg", r"DDR-dataset/lesion_segmentation/train/label"),
            ("lesion_seg", r"DDR-dataset/lesion_segmentation/test/label"),
            # The directory name for valid provided was "segmentation label"
            ("lesion_seg", r"DDR-dataset/lesion_segmentation/valid/segmentation label"),
        ],
    },
    "deepdrid": {
        "image_roots": [
            ("fundus", r"regular_fundus_images/regular-fundus-training/Images"),
            ("fundus", r"regular_fundus_images/regular-fundus-validation/Images"),
            ("fundus", r"regular_fundus_images/Online-Challenge1&2-Evaluation/Images"),
        ],
        "mask_roots": [],
    },
    "eyepacs": {
        "image_roots": [
            ("fundus", r"diabetic-retinopathy-detection/extracted/train"),
            ("fundus", r"diabetic-retinopathy-detection/extracted/test"),  # If not present, it will be skipped
        ],
        "mask_roots": [],
    },
    "idrid": {
        "image_roots": [
            ("grading", r"B. Disease Grading/1. Original Images/a. Training Set"),
            ("grading", r"B. Disease Grading/1. Original Images/b. Testing Set"),
            ("seg", r"A. Segmentation/1. Original Images/a. Training Set"),
            ("seg", r"A. Segmentation/1. Original Images/b. Testing Set"),
        ],
        "mask_roots": [
            ("seg", r"A. Segmentation/2. All Segmentation Groundtruths/a. Training Set"),
            ("seg", r"A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set"),
        ],
    },
    "messidor2": {
        "image_roots": [
            ("fundus", r"official/images"),
        ],
        "mask_roots": [],
    },
}


@dataclass(frozen=True)
class CropParams:
    x0: int
    y0: int
    x1: int
    y1: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def list_images_under(root: Path) -> List[Path]:
    if not root.exists():
        return []
    out: List[Path] = []
    for p in root.rglob("*"):
        if is_image_file(p):
            out.append(p)
    return out


def sha1_short(s: str, n: int = 8) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def sanitize_stem(stem: str) -> str:
    stem = stem.replace(" ", "_")
    stem = re.sub(r"[^0-9a-zA-Z_\-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem or "img"


def read_rgb(path: Path) -> Optional[np.ndarray]:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def read_mask(path: Path) -> Optional[np.ndarray]:
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def write_jpg(rgb: np.ndarray, out_path: Path, quality: int = 95) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])


def write_png(arr: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), arr)


def compute_crop_params(rgb: np.ndarray, thr: int = 10) -> CropParams:
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = (gray > thr).astype(np.uint8) * 255

    k = max(7, (min(h, w) // 80) | 1)
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        side = min(h, w)
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        x1 = x0 + side
        y1 = y0 + side
        return CropParams(x0, y0, x1, y1, 0, 0, 0, 0)

    c = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(c)

    side = int(max(bw, bh))
    cx = x + bw / 2.0
    cy = y + bh / 2.0
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x1 = x0 + side
    y1 = y0 + side

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w)
    pad_bottom = max(0, y1 - h)

    return CropParams(x0, y0, x1, y1, pad_left, pad_top, pad_right, pad_bottom)


def crop_with_padding(img: np.ndarray, cp: CropParams, pad_value=0) -> np.ndarray:
    if any([cp.pad_left, cp.pad_top, cp.pad_right, cp.pad_bottom]):
        if img.ndim == 3:
            val = (pad_value, pad_value, pad_value)
        else:
            val = pad_value
        img = cv2.copyMakeBorder(
            img,
            top=cp.pad_top,
            bottom=cp.pad_bottom,
            left=cp.pad_left,
            right=cp.pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=val,
        )
    x0 = cp.x0 + cp.pad_left
    y0 = cp.y0 + cp.pad_top
    x1 = cp.x1 + cp.pad_left
    y1 = cp.y1 + cp.pad_top
    return img[y0:y1, x0:x1]


def resize_img(img: np.ndarray, out_size: int, is_mask: bool) -> np.ndarray:
    h, w = img.shape[:2]
    if h == out_size and w == out_size:
        return img
    if is_mask:
        interp = cv2.INTER_NEAREST
    else:
        interp = cv2.INTER_AREA if max(h, w) > out_size else cv2.INTER_LINEAR
    return cv2.resize(img, (out_size, out_size), interpolation=interp)


def shard_dir(sample_id: str, shard_len: int) -> str:
    if shard_len <= 0:
        return ""
    h = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()
    return h[:shard_len]


def make_sample_id(dataset: str, group: str, raw_path: Path, raw_root: Path) -> str:
    """
    Readable naming: dataset__group__<orig_stem>__<hash6>
    """
    stem = sanitize_stem(raw_path.stem)
    rel = str(raw_path.relative_to(raw_root)).replace("\\", "/")
    h = sha1_short(rel, 6)
    return f"{dataset}__{group}__{stem}__{h}"


def normalize_image_key(dataset: str, p: Path) -> str:
    """
    Pairing key for original images (usually normalized stem)
    """
    return sanitize_stem(p.stem)


def normalize_mask_key(dataset: str, mp: Path) -> str:
    """
    Pairing key for masks
    - DDR: mask filename is usually the same as image filename => stem
    - IDRiD: common naming like IDRiD_01_MA / IDRiD_01_EX / ... => need to remove lesion suffix at the end
    """
    stem = sanitize_stem(mp.stem)

    if dataset.lower() == "idrid":
        # Remove common suffixes: _MA _HE _EX _SE _OD etc. (case insensitive)
        stem = re.sub(r"(_MA|_HE|_EX|_SE|_OD|_BG)$", "", stem, flags=re.IGNORECASE)
        # Some might be -MA or .MA, etc., handling as fallback
        stem = re.sub(r"(-MA|-HE|-EX|-SE|-OD)$", "", stem, flags=re.IGNORECASE)

    return stem


def infer_mask_type_ddr(mp: Path) -> str:
    parts = [p.upper() for p in mp.parts]
    for t in ["MA", "HE", "EX", "SE"]:
        if t in parts:
            return t
    return "MASK"


def infer_mask_type_idrid(mp: Path) -> str:
    parent = mp.parent.name.lower()
    # IDRiD official directory names: Microaneurysms / Haemorrhages / Hard Exudates / Soft Exudates / Optic Disc
    if "micro" in parent:
        return "MA"
    if "haem" in parent or "hem" in parent:
        return "HE"
    if "hard" in parent and "exud" in parent:
        return "EX"
    if "soft" in parent and "exud" in parent:
        return "SE"
    if "optic" in parent and "disc" in parent:
        return "OD"
    return sanitize_stem(mp.parent.name)[:32] or "MASK"


def infer_mask_type(dataset: str, mp: Path) -> str:
    ds = dataset.lower()
    if ds == "ddr":
        return infer_mask_type_ddr(mp)
    if ds == "idrid":
        return infer_mask_type_idrid(mp)
    return "MASK"


def iter_cfg_items(items: List[Union[str, Tuple[str, str]]], default_group: str) -> List[Tuple[str, str]]:
    """
    Compatible with string or tuple configuration
    """
    out: List[Tuple[str, str]] = []
    for it in items:
        if isinstance(it, (list, tuple)) and len(it) == 2:
            out.append((str(it[0]), str(it[1])))
        else:
            out.append((default_group, str(it)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default="DR_RoundTune_Project")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--thr", type=int, default=10)
    ap.add_argument("--jpeg_quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=1, help="Suggested 1; script processes sequentially by default to ensure stable alignment")

    ap.add_argument("--shard_len", type=int, default=2, help="2 => 00/0a/...; 0 => no shard")
    ap.add_argument("--export_masks", action="store_true", help="export masks to processed/cfp_{size}/masks/")
    ap.add_argument("--align_masks", action="store_true", help="align masks using image crop params + nearest resize")

    ap.add_argument("--overwrite_manifests", action="store_true", help="overwrite manifests if exist")
    ap.add_argument("--overwrite_manifest", action="store_true", help="compat: same as --overwrite_manifests")

    args = ap.parse_args()
    if args.overwrite_manifest:
        args.overwrite_manifests = True

    project_root = Path(args.project_root).resolve()
    raw_root = project_root / "data" / "raw"
    out_root = project_root / "data" / "processed" / f"cfp_{args.size}"
    out_img_root = out_root / "images"
    out_mask_root = out_root / "masks"
    qc_root = out_root / "qc"
    qc_root.mkdir(parents=True, exist_ok=True)

    manifest_dir = project_root / "data" / "meta" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    mani_img = manifest_dir / "manifest_offline_images.csv"
    mani_msk = manifest_dir / "manifest_offline_masks.csv"
    mani_compat = manifest_dir / "manifest_offline.csv"  # Compatible with old workflow

    if (mani_img.exists() or mani_msk.exists() or mani_compat.exists()) and not args.overwrite_manifests:
        raise FileExistsError(
            f"Manifests exist. Use --overwrite_manifests to overwrite:\n{mani_img}\n{mani_msk}\n{mani_compat}"
        )

    if args.workers != 1:
        print("[WARN] workers != 1. Current version processes sequentially (to ensure crop/mask alignment stability), workers argument will be ignored.")

    # -----------------------------
    # 1) collect whitelist images/masks
    # -----------------------------
    image_items: List[Tuple[str, str, Path]] = []  # (dataset, group, path)
    mask_items: List[Tuple[str, str, Path]] = []   # (dataset, group, mask_path)

    for ds, cfg in DATASET_CFG.items():
        ds_dir = raw_root / ds
        if not ds_dir.exists():
            continue

        for group, rel_root in iter_cfg_items(cfg.get("image_roots", []), default_group="fundus"):
            root = ds_dir / rel_root
            for p in list_images_under(root):
                image_items.append((ds, group, p))

        for group, rel_root in iter_cfg_items(cfg.get("mask_roots", []), default_group="seg"):
            root = ds_dir / rel_root
            for p in list_images_under(root):
                mask_items.append((ds, group, p))

    # Deduplicate + Sort
    image_items = sorted(set(image_items), key=lambda x: (x[0], x[1], str(x[2])))
    mask_items = sorted(set(mask_items), key=lambda x: (x[0], x[1], str(x[2])))

    print(f"[INFO] Found CFP images (whitelist): {len(image_items)}")
    print(f"[INFO] Found masks (whitelist): {len(mask_items)}")

    # -----------------------------
    # 2) process images
    # -----------------------------
    img_records: List[Dict] = []
    bad_images: List[Dict] = []

    # pairing tables
    # key: (dataset, group, norm_key) -> sample_id
    img_key_to_sample: Dict[Tuple[str, str, str], str] = {}
    sample_to_crop: Dict[str, Dict] = {}
    sample_to_raw: Dict[str, Path] = {}

    for i, (ds, group, p) in enumerate(image_items, 1):
        sample_id = make_sample_id(ds, group, p, raw_root)
        rgb = read_rgb(p)
        if rgb is None:
            rec = {
                "sample_id": sample_id,
                "dataset": ds,
                "group": group,
                "key": normalize_image_key(ds, p),
                "raw_path": str(p),
                "proc_path": "",
                "status": "bad_decode",
                "out_size": args.size,
                "jpeg_quality": args.jpeg_quality,
                "crop_json": "",
            }
            img_records.append(rec)
            bad_images.append(rec)
            continue

        cp = compute_crop_params(rgb, thr=args.thr)
        cropped = crop_with_padding(rgb, cp, pad_value=0)
        resized = resize_img(cropped, args.size, is_mask=False)

        sh = shard_dir(sample_id, args.shard_len)
        out_dir = out_img_root / ds / group
        if sh:
            out_dir = out_dir / sh
        out_path = out_dir / f"{sample_id}.jpg"
        write_jpg(resized, out_path, quality=args.jpeg_quality)

        crop_json = {
            "x0": cp.x0, "y0": cp.y0, "x1": cp.x1, "y1": cp.y1,
            "pad_left": cp.pad_left, "pad_top": cp.pad_top,
            "pad_right": cp.pad_right, "pad_bottom": cp.pad_bottom,
            "thr": args.thr,
        }

        key = normalize_image_key(ds, p)

        rec = {
            "sample_id": sample_id,
            "dataset": ds,
            "group": group,
            "key": key,
            "raw_path": str(p),
            "proc_path": str(out_path),
            "status": "ok",
            "out_size": args.size,
            "jpeg_quality": args.jpeg_quality,
            "crop_json": json.dumps(crop_json, ensure_ascii=False),
        }
        img_records.append(rec)

        img_key_to_sample[(ds, group, key)] = sample_id
        sample_to_crop[sample_id] = crop_json
        sample_to_raw[sample_id] = p

        if i % 2000 == 0:
            print(f"[INFO] Processed images {i}/{len(image_items)} ...")

    # write image manifests
    img_fields = ["sample_id", "dataset", "group", "key", "raw_path", "proc_path",
                  "status", "out_size", "jpeg_quality", "crop_json"]

    with open(mani_img, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=img_fields)
        w.writeheader()
        for r in img_records:
            w.writerow({k: r.get(k, "") for k in img_fields})

    # compat manifest_offline.csv (old columns)
    with open(mani_compat, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "dataset", "raw_path", "proc_path", "status", "out_size"])
        w.writeheader()
        for r in img_records:
            w.writerow({
                "sample_id": r.get("sample_id", ""),
                "dataset": r.get("dataset", ""),
                "raw_path": r.get("raw_path", ""),
                "proc_path": r.get("proc_path", ""),
                "status": r.get("status", ""),
                "out_size": r.get("out_size", ""),
            })

    bad_decode_img_path = qc_root / "bad_decode_images.csv"
    with open(bad_decode_img_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "dataset", "group", "raw_path", "status"])
        w.writeheader()
        for r in bad_images:
            w.writerow({
                "sample_id": r.get("sample_id", ""),
                "dataset": r.get("dataset", ""),
                "group": r.get("group", ""),
                "raw_path": r.get("raw_path", ""),
                "status": r.get("status", ""),
            })

    print(f"[DONE] Images -> {out_img_root}")
    print(f"[DONE] Manifest images -> {mani_img}")
    print(f"[DONE] Manifest compat -> {mani_compat}")
    print(f"[DONE] QC bad_decode images -> {bad_decode_img_path}")

    # -----------------------------
    # 3) export masks (optional)
    # -----------------------------
    mask_records: List[Dict] = []
    bad_masks: List[Dict] = []

    msk_fields = ["sample_id", "dataset", "group", "mask_type", "raw_mask_path",
                  "proc_mask_path", "status", "aligned", "paired_sample_id"]

    if args.export_masks and len(mask_items) > 0:
        for i, (ds, group, mp) in enumerate(mask_items, 1):
            m = read_mask(mp)
            if m is None:
                bad_masks.append({
                    "sample_id": "",
                    "dataset": ds,
                    "group": group,
                    "mask_type": infer_mask_type(ds, mp),
                    "raw_mask_path": str(mp),
                    "status": "bad_decode",
                })
                continue

            mask_type = infer_mask_type(ds, mp)
            mkey = normalize_mask_key(ds, mp)

            paired_sample = img_key_to_sample.get((ds, group, mkey), "")
            aligned_flag = False
            out_mask = m

            # align if possible
            if args.align_masks and paired_sample and paired_sample in sample_to_crop and paired_sample in sample_to_raw:
                raw_img_path = sample_to_raw[paired_sample]
                rgb = read_rgb(raw_img_path)
                if rgb is not None:
                    ih, iw = rgb.shape[:2]
                    mh, mw = out_mask.shape[:2]
                    # If mask size differs from original image, resize to original size first, then crop/resize
                    if (mh != ih) or (mw != iw):
                        out_mask = cv2.resize(out_mask, (iw, ih), interpolation=cv2.INTER_NEAREST)

                    cpj = sample_to_crop[paired_sample]
                    cp = CropParams(
                        x0=int(cpj["x0"]), y0=int(cpj["y0"]), x1=int(cpj["x1"]), y1=int(cpj["y1"]),
                        pad_left=int(cpj["pad_left"]), pad_top=int(cpj["pad_top"]),
                        pad_right=int(cpj["pad_right"]), pad_bottom=int(cpj["pad_bottom"]),
                    )
                    out_mask = crop_with_padding(out_mask, cp, pad_value=0)
                    out_mask = resize_img(out_mask, args.size, is_mask=True)
                    aligned_flag = True

            # decide sample_id for saving
            if paired_sample:
                sample_id = paired_sample
                status = "ok"
                paired_id = paired_sample
            else:
                # unmatched mask still exported to UNPAIRED
                rel = str(mp).replace("\\", "/")
                sample_id = f"{ds}__{group}__UNPAIRED__{mkey}__{sha1_short(rel,6)}"
                status = "ok_unpaired"
                paired_id = ""

            sh = shard_dir(sample_id, args.shard_len)

            if status == "ok":
                out_dir = out_mask_root / ds / group / mask_type
            else:
                out_dir = out_mask_root / ds / group / "UNPAIRED" / mask_type

            if sh:
                out_dir = out_dir / sh

            out_path = out_dir / f"{sample_id}.png"
            write_png(out_mask, out_path)

            mask_records.append({
                "sample_id": sample_id,
                "dataset": ds,
                "group": group,
                "mask_type": mask_type,
                "raw_mask_path": str(mp),
                "proc_mask_path": str(out_path),
                "status": status,
                "aligned": aligned_flag,
                "paired_sample_id": paired_id,
            })

            if i % 2000 == 0:
                print(f"[INFO] Exported masks {i}/{len(mask_items)} ...")

    # write mask manifest (even if empty)
    with open(mani_msk, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=msk_fields)
        w.writeheader()
        for r in mask_records:
            w.writerow({k: r.get(k, "") for k in msk_fields})

    bad_decode_msk_path = qc_root / "bad_decode_masks.csv"
    with open(bad_decode_msk_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "group", "mask_type", "raw_mask_path", "status"])
        w.writeheader()
        for r in bad_masks:
            w.writerow({
                "dataset": r.get("dataset", ""),
                "group": r.get("group", ""),
                "mask_type": r.get("mask_type", ""),
                "raw_mask_path": r.get("raw_mask_path", ""),
                "status": r.get("status", ""),
            })

    print(f"[DONE] Manifest masks -> {mani_msk}")
    print(f"[DONE] QC bad_decode masks -> {bad_decode_msk_path}")

    # -----------------------------
    # 4) stats
    # -----------------------------
    stats = {
        "raw_root": str(raw_root),
        "out_root": str(out_root),
        "n_images_found": len(image_items),
        "n_images_records": len(img_records),
        "n_images_bad_decode": len(bad_images),
        "n_masks_found": len(mask_items),
        "n_masks_exported": len(mask_records),
        "params": {
            "size": args.size,
            "thr": args.thr,
            "jpeg_quality": args.jpeg_quality,
            "export_masks": args.export_masks,
            "align_masks": args.align_masks,
            "shard_len": args.shard_len,
        },
    }
    stats_path = qc_root / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[DONE] QC stats -> {stats_path}")


if __name__ == "__main__":
    main()