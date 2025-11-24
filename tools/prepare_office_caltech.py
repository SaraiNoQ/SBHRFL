"""Rearrange Office-Caltech into ImageFolder-friendly train/test splits.

Expected input layout (common downloads):
data/office_caltech_raw/
  amazon/
    class1/img1.jpg
  dslr/
    class1/img2.jpg
  webcam/
    class2/img3.jpg
  caltech/
    class3/img4.jpg

This script flattens all domains, merges by class, and writes:
data/office_caltech/train/<class_name>/*
data/office_caltech/test/<class_name>/*

Splits use a deterministic 80/20 split per class. Adjust --train-ratio if needed.
"""

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def gather_images(root: Path):
    items = []
    for path in root.rglob("*"):
        if path.suffix.lower() in IMAGE_EXTS and path.is_file():
            cls = path.parent.name
            items.append((path, cls))
    return items


def split_by_class(items, train_ratio: float, seed: int):
    random.seed(seed)
    by_class = {}
    for path, cls in items:
        by_class.setdefault(cls, []).append(path)
    splits = {"train": [], "test": []}
    for cls, paths in by_class.items():
        random.shuffle(paths)
        cut = max(1, int(len(paths) * train_ratio)) if len(paths) > 1 else len(paths)
        splits["train"].extend((p, cls) for p in paths[:cut])
        splits["test"].extend((p, cls) for p in paths[cut:])
    return splits


def copy_split(splits, out_root: Path):
    for split_name, items in splits.items():
        for src, cls in items:
            dst = out_root / split_name / cls / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Prepare Office-Caltech for ImageFolder.")
    parser.add_argument("--source", type=str, default="data/office_caltech_raw", help="Raw dataset root (domains under here).")
    parser.add_argument("--target", type=str, default="data/office_caltech", help="Output root with train/test folders.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Per-class train split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    args = parser.parse_args()

    src_root = Path(args.source)
    dst_root = Path(args.target)
    if not src_root.exists():
        raise FileNotFoundError(f"Source root not found: {src_root}")
    if dst_root.exists():
        print(f"[Info] Target {dst_root} exists; files may be overwritten.")

    items = gather_images(src_root)
    if not items:
        raise RuntimeError(f"No images found under {src_root}. Check source layout.")
    splits = split_by_class(items, args.train_ratio, args.seed)
    copy_split(splits, dst_root)
    print(f"Done. Train: {len(splits['train'])} images, Test: {len(splits['test'])} images -> {dst_root}")


if __name__ == "__main__":
    main()
