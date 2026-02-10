"""Dataset balancing script that augments minority classes with one-op transforms."""

import argparse
import random
import shutil
from pathlib import Path
import numpy as np, imageio
import imgaug.augmenters as iaa

WHITE = 255
aug_ops = {
    "brightness": iaa.AddToBrightness(add=(-30, 30)),
    "contrast":   iaa.LinearContrast(alpha=(0.5, 2.0)),
    "scale":      iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, fit_output=False, cval=WHITE),
    "perspective":iaa.PerspectiveTransform(scale=(0.01, 0.15), keep_size=True, cval=WHITE),
    "rotation":   iaa.Affine(rotate=(-25, 25), fit_output=False, cval=WHITE),
    "translation":iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, fit_output=False, cval=WHITE),
    "shear":      iaa.Affine(shear=(-15, 15), fit_output=False, cval=WHITE),
}
op_names = list(aug_ops.keys())
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def list_images(d: Path):
    """
    List image files in a directory, sorted by name.
    """
    return sorted([p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS])

def copy_all(src: Path, dst: Path):
    """
    Copy a file to the destination, creating parent directories as needed.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def augment_one(src_img: Path, dst_img: Path, op_name: str):
    """
    Apply a single augmentation op and write the augmented image.

    Steps:
    1) Load the source image.
    2) Apply the chosen augmentation.
    3) Save the result to disk.
    """
    img = imageio.v2.imread(src_img)
    aug = aug_ops[op_name](image=img)
    imageio.imwrite(dst_img, aug)

def main():
    """
    CLI entry point to balance classes via single-op augmentation.

    Steps:
    1) Parse CLI args and determine target class count.
    2) Copy originals into the output tree.
    3) Generate augmented images for minority classes.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force_majority_class", type=int, default=None)
    ap.add_argument("--even_ops", action="store_true",
                    help="Cycle ops evenly instead of random choice")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    in_root = Path(args.input_root); out_root = Path(args.output_root)

    classes = [0,1,2,3,4]
    imgs = {c: list_images(in_root/str(c)) for c in classes}
    counts = {c: len(v) for c,v in imgs.items()}

    if args.force_majority_class is not None:
        target_c = args.force_majority_class
        target_n = counts[target_c]
    else:
        target_c = max(counts, key=counts.get)
        target_n = counts[target_c]

    print("Input counts:", counts)
    print(f"Target class {target_c} with {target_n} images")

    # copy originals
    for c in classes:
        for p in imgs[c]:
            dst = out_root / str(c) / p.name
            if not args.dry_run:
                copy_all(p, dst)

    # balance minorities with single op per image
    for c in classes:
        if c == target_c: 
            print(f"class {c}: unchanged")
            continue
        need = target_n - counts[c]
        print(f"class {c}: need {need} augmented images")

        for i in range(max(0, need)):
            src = random.choice(imgs[c])
            if args.even_ops:
                op = op_names[i % len(op_names)]
            else:
                op = random.choice(op_names)
            stem, ext = src.stem, src.suffix
            new_name = f"{stem}_{op}_aug{i:05d}{ext}"
            dst = out_root / str(c) / new_name
            if not args.dry_run:
                augment_one(src, dst, op)

    if args.dry_run:
        print("Dry run. No files written.")

if __name__ == "__main__":
    main()
