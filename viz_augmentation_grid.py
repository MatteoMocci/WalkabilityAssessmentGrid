"""Visualization utility for comparing augmentation effects on matched image pairs."""

# viz_augmentation_grid.py
# Run: python viz_augmentation_grid.py
import os
import re
import random
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import imageio
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt

# =========================
# CONFIG (kept exactly as requested)
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SV_ROOT = Path(os.getenv("WALKCNN_STREET_AUG_DIR", PROJECT_ROOT / "augmented-streetview"))  # contains class folders (e.g., 0..4)
SAT_ROOT = Path(os.getenv("WALKCNN_SAT_AUG_DIR", PROJECT_ROOT / "augmented-satellite"))      # contains class folders (e.g., 0..4)
CLASS_ID = "4"                                  # class folder to sample from
PICK_RANDOM_IMAGE = True                        # True: pick random, False: pick first
MATCH_TOL_DEG = 1e-4                            # coordinate tolerance (degrees)
SEED = 46                                       # reproducibility
OUTPUT_DIR = Path(os.getenv("WALKCNN_VIZ_OUT_DIR", Path(__file__).resolve().parent))          # where to save the figures
# =========================

WHITE = 255
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# -----------------------
# Filename parsing (id_lat_lon_[...].jpg)
# -----------------------
FLOAT_RE = r"[-+]?\d+(?:\.\d+)?"

def extract_lat_lon(name: str) -> Optional[Tuple[float, float]]:
    """
    Filenames follow id_lat_lon_[...].jpg
    We take the 2nd and 3rd float tokens as (lat, lon).
    Fallback: first plausible pair within [-90,90] x [-180,180].
    """
    nums = re.findall(FLOAT_RE, name)
    vals = [float(x) for x in nums]
    if len(vals) >= 3:
        lat, lon = vals[1], vals[2]
        if abs(lat) <= 90 and abs(lon) <= 180:
            return lat, lon
    for i in range(len(vals) - 1):
        lat, lon = vals[i], vals[i + 1]
        if abs(lat) <= 90 and abs(lon) <= 180:
            return lat, lon
    return None

def is_image(path: Path) -> bool:
    """Return True if the path is an image file with a supported extension."""
    return path.is_file() and path.suffix.lower() in IMG_EXTS

def looks_original(path: Path) -> bool:
    """Return True if the filename does not appear to be augmented."""
    # prefer originals; skip any augmented files
    return "_aug" not in path.name.lower()

def list_candidates(folder: Path):
    """Return originals if present; else fall back to any images."""
    originals = [p for p in folder.iterdir() if is_image(p) and looks_original(p)]
    if originals:
        return originals
    return [p for p in folder.iterdir() if is_image(p)]

def build_coord_index(root: Path) -> Dict[Tuple[int, int], Dict[Tuple[float, float], Path]]:
    """
    Index images by coarse bins of lat/lon for tolerance search.
    Prefer originals; if none exist, include all images.
    """
    index: Dict[Tuple[int, int], Dict[Tuple[float, float], Path]] = {}
    files = [p for p in root.rglob("*") if is_image(p) and looks_original(p)]
    if not files:
        files = [p for p in root.rglob("*") if is_image(p)]
    for p in files:
        ll = extract_lat_lon(p.name)
        if ll is None:
            continue
        lat, lon = ll
        key = (int(lat * 1000), int(lon * 1000))  # coarse bin at 1e-3 deg
        index.setdefault(key, {})[(lat, lon)] = p
    return index

def find_match_by_coords(target_path: Path, sat_index, tol: float) -> Optional[Path]:
    """
    Find a matching satellite image by coordinate proximity.

    Steps:
    1) Extract coordinates from target filename.
    2) Search nearby coarse bins.
    3) Expand tolerance until a match is found.
    """
    ll = extract_lat_lon(target_path.name)
    if ll is None:
        return None
    lat, lon = ll
    base_key = (int(lat * 1000), int(lon * 1000))
    neighbor_keys = [(base_key[0] + di, base_key[1] + dj)
                     for di in (-1, 0, 1) for dj in (-1, 0, 1)]

    def try_tol(t):
        """
        Try to find a match within a given tolerance value.
        """
        best, best_d = None, float("inf")
        for k in neighbor_keys:
            bucket = sat_index.get(k, {})
            for (slat, slon), path in bucket.items():
                d = max(abs(slat - lat), abs(slon - lon))  # fast Chebyshev in degrees
                if d <= t and d < best_d:
                    best, best_d = path, d
        return best

    # grow tolerance if needed
    for factor in (1, 2, 5, 10):
        cand = try_tol(tol * factor)
        if cand is not None:
            return cand
    return None

# -----------------------
# Augmentations (sample params explicitly so we can display them)
# -----------------------
def aug_original(img):
    """Return the original image without modification."""
    return img, "original"

def aug_brightness(img):
    """Apply a random brightness adjustment and return label text."""
    Δ = random.randint(-30, 30)
    aug = iaa.AddToBrightness(add=Δ)
    return aug(image=img), f"brightness Δ={Δ}"

def aug_contrast(img):
    """Apply a random contrast adjustment and return label text."""
    α = round(random.uniform(0.5, 2.0), 2)
    aug = iaa.LinearContrast(alpha=α)
    return aug(image=img), f"contrast α={α}"

def aug_scale(img):
    """Apply a random scale transform and return label text."""
    x = round(random.uniform(0.8, 1.2), 2)
    y = round(random.uniform(0.8, 1.2), 2)
    aug = iaa.Affine(scale={"x": x, "y": y}, fit_output=False, cval=WHITE)
    return aug(image=img), f"scale x={x}, y={y}"

def aug_perspective(img):
    """Apply a random perspective transform and return label text."""
    λ = round(random.uniform(0.01, 0.15), 3)
    aug = iaa.PerspectiveTransform(scale=λ, keep_size=True, cval=WHITE)
    return aug(image=img), f"perspective λ={λ}"

def aug_rotation(img):
    """Apply a random rotation and return label text."""
    θ = round(random.uniform(-25, 25), 1)
    aug = iaa.Affine(rotate=θ, fit_output=False, cval=WHITE)
    return aug(image=img), f"rotation θ={θ}°"

def aug_translation(img):
    """Apply a random translation and return label text."""
    tx = round(random.uniform(-0.2, 0.2), 2)
    ty = round(random.uniform(-0.2, 0.2), 2)
    aug = iaa.Affine(translate_percent={"x": tx, "y": ty}, fit_output=False, cval=WHITE)
    return aug(image=img), f"translation x={tx}, y={ty}"

def aug_shear(img):
    """Apply a random shear and return label text."""
    φ = round(random.uniform(-15, 15), 1)
    aug = iaa.Affine(shear=φ, fit_output=False, cval=WHITE)
    return aug(image=img), f"shear φ={φ}°"

AUG_FUNCS = [
    aug_original,
    aug_brightness,
    aug_contrast,
    aug_scale,
    aug_perspective,
    aug_rotation,
    aug_translation,
    aug_shear,
]  # 8 entries; 9th grid cell stays empty

def apply_all_ops_with_params(img: np.ndarray):
    """
    Apply each augmentation op and collect images and titles.

    Steps:
    1) Run each augmentation function.
    2) Collect the augmented image and label.
    """
    images, titles = [], []
    for f in AUG_FUNCS:
        out, title = f(img)
        images.append(out)
        titles.append(title)
    return images, titles

# -----------------------
# Plotting
# -----------------------
def plot_3x3(images, titles, header_text: str):
    """
    Plot a 3x3 grid with a header and per-cell titles.
    """
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))  # uniform cell size
    axes = axes.ravel()

    # header at top margin with the source filename
    fig.text(0.5, 0.985, header_text, ha="center", va="top", fontsize=12)

    for i in range(8):
        axes[i].imshow(images[i])
        axes[i].set_title(titles[i], fontsize=10, pad=2)  # show op name + sampled params
        axes[i].axis("off")

    # leave last cell empty
    axes[8].axis("off")

    # tighter spacing
    plt.subplots_adjust(top=0.96, hspace=0.08, wspace=0.04)
    return fig

# -----------------------
# Main
# -----------------------
def main():
    """
    CLI entry point to generate street/satellite augmentation grids.

    Steps:
    1) Pick a matched street/satellite pair by coordinates.
    2) Apply all augmentations and plot 3x3 grids.
    3) Save the output images.
    """
    random.seed(SEED); np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sv_class = SV_ROOT / CLASS_ID
    sat_class = SAT_ROOT / CLASS_ID
    if not sv_class.exists():
        raise FileNotFoundError(f"Missing streetview class folder: {sv_class}")
    if not sat_class.exists():
        raise FileNotFoundError(f"Missing satellite class folder: {sat_class}")

    sv_candidates = list_candidates(sv_class)
    if not sv_candidates:
        raise FileNotFoundError(f"No images found in {sv_class}")

    sv_img_path = random.choice(sv_candidates) if PICK_RANDOM_IMAGE else sorted(sv_candidates)[0]

    # match satellite by coordinates with tolerance growth
    sat_index = build_coord_index(sat_class)
    sat_img_path = find_match_by_coords(sv_img_path, sat_index, MATCH_TOL_DEG)
    if sat_img_path is None:
        raise FileNotFoundError(
            f"No satellite image matched coordinates of {sv_img_path.name} within tol={MATCH_TOL_DEG} (even after widening)."
        )

    # load images
    sv_img = imageio.v2.imread(sv_img_path)
    sat_img = imageio.v2.imread(sat_img_path)

    # apply original + 7 augmentations with sampled parameter labels
    sv_imgs, sv_titles = apply_all_ops_with_params(sv_img)
    sat_imgs, sat_titles = apply_all_ops_with_params(sat_img)

    # streetview plot
    sv_header = f"Streetview source: {sv_img_path.name}"
    fig_sv = plot_3x3(sv_imgs, sv_titles, sv_header)
    out_sv = OUTPUT_DIR / f"sv_aug_grid__{sv_img_path.stem}.png"
    fig_sv.savefig(out_sv, dpi=200, bbox_inches="tight")
    plt.close(fig_sv)

    # satellite plot
    sat_header = f"Satellite source: {sat_img_path.name}"
    fig_sat = plot_3x3(sat_imgs, sat_titles, sat_header)
    out_sat = OUTPUT_DIR / f"sat_aug_grid__{sat_img_path.stem}.png"
    fig_sat.savefig(out_sat, dpi=200, bbox_inches="tight")
    plt.close(fig_sat)

    print("Saved:")
    print(f" - {out_sv}")
    print(f" - {out_sat}")

if __name__ == "__main__":
    main()
