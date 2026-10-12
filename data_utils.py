"""Dataset and dataloader utilities for paired street/satellite training splits."""

# data_utils.py
import csv
import glob
import os
import random
from collections import defaultdict

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def _build_eval_tf(img_size: int):
    """
    Build the evaluation transform pipeline for input images.

    Steps:
    1) Resize to square.
    2) Convert to tensor.
    3) Normalize with ImageNet stats.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def _make_aug_index(aug_root, rels=None, logger=None):
    """
    Build an index mapping ORIGINAL rel path (e.g. '2/foo.jpg') -> list of
    augmented rel paths located under `aug_root`.

    - aug_root: path to the augmented dataset root (class subdirs inside).
    - rels: iterable of original rel paths for *this split* to restrict indexing.
            If None, we index everything we find.
    """
    if rels is not None:
        rel_set = {r.replace("\\", "/") for r in rels}
    else:
        rel_set = None

    idx = defaultdict(list)
    total_files = 0

    # scan aug_root/<class>/*.jpg|*.png
    for cls in sorted(os.listdir(aug_root)):
        cls_dir = os.path.join(aug_root, cls)
        if not os.path.isdir(cls_dir):
            continue

        jpgs = glob.glob(os.path.join(cls_dir, "*.jpg"))
        pngs = glob.glob(os.path.join(cls_dir, "*.png"))
        for fp in jpgs + pngs:
            base = os.path.basename(fp)
            # tie augmented file back to its original filename (strip "__aug..."):
            base_orig = base.split("__aug")[0]  # e.g. foo__aug3.jpg -> foo
            # If extensions were preserved, base_orig still ends with .jpg/.png.
            # If not, we keep it as-is; the rel comparison below is string-based.
            orig_rel = f"{cls}/{base_orig}".replace("\\", "/")

            if rel_set is not None and orig_rel not in rel_set:
                # this augmented sample is not part of the current split's originals
                continue

            aug_rel = f"{cls}/{base}".replace("\\", "/")
            idx[orig_rel].append(aug_rel)
            total_files += 1

    if logger:
        logger.info(f"[preaug] indexed keys={len(idx)} total_aug_files={total_files}")
    return idx



class PairDiskDataset(Dataset):
    """
    pairs: list of (street_rel, sat_rel, label) from ORIGINAL keys
    split: 'train' or 'val'
    policy: 'none' or 'preaug'
    preaug_strategy: 'sample_one' | 'use_all' | 'balanced_sampler'
    """
    def __init__(self, pairs, view,
                 street_root_tr, sat_root_tr,
                 street_root_vl, sat_root_vl,
                 img_size: int, split: str,
                 policy: str, preaug_strategy: str = "sample_one",
                 logger=None):
        """
        Initialize a paired dataset with optional pre-augmented sampling.

        Steps:
        1) Store roots, view, and transform.
        2) Build augmentation indexes when needed.
        3) Expand pairs based on strategy (use_all vs sample_one).
        """
        self.view   = view
        self.split  = split
        self.policy = policy
        self.strategy = preaug_strategy
        self.tf     = _build_eval_tf(img_size)
        self.logger = logger

        self.street_root_tr = street_root_tr
        self.sat_root_tr    = sat_root_tr
        self.street_root_vl = street_root_vl
        self.sat_root_vl    = sat_root_vl

        self.base_pairs = pairs
        self.aug_index_st = None
        self.aug_index_sa = None

        if split == "train" and policy == "preaug":
            st_rels = [p[0] for p in pairs]
            sa_rels = [p[1] for p in pairs]
            self.aug_index_st = _make_aug_index(street_root_tr, st_rels, logger=logger)
            self.aug_index_sa = _make_aug_index(sat_root_tr, sa_rels, logger=logger)


        if split == "train" and policy == "preaug" and self.strategy == "use_all":
            expanded = []
            for st_rel, sa_rel, y in self.base_pairs:
                st_list = self.aug_index_st.get(st_rel, []) or [st_rel]
                sa_list = self.aug_index_sa.get(sa_rel, []) or [sa_rel]
                maxn = max(len(st_list), len(sa_list))
                for i in range(maxn):
                    st_r = st_list[i % len(st_list)]
                    sa_r = sa_list[i % len(sa_list)]
                    expanded.append((st_r, sa_r, y))
            self.pairs = expanded
        else:
            self.pairs = list(self.base_pairs)

        self.labels = [int(y) for _, _, y in self.pairs]

    def __len__(self):
        """Return the number of paired samples in the dataset."""
        return len(self.pairs)

    def _as_fullpath(self, rel: str, use_train_root: bool, is_street: bool):
        """
        Resolve a relative path to the correct street/satellite root.
        """
        root = (self.street_root_tr if is_street else self.sat_root_tr) if use_train_root \
               else (self.street_root_vl if is_street else self.sat_root_vl)
        return os.path.join(root, rel)

    def _load_img(self, fp: str):
        """
        Load an image from disk and apply the evaluation transform.
        """
        img = Image.open(fp).convert("RGB")
        return self.tf(img)

    def __getitem__(self, i):
        """
        Fetch a single sample, returning view-specific tensors and label.

        Steps:
        1) Resolve paired paths (with augmentation strategy if enabled).
        2) Load required image(s).
        3) Return tensors and integer label.
        """
        st_rel, sa_rel, y = self.pairs[i]
        if self.split == "train" and self.policy == "preaug" and self.strategy == "sample_one":
            # we will not use this path now, since strategy is use_all, left here for completeness
            st_cands = self.aug_index_st.get(st_rel, []) or [st_rel]
            sa_cands = self.aug_index_sa.get(sa_rel, []) or [sa_rel]
            st_fp = self._as_fullpath(random.choice(st_cands), True,  True)
            sa_fp = self._as_fullpath(random.choice(sa_cands), True,  False)
        else:
            st_fp = self._as_fullpath(st_rel, self.split == "train", True)
            sa_fp = self._as_fullpath(sa_rel, self.split == "train", False)

        if self.view == "street":
            return self._load_img(st_fp), int(y)
        if self.view == "satellite":
            return self._load_img(sa_fp), int(y)
        return (self._load_img(st_fp), self._load_img(sa_fp)), int(y)

def _make_balanced_sampler(labels):
    """
    Create a weighted sampler to balance class frequencies.

    Steps:
    1) Count samples per class.
    2) Invert counts to get weights.
    3) Build a WeightedRandomSampler.
    """
    labels = torch.as_tensor(labels, dtype=torch.long)
    class_sample_count = torch.bincount(labels).clamp(min=1)
    weights = 1.0 / class_sample_count[labels].float()
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)

def make_loaders_from_roots(train_pairs, test_pairs, view,
                            street_root_tr, sat_root_tr,
                            street_root_vl, sat_root_vl,
                            policy: str, img_size: int,
                            batch_size: int, num_workers: int,
                            preaug_strategy: str = "sample_one",
                            logger=None):
    """
    Build train/val dataloaders for paired street/satellite datasets.

    Steps:
    1) Create PairDiskDataset instances for train/val.
    2) Optionally create a balanced sampler.
    3) Return DataLoader objects.
    """
    tr_ds = PairDiskDataset(train_pairs, view, street_root_tr, sat_root_tr,
                            street_root_vl, sat_root_vl,
                            img_size, split="train", policy=policy,
                            preaug_strategy=preaug_strategy, logger=logger)
    vl_ds = PairDiskDataset(test_pairs,  view, street_root_tr, sat_root_tr,
                            street_root_vl, sat_root_vl,
                            img_size, split="val",   policy=policy,
                            preaug_strategy=preaug_strategy, logger=logger)

    sampler = None
    if policy == "preaug" and preaug_strategy == "balanced_sampler":
        sampler = _make_balanced_sampler(tr_ds.labels)

    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=(sampler is None),
                       sampler=sampler, num_workers=num_workers, pin_memory=True)
    vl_ld = DataLoader(vl_ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True)
    return tr_ld, vl_ld


# ====================================================================
# Ordinal walkability: point-based data (point_labels.csv + folds.csv)
# ====================================================================

N_THRESHOLDS = 4  # P(Y>1) .. P(Y>4) on the 1-5 scale
N_CLASSES = 5
WHITE = 255  # fill value used by the article's geometric augmentations


class _AddToBrightness(A.ImageOnlyTransform):
    """
    Port of imgaug AddToBrightness(add=(-30, 30)): convert to a randomly chosen
    colorspace, add an integer from [-30, 30] to its brightness channel, convert back.
    Colorspaces and channel ids match imgaug's defaults.
    """
    _SPACES = (  # (to, from, brightness channel)
        (cv2.COLOR_RGB2YCrCb, cv2.COLOR_YCrCb2RGB, 0),
        (cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2RGB, 2),
        (cv2.COLOR_RGB2HLS, cv2.COLOR_HLS2RGB, 1),
        (cv2.COLOR_RGB2Lab, cv2.COLOR_Lab2RGB, 0),
        (cv2.COLOR_RGB2Luv, cv2.COLOR_Luv2RGB, 0),
        (cv2.COLOR_RGB2YUV, cv2.COLOR_YUV2RGB, 0),
    )

    def __init__(self, add=(-30, 30), p=1.0):
        super().__init__(p=p)
        self.add = add

    def get_params(self):
        return {"space": self.py_random.randrange(len(self._SPACES)),
                "value": self.py_random.randint(self.add[0], self.add[1])}

    def apply(self, img, space=0, value=0, **params):
        to_code, from_code, ch = self._SPACES[space]
        conv = cv2.cvtColor(img, to_code).astype(np.int16)
        conv[..., ch] = np.clip(conv[..., ch] + value, 0, 255)
        return cv2.cvtColor(conv.astype(np.uint8), from_code)


class _LinearContrast(A.ImageOnlyTransform):
    """
    Port of imgaug LinearContrast(alpha=(0.5, 2.0)): v' = 127 + alpha * (v - 127),
    same alpha for all channels, clipped and truncated to uint8 like imgaug's LUT.
    """

    def __init__(self, alpha=(0.5, 2.0), p=1.0):
        super().__init__(p=p)
        self.alpha = alpha

    def get_params(self):
        return {"alpha": self.py_random.uniform(self.alpha[0], self.alpha[1])}

    def apply(self, img, alpha=1.0, **params):
        table = 127 + np.float32(alpha) * (np.arange(256, dtype=np.float32) - 127)
        table = np.clip(table, 0, 255).astype(np.uint8)
        return cv2.LUT(img, table)


def build_article_augmentation(seed: int) -> "A.Compose":
    """
    The seven single-op augmentations from the article (balance_dataset_single_op.py),
    same ranges, one op drawn uniformly per image. Geometric ops fill with white.
    """
    geo = dict(fill=WHITE, border_mode=cv2.BORDER_CONSTANT, p=1.0)
    ops = [
        _AddToBrightness(add=(-30, 30)),
        _LinearContrast(alpha=(0.5, 2.0)),
        A.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, keep_ratio=False, **geo),
        A.Perspective(scale=(0.01, 0.15), keep_size=True, fit_output=False, **geo),
        A.Affine(rotate=(-25, 25), **geo),
        A.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, **geo),
        A.Affine(shear={"x": (-15, 15), "y": (-15, 15)}, **geo),
    ]
    return A.Compose([A.OneOf(ops, p=1.0)], seed=seed)


def load_points(labels_csv: str, folds_csv: str):
    """
    Join point_labels.csv with folds.csv.

    Returns a list of dicts with: point_id, fold, city, y_cum (4 soft cumulative
    targets), y_soft (5-class soft distribution) and the two ratings (lo, hi)
    recovered from the cumulative targets (two raters per point).
    """
    folds = {}
    with open(folds_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            folds[r["point_id"]] = (int(r["fold"]), r["city"])
    points = []
    with open(labels_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            y_cum = np.array([float(r[f"p_above_{k}"]) for k in range(1, N_THRESHOLDS + 1)], dtype=np.float32)
            # each rater contributes 0.5 to every threshold it exceeds
            lo = 1 + int((y_cum == 1.0).sum())
            hi = 1 + int((y_cum >= 0.5).sum())
            y_soft = np.zeros(N_CLASSES, dtype=np.float32)
            y_soft[lo - 1] += 0.5
            y_soft[hi - 1] += 0.5
            fold, city = folds[r["point_id"]]
            points.append({"point_id": r["point_id"], "fold": fold, "city": city,
                           "y_cum": y_cum, "y_soft": y_soft, "rater_lo": lo, "rater_hi": hi})
    return points


def street_paths(street_dir: str, point_id: str):
    """Return the (front, back) Street View image paths of a point."""
    return (os.path.join(street_dir, f"{point_id}_front.png"),
            os.path.join(street_dir, f"{point_id}_back.png"))


def has_street_images(street_dir: str, point_id: str) -> bool:
    """True when both 180-degree Street View images of a point exist."""
    return all(os.path.isfile(p) for p in street_paths(street_dir, point_id))


def aerial_index(aerial_dir: str):
    """{point_id: path} of the aerial patches (files named P######_lat<lat>_lon<lon>.png)."""
    out = {}
    for name in os.listdir(aerial_dir):
        if name.startswith("P") and name.endswith(".png") and "_lat" in name:
            out[name.split("_")[0]] = os.path.join(aerial_dir, name)
    return out


def split_tiles(img: np.ndarray):
    """Split a 180-degree image (two 90-degree views side by side) into its two square tiles."""
    half = img.shape[1] // 2
    return [img[:, :half], img[:, half:]]


VIEWS_WITH_STREET = ("street", "dual", "late")
VIEWS_WITH_AERIAL = ("aerial", "dual", "late")


class PointViewDataset(Dataset):
    """
    One item per point, depending on the view:
      street       the 4 native 90-degree Street View tiles [4, 3, H, W]
      aerial       the aerial patch as a single tile [1, 3, H, W]
      dual / late  (street tiles, aerial tile)
    plus the point's targets and its index.

    In training with aug_copies > 0 the dataset holds every original once plus
    aug_copies augmented versions (one article op drawn per tile, aerial included).
    """

    def __init__(self, points, street_dir: str, img_size: int, train: bool,
                 aug_copies: int = 1, seed: int = 42, view: str = "street", aerial_paths=None):
        self.points = list(points)
        self.street_dir = street_dir
        self.view = view
        self.aerial_paths = aerial_paths or {}
        self.copies = aug_copies if train else 0
        self.aug = build_article_augmentation(seed) if self.copies else None
        self.to_tensor = _build_eval_tf(img_size)

    def __len__(self):
        return len(self.points) * (1 + self.copies)

    def reseed(self, seed: int):
        """Reseed the augmentation RNG (called once per DataLoader worker)."""
        if self.aug is not None:
            self.aug.set_random_seed(seed)

    def _tile(self, tile: np.ndarray, augment: bool) -> torch.Tensor:
        if augment:
            tile = self.aug(image=np.ascontiguousarray(tile))["image"]
        return self.to_tensor(Image.fromarray(tile))

    def __getitem__(self, i):
        p = self.points[i % len(self.points)]
        augment = i >= len(self.points)
        inputs = []
        if self.view in VIEWS_WITH_STREET:
            tiles = []
            for path in street_paths(self.street_dir, p["point_id"]):
                img = np.asarray(Image.open(path).convert("RGB"))
                tiles += [self._tile(t, augment) for t in split_tiles(img)]
            inputs.append(torch.stack(tiles))
        if self.view in VIEWS_WITH_AERIAL:
            img = np.asarray(Image.open(self.aerial_paths[p["point_id"]]).convert("RGB"))
            inputs.append(self._tile(img, augment).unsqueeze(0))
        x = inputs[0] if len(inputs) == 1 else tuple(inputs)
        return (x,
                torch.from_numpy(p["y_cum"]),
                torch.from_numpy(p["y_soft"]),
                i % len(self.points))


StreetTileDataset = PointViewDataset  # street view is the default


def seed_worker(worker_id: int):
    """
    DataLoader worker_init_fn: derive numpy/random/augmentation seeds from the
    per-worker torch seed, so runs are reproducible given the loader generator.
    """
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, "reseed"):
        info.dataset.reseed(seed)
