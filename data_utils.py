"""Dataset and dataloader utilities for paired street/satellite training splits."""

# data_utils.py
import glob
import os
import random
from collections import defaultdict

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
