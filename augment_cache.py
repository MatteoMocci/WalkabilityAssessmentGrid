"""Disk-backed augmentation cache builder and lazy tensor loader."""

# augment_cache.py (streaming, memory-safe)
import os, json, hashlib
from typing import Dict
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy, RandAugment
from tqdm import tqdm

def _sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

def _ensure_dir(p): os.makedirs(p, exist_ok=True)

def _base_tf(img_size: int):
    return transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

def _policy_tf(name: str, img_size: int):
    base = _base_tf(img_size)
    if name == "auto":
        return transforms.Compose([AutoAugment(policy=AutoAugmentPolicy.IMAGENET), *base.transforms])
    if name == "rand":
        return transforms.Compose([RandAugment(num_ops=2, magnitude=9), *base.transforms])
    raise ValueError("policy must be auto or rand")

def _save_tensor(t: torch.Tensor, path: str):
    # store half precision to cut size by 2; convert back to float32 when loading
    torch.save(t.half(), path)

def _load_tensor(path: str) -> torch.Tensor:
    return torch.load(path).float()

def _write_json(path: str, obj: Dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)

def precompute_global_augments(
    full_pairs,
    street_root,
    sat_root,
    cache_root="./aug_cache",
    logger=None,
    force_rebuild=False,
    autosave_every=200,
    img_size: int = 224,
    max_images_per_view: int | None = None,   # helpful for smoke mode
):
    """
    Stream-save augmented tensors to per-image .pt files. No giant dict in RAM.
    Layout:
      {cache_root}/{policy}/{view}/<sha16>.pt
      {cache_root}/{policy}/index_street.json  (maps original relpath -> sha16)
      {cache_root}/{policy}/index_satellite.json
    """
    _ensure_dir(cache_root)

    st_paths = list(dict.fromkeys(p[0] for p in full_pairs))
    sa_paths = list(dict.fromkeys(p[1] for p in full_pairs))
    if max_images_per_view is not None:
        st_paths = st_paths[:max_images_per_view]
        sa_paths = sa_paths[:max_images_per_view]

    if logger:
        logger.info(f"Augment cache root: {os.path.abspath(cache_root)}")
        logger.info(f"Unique images - street={len(st_paths)} satellite={len(sa_paths)}")

    for policy in ("auto","rand"):
        pol_dir = os.path.join(cache_root, policy)
        st_dir  = os.path.join(pol_dir, "street")
        sa_dir  = os.path.join(pol_dir, "satellite")
        _ensure_dir(st_dir); _ensure_dir(sa_dir)

        st_idx_path = os.path.join(pol_dir, "index_street.json")
        sa_idx_path = os.path.join(pol_dir, "index_satellite.json")
        st_index = {}
        sa_index = {}

        # if index exists and not rebuilding, load it
        if not force_rebuild:
            if os.path.exists(st_idx_path):
                try: st_index = json.load(open(st_idx_path, "r", encoding="utf-8"))
                except Exception: st_index = {}
            if os.path.exists(sa_idx_path):
                try: sa_index = json.load(open(sa_idx_path, "r", encoding="utf-8"))
                except Exception: sa_index = {}

        tf = _policy_tf(policy, img_size)

        # street
        if logger: logger.info(f"[{policy}] streaming street to {st_dir}")
        errors = 0
        for i, p in enumerate(tqdm(st_paths, desc=f"[{policy}] street", unit="img", ncols=90, ascii=True)):
            key = st_index.get(p, _sha(p))
            out_path = os.path.join(st_dir, key + ".pt")
            if force_rebuild and os.path.exists(out_path):
                try: os.remove(out_path)
                except Exception: pass
            if not os.path.exists(out_path):
                try:
                    img = Image.open(os.path.join(street_root, p)).convert("RGB")
                    _save_tensor(tf(img), out_path)
                    st_index[p] = key
                except Exception as e:
                    errors += 1
                    if logger: logger.warning(f"[{policy}] street fail: {p} - {e}")
            if autosave_every and (i+1) % autosave_every == 0:
                _write_json(st_idx_path, st_index)
        _write_json(st_idx_path, st_index)
        if logger: logger.info(f"[{policy}] street done - indexed={len(st_index)} errors={errors}")

        # satellite
        if logger: logger.info(f"[{policy}] streaming satellite to {sa_dir}")
        errors = 0
        for i, p in enumerate(tqdm(sa_paths, desc=f"[{policy}] satellite", unit="img", ncols=90, ascii=True)):
            key = sa_index.get(p, _sha(p))
            out_path = os.path.join(sa_dir, key + ".pt")
            if force_rebuild and os.path.exists(out_path):
                try: os.remove(out_path)
                except Exception: pass
            if not os.path.exists(out_path):
                try:
                    img = Image.open(os.path.join(sat_root, p)).convert("RGB")
                    _save_tensor(tf(img), out_path)
                    sa_index[p] = key
                except Exception as e:
                    errors += 1
                    if logger: logger.warning(f"[{policy}] satellite fail: {p} - {e}")
            if autosave_every and (i+1) % autosave_every == 0:
                _write_json(sa_idx_path, sa_index)
        _write_json(sa_idx_path, sa_index)
        if logger: logger.info(f"[{policy}] satellite done - indexed={len(sa_index)} errors={errors}")

class _LazyCache(dict):
    """
    Dict-like view that loads tensors lazily from disk:
      cache["street"][relpath] -> torch.FloatTensor [3,H,W]
    """
    def __init__(self, index_path: str, data_dir: str):
        super().__init__()
        self._dir = data_dir
        self._idx = json.load(open(index_path, "r", encoding="utf-8"))

    def __contains__(self, k): return k in self._idx
    def __len__(self): return len(self._idx)
    def __getitem__(self, k):
        key = self._idx[k]
        path = os.path.join(self._dir, key + ".pt")
        return _load_tensor(path)

def load_aug_cache(policy: str, cache_root="./aug_cache"):
    pol_dir = os.path.join(cache_root, policy)
    st_idx_path = os.path.join(pol_dir, "index_street.json")
    sa_idx_path = os.path.join(pol_dir, "index_satellite.json")
    st_dir = os.path.join(pol_dir, "street")
    sa_dir = os.path.join(pol_dir, "satellite")
    if not (os.path.exists(st_idx_path) and os.path.exists(sa_idx_path)):
        raise FileNotFoundError(f"Aug cache missing for policy={policy}. Expected {st_idx_path} and {sa_idx_path}")
    return {"street": _LazyCache(st_idx_path, st_dir), "satellite": _LazyCache(sa_idx_path, sa_dir)}
