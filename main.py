"""WalkCNN experiment runner with grouped CV, resume-safe training, and reporting."""

# main.py
from __future__ import annotations
import argparse
import csv
import datetime
import faulthandler
import gc
import glob
import io
import json
import logging
import multiprocessing as mp
import os
import pickle
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import GroupKFold
from transformers import logging as hf_logging

from logging_utils import get_logger
from data_utils import make_loaders_from_roots
from model_utils import load_model
from losses import get_criterion
from metrics_utils import compute_metrics
from dual_encoder import DualFeatureConcat, CombinedFeatureConcat

hf_logging.set_verbosity_error()
faulthandler.enable()  # dumps a traceback to stderr on fatal crashes (no Python exception)

# Use a monotonic clock to avoid false timeouts from wall-clock jumps.
PROGRESS = {"ts": time.monotonic()}


def timestamp():
    return time.strftime("%H:%M:%S")


def note_progress():
    """Mark forward progress for the stall watchdog."""
    PROGRESS["ts"] = time.monotonic()


@contextmanager
def progress_span(tag: str):
    # tell the watchdog we’re starting a long step
    note_progress()
    try:
        yield
    finally:
        # and again when we exit it
        note_progress()
        # optional: a small log to make it visible in stdout
        print(f"{timestamp()}  INFO   [progress] {tag} done", flush=True)


def start_stall_watchdog(
    timeout_sec=900, stacks_path="stacks.txt", check_every=5, logger=None
):
    """
    Kill the process if no note_progress() happens within timeout_sec.
    Uses time.monotonic() so OS/NTP clock changes don't cause false positives.
    """

    def _dump_threads():
        out = []
        for thr in threading.enumerate():
            out.append(f"--- thread: {thr.name} (daemon={thr.daemon}) ---")
            stack = sys._current_frames().get(thr.ident)
            if stack:
                out.extend(traceback.format_stack(stack))
        return "\n".join(out)

    stop_evt = threading.Event()

    def _watch():
        last_log = 0.0
        while not stop_evt.wait(check_every):
            idle = time.monotonic() - PROGRESS["ts"]
            if idle >= timeout_sec:
                # one last snapshot to help debugging
                try:
                    with open(stacks_path, "w", encoding="utf-8") as f:
                        f.write(_dump_threads())
                except Exception:
                    pass
                msg = f"{timestamp()}  ERROR  [watchdog] no progress for {timeout_sec}s. exiting"
                print(msg, flush=True)
                if logger:
                    logger.error(msg)
                # Exit with the code your external supervisor expects.
                sys.exit(99)
            # Optional: sparse debug if we’re past half the timeout
            if idle >= (timeout_sec // 2) and (time.monotonic() - last_log) > 60:
                last_log = time.monotonic()
                print(
                    f"{timestamp()}  INFO   [watchdog] idle={idle:.1f}s (timeout={timeout_sec}s)",
                    flush=True,
                )

    t = threading.Thread(target=_watch, name="stall-watchdog", daemon=True)
    t.start()
    print(f"{timestamp()}  INFO   [watchdog] active timeout={timeout_sec}s", flush=True)
    if logger:
        logger.info(f"[watchdog] active timeout={timeout_sec}s")
    return stop_evt


# ====================================================
# ===== GPU memory helpers (paste once near the imports / small utils) =====


def gpu_mem_gb():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    a = torch.cuda.memory_allocated() / (1024**3)
    r = torch.cuda.memory_reserved() / (1024**3)
    return a, r


def free_cuda(logger=None, tag=""):
    try:
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(0.01)  # give the driver a breath
        a, r = gpu_mem_gb()
        if logger:
            logger.info(
                f"[mem]{(' '+tag) if tag else ''} allocated={a:.2f}GB reserved={r:.2f}GB"
            )
    except Exception:
        pass


# streaming confusion matrix & metrics for eval to avoid retaining tensors
def _update_cm(cm, y_true_cpu, y_pred_cpu, num_classes):
    with torch.no_grad():
        k = num_classes
        idx = y_true_cpu.to(torch.int64) * k + y_pred_cpu.to(torch.int64)
        binc = torch.bincount(idx, minlength=k * k)
        cm += binc.view(k, k)
    return cm


def _metrics_from_cm(cm):
    tp = cm.diag().float()
    total = cm.sum().clamp_min(1).float()
    acc = tp.sum() / total
    fp = cm.sum(0).float() - tp
    fn = cm.sum(1).float() - tp
    denom = (2 * tp + fp + fn).clamp_min(1e-9)
    f1_macro = ((2 * tp) / denom).mean()
    return float(acc), float(f1_macro)


SEED = 42  # used in folds / loaders
# ----------------------------
# Paths & dataset definitions
# ----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_DIR = os.getenv("WALKCNN_BASE_DIR", PROJECT_ROOT)
STREET_ORIG = os.path.join(BASE_DIR, "streetview")
SAT_ORIG = os.path.join(BASE_DIR, "satellite")
STREET_AUG = os.path.join(BASE_DIR, "augmented-streetview")
SAT_AUG = os.path.join(BASE_DIR, "augmented-satellite")
TIMING_LOG = "timing_folds.csv"
SAVE_OPTIM_STATE = (
    True  # set True on Linux; False on Windows to avoid access violations
)
FOLDS_PER_VIEW = {
    "street": 10,
    "satellite": 10,
    "dual": 10,
    "combined": 10,
}
def _atomic_replace(src, dst):
    if os.path.exists(dst):
        os.replace(src, dst)
    else:
        shutil.move(src, dst)


def _write_bytes_atomic(payload_bytes: bytes, path: str):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=d, delete=False) as tmp:
        tmp.write(payload_bytes)
        tmp_path = tmp.name
    _atomic_replace(tmp_path, path)


def _write_json_atomic(path: str, obj: dict):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _done_marker_path(run_key: str) -> str:
    return os.path.join(CHECKPOINT_DIR, f"{run_key}_done.json")


def _save_worker_once(kind: str, payload_bytes: bytes, path: str):
    """
    Child **process** entry (used on non-Windows only).
    We do not reference tmp_path here because it's local to _write_bytes_atomic.
    """
    try:
        _write_bytes_atomic(payload_bytes, path)
        os._exit(0)
    except Exception:
        # best-effort: nothing to clean here (tmp file handled inside _write_bytes_atomic)
        os._exit(1)


def safe_save_bytes_async(
    kind: str, payload_bytes: bytes, path: str, *, timeout: float = 10.0, logger=None
):
    """
    Fire-and-forget save. On Windows, use a daemon **thread** (no re-import of main.py).
    On Unix, use a short-lived child **process** with a timeout.
    Returns True if the save was launched and (on Unix) finished successfully.
    """
    if os.name == "nt":
        # Windows path: no multiprocessing to avoid re-import of main.py
        t = threading.Thread(
            target=_write_bytes_atomic, args=(payload_bytes, path), daemon=True
        )
        t.start()
        if logger:
            logger.info(f"[ckpt] queued {kind} save to {path} (thread)")
        return True
    # Unix-like path: separate process with timeout
    ctx = mp.get_context("spawn")
    p = ctx.Process(
        target=_save_worker_once, args=(kind, payload_bytes, path), daemon=True
    )
    p.start()
    p.join(timeout)
    if p.is_alive():
        try:
            p.kill()
        except Exception:
            pass
        if logger:
            logger.warning(f"[ckpt] save timed out, skipped path={path}")
        return False
    if p.exitcode != 0:
        if logger:
            logger.warning(
                f"[ckpt] save failed in child (exit={p.exitcode}), skipped path={path}"
            )
        return False
    return True


def dump_torch_state_to_bytes(state: dict) -> bytes:
    """
    Serialize a CPU-only state dict to bytes with the legacy pickler.
    If torch’s pickler trips on Windows, fall back to a tiny JSON meta.
    """
    def _assert_cpu_only(x):
        if torch.is_tensor(x):
            assert not x.is_cuda
        elif isinstance(x, dict):
            for v in x.values():
                _assert_cpu_only(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                _assert_cpu_only(v)

    try:
        _assert_cpu_only(state)
        buf = io.BytesIO()
        torch.save(state, buf, _use_new_zipfile_serialization=False)
        return buf.getvalue()
    except Exception:
        minimal = {
            k: v
            for k, v in state.items()
            if k in ("epoch", "best_val", "best_epoch", "best_metrics")
        }
        return json.dumps(minimal).encode("utf-8")


def save_best_async(model, path: str, logger=None):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload = dump_torch_state_to_bytes(cpu_state)
    ok = safe_save_bytes_async("best", payload, path, timeout=15.0, logger=logger)
    if ok and logger:
        logger.info(f"[ckpt] saved best to {path} (async)")


def save_state_async(
    epoch, model, optimizer, meta: dict, path: str, *, save_optimizer: bool, logger=None
):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    state = {
        "epoch": int(epoch),
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "best_val": float(meta.get("best_val", float("inf"))),
        "best_epoch": int(meta.get("best_epoch", -1)),
        "best_metrics": meta.get("best_metrics", {}),
    }
    if save_optimizer:
        try:
            state["optimizer"] = {
                k: (v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in optimizer.state_dict().items()
            }
        except Exception:
            if logger:
                logger.warning(
                    "[ckpt] optimizer state skipped due to serialization issue"
                )
    payload = dump_torch_state_to_bytes(state)
    ok = safe_save_bytes_async("state", payload, path, timeout=15.0, logger=logger)
    if not ok and logger:
        logger.warning(f"[ckpt] state save skipped for {path}")


# ---------------------------------------------------------
def _forward_for_view(model, batch, device, view: str):
    """Route inputs correctly for single-view vs dual/combined models."""
    st = batch["street"].to(device, non_blocking=True)
    sa = batch["sat"].to(device, non_blocking=True)
    # Dual / Combined wrappers take two tensors
    if view in ("dual", "combined") or getattr(model, "is_pair_input", False):
        # accept either positional (st, sa) or keyword (street=, sat=)
        try:
            return model(st, sa)
        except TypeError:
            return model(street=st, sat=sa)
    # Single-view
    if view == "street":
        return model(st)
    else:  # 'satellite'
        return model(sa)


# main.py
def build_model_for_view(
    view: str, model_name: str, pretrain: str, num_classes: int, device
):
    def _single_backbone(*, replace_classifier: bool):
        # Decide replacement here
        cls = num_classes if replace_classifier else None
        m = load_model(
            model_name, pretrain, cls
        )  # head replaced INSIDE load_model only if cls is not None
        if hasattr(m, "aux_logits"):
            m.aux_logits = False
        m = m.to(device)  # move AFTER any replacement
        m.is_pair_input = False
        return m

    if view in ("street", "satellite"):
        # single view: we need a 5-class head
        return _single_backbone(replace_classifier=True)
    if view == "dual":
        st = _single_backbone(replace_classifier=False)
        sa = _single_backbone(replace_classifier=False)
        m = DualFeatureConcat(st, sa, num_classes=num_classes).to(device)
        m.is_pair_input = True
        return m
    if view == "combined":
        st = _single_backbone(replace_classifier=False)
        sa = _single_backbone(replace_classifier=False)
        m = CombinedFeatureConcat(st, sa, num_classes=num_classes).to(device)
        m.is_pair_input = True
        return m


def aggregate_metrics(metric_dicts):
    """
    Average a list of metric dicts -> single dict. Missing keys default to 0.
    """
    if not metric_dicts:
        return {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "mean_absolute_error": 0,
            "one_off_accuracy": 0,
        }
    keys = set().union(*[d.keys() for d in metric_dicts])
    out = {}
    for k in keys:
        vals = [float(d.get(k, 0.0)) for d in metric_dicts]
        out[k] = sum(vals) / max(1, len(vals))
    return out


@torch.no_grad()
def compute_class_weights_from_loader(loader, num_classes=None, eps=1e-8):
    """
    Inverse-frequency class weights from a training loader.
    Tries dataset.targets / .samples first; falls back to one pass over labels.
    """
    ds = getattr(loader, "dataset", None)
    labels = None
    # Fast paths
    if ds is not None and hasattr(ds, "targets"):
        labels = list(map(int, ds.targets))
    elif ds is not None and hasattr(ds, "samples"):
        labels = [int(y) for _, y in ds.samples]
    # Fallback: one pass over loader (labels only)
    if labels is None:
        labels = []
        for batch in loader:
            if isinstance(batch[0], (tuple, list)) and len(batch[0]) == 2:
                _, lab = batch
            else:
                _, lab = batch
            labels.extend(lab.cpu().numpy().tolist())
    if num_classes is None:
        num_classes = max(labels) + 1 if labels else 1
    counts = torch.bincount(
        torch.tensor(labels, dtype=torch.long), minlength=num_classes
    ).float()
    # classic inverse freq
    weights = counts.sum() / (num_classes * (counts + eps))
    return weights










# --------------------------------------------
# ===== Fold inspection helpers (no training) =====








def is_transformer(obj) -> bool:
    name = obj.lower() if isinstance(obj, str) else type(obj).__name__.lower()
    tf_markers = (
        "vit",
        "swin",
        "beit",
        "maxvit",
        "deit",
        "twins",
        "xcit",
        "cait",
        "mixer",
        "tnt",
    )
    return any(tok in name for tok in tf_markers)


def is_cnn(obj) -> bool:
    return not is_transformer(obj)


def pick_epochs_for_model(model_or_name) -> int:
    return 30 if is_cnn(model_or_name) else 20




PRED_DIR = "pred_reports"
os.makedirs(PRED_DIR, exist_ok=True)










# ----------------------------
# Grid config
# ----------------------------
VIEWS = ["street", "satellite", "dual", "combined"]
MODELS = ["alexnet", "vgg16", "googlenet", "vit_base", "deit_base", "swin_base"]
PRETRAINING_BY_MODEL = {m: ["imagenet1k"] for m in MODELS}
LOSSES = ["WCE"]  # restrict grid to WCE-only rerun
AUGS_FOR_LOSS = {
    "CE": ["preaug"],  # only augmented
    "SCE": ["preaug"],  # only augmented
    "WCE": ["preaug"],  # rerun WCE using augmented dataset
}
AUG_POLICIES = ["none", "preaug"]
PREAUG_STRATEGY = "use_all"  # use all augmented variants for each training image
# ----------------------------
# Run mode (smoke vs full)
# ----------------------------
RUN_MODE = "full"  # change to "full" for your real grid
# Resume & skip controls
RESUME_RUNS = True  # if True, resume a run from saved state if found
START_AT_RUN = None  # e.g. "street_resnet50_imagenet1k_CE_preaug" to skip earlier runs
# Early stopping controls
EARLY_MONITOR = "f1"  # monitor F1 for early stopping
EARLY_MODE = "max"  # maximize F1
EARLY_MIN_DELTA = 1e-4  # require improvement by at least this margin
EARLY_PATIENCE = 5  # stop after this many epochs without improvement
CHECKPOINT_DIR = "checkpoints"
RUN_CFG = {
    "smoke": {
        "IMG_SIZE": 128,
        "BATCH_SIZE": 8,
        "EPOCHS": 3,
        "N_SPLITS": 10,
        "TRAIN_BATCHES": 1,  # limit steps for speed
        "VAL_BATCHES": 1,
    },
    "full": {
        "IMG_SIZE": 224,
        "BATCH_SIZE": None,  # decided per model
        "EPOCHS": 10,
        "N_SPLITS": 10,
        "TRAIN_BATCHES": None,
        "VAL_BATCHES": None,
    },
    "timing": {
        "IMG_SIZE": 224,
        "BATCH_SIZE": None,
        "EPOCHS": 10,  # one epoch is enough for timing per fold
        "N_SPLITS": 10,
        "TRAIN_BATCHES": None,
        "VAL_BATCHES": None,
    },
}
RUN_CFG["train"] = {
    "IMG_SIZE": 224,  # 224 for most transformer backbones; CNNs will override to base if needed
    "BATCH_SIZE": None,  # auto-picked per model
    "EPOCHS": 10,  # ignored in train-mode; we set per-model below (30/20)
    "N_SPLITS": 10,  # we’ll use only ONE split as val; the others are train
    "TRAIN_BATCHES": None,
    "VAL_BATCHES": None,
}
TIMING_CONFIG = {
    "street": ("resnet50", "imagenet1k", "CE", "none"),
    "satellite": ("resnet50", "imagenet1k", "CE", "none"),
    "dual": ("resnet50", "imagenet1k", "CE", "none"),
    "combined": ("resnet50", "imagenet1k", "CE", "none"),
}
OUT_CV_FILE = f"cv_summary_{RUN_MODE}.csv"
OUT_FOLD_FILE = f"fold_metrics_{RUN_MODE}.csv"
# Optional: resume from a given run key (e.g., "street_vgg16_imagenet1k_CE_preaug")
SKIP_TO_RUN: Optional[str] = None
# ----------------------------
# Training hyperparams
# ----------------------------
NUM_WORKERS = 0
CNN_LR, CNN_MOMENTUM, CNN_WD = 1e-3, 0.9, 1e-4
TF_LR, TF_WD = 5e-5, 0.01
TRANSFORMER_HINTS = ("vit", "deit", "swin", "beit")
# --- robust resume controls ---
RUN_REGISTRY_BASENAME = "run_registry"  # persistent status prefix; suffixed by RUN_MODE
RUN_REGISTRY = f"{RUN_REGISTRY_BASENAME}_{RUN_MODE}.json"
AUTO_RESUME = True  # auto-pick the first unfinished run at startup
START_AT_RUN = None  # manual override: e.g. "street_resnet50_imagenet1k_CE_preaug"
FORCE_RERUN_FOLDS = True  # set True to retrain even if *_done.json exists
# heartbeat writes a small file every N seconds so you can see it's alive
HEARTBEAT_FILE = "heartbeat.txt"
HEARTBEAT_SECS = 60

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _uses_preaug_in_current_config() -> bool:
    """Return True when the selected run config needs augmented datasets."""
    if RUN_MODE == "train":
        # train mode always runs both none + preaug
        return True
    if RUN_MODE == "timing":
        return any(cfg[3] == "preaug" for cfg in TIMING_CONFIG.values())
    return any("preaug" in AUGS_FOR_LOSS.get(loss_name, []) for loss_name in LOSSES)


def _inspect_dataset_root(path: str) -> dict:
    """
    Summarize a dataset root: existence, numeric class dirs, and image count.
    Expected layout is: root/<class_id>/<image_file>.
    """
    info = {
        "path": path,
        "exists": os.path.isdir(path),
        "class_dirs": [],
        "image_count": 0,
    }
    if not info["exists"]:
        return info
    for d in sorted(os.listdir(path)):
        cls_dir = os.path.join(path, d)
        if not os.path.isdir(cls_dir):
            continue
        if not d.isdigit():
            continue
        info["class_dirs"].append(d)
        try:
            info["image_count"] += sum(
                1 for fn in os.listdir(cls_dir) if fn.lower().endswith(IMAGE_EXTS)
            )
        except OSError:
            pass
    return info


def validate_data_roots_or_exit(logger) -> None:
    """
    Check dataset folders early and print actionable setup hints on failure.
    """
    need_preaug = _uses_preaug_in_current_config()
    checks = [
        ("street originals", STREET_ORIG, True),
        ("satellite originals", SAT_ORIG, True),
        ("street augmented", STREET_AUG, need_preaug),
        ("satellite augmented", SAT_AUG, need_preaug),
    ]
    problems = []

    logger.info("[data] base_dir=%s", BASE_DIR)
    logger.info("[data] preaug_required=%s", need_preaug)
    for label, path, required in checks:
        info = _inspect_dataset_root(path)
        logger.info(
            "[data] %-18s required=%s exists=%s classes=%d images=%d path=%s",
            label,
            required,
            info["exists"],
            len(info["class_dirs"]),
            info["image_count"],
            path,
        )
        if required and not info["exists"]:
            problems.append(f"- Missing required folder: {path}")
        elif required and info["exists"] and info["image_count"] == 0:
            problems.append(
                f"- Folder is present but no images were found in numeric class dirs: {path}"
            )

    if not problems:
        return

    print("\n[setup] Dataset configuration is incomplete.", flush=True)
    for p in problems:
        print(p, flush=True)
    print("\n[setup] Expected folder layout:", flush=True)
    print(f"  {BASE_DIR}/streetview/<class_id>/*.jpg", flush=True)
    print(f"  {BASE_DIR}/satellite/<class_id>/*.jpg", flush=True)
    if need_preaug:
        print(f"  {BASE_DIR}/augmented-streetview/<class_id>/*.jpg", flush=True)
        print(f"  {BASE_DIR}/augmented-satellite/<class_id>/*.jpg", flush=True)
    print(
        "\n[setup] Example class folders: 0, 1, 2, 3, 4 (numeric folder names).",
        flush=True,
    )
    print(
        "[setup] To change dataset root, set WALKCNN_BASE_DIR before running.",
        flush=True,
    )
    if os.name == "nt":
        print(
            r'[setup] PowerShell example: $env:WALKCNN_BASE_DIR = "C:\path\to\data_root"',
            flush=True,
        )
    else:
        print(
            '[setup] Bash example: export WALKCNN_BASE_DIR="/path/to/data_root"',
            flush=True,
        )
    raise SystemExit(2)


def _load_registry(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_registry(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def mark_run_status(
    reg: dict, run_key: str, status: str, extra: dict | None = None
) -> dict:
    entry = reg.get(run_key, {})
    entry["status"] = status  # "planned" | "running" | "done" | "failed"
    if extra:
        entry.update(extra)
    entry["ts"] = datetime.now().isoformat(timespec="seconds")
    reg[run_key] = entry
    return reg


def reset_registry_status(
    reg: dict,
    run_keys: list[str],
    *,
    clear_done_markers: bool = True,
    logger=None,
) -> tuple[dict, bool]:
    changed = False
    for rk in run_keys:
        entry = reg.get(rk)
        if not entry:
            continue
        entry = {
            k: v
            for k, v in entry.items()
            if k not in {"error", "best_epoch", "best_f1"}
        }
        entry["status"] = "planned"
        entry["ts"] = datetime.now().isoformat(timespec="seconds")
        reg[rk] = entry
        changed = True
        if clear_done_markers:
            marker_paths = [
                _done_marker_path(rk),
            ]
            fold_pattern = os.path.join(CHECKPOINT_DIR, f"{rk}_fold*_done.json")
            marker_paths.extend(glob.glob(fold_pattern))
            for mp in marker_paths:
                try:
                    if os.path.exists(mp):
                        os.remove(mp)
                        changed = True
                        if logger:
                            logger.info(f"[reset] removed done marker {mp}")
                except OSError as e:
                    if logger:
                        logger.warning(f"[reset] could not remove {mp}: {e}")
    return reg, changed


def first_unfinished(planned: list[str], reg: dict) -> str | None:
    for rk in planned:
        st = reg.get(rk, {}).get("status", "planned")
        if st not in ("done",):
            return rk
    return None


def start_heartbeat(path: str, interval_sec: int, logger):
    stop_evt = threading.Event()

    def _beat():
        while not stop_evt.wait(interval_sec):
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(datetime.now().isoformat(timespec="seconds"))
            except Exception as e:
                logger.warning(f"[heartbeat] write failed: {e}")

    t = threading.Thread(target=_beat, daemon=True)
    t.start()
    logger.info(f"[heartbeat] writing to {path} every {interval_sec}s")
    return stop_evt


def build_aug_catalog(root_dir: str):
    """
    Index augmented images by (label, rounded_latlon_key) -> [relative_paths].
    Uses the same lat/lon parsing used for folds.
    """
    catalog = defaultdict(list)
    if not os.path.isdir(root_dir):
        return catalog
    for cls in sorted(os.listdir(root_dir)):
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        try:
            y = int(cls)
        except Exception:
            continue
        for fn in sorted(os.listdir(cls_dir)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            rel = os.path.join(cls, fn)
            coords = _coords_from_name(rel)
            if coords is None:
                continue
            k = _key_latlon(*coords)
            catalog[(y, k)].append(rel)
    return catalog




def pick_batch_size(model_name: str) -> int:
    name = model_name.lower()
    if any(h in name for h in TRANSFORMER_HINTS):
        return 16
    return 64


def pick_lr_and_wd(model_name: str) -> Tuple[float, float]:
    # use our local string-based is_cnn()
    return (CNN_LR, CNN_WD) if is_cnn(model_name) else (TF_LR, TF_WD)


def pick_img_size_for_model(model_name: str, base: int) -> int:
    # many transformers expect 224
    return 224 if any(h in model_name.lower() for h in TRANSFORMER_HINTS) else base


# ----------------------------
# Folds built on coordinates
# ----------------------------
FOLDS_CACHE = f"folds_cache_{RUN_MODE}.pkl"
TOL_DEG = 5e-5  # ~5e-5 deg ~ 5-6 meters; tweak if needed
Float = float
Pair = Tuple[str, str, int]  # (street_rel, sat_rel, label)


def _list_images(root: str):
    """Return list of (label_int, rel_path) for images under class dirs 0..N."""
    out = []
    if not os.path.isdir(root):
        return out
    for cls in sorted(os.listdir(root)):  # stable class order
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):  # <-- skip files like .zip
            continue
        try:
            y = int(cls)  # only directories named as integers are classes
        except Exception:
            continue
        for fn in sorted(os.listdir(cls_dir)):  # stable file order
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                out.append((y, os.path.join(cls, fn)))
    return out


_float_re = re.compile(r"-?\d+\.\d+")


def _coords_from_name(name: str) -> Optional[Tuple[Float, Float]]:
    # filenames like: id_lat_lon_*.jpg  (e.g., 2060.0_40.590713_8.349011_1.jpg)
    nums = [float(x) for x in _float_re.findall(os.path.basename(name))]
    if len(nums) >= 3:
        lat, lon = nums[1], nums[2]
        return (lat, lon)
    if len(nums) >= 2:
        # fallback: assume first two are lat/lon (rare)
        return (nums[0], nums[1])
    return None


def _key_latlon(lat: Float, lon: Float) -> Tuple[int, int]:
    return (int(round(lat / TOL_DEG)), int(round(lon / TOL_DEG)))


def ensure_pairs_and_groups(
    street_root: str, sat_root: str, logger
) -> Tuple[List[Pair], Dict[str, str], List[str]]:
    """Return:
    - pairs: [(street_rel, sat_rel, y)]
    - group_id_by_pair_key: maps 'street_rel' (originals) to group_id
    - groups: list of unique group ids (for GroupKFold)
    """
    logger.info(f"street_root={street_root}  exists={os.path.isdir(street_root)}")
    logger.info(f"sat_root   ={sat_root}     exists={os.path.isdir(sat_root)}")
    st_items = _list_images(street_root)
    sa_items = _list_images(sat_root)
    # index satellite by rounded coords
    sat_by_key = defaultdict(list)
    for k in sat_by_key:
        sat_by_key[k].sort(key=lambda t: t[1])  # sort by satellite relative path
    for y, rel in sa_items:
        c = _coords_from_name(rel)
        if c is None:
            continue
        sat_by_key[_key_latlon(*c)].append((y, rel, c))
    pairs: List[Pair] = []
    group_by_street: Dict[str, str] = {}
    for y, st_rel in st_items:
        c = _coords_from_name(st_rel)
        if c is None:
            continue
        key = _key_latlon(*c)
        if key not in sat_by_key:
            continue
        # pick first sat with same class if possible, else first
        sats_here = sat_by_key[key]
        match = None
        for ys, sa_rel, _c in sats_here:
            if ys == y:
                match = (ys, sa_rel)
                break
        if match is None:
            ys, sa_rel, _c = sats_here[0]
            match = (y, sa_rel)
        _, sa_rel = match
        pairs.append((st_rel, sa_rel, y))
        group_by_street[st_rel] = f"{key[0]}_{key[1]}"
    if not pairs:
        raise RuntimeError(
            "No matching street and satellite files found to build folds"
        )
    groups = sorted(set(group_by_street.values()))
    logger.info(
        f"Built {len(pairs)} matched pairs across {len(groups)} coordinate groups"
    )
    return pairs, group_by_street, groups


def kfold_on_groups(groups: List[str], n_splits: int, logger):
    gkf = GroupKFold(n_splits=n_splits)
    group_indices = np.arange(len(groups))
    # identity mapping group idx -> group id string
    folds = []
    for tr_idx, vl_idx in gkf.split(group_indices, groups=group_indices):
        folds.append(
            {
                "train_groups": [groups[i] for i in tr_idx],
                "test_groups": [groups[i] for i in vl_idx],
            }
        )
    logger.info(f"Prepared {len(folds)} folds (GroupKFold)")
    return folds


# ----------------------------
# Training helpers
# ----------------------------




# --- drop-in replacements in main.py ---
@torch.no_grad()
def evaluate_one_epoch(
    model,
    loader,
    device,
    criterion,
    *,
    view: str,
    num_classes: int,
    limit_batches: int | None = None,
    logger=None,
):
    model.eval()
    cm = torch.zeros(num_classes, num_classes, device="cpu")
    running_loss = 0.0
    n_seen = 0
    start_alloc, start_res = gpu_mem_gb()
    with torch.inference_mode():
        for i, batch in enumerate(loader, start=1):
            note_progress()
            if limit_batches is not None and i > limit_batches:
                break
            # forward (same routing as train)
            st = sa = x = labels = None
            if isinstance(batch, dict):
                labels = batch["label"].to(device, non_blocking=True)
                st = batch.get("street")
                sa = batch.get("satellite") or batch.get("sat")
                x = batch.get("image")
                if st is not None:
                    st = st.to(device, non_blocking=True)
                if sa is not None:
                    sa = sa.to(device, non_blocking=True)
                if x is not None:
                    x = x.to(device, non_blocking=True)
            elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                xs, labels = batch
                labels = labels.to(device, non_blocking=True)
                if isinstance(xs, (tuple, list)) and len(xs) == 2:
                    st, sa = xs
                    st = st.to(device, non_blocking=True)
                    sa = sa.to(device, non_blocking=True)
                else:
                    x = xs.to(device, non_blocking=True)
            pair = (view in ("dual", "combined")) or bool(
                getattr(model, "is_pair_input", False)
            )
            if pair:
                try:
                    logits = model(street=st, sat=sa)
                except TypeError:
                    logits = model(st, sa)
            else:
                if x is None:
                    x = st if st is not None else sa
                logits = model(x)
            loss = criterion(logits, labels)
            # move small things to CPU, update CM, drop tensors
            y_pred = logits.argmax(dim=1).to("cpu")
            y_true = labels.to("cpu")
            cm = _update_cm(cm, y_true, y_pred, num_classes)
            bs = y_true.size(0)
            running_loss += float(loss) * bs
            n_seen += bs
            del loss, logits, labels, st, sa, x, y_pred, y_true
            if i % 50 == 0:
                torch.cuda.empty_cache()
    val_acc, val_f1 = _metrics_from_cm(cm)
    end_alloc, end_res = gpu_mem_gb()
    if logger:
        logger.info(
            f"[mem] val epoch drift alloc={end_alloc-start_alloc:+.2f}GB res={end_res-start_res:+.2f}GB"
        )
    return running_loss / max(1, n_seen), val_acc, val_f1


@torch.no_grad()
def eval_model(
    model,
    loader,
    device,
    *,
    view: str | None = None,
    limit_batches: int | None = None,
    logger=None,
    progress_every: int = 100,
):
    """
    Lightweight evaluation that returns (preds, labels) without computing loss.
    - Use this for fold checks, timing runs, or after-training reports.
    - Supports street / satellite / dual / combined.
    """
    model.eval()
    # Try to infer view from the dataset if not given
    if view is None:
        try:
            view = getattr(loader.dataset, "view", "street")
        except Exception:
            view = "street"
    preds, refs = [], []
    for i, batch in enumerate(loader, start=1):
        note_progress()
        if limit_batches is not None and i > limit_batches:
            break
        if isinstance(batch, dict):
            labels = batch["label"].to(device, non_blocking=True)
            logits = _forward_for_view(model, batch, device, view)
        else:
            if isinstance(batch[0], (tuple, list)) and len(batch[0]) == 2:
                (st, sa), labels = batch
                st = st.to(device, non_blocking=True)
                sa = sa.to(device, non_blocking=True)
                logits = (
                    model(street=st, sat=sa)
                    if view in ("dual", "combined")
                    else model(st if view == "street" else sa)
                )
            else:
                images, labels = batch
                images = images.to(device, non_blocking=True)
                logits = model(images)
        if hasattr(logits, "logits"):
            logits = logits.logits
        elif isinstance(logits, dict) and "logits" in logits:
            logits = logits["logits"]
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
        refs.extend(labels.cpu().numpy().tolist())
        if logger and (i % progress_every == 0):
            logger.info(f"[eval] step {i}")
    return np.array(preds), np.array(refs)


def train_one_epoch(
    model,
    loader,
    device,
    criterion,
    optimizer,
    *,
    view: str,
    limit_batches: int | None = None,
    logger=None,
    scaler=None,
):
    model.train()
    running_loss = 0.0
    n_seen = 0
    start_alloc, start_res = gpu_mem_gb()

    # inner forward that also moves tensors to device
    def _forward_for_batch(model, batch, view):
        st = sa = x = labels = None
        if isinstance(batch, dict):
            labels = batch["label"].to(device, non_blocking=True)
            st = batch.get("street")
            sa = batch.get("satellite") or batch.get("sat")
            x = batch.get("image")
            if st is not None:
                st = st.to(device, non_blocking=True)
            if sa is not None:
                sa = sa.to(device, non_blocking=True)
            if x is not None:
                x = x.to(device, non_blocking=True)
        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
            xs, labels = batch
            labels = labels.to(device, non_blocking=True)
            if isinstance(xs, (tuple, list)) and len(xs) == 2:
                st, sa = xs
                st = st.to(device, non_blocking=True)
                sa = sa.to(device, non_blocking=True)
            else:
                x = xs.to(device, non_blocking=True)
        else:
            raise TypeError(f"Unexpected batch structure: {type(batch)}")
        pair = (view in ("dual", "combined")) or bool(
            getattr(model, "is_pair_input", False)
        )
        if pair:
            try:
                logits = model(street=st, sat=sa)
            except TypeError:
                logits = model(st, sa)
        else:
            if x is None:
                x = st if st is not None else sa
            logits = model(x)
        return logits, labels, st, sa, x

    for i, batch in enumerate(loader, start=1):
        note_progress()
        if limit_batches is not None and i > limit_batches:
            break
        optimizer.zero_grad(set_to_none=True)
        # AMP optional
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits, labels, st, sa, x = _forward_for_batch(model, batch, view)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, labels, st, sa, x = _forward_for_batch(model, batch, view)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        bs = labels.size(0)
        running_loss += float(loss.detach()) * bs
        n_seen += bs
        # aggressively drop references so CUDA can free
        del loss, logits, labels, st, sa, x
        if i % 50 == 0:
            torch.cuda.empty_cache()
    end_alloc, end_res = gpu_mem_gb()
    if logger:
        logger.info(
            f"[mem] train epoch drift alloc={end_alloc-start_alloc:+.2f}GB res={end_res-start_res:+.2f}GB"
        )
    return running_loss / max(1, n_seen)


def train_with_early_stopping(
    model,
    tr_loader,
    vl_loader,
    device,
    *,
    criterion,
    optimizer,
    epochs,
    patience,
    logger,
    view,
    train_limit=None,
    val_limit=None,
    run_key=None,
    monitor="f1",
    mode="max",
    min_delta=1e-4,
    resume=True,
    save_dir="checkpoints",
    num_classes=None,
):
    """
    Early stopping on a metric (default F1). Saves:
      - {run_key}_best.pt   -> best model weights
      - {run_key}_state.pt  -> last epoch state for resume
    Returns: (best_metrics: dict, best_epoch: int)
    """
    os.makedirs(save_dir, exist_ok=True)
    assert mode in ("max", "min")
    better = (
        (lambda cur, best: cur > best + min_delta)
        if mode == "max"
        else (lambda cur, best: cur < best - min_delta)
    )
    best_val = -float("inf") if mode == "max" else float("inf")
    best_metrics = None
    best_epoch = 0
    bad = 0
    start_epoch = 1
    state_path = os.path.join(save_dir, f"{run_key}_state.pt") if run_key else None
    best_path = os.path.join(save_dir, f"{run_key}_best.pt") if run_key else None
    # ---- resume if possible ----
    if resume and state_path and os.path.exists(state_path):
        try:
            state = torch.load(state_path, map_location="cpu")
            model.load_state_dict(state["model"])
            opt_state = state.get("optimizer")
            if opt_state:
                optimizer.load_state_dict(opt_state)
            start_epoch = int(state.get("epoch", 0)) + 1
            best_val = float(state.get("best_val", best_val))
            bad = int(state.get("bad", 0))
            best_epoch = int(state.get("best_epoch", 0))
            best_metrics = state.get("best_metrics", None)
            if logger:
                logger.info(
                    f"[resume] {run_key} from epoch {start_epoch} (best {monitor}={best_val:.4f} @epoch {best_epoch})"
                )
        except Exception as e:
            if logger:
                logger.warning(f"[resume] failed to load state: {e}")
    # ---- train loop ----
    for ep in range(start_epoch, epochs + 1):
        if logger:
            logger.info(f"epoch {ep}/{epochs}")
        # train
        tr_loss = train_one_epoch(
            model,
            tr_loader,
            device,
            criterion,
            optimizer,
            view=view,
            limit_batches=train_limit,
            logger=logger,
        )
        free_cuda(logger, tag=f"after-train ep {ep}")
        # validate (streaming, low-memory)
        val_loss, val_acc, val_f1 = evaluate_one_epoch(
            model,
            vl_loader,
            device,
            criterion,
            view=getattr(vl_loader.dataset, "view", "street"),
            num_classes=num_classes,
            limit_batches=(
                val_limit if val_limit is not None else RUN_CFG[RUN_MODE]["VAL_BATCHES"]
            ),
            logger=logger,
        )
        free_cuda(logger, tag=f"after-val ep {ep}")
        # metrics dict + monitor value
        m = {"loss": float(val_loss), "accuracy": float(val_acc), "f1": float(val_f1)}
        cur = float(m.get(monitor, 0.0))
        if logger:
            logger.info(
                f"val  loss={val_loss:.4f}  accuracy={val_acc:.4f}  f1={val_f1:.4f}"
            )
            note_progress()
            logger.info("[ckpt] begin post-val")
        # improvement?
        if best_metrics is None or better(cur, best_val):
            best_val = cur
            best_metrics = m
            best_epoch = ep
            bad = 0
            if best_path:
                note_progress()
                try:
                    save_best_async(model, best_path, logger=logger)
                except Exception as e:
                    logger.warning(f"[ckpt] save_best_async skipped: {e}")
        else:
            bad += 1
        # save state every epoch (async child, non-fatal)
        if state_path:
            note_progress()
            try:
                save_state_async(
                    ep,
                    model,
                    optimizer,
                    {
                        "best_val": best_val,
                        "best_epoch": best_epoch,
                        "best_metrics": best_metrics or {},
                    },
                    state_path,
                    save_optimizer=SAVE_OPTIM_STATE,
                    logger=logger,
                )
            except Exception as e:
                logger.warning(f"[ckpt] save_state_async skipped: {e}")
        # early stop?
        if bad >= patience:
            if logger:
                logger.info(
                    f"[early stopping] no improvement in {patience} epochs "
                    f"(best {monitor}={best_val:.4f} @epoch {best_epoch})"
                )
            break
        reason = "early_stop" if bad >= patience else "finished"
        if run_key:
            try:
                _write_json_atomic(
                    _done_marker_path(run_key),
                    {
                        "run_key": run_key,
                        "reason": reason,
                        "best_epoch": int(best_epoch),
                        "best_val": float(best_val),
                        "ts": datetime.now().isoformat(timespec="seconds"),
                    },
                )
            except Exception as e:
                if logger:
                    logger.warning(f"[ckpt] could not write done marker: {e}")
    if best_metrics is None:
        best_metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_absolute_error": 0.0,
            "one_off_accuracy": 0.0,
        }
    return best_metrics, best_epoch


def build_loaders_with_guard(
    train_pairs,
    test_pairs,
    view_str,
    street_tr,
    sat_tr,
    street_vl,
    sat_vl,
    aug_name,
    img_size,
    batch_size,
    logger,
    *,
    preaug_strategy,
    num_workers,
):
    """Build loaders with timing, size logs, and graceful fallback."""
    t0 = time.time()
    logger.info(
        f"[loader] building with policy={aug_name} strategy={preaug_strategy} "
        f"img={img_size} bs={batch_size} workers={num_workers}"
    )
    try:
        tr_loader, vl_loader = make_loaders_from_roots(
            train_pairs,
            test_pairs,
            view_str,
            street_root_tr=street_tr,
            sat_root_tr=sat_tr,
            street_root_vl=street_vl,
            sat_root_vl=sat_vl,
            policy=aug_name,
            img_size=img_size,
            batch_size=batch_size,
            num_workers=num_workers,
            preaug_strategy=preaug_strategy,
            logger=logger,
        )
    except BaseException as e:
        logger.exception(f"[loader] failed to build loaders: {e}")
        raise
    # size logs (these help confirm we got past construction)
    try:
        tr_len = len(getattr(tr_loader, "dataset", []))
    except Exception:
        tr_len = -1
    try:
        vl_len = len(getattr(vl_loader, "dataset", []))
    except Exception:
        vl_len = -1
    try:
        tr_batches = len(tr_loader)
    except Exception:
        tr_batches = -1
    try:
        vl_batches_len = len(vl_loader)
    except Exception:
        vl_batches_len = -1
    logger.info(
        f"[loader] built in {time.time()-t0:.1f}s  "
        f"train_pairs={len(train_pairs)} test_pairs={len(test_pairs)} "
        f"tr_dataset={tr_len} vl_dataset={vl_len} "
        f"tr_batches={tr_batches} vl_batches={vl_batches_len}"
    )
    return tr_loader, vl_loader


# ----------------------------
# Main
# ----------------------------









def main(*, reset_progress: bool = False):
    logger = get_logger()
    view_fold_times = defaultdict(list)  # view -> list of fold durations in seconds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = RUN_CFG[RUN_MODE]
    base_img_size = cfg["IMG_SIZE"]
    base_epochs = cfg["EPOCHS"]
    base_bs = cfg["BATCH_SIZE"]
    n_splits = cfg["N_SPLITS"]
    logger.info(f"RUN_MODE={RUN_MODE}  device={device}")
    logger.info(
        f"IMG_SIZE={base_img_size}  BATCH_SIZE={base_bs}  EPOCHS={base_epochs}  N_SPLITS={n_splits}"
    )
    validate_data_roots_or_exit(logger)
    wd_stop = start_stall_watchdog(timeout_sec=600, logger=logger)  # 15 minutes
    hb_stop = start_heartbeat(HEARTBEAT_FILE, HEARTBEAT_SECS, logger)
    # Build pairs & groups from ORIGINAL sets (no leakage from augmented variants)
    pairs, group_by_street, groups = ensure_pairs_and_groups(
        STREET_ORIG, SAT_ORIG, logger
    )
    # Index all pairs by their group id
    group_to_pairs: Dict[str, List[Pair]] = defaultdict(list)
    for st_rel, sa_rel, y in pairs:
        gid = group_by_street[st_rel]
        group_to_pairs[gid].append((st_rel, sa_rel, y))
    if os.path.exists(FOLDS_CACHE):
        with open(FOLDS_CACHE, "rb") as f:
            cached = pickle.load(f)
        groups = cached["groups"]
        folds = cached["folds"]
        logger.info(f"Loaded {len(folds)} folds from {FOLDS_CACHE}")
    else:
        folds = kfold_on_groups(groups, n_splits, logger)
        with open(FOLDS_CACHE, "wb") as f:
            pickle.dump({"groups": groups, "folds": folds}, f)
        logger.info(f"Saved folds to {FOLDS_CACHE}")
    # ----- OPTIONAL: inspect fold class distributions and exit -----
    INSPECT_FOLDS_ONLY = False  # set True to print distributions and stop
    if INSPECT_FOLDS_ONLY:
        num_classes = len({y for _, _, y in pairs})
        os.makedirs("fold_inspect", exist_ok=True)
        summary_rows = []
        for i, fold in enumerate(folds, start=1):
            train_pairs = [p for g in fold["train_groups"] for p in group_to_pairs[g]]
            val_pairs = [p for g in fold["test_groups"] for p in group_to_pairs[g]]
            tr_counts = np.bincount(
                [y for _, _, y in train_pairs], minlength=num_classes
            )
            vl_counts = np.bincount([y for _, _, y in val_pairs], minlength=num_classes)
            logger.info(
                "[fold %d] train: %s",
                i,
                " ".join(f"{c}:{int(n)}" for c, n in enumerate(tr_counts)),
            )
            logger.info(
                "[fold %d]   val: %s",
                i,
                " ".join(f"{c}:{int(n)}" for c, n in enumerate(vl_counts)),
            )
            # per fold csv
            with open(
                os.path.join("fold_inspect", f"fold_{i}_dist.csv"),
                "w",
                newline="",
                encoding="utf-8",
            ) as fcsv:
                w = csv.writer(fcsv)
                w.writerow(["class", "train_count", "val_count"])
                for c in range(num_classes):
                    w.writerow([c, int(tr_counts[c]), int(vl_counts[c])])
            summary_rows.append([i] + tr_counts.tolist() + vl_counts.tolist())
    INSPECT_FOLDS_ONLY = False  # set True to print distributions and stop
    if INSPECT_FOLDS_ONLY:
        num_classes = len({y for _, _, y in pairs})
        os.makedirs("fold_inspect", exist_ok=True)
        summary_rows = []
        for i, fold in enumerate(folds, start=1):
            train_pairs = [p for g in fold["train_groups"] for p in group_to_pairs[g]]
            val_pairs = [p for g in fold["test_groups"] for p in group_to_pairs[g]]
            tr_counts = np.bincount(
                [y for _, _, y in train_pairs], minlength=num_classes
            )
            vl_counts = np.bincount([y for _, _, y in val_pairs], minlength=num_classes)
            logger.info(
                "[fold %d] train: %s",
                i,
                " ".join(f"{c}:{int(n)}" for c, n in enumerate(tr_counts)),
            )
            logger.info(
                "[fold %d]   val: %s",
                i,
                " ".join(f"{c}:{int(n)}" for c, n in enumerate(vl_counts)),
            )
            with open(
                os.path.join("fold_inspect", f"fold_{i}_dist.csv"),
                "w",
                newline="",
                encoding="utf-8",
            ) as fcsv:
                w = csv.writer(fcsv)
                w.writerow(["class", "train_count", "val_count"])
                for c in range(num_classes):
                    w.writerow([c, int(tr_counts[c]), int(vl_counts[c])])
            summary_rows.append([i] + tr_counts.tolist() + vl_counts.tolist())
        header = (
            ["fold"]
            + [f"train_c{c}" for c in range(num_classes)]
            + [f"val_c{c}" for c in range(num_classes)]
        )
        with open(
            os.path.join("fold_inspect", "summary.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as fsum:
            w = csv.writer(fsum)
            w.writerow(header)
            w.writerows(summary_rows)
        logger.info("Wrote fold distributions in fold_inspect/. Exiting.")
        hb_stop.set()
        return
    aug_catalog_street = build_aug_catalog(STREET_AUG)
    aug_catalog_sat = build_aug_catalog(SAT_AUG)
    logger.info(
        f"[preaug] indexed street_aug keys={len(aug_catalog_street)}  sat_aug keys={len(aug_catalog_sat)}"
    )
    hb_stop.set()


    # CSVs
    need_fold_header = not os.path.exists(OUT_FOLD_FILE)
    need_cv_header = not os.path.exists(OUT_CV_FILE)
    fold_f = open(OUT_FOLD_FILE, "a", newline="", encoding="utf-8")
    cv_f = open(OUT_CV_FILE, "a", newline="", encoding="utf-8")
    fold_w = csv.DictWriter(
        fold_f,
        fieldnames=[
            "run",
            "fold",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "mean_absolute_error",
            "one_off_accuracy",
        ],
    )
    cv_w = csv.DictWriter(
        cv_f,
        fieldnames=[
            "run",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "mean_absolute_error",
            "one_off_accuracy",
        ],
    )
    if need_fold_header:
        fold_w.writeheader()
    if need_cv_header:
        cv_w.writeheader()
    # resume support
    skipping = SKIP_TO_RUN is not None
    # ----------------------------
    # === SINGLE-TRAIN MODE ===
    # Trains each (view, model) once with CE loss, two augs: none + preaug
    # CNNs: 30 epochs, TFs: 20 epochs, with early stopping
    # ----------------------------
    if RUN_MODE == "train":
        # choose the single held-out validation split ONCE and reuse it for all runs
        if not folds:
            folds = kfold_on_groups(groups, n_splits, logger)
        val_fold = folds[0]
        train_groups = val_fold["train_groups"]
        test_groups = val_fold["test_groups"]
        base_train_pairs = [p for g in train_groups for p in group_to_pairs[g]]
        base_val_pairs = [p for g in test_groups for p in group_to_pairs[g]]
        num_classes = len(sorted({y for _, _, y in base_train_pairs + base_val_pairs}))
        LOSS_FIXED = "CE"
        AUG_LIST = ["none", "preaug"]
        # plan runs: all views x all models x {none, preaug}
        planned_runs = []
        for view in VIEWS:
            for model_name in MODELS:
                for aug_name in AUG_LIST:
                    rk = f"{view}_{model_name}_imagenet1k_{LOSS_FIXED}_{aug_name}"
                    planned_runs.append((rk, view, model_name, aug_name))
        # registry & resume
        reg = _load_registry(RUN_REGISTRY)
        run_keys = [rk for rk, *_ in planned_runs]
        for rk in run_keys:
            if rk not in reg:
                reg[rk] = {
                    "status": "planned",
                    "ts": datetime.now().isoformat(timespec="seconds"),
                }
        _save_registry(RUN_REGISTRY, reg)
        if reset_progress:
            reg, reset_changed = reset_registry_status(
                reg, run_keys, clear_done_markers=True, logger=logger
            )
            if reset_changed:
                _save_registry(RUN_REGISTRY, reg)
            start_idx = 0
            reset_progress = False
        elif START_AT_RUN:
            start_idx = next(
                (
                    i
                    for i, (rk, *_rest) in enumerate(planned_runs)
                    if rk == START_AT_RUN
                ),
                0,
            )
        elif AUTO_RESUME:
            unfinished = first_unfinished(run_keys, reg)
            start_idx = (
                next(
                    (
                        i
                        for i, (rk, *_rest) in enumerate(planned_runs)
                        if rk == unfinished
                    ),
                    0,
                )
                if unfinished
                else 0
            )
        else:
            start_idx = 0
        # one CSV for all train-mode runs
        TRAIN_SUMMARY = "train_summary.csv"
        need_hdr = not os.path.exists(TRAIN_SUMMARY)
        with open(TRAIN_SUMMARY, "a", newline="", encoding="utf-8") as ts_f:
            ts_w = csv.DictWriter(
                ts_f,
                fieldnames=[
                    "run",
                    "view",
                    "model",
                    "aug",
                    "epochs",
                    "best_epoch",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "one_off_accuracy",
                    "mean_absolute_error",
                ],
            )
            if need_hdr:
                ts_w.writeheader()
            for rk, view, model_name, aug_name in planned_runs[start_idx:]:
                logger.info(f"=== TRAIN {rk} ===")
                reg = mark_run_status(reg, rk, "running")
                _save_registry(RUN_REGISTRY, reg)
                try:
                    # choose roots per augmentation
                    if aug_name == "preaug":
                        street_tr = STREET_AUG
                        sat_tr = SAT_AUG
                    else:
                        street_tr = STREET_ORIG
                        sat_tr = SAT_ORIG
                    street_vl = STREET_ORIG
                    sat_vl = SAT_ORIG
                    # loaders built once per run (fixed split)
                    img_size = pick_img_size_for_model(model_name, cfg["IMG_SIZE"])
                    batch_size = cfg["BATCH_SIZE"] or pick_batch_size(model_name)
                    with progress_span("build_loaders"):
                        tr_loader, vl_loader = build_loaders_with_guard(
                            base_train_pairs,
                            base_val_pairs,
                            view,
                            street_tr,
                            sat_tr,
                            street_vl,
                            sat_vl,
                            aug_name,
                            img_size,
                            batch_size,
                            logger,
                            preaug_strategy=PREAUG_STRATEGY,
                            num_workers=NUM_WORKERS,
                        )
                    # model + head (build per view so dual/combined get two-input wrappers)
                    with progress_span("build_loaders"):
                        model = build_model_for_view(
                            view, model_name, "imagenet1k", num_classes, device
                        )
                    # Fail fast if view/model mismatch
                    pair_flag = bool(getattr(model, "is_pair_input", False))
                    if view in ("dual", "combined"):
                        assert (
                            pair_flag
                        ), f"view={view} expects two input model, got {type(model).__name__}"
                    else:
                        assert (
                            not pair_flag
                        ), f"view={view} expects single input model, got {type(model).__name__}"
                    # optimizer & loss
                    lr, wd = pick_lr_and_wd(model_name)
                    optimizer = (optim.SGD if is_cnn(model_name) else optim.AdamW)(
                        model.parameters(),
                        lr=lr,
                        weight_decay=wd,
                        **({"momentum": CNN_MOMENTUM} if is_cnn(model_name) else {}),
                    )
                    criterion = get_criterion("CE", num_classes).to(device)
                    # epochs by family + early stop on F1
                    epochs = pick_epochs_for_model(
                        model_name
                    )  # 30 for CNNs, 20 for TFs
                    best_metrics, best_epoch = train_with_early_stopping(
                        model,
                        tr_loader,
                        vl_loader,
                        device,
                        criterion=criterion,
                        optimizer=optimizer,
                        epochs=epochs,
                        patience=EARLY_PATIENCE,
                        logger=logger,
                        monitor=EARLY_MONITOR,
                        mode=EARLY_MODE,
                        min_delta=EARLY_MIN_DELTA,
                        resume=True,
                        save_dir=CHECKPOINT_DIR,
                        run_key=rk,
                        view=view,
                        num_classes=num_classes,
                    )
                    # log & persist
                    ts_w.writerow(
                        {
                            "run": rk,
                            "view": view,
                            "model": model_name,
                            "aug": aug_name,
                            "epochs": epochs,
                            "best_epoch": best_epoch,
                            "accuracy": best_metrics.get("accuracy", 0.0),
                            "precision": best_metrics.get("precision", 0.0),
                            "recall": best_metrics.get("recall", 0.0),
                            "f1": best_metrics.get("f1", 0.0),
                            "one_off_accuracy": best_metrics.get(
                                "one_off_accuracy", 0.0
                            ),
                            "mean_absolute_error": best_metrics.get(
                                "mean_absolute_error", 0.0
                            ),
                        }
                    )
                    ts_f.flush()
                    reg = mark_run_status(
                        reg,
                        rk,
                        "done",
                        extra={
                            "best_epoch": best_epoch,
                            "best_f1": float(best_metrics.get("f1", 0.0)),
                        },
                    )
                except Exception as e:
                    logger.exception(f"[failed] {rk}: {e}")
                    reg = mark_run_status(
                        reg, rk, "failed", extra={"error": str(e)[:400]}
                    )
                finally:
                    _save_registry(RUN_REGISTRY, reg)
        # IMPORTANT: stop here so we DON'T fall through to the grid/CV loop
        return
        # same two configs for every model
        # ---------------------- GRID / FULL MODE: planned runs + resume ----------------------
    planned_runs: list[tuple[str, str, str, str, str, str]] = (
        []
    )  # (rk, view, model_name, pre, loss_name, aug_name)
    for view in VIEWS:
        for model_name in MODELS:
            for pre in PRETRAINING_BY_MODEL[model_name]:
                for loss_name in LOSSES:
                    for aug_name in AUGS_FOR_LOSS[loss_name]:
                        if RUN_MODE == "timing":
                            if (model_name, pre, loss_name, aug_name) != TIMING_CONFIG[
                                view
                            ]:
                                continue
                        rk = f"{view}_{model_name}_{pre}_{loss_name}_{aug_name}"
                        planned_runs.append(
                            (rk, view, model_name, pre, loss_name, aug_name)
                        )
    reg = _load_registry(RUN_REGISTRY)
    run_keys = [rk for rk, *_ in planned_runs]
    for rk in run_keys:
        if rk not in reg:
            reg[rk] = {
                "status": "planned",
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
    _save_registry(RUN_REGISTRY, reg)
    if reset_progress:
        reg, reset_changed = reset_registry_status(
            reg, run_keys, clear_done_markers=True, logger=logger
        )
        if reset_changed:
            _save_registry(RUN_REGISTRY, reg)
        start_idx = 0
        reset_progress = False
    elif START_AT_RUN:
        start_idx = next(
            (i for i, (rk, *_rest) in enumerate(planned_runs) if rk == START_AT_RUN), 0
        )
    elif AUTO_RESUME:
        unfinished = first_unfinished(run_keys, reg)
        start_idx = (
            next(
                (i for i, (rk, *_rest) in enumerate(planned_runs) if rk == unfinished),
                0,
            )
            if unfinished
            else 0
        )
    else:
        start_idx = 0
    hb_stop = start_heartbeat(HEARTBEAT_FILE, HEARTBEAT_SECS, logger)
    try:
        for run_key, view, model_name, pre, loss_name, aug_name in planned_runs[
            start_idx:
        ]:
            logger.info(f"=== RUN {run_key} ===")
            reg = mark_run_status(reg, run_key, "running")
            _save_registry(RUN_REGISTRY, reg)
            try:
                # ===================== BEGIN your existing per-run body =====================
                # meta line, roots, and CV writers setup as in your current code
                res_meta = {
                    "run_key": run_key,
                    "view": view,
                    "model": model_name,
                    "pretraining": pre,
                    "loss": loss_name,
                    "augmentation": aug_name,
                    "img_size": pick_img_size_for_model(model_name, cfg["IMG_SIZE"]),
                    "batch_size": (cfg["BATCH_SIZE"] or pick_batch_size(model_name)),
                    "folds": n_splits,
                }
                logger.info(json.dumps(res_meta))
                # choose roots per augmentation policy
                if aug_name == "preaug":
                    street_train_root = STREET_AUG
                    sat_train_root = SAT_AUG
                else:
                    street_train_root = STREET_ORIG
                    sat_train_root = SAT_ORIG
                street_val_root = STREET_ORIG
                sat_val_root = SAT_ORIG
                img_size = res_meta["img_size"]
                batch_size = res_meta["batch_size"]
                # run over cached folds
                best_acc = 0.0
                fold_metrics = []
                for fold_idx, fold in enumerate(folds, 1):
                    train_pairs = [
                        p for g in fold["train_groups"] for p in group_to_pairs[g]
                    ]
                    val_pairs = [
                        p for g in fold["test_groups"] for p in group_to_pairs[g]
                    ]
                    with progress_span("build_loaders"):
                        tr_loader, vl_loader = build_loaders_with_guard(
                            train_pairs,
                            val_pairs,
                            view,
                            street_train_root,
                            sat_train_root,
                            street_val_root,
                            sat_val_root,
                            aug_name,
                            img_size,
                            batch_size,
                            logger,
                            preaug_strategy=PREAUG_STRATEGY,
                            num_workers=NUM_WORKERS,
                        )
                    ys_fold = [y for _, _, y in (train_pairs + val_pairs)]
                    num_classes = (max(ys_fold) + 1) if ys_fold else 5
                    # Build model for the view
                    with progress_span("build_loaders"):
                        model = build_model_for_view(
                            view, model_name, pre, num_classes, device
                        )
                    pair_flag = bool(getattr(model, "is_pair_input", False))
                    if view in ("dual", "combined"):
                        assert (
                            pair_flag
                        ), f"view={view} expects two input model, got {type(model).__name__}"
                    else:
                        assert (
                            not pair_flag
                        ), f"view={view} expects single input model, got {type(model).__name__}"
                    logger.info(
                        f"[model] built {type(model).__name__} pair_input={pair_flag} view={view}"
                    )
                    # Optimizer and loss
                    lr, wd = pick_lr_and_wd(model_name)
                    optimizer = (optim.SGD if is_cnn(model_name) else optim.AdamW)(
                        model.parameters(),
                        lr=lr,
                        weight_decay=wd,
                        **({"momentum": CNN_MOMENTUM} if is_cnn(model_name) else {}),
                    )
                    if loss_name == "CE":
                        criterion = get_criterion("CE", num_classes).to(device)
                    elif loss_name == "SCE":
                        criterion = get_criterion("SCE", num_classes).to(device)
                    elif loss_name == "WCE":
                        class_weights = compute_class_weights_from_loader(tr_loader).to(
                            device
                        )
                        criterion = get_criterion(
                            "WCE", num_classes=num_classes, class_weights=class_weights
                        ).to(device)
                    else:
                        raise RuntimeError(f"Unknown loss {loss_name}")
                    # Early stopping, resume, and checkpointing per fold
                    epochs = pick_epochs_for_model(model_name)
                    rk_fold = f"{run_key}_fold{fold_idx}"
                    best_path = os.path.join(CHECKPOINT_DIR, f"{rk_fold}_best.pt")
                    state_path = os.path.join(CHECKPOINT_DIR, f"{rk_fold}_state.pt")
                    done_path = _done_marker_path(rk_fold)
                    if (not FORCE_RERUN_FOLDS) and os.path.exists(done_path):
                        logger.info(
                            f"[resume] {rk_fold} previously finalized; skipping training"
                        )
                        # Load best (preferred) or last state
                        loaded = False
                        try:
                            if os.path.exists(best_path):
                                state = torch.load(best_path, map_location="cpu")
                                model.load_state_dict(state)
                                logger.info(f"[ckpt] loaded best weights: {best_path}")
                                loaded = True
                        except Exception as e:
                            logger.warning(f"[ckpt] failed to load best: {e}")
                        if not loaded and os.path.exists(state_path):
                            try:
                                state = torch.load(state_path, map_location="cpu")
                                model.load_state_dict(state["model"])
                                logger.info(f"[ckpt] loaded last state: {state_path}")
                                loaded = True
                            except Exception as e:
                                logger.warning(f"[ckpt] failed to load last state: {e}")
                        # Evaluate only (no training)
                        preds, refs = eval_model(
                            model,
                            vl_loader,
                            device,
                            view=view,
                            limit_batches=RUN_CFG[RUN_MODE]["VAL_BATCHES"],
                            logger=logger,
                        )
                        m = compute_metrics(preds, refs)
                        fold_metrics.append(m)
                        fold_w.writerow(
                            {
                                "run": run_key,
                                "fold": fold_idx,
                                "accuracy": m.get("accuracy", 0.0),
                                "precision": m.get("precision", 0.0),
                                "recall": m.get("recall", 0.0),
                                "f1": m.get("f1", 0.0),
                                "mean_absolute_error": m.get(
                                    "mean_absolute_error", 0.0
                                ),
                                "one_off_accuracy": m.get("one_off_accuracy", 0.0),
                            }
                        )
                        fold_f.flush()
                        logger.info(
                            f"val  accuracy={m.get('accuracy',0):.4f}  f1={m.get('f1',0):.4f}"
                        )
                        free_cuda(logger, tag=f"after-fold {fold_idx}")
                        continue
                    train_limit = RUN_CFG[RUN_MODE]["TRAIN_BATCHES"]
                    val_limit = RUN_CFG[RUN_MODE]["VAL_BATCHES"]
                    best_metrics, best_epoch = train_with_early_stopping(
                        model,
                        tr_loader,
                        vl_loader,
                        device,
                        criterion=criterion,
                        optimizer=optimizer,
                        epochs=epochs,
                        patience=EARLY_PATIENCE,
                        logger=logger,
                        view=view,
                        train_limit=train_limit,
                        val_limit=val_limit,
                        run_key=rk_fold,
                        monitor=EARLY_MONITOR,
                        mode=EARLY_MODE,
                        min_delta=EARLY_MIN_DELTA,
                        resume=RESUME_RUNS,
                        save_dir=CHECKPOINT_DIR,
                        num_classes=num_classes,
                    )
                    _best_loaded = False
                    if os.path.exists(best_path):
                        try:
                            state = torch.load(best_path, map_location="cpu")
                            model.load_state_dict(state)
                            logger.info(
                                f"[ckpt] loaded best weights for eval: {best_path}"
                            )
                            _best_loaded = True
                        except Exception as e:
                            logger.warning(f"[ckpt] failed to load best for eval: {e}")
                    if not _best_loaded and os.path.exists(state_path):
                        try:
                            state = torch.load(state_path, map_location="cpu")
                            model.load_state_dict(state.get("model", state))
                            logger.info(
                                f"[ckpt] loaded last state for eval: {state_path}"
                            )
                        except Exception as e:
                            logger.warning(
                                f"[ckpt] failed to load last state for eval: {e}"
                            )
                    preds, refs = eval_model(
                        model,
                        vl_loader,
                        device,
                        view=view,
                        limit_batches=val_limit,
                        logger=logger,
                    )
                    m = compute_metrics(preds, refs)
                    fold_metrics.append(m)
                    fold_w.writerow(
                        {
                            "run": run_key,
                            "fold": fold_idx,
                            "accuracy": m.get("accuracy", 0.0),
                            "precision": m.get("precision", 0.0),
                            "recall": m.get("recall", 0.0),
                            "f1": m.get("f1", 0.0),
                            "mean_absolute_error": m.get("mean_absolute_error", 0.0),
                            "one_off_accuracy": m.get("one_off_accuracy", 0.0),
                        }
                    )
                    fold_f.flush()
                    logger.info(
                        f"val  accuracy={m.get('accuracy',0):.4f}  f1={m.get('f1',0):.4f}"
                    )
                    free_cuda(logger, tag=f"after-fold {fold_idx}")

                if not fold_metrics:
                    logger.warning(
                        f"[cv] no folds evaluated for {run_key}; skipping CV summary row"
                    )
                    continue
                cv_m = aggregate_metrics(fold_metrics)
                cv_w.writerow(
                    {
                        "run": run_key,
                        "accuracy": cv_m.get("accuracy", 0.0),
                        "precision": cv_m.get("precision", 0.0),
                        "recall": cv_m.get("recall", 0.0),
                        "f1": cv_m.get("f1", 0.0),
                        "mean_absolute_error": cv_m.get("mean_absolute_error", 0.0),
                        "one_off_accuracy": cv_m.get("one_off_accuracy", 0.0),
                    }
                )
                cv_f.flush()
                # ===================== END your existing per-run body =======================
                reg = mark_run_status(reg, run_key, "done")
            except Exception as e:
                logger.exception(f"[failed] {run_key}: {e}")
                reg = mark_run_status(
                    reg, run_key, "failed", extra={"error": str(e)[:400]}
                )
            finally:
                _save_registry(RUN_REGISTRY, reg)
    finally:
        hb_stop.set()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reset-progress", action="store_true")
    parser.add_argument("--supervise", action="store_true")
    parser.add_argument("--backoff", type=float, default=5.0)  # seconds
    # parse only our wrapper flags; pass the rest to the child unchanged
    known, unknown = parser.parse_known_args()
    if known.supervise:
        # Supervisor: relaunch child on crash with exponential backoff
        restarts = 0
        while True:
            cmd = [sys.executable, __file__]
            if known.reset_progress:
                cmd.append("--reset-progress")
            cmd += [arg for arg in unknown if arg != "--supervise"]
            print(
                f"[supervisor] launching: {' '.join(shlex.quote(c) for c in cmd)}  (attempt {restarts+1})"
            )
            proc = subprocess.Popen(cmd)
            proc.wait()
            code = proc.returncode
            if code == 0:
                print("[supervisor] child exited cleanly.")
                sys.exit(0)
            restarts += 1
            # No maximum attempts: keep supervising forever
            # Use a modest, bounded linear delay instead of exponential backoff
            delay = min(
                known.backoff * restarts, 30.0
            )  # e.g., 5s,10s,... capped at 30s
            print(
                f"[supervisor] child crashed (code={code}). restarting in {delay:.1f}s..."
            )
            time.sleep(delay)
    else:
        # Normal run: call your existing main(entrypoint) here
        sys.exit(main(reset_progress=known.reset_progress))
