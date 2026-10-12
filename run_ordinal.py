"""Ordinal walkability experiment: street-only model, ordinal head vs CE baseline.

For each of the 10 spatial folds k (folds.csv):
  - test on fold k, early-stop on fold (k+1) % 10, train on the other 8
  - train with online article augmentation (original + aug copies per epoch)
  - early stopping on validation loss; the test fold is used once, at the end
  - write per-point test predictions and fold metrics

Heads:
  ordinal : Linear(..., 4), loss = mean_k BCEWithLogits(z_k, y_k) on soft cumulative targets
  ce      : Linear(..., 5), loss = CE against the raters' soft class distribution

Outputs (results/<run>/): fold_<k>_preds.csv, fold_<k>_metrics.json, summary.csv
Re-running skips folds whose metrics file exists (use --overwrite to redo).

Usage:
    python run_ordinal.py --head ordinal
    python run_ordinal.py --head ce
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import time

# torch.compile cache on disk: the default /tmp is a small RAM disk on this machine
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "torchinductor"))

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from data_utils import (VIEWS_WITH_AERIAL, VIEWS_WITH_STREET, PointViewDataset, aerial_index, has_street_images,
                        load_points, seed_worker)
from dual_encoder import FusionModel, TileMeanModel
from logging_utils import get_logger
from losses import get_criterion
from metrics_utils import (cumulative_from_output, decode_thresholds, discrete_metrics_vs_raters, ordinal_metrics,
                           tune_thresholds)
from model_utils import load_model
from run_status import fold_complete

N_FOLDS = 10
AERIAL_DIR = os.path.join("views_output", "wms_patches", "images")
CNN_LR, CNN_MOMENTUM, CNN_WD = 1e-3, 0.9, 1e-4
TF_LR, TF_WD = 5e-5, 0.01
# --schedule article: recipe of Damandeh & Saberi (2026) for Swin on 5-point walkability ratings
ART_LR, ART_WD, ART_LAYER_DECAY, ART_WARMUP_FRAC, ART_CLIP = 1e-4, 1e-3, 0.9, 0.1, 1.0
TRANSFORMER_HINTS = ("vit", "deit", "swin", "beit")
METRIC_KEYS = ["accuracy", "f1", "mean_absolute_error", "one_off_accuracy", "amae", "qwk", "tuned_qwk",
               "alpha_ordinal", "score_mae", "score_spearman", "monotonicity_violation_pct"]
DISCRETE_KEYS = ("accuracy", "f1", "mean_absolute_error", "one_off_accuracy", "amae", "mmae", "qwk",
                 "alpha_ordinal", "alpha_interval", "recall_1", "recall_5")


def seed_everything(seed: int):
    """Seed python, numpy and torch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_transformer(model_name: str) -> bool:
    """True for transformer backbones (AdamW), False for CNNs (SGD)."""
    return any(h in model_name.lower() for h in TRANSFORMER_HINTS)


def make_optimizer(model, model_name: str, schedule: str = "constant", lr=None):
    """
    constant: AdamW for transformers, SGD with momentum for CNNs (settings of the previous study);
              lr overrides the default learning rate.
    article:  AdamW with layer-wise lr decay over the backbone (top layer x1, each layer below x0.9),
              no weight decay on norms/biases; the tile head trains at the full lr.
    """
    if schedule == "article":
        if not is_transformer(model_name):
            raise SystemExit("--schedule article is defined for transformer backbones only")
        from timm.optim import param_groups_layer_decay
        groups = param_groups_layer_decay(model.ex.backbone, weight_decay=ART_WD, layer_decay=ART_LAYER_DECAY)
        groups += [{"params": [model.head.weight], "weight_decay": ART_WD, "lr_scale": 1.0},
                   {"params": [model.head.bias], "weight_decay": 0.0, "lr_scale": 1.0}]
        for g in groups:
            g["lr"] = ART_LR * g.get("lr_scale", 1.0)
        return optim.AdamW(groups, lr=ART_LR)
    if is_transformer(model_name):
        return optim.AdamW(model.parameters(), lr=lr or TF_LR, weight_decay=TF_WD)
    return optim.SGD(model.parameters(), lr=lr or CNN_LR, momentum=CNN_MOMENTUM, weight_decay=CNN_WD)


def last_stage_modules(backbone):
    """Final stage of a transformer backbone and its output norm (Swin: layers[-1]; ViT/DeiT: blocks[-1])."""
    if hasattr(backbone, "layers"):
        return [backbone.layers[-1], backbone.norm]
    if hasattr(backbone, "blocks"):
        return [backbone.blocks[-1], backbone.norm]
    raise SystemExit(f"don't know the last stage of {type(backbone).__name__}")


def set_trainable(model, modules):
    """Freeze the whole backbone, then unfreeze the given backbone modules; the head always trains."""
    for p in model.ex.parameters():
        p.requires_grad_(False)
    for m in modules:
        for p in m.parameters():
            p.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


def make_scheduler(optimizer, total_steps: int):
    """Per-batch linear warmup over the first 10% of steps, then cosine decay to 0."""
    warmup = max(1, int(ART_WARMUP_FRAC * total_steps))

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total_steps - warmup))))

    return optim.lr_scheduler.LambdaLR(optimizer, factor)


def make_loader(points, args, train: bool, seed: int, persistent: bool = False, aerial_paths=None):
    """
    Build a DataLoader over the given points (shuffled + augmented when training).
    persistent keeps the workers alive across epochs (train/val loaders); each
    worker's augmentation stream then continues instead of being reseeded per epoch.
    """
    ds = PointViewDataset(points, args.street_dir, args.img_size, train=train, aug_copies=args.aug_copies,
                          seed=seed, view=args.view, aerial_paths=aerial_paths)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=train, num_workers=args.num_workers,
                      worker_init_fn=seed_worker, generator=gen, pin_memory=torch.cuda.is_available(),
                      persistent_workers=persistent and args.num_workers > 0)


def run_epoch(model, loader, criterion, head, device, optimizer=None, scaler=None, limit=None, desc=None,
              scheduler=None, clip=None):
    """
    One pass over the loader. Trains when an optimizer is given, otherwise evaluates.
    When training, steps the per-batch scheduler and clips the gradient norm if given.
    Shows a transient batch progress bar labelled desc.
    Returns (mean loss, outputs [N, out_dim], dataset indices [N]).
    """
    train = optimizer is not None
    model.train(train)
    total, n, outs, idxs = 0.0, 0, [], []
    use_amp = scaler is not None
    total_batches = len(loader) if limit is None else min(limit, len(loader))
    bar = tqdm(total=total_batches, desc=desc, unit="batch", leave=False, dynamic_ncols=True)
    with torch.set_grad_enabled(train):
        for b, (x, y_cum, y_soft, idx) in enumerate(loader, start=1):
            if limit is not None and b > limit:
                break
            # single view: tiles [B, T, 3, H, W]; fused views: (street tiles, aerial tile)
            inputs = [t.to(device, non_blocking=True) for t in (x if isinstance(x, (list, tuple)) else [x])]
            target = (y_cum if head == "ordinal" else y_soft).to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(*inputs)
            loss = criterion(out.float(), target)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    if clip:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            total += float(loss.detach()) * y_cum.size(0)
            n += y_cum.size(0)
            bar.update(1)
            bar.set_postfix(loss=f"{total / n:.4f}")
            if not train:
                outs.append(out.float().cpu().numpy())
                idxs.append(idx.numpy())
    bar.close()
    outs = np.concatenate(outs) if outs else np.zeros((0, 0))
    idxs = np.concatenate(idxs) if idxs else np.zeros(0, dtype=int)
    return total / max(1, n), outs, idxs


def evaluate_outputs(outputs, points, head, thresholds=None):
    """
    Compute metrics for one head's outputs. Discrete metrics use the 0.5 rule; with
    thresholds (tuned on validation), the same discrete metrics are added as tuned_*.
    For the CE baseline, the native argmax decision is added as argmax_* (reference).
    """
    cum = cumulative_from_output(outputs, head)
    lo = np.array([p["rater_lo"] for p in points])
    hi = np.array([p["rater_hi"] for p in points])
    metrics = ordinal_metrics(cum, lo, hi)
    if thresholds is not None:
        tuned = ordinal_metrics(cum, lo, hi, pred_override=decode_thresholds(cum, thresholds))
        metrics.update({f"tuned_{k}": tuned[k] for k in DISCRETE_KEYS})
    if head == "ce":
        argmax = ordinal_metrics(cum, lo, hi, pred_override=outputs.argmax(axis=1) + 1)
        metrics.update({f"argmax_{k}": argmax[k] for k in DISCRETE_KEYS})
    return cum, metrics


def sync_to_disk(path):
    """Force a written file to disk, so a power cut cannot leave it half-written."""
    with open(path, "rb") as f:
        os.fsync(f.fileno())


def write_preds(path, points, outputs, cum, fold):
    """Per-point outputs, cumulative probabilities, 0.5-rule class and expected score."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["point_id", "fold", "city", "rater_lo", "rater_hi"]
                   + [f"out_{j}" for j in range(outputs.shape[1])]
                   + [f"p_above_{j}" for j in range(1, 5)] + ["pred_class", "score"])
        for p, o, c in zip(points, outputs, cum):
            w.writerow([p["point_id"], fold, p["city"], p["rater_lo"], p["rater_hi"]]
                       + [f"{v:.5f}" for v in o] + [f"{v:.5f}" for v in c]
                       + [int(1 + (c > 0.5).sum()), f"{1 + c.sum():.4f}"])


def validation_qwk(outputs, idx, points, head):
    """QWK of the decoded classes on the validation fold, computed exactly like the test QWK."""
    cum = cumulative_from_output(outputs, head)
    pred = 1 + (cum > 0.5).sum(axis=1)
    lo = np.array([points[i]["rater_lo"] for i in idx])
    hi = np.array([points[i]["rater_hi"] for i in idx])
    return discrete_metrics_vs_raters(pred, lo, hi)["qwk"]


def build_model(args, k, out_dim, logger):
    """
    Model for the view (on CPU):
      street / aerial  TileMeanModel (pretrained backbone, linear or CORAL output)
      dual             FusionModel with two pretrained encoders trained jointly
      late             FusionModel with the encoders of this fold's trained street-only and
                       aerial-only models (fold_<k>_best.pt), kept fixed
    Returns (model, description of the starting weights).
    """
    def backbone():
        b = load_model(args.model, "imagenet1k", None, pretrained=not args.no_pretrained)
        cfg = getattr(b, "pretrained_cfg", {}) or {}
        return b, "random" if args.no_pretrained else cfg.get("hf_hub_id") or "torchvision IMAGENET1K"

    if args.view in ("street", "aerial"):
        b, weights = backbone()
        return TileMeanModel(b, out_dim=out_dim, head_type=args.head_type), weights
    if args.view == "dual":
        (b1, weights), (b2, _) = backbone(), backbone()
        return FusionModel(TileMeanModel(b1, out_dim).ex, TileMeanModel(b2, out_dim).ex, out_dim), weights
    encoders = []
    for view, run in (("street", args.street_run), ("aerial", args.aerial_run)):
        path = os.path.join(run, f"fold_{k}_best.pt")
        if not os.path.exists(path):
            raise SystemExit(f"late fusion needs the trained {view}-only model of fold {k}: {path} is missing "
                             f"(run the {view} view with --save-weights)")
        single = TileMeanModel(backbone()[0], out_dim)
        single.load_state_dict(torch.load(path, map_location="cpu"))
        encoders.append(single.ex)
        logger.info(f"[fold {k}] late fusion: {view} encoder from {path}")
    return FusionModel(*encoders, out_dim), f"trained {args.street_run} + {args.aerial_run}"


def train_fold(k, points, args, device, logger, run_dir, aerial_paths=None):
    """Train and evaluate one fold; write predictions and metrics."""
    metrics_path = os.path.join(run_dir, f"fold_{k}_metrics.json")
    if not args.overwrite and fold_complete(run_dir, k, need_weights=args.save_weights):
        logger.info(f"[fold {k}] already done, skipping ({metrics_path})")
        return
    if os.path.exists(metrics_path) and not args.overwrite:
        logger.warning(f"[fold {k}] files are incomplete (interrupted run?), training this fold again")
    val_fold = (k + 1) % N_FOLDS
    test = [p for p in points if p["fold"] == k]
    val = [p for p in points if p["fold"] == val_fold]
    train = [p for p in points if p["fold"] not in (k, val_fold)]
    logger.info(f"[fold {k}] train={len(train)} val={len(val)} (fold {val_fold}) test={len(test)}")

    seed = args.seed + k
    seed_everything(seed)
    model, weights = build_model(args, k, 4 if args.head == "ordinal" else 5, logger)
    if args.head_type == "coral" and args.coral_init == "prior":
        # CORAL thresholds differ only by their biases: start them at the training fold's
        # threshold rates, logit P(Y > k), so their spacing need not be learned at a small lr
        rate = np.clip(np.mean([p["y_cum"] for p in train], axis=0), 1e-3, 1 - 1e-3)
        with torch.no_grad():
            model.head.bias.copy_(torch.from_numpy(np.log(rate / (1 - rate))).float())
        logger.info(f"[fold {k}] CORAL biases initialised at logit P(Y>k) of the training fold: "
                    + " ".join(f"{b:.2f}" for b in model.head.bias.tolist()))
    model = model.to(device)
    logger.info(f"[fold {k}] view={args.view} backbone={args.model} weights={weights}")
    if device.type == "cuda":
        # speed only, same model: measured gains are compile for transformers, channels_last for CNNs
        if not is_transformer(args.model):
            model = model.to(memory_format=torch.channels_last)
        elif not args.no_compile:
            model.compile()  # in place, so state_dict keys are unchanged
    criterion = get_criterion("ORDINAL" if args.head == "ordinal" else "SOFT_CE", num_classes=5)
    scaler = torch.amp.GradScaler("cuda") if (device.type == "cuda" and not args.no_amp) else None

    tr_ld = make_loader(train, args, train=True, seed=seed, persistent=True, aerial_paths=aerial_paths)
    vl_ld = make_loader(val, args, train=False, seed=seed, persistent=True, aerial_paths=aerial_paths)
    te_ld = make_loader(test, args, train=False, seed=seed, aerial_paths=aerial_paths)

    # fine-tuning mode -> list of phases (name, optimizer, scheduler, clip)
    if args.view == "late":
        # fixed encoders from the single-view models: only the fusion head trains (AdamW, head lr)
        for p in list(model.st_ex.parameters()) + list(model.sa_ex.parameters()):
            p.requires_grad_(False)
        model.freeze_encoders = True  # encoders stay in eval mode: no BatchNorm updates, no dropout
        phases = [("fusion_head", lambda: optim.AdamW(model.head.parameters(), lr=args.head_lr,
                                                      weight_decay=TF_WD), None, None)]
    elif args.finetune == "full":
        if args.schedule == "article" and args.view != "street":
            raise SystemExit("--schedule article is implemented for the street view only")
        optimizer = make_optimizer(model, args.model, args.schedule, args.lr)
        scheduler, clip = None, None
        if args.schedule == "article":
            steps_per_epoch = len(tr_ld) if args.limit_batches is None else min(args.limit_batches, len(tr_ld))
            scheduler, clip = make_scheduler(optimizer, args.epochs * steps_per_epoch), ART_CLIP
            scales = [g.get("lr_scale", 1.0) for g in optimizer.param_groups]
            logger.info(f"[fold {k}] schedule=article peak_lr={ART_LR:g} layer lr scales {min(scales):.3f}-"
                        f"{max(scales):.3f} ({len(scales)} groups) warmup={ART_WARMUP_FRAC:.0%} cosine over "
                        f"{args.epochs} epochs clip={ART_CLIP}")
        phases = [("full", lambda: optimizer, scheduler, clip)]
    else:
        if args.schedule != "constant" or not is_transformer(args.model) or args.view not in ("street", "aerial"):
            raise SystemExit("--finetune frozen/lpft is defined for single-view transformers with the constant "
                             "schedule")

        def head_phase():
            params = set_trainable(model, [])
            return optim.AdamW(params, lr=args.head_lr, weight_decay=TF_WD)

        def last_stage_phase():
            model.load_state_dict(state["best_state"])  # continue from the best head-only model
            params = set_trainable(model, last_stage_modules(model.ex.backbone))
            return optim.AdamW(params, lr=args.lr or TF_LR, weight_decay=TF_WD)

        phases = [("head", head_phase, None, None)]
        if args.finetune == "lpft":
            phases.append(("last_stage", last_stage_phase, None, None))

    # model selection over all phases: lowest validation loss, or highest validation QWK (--monitor);
    # patience restarts in each phase, the best model can come from any phase
    state = {"best_score": float("-inf"), "best_loss": float("inf"), "best_qwk": float("nan"),
             "best_state": None, "best_epoch": 0, "best_phase": None}
    done = 0
    for phase, make_opt, scheduler, clip in phases:
        optimizer = make_opt()
        n_train = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
        logger.info(f"[fold {k}] phase={phase} trainable params={n_train:,} "
                    f"lr={max(g['lr'] for g in optimizer.param_groups):g}")
        bad = 0
        ep_bar = tqdm(total=args.epochs, desc=f"fold {k} {phase} epochs", unit="ep", leave=False, dynamic_ncols=True)
        for ep in range(1, args.epochs + 1):
            t0 = time.time()
            tr_loss, _, _ = run_epoch(model, tr_ld, criterion, args.head, device, optimizer, scaler,
                                      limit=args.limit_batches, desc=f"{phase} epoch {ep} train",
                                      scheduler=scheduler, clip=clip)
            lr_now = max(g["lr"] for g in optimizer.param_groups)
            vl_loss, vl_out, vl_idx = run_epoch(model, vl_ld, criterion, args.head, device,
                                                limit=args.limit_batches, desc=f"{phase} epoch {ep} val")
            vl_qwk = validation_qwk(vl_out, vl_idx, val, args.head)
            score = vl_qwk if args.monitor == "qwk" else -vl_loss
            improved = score > state["best_score"] + args.min_delta
            if improved:
                state.update(best_score=score, best_loss=vl_loss, best_qwk=vl_qwk, best_epoch=done + ep,
                             best_phase=phase,
                             best_state={n: t.detach().cpu().clone() for n, t in model.state_dict().items()})
                bad = 0
            else:
                bad += 1
            logger.info(f"[fold {k}] {phase} epoch {ep}/{args.epochs} train_loss={tr_loss:.4f} "
                        f"val_loss={vl_loss:.4f} val_qwk={vl_qwk:.4f} lr={lr_now:.2e}{' *' if improved else ''} "
                        f"({time.time() - t0:.0f}s)")
            ep_bar.update(1)
            ep_bar.set_postfix(best=f"{args.monitor}={abs(state['best_score']):.4f}@{state['best_epoch']}",
                               patience=f"{bad}/{args.patience}")
            if bad >= args.patience:
                logger.info(f"[fold {k}] {phase}: early stop after {ep} epochs, best so far epoch "
                            f"{state['best_epoch']} ({state['best_phase']}) val_loss={state['best_loss']:.4f} "
                            f"val_qwk={state['best_qwk']:.4f}")
                break
        ep_bar.close()
        done += ep

    best_state, best_epoch, best_loss, best_qwk = (state["best_state"], state["best_epoch"],
                                                   state["best_loss"], state["best_qwk"])
    model.load_state_dict(best_state)
    if args.save_weights:
        torch.save(best_state, os.path.join(run_dir, f"fold_{k}_best.pt"))
        sync_to_disk(os.path.join(run_dir, f"fold_{k}_best.pt"))
    # decision thresholds: tuned for QWK on the validation fold with the selected model
    _, vl_out, vl_idx = run_epoch(model, vl_ld, criterion, args.head, device, limit=args.limit_batches,
                                  desc="val (best)")
    val_eval = [val[i] for i in vl_idx]
    vl_cum = cumulative_from_output(vl_out, args.head)
    write_preds(os.path.join(run_dir, f"fold_{k}_val_preds.csv"), val_eval, vl_out, vl_cum, val_fold)
    sync_to_disk(os.path.join(run_dir, f"fold_{k}_val_preds.csv"))
    thresholds = tune_thresholds(vl_cum, np.array([p["rater_lo"] for p in val_eval]),
                                 np.array([p["rater_hi"] for p in val_eval]))
    logger.info(f"[fold {k}] thresholds tuned on validation: " + " ".join(f"{t:.2f}" for t in thresholds))

    te_loss, outputs, idx = run_epoch(model, te_ld, criterion, args.head, device, limit=args.limit_batches,
                                      desc="test")
    test_eval = [test[i] for i in idx]
    cum, metrics = evaluate_outputs(outputs, test_eval, args.head, thresholds)
    metrics.update({"fold": k, "val_fold": val_fold, "weights": weights, "best_epoch": best_epoch,
                    "monitor": args.monitor, "best_val_loss": best_loss, "best_val_qwk": best_qwk,
                    "view": args.view, "head_type": args.head_type,
                    "finetune": args.finetune, "lr": args.lr or TF_LR, "head_lr": args.head_lr,
                    "best_phase": state["best_phase"],
                    "test_loss": te_loss, "tuned_thresholds": [float(t) for t in thresholds],
                    "n_train": len(train), "n_val": len(val)})
    write_preds(os.path.join(run_dir, f"fold_{k}_preds.csv"), test_eval, outputs, cum, k)
    sync_to_disk(os.path.join(run_dir, f"fold_{k}_preds.csv"))
    # metrics last and atomically: its presence marks the fold as finished
    with open(metrics_path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(metrics_path + ".tmp", metrics_path)
    logger.info(f"[fold {k}] test " + " ".join(f"{m}={metrics[m]:.4f}" for m in METRIC_KEYS))


def write_summary(run_dir, logger):
    """Mean and std of every fold metric over the folds present in run_dir."""
    rows = []
    for k in range(N_FOLDS):
        p = os.path.join(run_dir, f"fold_{k}_metrics.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                rows.append(json.load(f))
    if not rows:
        return
    keys = [k for k in rows[0] if k not in ("fold", "val_fold", "weights", "monitor", "n", "n_train", "n_val",
                                             "best_epoch", "tuned_thresholds", "finetune", "lr", "head_lr",
                                             "best_phase", "view", "head_type")]
    with open(os.path.join(run_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "std", "n_folds"])
        for key in keys:
            vals = np.array([r[key] for r in rows], dtype=float)
            w.writerow([key, f"{np.nanmean(vals):.4f}", f"{np.nanstd(vals, ddof=1) if len(vals) > 1 else 0.0:.4f}",
                        int(np.sum(~np.isnan(vals)))])
    logger.info(f"[summary] {len(rows)} folds -> {os.path.join(run_dir, 'summary.csv')}")


def main():
    """
    CLI entry point.

    Steps:
    1) Load labels + folds and keep points whose Street View images exist.
    2) Train/evaluate the requested folds.
    3) Write the per-run summary.
    """
    ap = argparse.ArgumentParser(description="Ordinal walkability: street-only, ordinal vs CE head")
    ap.add_argument("--head", choices=["ordinal", "ce"], required=True)
    ap.add_argument("--model", default="swin_base")
    ap.add_argument("--labels", default="point_labels.csv")
    ap.add_argument("--folds-csv", default="folds.csv")
    ap.add_argument("--view", choices=["street", "aerial", "dual", "late"], default="street",
                    help="street tiles, aerial patch, dual encoder (joint), or late fusion of the trained "
                         "street-only and aerial-only encoders")
    ap.add_argument("--street-dir", default=os.path.join("views_output", "streetview", "images"))
    ap.add_argument("--aerial-dir", default=AERIAL_DIR)
    ap.add_argument("--street-run", default=None,
                    help="late fusion: run directory of the street-only model (default <out-dir>/street_<model>_<head>)")
    ap.add_argument("--aerial-run", default=None,
                    help="late fusion: run directory of the aerial-only model (default <out-dir>/aerial_<model>_<head>)")
    ap.add_argument("--head-type", choices=["linear", "coral"], default="linear",
                    help="ordinal output layer: 4 independent thresholds, or CORAL (shared weights)")
    ap.add_argument("--coral-init", choices=["prior", "fixed"], default="prior",
                    help="CORAL biases start at the training fold's logit P(Y>k) (prior) or at 1, .75, .5, .25 "
                         "(fixed, the reference implementation)")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--folds", default="all", help="'all' or comma-separated fold ids")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--schedule", choices=["constant", "article"], default="constant",
                    help="constant lr (previous study) or the article recipe: warmup + cosine, layer-wise lr decay, "
                         "clipping")
    ap.add_argument("--finetune", choices=["full", "frozen", "lpft"], default="full",
                    help="full: train everything; frozen: backbone fixed, train the head only; lpft: head only, "
                         "then also the last backbone stage")
    ap.add_argument("--lr", type=float, default=None,
                    help="learning rate of full fine-tuning / of the lpft last-stage phase (default 5e-5)")
    ap.add_argument("--head-lr", type=float, default=1e-3, help="learning rate of head-only training (frozen, lpft)")
    ap.add_argument("--monitor", choices=["loss", "qwk"], default="loss",
                    help="early stopping / model selection on validation loss or validation QWK")
    ap.add_argument("--batch-size", type=int, default=16, help="points per batch (x4 tiles)")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--aug-copies", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=16,
                    help="keep fixed across runs: augmentation draws depend on the worker count")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-compile", action="store_true", help="disable torch.compile for transformers")
    ap.add_argument("--save-weights", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow-missing", action="store_true",
                    help="train on the points that have street images instead of aborting")
    # smoke-test helpers
    ap.add_argument("--no-pretrained", action="store_true", help="random init (smoke tests only)")
    ap.add_argument("--max-points", type=int, default=None, help="use only the first N points (smoke tests)")
    ap.add_argument("--limit-batches", type=int, default=None, help="cap batches per pass (smoke tests)")
    ap.add_argument("--tag", default="", help="suffix for the run directory")
    args = ap.parse_args()

    if args.head_type == "coral" and (args.head != "ordinal" or args.view not in ("street", "aerial")):
        raise SystemExit("--head-type coral is for the ordinal head of single-view models")
    args.street_run = args.street_run or os.path.join(args.out_dir, f"street_{args.model}_{args.head}")
    args.aerial_run = args.aerial_run or os.path.join(args.out_dir, f"aerial_{args.model}_{args.head}")
    run_name = (f"{args.view}_{args.model}_{args.head}" + ("_coral" if args.head_type == "coral" else "")
                + ("_qwk" if args.monitor == "qwk" else "")
                + ("_sched" if args.schedule == "article" else "")
                + ("" if args.finetune == "full" else f"_{args.finetune}") + (f"_{args.tag}" if args.tag else ""))
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    logger = get_logger(name=run_name, log_file=os.path.join(run_dir, "train.log"))
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points = load_points(args.labels, args.folds_csv)
    aerial_paths = aerial_index(args.aerial_dir) if args.view in VIEWS_WITH_AERIAL else None
    usable = [p for p in points
              if (args.view not in VIEWS_WITH_STREET or has_street_images(args.street_dir, p["point_id"]))
              and (aerial_paths is None or p["point_id"] in aerial_paths)]
    n_missing = len(points) - len(usable)
    if args.max_points:
        usable = usable[: args.max_points]
    logger.info(f"run={run_name} view={args.view} device={device} points={len(points)} with images="
                f"{len(points) - n_missing}" + (f" (using first {len(usable)})" if args.max_points else ""))
    if n_missing and not args.allow_missing:
        logger.error(f"{n_missing} labelled points are missing images for view {args.view}; aborting "
                     f"(pass --allow-missing to exclude them)")
        raise SystemExit(1)
    if n_missing:
        logger.warning(f"{n_missing} labelled points are missing images and are excluded")

    folds = range(N_FOLDS) if args.folds == "all" else [int(x) for x in args.folds.split(",")]
    with logging_redirect_tqdm(loggers=[logger]):
        try:
            for k in tqdm(folds, desc=f"{run_name} folds", unit="fold", dynamic_ncols=True):
                train_fold(k, usable, args, device, logger, run_dir, aerial_paths)
        except Exception:
            logger.exception("run failed")
            raise
    write_summary(run_dir, logger)


if __name__ == "__main__":
    main()
