"""Choices for the fine-tuning study, made on validation loss only (test folds are never used).

choose MODE=RUN_DIR ...
    Mean best validation loss over the folds for each run; prints a summary and, on the
    last line, the MODE with the lowest mean.
nested --out OUT_DIR RUN_DIR ...
    Learning-rate selection per fold: for each fold, the run with the lowest best validation
    loss is selected and its fold files (test/val predictions, metrics) are copied to OUT_DIR,
    which then behaves like a normal run directory for evaluate_runs.py. lr_choice.csv lists
    the candidates' validation losses and the choice per fold.
"""

import argparse
import csv
import glob
import json
import os
import re
import shutil
import sys

import numpy as np


def val_losses(run_dir):
    """{fold: best validation loss} from a run's fold metrics."""
    out = {}
    for p in glob.glob(os.path.join(run_dir, "fold_*_metrics.json")):
        with open(p, encoding="utf-8") as f:
            out[int(re.search(r"fold_(\d+)_metrics", p).group(1))] = json.load(f)["best_val_loss"]
    return out


def choose(pairs):
    runs = dict(p.split("=", 1) for p in pairs)
    losses = {m: val_losses(d) for m, d in runs.items()}
    folds = set.intersection(*(set(v) for v in losses.values()))
    if not folds:
        raise SystemExit("no fold is complete in every run")
    means = {m: float(np.mean([v[k] for k in folds])) for m, v in losses.items()}
    for m in runs:
        print(f"  {m:8s} mean best validation loss {means[m]:.4f} over {len(folds)} folds ({runs[m]})")
    best = min(means, key=means.get)
    print(f"  -> best mode on validation: {best}")
    print(best)


def nested(out_dir, run_dirs):
    losses = {d: val_losses(d) for d in run_dirs}
    folds = sorted(set.intersection(*(set(v) for v in losses.values())))
    if not folds:
        raise SystemExit("no fold is complete in every run")
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(os.path.join(run_dirs[0], "config.json"), os.path.join(out_dir, "config.json"))
    names = [os.path.basename(os.path.normpath(d)) for d in run_dirs]
    with open(os.path.join(out_dir, "lr_choice.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fold"] + [f"val_loss {n}" for n in names] + ["chosen"])
        for k in folds:
            best = min(run_dirs, key=lambda d: losses[d][k])
            for name in (f"fold_{k}_preds.csv", f"fold_{k}_val_preds.csv", f"fold_{k}_metrics.json"):
                src = os.path.join(best, name)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(out_dir, name))
            w.writerow([k] + [f"{losses[d][k]:.4f}" for d in run_dirs] + [os.path.basename(os.path.normpath(best))])
            print(f"  fold {k}: chosen {os.path.basename(os.path.normpath(best))} "
                  f"(val loss {losses[best][k]:.4f})")
    for d, n in zip(run_dirs, names):
        print(f"  {n}: mean best validation loss {np.mean([losses[d][k] for k in folds]):.4f}")
    print(f"  -> per-fold selection written to {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("choose")
    c.add_argument("runs", nargs="+", help="MODE=RUN_DIR")
    n = sub.add_parser("nested")
    n.add_argument("--out", required=True)
    n.add_argument("runs", nargs="+")
    args = ap.parse_args()
    if args.cmd == "choose":
        choose(args.runs)
    else:
        nested(args.out, args.runs)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
