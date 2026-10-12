"""Decision-level late fusion from saved predictions (no training).

For each fold, averages the cumulative probabilities P(Y > k) of a street-only and an
aerial-only ordinal run on the same points (test and validation files), and writes a run
directory that evaluate_runs.py reads like any other (outputs stored as logits).

Usage:
    python fuse_predictions.py --street results/grid/street_swin_base_ordinal \
        --aerial results/grid/aerial_swin_base_ordinal --out results/grid/avg_swin_base_ordinal
"""

import argparse
import csv
import glob
import json
import os
import re

import numpy as np

from metrics_utils import cumulative_from_output


def read(path):
    """{point_id: row} and cumulative probabilities keyed by point_id."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cols = sorted((c for c in rows[0] if c.startswith("out_")), key=lambda c: int(c[4:]))
    cum = cumulative_from_output(np.array([[float(r[c]) for c in cols] for r in rows]), "ordinal")
    return {r["point_id"]: (r, c) for r, c in zip(rows, cum)}


def fuse_file(street_path, aerial_path, out_path):
    s, a = read(street_path), read(aerial_path)
    if set(s) != set(a):
        raise SystemExit(f"street and aerial predictions cover different points: {street_path}")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["point_id", "fold", "city", "rater_lo", "rater_hi"] + [f"out_{j}" for j in range(4)]
                   + [f"p_above_{j}" for j in range(1, 5)] + ["pred_class", "score"])
        for pid, (row, cs) in s.items():
            c = np.clip(0.5 * (cs + a[pid][1]), 1e-6, 1 - 1e-6)
            w.writerow([pid, row["fold"], row["city"], row["rater_lo"], row["rater_hi"]]
                       + [f"{v:.5f}" for v in np.log(c / (1 - c))] + [f"{v:.5f}" for v in c]
                       + [int(1 + (c > 0.5).sum()), f"{1 + c.sum():.4f}"])


def main():
    ap = argparse.ArgumentParser(description="Average street-only and aerial-only cumulative probabilities")
    ap.add_argument("--street", required=True)
    ap.add_argument("--aerial", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"head": "ordinal", "view": "average", "street_run": args.street, "aerial_run": args.aerial}, f,
                  indent=2)
    n = 0
    for p in glob.glob(os.path.join(args.street, "fold_*_preds.csv")):
        name = os.path.basename(p)
        other = os.path.join(args.aerial, name)
        if os.path.exists(other):
            fuse_file(p, other, os.path.join(args.out, name))
            n += re.fullmatch(r"fold_\d+_preds\.csv", name) is not None
    print(f"{args.out}: {n} folds averaged")


if __name__ == "__main__":
    main()
