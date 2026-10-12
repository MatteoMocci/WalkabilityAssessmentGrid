"""Compare rules for turning cumulative probabilities into a 1-5 class, on saved test predictions.

Decoders (applied to P(Y > k), k = 1..4):
  strict      1 + sum_k I[P(Y > k) > 0.5]                      (current rule)
  round       round(1 + sum_k P(Y > k)), halves rounded up      (rounded expected score)
  tuned_prob  1 + sum_k I[P(Y > k) > t_k], t_k tuned for QWK
  tuned_score 1 + sum_k I[score > c_k], increasing cut points c_k on the expected score, tuned for QWK
  argmax      CE head only, reference

No retraining. Validation predictions were not saved, so thresholds for test fold k are tuned
on the pooled test predictions of the other folds (cross-fitting): fold k's labels never
choose its own thresholds. Metrics use both raters (soft confusion, as everywhere else).

Usage:
    python compare_decoders.py results/street_swin_base_ordinal results/street_swin_base_ce
"""

import argparse
import csv
import glob
import json
import os
import re

import numpy as np
from scipy.stats import wilcoxon

from metrics_utils import cumulative_from_output, discrete_metrics_vs_raters, soft_confusion, tune_thresholds

GRID_PROB = np.round(np.arange(0.05, 0.951, 0.01), 2)
GRID_SCORE = np.round(np.arange(1.0, 5.001, 0.01), 2)


def load_run(run_dir):
    """Return head and {fold: (cum, outputs, lo, hi)} from a run's prediction files."""
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        head = json.load(f)["head"]
    folds = {}
    for p in glob.glob(os.path.join(run_dir, "fold_*_preds.csv")):
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        cols = sorted((c for c in rows[0] if c.startswith("out_")), key=lambda c: int(c[4:]))
        out = np.array([[float(r[c]) for c in cols] for r in rows])
        folds[int(re.search(r"fold_(\d+)_preds", p).group(1))] = (
            cumulative_from_output(out, head), out,
            np.array([int(r["rater_lo"]) for r in rows]), np.array([int(r["rater_hi"]) for r in rows]))
    return head, folds


def decode_counts(values, cuts):
    """1 + number of cut points exceeded (values [N] against cuts [4], or [N,4] against [4])."""
    v = values[:, None] if values.ndim == 1 else values
    return 1 + (v > np.asarray(cuts)[None, :]).sum(axis=1)


def qwk(pred, lo, hi):
    return discrete_metrics_vs_raters(pred, lo, hi)["qwk"]


def tune(values, lo, hi, start, grid, increasing):
    """Coordinate ascent on the 4 cut points to maximise QWK on the tuning set."""
    cuts = np.array(start, dtype=float)
    best = qwk(decode_counts(values, cuts), lo, hi)
    for _ in range(4):
        changed = False
        for k in range(4):
            lo_b = cuts[k - 1] if (increasing and k > 0) else -np.inf
            hi_b = cuts[k + 1] if (increasing and k < 3) else np.inf
            for c in grid[(grid > lo_b) & (grid < hi_b)]:
                trial = cuts.copy()
                trial[k] = c
                s = qwk(decode_counts(values, trial), lo, hi)
                if s > best + 1e-9:
                    best, cuts, changed = s, trial, True
        if not changed:
            break
    return cuts


def fold_report(pred, lo, hi):
    """Hard-class metrics plus per-class recall and predicted-class share."""
    m = discrete_metrics_vs_raters(pred, lo, hi)
    cm = soft_confusion(pred, lo, hi)
    recall = np.divide(np.diag(cm), cm.sum(1), out=np.full(5, np.nan), where=cm.sum(1) > 0)
    m.update({f"recall_{c}": float(recall[c - 1]) for c in range(1, 6)})
    m.update({f"share_{c}": float(np.mean(pred == c)) for c in range(1, 6)})
    return m


def analyse(run_dir):
    head, folds = load_run(run_dir)
    names = ["strict", "round", "tuned_prob", "tuned_score"] + (["argmax"] if head == "ce" else [])
    per = {n: {} for n in names}
    preds = {n: [] for n in names}
    cuts_used = {"tuned_prob": [], "tuned_score": []}
    for k in sorted(folds):
        cum, out, lo, hi = folds[k]
        score = 1.0 + cum.sum(axis=1)
        others = [j for j in folds if j != k]
        t_cum = np.concatenate([folds[j][0] for j in others])
        t_lo = np.concatenate([folds[j][2] for j in others])
        t_hi = np.concatenate([folds[j][3] for j in others])
        t_prob = tune_thresholds(t_cum, t_lo, t_hi, GRID_PROB)
        t_score = tune(1.0 + t_cum.sum(axis=1), t_lo, t_hi, [1.5, 2.5, 3.5, 4.5], GRID_SCORE, increasing=True)
        cuts_used["tuned_prob"].append(t_prob)
        cuts_used["tuned_score"].append(t_score)
        fold_preds = {
            "strict": decode_counts(cum, [0.5] * 4),
            "round": np.clip(np.floor(score + 0.5), 1, 5).astype(int),
            "tuned_prob": decode_counts(cum, t_prob),
            "tuned_score": decode_counts(score, t_score),
        }
        if head == "ce":
            fold_preds["argmax"] = out.argmax(axis=1) + 1
        for n in names:
            per[n][k] = fold_report(fold_preds[n], lo, hi)
            preds[n].append((fold_preds[n], lo, hi))

    keys = ["qwk", "mean_absolute_error", "accuracy", "one_off_accuracy", "f1", "amae",
            "recall_1", "recall_5", "share_1", "share_5"]
    lines = [f"=== {os.path.basename(os.path.normpath(run_dir))} (head={head}, {len(folds)} folds; "
             f"mean ± SD over folds; p = paired Wilcoxon vs strict)", ""]
    lines.append(f"{'metric':22s}" + "".join(f"{n:>24s}" for n in names))
    for key in keys:
        row = f"{key:22s}"
        for n in names:
            v = np.array([per[n][k][key] for k in sorted(folds)])
            cell = f"{np.mean(v):.3f}±{np.std(v, ddof=1):.3f}"
            if n != "strict":
                d = v - np.array([per["strict"][k][key] for k in sorted(folds)])
                p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                cell += f" (p={p:.3f})"
            row += f"{cell:>24s}"
        lines.append(row)
    lines.append("")
    lines.append("raters' class shares: " + " ".join(
        f"{c}:{np.mean(np.r_[np.concatenate([x[1] for x in preds['strict']]), np.concatenate([x[2] for x in preds['strict']])] == c):.3f}"
        for c in range(1, 6)))
    for n in ("tuned_prob", "tuned_score"):
        c = np.array(cuts_used[n])
        lines.append(f"{n} cut points (mean over folds): " + " ".join(f"{x:.2f}" for x in c.mean(0))
                     + "   (SD " + " ".join(f"{x:.2f}" for x in c.std(0)) + ")")
    lines.append("")
    for n in names:
        p = np.concatenate([x[0] for x in preds[n]])
        lo = np.concatenate([x[1] for x in preds[n]])
        hi = np.concatenate([x[2] for x in preds[n]])
        cm = soft_confusion(p, lo, hi)
        cm = 100 * cm / cm.sum(1, keepdims=True)
        lines.append(f"confusion, {n} (rows = rater class, % of row; cols = predicted 1..5)")
        lines += [f"  {r + 1}: " + " ".join(f"{x:5.1f}" for x in cm[r]) for r in range(5)]
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Compare decoding rules on saved test predictions")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=os.path.join("results", "decoding_comparison.txt"))
    args = ap.parse_args()
    text = "\n".join(analyse(r) for r in args.runs)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
