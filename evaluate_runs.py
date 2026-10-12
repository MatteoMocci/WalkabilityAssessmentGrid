"""Recompute the full metric set for finished runs from their saved test predictions.

No retraining: reads results/<run>/fold_<k>_preds.csv, recomputes every metric in
metrics_utils.ordinal_metrics (discrete, continuous, probabilistic, human reference)
for the 0.5 decoding rule and for QWK-tuned thresholds (tuned_*). Thresholds come from
the run's validation predictions (fold_<k>_val_preds.csv) when present; older runs
without them are cross-fitted on the other folds' test predictions. Writes per run:
  metrics_by_fold.csv   one row per fold
  summary_full.csv      mean, sample SD and number of folds per metric
With two or more runs it also writes <out-dir>/comparison.md (and .csv): mean +- SD per
run next to the human reference, and a paired Wilcoxon test of each run against the
first one over the folds.

Usage:
    python evaluate_runs.py results/street_swin_base_ordinal results/street_swin_base_ce
"""

import argparse
import csv
import glob
import json
import os
import re

import numpy as np
from scipy.stats import wilcoxon

from metrics_utils import cumulative_from_output, decode_thresholds, ordinal_metrics, tune_thresholds

DISCRETE_KEYS = ("accuracy", "f1", "mean_absolute_error", "one_off_accuracy", "amae", "mmae", "qwk",
                 "alpha_ordinal", "alpha_interval", "recall_1", "recall_5")
# (section, metric, label, higher is better); human_<metric> is shown next to it when it exists
REPORT = [
    ("Discrete class vs raters", "accuracy", "Accuracy (exact)", True),
    ("Discrete class vs raters", "one_off_accuracy", "One-off accuracy", True),
    ("Discrete class vs raters", "f1", "Macro-F1", True),
    ("Discrete class vs raters", "mean_absolute_error", "MAE", False),
    ("Discrete class vs raters", "amae", "AMAE (macro MAE)", False),
    ("Discrete class vs raters", "mmae", "MMAE (worst-class MAE)", False),
    ("Discrete class vs raters", "qwk", "Quadratic weighted kappa", True),
    ("Discrete class vs raters", "alpha_ordinal", "Krippendorff alpha (ordinal)", True),
    ("Discrete class vs raters", "alpha_interval", "Krippendorff alpha (interval)", True),
    ("Discrete class vs raters", "recall_1", "Recall class 1", True),
    ("Discrete class vs raters", "recall_5", "Recall class 5", True),
    ("Discrete class, QWK-tuned thresholds", "tuned_qwk", "Quadratic weighted kappa", True),
    ("Discrete class, QWK-tuned thresholds", "tuned_mean_absolute_error", "MAE", False),
    ("Discrete class, QWK-tuned thresholds", "tuned_accuracy", "Accuracy (exact)", True),
    ("Discrete class, QWK-tuned thresholds", "tuned_one_off_accuracy", "One-off accuracy", True),
    ("Discrete class, QWK-tuned thresholds", "tuned_f1", "Macro-F1", True),
    ("Discrete class, QWK-tuned thresholds", "tuned_amae", "AMAE (macro MAE)", False),
    ("Discrete class, QWK-tuned thresholds", "tuned_recall_1", "Recall class 1", True),
    ("Discrete class, QWK-tuned thresholds", "tuned_recall_5", "Recall class 5", True),
    ("Continuous score vs mean rating", "score_spearman", "Spearman rho", True),
    ("Continuous score vs mean rating", "score_kendall", "Kendall tau-b", True),
    ("Continuous score vs mean rating", "score_pearson", "Pearson r", True),
    ("Continuous score vs mean rating", "score_r2", "R2", True),
    ("Continuous score vs mean rating", "score_mae", "MAE", False),
    ("Continuous score vs mean rating", "score_rmse", "RMSE", False),
    ("Probabilities vs soft targets", "cum_bce", "Cumulative BCE", False),
    ("Probabilities vs soft targets", "brier_t1", "Brier P(Y>1)", False),
    ("Probabilities vs soft targets", "brier_t2", "Brier P(Y>2)", False),
    ("Probabilities vs soft targets", "brier_t3", "Brier P(Y>3)", False),
    ("Probabilities vs soft targets", "brier_t4", "Brier P(Y>4)", False),
    ("Probabilities vs soft targets", "auc_t1", "AUC P(Y>1), unanimous points", True),
    ("Probabilities vs soft targets", "auc_t2", "AUC P(Y>2), unanimous points", True),
    ("Probabilities vs soft targets", "auc_t3", "AUC P(Y>3), unanimous points", True),
    ("Probabilities vs soft targets", "auc_t4", "AUC P(Y>4), unanimous points", True),
    ("Probabilities vs soft targets", "sd_vs_disagreement_spearman", "Predicted spread vs rater disagreement (rho)", True),
    ("Probabilities vs soft targets", "monotonicity_violation_pct", "Monotonicity violations (%)", False),
]


def read_preds(path: str, head: str):
    """(cumulative probabilities, raw outputs, rater_lo, rater_hi) from a prediction file."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out_cols = sorted((c for c in rows[0] if c.startswith("out_")), key=lambda c: int(c[4:]))
    outputs = np.array([[float(r[c]) for c in out_cols] for r in rows])
    return (cumulative_from_output(outputs, head), outputs,
            np.array([int(r["rater_lo"]) for r in rows]), np.array([int(r["rater_hi"]) for r in rows]))


def fold_metrics(test, thresholds, head: str, fold: int) -> dict:
    """All metrics for one fold: 0.5 rule, tuned thresholds (tuned_*), CE argmax (argmax_*)."""
    cum, outputs, lo, hi = test
    m = ordinal_metrics(cum, lo, hi)
    tuned = ordinal_metrics(cum, lo, hi, pred_override=decode_thresholds(cum, thresholds))
    m.update({f"tuned_{k}": tuned[k] for k in DISCRETE_KEYS})
    m.update({f"threshold_{j + 1}": float(t) for j, t in enumerate(thresholds)})
    if head == "ce":
        argmax = ordinal_metrics(cum, lo, hi, pred_override=outputs.argmax(axis=1) + 1)
        m.update({f"argmax_{k}": argmax[k] for k in DISCRETE_KEYS})
    m["fold"] = fold
    return m


def evaluate_run(run_dir: str):
    """Recompute and write per-fold and summary metrics for one run; return (name, list of fold dicts, source)."""
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        head = json.load(f)["head"]
    paths = glob.glob(os.path.join(run_dir, "fold_*_preds.csv"))
    tests = {int(re.search(r"fold_(\d+)_preds", p).group(1)): read_preds(p, head)
             for p in paths if not p.endswith("_val_preds.csv")}
    if not tests:
        raise SystemExit(f"no fold_*_preds.csv in {run_dir}")
    folds, sources = [], set()
    for k in sorted(tests):
        val_path = os.path.join(run_dir, f"fold_{k}_val_preds.csv")
        if os.path.exists(val_path):
            cum, _, lo, hi = read_preds(val_path, head)
            sources.add("validation fold")
        else:  # older runs: cross-fit on the other folds' test predictions
            others = [tests[j] for j in tests if j != k]
            cum, lo, hi = (np.concatenate([o[i] for o in others]) for i in (0, 2, 3))
            sources.add("cross-fitted on other folds")
        folds.append(fold_metrics(tests[k], tune_thresholds(cum, lo, hi), head, k))
    keys = ["fold"] + [k for k in folds[0] if k != "fold"]
    with open(os.path.join(run_dir, "metrics_by_fold.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(folds)
    with open(os.path.join(run_dir, "summary_full.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "std", "n_folds"])
        for k in keys[1:]:
            v = np.array([m[k] for m in folds], dtype=float)
            w.writerow([k, f"{np.nanmean(v):.4f}", f"{np.nanstd(v, ddof=1):.4f}", int(np.sum(~np.isnan(v)))])
    print(f"{run_dir}: {len(folds)} folds -> metrics_by_fold.csv, summary_full.csv")
    return os.path.basename(os.path.normpath(run_dir)), folds, " + ".join(sorted(sources))


def mean_sd(folds, key):
    """'mean +- sd' over folds, or '' if the metric is missing."""
    if key not in folds[0]:
        return ""
    v = np.array([m[key] for m in folds], dtype=float)
    return f"{np.nanmean(v):.3f} ± {np.nanstd(v, ddof=1):.3f}"


def write_comparison(runs, out_dir):
    """Markdown + CSV table: every run, the human reference, and paired tests against the first run."""
    sources = {n: src for n, _, src in runs}
    runs = [(n, f) for n, f, _ in runs]
    names = [n for n, _ in runs]
    ref_name, ref = runs[0]
    by_fold = {n: {m["fold"]: m for m in f} for n, f in runs}
    common = sorted(set.intersection(*(set(d) for d in by_fold.values())))
    header = ["Metric"] + names + ["Human (rater vs rater)"] + [f"Δ {n} − {ref_name} (p)" for n in names[1:]]
    lines, rows, section = [], [], None
    for sec, key, label, _ in REPORT:
        if sec != section:
            section = sec
            lines += ["", f"**{sec}**", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        human = mean_sd(ref, f"human_{key}")
        cells = [label] + [mean_sd(f, key) for _, f in runs] + [human]
        for n in names[1:]:
            a = np.array([by_fold[n][k][key] for k in common], dtype=float)
            b = np.array([by_fold[ref_name][k][key] for k in common], dtype=float)
            d = a - b
            ok = ~np.isnan(d)
            p = wilcoxon(d[ok]).pvalue if ok.sum() >= 5 and np.any(d[ok] != 0) else float("nan")
            cells.append(f"{np.mean(d[ok]):+.3f} ({p:.3f})")
        lines.append("| " + " | ".join(cells) + " |")
        rows.append(cells)
    if any(k.startswith("argmax_") for k in ref[0]) or any(any(k.startswith("argmax_") for k in f[0]) for _, f in runs):
        lines += ["", "CE argmax decision (reference only): " + "; ".join(
            f"{n}: accuracy {mean_sd(f, 'argmax_accuracy')}, QWK {mean_sd(f, 'argmax_qwk')}"
            for n, f in runs if "argmax_accuracy" in f[0])]
    lines += ["", f"Folds: {len(common)} paired. Human column = one rater scored against the other on the same "
                  f"test points. Correlation ceiling for the continuous score, sqrt(ICC(1,2)): "
                  f"{mean_sd(ref, 'human_corr_ceiling')}. p = paired Wilcoxon signed-rank over folds.",
              "QWK-tuned thresholds on P(Y>k) were tuned on: " + "; ".join(f"{n}: {s}" for n, s in sources.items())]
    md = "\n".join(lines).lstrip() + "\n"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "comparison.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(out_dir, "comparison.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(md)
    print(f"-> {os.path.join(out_dir, 'comparison.md')}")


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Recompute full metrics from saved predictions")
    ap.add_argument("runs", nargs="+", help="run directories (the first is the reference for paired tests)")
    ap.add_argument("--out-dir", default=None, help="where comparison.md goes (default: parent of the first run)")
    args = ap.parse_args()
    runs = [evaluate_run(r) for r in args.runs]
    out_dir = args.out_dir or os.path.dirname(os.path.normpath(args.runs[0]))
    write_comparison(runs, out_dir)


if __name__ == "__main__":
    main()
