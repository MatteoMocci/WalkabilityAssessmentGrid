"""Compute per-run mean/std summaries from per-fold metric CSV outputs."""

import argparse
import csv
import math
import os
from collections import defaultdict
from statistics import mean, pstdev


EXPECTED_FIELDS = [
    "run",
    "fold",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "mean_absolute_error",
    "one_off_accuracy",
]


def read_fold_metrics(path: str):
    """
    Read fold_metrics_*.csv robustly, even if a header row is misplaced.
    Yields dicts with EXPECTED_FIELDS keys; values for metrics are floats.
    """
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        first = True
        for row in r:
            if not row:
                continue
            # Skip header if present anywhere
            if row[0] == "run":
                first = False
                continue
            # If the first row is not header but has the right length, accept it
            if len(row) < len(EXPECTED_FIELDS):
                # Skip malformed lines
                continue
            rows.append({
                "run": row[0],
                "fold": int(row[1]),
                "accuracy": float(row[2]),
                "precision": float(row[3]),
                "recall": float(row[4]),
                "f1": float(row[5]),
                "mean_absolute_error": float(row[6]),
                "one_off_accuracy": float(row[7]),
            })
            first = False
    return rows


def compute_stats(rows):
    by_run = defaultdict(lambda: defaultdict(list))
    for r in rows:
        run = r["run"]
        for k in ("accuracy", "precision", "recall", "f1", "mean_absolute_error", "one_off_accuracy"):
            by_run[run][k].append(float(r[k]))
    out = []
    for run, metrics in by_run.items():
        rec = {"run": run, "n_folds": len(next(iter(metrics.values()), []))}
        for k, vals in metrics.items():
            # population std (pstdev) across folds; change to stdev if desired
            rec[f"{k}_mean"] = mean(vals) if vals else 0.0
            rec[f"{k}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
        out.append(rec)
    # Sort for stable output
    out.sort(key=lambda d: d["run"])
    return out


def write_stats(path_out: str, stats):
    fieldnames = [
        "run",
        "accuracy_mean",
        "accuracy_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
        "mean_absolute_error_mean",
        "mean_absolute_error_std",
        "one_off_accuracy_mean",
        "one_off_accuracy_std",
        "n_folds",
    ]
    with open(path_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        # Format all numeric metrics to 3 decimals; keep n_folds as integer
        for rec in stats:
            out_row = {}
            for k in fieldnames:
                v = rec.get(k, "")
                if k == "run":
                    out_row[k] = v
                elif k == "n_folds":
                    try:
                        out_row[k] = int(v)
                    except Exception:
                        out_row[k] = v
                else:
                    # format floats to 3 decimals; pass through non-finite
                    try:
                        fv = float(v)
                        if math.isfinite(fv):
                            out_row[k] = f"{fv:.3f}"
                        else:
                            out_row[k] = "nan"
                    except Exception:
                        out_row[k] = v
            w.writerow(out_row)


def main():
    p = argparse.ArgumentParser(description="Compute CV mean/std from per-fold metrics")
    p.add_argument(
        "--fold-metrics",
        default="fold_metrics_full.csv",
        help="Path to per-fold metrics CSV (default: fold_metrics_full.csv)",
    )
    p.add_argument(
        "--out",
        default="cv_summary_full_with_std.csv",
        help="Output CSV with mean/std (default: cv_summary_full_with_std.csv)",
    )
    args = p.parse_args()

    if not os.path.exists(args.fold_metrics):
        raise SystemExit(f"Missing fold metrics file: {args.fold_metrics}")

    rows = read_fold_metrics(args.fold_metrics)
    if not rows:
        raise SystemExit("No rows read from fold metrics; nothing to do.")
    stats = compute_stats(rows)
    write_stats(args.out, stats)
    print(f"Wrote stats for {len(stats)} runs to {args.out}")


if __name__ == "__main__":
    main()
