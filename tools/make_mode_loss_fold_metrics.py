"""Reshape fold metrics into mode/model/loss format for statistical analysis."""

import argparse
import csv
import os


IN_FIELDS = [
    "run",
    "fold",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "mean_absolute_error",
    "one_off_accuracy",
]

OUT_FIELDS = [
    "mode",
    "model",
    "loss",
    "fold",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "mean_absolute_error",
    "one_off_accuracy",
]


def parse_run_key(run_key: str) -> tuple[str, str, str, str, str]:
    """Parse 'view_model_pre_loss_aug' allowing model to contain underscores.

    Returns: (mode, model, pre, loss, aug)
    """
    parts = run_key.split("_")
    if not parts:
        return "", "", "", "", ""
    mode = parts[0]
    # Find pretraining token (currently 'imagenet1k')
    try:
        pre_idx = parts.index("imagenet1k")
    except ValueError:
        # Fallback: assume fixed 5-part format without underscores in model
        if len(parts) >= 5:
            return parts[0], parts[1], parts[2], parts[3], parts[4]
        # Last resort
        return mode, "", "", parts[-2] if len(parts) >= 2 else "", parts[-1] if parts else ""
    model = "_".join(parts[1:pre_idx]) if pre_idx > 1 else ""
    pre = parts[pre_idx]
    loss = parts[pre_idx + 1] if pre_idx + 1 < len(parts) else ""
    aug = parts[pre_idx + 2] if pre_idx + 2 < len(parts) else ""
    return mode, model, pre, loss, aug


def read_fold_metrics(path: str):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            if row[0] == "run":
                # header line anywhere -> skip
                continue
            if len(row) < len(IN_FIELDS):
                continue
            rows.append(row)
    return rows


def write_mode_loss_metrics(rows, out_path: str):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        for row in rows:
            run_key = row[0]
            fold = row[1]
            mode, model, _pre, loss, _aug = parse_run_key(run_key)
            # format metrics to 3 decimals
            def fmt(x: str) -> str:
                try:
                    v = float(x)
                    return f"{v:.3f}"
                except Exception:
                    return x

            w.writerow(
                {
                    "mode": mode,
                    "model": model,
                    "loss": loss,
                    "fold": int(fold),
                    "accuracy": fmt(row[2]),
                    "precision": fmt(row[3]),
                    "recall": fmt(row[4]),
                    "f1": fmt(row[5]),
                    "mean_absolute_error": fmt(row[6]),
                    "one_off_accuracy": fmt(row[7]),
                }
            )


def main():
    p = argparse.ArgumentParser(description="Make per-fold metrics with mode and loss")
    p.add_argument(
        "--input",
        default="fold_metrics_full.csv",
        help="Input per-fold metrics CSV (default: fold_metrics_full.csv)",
    )
    p.add_argument(
        "--out",
        default="metrics_by_mode_loss_fold.csv",
        help="Output CSV (default: metrics_by_mode_loss_fold.csv)",
    )
    args = p.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Missing input file: {args.input}")
    rows = read_fold_metrics(args.input)
    if not rows:
        raise SystemExit("No data rows found in input.")
    write_mode_loss_metrics(rows, args.out)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
