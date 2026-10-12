"""One-page overview of the backbone x view grid from each run's summary_full.csv.

For each key metric, prints a backbone x view matrix (mean over folds) and writes it to
<grid>/overview.txt as plain fixed-width text.

Usage:
    python grid_overview.py results/grid
"""

import argparse
import csv
import os

BACKBONES = ["swin_base", "vit_base", "deit_base", "vgg16", "googlenet", "alexnet"]
VIEWS = ["street", "aerial", "avg", "late", "dual"]
METRICS = [("qwk", "QWK, 0.5 rule"), ("tuned_qwk", "QWK, tuned thresholds"),
           ("mean_absolute_error", "MAE, 0.5 rule"), ("one_off_accuracy", "Within-one accuracy, 0.5 rule"),
           ("score_spearman", "Spearman, continuous score"), ("cum_bce", "Cumulative BCE")]


def load(grid, view, backbone):
    path = os.path.join(grid, f"{view}_{backbone}_ordinal", "summary_full.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return {r["metric"]: (float(r["mean"]), float(r["std"])) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("grid")
    args = ap.parse_args()
    data = {(v, b): load(args.grid, v, b) for v in VIEWS for b in BACKBONES}
    lines = ["Backbone x view overview (mean ± SD over 10 spatial folds; avg = average of street-only and "
             "aerial-only probabilities)", ""]
    for key, label in METRICS:
        lines.append(label)
        lines.append(f"  {'':10s}" + "".join(f"{v:>16s}" for v in VIEWS))
        for b in BACKBONES:
            cells = []
            for v in VIEWS:
                d = data[(v, b)]
                cells.append(f"{d[key][0]:.3f} ± {d[key][1]:.3f}" if d and key in d else "-")
            lines.append(f"  {b:10s}" + "".join(f"{c:>16s}" for c in cells))
        lines.append("")
    text = "\n".join(lines)
    with open(os.path.join(args.grid, "overview.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
