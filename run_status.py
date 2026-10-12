"""Which folds of which runs are complete, without opening the model weights.

A fold counts as complete when its metrics file parses and its test and validation
prediction files have the expected number of rows (the metrics file stores both counts).
For runs that must keep weights (street/aerial in the grid, needed by late fusion),
fold_<k>_best.pt must also be an intact archive: the check reads only the archive's index
at the end of the file, so a file truncated by a crash or power cut is detected quickly.

run_ordinal.py uses fold_complete() to decide which folds to skip. As a command it prints
(and writes to <grid>/status.txt) a backbone x view table of completed folds:
    python run_status.py results/grid
"""

import argparse
import json
import os
import zipfile

N_FOLDS = 10
BACKBONES = ["swin_base", "vit_base", "deit_base", "vgg16", "googlenet", "alexnet"]
VIEWS = ["street", "aerial", "avg", "late", "dual"]
VIEWS_WITH_WEIGHTS = ("street", "aerial")  # late fusion reuses these encoders


def _rows(path):
    """Data rows of a CSV file (header excluded), or -1 if missing."""
    if not os.path.exists(path):
        return -1
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


def weights_ok(path):
    """True when the saved weights exist and are a complete torch archive."""
    return os.path.exists(path) and os.path.getsize(path) > 0 and zipfile.is_zipfile(path)


def fold_complete(run_dir, k, need_weights=False):
    """True when fold k of run_dir finished and its files are intact."""
    try:
        with open(os.path.join(run_dir, f"fold_{k}_metrics.json"), encoding="utf-8") as f:
            m = json.load(f)
    except (OSError, ValueError):
        return False
    if _rows(os.path.join(run_dir, f"fold_{k}_preds.csv")) != m.get("n", -2):
        return False
    if "n_val" in m and _rows(os.path.join(run_dir, f"fold_{k}_val_preds.csv")) != m["n_val"]:
        return False
    return not need_weights or weights_ok(os.path.join(run_dir, f"fold_{k}_best.pt"))


def avg_complete(run_dir, k):
    """The averaged view has no metrics file: it is complete when its prediction file exists."""
    return _rows(os.path.join(run_dir, f"fold_{k}_preds.csv")) > 0


def status_table(grid):
    """Backbone x view table: completed folds out of 10 (weights in brackets where needed)."""
    lines = ["Completed folds per run (out of 10); [w] = folds whose saved weights are intact", ""]
    lines.append(f"  {'':10s}" + "".join(f"{v:>14s}" for v in VIEWS))
    todo = []
    for b in BACKBONES:
        cells = []
        for v in VIEWS:
            run = os.path.join(grid, f"{v}_{b}_ordinal")
            if v == "avg":
                done = sum(avg_complete(run, k) for k in range(N_FOLDS))
                cells.append(f"{done}/10")
            else:
                done = sum(fold_complete(run, k) for k in range(N_FOLDS))
                cell = f"{done}/10"
                if v in VIEWS_WITH_WEIGHTS:
                    w = sum(weights_ok(os.path.join(run, f"fold_{k}_best.pt")) for k in range(N_FOLDS))
                    cell += f" [{w}w]"
                cells.append(cell)
            if done < N_FOLDS:
                todo.append(f"{v} {b} ({done}/10)")
        lines.append(f"  {b:10s}" + "".join(f"{c:>14s}" for c in cells))
    lines += ["", "Still to do: " + (", ".join(todo) if todo else "nothing, the grid is complete")]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("grid", nargs="?", default=os.path.join("results", "grid"))
    args = ap.parse_args()
    text = status_table(args.grid)
    os.makedirs(args.grid, exist_ok=True)
    with open(os.path.join(args.grid, "status.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
