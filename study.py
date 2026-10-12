#!/usr/bin/env python3
"""One entry point for the experiments of this repository, on Linux, Windows and macOS.

    python study.py status                      # completed folds of the backbone x view grid
    python study.py first-experiment            # Swin street: ordinal head, then the soft-label CE baseline
    python study.py finetune                    # full / frozen / lpft, then a learning-rate grid on the best mode
    python study.py grid                        # 6 backbones x street / aerial / avg / late fusion / dual encoder
    python study.py grid --backbones swin_base vgg16

Every study resumes where it stopped: finished folds are skipped (see run_status.py).
Arguments this script does not know are passed to run_ordinal.py, e.g.
    python study.py grid --device cpu --folds 0 --epochs 1      # quick check on a machine without GPU
Run it with the interpreter of the environment created by setup_env.py
(.venv/bin/python on Linux/macOS, .venv\\Scripts\\python.exe on Windows).
"""

import argparse
import datetime
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
BACKBONES = ["swin_base", "vit_base", "deit_base", "vgg16", "googlenet", "alexnet"]
GRID_VIEWS = ["street", "aerial", "avg", "late", "dual"]


class Study:
    """Shared logging and process handling: one timestamped log per study launch."""

    def __init__(self, out_dir, name):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.log_path = os.path.join(out_dir, f"{name}_{datetime.datetime.now():%Y%m%d_%H%M%S}.log")

    def log(self, msg):
        line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def tool(self, *args, echo=True):
        """Run a helper script; its output goes to the console (if echo) and the study log. Returns stdout."""
        res = subprocess.run([PY, *args], cwd=HERE, capture_output=True, text=True)
        if res.returncode != 0:
            sys.stderr.write(res.stdout + res.stderr)
            self.log(f"FAILED: {' '.join(args)}")
            raise SystemExit(1)
        if echo:
            print(res.stdout, end="", flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(res.stdout)
        return res.stdout

    def train(self, label, args):
        """One run_ordinal.py run (progress bars on the console, details in the run's train.log)."""
        self.log(f"[{label}] start")
        t0 = time.time()
        code = subprocess.run([PY, "run_ordinal.py", *args], cwd=HERE).returncode
        minutes = int((time.time() - t0) // 60)
        if code != 0:
            self.log(f"[{label}] FAILED after {minutes} min, see the run's train.log")
            raise SystemExit(1)
        self.log(f"[{label}] done in {minutes} min")


def first_experiment(opts, extra):
    """Swin (or --model) on street images: ordinal head, then the soft-label CE baseline, 10 folds each."""
    s = Study(opts.out_dir, f"experiment_{opts.model}")
    s.log(f"experiment start: model={opts.model} extra args: {' '.join(extra)}")
    for head in ("ordinal", "ce"):
        s.train(head, ["--head", head, "--model", opts.model, "--out-dir", opts.out_dir, *extra])
    s.log("experiment finished")


def finetune(opts, extra):
    """
    1) full / frozen / lpft at their default learning rates
    2) best mode = lowest mean validation loss (test folds never used)
    3) learning-rate grid on that mode
    4) per fold, the learning rate with the lowest validation loss (nested), then comparison tables
    """
    out, m = opts.out_dir, opts.model
    d = os.path.join(out, f"street_{m}_ordinal")
    s = Study(out, "study")
    s.log(f"fine-tuning study start: model={m} extra args: {' '.join(extra)}")
    base = ["--head", "ordinal", "--model", m, "--out-dir", out]
    s.train("full lr=5e-5", [*base, "--finetune", "full", "--lr", "5e-5", "--tag", "lr5e-5", *extra])
    s.train("frozen head_lr=1e-3", [*base, "--finetune", "frozen", "--head-lr", "1e-3", "--tag", "lr1e-3", *extra])
    s.train("lpft head_lr=1e-3 lr=5e-5",
            [*base, "--finetune", "lpft", "--head-lr", "1e-3", "--lr", "5e-5", "--tag", "lr5e-5", *extra])

    s.log("choosing the best mode on validation loss")
    best = s.tool("select_finetune.py", "choose", f"full={d}_lr5e-5", f"frozen={d}_frozen_lr1e-3",
                  f"lpft={d}_lpft_lr5e-5").strip().splitlines()[-1]
    s.log(f"best mode: {best}")
    prefix, flags, lr_flag, default, grid = {
        "full": (d, ["--finetune", "full"], "--lr", "5e-5", ["1e-5", "2e-5", "1e-4"]),
        "frozen": (f"{d}_frozen", ["--finetune", "frozen"], "--head-lr", "1e-3", ["3e-4", "3e-3", "1e-2"]),
        "lpft": (f"{d}_lpft", ["--finetune", "lpft", "--head-lr", "1e-3"], "--lr", "5e-5", ["1e-5", "2e-5", "1e-4"]),
    }[best]
    runs = [f"{prefix}_lr{default}"]
    for lr in grid:
        s.train(f"{best} {lr_flag}={lr}", [*base, *flags, lr_flag, lr, "--tag", f"lr{lr}", *extra])
        runs.append(f"{prefix}_lr{lr}")

    s.log("selecting the learning rate per fold on validation loss")
    selected = f"{prefix}_lrselected"
    s.tool("select_finetune.py", "nested", "--out", selected, *runs)
    s.tool("evaluate_runs.py", *runs, "--out-dir", os.path.join(out, "grid"), echo=False)
    s.tool("evaluate_runs.py", f"{d}_lr5e-5", f"{d}_frozen_lr1e-3", f"{d}_lpft_lr5e-5", selected, "--out-dir", out)
    s.log(f"fine-tuning study finished: {out}/comparison.md, {out}/grid/comparison.md, {selected}/lr_choice.csv")


def grid(opts, extra):
    """
    Backbones x views, ordinal head, full fine-tuning, views in order across all backbones:
    street and aerial (weights saved), avg (no training), late fusion, dual encoder; then comparisons.
    """
    out, bbs = opts.out_dir, opts.backbones
    run = lambda view, b: os.path.join(out, f"{view}_{b}_ordinal")
    s = Study(out, "grid")
    s.log(f"backbone grid start: backbones={' '.join(bbs)} extra args: {' '.join(extra)}")
    s.tool("run_status.py", out)
    base = ["--head", "ordinal", "--out-dir", out]
    for view in ("street", "aerial"):
        for b in bbs:
            s.train(f"{view} {b}", [*base, "--view", view, "--model", b, "--save-weights", *extra])
            s.tool("run_status.py", out, echo=False)
    for b in bbs:
        s.tool("fuse_predictions.py", "--street", run("street", b), "--aerial", run("aerial", b),
               "--out", run("avg", b))
    s.tool("run_status.py", out, echo=False)
    for view in ("late", "dual"):
        for b in bbs:
            s.train(f"{view} {b}", [*base, "--view", view, "--model", b, *extra])
            s.tool("run_status.py", out, echo=False)

    s.log("writing comparisons")
    for view in GRID_VIEWS:
        s.tool("evaluate_runs.py", *[run(view, b) for b in bbs], "--out-dir", os.path.join(out, "by_view", view),
               echo=False)
    for b in bbs:
        s.tool("evaluate_runs.py", *[run(v, b) for v in GRID_VIEWS], "--out-dir", os.path.join(out, "by_backbone", b),
               echo=False)
    s.tool("grid_overview.py", out)
    s.log(f"backbone grid finished: {out}/overview.txt, {out}/by_view/*/comparison.md, {out}/by_backbone/*/comparison.md")


def status(opts, extra):
    subprocess.run([PY, "run_status.py", opts.out_dir], cwd=HERE, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="study", required=True)
    p = sub.add_parser("first-experiment", help="Swin street: ordinal head vs soft-label CE")
    p.add_argument("--model", default="swin_base")
    p.add_argument("--out-dir", default="results")
    p.set_defaults(func=first_experiment)
    p = sub.add_parser("finetune", help="full / frozen / lpft, then a learning-rate grid on the best")
    p.add_argument("--model", default="swin_base")
    p.add_argument("--out-dir", default=os.path.join("results", "finetune"))
    p.set_defaults(func=finetune)
    p = sub.add_parser("grid", help="backbones x views")
    p.add_argument("--backbones", nargs="+", default=BACKBONES, choices=BACKBONES)
    p.add_argument("--out-dir", default=os.path.join("results", "grid"))
    p.set_defaults(func=grid)
    p = sub.add_parser("status", help="completed folds of the grid")
    p.add_argument("--out-dir", default=os.path.join("results", "grid"))
    p.set_defaults(func=status)
    opts, extra = ap.parse_known_args()
    os.chdir(HERE)
    opts.func(opts, extra)


if __name__ == "__main__":
    main()
