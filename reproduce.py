"""
One-command reproducibility runner for the study.

This script orchestrates the end-to-end workflow using relative paths:
1) Train/evaluate models with main.py (produces fold_metrics_full.csv).
2) Compute CV mean/std summaries (Table 1 inputs).
3) Reshape metrics for GLMM and run GLMM analysis (Table 2 + Figure 8).

Use --skip-train if you already have fold_metrics_full.csv.
Use --zero-shot to evaluate immediately without any training steps.
Use --skip-glmm if you only want model metrics and Table 1.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path.cwd().resolve()
PY = sys.executable


def _run(cmd: list[str], *, cwd: Path, label: str) -> None:
    """
    Run a subprocess command and exit on failure.

    Steps:
    1) Print the command for traceability.
    2) Execute with inherited stdout/stderr.
    3) Raise if the command fails.
    """
    print(f"[reproduce] {label}: {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(cwd))


def _ensure_file(path: Path, hint: str) -> None:
    """
    Ensure a required file exists or exit with a helpful hint.
    """
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}\nHint: {hint}")


def _resolve_rscript() -> str:
    """
    Resolve an executable path for Rscript.

    Resolution order:
    1) Environment override via RSCRIPT_BIN.
    2) PATH lookup.
    3) Common Windows install locations under Program Files.
    """
    env_bin = os.getenv("RSCRIPT_BIN", "").strip()
    if env_bin:
        if os.path.isfile(env_bin):
            return env_bin
        raise SystemExit(
            f"RSCRIPT_BIN is set but not found: {env_bin}\n"
            "Hint: set RSCRIPT_BIN to a valid Rscript executable path."
        )

    in_path = shutil.which("Rscript")
    if in_path:
        return in_path

    candidates: list[Path] = []
    for env_name in ("ProgramFiles", "ProgramFiles(x86)"):
        root = os.getenv(env_name, "")
        if not root:
            continue
        r_root = Path(root) / "R"
        if not r_root.exists():
            continue
        # Typical layouts: C:\\Program Files\\R\\R-x.y.z\\bin\\Rscript.exe
        candidates.extend(r_root.glob("R-*\\bin\\Rscript.exe"))
        candidates.extend(r_root.glob("R-*\\bin\\x64\\Rscript.exe"))

    if candidates:
        # Pick latest version by path sort (R-4.3.3 > R-4.2.0, etc.).
        return str(sorted(candidates)[-1])

    raise SystemExit(
        "Rscript was not found.\n"
        "Install R and ensure Rscript is on PATH, or set RSCRIPT_BIN explicitly.\n"
        "PowerShell example:\n"
        '$env:RSCRIPT_BIN = "C:\\Program Files\\R\\R-4.5.1\\bin\\Rscript.exe"'
    )


def main() -> int:
    """
    CLI entry point for reproducibility workflow.

    Steps:
    1) Optionally run training/evaluation grid (main.py).
    2) Compute per-run mean/std summaries for Table 1.
    3) Build mode/model/loss metrics and run GLMM for Table 2 + Figure 8.
    """
    ap = argparse.ArgumentParser(description="Reproduce study findings end-to-end")
    ap.add_argument("--skip-train", action="store_true", help="Skip model training/evaluation")
    ap.add_argument(
        "--zero-shot",
        action="store_true",
        help="Run evaluation only (no training steps) in main.py",
    )
    ap.add_argument("--skip-glmm", action="store_true", help="Skip GLMM analysis (Table 2/Figure 8)")
    ap.add_argument("--reset-progress", action="store_true", help="Reset training progress before running")
    args = ap.parse_args()
    if args.skip_train and args.zero_shot:
        raise SystemExit("--skip-train and --zero-shot cannot be used together.")

    # Paths (relative to repo root)
    fold_metrics = REPO_ROOT / "fold_metrics_full.csv"
    cv_summary_std = REPO_ROOT / "cv_summary_full_with_std.csv"
    mode_loss_metrics = REPO_ROOT / "output" / "metrics_by_mode_loss_fold.csv"

    # 1) Training/evaluation grid
    if not args.skip_train:
        cmd = [PY, "main.py"]
        if args.reset_progress:
            cmd.append("--reset-progress")
        if args.zero_shot:
            cmd.append("--zero-shot")
        _run(cmd, cwd=REPO_ROOT, label="train/eval grid")
        _ensure_file(
            fold_metrics,
            "Training should produce fold_metrics_full.csv in the repo root.",
        )

    # 2) Table 1 inputs: mean/std per run
    _ensure_file(
        fold_metrics,
        "Run training first or pass --skip-train only if fold_metrics_full.csv exists.",
    )
    _run(
        [
            PY,
            "tools/compute_cv_stats.py",
            "--fold-metrics",
            "fold_metrics_full.csv",
            "--out",
            "cv_summary_full_with_std.csv",
        ],
        cwd=REPO_ROOT,
        label="compute CV mean/std",
    )
    _ensure_file(cv_summary_std, "Expected cv_summary_full_with_std.csv to be created.")

    # 3) GLMM inputs + analysis (Table 2 + Figure 8)
    _run(
        [
            PY,
            "tools/make_mode_loss_fold_metrics.py",
            "--input",
            "fold_metrics_full.csv",
            "--out",
            "output/metrics_by_mode_loss_fold.csv",
        ],
        cwd=REPO_ROOT,
        label="reshape metrics for GLMM",
    )
    _ensure_file(mode_loss_metrics, "Expected output/metrics_by_mode_loss_fold.csv to be created.")

    if not args.skip_glmm:
        rscript = _resolve_rscript()
        _run([rscript, "glmm_walk.R"], cwd=REPO_ROOT, label="GLMM + Figure 8")

    print("[reproduce] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
