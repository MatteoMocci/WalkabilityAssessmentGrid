# WalkCNN Grid Workdir

This repository trains and evaluates image classifiers for **perceived walkability level prediction** (5 classes) using:

- street-view images
- satellite images
- dual/combined fusion of both views

The main workflow is a configurable grid in `main.py` with grouped cross-validation, early stopping, checkpointing, and resume support.

## What the repository does

- Builds matched street/satellite pairs from filenames containing coordinates.
- Splits data with `GroupKFold` so nearby coordinate groups stay together.
- Trains multiple backbones (CNNs + transformers) on four views:
  - `street`
  - `satellite`
  - `dual`
  - `combined`
- Computes metrics per fold (`accuracy`, `precision`, `recall`, `f1`, `MAE`, `one_off_accuracy`).
- Saves run-level and fold-level CSV outputs plus JSON metadata.

## Dataset expectations

`main.py` expects class-folder datasets under a base directory:

- `streetview/` (street originals)
- `satellite/` (satellite originals)
- `augmented-streetview/` (street augmented, optional depending on config)
- `augmented-satellite/` (satellite augmented, optional depending on config)

By default, the base directory is the repo root.  
You can override it with:

```powershell
$env:WALKCNN_BASE_DIR = "C:\path\to\datasets_root"
```

Filenames should include coordinate tokens (example pattern used by parser: `id_lat_lon_*.jpg`).

### Startup data checks

`main.py` now validates dataset folders at startup and prints actionable setup hints if data is missing.

It reports for each required root:
- whether it exists
- how many numeric class folders were found
- how many images were found

If setup is incomplete, it prints the expected layout and exits early:

```text
[setup] Dataset configuration is incomplete.
[setup] Expected folder layout:
  <BASE>/streetview/<class_id>/*.jpg
  <BASE>/satellite/<class_id>/*.jpg
  <BASE>/augmented-streetview/<class_id>/*.jpg
  <BASE>/augmented-satellite/<class_id>/*.jpg
```

Class folders are expected to be numeric (for example: `0`, `1`, `2`, `3`, `4`).

## Quick start

1) Install dependencies:

```bash
pip install -r requirements.txt
```

2) Configure experiment grid in `main.py` (at minimum check):

- `RUN_MODE`
- `VIEWS`, `MODELS`
- `LOSSES`, `AUGS_FOR_LOSS`
- dataset paths / `WALKCNN_BASE_DIR`

3) Run:

```bash
python main.py
```

Optional supervised restart mode:

```bash
python main.py --supervise
```

Reset progress markers/registry:

```bash
python main.py --reset-progress
```

## One-command reproducibility

To reproduce the full pipeline (training/eval + Table 1 + Table 2 + Figure 8) in one go:

```bash
python reproduce.py
```

Optional flags:

- `--skip-train` if `fold_metrics_full.csv` already exists
- `--skip-glmm` to skip Table 2/Figure 8 outputs
- `--reset-progress` to reset the training registry before running

## Appendix tables A.1–A.4 (Mode-specific CV aggregates)

The appendix tables are derived by aggregating cross-validation metrics per mode:

- Mode = `street` -> Appendix Table A.1
- Mode = `satellite` -> Appendix Table A.2
- Mode = `combined` -> Appendix Table A.3
- Mode = `dual` -> Appendix Table A.4

## Outputs

Main artifacts produced in the repository root:

- `fold_metrics_<mode>.csv` — metrics per fold and run
- `cv_summary_<mode>.csv` — averaged metrics per run
- `results_meta/*.json` — per-run metadata
- `checkpoints/` — `*_best.pt`, `*_state.pt`, and done markers
- `run_registry_<mode>.json` — status tracking for resume
- `heartbeat.txt` / logs — liveness and run logging

## Repository layout (key files)

- `main.py` — end-to-end training/evaluation grid runner
- `data_utils.py` — paired dataset and dataloaders, pre-aug handling
- `model_utils.py` — model loading and classifier-head replacement
- `dual_encoder.py` — dual/combined fusion modules
- `losses.py` — CE/WCE/SCE loss factory
- `metrics_utils.py` — sklearn-based metrics
- `tools/*.py` — post-processing/statistics/conversion utilities
- `glmm_walk.R` — GLMM-based statistical analysis of fold metrics

## Utility scripts

- `tools/compute_cv_stats.py`  
  Builds mean/std summary from `fold_metrics_*.csv`.
- `tools/make_mode_loss_fold_metrics.py`  
  Reshapes fold metrics into mode/model/loss format for stats.
- `balance_dataset_single_op.py`  
  Balances class counts with one-op image augmentation.
- `viz_augmentation_grid.py`  
  Visualizes augmentation operations on matched street/satellite examples.
