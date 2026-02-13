# WalkCNN Reproducibility Pipeline

This README is the main guide to reproduce the study outputs from this repository.

## 1) What You Need Before Running

1. Clone this repository.
2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Install R and required packages (needed for Table 2 and Figure 8):

```r
install.packages(c("tidyverse", "glmmTMB", "performance", "car", "DHARMa", "rstatix", "emmeans"))
```

4. Download datasets:
- Streetview images: `https://github.com/gatrunfio/AAPW`
- Satellite and augmented datasets: `https://figshare.com/s/2753c0fc785521d63636?file=61611229`

5. Arrange folders as:

```text
<BASE>/
  streetview/0..4/*.jpg
  satellite/0..4/*.jpg
  augmented-streetview/0..4/*.jpg
  augmented-satellite/0..4/*.jpg
```

By default, `<BASE>` is the current working directory. To use another location:

```powershell
$env:WALKCNN_BASE_DIR = "..\\data"
```

## 2) How to Start the Reproducibility Pipeline

Run:

```bash
python reproduce.py
```

This starts the full end-to-end pipeline (training/evaluation + post-processing + GLMM outputs).

Important runtime note: full training can take several days depending on hardware.
If you want a quick test without training the models, run the zero-shot workflow in `zero-shot.md`.

Useful flags:
- `--reset-progress`: reset registry/checkpoint progress before running
- `--skip-train`: skip training/evaluation only if `fold_metrics_full.csv` already exists
- `--skip-glmm`: skip GLMM/Table 2/Figure 8 outputs

## 3) What Each Pipeline Step Does

When you run `python reproduce.py`, the workflow is:

1. Training and evaluation (`main.py`)
- Runs the model grid over modes/views and folds.
- Writes fold metrics and CV summary metrics.

2. CV aggregation (`tools/compute_cv_stats.py`)
- Computes mean and standard deviation per run from fold metrics.
- Produces Table 1 input data.

3. GLMM input reshaping (`tools/make_mode_loss_fold_metrics.py`)
- Reshapes fold outputs into mode/model/loss format for statistical modeling.

4. GLMM analysis and figure generation (`glmm_walk.R`)
- Fits the GLMM and writes Table 2 contrasts.
- Exports Figure 8 forest plots.

## 4) Results Produced

Main outputs are:

- `fold_metrics_full.csv`: per-fold metrics for each run
- `cv_summary_full.csv`: average metrics per run
- `cv_summary_full_with_std.csv`: mean and std per run (Table 1 input)
- `output/metrics_by_mode_loss_fold.csv`: reshaped fold metrics for GLMM
- `output/table2_glmm_contrasts.csv`: Table 2 contrasts
- `output/figure8_forest.png`
- `output/figure8_forest.pdf`
- `results_meta/*.json`: run metadata
- `checkpoints/*.pt` and done markers: resume/training state

## Related Docs

- `zero-shot.md`: no-training evaluation workflow
- `overview.md`: concise description of each repo file and what is required for computation
