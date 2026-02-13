# Zero-Shot Pipeline (No Training)

This guide runs evaluation without any training updates.
Use this to quickly validate the pipeline and outputs.

## 1) What You Need

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Install R packages if you also want GLMM/Table 2/Figure 8:

```r
install.packages(c("tidyverse", "glmmTMB", "performance", "car", "DHARMa", "rstatix", "emmeans"))
```

3. Download and place datasets in:

```text
<BASE>/
  streetview/0..4/*.jpg
  satellite/0..4/*.jpg
  augmented-streetview/0..4/*.jpg
  augmented-satellite/0..4/*.jpg
```

By default, `<BASE>` is the current working directory.
Set a custom base if needed:

```powershell
$env:WALKCNN_BASE_DIR = "..\\data"
```

## 2) How to Start Zero-Shot

Run:

```bash
python reproduce.py --zero-shot
```

This calls `main.py --zero-shot` internally.

## 3) What the Zero-Shot Pipeline Does

1. `main.py --zero-shot`
- Builds folds, models, and loaders as usual.
- Skips all training loops.
- Evaluates each fold directly with current model weights.
- Writes fold/CV metrics files.

2. `tools/compute_cv_stats.py`
- Computes mean and std from fold metrics.

3. `tools/make_mode_loss_fold_metrics.py`
- Reshapes fold metrics for GLMM input.

4. `glmm_walk.R` (unless `--skip-glmm` is used)
- Produces Table 2 contrasts and Figure 8 outputs.

## 4) Zero-Shot Results

You get the same output file types as full reproducibility, but metrics reflect no-training evaluation:

- `fold_metrics_full.csv`
- `cv_summary_full.csv`
- `cv_summary_full_with_std.csv`
- `output/metrics_by_mode_loss_fold.csv`
- `output/table2_glmm_contrasts.csv`
- `output/figure8_forest.png`
- `output/figure8_forest.pdf`

Useful options:
- `python reproduce.py --zero-shot --skip-glmm`
- `python reproduce.py --zero-shot --reset-progress`
