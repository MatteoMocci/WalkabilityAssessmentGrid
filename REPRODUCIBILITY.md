# Reproducibility Guide (Tables 1-2, Figure 8, Section 4 Metrics)

This document provides step-by-step instructions to reproduce each reported finding
using only the shared data and code, with **relative paths** throughout. The workflow
starts from the original source data (street-view and aerial images), proceeds through
data preparation and model training, and finishes with the statistical analysis and
figure generation used in Section 4.

If you are running on Windows PowerShell, the commands below work as written. For
macOS/Linux, use the same commands without `$env:` syntax for environment variables.

## 1) Environment Setup

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Install R and required R packages:

```r
install.packages(c("tidyverse", "glmmTMB", "performance", "car", "DHARMa", "rstatix", "emmeans"))
```

## 2) Data Workflow (From Original Source Data)

You only need the `streetview/` and `satellite/` folders to run the pipeline.
The `augmented-streetview/` and `augmented-satellite/` folders can be downloaded
directly (or regenerated locally as described below).

Data sources:

- Streetview images: `https://github.com/gatrunfio/AAPW`
- Satellite + augmented datasets: `https://figshare.com/s/2753c0fc785521d63636?file=61611229`

The repository expects the **original street-view and aerial images** organized into
class folders. Filenames must contain latitude/longitude tokens, as used by the pairing
logic. The expected folder layout is:

```
<BASE>/
  streetview/
    0/*.jpg
    1/*.jpg
    2/*.jpg
    3/*.jpg
    4/*.jpg
  satellite/
    0/*.jpg
    1/*.jpg
    2/*.jpg
    3/*.jpg
    4/*.jpg
```

Where `<BASE>` is the dataset root. By default, `<BASE>` is the **parent of this repo**.
To set a custom location, use a **relative path**:

```powershell
$env:WALKCNN_BASE_DIR = "..\\data"
```

### 2.1 Create Augmented Datasets (for WCE/SCE/CE runs)

The full grid in the paper uses pre-augmented datasets. Generate them from the
original images using the provided script:

```powershell
python balance_dataset_single_op.py `
  --input_root ..\\data\\streetview `
  --output_root ..\\data\\augmented-streetview `
  --seed 42 `
  --even_ops

python balance_dataset_single_op.py `
  --input_root ..\\data\\satellite `
  --output_root ..\\data\\augmented-satellite `
  --seed 42 `
  --even_ops
```

This produces:

```
<BASE>/
  augmented-streetview/0..4/*.jpg
  augmented-satellite/0..4/*.jpg
```

## 3) Train and Evaluate Models (Section 4 Metrics)

This generates the cross-validated metrics used throughout Section 4, including
accuracy, precision, recall, macro F1, MAE, and one-off accuracy.

1. Ensure `RUN_MODE="full"` in `main.py`.
2. Ensure the grid matches the paper (modes, models, losses).

Note: The current `main.py` may be temporarily configured for a WCE-only rerun.
The paper results use **CE, SCE, and WCE** across all four modes. If needed, set:

```
LOSSES = ["CE", "SCE", "WCE"]
AUGS_FOR_LOSS = {"CE": ["preaug"], "SCE": ["preaug"], "WCE": ["preaug"]}
```

Then run:

```powershell
python main.py --reset-progress
python main.py
```

Outputs (relative to repo root):

```
fold_metrics_full.csv
cv_summary_full.csv
```

These files are the **base metrics** for all Section 4 results.

### One-command option

If you want to run the entire pipeline end-to-end (training/eval + Table 1 + Table 2 + Figure 8):

```bash
python reproduce.py
```

Optional flags:

- `--skip-train` if `fold_metrics_full.csv` already exists
- `--skip-glmm` to skip Table 2/Figure 8 outputs
- `--reset-progress` to reset the training registry before running

## 4) Table 1 (Best Configuration Per Mode)

Table 1 reports the **best run per mode** with mean ± std across folds.

1. Compute mean/std per run:

```powershell
python tools\\compute_cv_stats.py `
  --fold-metrics fold_metrics_full.csv `
  --out cv_summary_full_with_std.csv
```

2. For each mode (`street`, `satellite`, `combined`, `dual`), select the run with
the highest **accuracy_mean**. This selection produces the "best per mode" rows
reported in Table 1.

Input: `cv_summary_full_with_std.csv`  
Output: Table 1 values.

## 5) Table 2 (GLMM Pairwise Contrasts, Holm Adjusted)

Table 2 is based on a GLMM fitted to accuracy with extreme outliers removed and
pairwise contrasts between modes, Holm-adjusted.

1. Create per-fold mode/model/loss metrics:

```powershell
python tools\\make_mode_loss_fold_metrics.py `
  --input fold_metrics_full.csv `
  --out output\\metrics_by_mode_loss_fold.csv
```

2. Fit the GLMM and compute Table 2 contrasts:

```powershell
Rscript glmm_walk.R
```

Outputs:

```
output\\table2_glmm_contrasts.csv
```

This CSV contains the estimates and Holm-adjusted p-values used in Table 2.

## 6) Figure 8 (Forest Plots of Odds Ratios)

Figure 8 compares late fusion and dual encoder fusion against street, within each
model-by-loss cell, using Holm-adjusted contrasts.

1. Ensure `output\\metrics_by_mode_loss_fold.csv` exists (from step 5.1).
2. Run the figure script (same GLMM script):

```powershell
Rscript glmm_walk.R
```

Outputs:

```
output\\figure8_forest.png
output\\figure8_forest.pdf
```

## 7) Notes on Modes and Labels

The mapping between code and manuscript terminology is:

- `street`   -> Street view
- `satellite` -> Aerial view
- `combined` -> Late fusion
- `dual` -> Dual encoder fusion

These mappings are applied in the R scripts for Table 2 and Figure 8.

## 8) Off-the-Shelf Software

All steps above are fully scripted (Python/R). No GUI-based software is required,
so no screenshots are needed.
