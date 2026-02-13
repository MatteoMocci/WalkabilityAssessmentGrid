# Required Files Overview

Only files required for the reproducibility computation are listed here.

## Pipeline Entry and Core Training/Evaluation

- `reproduce.py` - Orchestrates the end-to-end reproducibility workflow.
- `main.py` - Runs model training/evaluation and zero-shot evaluation.
- `data_utils.py` - Builds datasets and dataloaders.
- `model_utils.py` - Loads backbones and classifier heads.
- `dual_encoder.py` - Implements dual/combined fusion models.
- `losses.py` - Defines CE/SCE/WCE loss functions.
- `metrics_utils.py` - Computes evaluation metrics.
- `logging_utils.py` - Provides logging setup/utilities.

## Post-Processing and Statistical Outputs

- `tools/compute_cv_stats.py` - Computes mean/std summaries from fold metrics (Table 1 input).
- `tools/make_mode_loss_fold_metrics.py` - Reshapes fold metrics for GLMM analysis.
- `glmm_walk.R` - Produces Table 2 contrasts and Figure 8 outputs.

## Dependencies and Main Documentation

- `requirements.txt` - Python dependencies.
- `README.md` - Main reproducibility instructions.
