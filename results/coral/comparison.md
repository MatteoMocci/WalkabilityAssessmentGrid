**Discrete class vs raters**

| Metric | street_swin_base_ordinal_lr5e-5 | street_swin_base_ordinal_coral_prior | street_swin_base_ordinal_coral | Human (rater vs rater) | Δ street_swin_base_ordinal_coral_prior − street_swin_base_ordinal_lr5e-5 (p) | Δ street_swin_base_ordinal_coral − street_swin_base_ordinal_lr5e-5 (p) |
|---|---|---|---|---|---|---|
| Accuracy (exact) | 0.369 ± 0.024 | 0.360 ± 0.030 | 0.323 ± 0.030 | 0.306 ± 0.017 | -0.009 (0.105) | -0.047 (0.004) |
| One-off accuracy | 0.848 ± 0.019 | 0.833 ± 0.024 | 0.754 ± 0.039 | 0.742 ± 0.020 | -0.015 (0.008) | -0.094 (0.002) |
| Macro-F1 | 0.300 ± 0.026 | 0.309 ± 0.034 | 0.306 ± 0.024 | 0.293 ± 0.017 | +0.009 (0.625) | +0.006 (0.557) |
| MAE | 0.805 ± 0.042 | 0.835 ± 0.051 | 0.995 ± 0.088 | 1.030 ± 0.035 | +0.030 (0.002) | +0.190 (0.002) |
| AMAE (macro MAE) | 0.924 ± 0.042 | 0.940 ± 0.054 | 1.009 ± 0.053 | 1.108 ± 0.038 | +0.016 (0.131) | +0.085 (0.002) |
| MMAE (worst-class MAE) | 1.452 ± 0.161 | 1.416 ± 0.184 | 1.377 ± 0.193 | 1.514 ± 0.164 | -0.036 (0.496) | -0.075 (0.275) |
| Quadratic weighted kappa | 0.455 ± 0.041 | 0.453 ± 0.052 | 0.457 ± 0.042 | 0.342 ± 0.029 | -0.002 (0.695) | +0.003 (0.695) |
| Krippendorff alpha (ordinal) | 0.448 ± 0.044 | 0.437 ± 0.054 | 0.428 ± 0.047 | 0.336 ± 0.031 | -0.011 (0.131) | -0.020 (0.375) |
| Krippendorff alpha (interval) | 0.451 ± 0.043 | 0.448 ± 0.052 | 0.452 ± 0.042 | 0.343 ± 0.029 | -0.003 (0.492) | +0.001 (0.625) |
| Recall class 1 | 0.214 ± 0.093 | 0.269 ± 0.098 | 0.586 ± 0.094 | 0.357 ± 0.020 | +0.055 (0.160) | +0.373 (0.002) |
| Recall class 5 | 0.028 ± 0.035 | 0.083 ± 0.091 | 0.455 ± 0.201 | 0.182 ± 0.070 | +0.054 (0.078) | +0.427 (0.002) |

**Discrete class, QWK-tuned thresholds**

| Metric | street_swin_base_ordinal_lr5e-5 | street_swin_base_ordinal_coral_prior | street_swin_base_ordinal_coral | Human (rater vs rater) | Δ street_swin_base_ordinal_coral_prior − street_swin_base_ordinal_lr5e-5 (p) | Δ street_swin_base_ordinal_coral − street_swin_base_ordinal_lr5e-5 (p) |
|---|---|---|---|---|---|---|
| Quadratic weighted kappa | 0.485 ± 0.036 | 0.486 ± 0.048 | 0.466 ± 0.060 |  | +0.001 (0.922) | -0.019 (0.160) |
| MAE | 0.888 ± 0.061 | 0.889 ± 0.042 | 0.899 ± 0.050 |  | +0.002 (0.375) | +0.012 (0.432) |
| Accuracy (exact) | 0.361 ± 0.023 | 0.355 ± 0.022 | 0.351 ± 0.020 |  | -0.005 (0.275) | -0.009 (0.375) |
| One-off accuracy | 0.802 ± 0.030 | 0.804 ± 0.017 | 0.801 ± 0.024 |  | +0.002 (1.000) | -0.001 (0.770) |
| Macro-F1 | 0.334 ± 0.025 | 0.327 ± 0.025 | 0.322 ± 0.036 |  | -0.007 (0.232) | -0.012 (0.275) |
| AMAE (macro MAE) | 0.935 ± 0.042 | 0.939 ± 0.048 | 0.955 ± 0.063 |  | +0.004 (0.557) | +0.020 (0.322) |
| Recall class 1 | 0.549 ± 0.087 | 0.542 ± 0.117 | 0.521 ± 0.150 |  | -0.007 (0.846) | -0.027 (0.496) |
| Recall class 5 | 0.231 ± 0.188 | 0.200 ± 0.156 | 0.223 ± 0.196 |  | -0.031 (1.000) | -0.009 (0.547) |

**Continuous score vs mean rating**

| Metric | street_swin_base_ordinal_lr5e-5 | street_swin_base_ordinal_coral_prior | street_swin_base_ordinal_coral | Human (rater vs rater) | Δ street_swin_base_ordinal_coral_prior − street_swin_base_ordinal_lr5e-5 (p) | Δ street_swin_base_ordinal_coral − street_swin_base_ordinal_lr5e-5 (p) |
|---|---|---|---|---|---|---|
| Spearman rho | 0.601 ± 0.053 | 0.591 ± 0.066 | 0.581 ± 0.064 |  | -0.010 (0.105) | -0.021 (0.037) |
| Kendall tau-b | 0.457 ± 0.046 | 0.446 ± 0.056 | 0.438 ± 0.055 |  | -0.010 (0.084) | -0.019 (0.020) |
| Pearson r | 0.629 ± 0.046 | 0.617 ± 0.055 | 0.609 ± 0.059 |  | -0.012 (0.064) | -0.020 (0.037) |
| R2 | 0.370 ± 0.064 | 0.340 ± 0.076 | 0.348 ± 0.080 |  | -0.031 (0.014) | -0.022 (0.193) |
| MAE | 0.617 ± 0.030 | 0.631 ± 0.033 | 0.628 ± 0.031 |  | +0.014 (0.064) | +0.011 (0.105) |
| RMSE | 0.775 ± 0.036 | 0.793 ± 0.040 | 0.788 ± 0.038 |  | +0.018 (0.014) | +0.013 (0.193) |

**Probabilities vs soft targets**

| Metric | street_swin_base_ordinal_lr5e-5 | street_swin_base_ordinal_coral_prior | street_swin_base_ordinal_coral | Human (rater vs rater) | Δ street_swin_base_ordinal_coral_prior − street_swin_base_ordinal_lr5e-5 (p) | Δ street_swin_base_ordinal_coral − street_swin_base_ordinal_lr5e-5 (p) |
|---|---|---|---|---|---|---|
| Cumulative BCE | 0.439 ± 0.014 | 0.446 ± 0.014 | 0.527 ± 0.019 |  | +0.007 (0.006) | +0.087 (0.002) |
| Brier P(Y>1) | 0.048 ± 0.007 | 0.051 ± 0.007 | 0.099 ± 0.012 |  | +0.003 (0.014) | +0.051 (0.002) |
| Brier P(Y>2) | 0.099 ± 0.012 | 0.098 ± 0.012 | 0.103 ± 0.013 |  | -0.000 (1.000) | +0.004 (0.064) |
| Brier P(Y>3) | 0.111 ± 0.011 | 0.114 ± 0.011 | 0.112 ± 0.008 |  | +0.003 (0.375) | +0.001 (0.770) |
| Brier P(Y>4) | 0.049 ± 0.009 | 0.054 ± 0.008 | 0.125 ± 0.027 |  | +0.005 (0.064) | +0.076 (0.002) |
| AUC P(Y>1), unanimous points | 0.947 ± 0.022 | 0.944 ± 0.018 | 0.940 ± 0.022 |  | -0.003 (0.557) | -0.007 (0.432) |
| AUC P(Y>2), unanimous points | 0.904 ± 0.030 | 0.905 ± 0.029 | 0.897 ± 0.034 |  | +0.001 (0.770) | -0.008 (0.232) |
| AUC P(Y>3), unanimous points | 0.845 ± 0.051 | 0.841 ± 0.056 | 0.835 ± 0.053 |  | -0.004 (0.322) | -0.010 (0.105) |
| AUC P(Y>4), unanimous points | 0.819 ± 0.065 | 0.824 ± 0.075 | 0.847 ± 0.057 |  | +0.005 (1.000) | +0.028 (0.084) |
| Predicted spread vs rater disagreement (rho) | 0.057 ± 0.051 | 0.073 ± 0.041 | 0.081 ± 0.043 |  | +0.016 (0.232) | +0.024 (0.492) |
| Monotonicity violations (%) | 0.165 ± 0.319 | 0.000 ± 0.000 | 0.000 ± 0.000 |  | -0.165 (0.125) | -0.165 (0.125) |

Folds: 10 paired. Human column = one rater scored against the other on the same test points. Correlation ceiling for the continuous score, sqrt(ICC(1,2)): 0.714 ± 0.023. p = paired Wilcoxon signed-rank over folds.
QWK-tuned thresholds on P(Y>k) were tuned on: street_swin_base_ordinal_lr5e-5: validation fold; street_swin_base_ordinal_coral_prior: validation fold; street_swin_base_ordinal_coral: validation fold
