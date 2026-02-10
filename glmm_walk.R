# ============================================================
# GLMM analysis for Section 4 (Tables 2 + Figure 8)
# Single-file analysis script with commented steps.
#
# Inputs:
#   - output/metrics_by_mode_loss_fold.csv
# Outputs:
#   - output/table2_glmm_contrasts.csv
#   - output/figure8_forest.png
#   - output/figure8_forest.pdf
# ============================================================

# 1) Pre-processing / package load
library(tidyverse)
library(glmmTMB)
library(performance)
library(car)
library(DHARMa)
library(rstatix)   # identify_outliers()
library(emmeans)   # marginal means + contrasts

input_csv <- Sys.getenv("GLMM_INPUT_CSV", unset = "output/metrics_by_mode_loss_fold.csv")
out_table2_csv <- Sys.getenv("TABLE2_OUT_CSV", unset = "output/table2_glmm_contrasts.csv")
out_fig8_png <- Sys.getenv("FIG8_OUT_PNG", unset = "output/figure8_forest.png")
out_fig8_pdf <- Sys.getenv("FIG8_OUT_PDF", unset = "output/figure8_forest.pdf")

if (!file.exists(input_csv)) {
  stop(paste("Missing input file:", input_csv))
}

df <- read_csv(input_csv, show_col_types = FALSE)
assumption_summary <- list()

# create run id for random intercept
df$run <- interaction(df$model, df$loss, df$fold, sep=":")

# --- Outlier detection (by condition) ---------------------------------------
# Detect outliers within each mode x model x loss group
out_tbl <- df %>%
  group_by(mode, model, loss) %>%
  identify_outliers(accuracy) %>%
  ungroup()

# Keep only the extreme ones
extreme_keys <- out_tbl %>%
  filter(is.extreme) %>%
  select(mode, model, loss, fold)

assumption_summary$outliers <- list(
  detected = nrow(out_tbl),
  extreme  = nrow(extreme_keys),
  table    = out_tbl
)

# Remove extreme outliers from df (keeps everything else)
df_noext <- df %>%
  anti_join(extreme_keys, by = c("mode","model","loss","fold"))

# 2) Beta GLMM with random intercept on run (full data)
fit_tmb <- glmmTMB(
  accuracy ~ mode * model * loss + (1 | run),
  family = beta_family(link = "logit"),
  data = df,
  control = glmmTMBControl(
    optimizer = optim,
    optArgs = list(method = "BFGS"),
    optCtrl = list(maxit = 1e5)
  )
)
summary(fit_tmb)

# Same model after removing extreme outliers (used for Table 2 + Figure 8)
fit_tmb_noext <- glmmTMB(
  accuracy ~ mode * model * loss + (1 | run),
  family = beta_family(link = "logit"),
  data = df_noext,
  control = glmmTMBControl(
    optimizer = optim,
    optArgs = list(method = "BFGS"),
    optCtrl = list(maxit = 1e5)
  )
)
summary(fit_tmb_noext)

# 3) Convergence check
assumption_summary$convergence <- list(
  full   = fit_tmb$fit$convergence,
  noext  = fit_tmb_noext$fit$convergence
)

# 4) Singularity check
assumption_summary$singularity <- list(
  full  = check_singularity(fit_tmb,       tolerance = 1e-3),
  noext = check_singularity(fit_tmb_noext, tolerance = 1e-3)
)

# 5) Normality (not required for beta GLMM; kept for continuity)
par(mfrow = c(1,2))
qqnorm(resid(fit_tmb));       qqline(resid(fit_tmb))
plot(fitted(fit_tmb), resid(fit_tmb), xlab = "Fitted", ylab = "Residuals"); abline(h = 0, lty = 2)
par(mfrow = c(1,1))

sw_full <- shapiro.test(sample(resid(fit_tmb), size = min(5000, length(resid(fit_tmb)))))
assumption_summary$normality_full <- list(
  shapiro_p = sw_full$p.value,
  pass = sw_full$p.value > 0.05
)

par(mfrow = c(1,2))
qqnorm(resid(fit_tmb_noext)); qqline(resid(fit_tmb_noext))
plot(fitted(fit_tmb_noext), resid(fit_tmb_noext), xlab = "Fitted", ylab = "Residuals"); abline(h = 0, lty = 2)
par(mfrow = c(1,1))

sw_noext <- shapiro.test(sample(resid(fit_tmb_noext), size = min(5000, length(resid(fit_tmb_noext)))))
assumption_summary$normality_noext <- list(
  shapiro_p = sw_noext$p.value,
  pass = sw_noext$p.value > 0.05
)

# 6) Homogeneity of variance (not required for beta GLMM; kept for continuity)
df$grp <- interaction(df$mode, df$model, df$loss, drop = TRUE)
lv_full <- leveneTest(resid(fit_tmb) ~ grp, data = df)
lev_p_full <- lv_full[["Pr(>F)"]][1]
assumption_summary$homogeneity_full <- list(
  levene_p = lev_p_full,
  pass = lev_p_full > 0.05
)

df_noext$grp <- interaction(df_noext$mode, df_noext$model, df_noext$loss, drop = TRUE)
lv_noext <- leveneTest(resid(fit_tmb_noext) ~ grp, data = df_noext)
lev_p_noext <- lv_noext[["Pr(>F)"]][1]
assumption_summary$homogeneity_noext <- list(
  levene_p = lev_p_noext,
  pass = lev_p_noext > 0.05
)

# 7) Collinearity
assumption_summary$collinearity <- list(
  full  = check_collinearity(fit_tmb),
  noext = check_collinearity(fit_tmb_noext)
)

# 8) DHARMa (most relevant for GLMMs)
set.seed(1)
sim_full  <- simulateResiduals(fittedModel = fit_tmb,        n = 1000)
sim_noext <- simulateResiduals(fittedModel = fit_tmb_noext,  n = 1000)

par(mfrow = c(2,2))
plot(sim_full,  main = "DHARMa full")
plot(sim_noext, main = "DHARMa no-ext")
par(mfrow = c(1,1))

assumption_summary$DHARMa <- list(
  full = list(
    uniformity_p = testUniformity(sim_full)$p.value,
    dispersion_p = testDispersion(sim_full)$p.value,
    outliers_p   = testOutliers(sim_full)$p.value
  ),
  noext = list(
    uniformity_p = testUniformity(sim_noext)$p.value,
    dispersion_p = testDispersion(sim_noext)$p.value,
    outliers_p   = testOutliers(sim_noext)$p.value
  )
)

# Optional: residuals vs predictors with factors to silence warnings
df$mode_f      <- factor(df$mode); df$model_f <- factor(df$model); df$loss_f <- factor(df$loss)
df_noext$mode_f <- factor(df_noext$mode); df_noext$model_f <- factor(df_noext$model); df_noext$loss_f <- factor(df_noext$loss)

par(mfrow = c(2,3))
plotResiduals(sim_full,  df$mode_f,      main = "Full: resids vs mode")
plotResiduals(sim_full,  df$model_f,     main = "Full: resids vs model")
plotResiduals(sim_full,  df$loss_f,      main = "Full: resids vs loss")
plotResiduals(sim_noext, df_noext$mode_f,  main = "No-ext: resids vs mode")
plotResiduals(sim_noext, df_noext$model_f, main = "No-ext: resids vs model")
plotResiduals(sim_noext, df_noext$loss_f,  main = "No-ext: resids vs loss")
par(mfrow = c(1,1))

# ============================================================
# 9) Table 2: Holm-adjusted pairwise contrasts on accuracy
# ============================================================

# Map mode names to manuscript terminology for reporting
df_noext_tbl2 <- df_noext %>%
  mutate(
    mode = recode(
      mode,
      satellite = "aerial",
      combined = "late fusion",
      dual = "dual encoder",
      .default = mode
    )
  )

emm_tbl2 <- emmeans(fit_tmb_noext, ~ mode, type = "response")
contr_tbl2 <- pairs(emm_tbl2, adjust = "holm")
sum_tbl2 <- summary(contr_tbl2, infer = c(TRUE, TRUE), type = "response")

table2_out <- as.data.frame(sum_tbl2) %>%
  transmute(
    contrast = contrast,
    estimate = estimate,
    SE = SE,
    df = df,
    z = z.ratio,
    p_adj = p.value
  )

write_csv(table2_out, out_table2_csv)
message("Wrote Table 2 contrasts to: ", out_table2_csv)

# ============================================================
# 10) Figure 8: Forest plots of odds ratios (fusion vs street)
# ============================================================

# EMMs on logit scale for odds ratios
emm_fig8 <- emmeans(fit_tmb_noext, ~ mode | model * loss, type = "link")

# Contrasts: late fusion vs street, dual encoder vs street
contr_fig8 <- contrast(
  emm_fig8,
  method = list(
    "late_vs_street" = c(-1, 0, 1, 0),
    "dual_vs_street" = c(-1, 0, 0, 1)
  ),
  by = c("model", "loss"),
  adjust = "none"
)

sum_fig8 <- as.data.frame(summary(contr_fig8, infer = c(TRUE, TRUE)))

# Holm adjustment within each model x loss cell
sum_fig8 <- sum_fig8 %>%
  group_by(model, loss) %>%
  mutate(p_adj = p.adjust(p.value, method = "holm")) %>%
  ungroup()

plot_df <- sum_fig8 %>%
  mutate(
    odds_ratio = exp(estimate),
    or_low = exp(lower.CL),
    or_high = exp(upper.CL),
    significant = p_adj < 0.05,
    contrast_label = recode(
      contrast,
      "late_vs_street" = "late fusion vs street",
      "dual_vs_street" = "dual encoder vs street"
    ),
    cell = paste(model, loss, sep = " / ")
  )

g <- ggplot(plot_df, aes(x = odds_ratio, y = cell)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "gray50") +
  geom_errorbarh(aes(xmin = or_low, xmax = or_high), height = 0.2, color = "gray40") +
  geom_point(aes(shape = significant), size = 2.5, fill = "black") +
  facet_wrap(~ contrast_label, ncol = 1, scales = "free_y") +
  scale_x_log10() +
  scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 1)) +
  labs(
    x = "Odds ratio (log scale)",
    y = "Model / Loss",
    title = "Fusion vs street (odds ratios by model and loss)"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    axis.text.y = element_text(size = 9),
    legend.position = "none"
  )

ggsave(out_fig8_png, g, width = 7.5, height = 9, dpi = 300)
ggsave(out_fig8_pdf, g, width = 7.5, height = 9)
message("Wrote Figure 8 plots to: ", out_fig8_png, " and ", out_fig8_pdf)

assumption_summary
