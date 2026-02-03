# 1) Pre - processing
# leggi dati
library(tidyverse)
library(glmmTMB)
library(performance)
library(car)
library(DHARMa)
library(rstatix)   # for identify_outliers()

input_csv <- Sys.getenv("GLMM_INPUT_CSV", unset = "output/metrics_by_mode_loss_fold.csv")
df <- read_csv(input_csv)
assumption_summary <- list()

# crea run id
df$run <- interaction(df$model, df$loss, df$fold, sep=":")

# --- Outlier detection (by condition) ---------------------------------------
# Detect outliers within each mode x model x loss group
out_tbl <- df %>%
  group_by(mode, model, loss) %>%
  identify_outliers(accuracy) %>%
  ungroup()

# Keep only the extreme ones (as you mentioned)
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

# 2) modello beta con random intercept su run — original fit on full data
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

# Same model after removing extreme outliers
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

# 3) Controllo convergenza
assumption_summary$convergence <- list(
  full   = fit_tmb$fit$convergence,
  noext  = fit_tmb_noext$fit$convergence
)

# 4) Controllo singolarità
assumption_summary$singularity <- list(
  full  = check_singularity(fit_tmb,       tolerance = 1e-3),
  noext = check_singularity(fit_tmb_noext, tolerance = 1e-3)
)

# 5) Normality (note: not required for beta GLMM; kept here for continuity)
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

# 6) Omogeneità Varianze (idem, not required for beta GLMM; kept for continuity)
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

# 7) Collinearità
assumption_summary$collinearity <- list(
  full  = check_collinearity(fit_tmb),
  noext = check_collinearity(fit_tmb_noext)
)

# 8) DHARMa (this is the most relevant for GLMMs)
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



assumption_summary
