# Perceived walkability from street-level and aerial imagery

This repository contains two studies:

- **Ordinal walkability study** (this branch, sections 1–5): predicts perceived walkability on a 1–5 scale from
  Street View and aerial images of 4,410 points in Cagliari, Sassari and Alghero, each rated by two people.
  The model learns the four cumulative probabilities P(Y > k) from soft targets built from both ratings.
- **Previous study (WalkCNN)**: the earlier classification pipeline, kept for reproducibility (section 6).

## 1) Installation

Works on Linux, Windows and macOS with Python 3.10 or newer. An NVIDIA GPU is strongly recommended
(the published runs used an RTX 4090, 24 GB); a CPU works for quick checks only.

```bash
git clone -b ordinal-study https://github.com/MatteoMocci/WalkabilityAssessmentGrid.git
cd WalkabilityAssessmentGrid
python study.py grid
```

That single command sets everything up on the first run and then starts the experiment (any study in
section 3 works the same way). If the environment `.venv` does not exist, `study.py` first runs
`setup_env.py`, which creates it, installs the PyTorch build that matches your NVIDIA driver (or the CPU /
Apple GPU build), installs the other packages and reports which device training will use. Afterwards
`study.py` always runs inside `.venv`, whichever `python` starts it. To only install, run `python setup_env.py`.

- On **Python 3.12** it installs the exact versions of the published results (`requirements-lock.txt`);
  on other versions it installs compatible versions (`requirements.txt`), whose results can differ slightly.
- `python setup_env.py --torch cpu` (or `cu126`, `cu128`, ...) forces a PyTorch build;
  `--compatible` / `--exact` force the version set.

On Linux/macOS, `python` may be called `python3`. To use your own environment instead of `.venv`
(e.g. conda), set `STUDY_NO_VENV=1`.

## 2) Data

In the repository:

- `point_labels.csv`: for each point, the soft cumulative targets `p_above_1..4`, the share of the two raters
  whose rating exceeds k (e.g. ratings 3 and 4 give 1, 1, 0.5, 0).
- `folds.csv`: the 10 spatial cross-validation folds (1 km blocks, stratified by city × rating), built by
  `make_folds.py`. All experiments use these folds.

Not in the repository (images), expected at:

```text
views_output/streetview/images/P000001_front.png, P000001_back.png, ...   (two 180° views per point)
views_output/wms_patches/images/P000001_lat<lat>_lon<lon>.png, ...         (aerial patch per point)
```

Pretrained backbone weights are downloaded automatically on the first run (internet needed).

## 3) Running the experiments

```bash
python study.py status              # what is done / still to do in the backbone x view grid
python study.py first-experiment    # Swin, street images: ordinal head vs soft-label CE baseline
python study.py finetune            # full / frozen / head-then-last-stage fine-tuning, then a learning-rate grid
python study.py grid                # 6 backbones x street / aerial / averaged / late fusion / dual encoder
```

- **Resuming**: run the same command again. Finished folds are skipped; a fold left incomplete by a crash or
  power cut is trained again.
- **Options**: arguments that `study.py` does not know go to `run_ordinal.py`, e.g. a quick check on any
  machine: `python study.py grid --backbones alexnet --device cpu --folds 0 --epochs 1 --max-points 40`.
  `python run_ordinal.py --help` lists all options.
- **Long runs**: on Linux/macOS run inside `tmux` so they survive a closed terminal. The `run_*.sh` scripts are
  Linux shortcuts for the same studies.

Durations on an RTX 4090: about 1 hour per backbone and view; the full grid about 12–18 hours.

## 4) Outputs

Each run writes a folder in `results/` with, per fold, the test and validation predictions
(`fold_<k>_preds.csv`, `fold_<k>_val_preds.csv`), the metrics (`fold_<k>_metrics.json`) and `train.log`.
Summaries and comparisons:

- `results/grid/overview.txt`: backbone × view tables of the main metrics
- `results/grid/by_view/<view>/comparison.md`, `results/grid/by_backbone/<backbone>/comparison.md`
- `results/finetune/comparison.md`, `results/coral/comparison.md`, `results/comparison.md`
- `python evaluate_runs.py <run folders>` recomputes all metrics from the saved predictions and writes a
  comparison table with paired tests between runs.

Metrics: accuracy, QWK, within-one accuracy and MAE of the predicted class (with the 0.5 rule and with
thresholds tuned on the validation fold), plus AMAE, Krippendorff's alpha, correlations of the continuous score
1 + Σ P(Y > k), calibration, and a human reference (one rater scored against the other).

Model weights (`*.pt`) are saved only for the street and aerial grid runs (late fusion reuses them) and are not
in the repository.

## 5) Reproducibility notes

- Folds and seeds are fixed (seed 42 + fold). Augmentation draws depend on the number of data-loading
  workers: keep the default `--num-workers 16` to reproduce the published runs.
- Exact numbers also depend on the GPU, the library versions and whether `torch.compile` is available
  (CUDA with Triton); without it the models run uncompiled, with the same architecture.
- `requirements-lock.txt` is the environment of the published results (Python 3.12.13, CUDA 13.0).

## 6) Previous study (WalkCNN pipeline)

The earlier study's pipeline (`main.py`, `reproduce.py`, `tools/`, `glmm_walk.R`) is unchanged. It needs its own
dependencies and R:

```bash
pip install -r requirements-legacy.txt
```

```r
install.packages(c("tidyverse", "glmmTMB", "performance", "car", "DHARMa", "rstatix", "emmeans"))
```

Data:
- Streetview pictures: `https://anonymous.4open.science/r/Streetview180-7357/`
- Generated aerial views and augmented dataset: `https://figshare.com/s/2753c0fc785521d63636?file=61611229`

Arranged as `<BASE>/{streetview,satellite,augmented-streetview,augmented-satellite}/0..4/*.jpg`, where `<BASE>`
is the working directory or the `WALKCNN_BASE_DIR` environment variable. Run `python reproduce.py`
(flags: `--reset-progress`, `--skip-train`, `--skip-glmm`). It runs training and evaluation (`main.py`), CV
aggregation (`tools/compute_cv_stats.py`), GLMM input reshaping (`tools/make_mode_loss_fold_metrics.py`) and the
GLMM with Table 2 / Figure 8 (`glmm_walk.R`); see `zero-shot.md` for a no-training workflow and `overview.md`
for a description of each file.
