
# WalkCNN grid workdir

Edit and run `main.py` directly in VS Code. It runs your full K-fold grid:
- Views: street, satellite, dual, combined
- Models: alexnet, vgg16, googlenet, resnet50, vit_base, deit_base, swin_base, beit_base, cvt_base, rope_vit_places365
- Pretraining per model (imagenet1k or places365)
- Losses: CE, WCE, SCE
- Aug policies with caching: none, auto, rand (WCE uses none)

You can resume with `SKIP_TO_RUN` inside `main.py`.
Augmentation test helper was removed on purpose.

Outputs:
- `fold_metrics.csv` per fold and run
- `cv_summary.csv` average across folds
- `walkcnn.log` logs
- `aug_cache/` with cached augmented tensors

Pro tip: change `street_root` and `sat_root` in `main.py` to your local paths before running.
