#!/usr/bin/env bash
# Backbone x view grid, ordinal head, one entry point (nothing to launch in between):
#   backbones: swin_base vit_base deit_base vgg16 googlenet alexnet
#   views:     street  (4 Street View tiles)
#              aerial  (aerial patch)
#              late    (fixed encoders of the trained street-only and aerial-only models + fusion head)
#              dual    (street and aerial encoders trained jointly, embeddings concatenated)
#              avg     (no training: average of street-only and aerial-only probabilities)
# Recipe: full fine-tuning (transformers AdamW 5e-5, CNNs SGD 1e-3), selection on validation loss,
# both decoders. Views run in order across all backbones, so complete comparisons appear early.
#
# Outputs in results/grid/: one folder per run, grid_<time>.log, status.txt (completed folds per run,
#   refreshed after every run; also: python run_status.py), overview.txt (backbone x view matrices),
#   by_view/<view>/comparison.md, by_backbone/<backbone>/comparison.md
# Re-running resumes: finished folds are skipped. Extra arguments go to run_ordinal.py.
#   ./run_backbone_grid.sh
#   BACKBONES="swin_base vgg16" ./run_backbone_grid.sh

set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
OUT="${GRID_OUT:-results/grid}"  # GRID_OUT overrides (tests)
read -r -a BB <<< "${BACKBONES:-swin_base vit_base deit_base vgg16 googlenet alexnet}"

mkdir -p "$OUT"
LOG="$OUT/grid_$(date +%Y%m%d_%H%M%S).log"
log() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$LOG"; }

run() {  # run <label> <run_ordinal args...>
    local label=$1
    shift
    log "[$label] start"
    local t0=$SECONDS
    if "$PY" run_ordinal.py --head ordinal --out-dir "$OUT" "$@"; then
        log "[$label] done in $(( (SECONDS - t0) / 60 )) min"
        "$PY" run_status.py "$OUT" > /dev/null
    else
        log "[$label] FAILED after $(( (SECONDS - t0) / 60 )) min, see the run's train.log"
        exit 1
    fi
}

log "backbone grid start: backbones=${BB[*]} extra args: $*"
"$PY" run_status.py "$OUT" | tee -a "$LOG"

for view in street aerial; do
    for b in "${BB[@]}"; do
        run "$view $b" --view "$view" --model "$b" --save-weights "$@"
    done
done
for b in "${BB[@]}"; do
    "$PY" fuse_predictions.py --street "$OUT/street_${b}_ordinal" --aerial "$OUT/aerial_${b}_ordinal" \
        --out "$OUT/avg_${b}_ordinal" | tee -a "$LOG"
done
"$PY" run_status.py "$OUT" > /dev/null
for view in late dual; do
    for b in "${BB[@]}"; do
        run "$view $b" --view "$view" --model "$b" "$@"
    done
done

log "writing comparisons"
for view in street aerial avg late dual; do
    "$PY" evaluate_runs.py $(for b in "${BB[@]}"; do echo "$OUT/${view}_${b}_ordinal"; done) \
        --out-dir "$OUT/by_view/$view" > /dev/null
done
for b in "${BB[@]}"; do
    "$PY" evaluate_runs.py $(for v in street aerial avg late dual; do echo "$OUT/${v}_${b}_ordinal"; done) \
        --out-dir "$OUT/by_backbone/$b" > /dev/null
done
"$PY" grid_overview.py "$OUT" | tee -a "$LOG"
log "backbone grid finished: $OUT/overview.txt, $OUT/by_view/*/comparison.md, $OUT/by_backbone/*/comparison.md"
