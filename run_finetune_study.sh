#!/usr/bin/env bash
# Fine-tuning study, ordinal head, one entry point (nothing to launch in between):
#   1) full / frozen / lpft, each at its default learning rate
#      full: everything trains, lr 5e-5 | frozen: head only, lr 1e-3 |
#      lpft: head only (1e-3), then also the last backbone stage (5e-5)
#   2) best mode = lowest mean validation loss over the folds (test folds never used)
#   3) learning-rate grid on that mode
#   4) per fold, the learning rate with the lowest validation loss is selected (nested),
#      then comparison tables are written
#
# Outputs in results/finetune/: one folder per run, study_<time>.log (stage-by-stage log),
#   comparison.md (3 modes + selected learning rate), grid/comparison.md (all learning rates)
# Re-running resumes: finished folds are skipped. Extra arguments go to run_ordinal.py.
#   ./run_finetune_study.sh
#   ./run_finetune_study.sh --model vit_base

set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
OUT="${FT_OUT:-results/finetune}"  # FT_OUT overrides (tests)
MODEL=swin_base
prev=""
for a in "$@"; do
    [[ "$prev" == "--model" ]] && MODEL="$a"
    prev="$a"
done
D="$OUT/street_${MODEL}_ordinal"

mkdir -p "$OUT"
LOG="$OUT/study_$(date +%Y%m%d_%H%M%S).log"
log() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$LOG"; }

run() {  # run <label> <run_ordinal args...>
    local label=$1
    shift
    log "[$label] start"
    local t0=$SECONDS
    if "$PY" run_ordinal.py --head ordinal --out-dir "$OUT" "$@"; then
        log "[$label] done in $(( (SECONDS - t0) / 60 )) min"
    else
        log "[$label] FAILED after $(( (SECONDS - t0) / 60 )) min, see the run's train.log"
        exit 1
    fi
}

log "fine-tuning study start: model=$MODEL extra args: $*"

# 1) three modes at their default learning rates
run "full lr=5e-5"            --finetune full   --lr 5e-5                 --tag lr5e-5 "$@"
run "frozen head_lr=1e-3"     --finetune frozen --head-lr 1e-3            --tag lr1e-3 "$@"
run "lpft head_lr=1e-3 lr=5e-5" --finetune lpft --head-lr 1e-3 --lr 5e-5  --tag lr5e-5 "$@"

# 2) best mode on validation loss
log "choosing the best mode on validation loss"
BEST=$("$PY" select_finetune.py choose full="${D}_lr5e-5" frozen="${D}_frozen_lr1e-3" \
       lpft="${D}_lpft_lr5e-5" | tee -a "$LOG" | tail -1)
log "best mode: $BEST"

# 3) learning-rate grid on the best mode (the default learning rate is already done)
case "$BEST" in
    full)   PREFIX="${D}";        FLAG=(--finetune full);                   LR_FLAG=--lr;      DEFAULT=5e-5; GRID=(1e-5 2e-5 1e-4) ;;
    frozen) PREFIX="${D}_frozen"; FLAG=(--finetune frozen);                 LR_FLAG=--head-lr; DEFAULT=1e-3; GRID=(3e-4 3e-3 1e-2) ;;
    lpft)   PREFIX="${D}_lpft";   FLAG=(--finetune lpft --head-lr 1e-3);    LR_FLAG=--lr;      DEFAULT=5e-5; GRID=(1e-5 2e-5 1e-4) ;;
    *)      log "unexpected mode '$BEST'"; exit 1 ;;
esac
RUNS=("${PREFIX}_lr${DEFAULT}")
for lr in "${GRID[@]}"; do
    run "$BEST $LR_FLAG=$lr" "${FLAG[@]}" "$LR_FLAG" "$lr" --tag "lr$lr" "$@"
    RUNS+=("${PREFIX}_lr${lr}")
done

# 4) per-fold learning-rate selection on validation loss, then comparison tables
log "selecting the learning rate per fold on validation loss"
SELECTED="${PREFIX}_lrselected"
"$PY" select_finetune.py nested --out "$SELECTED" "${RUNS[@]}" | tee -a "$LOG"
"$PY" evaluate_runs.py "${RUNS[@]}" --out-dir "$OUT/grid" > /dev/null
"$PY" evaluate_runs.py "${D}_lr5e-5" "${D}_frozen_lr1e-3" "${D}_lpft_lr5e-5" "$SELECTED" --out-dir "$OUT" \
    | tee -a "$LOG"
log "fine-tuning study finished: $OUT/comparison.md, $OUT/grid/comparison.md, $SELECTED/lr_choice.csv"
