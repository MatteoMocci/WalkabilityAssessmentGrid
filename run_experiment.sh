#!/usr/bin/env bash
# First experiment: street-only model, ordinal head then CE baseline, all 10 folds.
#
# Shows progress bars (folds / epochs / batches) in the terminal. Logs go to:
#   <out-dir>/street_<model>_<head>/train.log   full training log of each run
#   <out-dir>/experiment_<model>_<time>.log   start/end/outcome of each head
#
# Re-running resumes: folds whose metrics file exists are skipped.
# Extra arguments are passed to run_ordinal.py for both heads, e.g.
#   ./run_experiment.sh                      Swin-B, both heads
#   ./run_experiment.sh --model vit_base     another backbone
#   ./run_experiment.sh --folds 0            a single fold
#   ./run_experiment.sh --monitor qwk        select epochs on validation QWK (runs saved as *_qwk)
#   ./run_experiment.sh --schedule article   warmup + cosine + layer-wise lr decay (runs saved as *_sched)

set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
# defaults of run_ordinal.py, overridden by the same flags if given (used for log/result paths)
OUT_DIR=results
MODEL=swin_base
TAG=""
MON=""
SCH=""
prev=""
for a in "$@"; do
    case "$prev" in
        --model) MODEL="$a" ;;
        --out-dir) OUT_DIR="$a" ;;
        --tag) TAG="_$a" ;;
        --monitor) [[ "$a" == "qwk" ]] && MON="_qwk" ;;
        --schedule) [[ "$a" == "article" ]] && SCH="_sched" ;;
    esac
    prev="$a"
done

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/experiment_${MODEL}_$(date +%Y%m%d_%H%M%S).log"
log() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$LOG"; }

log "experiment start: model=$MODEL args: $*"
for head in ordinal ce; do
    log "[$head] start"
    t0=$SECONDS
    if "$PY" run_ordinal.py --head "$head" "$@"; then
        log "[$head] done in $(( (SECONDS - t0) / 60 )) min -> $OUT_DIR/street_${MODEL}_${head}${MON}${SCH}${TAG}/summary.csv"
    else
        log "[$head] FAILED after $(( (SECONDS - t0) / 60 )) min, see $OUT_DIR/street_${MODEL}_${head}${MON}${SCH}${TAG}/train.log"
        exit 1
    fi
done
log "experiment finished"
