#!/usr/bin/env bash
# Linux/macOS shortcut for: python study.py grid [options]  (any OS: see README)
# The environment variables of earlier versions still work:
#   BACKBONES="swin_base vgg16" ./run_backbone_grid.sh     GRID_OUT=results/other ./run_backbone_grid.sh
cd "$(dirname "$0")"
PY=.venv/bin/python
[[ -x "$PY" ]] || PY=python3
args=()
[[ -n "${BACKBONES:-}" ]] && read -r -a bb <<< "$BACKBONES" && args+=(--backbones "${bb[@]}")
[[ -n "${GRID_OUT:-}" ]] && args+=(--out-dir "$GRID_OUT")
exec "$PY" study.py grid "${args[@]}" "$@"
