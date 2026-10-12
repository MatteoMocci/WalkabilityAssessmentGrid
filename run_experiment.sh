#!/usr/bin/env bash
# Linux/macOS shortcut for: python study.py first-experiment [options]  (any OS: see README)
cd "$(dirname "$0")"
PY=.venv/bin/python
[[ -x "$PY" ]] || PY=python3
exec "$PY" study.py first-experiment "$@"
