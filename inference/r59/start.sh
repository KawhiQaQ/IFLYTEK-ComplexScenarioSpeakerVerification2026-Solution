#!/bin/sh
# POSIX entry point: evaluators may invoke this file with sh, bash, or directly.
set -eu
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
printf '%s\n' 'MIWU R58: starting V361 inference'
exec python3 -u "${SCRIPT_DIR}/run.py" "$@"
