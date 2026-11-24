#!/bin/sh
# Run from this folder. Uses the CUDA torch already installed. Does not download weights.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [ -z "${H3_T4_EVIDENCE_DIR:-}" ]; then
  echo "Set H3_T4_EVIDENCE_DIR to a directory under /kaggle/working" >&2
  exit 2
fi
mkdir -p "$H3_T4_EVIDENCE_DIR"
exec python "$ROOT/scripts/dual_gpu_smoke.py" "$@"
