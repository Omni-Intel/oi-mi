#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-/data/abc/oi-mi-cbramod-20260802}"
PY="${PY:-$ROOT/.venv/bin/python}"
CODE="${CODE:-$ROOT/code}"
OUTPUT="${OUTPUT:-$ROOT/outputs/offline_group_cv_20260728_v3_lr1e6_bs128/offline_search}"
DATASET="${DATASET:-$ROOT/data/records_storage/S001/calibration/20260728_182748/training_windows_main.npz}"

export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$CODE"
mkdir -p "$OUTPUT/logs"

total="$($PY -c 'from tools.search_cbramod_offline_group_cv import OFFLINE_SEARCH_CANDIDATES; print(len(OFFLINE_SEARCH_CANDIDATES))')"
for ((candidate = 0; candidate < total; candidate++)); do
  printf '%s candidate=%d/%d\n' "$(date --iso-8601=seconds)" "$candidate" "$((total - 1))"
  "$PY" tools/search_cbramod_offline_group_cv.py \
    --dataset "$DATASET" \
    --output "$OUTPUT" \
    --config config.yaml \
    --candidate-index "$candidate" \
    --folds 5 \
    --seeds 17,42,2026 \
    --epochs 50
done

"$PY" tools/search_cbramod_offline_group_cv.py \
  --dataset "$DATASET" \
  --output "$OUTPUT" \
  --summarize

printf '%s OFFLINE_SEARCH_COMPLETE\n' "$(date --iso-8601=seconds)"
