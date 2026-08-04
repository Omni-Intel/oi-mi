#!/usr/bin/env bash

set -euo pipefail

ROOT=/data/abc/oi-mi-cbramod-20260802
PY="$ROOT/.venv/bin/python"
CODE="$ROOT/code"
BASE="${BASE:-$ROOT/outputs/offline_group_cv_20260728_v3_one_stage}"
CALIBRATION="$ROOT/data/records_storage/S001/calibration/20260728_182748/training_windows_main.npz"
RECORDING="$ROOT/data/records_storage/S001/realtime/20260728_184606"
WORKERS="${WORKERS:-1}"

export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$CODE"
mkdir -p "$BASE/pipeline_logs"

run_offline_search() {
  local output="$BASE/offline_search"
  local total
  total="$($PY -c 'from tools.search_cbramod_offline_group_cv import OFFLINE_SEARCH_CANDIDATES; print(len(OFFLINE_SEARCH_CANDIDATES))')"
  mkdir -p "$output/logs"
  local pids=()
  for ((worker = 0; worker < WORKERS; worker++)); do
    (
      for ((candidate = worker; candidate < total; candidate += WORKERS)); do
        "$PY" tools/search_cbramod_offline_group_cv.py \
          --dataset "$CALIBRATION" \
          --output "$output" \
          --config config.yaml \
          --candidate-index "$candidate" \
          --folds 5 \
          --seeds 17,42,2026 \
          --epochs 50
      done
    ) >"$output/logs/local_worker_$(printf '%02d' "$worker").log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  "$PY" tools/search_cbramod_offline_group_cv.py \
    --dataset "$CALIBRATION" \
    --output "$output" \
    --summarize
}

run_online_stage() {
  local manifest="$1"
  local output="$2"
  local checkpoint="$3"
  local total
  total="$($PY -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["candidates"]))' "$manifest")"
  mkdir -p "$output/logs"
  local pids=()
  for ((worker = 0; worker < WORKERS; worker++)); do
    (
      for ((candidate = worker; candidate < total; candidate += WORKERS)); do
        "$PY" tools/search_cbramod_online_paired.py \
          --recording "$RECORDING" \
          --checkpoint "$checkpoint" \
          --candidate-manifest "$manifest" \
          --output "$output" \
          --candidate-index "$candidate" \
          --seeds 17,42,2026 \
          --report-block-size 64
      done
    ) >"$output/logs/local_worker_$(printf '%02d' "$worker").log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  "$PY" tools/search_cbramod_online_paired.py \
    --recording "$RECORDING" \
    --checkpoint "$checkpoint" \
    --candidate-manifest "$manifest" \
    --output "$output" \
    --summarize
}

run_offline_search

"$PY" tools/train_cbramod_offline_final.py \
  --dataset "$CALIBRATION" \
  --offline-summary "$BASE/offline_search/summary.json" \
  --output "$BASE/final_seed42" \
  --config config.yaml \
  --seed 42 \
  --max-epochs 50

"$PY" tools/build_cbramod_online_paired_manifest.py \
  --offline-summary "$BASE/offline_search/summary.json" \
  --final-training-report "$BASE/final_seed42/training_report.json" \
  --output "$BASE/online_search_manifest.json"

CHECKPOINT="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_checkpoint"]["model"]["path"])' "$BASE/online_search_manifest.json")"
run_online_stage "$BASE/online_search_manifest.json" "$BASE/online_search" "$CHECKPOINT"
echo "PIPELINE_COMPLETE"
