#!/usr/bin/env bash

set -euo pipefail

ROOT=/data/abc/oi-mi-cbramod-20260802
PY="$ROOT/.venv/bin/python"
CODE="$ROOT/code"
BASE="$ROOT/outputs/offline_group_cv_20260728_v2"
OFFLINE_SUMMARY="$BASE/stage2/summary.json"
FINAL_REPORT="$BASE/final_seed42/training_report.json"
RECORDING="$ROOT/data/records_storage/S001/realtime/20260728_184606"
MANIFEST="$BASE/online_fullgrid_final_eval_manifest.json"
OUTPUT="$BASE/online_fullgrid_final_eval"
WORKERS="${WORKERS:-1}"

export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$CODE"
mkdir -p "$OUTPUT/logs"

"$PY" tools/build_cbramod_online_paired_manifest.py \
  --offline-summary "$OFFLINE_SUMMARY" \
  --final-training-report "$FINAL_REPORT" \
  --output "$MANIFEST"

CHECKPOINT="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_checkpoint"]["model"]["path"])' "$MANIFEST")"
TOTAL="$($PY -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["candidates"]))' "$MANIFEST")"

pids=()
for ((worker = 0; worker < WORKERS; worker++)); do
  (
    for ((candidate = worker; candidate < TOTAL; candidate += WORKERS)); do
      "$PY" tools/search_cbramod_online_paired.py \
        --recording "$RECORDING" \
        --checkpoint "$CHECKPOINT" \
        --candidate-manifest "$MANIFEST" \
        --output "$OUTPUT" \
        --candidate-index "$candidate" \
        --seeds 17,42,2026 \
        --report-block-size 64
    done
  ) >"$OUTPUT/logs/local_worker_$(printf '%02d' "$worker").log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if (( status != 0 )); then
  exit "$status"
fi

"$PY" tools/search_cbramod_online_paired.py \
  --recording "$RECORDING" \
  --checkpoint "$CHECKPOINT" \
  --candidate-manifest "$MANIFEST" \
  --output "$OUTPUT" \
  --summarize

echo "ONLINE_SEARCH_COMPLETE"
