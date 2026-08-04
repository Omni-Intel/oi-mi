#!/bin/bash
#SBATCH --job-name=cbra_online_p2
#SBATCH --partition=rtx2080ti
#SBATCH --qos=rtx2080ti
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00

set -euo pipefail

ROOT="$HOME/oi-mi-cbramod-20260801"
PY="$HOME/.conda/envs/oi-mi-cbramod/bin/python"
BASE_DIR="$ROOT/outputs/offline_group_cv_20260728_v3_one_stage"
FINAL_REPORT="$BASE_DIR/final_seed42/training_report.json"
CHECKPOINT="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["artifacts"]["model"]["path"])' "$FINAL_REPORT")"
RECORDING="$ROOT/data/records_storage/S001/realtime/20260728_184606"
MANIFEST="$BASE_DIR/online_stage2_manifest.json"
OUTPUT_DIR="$BASE_DIR/online_stage2"
WORKERS_PER_JOB=2
TOTAL_WORKERS=4
TOTAL_CANDIDATES="$($PY -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["candidates"]))' "$MANIFEST")"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUTPUT_DIR/logs"
cd "$ROOT/code"
IFS=',' read -r -a ALLOCATED_GPUS <<< "${CUDA_VISIBLE_DEVICES:?Slurm did not expose allocated GPUs}"
if (( ${#ALLOCATED_GPUS[@]} != WORKERS_PER_JOB )); then
  echo "Expected $WORKERS_PER_JOB allocated GPUs, got: $CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi

worker_pids=()
for ((local_worker = 0; local_worker < WORKERS_PER_JOB; local_worker++)); do
  global_worker=$((SLURM_ARRAY_TASK_ID * WORKERS_PER_JOB + local_worker))
  worker_log="$OUTPUT_DIR/logs/worker_$(printf '%02d' "$global_worker").log"
  (
    for ((candidate_index = global_worker; candidate_index < TOTAL_CANDIDATES; candidate_index += TOTAL_WORKERS)); do
      echo "worker=$global_worker gpu=${ALLOCATED_GPUS[$local_worker]} candidate=$candidate_index"
      CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS[$local_worker]}" \
        "$PY" tools/search_cbramod_online_paired.py \
          --recording "$RECORDING" \
          --checkpoint "$CHECKPOINT" \
          --candidate-manifest "$MANIFEST" \
          --output "$OUTPUT_DIR" \
          --candidate-index "$candidate_index" \
          --seeds 17,42,2026 \
          --report-block-size 64
    done
  ) >"$worker_log" 2>&1 &
  worker_pids+=("$!")
done

status=0
for pid in "${worker_pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
