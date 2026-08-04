#!/bin/bash
#SBATCH --job-name=cbra_offline_cv
#SBATCH --partition=rtx2080ti
#SBATCH --qos=rtx2080ti
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00

set -euo pipefail

ROOT="$HOME/oi-mi-cbramod-20260801"
PY="$HOME/.conda/envs/oi-mi-cbramod/bin/python"
DATASET="$ROOT/data/records_storage/S001/calibration/20260728_182748/training_windows_main.npz"
OUTPUT_DIR="$ROOT/outputs/offline_group_cv_20260728_v3_one_stage/offline_search"
WORKERS_PER_JOB=4
TOTAL_WORKERS=8
TOTAL_CANDIDATES=320

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
        "$PY" tools/search_cbramod_offline_group_cv.py \
          --dataset "$DATASET" \
          --output "$OUTPUT_DIR" \
          --config config.yaml \
          --candidate-index "$candidate_index" \
           --folds 5 \
           --seeds 17,42,2026 \
           --epochs 50
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
