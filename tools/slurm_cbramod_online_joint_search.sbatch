#!/bin/bash
#SBATCH --job-name=cbra_online_grid
#SBATCH --partition=rtx2080ti
#SBATCH --qos=rtx2080ti
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00

set -euo pipefail

ROOT="$HOME/oi-mi-cbramod-20260801"
PY="$HOME/.conda/envs/oi-mi-cbramod/bin/python"
SUMMARY="$ROOT/outputs/offline_staged_seed42/training_summary.json"
WORKERS_PER_JOB=4
TOTAL_WORKERS=8
TOTAL_CANDIDATES=1152

CHECKPOINT="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["deployment_candidate"]["model"]["path"])' "$SUMMARY")"
OUTPUT_DIR="$ROOT/outputs/online_joint_grid_8gpu_1152_20260801/shards"
mkdir -p "$OUTPUT_DIR"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$ROOT/code"
IFS=',' read -r -a ALLOCATED_GPUS <<< "${CUDA_VISIBLE_DEVICES:?Slurm did not expose allocated GPUs}"
if (( ${#ALLOCATED_GPUS[@]} != WORKERS_PER_JOB )); then
  echo "Expected $WORKERS_PER_JOB allocated GPUs, got: $CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi

worker_pids=()
for ((local_worker = 0; local_worker < WORKERS_PER_JOB; local_worker++)); do
  global_worker=$((SLURM_ARRAY_TASK_ID * WORKERS_PER_JOB + local_worker))
  base_size=$((TOTAL_CANDIDATES / TOTAL_WORKERS))
  remainder=$((TOTAL_CANDIDATES % TOTAL_WORKERS))
  START_INDEX=$((global_worker * base_size + (global_worker < remainder ? global_worker : remainder)))
  STOP_INDEX=$((START_INDEX + base_size + (global_worker < remainder ? 1 : 0)))
  worker_output="$OUTPUT_DIR/shard_$(printf '%03d' "$global_worker").json"
  worker_log="$OUTPUT_DIR/shard_$(printf '%03d' "$global_worker").worker.log"
  echo "worker=$global_worker gpu=${ALLOCATED_GPUS[$local_worker]} range=[$START_INDEX,$STOP_INDEX)"
  CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS[$local_worker]}" \
    "$PY" tools/search_neuroonline_online_hyperparams.py \
      --recording "$ROOT/data/records_storage/S001/realtime/20260728_184606" \
      --checkpoint "$CHECKPOINT" \
      --output "$worker_output" \
      --model-name cbramod \
      --sfreq 200 \
      --n-classes 3 \
      --learning-rates 1e-6,1e-5,3e-5,1e-4,3e-4,1e-3 \
      --epochs-grid 1,3,5 \
      --batch-sizes 16,32,64,128 \
      --mask-ratios 0.1,0.3,0.5,0.7 \
      --lambda-grid 0.1,0.25,0.5,1.0 \
      --selection-start 160 \
      --selection-end 320 \
      --start-index "$START_INDEX" \
      --stop-index "$STOP_INDEX" \
      >"$worker_log" 2>&1 &
  worker_pids+=("$!")
done

status=0
for pid in "${worker_pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
