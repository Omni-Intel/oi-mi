#!/bin/bash
#SBATCH --job-name=cbra_online_final
#SBATCH --partition=rtx2080ti
#SBATCH --qos=rtx2080ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00

set -euo pipefail

ROOT="$HOME/oi-mi-cbramod-20260801"
PY="$HOME/.conda/envs/oi-mi-cbramod/bin/python"
SUMMARY="$ROOT/outputs/offline_staged_seed42/training_summary.json"
CHECKPOINT="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["deployment_candidate"]["model"]["path"])' "$SUMMARY")"
REPORT="$ROOT/outputs/online_joint_grid_8gpu_1152_20260801/search_report.json"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$ROOT/code"
"$PY" tools/search_neuroonline_online_hyperparams.py \
  --recording "$ROOT/data/records_storage/S001/realtime/20260728_184606" \
  --checkpoint "$CHECKPOINT" \
  --output "$REPORT" \
  --model-name cbramod \
  --sfreq 200 \
  --n-classes 3 \
  --learning-rates 1e-6,1e-5,3e-5,1e-4,3e-4,1e-3 \
  --epochs-grid 1,3,5 \
  --batch-sizes 16,32,64,128 \
  --mask-ratios 0.1,0.3,0.5,0.7 \
  --lambda-grid 0.1,0.25,0.5,1.0 \
  --selection-start 160 \
  --selection-end 320
