#!/bin/bash
#SBATCH --job-name=cbra_offline_final
#SBATCH --partition=rtx2080ti
#SBATCH --qos=rtx2080ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00

set -euo pipefail

ROOT="$HOME/oi-mi-cbramod-20260801"
PY="$HOME/.conda/envs/oi-mi-cbramod/bin/python"
BASE_DIR="$ROOT/outputs/offline_group_cv_20260728_v3_one_stage"
DATASET="$ROOT/data/records_storage/S001/calibration/20260728_182748/training_windows_main.npz"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$ROOT/code"
"$PY" tools/search_cbramod_offline_group_cv.py \
  --dataset "$DATASET" \
  --output "$BASE_DIR/offline_search" \
  --summarize

"$PY" tools/train_cbramod_offline_final.py \
  --dataset "$DATASET" \
  --offline-summary "$BASE_DIR/offline_search/summary.json" \
  --output "$BASE_DIR/final_seed42" \
  --config config.yaml \
  --seed 42 \
  --max-epochs 50

"$PY" tools/build_cbramod_online_paired_manifest.py \
  --offline-summary "$BASE_DIR/offline_search/summary.json" \
  --final-training-report "$BASE_DIR/final_seed42/training_report.json" \
  --output "$BASE_DIR/online_stage1_manifest.json"

ONLINE_JOB="$(sbatch --parsable tools/slurm_cbramod_online_paired_stage1.sbatch)"
echo "ONLINE_STAGE1_JOB=$ONLINE_JOB"
