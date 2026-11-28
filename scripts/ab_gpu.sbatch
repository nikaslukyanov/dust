#!/bin/bash
#SBATCH --job-name=dust-ab
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --qos=free_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --time=10:00:00
#SBATCH --no-requeue
# Submitted by scripts/launch_ab.py. Arguments: REPO BATCH SEED.

set -euo pipefail
repo=$1
batch=$2
seed=$3
test -f "$batch/environment.ready"
module load anaconda/2023.09
unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
scratch=$(mktemp -d "/tmp/dust-ab-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
cp "$repo/data/"{train_1M.pt,val.pt,test.pt,manifest.json} "$scratch/"
nvidia-smi > "$batch/seed-$seed/nvidia-smi-${SLURM_JOB_ID}.txt"
cd "$batch/source"
"$repo/.venv/baseline-cu128/bin/python" scripts/ab_experiment.py run \
    --output "$batch" --seed "$seed" --data-dir "$scratch"
