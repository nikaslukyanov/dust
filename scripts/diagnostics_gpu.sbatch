#!/bin/bash
#SBATCH --job-name=dust-diagnostics
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --qos=free_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --time=04:00:00
#SBATCH --no-requeue

set -euo pipefail
repo=$1
batch=$2
data=$3
shift 3
module load anaconda/2023.09
unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
scratch=$(mktemp -d "/tmp/dust-diagnostics-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
nvidia-smi > "$batch/nvidia-smi.txt"
{
    printf 'Slurm job %s\n' "$SLURM_JOB_ID"
    printf 'Allocated CPU threads %s\n' "$SLURM_CPUS_PER_TASK"
    hostname
    uptime
} > "$batch/environment.txt"
cp "$data/"{train_1M.pt,manifest.json} "$scratch/"
cd "$batch/source"
"$repo/.venv/baseline-cu128/bin/python" scripts/gpu_diagnostics.py "$@" --data-dir "$scratch"
