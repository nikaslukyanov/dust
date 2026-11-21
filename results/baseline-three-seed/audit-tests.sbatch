#!/bin/bash
#SBATCH --job-name=dust-audit-tests
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --qos=free_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
set -euo pipefail
module load anaconda/2023.09
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export WANDB_MODE=offline
repo=/insomnia001/home/pts2125/hpml/dust
batch="$repo/runs/baseline-gpu-20261118-172822"
cd "$batch/audit-source"
"$repo/.venv/baseline-cu128/bin/python" -m pytest -q -p no:cacheprovider \
    new_implementation scripts/test_check_baseline_results.py \
    --basetemp="$batch/audit-cpu-tests"
