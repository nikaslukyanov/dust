#!/bin/bash
#SBATCH --job-name=dust-gpu-env
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --qos=free_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --no-requeue

set -euo pipefail
repo=$1
experiment=$2
cd "$repo"
module load anaconda/2023.09
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1
env_dir="$repo/.venv/baseline-cu128"
python --version
python -m venv "$env_dir"
"$env_dir/bin/python" -m pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cu128
"$env_dir/bin/python" -m pip install -r "$experiment/source/requirements.txt" pytest
"$env_dir/bin/python" -m pip check
"$env_dir/bin/python" -m pip freeze > "$experiment/environment.txt"
"$env_dir/bin/python" -c 'import torch, tokenizers, datasets; assert torch.__version__.startswith("2.10.0"); assert torch.version.cuda == "12.8"; print("CUDA wheel ready:", torch.__version__, torch.version.cuda)'

# Exercise the core CPU equivalence and training/checkpoint/probe paths before requesting GPU work.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd "$experiment/source"
"$env_dir/bin/python" -m pytest -q -p no:cacheprovider \
    new_implementation/test_suffix.py new_implementation/test_gradient_check.py \
    new_implementation/test_checkpoint.py new_implementation/test_training.py \
    --basetemp="$experiment/cpu-tests"
date --iso-8601=seconds > "$experiment/environment.ready"
