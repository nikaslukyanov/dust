#!/bin/bash
#SBATCH --job-name=dust-gpu-env
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --qos=free_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --mail-user=pts2125@columbia.edu
#SBATCH --mail-type=FAIL

set -euo pipefail
repo=$1
experiment=$2
cd "$repo"
module load anaconda/2023.09
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
unset PIP_NO_CACHE_DIR
export PIP_CACHE_DIR="$repo/.venv/pip-cache"
env_dir="$repo/.venv/baseline-cu128"
wheelhouse="$repo/.venv/wheelhouse-cu128"
test -f "$wheelhouse/wheels.ready"
python --version
python -m venv "$env_dir"
# Large CUDA wheels are staged and checksum-verified before submission. Keep
# downloads off the setup job's critical path and retain smaller pip downloads.
"$env_dir/bin/python" -m pip install --index-url https://pypi.org/simple "$wheelhouse"/*.whl
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
