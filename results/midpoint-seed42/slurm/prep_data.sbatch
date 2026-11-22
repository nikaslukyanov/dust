#!/bin/bash
#SBATCH --job-name=dust-prep-data
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --qos=free_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
cd /insomnia001/depts/edu/users/nl2951/dust
module load anaconda/2023.09
unset PYTHONPATH; export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
python --version
python -m venv .venv/prep
.venv/prep/bin/python -m pip install -q --index-url https://pypi.org/simple .venv/wheelhouse-cu128/*.whl
.venv/prep/bin/python -m pip install -q -r requirements.txt
export HF_HOME=/insomnia001/depts/edu/users/nl2951/.hf-cache
.venv/prep/bin/python prepare_data.py
ls -la data/
