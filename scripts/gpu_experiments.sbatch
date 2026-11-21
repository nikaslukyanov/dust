#!/bin/bash
#SBATCH --job-name=dust-midpoint
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --qos=free_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --time=04:00:00
#SBATCH --no-requeue
# Midpoint GPU experiments on one L40S, run from the repository root:
#   sbatch --output=runs/midpoint-%j.log scripts/gpu_experiments.sbatch "$PWD" [SEED] [TASKS]
#
# TASKS: any of profile track3 track2 (default: all three, cheapest first).
#   profile  torch.profiler on one estimate, suffix + full replay: which SDPA kernel runs, per-phase GPU time.
#   track3   multiplexing sweep on the real model under bf16: blocks 0/3/6, sigma 0.05-0.8, nrep 2 and 8.
#   track2   1M / population 256: fixed allocation, then --adaptive variance and agreement, same seed.
# Needs .venv/baseline-cu128 (built by scripts/launch_baseline.sh's setup job) and data/ from prepare_data.py.
# Writes runs/midpoint-seed<SEED>/. A failed training run continues with the same command: each
# dust.py call below adds --resume when its folder already holds a checkpoint.
set -euo pipefail
repo=$1
seed=${2:-42}
tasks=${3:-profile track3 track2}
cd "$repo"
module load anaconda/2023.09
unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
py="$repo/.venv/baseline-cu128/bin/python"
out="$repo/runs/midpoint-seed$seed"
mkdir -p "$out"
git rev-parse HEAD > "$out/commit.txt" 2>/dev/null || true
nvidia-smi > "$out/nvidia-smi.txt"
"$py" -c 'import torch; name = torch.cuda.get_device_name(0); assert "L40S" in name, f"Expected an L40S, got {name}"; print(name, torch.__version__)'

for task in $tasks; do
    case "$task" in
    profile)
        "$py" -u new_implementation/profile_step.py --device cuda --tokens 1M --population 256 \
            --full-replay --output "$out/profile" 2>&1 | tee "$out/profile.log"
        ;;
    track3)
        for nrep in 2 8; do
            "$py" -u new_implementation/multiplex_experiment.py --real --device cuda --tokens 1M \
                --layers 0 3 6 --sigmas 0.05 0.1 0.2 0.4 0.8 --nrep "$nrep" --passes 16 --seeds 16 \
                --output "$out/track3/nrep$nrep.json" 2>&1 | tee "$out/track3-nrep$nrep.log"
        done
        ;;
    track2)
        run() {  # $1 = folder name, rest = extra dust.py flags
            local name=$1; shift
            local resume=()
            if ls "$out/track2/$name"/state*.pt >/dev/null 2>&1; then resume=(--resume); fi
            "$py" -u dust.py --device cuda --tokens 1M --population 256 --seed "$seed" \
                --probe-interval 10 --checkpoint-interval 10 "$@" ${resume[@]+"${resume[@]}"} \
                --output "$out/track2/$name" 2>&1 | tee -a "$out/track2-$name.log"
        }
        run fixed
        run variance --adaptive --allocation-policy variance
        run agreement --adaptive --allocation-policy agreement
        "$py" new_implementation/compare_runs.py \
            "$out/track2/fixed" "$out/track2/variance" "$out/track2/agreement" | tee "$out/track2-comparison.txt"
        ;;
    *) echo "Unknown task: $task" >&2; exit 2 ;;
    esac
done
date --iso-8601=seconds > "$out/done.txt"
