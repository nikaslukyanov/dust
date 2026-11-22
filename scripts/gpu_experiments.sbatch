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
# TASKS: any of profile track3 track2 track3train (default: the first three, cheapest first).
#   profile  torch.profiler on one estimate, suffix + full replay: which SDPA kernel runs, per-phase GPU time.
#   track3   multiplexing sweep on the real model under bf16: blocks 0/3/6, sigma 0.05-0.8, nrep 2 and 8.
#   track2   1M / population 256: fixed allocation, then --adaptive variance and agreement, same seed.
#   track3train  cache vs cache + --multiplex hadamard training runs, writer/hidden draws rounded to 4.
#   track3control  cache with half the writer/hidden draws: the same forward passes as cache + C.
#   fastgrid  --fast check (equivalence, speedup), then fast runs at 1, 0.75, 0.5, 0.25 x draws, +/- --antithetic.
#   track2b  B retune round 1: objective gradient, FLOP budget, both, agreement + FLOP budget, measure-only.
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
    track3train)
        # cache vs cache + C in training. --direct-multiple 4 rounds writer/hidden draws (110 -> 120 per
        # type) so the code sequences fit; the cache run uses the same recipe, so the two match.
        run() {
            local name=$1; shift
            local resume=()
            if ls "$out/track3train/$name"/state*.pt >/dev/null 2>&1; then resume=(--resume); fi
            "$py" -u dust.py --device cuda --tokens 1M --population 256 --seed "$seed" --direct-multiple 4 \
                --probe-interval 10 --checkpoint-interval 10 "$@" ${resume[@]+"${resume[@]}"} \
                --output "$out/track3train/$name" 2>&1 | tee -a "$out/track3train-$name.log"
        }
        run cache
        run hadamard --multiplex hadamard
        "$py" new_implementation/compare_runs.py \
            "$out/track3train/cache" "$out/track3train/hadamard" | tee "$out/track3train-comparison.txt"
        ;;
    track2b)
        # B retune, round 1: existing flags only. Compared against track2/fixed (same seed, same recipe).
        run() {
            local name=$1; shift
            local resume=()
            if ls "$out/track2b/$name"/state*.pt >/dev/null 2>&1; then resume=(--resume); fi
            "$py" -u dust.py --device cuda --tokens 1M --population 256 --seed "$seed" \
                --probe-interval 10 --checkpoint-interval 10 --adaptive "$@" ${resume[@]+"${resume[@]}"} \
                --output "$out/track2b/$name" 2>&1 | tee -a "$out/track2b-$name.log"
        }
        run variance-gradient --allocation-policy variance --allocation-objective gradient
        run variance-flops --allocation-policy variance --allocation-budget flops
        run variance-gradient-flops --allocation-policy variance --allocation-objective gradient --allocation-budget flops
        run agreement-flops --allocation-policy agreement --allocation-budget flops
        run measure-only --allocation-policy variance --allocation-interval 0
        "$py" new_implementation/compare_runs.py "$out/track2/fixed" \
            "$out/track2b/variance-gradient" "$out/track2b/variance-flops" "$out/track2b/variance-gradient-flops" \
            "$out/track2b/agreement-flops" "$out/track2b/measure-only" | tee "$out/track2b-comparison.txt"
        ;;
    track3control)
        # The decisive control for C: plain cache DUST with the rounded recipe's writer/hidden draws halved,
        # i.e. the same forward passes as cache + hadamard in track3train.
        "$py" -u dust.py --device cuda --tokens 1M --population 256 --seed "$seed" --direct-multiple 4 \
            --direct-scale 0.5 --probe-interval 10 --checkpoint-interval 10 \
            $(ls "$out/track3train/cache-half"/state*.pt >/dev/null 2>&1 && echo --resume) \
            --output "$out/track3train/cache-half" 2>&1 | tee -a "$out/track3train-cache-half.log"
        "$py" new_implementation/compare_runs.py "$out/track3train/cache" \
            "$out/track3train/cache-half" "$out/track3train/hadamard" | tee "$out/track3control-comparison.txt"
        ;;
    fastgrid)
        # Fast path check (equivalence + speedup), then the draw-count curve with the fast path, with and
        # without antithetic draws. Reference: track2/fixed (plain path, full recipe, same seed).
        "$py" -u new_implementation/fast_check.py --tokens 1M --population 256 --output "$out/fast-check" \
            2>&1 | tee "$out/fast-check.log"
        run() {
            local name=$1; shift
            local resume=()
            if ls "$out/fastgrid/$name"/state*.pt >/dev/null 2>&1; then resume=(--resume); fi
            "$py" -u dust.py --device cuda --tokens 1M --population 256 --seed "$seed" --fast \
                --probe-interval 10 --checkpoint-interval 10 "$@" ${resume[@]+"${resume[@]}"} \
                --output "$out/fastgrid/$name" 2>&1 | tee -a "$out/fastgrid-$name.log"
        }
        run full
        run full-anti --antithetic
        run half --direct-scale 0.5
        run half-anti --direct-scale 0.5 --antithetic
        run quarter --direct-scale 0.25
        run quarter-anti --direct-scale 0.25 --antithetic
        run three-quarter --direct-scale 0.75
        "$py" new_implementation/compare_runs.py "$out/track2/fixed" "$out/fastgrid/full" "$out/fastgrid/full-anti" \
            "$out/fastgrid/three-quarter" "$out/fastgrid/half" "$out/fastgrid/half-anti" \
            "$out/fastgrid/quarter" "$out/fastgrid/quarter-anti" | tee "$out/fastgrid-comparison.txt"
        ;;
    *) echo "Unknown task: $task" >&2; exit 2 ;;
    esac
done
date --iso-8601=seconds > "$out/done.txt"
