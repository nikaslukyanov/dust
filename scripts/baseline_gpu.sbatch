#!/bin/bash
#SBATCH --job-name=dust-full-vs-suffix
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
# Submit through scripts/launch_baseline.sh, not by hand. Every run is pinned to one GPU model (L40S)
# so step times from different jobs are comparable; the environment stage also refuses any other GPU.
#
# Arguments: REPO BATCH SEED ORDER
#   BATCH  launcher's batch folder (holds source/ and environment.ready); this job writes BATCH/seed-SEED
#   ORDER  full-first | suffix-first: which execution mode runs first on the GPU
# Environment (optional): WANDB_PROJECT, WANDB_ENTITY (log every run to W&B), RUN_BACKPROP=0 (skip the
#   backprop reference), plus anything wandb itself reads (WANDB_MODE=offline if nodes lack internet).

set -euo pipefail
repo=$1
batch=$2
seed=$3
order=$4
experiment="$batch/seed-$seed"
mkdir -p "$experiment"
case "$order" in full-first|suffix-first) ;; *) echo "ORDER must be full-first or suffix-first" >&2; exit 2 ;; esac
stage=starting
finish() {
    result=$?
    trap - EXIT
    printf 'stage=%s\nexit_code=%s\nfinished=%s\njob_id=%s\n' \
        "$stage" "$result" "$(date --iso-8601=seconds)" "$SLURM_JOB_ID" > "$experiment/status.txt"
    exit "$result"
}
trap finish EXIT

test -f "$batch/environment.ready"
module load anaconda/2023.09
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TOKENIZERS_PARALLELISM=false
python_bin="$repo/.venv/baseline-cu128/bin/python"
scratch=$(mktemp -d "/tmp/dust-baseline-${SLURM_JOB_ID}-XXXXXX")
export HF_HOME="$scratch/hf-cache"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
cd "$batch/source"
wandb_args=()
if [ -n "${WANDB_PROJECT:-}" ]; then
    wandb_args=(--wandb "$WANDB_PROJECT" --wandb-group "$(basename "$batch")" --wandb-tags 1M K256 "seed$seed")
    if [ -n "${WANDB_ENTITY:-}" ]; then wandb_args+=(--wandb-entity "$WANDB_ENTITY"); fi
fi

stage=environment
date --iso-8601=seconds
hostname
nvidia-smi > "$experiment/nvidia-smi.txt"
"$python_bin" - <<'PY' | tee "$experiment/gpu-environment.json"
import json, platform, torch
assert torch.cuda.is_available(), 'CUDA is unavailable'
assert torch.cuda.device_count() == 1, 'Expected exactly one allocated GPU'
assert torch.cuda.is_bf16_supported(), 'GPU must support bfloat16'
assert 'L40S' in torch.cuda.get_device_name(0), f'Expected an L40S, got {torch.cuda.get_device_name(0)}'
assert torch.__version__.startswith('2.10.0') and torch.version.cuda == '12.8'
device = torch.cuda.get_device_properties(0)
x = torch.randn(64, 64, device='cuda', dtype=torch.bfloat16)
assert torch.isfinite(x @ x).all().item()
torch.cuda.synchronize()
print(json.dumps(dict(python=platform.python_version(), torch=torch.__version__,
    cuda=torch.version.cuda, gpu=device.name, gpu_memory_bytes=device.total_memory,
    compute_capability=list(torch.cuda.get_device_capability())), indent=2))
PY

stage=data_verification
mkdir -p "$scratch/data"
cp "$repo/data/"{train_1M.pt,train_10M.pt,val.pt,test.pt,manifest.json} "$scratch/data/"
"$python_bin" scripts/verify_data.py "$scratch/data"

stage=preflight
"$python_bin" -u new_implementation/gpu_preflight.py --device cuda --tokens 1M \
    --population 256 --launch-bound --data-dir "$scratch/data" --output "$experiment/preflight" \
    2>&1 | tee "$experiment/preflight.log"
"$python_bin" - "$experiment/preflight/preflight.json" <<'PY'
import json, math, sys
report = json.load(open(sys.argv[1]))
checks = report['checks']
assert set(checks) == {'environment', 'equivalence', 'timing', 'probe', 'training', 'launch-bound'}
assert checks['environment']['status'] == 'PASS', 'Environment must pass without warnings'
assert all(item['status'] != 'FAIL' for item in checks.values()), 'Preflight failed'
assert math.isfinite(checks['equivalence']['max_relative_difference'])
for name, item in checks.items():
    if item['status'] == 'WARN':
        print(f'Preflight warning retained in report: {name}: {item}')
PY

run_dust() {  # $1 = full-replay | cached-suffix
    local mode=$1 extra=()
    if [ "$mode" = full-replay ]; then extra=(--full-replay); fi
    if [ ${#wandb_args[@]} -gt 0 ]; then extra+=("${wandb_args[@]}" --wandb-name "$mode-s$seed"); fi
    stage=${mode//-/_}
    "$python_bin" -u dust.py --device cuda --tokens 1M --population 256 --seed "$seed" \
        --probe-interval 10 --checkpoint-interval 10 ${extra[@]+"${extra[@]}"} \
        --data-dir "$scratch/data" --output "$experiment/$mode" \
        2>&1 | tee "$experiment/$mode.log"
}
if [ "$order" = full-first ]; then
    run_dust full-replay; run_dust cached-suffix
else
    run_dust cached-suffix; run_dust full-replay
fi

if [ "${RUN_BACKPROP:-1}" != 0 ]; then
    stage=backprop
    extra=()
    if [ ${#wandb_args[@]} -gt 0 ]; then extra=("${wandb_args[@]}" --wandb-name "backprop-s$seed"); fi
    "$python_bin" -u baselines/backprop.py --device cuda --tokens 1M --seed "$seed" ${extra[@]+"${extra[@]}"} \
        --data-dir "$scratch/data" --output "$experiment/backprop" \
        2>&1 | tee "$experiment/backprop.log"
fi

stage=comparison
"$python_bin" scripts/check_baseline_results.py "$experiment" --seed "$seed" | tee "$experiment/summary.log"
"$python_bin" new_implementation/compare_runs.py \
    "$experiment/full-replay" "$experiment/cached-suffix" | tee "$experiment/comparison.txt"
stage=completed
date --iso-8601=seconds
