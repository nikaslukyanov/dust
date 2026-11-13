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
#SBATCH --constraint=h100|l40s|l40|A6000
#SBATCH --time=10:00:00
#SBATCH --no-requeue
#SBATCH --mail-user=pts2125@columbia.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
repo=$1
experiment=$2
stage=starting
finish() {
    result=$?
    trap - EXIT
    printf 'stage=%s\nexit_code=%s\nfinished=%s\njob_id=%s\n' \
        "$stage" "$result" "$(date --iso-8601=seconds)" "$SLURM_JOB_ID" > "$experiment/status.txt"
    exit "$result"
}
trap finish EXIT

test -f "$experiment/environment.ready"
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
cd "$experiment/source"

stage=environment
date --iso-8601=seconds
hostname
nvidia-smi > "$experiment/nvidia-smi.txt"
"$python_bin" - <<'PY' | tee "$experiment/gpu-environment.json"
import json, platform, torch
assert torch.cuda.is_available(), 'CUDA is unavailable'
assert torch.cuda.device_count() == 1, 'Expected exactly one allocated GPU'
assert torch.cuda.is_bf16_supported(), 'GPU must support bfloat16'
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
"$python_bin" - "$scratch/data" <<'PY'
import hashlib, json, sys
from pathlib import Path
data = Path(sys.argv[1])
manifest = json.loads((data / 'manifest.json').read_text())
for name, info in manifest['files'].items():
    actual = hashlib.sha256((data / name).read_bytes()).hexdigest()
    assert actual == info['sha256'], f'Data checksum mismatch: {name}'
assert hashlib.sha256(Path('tokenizer/tokenizer.json').read_bytes()).hexdigest() == manifest['tokenizer_sha256']
print('Verified data and tokenizer hashes')
PY

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

stage=full_replay
"$python_bin" -u dust.py --device cuda --tokens 1M --population 256 --seed 42 \
    --full-replay --probe-interval 10 --checkpoint-interval 10 \
    --data-dir "$scratch/data" --output "$experiment/full-replay" \
    2>&1 | tee "$experiment/full-replay.log"

stage=cached_suffix
"$python_bin" -u dust.py --device cuda --tokens 1M --population 256 --seed 42 \
    --probe-interval 10 --checkpoint-interval 10 \
    --data-dir "$scratch/data" --output "$experiment/cached-suffix" \
    2>&1 | tee "$experiment/cached-suffix.log"

stage=comparison
"$python_bin" "$experiment/check_baseline_results.py" "$experiment" | tee "$experiment/summary.log"
"$python_bin" new_implementation/compare_runs.py \
    "$experiment/full-replay" "$experiment/cached-suffix" | tee "$experiment/comparison.txt"
stage=completed
date --iso-8601=seconds
