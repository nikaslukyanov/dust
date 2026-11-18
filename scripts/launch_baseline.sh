#!/bin/bash
# Launch the full-replay vs cached-suffix baseline (plus the backprop reference) on Insomnia.
#
# Run on the login node from the repository root:
#   scripts/launch_baseline.sh --seeds "42 43 44" --order alternate --wandb dust-hpml --wandb-entity TEAM
#
# What it does:
#   1. Refuses uncommitted changes, so every result maps to one git commit.
#   2. Creates runs/baseline-gpu-<timestamp>/ with source/ (a `git archive` of HEAD) and launch.json.
#   3. Submits one setup job (venv + CPU tests), then one L40S job per seed that starts only if the
#      setup succeeds. Each seed writes runs/baseline-gpu-<timestamp>/seed-<seed>/.
#
# Before the first launch (once per checkout):
#   python3 scripts/stage_baseline_wheels.py .venv/wheelhouse-cu128   # then touch .../wheels.ready
#   python prepare_data.py ...                                        # data/ with manifest.json
#   wandb login                                                       # only if using --wandb
#
# Options:
#   --seeds "S1 S2 ..."   seeds, one GPU job each (default: 42)
#   --order MODE          full-first | suffix-first | alternate (default: alternate; the first seed
#                         runs full replay first, the next cached suffix first, and so on)
#   --wandb PROJECT       log every run to W&B under PROJECT
#   --wandb-entity NAME   W&B team or user
#   --no-backprop         skip the backprop reference run
#   --mail EMAIL          Slurm mail on END/FAIL
#   --dry-run             create the folder and print the sbatch commands without submitting
set -euo pipefail

seeds=42
order=alternate
wandb_project=
wandb_entity=
run_backprop=1
mail=
dry_run=0
while [ $# -gt 0 ]; do
    case "$1" in
        --seeds) seeds=$2; shift 2 ;;
        --order) order=$2; shift 2 ;;
        --wandb) wandb_project=$2; shift 2 ;;
        --wandb-entity) wandb_entity=$2; shift 2 ;;
        --no-backprop) run_backprop=0; shift ;;
        --mail) mail=$2; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done
case "$order" in full-first|suffix-first|alternate) ;; *) echo "--order must be full-first, suffix-first or alternate" >&2; exit 2 ;; esac
for seed in $seeds; do
    case "$seed" in ''|*[!0-9]*) echo "Seeds must be non-negative integers, got '$seed'" >&2; exit 2 ;; esac
done

repo=$(git rev-parse --show-toplevel)
cd "$repo"
if ! git diff --quiet HEAD -- || [ -n "$(git ls-files --others --exclude-standard)" ]; then
    echo "Uncommitted or untracked changes: commit them first, so the run maps to one commit." >&2
    exit 1
fi
for file in data/train_1M.pt data/train_10M.pt data/val.pt data/test.pt data/manifest.json \
            .venv/wheelhouse-cu128/wheels.ready; do
    test -e "$file" || { echo "Missing $file (see the header of this script)." >&2; exit 1; }
done

commit=$(git rev-parse HEAD)
batch="$repo/runs/baseline-gpu-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$batch/source"
git archive HEAD | tar -x -C "$batch/source"
python3 - "$batch/launch.json" "$commit" "$order" "$wandb_project" "$wandb_entity" "$run_backprop" $seeds <<'PY'
import json, sys, datetime
path, commit, order, project, entity, backprop, *seeds = sys.argv[1:]
json.dump(dict(commit=commit, seeds=[int(s) for s in seeds], order=order, gpu='l40s',
               wandb=dict(project=project or None, entity=entity or None), backprop=backprop == '1',
               launched=datetime.datetime.now().astimezone().isoformat(timespec='seconds')),
          open(path, 'w'), indent=2)
PY

mail_args=()
if [ -n "$mail" ]; then mail_args=(--mail-user="$mail" --mail-type=END,FAIL); fi
submit() {
    if [ "$dry_run" = 1 ]; then
        echo "sbatch $*" >&2
        echo "DRYRUN"
    else
        sbatch --parsable "$@"
    fi
}

setup_id=$(submit ${mail_args[@]+"${mail_args[@]}"} --output="$batch/setup-%j.log" \
    scripts/setup_baseline_gpu.sbatch "$repo" "$batch")
echo "setup job: $setup_id"
export WANDB_PROJECT="$wandb_project" WANDB_ENTITY="$wandb_entity" RUN_BACKPROP="$run_backprop"
index=0
for seed in $seeds; do
    seed_order=$order
    if [ "$order" = alternate ]; then
        if [ $((index % 2)) = 0 ]; then seed_order=full-first; else seed_order=suffix-first; fi
    fi
    mkdir -p "$batch/seed-$seed"
    job_id=$(submit ${mail_args[@]+"${mail_args[@]}"} --dependency="afterok:$setup_id" --export=ALL \
        --output="$batch/seed-$seed/job-%j.log" \
        scripts/baseline_gpu.sbatch "$repo" "$batch" "$seed" "$seed_order")
    echo "seed $seed ($seed_order): job $job_id"
    index=$((index + 1))
done
echo "batch folder: $batch"
