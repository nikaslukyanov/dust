"""Launch the fixed/adaptive DUST campaign on one Insomnia L40S per seed."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (42, 43, 44)


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def submission(batch, script, log, *args, dependency=None):
    command = ['sbatch', '--parsable', f'--output={batch / log}']
    if dependency is not None:
        command.append(f'--dependency=afterok:{dependency}')
    return [*command, str(batch / 'source' / 'scripts' / script), str(ROOT), str(batch), *args]


def submit(command, dry_run):
    print(shlex.join(command), flush=True)
    if dry_run:
        return 'DRYRUN'
    return subprocess.check_output(command, text=True).strip().split(';')[0]


def snapshot(batch, launch, *, allow_existing=True):
    if batch.exists():
        if not allow_existing:
            raise ValueError(f'{batch}: choose a new launch directory')
        if json.loads((batch / 'launch.json').read_text()) != launch:
            raise ValueError(f'{batch}: incompatible launch settings or source commit')
        return

    batch.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.ab-launch-', dir=batch.parent) as stage:
        pending = Path(stage) / 'batch'
        source = pending / 'source'
        source.mkdir(parents=True)
        with tempfile.TemporaryFile() as archive:
            subprocess.run(['git', '-C', str(ROOT), 'archive', launch['commit']], stdout=archive, check=True)
            archive.seek(0)
            subprocess.run(['tar', '-x', '-C', str(source)], stdin=archive, check=True)
        (pending / 'launch.json').write_text(json.dumps(launch, indent=2) + '\n')
        pending.rename(batch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    parser.add_argument('--dry-run', action='store_true', help='Print the matrix and shell-safe commands.')
    args = parser.parse_args()
    batch = (args.output or ROOT / f'runs/ab-gpu-{datetime.now():%Y%m%d-%H%M%S}').resolve()
    seeds = list(args.seeds)
    if len(set(seeds)) != len(seeds) or not all(0 <= seed < 2**64 - 3 for seed in seeds):
        parser.error('Seeds must be unique non-negative integers below 2**64 - 3')

    try:
        if args.dry_run:
            from ab_experiment import experiment_plan

            print(json.dumps(experiment_plan(False, seeds), indent=2))
        else:
            if shutil.which('sbatch') is None:
                raise ValueError('Run this command on an Insomnia login node')
            if git('status', '--porcelain'):
                raise ValueError('Commit changes before launching a source snapshot')
            for name in ('data/train_1M.pt', 'data/val.pt', 'data/test.pt', 'data/manifest.json',
                         '.venv/wheelhouse-cu128/wheels.ready'):
                if not (ROOT / name).is_file():
                    raise ValueError(f'Missing {name}')
            snapshot(batch, dict(commit=git('rev-parse', 'HEAD'), seeds=seeds))

        setup = None
        if args.dry_run or not (batch / 'environment.ready').is_file():
            setup = submit(submission(batch, 'setup_baseline_gpu.sbatch', 'setup-%j.log', 'ab'), args.dry_run)
        for seed in seeds:
            if not args.dry_run:
                (batch / f'seed-{seed}').mkdir(exist_ok=True)
            submit(submission(batch, 'ab_gpu.sbatch', f'seed-{seed}/job-%j.log', str(seed),
                              dependency=setup), args.dry_run)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
