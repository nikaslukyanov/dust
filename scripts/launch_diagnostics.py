import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

if __name__ == '__main__' and '--dry-run' in sys.argv[1:]:
    sys.dont_write_bytecode = True

from ab_experiment import write_json
from gpu_diagnostics import TRAIN_FILE, diagnostic_plan, parse_args
from launch_ab import ROOT, git, snapshot, submission, submit

PYTHON = ROOT / '.venv/baseline-cu128/bin/python'


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv, launcher=True)
    batch = args.output.resolve()
    plan = diagnostic_plan(args)
    data = Path(plan['data_dir'])

    inner = [args.command, '--output', str(batch / 'diagnostics'), '--seed', str(args.seed)]
    if plan['checkpoint']:
        inner.extend(['--checkpoint', plan['checkpoint']])
    for name, value in plan['options'].items():
        values = value if isinstance(value, list) else [value]
        inner.extend([f'--{name.replace("_", "-")}', *map(str, values)])
    command = submission(batch, 'diagnostics_gpu.sbatch', 'job-%j.log', str(data), *inner)
    child = [str(PYTHON), 'scripts/gpu_diagnostics.py', *inner, '--data-dir']

    if args.dry_run:
        print(json.dumps(plan, indent=2))
        submit(command, dry_run=True)
        print(f'{shlex.join(child)} "$scratch"')
        return 0

    try:
        if batch.exists():
            raise ValueError(f'{batch}: choose a new launch directory')
        if not shutil.which('sbatch'):
            raise ValueError('Run this command on an Insomnia login node')
        if git('status', '--porcelain'):
            raise ValueError('Commit changes before launching a source snapshot')
        required = [PYTHON, data / TRAIN_FILE, data / 'manifest.json']
        if plan['checkpoint']:
            required.append(Path(plan['checkpoint']))
        for path in required:
            if not path.is_file():
                raise ValueError(f'Missing {path}')
        launch = dict(commit=git('rev-parse', 'HEAD'), plan=plan, sbatch_argv=command,
                      inner_argv_before_scratch=child)
        snapshot(batch, launch, allow_existing=False)
        job_id = submit(command, dry_run=False)
        write_json(batch / 'submission.json', dict(job_id=job_id))
        print(f'Submitted Slurm job {job_id}')
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
