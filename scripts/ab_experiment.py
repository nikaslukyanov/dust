"""Run matched fixed/adaptive DUST experiments and summarize paired seed results."""
import argparse
import fcntl
import hashlib
import json
import os
import pickle
import platform
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dust import (  # noqa: E402
    DEFAULT_CONFIGS, GPTConfig, Recipe, atomic_write_text, data_hash, load_sequences, recipe, train,
)
from allocation import Allocator  # noqa: E402
from toy import TOY_BATCH, TOY_CONFIG, TOY_EVAL_INTERVAL, TOY_STEPS, toy_data, toy_recipe  # noqa: E402
from ab_results import check_history, check_records, read_records, summarize_campaign, validate_result  # noqa: E402
from verify_data import sha256, verify_data  # noqa: E402

DEFAULT_SEEDS = (42, 43, 44)
WARMUP_STEPS = 3
DATA_FILES = ('train_1M.pt', 'val.pt', 'test.pt')
SOURCE_FILES = (
    'dust.py', 'model.py', 'requirements.txt', 'scripts/ab_experiment.py', 'scripts/ab_results.py',
    'scripts/check_baseline_results.py', 'scripts/verify_data.py', 'scripts/launch_ab.py',
    'scripts/ab_gpu.sbatch', 'scripts/setup_baseline_gpu.sbatch',
    *(f'new_implementation/{name}.py' for name in
      ('accounting', 'allocation', 'checkpoint', 'groups', 'multiplexing', 'run_metrics', 'toy', 'tracking')),
)
RESULT_FILES = ('config.json', 'metrics.jsonl', 'result.json', 'best.pt', 'state.pt')
TRAINER_TEMP_FILES = ('config.json.tmp', 'metrics.jsonl.tmp', 'result.json.tmp', 'state.pt.tmp')


def validate_seeds(seeds):
    if not seeds or len(set(seeds)) != len(seeds) or not all(0 <= seed < 2**64 - 3 for seed in seeds):
        raise ValueError('Seeds must be unique non-negative integers below 2**64 - 3')


def experiment_plan(toy, seeds, steps=None, checkpoint_interval=None):
    validate_seeds(seeds)
    total_steps = TOY_STEPS if toy else DEFAULT_CONFIGS['1M']['steps']
    if steps is not None and (not toy or not WARMUP_STEPS < steps <= total_steps):
        raise ValueError(f'--steps is for toy runs and must be between {WARMUP_STEPS + 1} and {TOY_STEPS}')
    if checkpoint_interval is not None and checkpoint_interval < 1:
        raise ValueError('--checkpoint-interval must be positive')
    steps = steps or (5 if toy else total_steps)
    batch = TOY_BATCH if toy else 8
    config = TOY_CONFIG if toy else GPTConfig()
    stock = toy_recipe(scale=2) if toy else recipe('1M', 256)
    allocation = {
        policy: dict(policy=policy, interval=4, ema=0.7, rate=0.5, minimum=2 * stock.nrep,
                     budget='draws', objective='update')
        for policy in ('variance', 'agreement')
    }
    initial = {policy: Allocator(stock, policy=policy).recipe(stock) for policy in allocation}
    matrix = [
        ('fixed', stock, stock, None, None),
        ('adaptive-variance', stock, initial['variance'], allocation['variance'], 'fixed'),
        ('adaptive-agreement', stock, initial['agreement'], allocation['agreement'], 'fixed'),
        ('fixed-variance', initial['variance'], initial['variance'], None, None),
        ('measure-variance', stock, initial['variance'], dict(allocation['variance'], interval=0), 'fixed-variance'),
        ('fixed-agreement', initial['agreement'], initial['agreement'], None, None),
        ('measure-agreement', stock, initial['agreement'], dict(allocation['agreement'], interval=0), 'fixed-agreement'),
    ]
    cases = [dict(name=name, requested_recipe=asdict(requested), initial_recipe=asdict(effective),
                  allocation=options, control=control)
             for name, requested, effective, options, control in matrix]
    names = [case['name'] for case in cases]
    runs = []
    for index, seed in enumerate(seeds):
        offset = index % len(names)
        runs.append(dict(seed=seed, path=f'seed-{seed}', order=names[offset:] + names[:offset],
                         paths={name: f'seed-{seed}/{name}' for name in names}))
    return dict(
        version=1, seeds=list(seeds), settings=dict(
            model=asdict(config), steps=steps, batch_size=batch, train_sequences=total_steps * batch,
            training_tokens=steps * batch * config.sequence_len,
            evaluation_interval=TOY_EVAL_INTERVAL if toy else DEFAULT_CONFIGS['1M']['eval_interval'],
            validation_sequences=32 if toy else 544, test_sequences=32 if toy else 544,
            checkpoint_interval=checkpoint_interval or (1 if toy else 10),
            warmup_steps=WARMUP_STEPS, execution='cached-suffix', world=1, probe_interval=0,
            mode='cpu-toy' if toy else 'l40s', matmul_precision='high',
        ),
        cases=cases, runs=runs, summary_path='summary.json',
    )


def add_experiment_options(parser):
    parser.add_argument('--toy', action='store_true', help='Use synthetic data and a small CPU model.')
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    parser.add_argument('--steps', type=int, help='Limit the toy run to 4–40 steps (default: 5).')
    parser.add_argument('--checkpoint-interval', type=int, help='Save state every N steps and at the end.')


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    atomic_write_text(path, json.dumps(value, indent=2, allow_nan=False) + '\n')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_identity(commit):
    return dict(commit=commit, files={name: sha256(ROOT / name) for name in SOURCE_FILES})


def source_commit():
    if not (ROOT / '.git').exists():
        return None
    return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()


def actual_runtime(toy):
    runtime = dict(
        device='cpu' if toy else 'cuda:0', python=platform.python_version(), torch=str(torch.__version__),
        system=platform.system(), architecture=platform.machine(), torch_threads=torch.get_num_threads(),
    )
    if not toy:
        if torch.cuda.device_count() != 1:
            raise ValueError('The GPU campaign requires exactly one visible CUDA GPU')
        torch.cuda.set_device(0)
        gpu = torch.cuda.get_device_properties(0)
        if 'L40S' not in gpu.name or not torch.cuda.is_bf16_supported():
            raise ValueError(f'The GPU campaign requires one bfloat16-capable L40S, found {gpu.name}')
        if not torch.__version__.startswith('2.10.0') or torch.version.cuda != '12.8':
            raise ValueError('The GPU campaign requires PyTorch 2.10.0 with CUDA 12.8')
        runtime.update(gpu=gpu.name, gpu_memory_bytes=gpu.total_memory,
                       compute_capability=list(torch.cuda.get_device_capability(0)), cuda=torch.version.cuda)
    return runtime


def verified_data(directory):
    if directory is None:
        raise ValueError('The GPU campaign requires --data-dir')
    return verify_data(directory, ROOT / 'tokenizer/tokenizer.json', DATA_FILES)


def load_data(manifest, directory):
    settings = manifest['settings']
    if settings['mode'] == 'cpu-toy':
        rows = toy_data(settings['train_sequences'] // settings['batch_size'], settings['batch_size'],
                        held_out=settings['validation_sequences'])
        return rows, dict(
            source='toy rule', tensor_sha256={name: data_hash(tensor)
                                             for name, tensor in zip(('train', 'validation', 'test'), rows)},
        )
    identity = verified_data(directory)
    if identity != manifest['data']:
        raise ValueError('Data differs from the experiment manifest')
    counts = [settings[key] for key in ('train_sequences', 'validation_sequences', 'test_sequences')]
    sequence_len = settings['model']['sequence_len']
    rows = tuple(load_sequences(directory / name, count, sequence_len) for name, count in zip(DATA_FILES, counts))
    return rows, identity


def bind_manifest(output, manifest):
    path = output / 'manifest.json'
    output.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read_json(path) != manifest:
            raise ValueError(f'{path}: incompatible manifest')
        return
    with tempfile.NamedTemporaryFile(mode='w', dir=output, prefix='.manifest-', delete=False) as stream:
        pending = Path(stream.name)
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write('\n')
    try:
        os.link(pending, path)
    except FileExistsError:
        if read_json(path) != manifest:
            raise ValueError(f'{path}: incompatible manifest')
    finally:
        pending.unlink()


def prepare_gpu(output, directory):
    launch = read_json(output / 'launch.json')
    manifest = experiment_plan(False, launch['seeds'])
    manifest.update(source=source_identity(launch['commit']), data=verified_data(directory),
                    runtime_requirements=dict(device='cuda:0', gpu='NVIDIA L40S', gpus=1,
                                              precision='bfloat16 autocast', torch='2.10.0', cuda='12.8'))
    bind_manifest(output, manifest)


def validated_manifest(output):
    manifest = read_json(output / 'manifest.json')
    settings = manifest['settings']
    toy = settings['mode'] == 'cpu-toy'
    plan = experiment_plan(toy, manifest['seeds'], settings['steps'] if toy else None,
                           settings['checkpoint_interval'])
    if any(manifest[key] != value for key, value in plan.items()):
        raise ValueError('Manifest case matrix or training configuration differs')
    if toy:
        commit = source_commit()
    else:
        launch = read_json(output / 'launch.json')
        if launch['seeds'] != manifest['seeds']:
            raise ValueError('Launch seeds differ from the experiment manifest')
        commit = launch['commit']
    if manifest['source'] != source_identity(commit):
        raise ValueError('Source differs from the experiment manifest')
    return manifest


def case_metadata(manifest, case, seed, runtime):
    return dict(campaign_sha256=fingerprint(manifest), case=case, settings=manifest['settings'],
                seed=seed, runtime=runtime)


def result_hashes(folder):
    return {name: sha256(folder / name) for name in RESULT_FILES}


def completed_case(folder, metadata, *, seal=False):
    if read_json(folder / 'config.json') != metadata:
        raise ValueError('case configuration differs from the experiment manifest')
    receipt = folder / 'complete.json'
    if not seal or receipt.exists():
        if read_json(receipt) != result_hashes(folder):
            raise ValueError('completed artifacts differ from their checksums')
    report = validate_result(folder, metadata['case'], metadata['settings'])
    if seal and not receipt.exists():
        write_json(receipt, result_hashes(folder))
    return report


def rewind_checkpoint(folder, metadata, training, validation):
    try:
        state = torch.load(folder / 'state.pt', map_location='cpu', weights_only=True, mmap=True)
    except (pickle.UnpicklingError, EOFError) as error:
        raise ValueError('unreadable checkpoint') from error
    case, settings = metadata['case'], metadata['settings']
    step = state['step']
    interval = settings['checkpoint_interval']
    if (type(step) is not int or not 1 <= step <= settings['steps'] or state['world'] != 1 or state['rank'] != 0
            or (step != settings['steps'] and step % interval)):
        raise ValueError('invalid saved checkpoint step or rank')
    expected = dict(model=settings['model'], recipe=case['requested_recipe'], seed=metadata['seed'], world=1,
                    suffix=True, batch_size=settings['batch_size'], data_sha256=data_hash(training, validation),
                    allocation=case['allocation'], multiplex=None)
    if state['extra']['fingerprint'] != expected:
        raise ValueError('checkpoint configuration or data differs')
    records = read_records(folder, step)
    check_records(records, case, settings, step)
    check_history(state['extra']['history'], records)
    atomic_write_text(folder / 'metrics.jsonl', ''.join(json.dumps(row) + '\n' for row in records))


def run_case(folder, metadata, rows, device):
    config_path = folder / 'config.json'
    if config_path.exists() and read_json(config_path) != metadata:
        raise ValueError('case configuration differs from the experiment manifest')
    if (folder / 'complete.json').exists() or (folder / 'result.json').exists():
        completed_case(folder, metadata, seal=True)
        print(f'Skipping verified {folder}', flush=True)
        return
    resume = (folder / 'state.pt').exists()
    if resume:
        if not config_path.is_file():
            raise ValueError('checkpoint has no case configuration')
        rewind_checkpoint(folder, metadata, *rows[:2])
    elif folder.exists() and any(folder.iterdir()):
        allowed = {'config.json', 'metrics.jsonl', 'best.pt', *TRAINER_TEMP_FILES}
        files = {path.name for path in folder.iterdir()}
        if files - allowed or (not config_path.exists() and files != {'config.json.tmp'}):
            raise ValueError('unrecognized files in a case without a checkpoint')
        archive = Path(tempfile.mkdtemp(prefix=f'.interrupted-{folder.name}-', dir=folder.parent))
        folder.rename(archive / 'case')
        print(f'Restarting {folder} before its first checkpoint (previous attempt: {archive})', flush=True)

    case, settings = metadata['case'], metadata['settings']
    print(f'Running {folder}', flush=True)
    train(GPTConfig(**settings['model']), Recipe(**case['requested_recipe']), *rows, folder,
          steps=settings['steps'], eval_interval=settings['evaluation_interval'], seed=metadata['seed'],
          batch_size=settings['batch_size'], device=device, checkpoint_interval=settings['checkpoint_interval'],
          resume=resume, allocation=case['allocation'], metadata=metadata, echo=lambda line: print(line, flush=True))
    completed_case(folder, metadata, seal=True)


def run_seed(output, manifest, entry, rows, runtime):
    seed = entry['seed']
    training, validation, testing = rows
    order = torch.randperm(len(training), generator=torch.Generator().manual_seed(seed + 1))
    rows = (training[order], validation, testing)
    directory = output / entry['path']
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f'seed {seed} is already running') from None
        runtime_path = directory / 'runtime.json'
        if runtime_path.exists():
            if read_json(runtime_path) != runtime:
                raise ValueError(f'seed {seed}: incompatible runtime')
        else:
            write_json(runtime_path, runtime)
        cases = {case['name']: case for case in manifest['cases']}
        for name in entry['order']:
            folder = output / entry['paths'][name]
            try:
                run_case(folder, case_metadata(manifest, cases[name], seed, runtime), rows,
                         torch.device(runtime['device']))
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
                raise ValueError(f'{folder}: {error}') from error


def run_campaign(args):
    if args.toy:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_float32_matmul_precision('high')
    runtime = actual_runtime(args.toy)
    if args.toy:
        manifest = experiment_plan(True, args.seeds, args.steps, args.checkpoint_interval)
        rows, data = load_data(manifest, None)
        manifest.update(source=source_identity(source_commit()), data=data, runtime_requirements=runtime)
        bind_manifest(args.output, manifest)
    else:
        manifest = validated_manifest(args.output)
        if manifest['settings']['mode'] != 'l40s':
            raise ValueError('Use --toy to run this CPU experiment')
        rows, _ = load_data(manifest, args.data_dir)

    entries = [entry for entry in manifest['runs'] if args.seed is None or entry['seed'] == args.seed]
    if not entries:
        raise ValueError(f'seed {args.seed} is absent from the experiment manifest')
    for entry in entries:
        run_seed(args.output, manifest, entry, rows, runtime)


def write_summary(output):
    manifest = validated_manifest(output)
    seeds = []
    for entry in manifest['runs']:
        runtime = read_json(output / entry['path'] / 'runtime.json')
        if seeds and runtime != seeds[0]['runtime']:
            raise ValueError('Seed runtimes differ')
        seed = entry['seed']
        reports = {}
        for case in manifest['cases']:
            folder = output / entry['paths'][case['name']]
            try:
                reports[case['name']] = completed_case(folder, case_metadata(manifest, case, seed, runtime))
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise ValueError(f'{folder}: {error}') from error
        seeds.append(dict(seed=seed, runtime=runtime, order=entry['order'], cases=reports))
    destination = output / manifest['summary_path']
    write_json(destination, summarize_campaign(manifest, seeds))
    print(f'Wrote {destination}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    add_experiment_options(commands.add_parser('plan', help='Print the case matrix without running it.'))
    run = commands.add_parser('run', help='Run or resume cases for the selected seeds.')
    add_experiment_options(run)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--seed', type=int, help='Run one seed from the manifest.')
    run.add_argument('--data-dir', type=Path)
    prepare = commands.add_parser('prepare', help='Verify data and prepare the manifest in the Slurm setup job.')
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--data-dir', type=Path, required=True)
    summary = commands.add_parser('summary', help='Validate and summarize every case and seed.')
    summary.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'plan':
            print(json.dumps(experiment_plan(args.toy, args.seeds, args.steps, args.checkpoint_interval), indent=2))
        elif args.command == 'prepare':
            prepare_gpu(args.output, args.data_dir)
        elif args.command == 'run':
            if not args.toy and (args.seed is None or args.steps is not None or args.checkpoint_interval is not None
                                 or args.seeds != DEFAULT_SEEDS):
                raise ValueError('GPU runs require --seed and take campaign settings from the Slurm manifest')
            run_campaign(args)
        else:
            write_summary(args.output)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
