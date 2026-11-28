"""Profile DUST estimates and check their gradient quality on a frozen model."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

if __name__ == '__main__' and '--toy' in sys.argv[1:]:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ab_experiment import actual_runtime, source_commit, validate_seeds, write_json  # noqa: E402
from dust import (  # noqa: E402
    DUST, GPTConfig, Recipe, atomic_write_text, data_hash, load_sequences, make_model, recipe,
)
from accounting import RunMeters, analytic_cost, measure  # noqa: E402
from profiling import profile_estimate  # noqa: E402
from roofline import L40S, mfu  # noqa: E402
from toy import TOY_BATCH, TOY_CONFIG, toy_recipe, toy_sequences  # noqa: E402
from verify_data import sha256, verify_data  # noqa: E402

TRAIN_FILE = 'train_1M.pt'
SOURCE_FILES = (
    'dust.py', 'model.py', 'requirements.txt', 'scripts/ab_experiment.py',
    'scripts/gpu_diagnostics.py', 'scripts/verify_data.py',
    *(f'new_implementation/{name}.py' for name in
      ('accounting', 'groups', 'multiplexing', 'profiling', 'roofline', 'toy')),
)
SDPA_BACKENDS = {
    'aten::_scaled_dot_product_attention_math': 'math',
    'aten::_scaled_dot_product_cudnn_attention': 'cudnn',
    'aten::_scaled_dot_product_efficient_attention': 'efficient',
    'aten::_scaled_dot_product_flash_attention': 'flash',
    'aten::_scaled_dot_product_flash_attention_for_cpu': 'cpu_flash',
}


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest='command', required=True)
    profile = commands.add_parser('profile', allow_abbrev=False)
    profile.add_argument('--output', type=Path, required=True)
    profile.add_argument('--toy', action='store_true', help='Run on CPU with synthetic tokens.')
    profile.add_argument('--seed', type=int, default=42, help='Model seed and base profile RNG seed.')
    profile.add_argument('--data-dir', type=Path, help='Prepared data directory (default: data).')
    profile.add_argument('--checkpoint', type=Path, help='Load a model state dict, such as best.pt.')
    profile.add_argument('--dry-run', action='store_true', help='Print the plan without creating files.')
    profile.add_argument('--warmup', type=int, default=3)
    profile.add_argument('--repeats', type=int, default=5)
    profile.add_argument('--active', type=int, default=1)
    args = parser.parse_args(argv)
    try:
        validate_seeds([args.seed])
        if args.toy and args.data_dir is not None:
            raise ValueError('--data-dir is only for GPU diagnostics')
        if args.warmup < 0 or args.repeats < 1 or args.active < 1:
            raise ValueError('--warmup must be non-negative and --repeats and --active must be positive')
    except ValueError as error:
        parser.error(str(error))
    return args


def diagnostic_plan(args):
    config, batch = (TOY_CONFIG, TOY_BATCH) if args.toy else (GPTConfig(), 8)
    settings = toy_recipe() if args.toy else recipe('1M', 256)
    workload = dict(scale=1) if args.toy else dict(population=256)
    return dict(
        schema_version=1, command=args.command, mode='cpu-toy' if args.toy else 'l40s',
        config=dict(**asdict(config), batch_size=batch), seed=args.seed,
        options=dict(warmup=args.warmup, repeats=args.repeats, active=args.active),
        data_dir=None if args.toy else str((args.data_dir or Path('data')).resolve()),
        checkpoint=str(args.checkpoint.resolve()) if args.checkpoint else None,
        execution=dict(suffix=True, world=1, multiplex=None, precision='fp32' if args.toy else 'bf16',
                       float32_matmul_precision='high'),
        recipes=[dict(**workload, recipe=asdict(settings),
                      analytic_cost=analytic_cost(settings, config, batch=batch).as_dict())],
    )


def source_identity():
    commit = source_commit()
    launch = ROOT.parent / 'launch.json'
    staged = commit is None and launch.is_file()
    if staged:
        commit = json.loads(launch.read_text())['commit']
    return dict(commit=commit, files={name: sha256(ROOT / name) for name in SOURCE_FILES},
                launch_sha256=sha256(launch) if staged else None)


def frozen_rows(args, plan):
    batch = plan['config']['batch_size']
    if args.toy:
        rows = toy_sequences(batch, args.seed)
        identity = dict(source='toy rule', seed=args.seed)
    else:
        directory = Path(plan['data_dir'])
        identity = verify_data(directory, ROOT / 'tokenizer/tokenizer.json', (TRAIN_FILE,))
        rows = load_sequences(directory / TRAIN_FILE, batch, plan['config']['sequence_len'])
        identity.update(source=TRAIN_FILE, first_row=0, row_count=batch)
    return rows, dict(**identity, tensor_sha256=data_hash(rows), shape=list(rows.shape), dtype=str(rows.dtype))


def model_state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy())
    return digest.hexdigest()


def reserve_output(output, manifest):
    output.mkdir(parents=True, exist_ok=True)
    claim = output / '.starting'
    try:
        claim.touch(exist_ok=False)
    except FileExistsError:
        raise ValueError(f'{output}: output directory is nonempty') from None
    try:
        if any(path != claim for path in output.iterdir()):
            raise ValueError(f'{output}: output directory is nonempty')
        write_json(output / 'manifest.json', manifest)
    finally:
        claim.unlink()


def sdpa_evidence(averages, trace_path):
    names = ('scaled_dot_product', '_efficient_attention', '_flash_attention', '_cudnn_attention')
    operators = {
        event.key: event.count for event in averages
        if event.key.startswith('aten::') and any(name in event.key for name in names)
    }
    backends = sorted({SDPA_BACKENDS[key] for key in operators.keys() & SDPA_BACKENDS.keys()})
    events = json.loads(trace_path.read_text(encoding='utf-8'))['traceEvents']
    kernels = Counter(event['name'] for event in events
                      if event.get('cat') == 'kernel'
                      and any(term in event['name'].lower() for term in ('attention', 'fmha')))
    return dict(operators=operators, observed_backends=backends or ['unknown'],
                trace_kernel_name_matches=dict(kernels), trace_kernel_name_filter=['attention', 'fmha'])


def run_profile(args, model, x, y, workload, result):
    is_cuda = x.is_cuda
    profile = result['profile'] = {}
    estimator = DUST(model, Recipe(**workload['recipe']), seed=args.seed)
    try:
        meters = RunMeters(warmup=args.warmup)
        for _ in range(args.warmup + args.repeats):
            with measure(meters):
                estimator.estimate(x, y)
        timing = meters.summary()
        timing.pop('likely_launch_bound')
        timing.pop('gpu_hours')
        timing.update(host_ms=meters.step_host_ms[args.warmup:],
                      cuda_event_ms=meters.step_gpu_ms[args.warmup:],
                      peak_memory_gib=timing['peak_memory_gib'] if is_cuda else None,
                      scope='Estimate only. Host time ends before the final timing synchronization. '
                            'CUDA event elapsed time can include idle gaps.')
        profile['ordinary_timing'] = timing
        cost = workload['analytic_cost']
        gpu_seconds = timing.get('gpu_median_ms', 0) / 1000
        profile['mfu'] = dict(
            needed=mfu(cost['flops_total'], gpu_seconds) if gpu_seconds else None,
            executed=mfu(cost['flops_total_executed'], gpu_seconds) if gpu_seconds else None,
            reason=None if gpu_seconds else 'No positive CUDA event median',
            dense_bf16_peak_flops_per_second=L40S.bf16_flops,
            scope='Analytic FLOPs divided by the ordinary CUDA event median and L40S dense BF16 peak. '
                  'Excludes apply, optimizer, and data. Hardware utilization was not measured.',
        )
        estimator.phase_seconds = {}
        try:
            estimator.estimate(x, y)
            profile['synchronized_phase_timing'] = dict(
                seconds=estimator.phase_seconds,
                scope='One separate estimate. CUDA synchronization at phase boundaries changes execution.',
            )
        finally:
            estimator.phase_seconds = None

        table_path = args.output / 'profile.txt'
        trace_path = args.output / 'trace.json'
        pending = trace_path.with_suffix('.json.tmp')
        table, prof = profile_estimate(
            estimator, x, y, warmup=0, active=args.active, trace_path=pending,
            sort_by='self_device_time_total' if is_cuda else 'self_cpu_time_total',
        )
        pending.replace(trace_path)
        result['artifacts']['trace'] = trace_path.name
        atomic_write_text(table_path, table)
        result['artifacts']['table'] = table_path.name
        averages = prof.key_averages()
        profile['profiler'] = dict(
            active=args.active,
            phases={
                event.key.removeprefix('dust/'): dict(
                    count=event.count, cpu_total_us=event.cpu_time_total,
                    device_total_us=event.device_time_total if is_cuda else None,
                )
                for event in averages if event.key.startswith('dust/')
            },
            scope='Inclusive totals over instrumented estimates. Profiling adds overhead.',
            sdpa=sdpa_evidence(averages, trace_path),
        )
    finally:
        estimator.close()


def run(args, plan, argv):
    manifest = dict(**plan, reproduction_argv=[sys.executable, str(Path(__file__).resolve()), *argv],
                    cwd=str(Path.cwd()))
    reserve_output(args.output, manifest)
    manifest_path = args.output / 'manifest.json'
    result = dict(schema_version=1, command=args.command, artifacts={'manifest': manifest_path.name})
    try:
        manifest['source'] = source_identity()
        manifest['runtime'] = actual_runtime(args.toy)
        torch.set_float32_matmul_precision(plan['execution']['float32_matmul_precision'])
        config = TOY_CONFIG if args.toy else GPTConfig()
        rows, manifest['data'] = frozen_rows(args, plan)
        device = torch.device(manifest['runtime']['device'])
        model = make_model(config, device, args.seed).eval()
        checkpoint = None
        if plan['checkpoint']:
            path = Path(plan['checkpoint'])
            checkpoint = dict(path=str(path), sha256=sha256(path))
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        state_hash = model_state_hash(model)
        manifest['model_state'] = dict(
            sha256=state_hash, initialization='checkpoint' if checkpoint else 'seeded',
            model_seed=args.seed, checkpoint=checkpoint,
        )
        write_json(manifest_path, manifest)
        x = rows[:, :-1].contiguous().to(device)
        y = rows[:, 1:].contiguous().to(device)
        run_profile(args, model, x, y, plan['recipes'][0], result)
        result['model_state_sha256_after'] = model_state_hash(model)
        if result['model_state_sha256_after'] != state_hash:
            raise RuntimeError('Frozen model state changed')
        result['status'] = 'complete'
    except (Exception, KeyboardInterrupt) as error:
        result.update(status='failed', error=dict(type=type(error).__name__, message=str(error)))
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        write_json(manifest_path, manifest)
    write_json(args.output / 'result.json', result)
    return 0 if result['status'] == 'complete' else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    plan = diagnostic_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    try:
        return run(args, plan, argv)
    except (OSError, ValueError) as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
