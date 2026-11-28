"""Profile DUST estimates and check their gradient quality on a frozen model."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean, stdev
import sys

if __name__ == '__main__':
    if '--toy' in sys.argv[1:]:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if '--dry-run' in sys.argv[1:]:
        sys.dont_write_bytecode = True

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ab_experiment import actual_runtime, source_commit, validate_seeds, write_json  # noqa: E402
from dust import (  # noqa: E402
    DUST, GPTConfig, Recipe, atomic_write_text, data_hash, load_sequences, make_model, recipe,
)
from accounting import RunMeters, analytic_cost, measure  # noqa: E402
from gradient_check import compare, estimate_gradients, reference_gradients  # noqa: E402
from profiling import profile_estimate  # noqa: E402
from roofline import L40S, mfu  # noqa: E402
from toy import TOY_BATCH, TOY_CONFIG, toy_recipe, toy_sequences  # noqa: E402
from verify_data import sha256, verify_data  # noqa: E402

TRAIN_FILE = 'train_1M.pt'
POPULATIONS = (256, 1024, 4096, 16384)
SOURCE_FILES = (
    'dust.py', 'model.py', 'requirements.txt',
    *(f'scripts/{name}.py' for name in
      ('ab_experiment', 'ab_results', 'check_baseline_results', 'gpu_diagnostics', 'verify_data')),
    *(f'new_implementation/{name}.py' for name in
      ('accounting', 'allocation', 'checkpoint', 'gradient_check', 'groups', 'multiplexing',
       'profiling', 'roofline', 'run_metrics', 'toy', 'tracking')),
)
SDPA_BACKENDS = {
    'aten::_scaled_dot_product_attention_math': 'math',
    'aten::_scaled_dot_product_cudnn_attention': 'cudnn',
    'aten::_scaled_dot_product_efficient_attention': 'efficient',
    'aten::_scaled_dot_product_flash_attention': 'flash',
    'aten::_scaled_dot_product_flash_attention_for_cpu': 'cpu_flash',
}


def parse_args(argv, *, launcher=False):
    description = 'Submit a frozen-model diagnostic on one Insomnia L40S.' if launcher else __doc__
    parser = argparse.ArgumentParser(description=description, allow_abbrev=False)
    commands = parser.add_subparsers(dest='command', required=True)
    profile = commands.add_parser('profile', allow_abbrev=False)
    quality = commands.add_parser('quality', allow_abbrev=False)
    for command in (profile, quality):
        command.add_argument('--output', type=Path, required=True)
        if launcher:
            command.set_defaults(toy=False)
        else:
            command.add_argument('--toy', action='store_true', help='Run on CPU with synthetic tokens.')
        command.add_argument('--seed', type=int, default=42, help='Model seed and base profile RNG seed.')
        command.add_argument('--data-dir', type=Path, help='Prepared data directory (default: data).')
        command.add_argument('--checkpoint', type=Path, help='Load a model state dict, such as best.pt.')
        command.add_argument('--dry-run', action='store_true', help='Print the plan without creating files.')
    profile.add_argument('--warmup', type=int, default=3)
    profile.add_argument('--repeats', type=int, default=5)
    profile.add_argument('--active', type=int, default=1)
    quality.add_argument('--populations', type=int, nargs='+', choices=POPULATIONS)
    quality.add_argument('--scales', type=int, nargs='+', help='Positive toy recipe scales (default: 1 2).')
    quality.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    quality.add_argument('--reference-chunk', type=int, default=1)
    args = parser.parse_args(argv)
    try:
        validate_seeds([args.seed])
        if args.toy and args.data_dir is not None:
            raise ValueError('--data-dir is only for GPU diagnostics')
        if args.command == 'profile':
            if args.warmup < 0 or args.repeats < 1 or args.active < 1:
                raise ValueError('--warmup must be non-negative and --repeats and --active must be positive')
        else:
            if args.toy and args.populations is not None:
                raise ValueError('--populations is only for GPU diagnostics. Use --scales with --toy.')
            if not args.toy and args.scales is not None:
                raise ValueError('--scales requires --toy')
            args.populations = args.populations or list(POPULATIONS)
            args.scales = args.scales or [1, 2]
            if len(set(args.populations)) != len(args.populations):
                raise ValueError('--populations must be unique')
            if min(args.scales) < 1 or len(set(args.scales)) != len(args.scales):
                raise ValueError('--scales must be positive and unique')
            if len(args.seeds) < 2:
                raise ValueError('--seeds needs at least two perturbation seeds')
            try:
                validate_seeds(args.seeds)
            except ValueError as error:
                raise ValueError(f'--seeds: {error}') from error
            if not 1 <= args.reference_chunk <= 8:
                raise ValueError('--reference-chunk must be between 1 and 8')
    except ValueError as error:
        parser.error(str(error))
    return args


def diagnostic_plan(args):
    config, batch = (TOY_CONFIG, TOY_BATCH) if args.toy else (GPTConfig(), 8)
    label = 'scale' if args.toy else 'population'
    if args.command == 'profile':
        sizes = [1] if args.toy else [256]
        options = dict(warmup=args.warmup, repeats=args.repeats, active=args.active)
        rng = dict(initial_seed=args.seed + 3, reseed_before_estimate=False)
    else:
        sizes = args.scales if args.toy else args.populations
        options = {f'{label}s': sizes, 'seeds': args.seeds, 'reference_chunk': args.reference_chunk}
        rng = dict(seed_argument='--seeds', reseed_before_estimate=True)
    workloads = []
    for size in sizes:
        settings = toy_recipe(scale=size) if args.toy else recipe('1M', size)
        workloads.append({label: size, 'recipe': asdict(settings),
                          'analytic_cost': analytic_cost(settings, config, batch=batch).as_dict()})
    execution = dict(suffix=True, world=1, multiplex=None, precision='fp32' if args.toy else 'bf16',
                     float32_matmul_precision='high', perturbation_rng=rng)
    if args.command == 'quality':
        execution['reference'] = dict(precision='fp32', float32_matmul_precision='highest',
                                      chunk=args.reference_chunk)
    return dict(
        schema_version=1, command=args.command, mode='cpu-toy' if args.toy else 'l40s',
        config=dict(**asdict(config), batch_size=batch), seed=args.seed, options=options,
        data_dir=None if args.toy else str((args.data_dir or Path('data')).resolve()),
        checkpoint=str(args.checkpoint.resolve()) if args.checkpoint else None,
        execution=execution, recipes=workloads,
    )


def source_identity():
    commit = source_commit()
    launch = ROOT.parent / 'launch.json'
    staged = commit is None and ROOT.name == 'source' and launch.is_file()
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


def sdpa_evidence(averages, events):
    names = ('scaled_dot_product', '_efficient_attention', '_flash_attention', '_cudnn_attention')
    operators = {
        event.key: event.count for event in averages
        if event.key.startswith('aten::') and any(name in event.key for name in names)
    }
    backends = sorted({SDPA_BACKENDS[key] for key in operators.keys() & SDPA_BACKENDS.keys()})
    kernel_terms = ('attention', 'fmha')
    kernels = Counter(event.key for event in events if event.device_type == torch.autograd.DeviceType.CUDA
                      and any(term in event.key.lower() for term in kernel_terms))
    return dict(operators=operators, observed_backends=backends or ['unknown'],
                trace_kernel_name_matches=dict(kernels), trace_kernel_name_filter=kernel_terms)


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
        timing.pop('wall_seconds')
        peak_memory = timing.pop('peak_memory_gib')
        timing.update(host_ms=meters.step_host_ms[args.warmup:],
                      cuda_event_ms=meters.step_gpu_ms[args.warmup:],
                      peak_memory_gib_including_warmup=peak_memory if is_cuda else None,
                      scope='Estimate only. Host time ends before the final timing synchronization. '
                            'CUDA event elapsed time can include idle gaps.')
        profile['ordinary_timing'] = timing
        cost = workload['analytic_cost']
        gpu_seconds = timing.get('gpu_median_ms', 0) / 1000
        if gpu_seconds:
            mfu_reason = None
        else:
            mfu_reason = 'No positive CUDA event median' if is_cuda else 'CUDA timing unavailable on CPU'
        profile['mfu'] = dict(
            needed=mfu(cost['flops_total'], gpu_seconds) if gpu_seconds else None,
            executed=mfu(cost['flops_total_executed'], gpu_seconds) if gpu_seconds else None,
            reason=mfu_reason,
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

        write_json(args.output / 'result.json', result)
        table_path = args.output / 'profile.txt'
        trace_path = args.output / 'trace.json'
        pending = trace_path.with_suffix('.json.tmp')
        table, prof = profile_estimate(
            estimator, x, y, warmup=0, active=args.active, trace_path=pending,
            sort_by='self_device_time_total' if is_cuda else 'self_cpu_time_total',
            record_shapes=False, profile_memory=False,
        )
        pending.replace(trace_path)
        result['artifacts']['trace'] = trace_path.name
        atomic_write_text(table_path, table)
        result['artifacts']['table'] = table_path.name
        averages = prof.key_averages()
        profile['profiler'] = dict(
            active=args.active, record_shapes=False, profile_memory=False,
            phases={
                event.key.removeprefix('dust/'): dict(
                    count=event.count, cpu_total_us=event.cpu_time_total,
                    device_total_us=event.device_time_total if is_cuda else None,
                )
                for event in averages if event.key.startswith('dust/')
            },
            scope='Inclusive totals over instrumented estimates. Profiling adds overhead.',
            sdpa=sdpa_evidence(averages, prof.events() if is_cuda else ()),
        )
    finally:
        estimator.close()


def summarize_quality(samples):
    def summary(values):
        values = list(values)
        return dict(mean=mean(values), sample_sd=stdev(values))

    return dict(
        clean_loss=summary(sample['clean_loss'] for sample in samples),
        per_group={
            group: {
                metric: summary(sample['per_group'][group][metric] for sample in samples)
                for metric in metrics
            }
            for group, metrics in samples[0]['per_group'].items()
        },
    )


def error_details(error):
    return dict(type=type(error).__name__, message=str(error))


def run_population(args, model, x, y, reference, population):
    estimator = DUST(model, Recipe(**population['recipe']), seed=args.seed)
    try:
        for seed in args.seeds:
            meters = RunMeters(warmup=0)
            with measure(meters):
                estimate, loss = estimate_gradients(estimator, x, y, seed=seed)
            groups = compare(reference, estimate)['per_group']
            del estimate
            if not math.isfinite(loss) or any(not math.isfinite(value)
                                              for metrics in groups.values() for value in metrics.values()):
                raise FloatingPointError(f'Non-finite estimate metrics at perturbation seed {seed}')
            population['samples'].append(dict(
                seed=seed, clean_loss=loss, per_group=groups, host_ms=meters.step_host_ms[0],
                cuda_event_ms=meters.step_gpu_ms[0] if meters.step_gpu_ms else None,
            ))
        population.update(status='complete', summary=summarize_quality(population['samples']))
    finally:
        estimator.close()


def run_quality(args, model, x, y, plan, result):
    quality = result['quality'] = dict(
        reference=dict(plan['execution']['reference']),
        timing=dict(warmup=0, scope='Raw per-seed time includes estimate, normalization, and gradient clones. '
                                    'Excludes reference and comparison. Host time ends before final timing sync.'),
        populations=[],
    )
    matmul_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision(quality['reference']['float32_matmul_precision'])
        reference, loss = reference_gradients(model, x, y, precision='fp32', chunk=args.reference_chunk)
    finally:
        torch.set_float32_matmul_precision(matmul_precision)
    if not math.isfinite(loss):
        raise FloatingPointError('Non-finite FP32 reference loss')
    quality['reference']['loss'] = loss

    for workload in plan['recipes']:
        population = dict(**workload, status='failed', samples=[])
        quality['populations'].append(population)
        try:
            run_population(args, model, x, y, reference, population)
        except torch.cuda.OutOfMemoryError as error:
            population.update(status='unavailable', error=error_details(error))
            label = f'toy scale {workload["scale"]}' if args.toy else f'population {workload["population"]}'
            print(f'{label} unavailable: {error}', file=sys.stderr)
        if population['status'] == 'unavailable' and x.is_cuda:
            torch.cuda.empty_cache()
        write_json(args.output / 'result.json', result)
    return all(population['status'] == 'complete' for population in quality['populations'])


def run(args, plan, argv):
    manifest = dict(**plan, reproduction_argv=[sys.executable, str(Path(__file__).resolve()), *argv],
                    cwd=str(Path.cwd()))
    reserve_output(args.output, manifest)
    manifest_path = args.output / 'manifest.json'
    result_path = args.output / 'result.json'
    result = dict(schema_version=1, command=args.command, status='partial',
                  reason='Run is in progress or was interrupted.',
                  artifacts=dict(manifest=manifest_path.name, result=result_path.name))
    write_json(result_path, result)
    matmul_precision = torch.get_float32_matmul_precision()
    try:
        manifest['source'] = source_identity()
        manifest['runtime'] = actual_runtime(args.toy)
        manifest['runtime']['cuda_visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES')
        torch.set_float32_matmul_precision(plan['execution']['float32_matmul_precision'])
        config = TOY_CONFIG if args.toy else GPTConfig()
        rows, manifest['data'] = frozen_rows(args, plan)
        device = torch.device(manifest['runtime']['device'])
        model = make_model(config, device, args.seed).eval()
        state = manifest['model_state'] = dict(
            sha256=None, initialization='checkpoint' if plan['checkpoint'] else 'seeded',
            model_seed=args.seed, checkpoint=None,
            scope='Model state dict in sorted key order with dtype and shape.',
        )
        if plan['checkpoint']:
            path = Path(plan['checkpoint'])
            state['checkpoint'] = dict(path=str(path), sha256=sha256(path))
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        state['sha256'] = model_state_hash(model)
        write_json(manifest_path, manifest)
        x = rows[:, :-1].contiguous().to(device)
        y = rows[:, 1:].contiguous().to(device)
        complete = True
        if args.command == 'profile':
            run_profile(args, model, x, y, plan['recipes'][0], result)
        else:
            complete = run_quality(args, model, x, y, plan, result)
        result['model_state_sha256_after'] = model_state_hash(model)
        if result['model_state_sha256_after'] != state['sha256']:
            raise RuntimeError('Frozen model state changed')
        result.update(status='complete' if complete else 'partial',
                      reason=None if complete else 'At least one requested recipe was unavailable.')
    except (Exception, KeyboardInterrupt) as error:
        result.update(status='failed', reason='Diagnostic raised an exception.', error=error_details(error))
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        write_json(manifest_path, manifest)
    finally:
        torch.set_float32_matmul_precision(matmul_precision)
    write_json(result_path, result)
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
