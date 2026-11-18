"""GPU preflight: a few minutes on the real model that exercise every CUDA path the CPU tests cannot,
before any compute is spent on a training run.

    python new_implementation/gpu_preflight.py --population 256 --output runs/preflight
    python new_implementation/gpu_preflight.py --toy --device cpu --output /tmp/preflight   # plumbing check

Checks, each reported as PASS / WARN / FAIL (a failed check does not stop the others):

1. environment   GPU, memory, torch version, bf16 autocast; whether the data files are there.
2. equivalence   Cached-suffix vs full-replay estimates at a matched seed, under bf16 autocast. They
                 consume the same noise, so they should agree up to rounding (per-group cosine ~ 1).
                 Also reports the cosine between two different seeds: the noise floor that "different"
                 would look like.
3. timing        The estimate in both modes: median time, time per phase, peak memory, against the
                 analytic FLOP saving. This is the full-replay (upstream) vs cached-suffix comparison
                 in miniature, so its numbers say what the training runs will show.
4. probe         The gradient probe against backprop: it runs, its peak memory, per-group cosines, and
                 it leaves the weights and the perturbation generator untouched.
5. training      A 2-step dust.train with evaluation, probe and checkpoints; then the same run stopped
                 after step 1 and resumed. Checks the records carry every diagnostic and that the
                 resumed run ends at the same weights (the CUDA generator restore).
6. launch-bound  (--launch-bound) the double-batch test: is the step GPU-bound, i.e. will a FLOP
                 saving show up in wall-clock at all.

Writes OUTPUT/preflight.json. Exit code 1 if any check FAILs. Single GPU only: the multi-GPU paths
(per-rank checkpoints, all_reduce) are not covered.
"""
import argparse
import json
import platform
import shutil
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dust  # noqa: E402  (import after the sys.path shim above; it puts this folder on the path)
import gradient_check as gc  # noqa: E402
from accounting import _percentile, analytic_cost  # noqa: E402
from model import GPTConfig  # noqa: E402
from profiling import launch_bound_test  # noqa: E402
from run_metrics import gradient_probe  # noqa: E402

RECORD_KEYS = ('train_loss', 'flops', 'forward_equivalents', 'draws', 'population', 'direct_forwards',
               'train_seconds', 'gpu_hours', 'elapsed_seconds', 'step_host_ms', 'phase_seconds', 'probe')
CUDA_RECORD_KEYS = ('step_gpu_ms', 'peak_memory_gb')
PHASES = ('clean', 'direct', 'embedding', 'hubs', 'head', 'local', 'scalars')


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def peak_gb(device):
    return torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == 'cuda' else None


def reset_peak(device):
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)


def check_environment(device, paths):
    info = dict(python=platform.python_version(), torch=torch.__version__, device=str(device))
    status, notes = 'PASS', []
    if device.type == 'cuda':
        properties = torch.cuda.get_device_properties(device)
        info.update(gpu=properties.name, gpu_memory_gb=properties.total_memory / 1024 ** 3,
                    gpus_visible=torch.cuda.device_count(), cuda=torch.version.cuda,
                    bf16=torch.cuda.is_bf16_supported())
        if not info['bf16']:
            status = 'FAIL'
            notes.append('GPU does not support bfloat16')
    else:
        status = 'WARN'
        notes.append('not a CUDA device: this run checks plumbing only')
    if not torch.__version__.startswith('2.10'):
        status = 'WARN' if status == 'PASS' else status
        notes.append(f'requirements.txt pins torch 2.10.0, found {torch.__version__}')
    missing = [str(p) for p in paths if not p.is_file()]
    info['data_missing'] = missing
    if missing:
        status = 'WARN' if status == 'PASS' else status
        notes.append('data files missing (run prepare_data.py); preflight uses random tokens instead')
    return status, dict(info, notes=notes)


def estimate_once(model, settings, x, y, suffix, seed):
    estimator = dust.DUST(model, settings, seed=seed, suffix=suffix)
    try:
        gradients, valid, _ = estimator.estimate(x, y)
    finally:
        estimator.close()
    return {name: (g / valid).float() for name, g in gradients.items()}


def check_equivalence(model, settings, x, y):
    full = estimate_once(model, settings, x, y, suffix=False, seed=7)
    cached = estimate_once(model, settings, x, y, suffix=True, seed=7)
    other = estimate_once(model, settings, x, y, suffix=True, seed=8)
    same = gc.compare(full, cached)['per_group']
    floor = gc.compare(full, other)['per_group']
    worst = max(((full[n] - cached[n]).norm() / full[n].norm().clamp(min=1e-30)).item() for n in full)
    cosines = {g: m['cosine'] for g, m in same.items()}
    lowest = min(cosines.values())
    status = 'PASS' if lowest > 0.999 else ('WARN' if lowest > 0.98 else 'FAIL')
    return status, dict(
        cosine_suffix_vs_full=cosines, min_cosine=lowest, max_relative_difference=worst,
        cosine_between_seeds={g: m['cosine'] for g, m in floor.items()},
        note='suffix and full replay consume the same noise; under bf16 they differ by rounding only. '
             'cosine_between_seeds is what two unrelated estimates look like.',
    )


def check_timing(model, settings, config, x, y, repeats, warmup=1):
    device = x.device
    result = {}
    for mode, suffix in (('full_replay', False), ('cached_suffix', True)):
        estimator = dust.DUST(model, settings, seed=7, suffix=suffix)
        seconds, phases = [], []
        try:
            for index in range(warmup + repeats):
                if index == warmup:
                    reset_peak(device)
                estimator.phase_seconds = {}
                sync(device)
                start = time.monotonic()
                estimator.estimate(x, y)
                sync(device)
                seconds.append(time.monotonic() - start)
                phases.append(estimator.phase_seconds)
        finally:
            estimator.close()
        cost = analytic_cost(settings, config, batch=len(x), suffix=suffix)
        result[mode] = dict(
            estimate_seconds_median=_percentile(seconds[warmup:], 50), estimate_seconds=seconds[warmup:],
            phase_seconds_median={p: _percentile([s[p] for s in phases[warmup:]], 50) for p in PHASES},
            peak_memory_gb=peak_gb(device), forward_equivalents=cost.forward_equivalents,
            flops=cost.flops_total,
        )
    full, cached = result['full_replay'], result['cached_suffix']
    result['measured_time_saving'] = 1 - cached['estimate_seconds_median'] / full['estimate_seconds_median']
    result['predicted_flop_saving'] = 1 - cached['flops'] / full['flops']
    result['note'] = ('measured_time_saving below predicted_flop_saving means part of the step is not '
                      'GPU-bound (launch overhead, memory traffic); see --launch-bound.')
    return 'PASS', result


def check_probe(model, settings, x, y):
    device = x.device
    estimator = dust.DUST(model, settings, seed=7)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    generator = estimator.generator.get_state().clone()
    try:
        reset_peak(device)
        sync(device)
        start = time.monotonic()
        probe = gradient_probe(estimator, x, y)
        sync(device)
        seconds = time.monotonic() - start
        restored = torch.equal(generator, estimator.generator.get_state())
    finally:
        estimator.close()
    untouched = all(torch.equal(before[name], value) and before[name].dtype == value.dtype
                    for name, value in model.state_dict().items())
    frozen = not any(p.requires_grad for p in model.parameters())
    cosines = {g: m['cosine'] for g, m in probe['per_group'].items()}
    finite = all(c == c for c in cosines.values())
    status = 'PASS' if restored and untouched and frozen and finite else 'FAIL'
    return status, dict(
        seconds=seconds, peak_memory_gb=peak_gb(device), mean_group_cosine=probe['mean_group_cosine'],
        cosine_to_backprop=cosines, rel_norm={g: m['rel_norm'] for g, m in probe['per_group'].items()},
        generator_restored=restored, weights_untouched=untouched, model_left_frozen=frozen,
        note='cosine_to_backprop at init on one batch; compare the ordering with the paper (head highest).',
    )


def check_training(config, settings, training, validation, testing, output, device, suffix):
    options = dict(steps=2, eval_interval=1, seed=42, device=device, suffix=suffix, batch_size=len(training) // 2,
                   checkpoint_interval=1, probe_interval=1, echo=lambda line: None)
    straight, resumed = output / 'train-straight', output / 'train-resumed'
    for folder in (straight, resumed):
        shutil.rmtree(folder, ignore_errors=True)
    result = dust.train(config, settings, training, validation, testing, straight, **options)
    dust.train(config, settings, training, validation, testing, resumed, **dict(options, steps=1))
    dust.train(config, settings, training, validation, testing, resumed, resume=True, **options)

    records = [json.loads(line) for line in (straight / 'metrics.jsonl').read_text().splitlines()]
    expected = RECORD_KEYS + (CUDA_RECORD_KEYS if device.type == 'cuda' else ())
    missing = sorted({key for record in records[1:] for key in expected if key not in record})
    phases_missing = sorted(set(PHASES) - set(records[-1].get('phase_seconds', {})))
    a = torch.load(straight / 'state.pt', map_location='cpu', weights_only=False)
    b = torch.load(resumed / 'state.pt', map_location='cpu', weights_only=False)
    difference = max(
        ((a['model'][n].float() - b['model'][n].float()).abs().max()
         / a['model'][n].float().abs().max().clamp(min=1e-30)).item() for n in a['model'])
    generators = torch.equal(a['generator'], b['generator'])
    files = [name for name in ('metrics.jsonl', 'best.pt', 'result.json', 'state.pt')
             if not (straight / name).is_file()]
    ok = not missing and not phases_missing and not files and generators and difference < 1e-2
    status = 'PASS' if ok and difference == 0 else ('WARN' if ok else 'FAIL')
    return status, dict(
        record_keys_missing=missing, phases_missing=phases_missing, files_missing=files,
        resume_max_relative_weight_difference=difference, resume_generator_identical=generators,
        best_val=result['best_val'], timing=result['timing'], last_record=records[-1],
        note='resume difference 0 = bit-exact. A tiny nonzero value on CUDA is kernel nondeterminism '
             '(index_add_ on the embeddings), not a lost generator state, as long as the generators match.',
    )


def check_launch_bound(model, settings, x, y, repeats=3, warmup=1):
    def run(data):
        estimator = dust.DUST(model, settings, seed=7)
        try:
            estimator.estimate(*data)
        finally:
            estimator.close()

    doubled = (torch.cat([x, x]), torch.cat([y, y]))
    return 'PASS', launch_bound_test(run, (x, y), doubled, repeats=repeats, warmup=warmup)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tokens', choices=dust.DEFAULT_CONFIGS, default='1M')
    parser.add_argument('--population', type=int, choices=dust.DRAW_CHUNK_SIZES, default=256)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('runs/preflight'))
    parser.add_argument('--device', help='cuda (default) or cpu.')
    parser.add_argument('--toy', action='store_true', help='Tiny model and recipe: checks this script on a CPU.')
    parser.add_argument('--repeats', type=int, default=3, help='Timed estimates per mode (after one warmup).')
    parser.add_argument('--launch-bound', action='store_true',
                        help='Also run the double-batch test (8 more estimates, half of them at twice the batch).')
    parser.add_argument('--skip', nargs='*', default=[], help='Checks to skip, by name.')
    args = parser.parse_args()

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda':
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')
    paths = [] if args.toy else [args.data_dir / f'train_{args.tokens}.pt', args.data_dir / 'val.pt',
                                 args.data_dir / 'test.pt']
    if args.toy:
        from toy import TOY_BATCH, TOY_CONFIG, toy_data, toy_recipe
        config, settings, batch = TOY_CONFIG, toy_recipe(), TOY_BATCH
        training, validation, testing = toy_data(2, batch)
    else:
        config, settings, batch = GPTConfig(), dust.recipe(args.tokens, args.population), 8
        if all(p.is_file() for p in paths):
            training = dust.load_sequences(paths[0], 2 * batch)
            validation, testing = dust.load_sequences(paths[1], 64), dust.load_sequences(paths[2], 64)
        else:
            generator = torch.Generator().manual_seed(0)
            training, validation, testing = (
                torch.randint(0, config.vocab_size, (count, config.sequence_len + 1), generator=generator)
                for count in (2 * batch, 32, 32))
    rows = training[:batch].to(device)
    x, y = rows[:, :-1].contiguous(), rows[:, 1:].contiguous()
    model = dust.make_model(config, device, seed=42)
    args.output.mkdir(parents=True, exist_ok=True)

    checks = [
        ('environment', lambda: check_environment(device, paths)),
        ('equivalence', lambda: check_equivalence(model, settings, x, y)),
        ('timing', lambda: check_timing(model, settings, config, x, y, args.repeats)),
        ('probe', lambda: check_probe(model, settings, x, y)),
        ('training', lambda: check_training(config, settings, training, validation, testing, args.output,
                                            device, suffix=True)),
    ]
    if args.launch_bound:
        checks.append(('launch-bound', lambda: check_launch_bound(model, settings, x, y)))

    report = dict(population=sum(settings.writers + settings.hidden + settings.hubs) + settings.embedding,
                  recipe=asdict(settings), model=asdict(config), checks={})
    for name, check in checks:
        if name in args.skip:
            continue
        print(f'[{name}] running...', flush=True)
        start = time.monotonic()
        try:
            status, detail = check()
        except Exception as error:  # a failed check must not hide the ones after it
            status, detail = 'FAIL', dict(error=f'{type(error).__name__}: {error}',
                                          traceback=traceback.format_exc())
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        report['checks'][name] = dict(status=status, check_seconds=time.monotonic() - start, **detail)
        print(f'[{name}] {status} ({time.monotonic() - start:.1f}s)', flush=True)
        print(json.dumps({k: v for k, v in detail.items() if k not in ('last_record', 'timing')},
                         indent=2, default=str), flush=True)
        (args.output / 'preflight.json').write_text(json.dumps(report, indent=2, default=str) + '\n')

    print('\nSummary')
    for name, check in report['checks'].items():
        print(f"  {check['status']:4}  {name}")
    timing = report['checks'].get('timing', {})
    if timing.get('status') == 'PASS':
        full, cached = timing['full_replay'], timing['cached_suffix']
        print(f"\n  estimate: full replay {full['estimate_seconds_median']:.2f}s, cached suffix "
              f"{cached['estimate_seconds_median']:.2f}s -> {100 * timing['measured_time_saving']:.1f}% faster "
              f"(FLOP model predicts {100 * timing['predicted_flop_saving']:.1f}%)")
    print(f"  report: {args.output / 'preflight.json'}")
    sys.exit(1 if any(c['status'] == 'FAIL' for c in report['checks'].values()) else 0)


if __name__ == '__main__':
    main()
