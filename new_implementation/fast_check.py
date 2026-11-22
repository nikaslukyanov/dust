"""GPU check of the fast path (model.set_fast): same estimate as the plain path, and how much faster.

    python new_implementation/fast_check.py --tokens 1M --population 256 --output runs/fast-check

1. equivalence  One DUST estimate per path at a matched seed under bf16 autocast: per-group cosine and
                relative difference (they consume the same noise, so only rounding separates them),
                plus the clean-loss difference. Also the cosine between two seeds, the noise floor.
2. timing       Median estimate time and per-phase seconds per path, after warm-up (the fast path's
                first calls compile; those are reported separately as compile_seconds), and peak memory.

Writes OUTPUT/fast_check.json. Exit code 1 if the lowest per-group cosine is below --min-cosine (0.9).
The attention internals (q, k) read the smallest loss changes through a different attention kernel,
so they sit lowest (L40S: k 0.943, q 0.967, every other group >= 0.994; two seeds give ~0.00); the
training-loss comparison, not this cosine, is the real test.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dust  # noqa: E402  (import after the sys.path shim above; it puts this folder on the path)
import gradient_check as gc  # noqa: E402
import model as model_module  # noqa: E402
from model import GPTConfig  # noqa: E402


def sync():
    torch.cuda.synchronize()


def estimate(model, settings, x, y, seed):
    estimator = dust.DUST(model, settings, seed=seed)
    try:
        gradients, valid, loss = estimator.estimate(x, y)
    finally:
        estimator.close()
    return {n: (g / valid).float() for n, g in gradients.items()}, float(loss)


def timing(model, settings, x, y, warmup, repeats):
    estimator = dust.DUST(model, settings, seed=7)
    seconds, phases = [], []
    try:
        start = time.monotonic()
        for _ in range(warmup):
            estimator.estimate(x, y)
        sync()
        warm = time.monotonic() - start
        torch.cuda.reset_peak_memory_stats()
        for _ in range(repeats):
            estimator.phase_seconds = {}
            sync()
            start = time.monotonic()
            estimator.estimate(x, y)
            sync()
            seconds.append(time.monotonic() - start)
            phases.append(estimator.phase_seconds)
    finally:
        estimator.close()
    names = phases[0].keys()
    return dict(median_seconds=statistics.median(seconds), seconds=seconds, warmup_seconds=warm,
                phase_seconds={n: statistics.median(p[n] for p in phases) for n in names},
                peak_memory_gib=torch.cuda.max_memory_allocated() / 1024 ** 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tokens', choices=dust.DEFAULT_CONFIGS, default='1M')
    parser.add_argument('--population', type=int, choices=dust.DRAW_CHUNK_SIZES, default=256)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('runs/fast-check'))
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--min-cosine', type=float, default=0.9)
    args = parser.parse_args()

    device = torch.device('cuda', torch.cuda.current_device())
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')
    config, settings = GPTConfig(), dust.recipe(args.tokens, args.population)
    rows = dust.load_sequences(args.data_dir / f'train_{args.tokens}.pt', 8).to(device)
    x, y = rows[:, :-1].contiguous(), rows[:, 1:].contiguous()
    model = dust.make_model(config, device, seed=42)
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(gpu=torch.cuda.get_device_name(device), torch=torch.__version__, population=args.population)

    model_module.set_fast(False)
    plain, plain_loss = estimate(model, settings, x, y, seed=7)
    other, _ = estimate(model, settings, x, y, seed=8)
    model_module.set_fast(True)
    start = time.monotonic()
    fused, fused_loss = estimate(model, settings, x, y, seed=7)
    sync()
    report['compile_seconds'] = time.monotonic() - start  # first fast estimate: includes compilation
    same = gc.compare(plain, fused)['per_group']
    floor = gc.compare(plain, other)['per_group']
    cosines = {g: m['cosine'] for g, m in same.items()}
    report['equivalence'] = dict(
        cosine_fast_vs_plain=cosines, min_cosine=min(cosines.values()),
        max_relative_difference=max(((plain[n] - fused[n]).norm() / plain[n].norm().clamp(min=1e-30)).item()
                                    for n in plain),
        cosine_between_seeds={g: m['cosine'] for g, m in floor.items()},
        clean_loss_plain=plain_loss, clean_loss_fast=fused_loss)
    print(json.dumps(report, indent=2), flush=True)

    model_module.set_fast(False)
    report['timing_plain'] = timing(model, settings, x, y, warmup=1, repeats=args.repeats)
    model_module.set_fast(True)
    report['timing_fast'] = timing(model, settings, x, y, warmup=2, repeats=args.repeats)
    report['speedup'] = report['timing_plain']['median_seconds'] / report['timing_fast']['median_seconds']
    (args.output / 'fast_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('timing_plain', 'timing_fast', 'speedup')}, indent=2), flush=True)
    if report['equivalence']['min_cosine'] < args.min_cosine:
        print('FAIL: fast path disagrees with the plain path', flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
