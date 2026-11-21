"""Profile one DUST update on the real model: which SDPA kernel runs, per-phase and per-op GPU time, a trace.

    python new_implementation/profile_step.py --device cuda --tokens 1M --population 256 --output runs/profile
    python new_implementation/profile_step.py --toy --device cpu --output /tmp/profile   # plumbing check

Runs ``--warmup`` estimates (discarded), then profiles ``--active`` estimates with torch.profiler
(profiling.profile_estimate) in cached-suffix mode, and in full-replay mode too with ``--full-replay``.
Writes, per mode, to OUTPUT/<mode>/:

- ``table_cuda.txt`` / ``table_cpu.txt``: per-op tables sorted by self GPU time and by self CPU time.
- ``trace.json``: the Perfetto / chrome trace (open at ui.perfetto.dev); dust/<phase> labels the phases.
- ``summary.json``: the SDPA backends seen (the aten op name says which kernel ran: flash, efficient,
  cudnn, or the unfused math path) with call counts and GPU time; the dust/<phase> rows; the top GPU ops;
  and the profiled estimate's wall time and the CPU-vs-GPU total, the launch-bound tell.

The profiler adds overhead, so take step times from training runs (metrics.jsonl), and shares and the
kernel identity from here.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dust  # noqa: E402  (import after the sys.path shim above; it puts this folder on the path)
from model import GPTConfig  # noqa: E402
from profiling import profile_estimate  # noqa: E402

SDPA_OPS = {
    'aten::_scaled_dot_product_flash_attention': 'flash',
    'aten::_scaled_dot_product_efficient_attention': 'efficient',
    'aten::_scaled_dot_product_cudnn_attention': 'cudnn',
    'aten::_scaled_dot_product_flash_attention_for_cpu': 'flash (cpu)',
    'aten::_scaled_dot_product_attention_math': 'math',
}
PHASES = ('clean', 'direct', 'embedding', 'hubs', 'head', 'local', 'scalars')


def _us(event, name):
    """Event time in microseconds; torch renamed cuda_* to device_* around 2.4, so accept both."""
    for attribute in (name.replace('cuda', 'device'), name):
        value = getattr(event, attribute, None)
        if value is not None:
            return float(value)
    return 0.0


def summarize(prof, active, wall_seconds):
    events = prof.key_averages()
    sdpa, phases, ops = {}, {}, []
    for event in events:
        row = dict(count=event.count, gpu_ms=_us(event, 'cuda_time_total') / 1e3,
                   self_gpu_ms=_us(event, 'self_cuda_time_total') / 1e3,
                   cpu_ms=_us(event, 'cpu_time_total') / 1e3)
        if event.key in SDPA_OPS:
            sdpa[SDPA_OPS[event.key]] = dict(op=event.key, **row)
        elif event.key.startswith('dust/'):
            phases[event.key[5:]] = row
        ops.append(dict(name=event.key, **row))
    ops.sort(key=lambda r: r['self_gpu_ms'], reverse=True)
    self_gpu = sum(r['self_gpu_ms'] for r in ops)
    self_cpu = sum(_us(e, 'self_cpu_time_total') for e in events) / 1e3
    backends = sorted(sdpa)
    return dict(
        sdpa_backends=backends,
        sdpa_verdict=(backends[0] if len(backends) == 1 else ('none seen' if not backends else 'mixed: ' + ', '.join(backends))),
        sdpa=sdpa,
        phases={p: phases[p] for p in PHASES if p in phases},
        top_gpu_ops=ops[:25],
        estimates_profiled=active,
        wall_seconds_per_estimate=wall_seconds / active,
        self_gpu_ms_total=self_gpu,
        self_cpu_ms_total=self_cpu,
        note='Profiler overhead inflates wall time; use it for shares and kernel identity only. '
             'self_cpu well above self_gpu, or GPU gaps in the trace, point to launch overhead.',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tokens', choices=dust.DEFAULT_CONFIGS, default='1M')
    parser.add_argument('--population', type=int, choices=dust.DRAW_CHUNK_SIZES, default=256)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('runs/profile'))
    parser.add_argument('--device', help='cuda (default) or cpu.')
    parser.add_argument('--toy', action='store_true', help='Tiny model and recipe: checks this script on a CPU.')
    parser.add_argument('--warmup', type=int, default=2, help='Estimates run and discarded before profiling.')
    parser.add_argument('--active', type=int, default=1, help='Estimates profiled.')
    parser.add_argument('--full-replay', action='store_true', help='Also profile full-replay (upstream) mode.')
    parser.add_argument('--row-limit', type=int, default=40)
    args = parser.parse_args()

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda':
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')
    if args.toy:
        from toy import TOY_BATCH, TOY_CONFIG, toy_data, toy_recipe
        config, settings = TOY_CONFIG, toy_recipe()
        rows = toy_data(2, TOY_BATCH)[0][:TOY_BATCH]
    else:
        config, settings = GPTConfig(), dust.recipe(args.tokens, args.population)
        rows = dust.load_sequences(args.data_dir / f'train_{args.tokens}.pt', 8)
    rows = rows.to(device)
    x, y = rows[:, :-1].contiguous(), rows[:, 1:].contiguous()
    model = dust.make_model(config, device, seed=42)

    environment = dict(torch=torch.__version__, device=str(device), population=args.population, toy=args.toy)
    if device.type == 'cuda':
        environment.update(gpu=torch.cuda.get_device_name(device), cuda=torch.version.cuda,
                           flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
                           mem_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
                           math_sdp=torch.backends.cuda.math_sdp_enabled())
    modes = [('cached-suffix', True)] + ([('full-replay', False)] if args.full_replay else [])
    report = dict(environment=environment, modes={})
    for mode, suffix in modes:
        folder = args.output / mode
        folder.mkdir(parents=True, exist_ok=True)
        estimator = dust.DUST(model, settings, seed=7, suffix=suffix)
        try:
            sort = 'self_cuda_time_total' if device.type == 'cuda' else 'self_cpu_time_total'
            start = time.monotonic()
            table, prof = profile_estimate(estimator, x, y, warmup=args.warmup, active=args.active, sort_by=sort,
                                           row_limit=args.row_limit, trace_path=folder / 'trace.json')
            wall = time.monotonic() - start  # includes warmup; replaced below by the profiled part only
        finally:
            estimator.close()
        # profile_estimate runs warmup + active estimates; split the wall time evenly as a rough figure.
        summary = summarize(prof, args.active, wall * args.active / (args.warmup + args.active))
        (folder / 'table_cuda.txt').write_text(table + '\n')
        (folder / 'table_cpu.txt').write_text(
            prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=args.row_limit) + '\n')
        (folder / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        report['modes'][mode] = summary
        print(f'[{mode}] SDPA kernel: {summary["sdpa_verdict"]}', flush=True)
        for phase, row in summary['phases'].items():
            print(f'  dust/{phase:10s} gpu {row["gpu_ms"]:10.1f} ms   cpu {row["cpu_ms"]:10.1f} ms', flush=True)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    (args.output / 'profile.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Wrote {args.output}/profile.json and per-mode tables + traces', flush=True)


if __name__ == '__main__':
    main()
