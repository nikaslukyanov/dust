"""Compare finished training runs side by side, the first one as the reference.

    python new_implementation/compare_runs.py runs/full-replay runs/suffix

Reads each run's config.json, metrics.jsonl and result.json. For the upstream-vs-cached-suffix
comparison both runs consume the same noise, so their loss curves should agree up to bf16 rounding
(``max |val diff|``) and the difference is in time, memory and FLOPs. Warmup steps are excluded from the
step-time percentiles (result.json's timing).
"""
import json
import sys
from pathlib import Path

PHASES = ('clean', 'direct', 'embedding', 'hubs', 'head', 'local', 'scalars')


def load(folder):
    folder = Path(folder)
    records = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
    config = json.loads((folder / 'config.json').read_text()) if (folder / 'config.json').is_file() else {}
    result = json.loads((folder / 'result.json').read_text()) if (folder / 'result.json').is_file() else None
    return dict(name=folder.name, records=records, config=config, result=result)


def median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2] if ordered else float('nan')


def summarize(run, warmup=3):
    steps = [r for r in run['records'] if r['step'] > 0]
    kept = steps[warmup:] or steps
    last, result = steps[-1], run['result'] or {}
    row = {
        'execution': run['config'].get('execution', '?'),
        'steps': last['step'],
        'best val': result.get('best_val', min(r['val_loss'] for r in run['records'] if 'val_loss' in r)),
        'test at best val': result.get('test_at_best_val', float('nan')),
        'final train loss': last['train_loss'],
        'forward-equivalents / step': last['forward_equivalents'] / last['step'],
        'FLOPs / step': last['flops'] / last['step'],
        'step host ms (median)': median([r['step_host_ms'] for r in kept]),
        'step gpu ms (median)': median([r['step_gpu_ms'] for r in kept if 'step_gpu_ms' in r]),
        'train seconds': last['train_seconds'],
        'peak memory GB': last.get('peak_memory_gb', float('nan')),
    }
    for phase in PHASES:
        row[f'  {phase} s (median)'] = median([r['phase_seconds'][phase] for r in kept if 'phase_seconds' in r])
    probes = [r['probe'] for r in steps if 'probe' in r]
    if probes:
        row['probe cosine (mean over probes)'] = sum(p['mean_group_cosine'] for p in probes) / len(probes)
    return row


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    runs = [load(folder) for folder in sys.argv[1:]]
    rows = [summarize(run) for run in runs]
    width = max(len(key) for row in rows for key in row)
    print(' ' * width + ''.join(f'{run["name"]:>18}' for run in runs) + ('   vs first' if len(runs) > 1 else ''))
    for key in rows[0]:
        values = [row.get(key, float('nan')) for row in rows]
        line = f'{key:<{width}}' + ''.join(f'{v:>18.6g}' if isinstance(v, (int, float)) else f'{v:>18}' for v in values)
        if len(runs) > 1 and all(isinstance(v, (int, float)) for v in values) and values[0]:
            line += ''.join(f'{100 * (v / values[0] - 1):>+9.1f}%' for v in values[1:])
        print(line)
    reference = {r['step']: r['val_loss'] for r in runs[0]['records'] if 'val_loss' in r}
    for run in runs[1:]:
        shared = [abs(r['val_loss'] - reference[r['step']]) for r in run['records']
                  if 'val_loss' in r and r['step'] in reference]
        if shared:
            print(f'max |val diff| {run["name"]} vs {runs[0]["name"]}: {max(shared):.3g} over {len(shared)} evaluations')


if __name__ == '__main__':
    main()
