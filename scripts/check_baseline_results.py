"""Refuse incomplete or mismatched full-replay/suffix comparisons."""
import json
import math
import statistics
import sys
from pathlib import Path


def read_run(folder):
    config = json.loads((folder / 'config.json').read_text())
    result = json.loads((folder / 'result.json').read_text())
    records = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
    if [row['step'] for row in records] != list(range(62)):
        raise ValueError(f'{folder}: expected the complete 61-step run, including step zero')
    for row in records:
        for key in ('train_loss', 'val_loss', 'step_host_ms'):
            if key in row and not math.isfinite(row[key]):
                raise ValueError(f'{folder}: nonfinite {key} at step {row["step"]}')
    for key in ('best_val', 'test_at_best_val'):
        if not math.isfinite(result[key]):
            raise ValueError(f'{folder}: nonfinite {key}')
    return config, result, records


def main():
    experiment = Path(sys.argv[1])
    full, cached = (read_run(experiment / name) for name in ('full-replay', 'cached-suffix'))
    left, right = full[0], cached[0]
    if left['execution'] != 'full-replay' or right['execution'] != 'cached-suffix':
        raise ValueError('The two execution modes are not full-replay and cached-suffix')
    differences = [key for key in set(left) | set(right)
                   if key != 'execution' and left.get(key) != right.get(key)]
    if differences:
        raise ValueError(f'Run configurations differ beyond execution mode: {differences}')
    for key, value in dict(tokens='1M', population=256, seed=42, steps=61, world=1,
                           allocation=None, multiplex=None).items():
        if left[key] != value:
            raise ValueError(f'Unexpected experiment setting {key}={left[key]!r}')
    summary = {'matched_configuration': True, 'runs': {}}
    for name, run in zip(('full-replay', 'cached-suffix'), (full, cached)):
        _, result, records = run
        kept = [row for row in records if row['step'] > 3]
        summary['runs'][name] = {
            'best_val': result['best_val'], 'best_step': result['best_step'],
            'test_at_best_val': result['test_at_best_val'],
            'median_step_host_ms': statistics.median(row['step_host_ms'] for row in kept),
            'train_seconds': records[-1]['train_seconds'],
            'peak_training_memory_gb': records[-1]['peak_memory_gb'],
        }
    baseline = summary['runs']['full-replay']['median_step_host_ms']
    suffix = summary['runs']['cached-suffix']['median_step_host_ms']
    summary['median_step_time_reduction_percent'] = 100 * (1 - suffix / baseline)
    full_val = {row['step']: row['val_loss'] for row in full[2] if 'val_loss' in row}
    cached_val = {row['step']: row['val_loss'] for row in cached[2] if 'val_loss' in row}
    if full_val.keys() != cached_val.keys():
        raise ValueError('Validation schedules differ')
    summary['maximum_absolute_validation_difference'] = max(
        abs(full_val[step] - cached_val[step]) for step in full_val)
    (experiment / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
