"""Refuse incomplete or mismatched full-replay/suffix comparisons.

    python3 scripts/check_baseline_results.py EXPERIMENT_DIR [--seed 42]

Standard library only. Writes EXPERIMENT_DIR/summary.json.
"""
import argparse
import json
import math
import statistics
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


def peak_memory_gib(record):
    # Runs before 2026-11-18 named the field peak_memory_gb; it was always GiB.
    return record['peak_memory_gib'] if 'peak_memory_gib' in record else record['peak_memory_gb']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('experiment', type=Path)
    parser.add_argument('--seed', type=int, default=42, help='Seed both runs must have used.')
    args = parser.parse_args()
    experiment = args.experiment
    full, cached = (read_run(experiment / name) for name in ('full-replay', 'cached-suffix'))
    left, right = full[0], cached[0]
    if left['execution'] != 'full-replay' or right['execution'] != 'cached-suffix':
        raise ValueError('The two execution modes are not full-replay and cached-suffix')
    differences = [key for key in set(left) | set(right)
                   if key != 'execution' and left.get(key) != right.get(key)]
    if differences:
        raise ValueError(f'Run configurations differ beyond execution mode: {differences}')
    for key, value in dict(tokens='1M', population=256, seed=args.seed, steps=61, world=1,
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
            'peak_training_memory_gib': peak_memory_gib(records[-1]),
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
