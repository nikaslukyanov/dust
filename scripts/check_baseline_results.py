"""Refuse incomplete or mismatched full-replay/suffix comparisons.

    python3 scripts/check_baseline_results.py EXPERIMENT_DIR [--seed 42]
    python3 scripts/check_baseline_results.py BATCH_DIR --sweep

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


def summarize_run(run):
    _, result, records = run
    kept = [row for row in records if row['step'] > 3]
    summary = {
        'best_val': result['best_val'], 'best_step': result['best_step'],
        'test_at_best_val': result['test_at_best_val'],
        'median_step_host_ms': statistics.median(row['step_host_ms'] for row in kept),
        'train_seconds': records[-1]['train_seconds'],
        'peak_training_memory_gib': peak_memory_gib(records[-1]),
    }
    if not all(math.isfinite(value) for value in summary.values()):
        raise ValueError('Run summary contains a nonfinite metric')
    if summary['median_step_host_ms'] <= 0:
        raise ValueError('Median step time must be positive')
    return summary


def summarize_pair(experiment, seed):
    full, cached = (read_run(experiment / name) for name in ('full-replay', 'cached-suffix'))
    left, right = full[0], cached[0]
    if left['execution'] != 'full-replay' or right['execution'] != 'cached-suffix':
        raise ValueError('The two execution modes are not full-replay and cached-suffix')
    differences = [key for key in set(left) | set(right)
                   if key != 'execution' and left.get(key) != right.get(key)]
    if differences:
        raise ValueError(f'Run configurations differ beyond execution mode: {differences}')
    for key, value in dict(tokens='1M', population=256, seed=seed, steps=61, world=1,
                           allocation=None, multiplex=None).items():
        if left[key] != value:
            raise ValueError(f'Unexpected experiment setting {key}={left[key]!r}')
    summary = {'matched_configuration': True, 'runs': {}}
    for name, run in zip(('full-replay', 'cached-suffix'), (full, cached)):
        summary['runs'][name] = summarize_run(run)
    baseline = summary['runs']['full-replay']['median_step_host_ms']
    suffix = summary['runs']['cached-suffix']['median_step_host_ms']
    summary['median_step_time_reduction_percent'] = 100 * (1 - suffix / baseline)
    full_val = {row['step']: row['val_loss'] for row in full[2] if 'val_loss' in row}
    cached_val = {row['step']: row['val_loss'] for row in cached[2] if 'val_loss' in row}
    if full_val.keys() != cached_val.keys():
        raise ValueError('Validation schedules differ')
    summary['maximum_absolute_validation_difference'] = max(
        abs(full_val[step] - cached_val[step]) for step in full_val)
    return summary


def distribution(values):
    """Seed-level variation; steps within a run are not independent repetitions."""
    return dict(n=len(values), mean=statistics.mean(values),
                sample_std=statistics.stdev(values) if len(values) > 1 else None,
                minimum=min(values), maximum=max(values))


def summarize_sweep(batch):
    """Validate complete launcher jobs before aggregating paired seed measurements."""
    launch = json.loads((batch / 'launch.json').read_text())
    seeds = launch['seeds']
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('A sweep needs a nonempty list of unique seeds')
    reports, reference, environment = [], None, None
    backprop_reference = None
    for seed in seeds:
        folder = batch / f'seed-{seed}'
        status = dict(line.split('=', 1) for line in (folder / 'status.txt').read_text().splitlines())
        if status.get('stage') != 'completed' or status.get('exit_code') != '0':
            raise ValueError(f'{folder}: job has not completed successfully')
        current = json.loads((folder / 'gpu-environment.json').read_text())
        if 'L40S' not in current['gpu'] or (environment is not None and current != environment):
            raise ValueError(f'{folder}: GPU environments differ or are not L40S')
        environment = current
        pair = summarize_pair(folder, seed)
        config = json.loads((folder / 'full-replay/config.json').read_text())
        comparable = {key: value for key, value in config.items() if key != 'seed'}
        if reference is not None and comparable != reference:
            differences = sorted(key for key in set(comparable) | set(reference)
                                 if comparable.get(key) != reference.get(key))
            raise ValueError(f'{folder}: configurations differ across seeds: {differences}')
        reference = comparable
        if launch['backprop']:
            backprop = read_run(folder / 'backprop')
            if backprop[0].get('method') != 'backprop':
                raise ValueError(f'{folder}: expected the backprop reference')
            for key in ('seed', 'tokens', 'steps', 'training_tokens', 'validation_sequences',
                        'test_sequences', 'model', 'model_sha256', 'data', 'torch_version', 'device'):
                if backprop[0][key] != config[key]:
                    raise ValueError(f'{folder}: backprop differs in {key}')
            comparable_backprop = {key: value for key, value in backprop[0].items() if key != 'seed'}
            if backprop_reference is not None and comparable_backprop != backprop_reference:
                raise ValueError(f'{folder}: backprop configurations differ across seeds')
            backprop_reference = comparable_backprop
            pair['runs']['backprop'] = summarize_run(backprop)
        reports.append(dict(seed=seed, **pair))
    methods = {
        method: {metric: distribution([row['runs'][method][metric] for row in reports])
                 for metric in reports[0]['runs'][method] if metric != 'best_step'}
        for method in reports[0]['runs']
    }
    return dict(source_commit=launch.get('commit'), warmup_steps=3, environment=environment,
                seeds=reports, methods=methods,
                step_time_reduction_percent=distribution([
                    row['median_step_time_reduction_percent'] for row in reports]),
                maximum_absolute_validation_difference=max(
                    row['maximum_absolute_validation_difference'] for row in reports))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('experiment', type=Path)
    parser.add_argument('--seed', type=int, default=42, help='Seed both runs must have used.')
    parser.add_argument('--sweep', action='store_true', help='Validate and aggregate every seed in launch.json.')
    args = parser.parse_args()
    summary = summarize_sweep(args.experiment) if args.sweep else summarize_pair(args.experiment, args.seed)
    encoded = json.dumps(summary, indent=2, allow_nan=False)
    (args.experiment / 'summary.json').write_text(encoded + '\n')
    print(encoded)


if __name__ == '__main__':
    main()
