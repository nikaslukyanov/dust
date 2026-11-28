"""Validate A/B run records and aggregate within-seed comparisons."""
import json
import math
from dataclasses import replace
from pathlib import Path

from check_baseline_results import distribution

# Keep this module importable through both pytest and the runner.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dust import GPTConfig, Recipe  # noqa: E402
from accounting import RunMeters  # noqa: E402
from allocation import unit_counts, with_counts  # noqa: E402
from run_metrics import allocation, dust_update_cost  # noqa: E402


def read_records(folder, last_step=None):
    """Read all records, or only the prefix committed in a checkpoint."""
    with (folder / 'metrics.jsonl').open() as stream:
        records = []
        for line in stream:
            records.append(json.loads(line))
            if last_step is not None and len(records) == last_step + 1:
                break
    return records


def check_history(history, records):
    expected = [{'step': 0, 'val_loss': records[0]['val_loss']}, *records[1:]]
    if history != expected:
        raise ValueError('history differs from metrics.jsonl')


def number(record, key, *, positive=False):
    value = record[key]
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f'invalid {key}: {value!r}')
    return value


def check_totals(record, totals):
    for key, total in totals.items():
        value = number(record, key)
        if (value != total if type(total) is int else not math.isclose(value, total, rel_tol=1e-12)):
            raise ValueError(f'step {record.get("step", "final")} has inconsistent {key}')


def check_allocation(previous, new, options, step):
    if not options or not options['interval'] or (step - 1) % options['interval']:
        raise ValueError(f'unexpected allocation change at step {step}')
    hubs = options['policy'] == 'agreement'
    before, after = unit_counts(previous, hubs), unit_counts(new, hubs)
    if with_counts(previous, after) != new or sum(after.values()) != sum(before.values()):
        raise ValueError(f'step {step} changes draws outside the policy budget')
    if any(type(count) is not int or count < options['minimum'] or count % new.nrep for count in after.values()):
        raise ValueError(f'step {step} violates the allocation minimum or quantum')


def check_records(records, case, settings, last_step):
    if [row['step'] for row in records] != list(range(last_step + 1)):
        raise ValueError(f'expected complete metrics through step {last_step}')
    config = GPTConfig(**settings['model'])
    active = Recipe(**case['initial_recipe'])
    cost = dust_update_cost(active, config, batch=settings['batch_size'])
    totals = dict.fromkeys(('flops', 'flops_executed', 'forward_equivalents', 'draws', 'train_seconds'), 0)
    check_totals(records[0], totals)
    number(records[0], 'val_loss')
    changes = []
    for row in records[1:]:
        step = row['step']
        evaluates = step == 1 or step % settings['evaluation_interval'] == 0 or step == settings['steps']
        if ('val_loss' in row) != evaluates:
            raise ValueError(f'step {step} does not match the evaluation schedule')
        if evaluates:
            number(row, 'val_loss')
        number(row, 'train_loss')
        totals['train_seconds'] += number(row, 'step_host_ms', positive=True) / 1000
        if settings['mode'] == 'l40s':
            number(row, 'step_gpu_ms', positive=True)
        elif 'step_gpu_ms' in row:
            raise ValueError('CPU toy run contains GPU timing')
        if 'allocation' in row:
            new = replace(active, **row['allocation'])
            if row['allocation'] != allocation(new):
                raise ValueError(f'step {step} has an incomplete allocation')
            if step == 1:
                if new != active:
                    raise ValueError('step 1 does not use the expected initial allocation')
            else:
                check_allocation(active, new, case['allocation'], step)
            active = new
            cost = dust_update_cost(active, config, batch=settings['batch_size'])
            changes.append(dict(step=step, **row['allocation']))
        elif step == 1:
            raise ValueError('step 1 is missing the initial allocation')
        for key in ('flops', 'flops_executed', 'forward_equivalents', 'draws'):
            totals[key] += cost[key]
        check_totals(row, dict(totals, population=cost['population'], direct_forwards=cost['direct_forwards']))
    return changes, totals


def summarize_metrics(records, result, warmup):
    final = records[-1]
    meters = RunMeters(warmup=warmup)
    for record in records:
        if record['step']:
            meters.record_step(record['step_host_ms'], record.get('step_gpu_ms'))
    timing = meters.summary()
    return dict(
        best_val=result['best_val'], best_step=result['best_step'], test_at_best_val=result['test_at_best_val'],
        training_seconds=final['train_seconds'], draws=final['draws'], population=final['population'],
        analytic_needed_flops=final['flops'], analytic_executed_flops=final['flops_executed'],
        allocation_seconds=result['timing']['totals']['allocation_seconds'],
        **{key: timing[key] for key in timing
           if key.endswith(('_median_ms', '_p10_ms', '_p90_ms'))},
    )


def validate_result(folder, case, settings):
    result = json.loads((folder / 'result.json').read_text())
    records = read_records(folder)
    history, totals = check_records(records, case, settings, settings['steps'])
    check_history(result['history'], records)
    best = min((row for row in records if 'val_loss' in row), key=lambda row: row['val_loss'])
    if result['best_step'] != best['step'] or number(result, 'best_val') != best['val_loss']:
        raise ValueError('result does not select the first best validation checkpoint')
    number(result, 'test_at_best_val')
    timing = result['timing']
    check_totals(timing['totals'], totals)
    decision = number(timing['totals'], 'allocation_seconds', positive=case['allocation'] is not None)
    if decision > totals['train_seconds'] or (case['allocation'] is None and decision != 0):
        raise ValueError('invalid allocation decision time')
    elapsed = number(timing, 'elapsed_seconds', positive=True)
    if elapsed < totals['train_seconds']:
        raise ValueError('elapsed time is shorter than training time')
    return dict(metrics=dict(summarize_metrics(records, result, settings['warmup_steps']),
                             elapsed_seconds=elapsed), allocation_history=history)


def metric_columns(rows):
    return {key: distribution([row[key] for row in rows]) for key in rows[0]}


def compare_seeds(seeds, cases):
    comparisons = []
    for case in cases:
        if case['control'] is None:
            continue
        paired = []
        for seed in seeds:
            base = seed['cases'][case['control']]['metrics']
            measured = seed['cases'][case['name']]['metrics']
            delta = {key: measured[key] - base[key] for key in base if key != 'best_step'}
            percent = {key: 100 * (1 - measured[key] / base[key]) for key in delta
                       if key.endswith('_ms') or key in ('training_seconds', 'analytic_needed_flops',
                                                        'analytic_executed_flops')}
            paired.append(dict(seed=seed['seed'], delta=delta, reduction_percent=percent))
        comparisons.append(dict(case=case['name'], control=case['control'], seeds=paired,
                                delta=metric_columns([row['delta'] for row in paired]),
                                reduction_percent=metric_columns([row['reduction_percent'] for row in paired])))
    return comparisons


def summarize_campaign(manifest, seeds):
    return dict(
        mode=manifest['settings']['mode'], warmup_steps=manifest['settings']['warmup_steps'],
        source=manifest['source'], data=manifest['data'], runtime=seeds[0]['runtime'], seeds=seeds,
        cases={case['name']: metric_columns([seed['cases'][case['name']]['metrics'] for seed in seeds])
               for case in manifest['cases']},
        comparisons=compare_seeds(seeds, manifest['cases']),
    )
