"""Validate sweep statistics and reject comparisons that do not share a workload."""
import json
import statistics

import pytest

from check_baseline_results import summarize_sweep


def write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def sweep(tmp_path):
    write_json(tmp_path / 'launch.json', dict(seeds=[42, 43, 44], backprop=True, order='alternate'))
    for seed, full_ms, suffix_ms in [(42, 100, 80), (43, 200, 100), (44, 400, 100)]:
        folder = tmp_path / f'seed-{seed}'
        folder.mkdir()
        (folder / 'status.txt').write_text('stage=completed\nexit_code=0\n')
        write_json(folder / 'gpu-environment.json', dict(gpu='NVIDIA L40S', torch='2.10.0+cu128', cuda='12.8'))
        for method, ms in [('full-replay', full_ms), ('cached-suffix', suffix_ms), ('backprop', 5)]:
            run = folder / method
            run.mkdir()
            config = dict(tokens='1M', seed=seed, steps=61, training_tokens=999424,
                          validation_sequences=544, test_sequences=544, model={'n_layer': 8},
                          model_sha256='model-hash', data={'sha256': {'train': 'data-hash'}},
                          torch_version='2.10.0+cu128', device='cuda:0')
            if method == 'backprop':
                config.update(method=method, lr=0.175, momentum=0.98)
            else:
                config.update(execution=method, population=256, world=1, allocation=None, multiplex=None)
            write_json(run / 'config.json', config)
            write_json(run / 'result.json', dict(best_val=6, best_step=61, test_at_best_val=6.1))
            rows = [dict(step=0, val_loss=8)] + [
                dict(step=step, train_loss=6, val_loss=6, step_host_ms=1e6 if step <= 3 else ms,
                     train_seconds=step * ms / 1000, peak_memory_gib=7)
                for step in range(1, 62)]
            (run / 'metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return tmp_path


def change_json(path, **changes):
    value = json.loads(path.read_text())
    value.update(changes)
    write_json(path, value)


def test_sweep_uses_paired_seed_reductions_and_discards_warmup(sweep):
    report = summarize_sweep(sweep)
    paired = report['step_time_reduction_percent']
    assert paired['mean'] == pytest.approx(statistics.mean([20, 50, 75]))
    assert paired['sample_std'] == pytest.approx(statistics.stdev([20, 50, 75]))
    assert paired['n'] == 3
    assert report['methods']['full-replay']['median_step_host_ms']['mean'] == pytest.approx(700 / 3)
    assert report['methods']['backprop']['median_step_host_ms']['mean'] == 5
    assert report['seeds'][0]['runs']['cached-suffix']['train_seconds'] == pytest.approx(4.88)


def test_incomplete_run_is_rejected(sweep):
    path = sweep / 'seed-43/backprop/metrics.jsonl'
    path.write_text('\n'.join(path.read_text().splitlines()[:-1]) + '\n')
    with pytest.raises(ValueError, match='complete 61-step'):
        summarize_sweep(sweep)


@pytest.mark.parametrize('path,change', [
    ('seed-42/cached-suffix/config.json', {'seed': 99}),
    ('seed-42/backprop/config.json', {'model': {'n_layer': 4}}),
    ('seed-42/gpu-environment.json', {'gpu': 'NVIDIA H100'}),
])
def test_mismatched_runs_are_rejected(sweep, path, change):
    change_json(sweep / path, **change)
    with pytest.raises(ValueError):
        summarize_sweep(sweep)


def test_different_data_between_seeds_is_rejected(sweep):
    for method in ('full-replay', 'cached-suffix', 'backprop'):
        change_json(sweep / f'seed-44/{method}/config.json', data={'sha256': {'train': 'other-data'}})
    with pytest.raises(ValueError, match='data'):
        summarize_sweep(sweep)


@pytest.mark.parametrize('change', [{'lr': 0.2}, {'momentum': 0.9}])
def test_backprop_optimizer_changes_between_seeds_are_rejected(sweep, change):
    change_json(sweep / 'seed-44/backprop/config.json', **change)
    with pytest.raises(ValueError, match='backprop configurations'):
        summarize_sweep(sweep)


def test_duplicate_seeds_are_rejected(sweep):
    change_json(sweep / 'launch.json', seeds=[42, 42, 44])
    with pytest.raises(ValueError, match='unique'):
        summarize_sweep(sweep)


def test_failed_job_is_rejected(sweep):
    (sweep / 'seed-43/status.txt').write_text('stage=backprop\nexit_code=1\n')
    with pytest.raises(ValueError, match='completed'):
        summarize_sweep(sweep)
