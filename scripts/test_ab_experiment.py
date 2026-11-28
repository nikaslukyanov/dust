import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def cli(*args):
    return subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True)


def plan(*args):
    result = cli('scripts/ab_experiment.py', 'plan', *args)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def population(recipe):
    return sum(recipe['writers'] + recipe['hidden'] + recipe['hubs']) + recipe['embedding']


@pytest.mark.parametrize(('mode', 'budget', 'steps'), [((), 256, 61), (('--toy',), 68, 5)])
def test_plan_matches_start_and_overhead_controls(mode, budget, steps):
    manifest = plan(*mode)
    assert manifest['seeds'] == [42, 43, 44]
    assert manifest['settings']['steps'] == steps
    cases = {case['name']: case for case in manifest['cases']}
    assert set(cases) == {
        'fixed', 'adaptive-variance', 'adaptive-agreement', 'fixed-variance', 'measure-variance',
        'fixed-agreement', 'measure-agreement',
    }
    fixed = cases['fixed']
    assert fixed['requested_recipe'] == fixed['initial_recipe']
    assert all(population(case['initial_recipe']) == budget for case in cases.values())
    for policy in ('variance', 'agreement'):
        adaptive, control, measure = (cases[f'{kind}-{policy}'] for kind in ('adaptive', 'fixed', 'measure'))
        assert adaptive['requested_recipe'] == measure['requested_recipe'] == fixed['requested_recipe']
        assert adaptive['initial_recipe'] == measure['initial_recipe'] == control['requested_recipe']
        assert measure['control'] == control['name']
        assert measure['allocation']['interval'] == 0
        assert adaptive['allocation']['interval'] == 4
        assert adaptive['allocation']['budget'] == 'draws'
        assert control['allocation'] is None
    if not mode:
        variance = cases['adaptive-variance']['initial_recipe']
        agreement = cases['adaptive-agreement']['initial_recipe']
        assert variance['writers'] == variance['hidden'] == [36, 30, 16, 10, 6, 4, 4, 4]
        assert variance['hubs'] == fixed['requested_recipe']['hubs']
        assert agreement['writers'] == agreement['hidden'] == [34, 30, 14, 10, 6, 4, 4, 4]
        assert agreement['hubs'] == [6, 6, 4, 4, 4, 4, 4, 4]
        assert (variance['embedding'], agreement['embedding']) == (10, 8)
    paths = [path for seed in manifest['runs'] for path in seed['paths'].values()]
    assert len(set(paths)) == len(paths) == 21
    assert len({tuple(seed['order']) for seed in manifest['runs']}) == 3


@pytest.mark.parametrize('args', [('--seeds', 42, 42), ('--seeds', -1), ('--toy', '--steps', 3), ('--steps', 5)])
def test_plan_rejects_invalid_experiments(args):
    result = cli('scripts/ab_experiment.py', 'plan', *args)
    assert result.returncode == 2
    assert 'usage:' in result.stderr


def test_slurm_dry_run_shell_quotes_and_no_writes(tmp_path):
    out = tmp_path / "campaign space ' quoted"
    result = cli('scripts/launch_ab.py', '--dry-run', '--output', out, '--seeds', 43, 44)
    assert result.returncode == 0, result.stderr
    assert all(name in result.stdout for name in ('adaptive-variance', 'fixed-agreement', 'measure-agreement'))
    commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith('sbatch ')]
    setup, first, second = commands
    assert setup[-2:] == [str(out), 'ab']
    for command, seed in zip((first, second), (43, 44)):
        assert command[-2:] == [str(out), str(seed)]
        assert f'--output={out}/seed-{seed}/job-%j.log' in command
    assert not out.exists()


def run_toy(out, *args):
    return cli('scripts/ab_experiment.py', 'run', '--toy', '--output', out, '--seeds', 42, '--steps', 5, *args)


@pytest.fixture(scope='module')
def finished_toy(tmp_path_factory):
    out = tmp_path_factory.mktemp('finished-toy')
    result = run_toy(out)
    assert result.returncode == 0, result.stderr
    return out


def summary(out):
    result = cli('scripts/ab_experiment.py', 'summary', '--output', out)
    assert result.returncode == 0, result.stderr
    return json.loads((out / 'summary.json').read_text())


def test_completed_run_skips_and_matches_overhead_controls(finished_toy):
    report = summary(finished_toy)
    cases = report['seeds'][0]['cases']
    assert report['mode'] == 'cpu-toy'
    assert report['warmup_steps'] == 3
    assert all('gpu_median_ms' not in case['metrics'] for case in cases.values())
    assert {case['metrics']['draws'] for case in cases.values()} == {860}
    for policy in ('variance', 'agreement'):
        control, measured = (cases[f'{kind}-{policy}'] for kind in ('fixed', 'measure'))
        assert control['allocation_history'] == measured['allocation_history']
        for key in ('best_val', 'test_at_best_val', 'draws', 'analytic_needed_flops', 'analytic_executed_flops'):
            assert control['metrics'][key] == measured['metrics'][key]
        assert measured['metrics']['allocation_seconds'] > 0
        assert control['metrics']['allocation_seconds'] == 0
    result = run_toy(finished_toy)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count('Skipping verified') == 7
    for policy in ('variance', 'agreement'):
        adaptive = cases[f'adaptive-{policy}']
        assert [row['step'] for row in adaptive['allocation_history']] == [1, 5]
        assert all(population(row) == 68 for row in adaptive['allocation_history'])
        assert adaptive['metrics']['analytic_executed_flops'] != cases['fixed']['metrics']['analytic_executed_flops']
    result = cli('scripts/ab_experiment.py', 'run', '--toy', '--output', finished_toy, '--seeds', 42, '--steps', 6)
    assert result.returncode != 0
    assert 'incompatible manifest' in result.stderr


@pytest.mark.parametrize('problem', ['truncated-metrics', 'missing-result', 'wrong-data', 'wrong-source', 'wrong-cost'])
def test_summary_rejects_incomplete_or_incompatible_results(tmp_path, finished_toy, problem):
    out = tmp_path / 'bad-run'
    shutil.copytree(finished_toy, out)
    (out / 'summary.json').unlink(missing_ok=True)
    case = out / 'seed-42/fixed'
    if problem == 'truncated-metrics':
        path = case / 'metrics.jsonl'
        path.write_text('\n'.join(path.read_text().splitlines()[:-1]) + '\n')
    elif problem == 'missing-result':
        (case / 'result.json').unlink()
    elif problem == 'wrong-cost':
        metrics, result = case / 'metrics.jsonl', case / 'result.json'
        rows = [json.loads(line) for line in metrics.read_text().splitlines()]
        rows[2]['flops'] += 1_000_000
        metrics.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        saved = json.loads(result.read_text())
        saved['history'][2] = rows[2]
        result.write_text(json.dumps(saved))
        complete = case / 'complete.json'
        hashes = json.loads(complete.read_text())
        for path in (metrics, result):
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        complete.write_text(json.dumps(hashes))
    else:
        path = out / 'manifest.json'
        data = json.loads(path.read_text())
        data['data' if problem == 'wrong-data' else 'source'] = {}
        path.write_text(json.dumps(data))
    result = cli('scripts/ab_experiment.py', 'summary', '--output', out)
    assert result.returncode != 0
    assert not (out / 'summary.json').exists()
    if problem == 'wrong-cost':
        assert 'step 2 has inconsistent flops' in result.stderr


def test_verified_data_detects_changed_bytes(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    files = ('train_1M.pt', 'val.pt', 'test.pt')
    original = b'verified token bytes'
    sha = hashlib.sha256(original).hexdigest()
    for name in files:
        (data / name).write_bytes(original)
    (data / 'manifest.json').write_text(json.dumps(dict(
        files={name: {'sha256': sha} for name in files},
        tokenizer_sha256=hashlib.sha256((ROOT / 'tokenizer/tokenizer.json').read_bytes()).hexdigest(),
    )))
    out = tmp_path / 'campaign'
    out.mkdir()
    (out / 'launch.json').write_text(json.dumps(dict(seeds=[42], commit='1' * 40)))
    args = ('scripts/ab_experiment.py', 'prepare', '--output', out, '--data-dir', data)
    prepared = cli(*args)
    assert prepared.returncode == 0, prepared.stderr
    baseline = cli('scripts/verify_data.py', data)
    assert baseline.returncode == 0, baseline.stderr
    assert baseline.stdout.strip() == 'Verified data and tokenizer hashes'

    (data / 'test.pt').write_bytes(b'different held-out tokens')
    prepared = cli(*args)
    assert prepared.returncode != 0
    assert 'Data checksum mismatch: test.pt' in prepared.stderr
    baseline = cli('scripts/verify_data.py', data)
    assert baseline.returncode != 0
    assert 'Data checksum mismatch: test.pt' in baseline.stderr


@pytest.mark.parametrize('problem', ['corrupt', 'missing'])
def test_unsealed_result_rejects_an_invalid_checkpoint(tmp_path, finished_toy, problem):
    out = tmp_path / 'unsealed'
    shutil.copytree(finished_toy, out)
    case = out / 'seed-42/fixed'
    receipt, checkpoint = case / 'complete.json', case / 'state.pt'
    receipt.unlink()
    if problem == 'corrupt':
        checkpoint.write_text('truncated checkpoint\n')
    else:
        checkpoint.unlink()

    call = run_toy(out)
    assert (call.returncode, receipt.exists()) == (2, False), call.stdout + call.stderr
    assert str(case) in call.stderr
    assert 'checkpoint' in call.stderr


def test_unsealed_result_recovers_from_the_checkpoint(tmp_path, finished_toy):
    out = tmp_path / 'unsealed'
    shutil.copytree(finished_toy, out)
    case = out / 'seed-42/fixed'
    result = case / 'result.json'
    original = json.loads(result.read_text())
    (case / 'complete.json').unlink()

    call = run_toy(out)
    assert call.returncode == 0, call.stderr
    assert f'Resuming {case} at step 5 of 5' in call.stdout
    assert call.stdout.count('Skipping verified') == 6
    recovered = json.loads(result.read_text())
    assert recovered['history'] == original['history']
    assert recovered['test_at_best_val'] == original['test_at_best_val']
    assert summary(out)['seeds'][0]['cases']['fixed']['metrics']['draws'] == 860


def test_recovers_a_final_checkpoint_with_a_torn_tail(tmp_path, finished_toy):
    out = tmp_path / 'interrupted'
    shutil.copytree(finished_toy, out)
    case = out / 'seed-42/adaptive-variance'
    result = case / 'result.json'
    old = json.loads(result.read_text())
    result.unlink()
    (case / 'complete.json').unlink()
    metrics = case / 'metrics.jsonl'
    before = [json.loads(line) for line in metrics.read_text().splitlines()]
    with metrics.open('a') as stream:
        stream.write('{"step":')
    call = run_toy(out)
    assert call.returncode == 0, call.stderr
    assert f'Resuming {case} at step 5 of 5' in call.stdout
    assert call.stdout.count('Skipping verified') == 6
    assert json.loads(result.read_text())['test_at_best_val'] == old['test_at_best_val']
    assert [json.loads(line) for line in metrics.read_text().splitlines()] == before
    assert summary(out)['seeds'][0]['cases']['adaptive-variance']['metrics']['draws'] == 860
