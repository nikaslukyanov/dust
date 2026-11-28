"""Public CLI checks for the frozen-model diagnostics."""
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def cli(*args):
    return subprocess.run(
        [sys.executable, str(ROOT / 'scripts/gpu_diagnostics.py'), *map(str, args)],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )


def read_json(path):
    return json.loads(path.read_text())


def test_profile_toy_writes_evidence(tmp_path):
    output = tmp_path / 'toy profile'
    run = cli('profile', '--toy', '--output', output, '--warmup', 1, '--repeats', 2, '--active', 2)
    assert run.returncode == 0, run.stderr
    manifest = read_json(output / 'manifest.json')
    result = read_json(output / 'result.json')

    assert manifest['runtime']['device'] == 'cpu'
    assert manifest['config']['batch_size'] == 8
    assert manifest['recipes'][0]['scale'] == 1
    assert manifest['data']['shape'] == [8, 65]
    assert len(manifest['data']['tensor_sha256']) == 64
    assert result['model_state_sha256_after'] == manifest['model_state']['sha256']
    assert result['status'] == 'complete'
    profile = result['profile']
    ordinary = profile['ordinary_timing']
    assert ordinary['warmup_discarded'] == 1
    assert len(ordinary['host_ms']) == 2
    assert ordinary['host_p10_ms'] <= ordinary['host_median_ms'] <= ordinary['host_p90_ms']
    assert ordinary['cuda_event_ms'] == []
    assert profile['mfu']['needed'] is None
    assert 'CUDA' in profile['mfu']['reason']
    assert set(profile['synchronized_phase_timing']['seconds']) == {
        'clean', 'direct', 'embedding', 'hubs', 'head', 'local', 'scalars',
    }
    assert profile['profiler']['phases']['clean']['count'] == 2
    assert profile['profiler']['sdpa']['operators']['aten::scaled_dot_product_attention'] > 0
    assert 'Self CPU' in (output / result['artifacts']['table']).read_text()
    events = read_json(output / result['artifacts']['trace'])['traceEvents']
    assert sum(event.get('name') == 'dust/clean' for event in events) == 2


def test_profile_dry_run_leaves_no_output(tmp_path):
    output = tmp_path / 'plan'
    run = cli('profile', '--output', output, '--dry-run')
    assert run.returncode == 0, run.stderr
    plan = json.loads(run.stdout)
    assert plan['config']['sequence_len'] == 2048
    assert plan['recipes'][0]['population'] == 256
    assert plan['recipes'][0]['analytic_cost']['flops_total_executed'] > 0
    assert plan['options'] == {'warmup': 3, 'repeats': 5, 'active': 1}
    assert not output.exists()


def test_profile_refuses_nonempty_output(tmp_path):
    marker = tmp_path / 'manifest.json'
    marker.write_text('previous run')
    run = cli('profile', '--output', tmp_path, '--toy')
    assert run.returncode != 0
    assert 'nonempty' in run.stderr
    assert marker.read_text() == 'previous run'
    assert list(tmp_path.iterdir()) == [marker]


@pytest.mark.parametrize('args', [
    ('--repeats', 0), ('--active', 0), ('--warmup', -1), ('--seed', -1),
    ('--populations', 256), ('--toy', '--data-dir', 'somewhere'),
])
def test_profile_rejects_invalid_options_before_creating_output(tmp_path, args):
    output = tmp_path / 'invalid'
    run = cli('profile', '--output', output, *args)
    assert run.returncode != 0
    assert 'error:' in run.stderr
    assert not output.exists()
