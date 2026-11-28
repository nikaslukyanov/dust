import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def cli(*args, script='gpu_diagnostics.py', env=None):
    return subprocess.run(
        [sys.executable, str(ROOT / 'scripts' / script), *map(str, args)],
        env=env,
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
    with (output / result['artifacts']['trace']).open() as trace:
        assert sum(bool(re.search(r'"name"\s*:\s*"dust/clean"', line)) for line in trace) == 2
    assert profile['profiler']['record_shapes'] is False
    assert profile['profiler']['profile_memory'] is False


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


def test_quality_toy_writes_raw_samples_and_summary(tmp_path):
    output = tmp_path / 'quality'
    run = cli('quality', '--output', output, '--toy', '--scales', 1, 2, '--seeds', 10, 20)
    assert run.returncode == 0, run.stderr
    manifest = read_json(output / 'manifest.json')
    result = read_json(output / 'result.json')
    assert [entry['scale'] for entry in manifest['recipes']] == [1, 2]
    assert manifest['execution']['float32_matmul_precision'] == 'high'
    assert result['status'] == 'complete'
    assert result['model_state_sha256_after'] == manifest['model_state']['sha256']
    quality = result['quality']
    assert quality['reference']['float32_matmul_precision'] == 'highest'
    assert quality['reference']['chunk'] == 1
    assert quality['reference']['loss'] > 0
    assert quality['timing']['warmup'] == 0
    assert [row['scale'] for row in quality['populations']] == [1, 2]
    for row in quality['populations']:
        assert row['status'] == 'complete'
        assert [sample['seed'] for sample in row['samples']] == [10, 20]
        for sample in row['samples']:
            assert sample['clean_loss'] > 0
            assert sample['host_ms'] > 0
            assert sample['cuda_event_ms'] is None
            assert set(sample['per_group']['head']) == {'cosine', 'rel_norm', 'ref_norm', 'est_norm'}
        assert row['summary']['per_group']['head']['cosine']['sample_sd'] > 0


def test_quality_summary_uses_sample_sd_over_seeds():
    from gpu_diagnostics import summarize_quality

    samples = [
        dict(seed=0, clean_loss=3, per_group={'head': dict(cosine=0.25, rel_norm=1, ref_norm=4, est_norm=4)}),
        dict(seed=1, clean_loss=5, per_group={'head': dict(cosine=0.50, rel_norm=2, ref_norm=4, est_norm=8)}),
        dict(seed=2, clean_loss=7, per_group={'head': dict(cosine=0.75, rel_norm=3, ref_norm=4, est_norm=12)}),
    ]
    assert summarize_quality(samples) == dict(
        clean_loss=dict(mean=5, sample_sd=2),
        per_group={'head': dict(
            cosine=dict(mean=0.5, sample_sd=0.25), rel_norm=dict(mean=2, sample_sd=1),
            ref_norm=dict(mean=4, sample_sd=0), est_norm=dict(mean=8, sample_sd=4),
        )},
    )


def test_quality_oom_reports_unavailable_and_removes_hooks(tmp_path, monkeypatch):
    import torch
    import gpu_diagnostics as diagnostics

    output = tmp_path / 'oom'
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    estimators = []
    reference_calls = []

    def reference(model, x, y, *, precision, chunk):
        assert torch.get_float32_matmul_precision() == 'highest'
        assert precision == 'fp32'
        reference_calls.append(chunk)
        return {'lm_head.weight': torch.tensor([1., 2.])}, 2.0

    def estimate(estimator, x, y, *, seed):
        assert torch.get_float32_matmul_precision() == 'high'
        estimators.append(estimator)
        if estimator.settings.embedding == 2:
            raise torch.cuda.OutOfMemoryError('test OOM')
        return {'lm_head.weight': torch.tensor([1., 2.])}, 3.0

    monkeypatch.setattr(diagnostics, 'reference_gradients', reference)
    monkeypatch.setattr(diagnostics, 'estimate_gradients', estimate)
    assert diagnostics.main(['quality', '--toy', '--output', str(output), '--scales', '1', '2']) == 1
    result = read_json(output / 'result.json')
    assert result['status'] == 'partial'
    first, second = result['quality']['populations']
    assert first['status'] == 'unavailable'
    assert first['recipe']['nrep'] == 2
    assert first['error'] == {'type': 'OutOfMemoryError', 'message': 'test OOM'}
    assert second['status'] == 'complete'
    assert len(second['samples']) == 3
    assert reference_calls == [1]
    assert all(not module._forward_hooks and not module._forward_pre_hooks
               for estimator in estimators for module in estimator.modules.values())


def test_checkpoint_loading_is_strict(tmp_path):
    import torch

    checkpoint = tmp_path / 'bad checkpoint.pt'
    torch.save({}, checkpoint)
    output = tmp_path / 'strict'
    run = cli('quality', '--output', output, '--toy', '--checkpoint', checkpoint)
    assert run.returncode == 1
    assert 'Missing key(s)' in run.stderr
    result = read_json(output / 'result.json')
    assert result['status'] == 'failed'
    manifest = read_json(output / 'manifest.json')
    assert manifest['model_state']['initialization'] == 'checkpoint'
    assert len(manifest['model_state']['checkpoint']['sha256']) == 64


@pytest.mark.parametrize('opts, expected', [
    (['--toy', '--populations', '256'], '--populations'),
    (['--scales', '1'], '--scales'),
    (['--populations', '512'], '--populations'),
    (['--seeds', '0'], '--seeds'),
    (['--seeds', '0', '0'], '--seeds'),
    (['--reference-chunk', '9'], '--reference-chunk'),
])
def test_quality_rejects_mixed_or_unsupported_options(tmp_path, capsys, opts, expected):
    from gpu_diagnostics import parse_args

    output = tmp_path / 'invalid'
    with pytest.raises(SystemExit) as error:
        parse_args(['quality', '--output', str(output), *opts])
    assert error.value.code == 2
    assert expected in capsys.readouterr().err
    assert not output.exists()


def test_launcher_dry_run_writes_nothing_and_preserves_arguments(tmp_path):
    output = tmp_path / 'profile job'
    data = tmp_path / 'input data'
    checkpoint = tmp_path / 'checkpoint with spaces.pt'
    sbatch = tmp_path / 'sbatch'
    submission = tmp_path / 'submission marker'
    sbatch.write_text(f'#!/bin/sh\ntouch {shlex.quote(str(submission))}\n')
    sbatch.chmod(0o755)
    run = cli('quality', '--output', output, '--data-dir', data, '--checkpoint', checkpoint,
              '--populations', 256, 1024, '--seeds', 4, 9, '--reference-chunk', 2,
              '--dry-run', script='launch_diagnostics.py',
              env={**os.environ, 'PATH': str(tmp_path) + os.pathsep + os.environ['PATH']})
    assert run.returncode == 0, run.stderr
    *plan_lines, sbatch_line, inner_line = run.stdout.splitlines()
    plan = json.loads('\n'.join(plan_lines))
    assert [entry['population'] for entry in plan['recipes']] == [256, 1024]
    submit_args, inner_args = shlex.split(sbatch_line), shlex.split(inner_line)
    assert submit_args[0] == 'sbatch'
    assert str(data) in submit_args
    assert str(output / 'source/scripts/diagnostics_gpu.sbatch') in submit_args
    expected = [
        'quality', '--output', str(output / 'diagnostics'), '--seed', '42', '--checkpoint',
        str(checkpoint), '--populations', '256', '1024', '--seeds', '4', '9', '--reference-chunk', '2',
    ]
    assert submit_args[-len(expected):] == expected
    assert inner_args[1:] == ['scripts/gpu_diagnostics.py', *expected, '--data-dir', '$scratch']
    assert not output.exists()
    assert not submission.exists()
