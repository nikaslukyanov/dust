"""Checks for tracking.py (the optional W&B mirror), on CPU with the toy setup.

W&B itself is replaced by a fake that keeps one timeline per run id and, like W&B, ignores a step it has
already seen. The truth is metrics.jsonl / result.json: what reaches W&B must equal them, also across a
crash and resume. One test drives the real wandb package in offline mode when it is installed.

Run: python -m pytest new_implementation/test_tracking.py
"""
import json
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'baselines'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import backprop  # noqa: E402  (imports after the sys.path shim above)
import dust  # noqa: E402
from toy import TOY_CONFIG, toy_data, toy_recipe  # noqa: E402
from tracking import ID_FILE, flatten  # noqa: E402

CPU = torch.device('cpu')
DATA = toy_data(steps=24)
TIMING = {'train_seconds', 'gpu_hours', 'elapsed_seconds', 'step_host_ms', 'step_gpu_ms', 'peak_memory_gb'}


class FakeWandb(types.ModuleType):
    """The slice of the wandb API tracking.py uses, with W&B's rule that a repeated step is dropped."""

    def __init__(self):
        super().__init__('wandb')
        self.runs, self.inits = {}, []

    def init(self, *, id, resume, config, **options):
        self.inits.append(dict(id=id, resume=resume, config=config, **options))
        run = self.runs.setdefault(id, types.SimpleNamespace(timeline={}, summary={}, finished=0))
        run.define_metric = lambda *args, **kwargs: None

        def log(data, step):
            if not run.timeline or step > max(run.timeline):
                run.timeline[step] = data

        def finish():
            run.finished += 1

        run.log, run.finish = log, finish
        return run


@pytest.fixture
def fake(monkeypatch):
    module = FakeWandb()
    monkeypatch.setitem(sys.modules, 'wandb', module)
    return module


def run(output, steps=6, **kwargs):
    training, validation, testing = DATA
    return dust.train(TOY_CONFIG, toy_recipe(), training, validation, testing, output, steps=steps,
                      eval_interval=2, seed=42, device=CPU, echo=lambda _: None, **kwargs)


def records(output):
    return [json.loads(line) for line in (Path(output) / 'metrics.jsonl').read_text().splitlines()]


def untimed(rows):
    return [{k: v for k, v in row.items() if k.split('/')[0] not in TIMING} for row in rows]


def test_flatten():
    assert flatten(dict(step=3, probe=dict(cosine=0.5, per_group=dict(head=dict(cosine=0.9))))) == \
        {'step': 3, 'probe/cosine': 0.5, 'probe/per_group/head/cosine': 0.9}
    assert flatten(dict(allocation=dict(writers=[4, 2], embedding=2))) == \
        {'allocation/writers/0': 4, 'allocation/writers/1': 2, 'allocation/embedding': 2}


@pytest.mark.parametrize('method', [dict(allocation=dict(interval=2, rate=1.0)), dict(multiplex='hadamard')],
                         ids=['B-adaptive', 'C-multiplex'])
def test_methods_b_and_c_mirror_metrics(fake, tmp_path, method):
    """B's allocation records and C's cost fields reach W&B as flat numeric keys, nothing dropped."""
    training, validation, testing = DATA
    dust.train(TOY_CONFIG, toy_recipe(scale=2), training, validation, testing, tmp_path / 'run', steps=6,
               eval_interval=2, seed=42, device=CPU, echo=lambda _: None, wandb=dict(project='p'), **method)
    [init] = fake.inits
    timeline = list(fake.runs[init['id']].timeline.values())
    assert timeline == [flatten(r) for r in records(tmp_path / 'run')]
    for row in timeline:
        assert all(isinstance(v, (int, float)) for v in row.values()), row
    if 'allocation' in method:
        assert any('allocation/writers/0' in row for row in timeline)
        assert any(key.startswith('allocation_noise/') for row in timeline for key in row)
    else:
        assert all('direct_forwards' in row for row in timeline[1:])


def test_dust_run_mirrors_metrics_and_result(fake, tmp_path):
    result = run(tmp_path / 'run', probe_interval=3, metadata=dict(tokens='toy'),
                 wandb=dict(project='p', group='A'))
    [init] = fake.inits
    assert (init['project'], init['group'], init['name'], init['config']) == ('p', 'A', 'run', dict(tokens='toy'))
    timeline = fake.runs[init['id']].timeline
    assert list(timeline.values()) == [flatten(r) for r in records(tmp_path / 'run')]
    assert 'probe/mean_group_cosine' in timeline[3]
    summary = fake.runs[init['id']].summary
    assert (summary['best_val'], summary['test_at_best_val']) == (result['best_val'], result['test_at_best_val'])
    assert 'history' not in summary and fake.runs[init['id']].finished == 1


def test_no_wandb_option_never_touches_wandb(fake, tmp_path):
    """Negative control: the default run neither imports nor initialises W&B, nor leaves an id file."""
    run(tmp_path / 'run', steps=2)
    assert fake.inits == [] and not (tmp_path / 'run' / ID_FILE).exists()


def test_resume_continues_the_same_wandb_run(fake, tmp_path, monkeypatch):
    run(tmp_path / 'straight', checkpoint_interval=3, wandb=dict(project='p'))
    evaluate, calls = dust.evaluate, []

    def crash_at_step_six(*args, **kwargs):  # evaluations at steps 0, 1, 2, 4, 6: the 5th is step 6
        calls.append(1)
        if len(calls) == 5:
            raise KeyboardInterrupt('simulated preemption')
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(dust, 'evaluate', crash_at_step_six)
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path / 'crashed', checkpoint_interval=3, wandb=dict(project='p'))
    monkeypatch.setattr(dust, 'evaluate', evaluate)
    run(tmp_path / 'crashed', checkpoint_interval=3, resume=True, wandb=dict(project='p'))

    straight, first, again = (i['id'] for i in fake.inits)
    assert first == again != straight  # one W&B run per training run, kept across the resume
    assert (tmp_path / 'crashed' / ID_FILE).read_text().strip() == first
    # Steps 4 and 5 ran twice; W&B keeps the first copy, which an exact resume makes identical.
    assert untimed(fake.runs[first].timeline.values()) == untimed(fake.runs[straight].timeline.values())
    assert fake.runs[first].finished == 1


def test_backprop_run_mirrors_metrics(fake, tmp_path):
    training, validation, testing = DATA
    backprop.train(TOY_CONFIG, training, validation, testing, tmp_path / 'bp', steps=4, eval_interval=2,
                   lr=0.1, momentum=0.9, device=CPU, echo=lambda _: None, wandb=dict(project='p'))
    [init] = fake.inits
    assert list(fake.runs[init['id']].timeline.values()) == [flatten(r) for r in records(tmp_path / 'bp')]


def test_real_wandb_offline(tmp_path, monkeypatch):
    """The real package accepts everything tracking.py sends (offline: no account or network)."""
    pytest.importorskip('wandb')
    monkeypatch.setenv('WANDB_SILENT', 'true')
    run(tmp_path / 'run', steps=3, probe_interval=3, metadata=dict(tokens='toy'),
        wandb=dict(project='dust-test', mode='offline'))
    run_id = (tmp_path / 'run' / ID_FILE).read_text().strip()
    assert list((tmp_path / 'run' / 'wandb').glob(f'offline-run-*-{run_id}/run-{run_id}.wandb'))
