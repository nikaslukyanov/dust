"""The upstream reference: `dust.py --full-replay` is the original public DUST code, number for number.

The baseline of the study is the code as the authors published it (commit b20f7c0, the last upstream
commit before this fork's changes). Rather than keep a second copy that cannot log our metrics, the
fork's DUST(suffix=False) must reproduce it exactly: this test checks the original dust.py and model.py
out of git history, trains a tiny model for a few steps with each, and compares every weight and loss.
On CPU in fp32 the match is bit-for-bit, for full replay and for cached-suffix execution alike, so a
`--full-replay` run *is* the original code with our diagnostics attached.

Skipped when the repository history is not available (a shallow clone or an exported tree).
Run: python -m pytest new_implementation/test_upstream.py
"""
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = 'b20f7c0'

RUNNER = '''
import sys, torch
code, mode, out = sys.argv[1:4]
sys.path.insert(0, code)
import dust
from model import GPTConfig
config = GPTConfig(sequence_len=64, n_layer=4, n_head=4, n_kv_head=4, n_embd=64)
settings = dust.Recipe(writers=[4, 4, 2, 2], hidden=[4, 4, 2, 2], hubs=[2, 2, 2, 4], embedding=2, head=64,
                       local=dict(q=2, v=2, gate=2, k=2, ve=2), nrep=2, lr=0.25, momentum=0.95)
data = torch.randint(0, 64, (24, 65), generator=torch.Generator().manual_seed(0))
model = dust.make_model(config, torch.device('cpu'), 42)
optimizer = dust.make_optimizer(model, settings)
options = {} if mode == 'upstream' else dict(suffix=mode == 'suffix')
estimator = dust.DUST(model, settings, 42, **options)
losses = []
for step in range(3):
    rows = data[step * 8:(step + 1) * 8]
    losses.append(estimator.step(rows[:, :-1].contiguous(), rows[:, 1:].contiguous(), optimizer))
torch.save(dict(weights=model.state_dict(), losses=losses), out)
'''


def git_show(path):
    return subprocess.run(['git', 'show', f'{UPSTREAM}:{path}'], cwd=ROOT, capture_output=True, text=True)


@pytest.fixture(scope='module')
def runs(tmp_path_factory):
    folder = tmp_path_factory.mktemp('upstream')
    original = folder / 'original'
    original.mkdir()
    for name in ('dust.py', 'model.py'):
        shown = git_show(name)
        if shown.returncode:
            pytest.skip(f'upstream commit {UPSTREAM} is not in this checkout')
        (original / name).write_text(shown.stdout)
    runner = folder / 'runner.py'
    runner.write_text(RUNNER)
    results = {}
    for mode, code in (('upstream', original), ('full', ROOT), ('suffix', ROOT)):
        out = folder / f'{mode}.pt'
        done = subprocess.run([sys.executable, str(runner), str(code), mode, str(out)], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        results[mode] = torch.load(out)
    return results


@pytest.mark.parametrize('mode', ['full', 'suffix'])
def test_training_matches_the_original_code(runs, mode):
    original, ours = runs['upstream'], runs[mode]
    assert ours['losses'] == original['losses']
    assert original['losses'][0] > original['losses'][-1]  # it trained, so the weights below are not the init
    for name, weight in original['weights'].items():
        assert torch.equal(ours['weights'][name], weight), name
