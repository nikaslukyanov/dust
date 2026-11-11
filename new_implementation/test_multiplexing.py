"""Checks of structured multiplexing (Track 3): the codes, the decoder, the estimator path, the wiring.

The code and decoder are checked on controlled examples with known answers, as the plan asks: a linear
loss (exact recovery) and a quadratic one (which second-order terms survive, and where they land). The
estimator path is checked against invariances it must have (the clean baseline, the chunk layout),
against backprop, and against the FLOPs PyTorch executes.

Run: python -m pytest new_implementation/test_multiplexing.py
"""
import json
import re
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dust  # noqa: E402  (imports after the sys.path shim above)
import model as model_module  # noqa: E402
import multiplex_experiment as experiment  # noqa: E402
import test_accounting as accounting_checks  # noqa: E402
from accounting import analytic_cost  # noqa: E402
from multiplexing import (code_length, codebook, cross_talk, decode, hadamard, sum_free_columns,  # noqa: E402
                          triple_products)
from run_metrics import dust_update_cost  # noqa: E402
from toy import TOY_CONFIG, toy_data, toy_recipe  # noqa: E402

CPU = torch.device('cpu')
DATA = toy_data(steps=8)


def normal(seed):
    generator = torch.Generator().manual_seed(seed)
    return lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.float64)


# ---- the codes -----------------------------------------------------------------------------------

def test_hadamard_is_orthogonal_and_columns_multiply_by_xor():
    for order in (1, 2, 4, 8, 16):
        matrix = hadamard(order)
        assert torch.equal(matrix.T @ matrix, order * torch.eye(order))
        for g in range(order):
            for h in range(order):
                assert torch.equal(matrix[:, g] * matrix[:, h], matrix[:, g ^ h])
    with pytest.raises(ValueError):
        hadamard(6)


def test_sum_free_codes_cancel_everything_up_to_second_order():
    assert [code_length(groups) for groups in (1, 2, 3, 4, 5, 8)] == [2, 4, 8, 8, 16, 16]
    assert sum_free_columns(2) == [1, 2] and sum_free_columns(4) == [1, 2, 4, 7]
    for groups in range(1, 9):
        code = codebook(groups)
        assert code.shape == (code_length(groups), groups)
        assert torch.equal(code.sum(0), torch.zeros(groups)), 'zero column sums reject the baseline'
        assert torch.equal(cross_talk(code), torch.eye(groups)), 'no first-order leakage'
        assert not triple_products(code).any(), 'no second-order cross term reaches any group'
    # Negative control: all three non-constant columns of H4 are orthogonal and balanced, but 1 XOR 2 = 3,
    # so the cross-curvature of groups 0 and 1 reaches group 2 at full strength.
    aliased = triple_products(codebook(3, columns=[1, 2, 3]))
    assert aliased[2, 0, 1] == 1 and torch.equal(cross_talk(codebook(3, columns=[1, 2, 3])), torch.eye(3))


def test_rademacher_codes_are_orthogonal_only_on_average():
    draw = normal(0)
    codes = [codebook(2, 'rademacher', draw) for _ in range(400)]
    assert all(set(code.unique().tolist()) <= {-1.0, 1.0} and code.shape == (4, 2) for code in codes)
    leak = torch.stack([cross_talk(code)[0, 1] for code in codes])
    assert leak.abs().max() == 1, 'some sequences give both groups the same signs'
    assert abs(leak.mean()) < 0.1, 'the leakage averages out over redraws'
    with pytest.raises(ValueError):
        codebook(2, 'rademacher')
    with pytest.raises(ValueError):
        codebook(2, 'walsh')


# ---- the decoder, on controlled losses -----------------------------------------------------------

def coded_losses(code, fields, loss, sigma):
    """Loss of every pass of one code sequence: pass m perturbs group g by sigma * code[m, g] * fields[g]."""
    return torch.stack([loss([sigma * sign * field for sign, field in zip(row, fields)]) for row in code.double()])


def test_linear_loss_is_decoded_exactly():
    """L = L0 + sum_g w_g . y_g: every group's directional loss change comes back exactly, whatever the
    other groups do, and whatever the baseline."""
    draw, sigma = normal(1), 0.2
    for groups in (2, 3, 5):
        weights = [draw(7) for _ in range(groups)]
        fields = [draw(7) for _ in range(groups)]
        loss = lambda deltas: 3.7 + sum(w @ d for w, d in zip(weights, deltas))
        expected = torch.stack([sigma * w @ a for w, a in zip(weights, fields)])
        recovered = decode(coded_losses(codebook(groups), fields, loss, sigma), codebook(groups))
        assert torch.allclose(recovered, expected, atol=1e-12)
        # Negative control: without the 1 / M and the right columns the answer is wrong.
        other = codebook(groups)[:, list(reversed(range(groups)))]
        assert not torch.allclose(decode(coded_losses(codebook(groups), fields, loss, sigma), other), expected)


def test_quadratic_loss_shows_which_second_order_terms_survive():
    """L = L0 + w . y + y^T H y / 2 over the stacked groups, H with self and cross blocks."""
    draw, sigma, width = normal(2), 0.3, 5
    weights = [draw(width) for _ in range(3)]
    fields = [draw(width) for _ in range(3)]
    hessian = draw(3 * width, 3 * width)
    hessian = hessian + hessian.T

    def loss(deltas):
        stacked = torch.cat(deltas)
        return 1.0 + torch.cat(weights) @ stacked + stacked @ hessian @ stacked / 2

    first_order = torch.stack([sigma * w @ a for w, a in zip(weights, fields)])
    block = lambda g, h: hessian[g * width:(g + 1) * width, h * width:(h + 1) * width]
    cross = lambda g, h: sigma ** 2 * fields[g] @ block(g, h) @ fields[h]

    # Sum-free columns: exact, although every self- and cross-curvature term is present.
    clean = codebook(3)
    assert torch.allclose(decode(coded_losses(clean, fields, loss, sigma), clean), first_order, atol=1e-10)

    # Columns 1, 2, 3 of H4 (1 XOR 2 = 3, and likewise for the other two pairs): each group receives
    # the cross-curvature of the other two, at full strength.
    aliased = codebook(3, columns=[1, 2, 3])
    leaked = decode(coded_losses(aliased, fields, loss, sigma), aliased) - first_order
    assert torch.allclose(leaked, torch.stack([cross(1, 2), cross(0, 2), cross(0, 1)]), atol=1e-10)
    assert leaked.abs().min() > 1e-3

    # One group alone, one-sided (as an uncentered DUST draw): its own curvature stays in.
    alone = loss([sigma * fields[0], torch.zeros(width, dtype=torch.float64), torch.zeros(width, dtype=torch.float64)])
    assert torch.allclose(alone - 1.0 - first_order[0], sigma ** 2 * fields[0] @ block(0, 0) @ fields[0] / 2, atol=1e-10)


def test_decode_centers_codes_that_do_not_sum_to_zero():
    """A constant added to every loss of a sequence (the baseline) must not reach any group."""
    code = torch.tensor([[1., 1.], [1., -1.], [1., 1.], [-1., 1.]])  # column sums 2 and 2
    losses = normal(3)(4, 6)
    assert torch.allclose(decode(losses + 9.0, code), decode(losses, code), atol=1e-12)
    raw = torch.einsum('mg,mt->gt', code.double(), losses + 9.0) / 4
    assert not torch.allclose(raw, decode(losses, code), atol=1e-3), 'negative control: uncentered decoding leaks it'


# ---- the estimator path --------------------------------------------------------------------------

def held_out_batch():
    rows = DATA[1][:8]
    return rows[:, :-1].contiguous(), rows[:, 1:].contiguous()


def coded_groups(layer, sigma=0.2):
    groups = experiment.block_groups(layer)
    return [(groups['writers'], sigma), (groups['hidden'], sigma)]


def multiplexed(model, kind, passes=8, nrep=2, seed=5, layer=1, shift=0.0, suffix=True):
    estimator = dust.DUST(model, experiment.experiment_recipe(nrep), seed, suffix=suffix)
    try:
        estimator.capture(*held_out_batch())
        estimator.clean_loss = estimator.clean_loss + shift
        return estimator.multiplexed_errors(coded_groups(layer), passes, layer, kind)
    finally:
        estimator.close()


def close(a, b, tolerance=1e-4):
    return all((a[site] - b[site]).abs().max() <= tolerance * b[site].abs().max() for site in a)


@pytest.mark.parametrize('kind', ['hadamard', 'rademacher'])
def test_estimate_ignores_the_baseline_and_is_reproducible(kind):
    """Reward centering, done by the code: shifting the clean loss changes nothing. Same seed, same answer."""
    model = dust.make_model(TOY_CONFIG, CPU, 0)
    base = multiplexed(model, kind)
    assert set(base) == {site for sites, _ in coded_groups(1) for site in sites}
    assert all(torch.equal(base[site], again) for site, again in multiplexed(model, kind).items())
    assert close(multiplexed(model, kind, shift=3.0), base)
    assert not close(multiplexed(model, kind, seed=6), base), 'negative control: another seed differs'


def test_code_sequences_do_not_depend_on_the_chunk_size():
    """A sequence of 4 passes may span two forward passes (nrep 2), fill one (4) or share one (8)."""
    model = dust.make_model(TOY_CONFIG, CPU, 0)
    reference = multiplexed(model, 'hadamard', passes=8, nrep=2)
    for nrep in (1, 4, 8):
        assert close(multiplexed(model, 'hadamard', passes=8, nrep=nrep), reference)
    assert close(multiplexed(model, 'hadamard', passes=8, suffix=False), reference), 'full replay agrees'


def test_estimator_refuses_what_cannot_be_multiplexed():
    model = dust.make_model(TOY_CONFIG, CPU, 0)
    estimator = dust.DUST(model, experiment.experiment_recipe(2), 0)
    try:
        estimator.capture(*held_out_batch())
        with pytest.raises(ValueError, match='code sequences'):
            estimator.multiplexed_errors(coded_groups(1), 6, 1)  # not a multiple of the code length 4
        distant = [(['transformer.h.0.mlp.c_fc'], 0.2), (['transformer.h.3.mlp.c_fc'], 0.2)]
        with pytest.raises(ValueError, match='start block'):
            estimator.multiplexed_errors(distant, 8, 0)
        with pytest.raises(ValueError, match='start block'):
            estimator.multiplexed_errors(coded_groups(1), 8, 0)  # would rerun a block it does not need
        adjacent = [(['transformer.h.1.mlp.c_fc'], 0.2), (['transformer.h.2.mlp.c_fc'], 0.2)]
        assert set(estimator.multiplexed_errors(adjacent, 8, 1)) == {site for sites, _ in adjacent for site in sites}
    finally:
        estimator.close()
    with pytest.raises(ValueError, match='multiplex'):
        dust.DUST(model, toy_recipe(scale=2), multiplex='walsh')
    uneven = toy_recipe(scale=2)
    uneven.hidden[0] = 4
    estimator = dust.DUST(model, uneven, multiplex='hadamard')
    try:
        with pytest.raises(ValueError, match='equal writer and hidden'):
            estimator.estimate(*held_out_batch())
    finally:
        estimator.close()


def test_multiplexed_estimate_points_along_the_gradient():
    """Against backprop, over seeds: the coded estimate is a gradient estimate; exactly orthogonal codes
    beat random signs; and separate evaluation with the same draws per group (twice the forward passes)
    beats both -- multiplexing saves passes, it does not add information per draw."""
    model = dust.make_model(TOY_CONFIG, CPU, 0)
    report = experiment.compare(model, experiment.experiment_recipe(2), *held_out_batch(), layer=0, passes=16,
                                seeds=range(12), methods=('separate', 'hadamard', 'rademacher'))
    assert report['separate']['forwards'] == 32 and report['hadamard']['forwards'] == 16
    for group in ('writers', 'hidden'):
        score = {method: report[method]['groups'][group] for method in report}
        assert score['hadamard']['mean_cosine'] > 0.3
        assert score['separate']['cosine'] > score['hadamard']['cosine'] > score['rademacher']['cosine'] > 0


# ---- accounting and the training loop ------------------------------------------------------------

def coded_recipe():
    return dust.Recipe(writers=[8, 4, 4, 4], hidden=[8, 4, 4, 4], hubs=[2, 2, 2, 4], embedding=2, head=32,
                       local=dict(q=2, v=2, gate=2, k=2, ve=2), nrep=2, lr=0.1, momentum=0.9)


@pytest.mark.parametrize('suffix', [True, False], ids=['cached-suffix', 'full-replay'])
def test_accounting_matches_the_flops_a_multiplexed_estimate_runs(monkeypatch, suffix):
    monkeypatch.setattr(model_module, 'attention', accounting_checks.matmul_attention)
    config = accounting_checks.FULL_WINDOWS
    model = dust.make_model(config, CPU, seed=0)
    estimator = dust.DUST(model, coded_recipe(), suffix=suffix, multiplex='hadamard')
    try:
        with FlopCounterMode(display=False) as counter:
            estimator.estimate(*accounting_checks.tokens(config, 2))
    finally:
        estimator.close()
    counts = {name: sum(ops.values()) for name, ops in counter.get_flop_counts().items()}
    blocks = sum(v for k, v in counts.items() if re.fullmatch(r'GPT\.transformer\.h\.\d+|GPT\.lm_head', k))
    executed = blocks - sum(v for k, v in counts.items() if k.endswith('.ve_gate'))

    shared = analytic_cost(coded_recipe(), config, batch=2, suffix=suffix, multiplex=True)
    separate = analytic_cost(coded_recipe(), config, batch=2, suffix=suffix)
    assert executed == accounting_checks.predicted_forward_flops(shared)
    assert executed < accounting_checks.predicted_forward_flops(separate), 'negative control'
    # Both groups' draws still count; only the forward evaluations are shared.
    assert shared.draws == separate.draws and shared.population_K == separate.population_K
    assert separate.direct_forwards - shared.direct_forwards == sum(coded_recipe().hidden)
    assert separate.direct_forwards == separate.population_K


def run(output, steps=4, **kwargs):
    training, validation, testing = DATA
    return dust.train(TOY_CONFIG, toy_recipe(scale=2), training, validation, testing, output, steps=steps,
                      eval_interval=2, seed=42, device=CPU, echo=lambda _: None, **kwargs)


def records(output):
    return [json.loads(line) for line in (Path(output) / 'metrics.jsonl').read_text().splitlines()]


def test_multiplexed_training_run(tmp_path):
    result = run(tmp_path / 'coded', steps=6, checkpoint_interval=3, multiplex='hadamard')
    run(tmp_path / 'separate', steps=2)
    coded, separate = records(tmp_path / 'coded'), records(tmp_path / 'separate')
    settings = toy_recipe(scale=2)

    # Charged what it ran: shared passes once, both groups' draws.
    cost = dust_update_cost(settings, TOY_CONFIG, batch=8, multiplex=True)
    assert [row['flops'] for row in coded[1:]] == pytest.approx([row['step'] * cost['flops'] for row in coded[1:]])
    assert coded[1]['flops'] < separate[1]['flops']
    assert coded[1]['draws'] == separate[1]['draws'] and coded[1]['population'] == separate[1]['population']
    assert separate[1]['direct_forwards'] - coded[1]['direct_forwards'] == sum(settings.hidden)
    assert result['best_val'] < result['history'][0]['val_loss'] - 1, 'it learns'

    # train() is the plain loop of the multiplexed step.
    model = dust.make_model(TOY_CONFIG, CPU, 42)
    optimizer = dust.make_optimizer(model, settings)
    estimator = dust.DUST(model, settings, 42, multiplex='hadamard')
    for step in range(1, 7):
        batch = DATA[0][(step - 1) * 8:step * 8]
        estimator.step(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), optimizer)
    estimator.close()
    trained = torch.load(tmp_path / 'coded' / 'state.pt', weights_only=False)['model']
    assert all(torch.equal(trained[name], value.detach().cpu()) for name, value in model.state_dict().items())

    with pytest.raises(ValueError, match='multiplex'):
        run(tmp_path / 'coded', steps=6, resume=True)  # a separate-evaluation run cannot continue it
    with pytest.raises(ValueError, match='variant D'):
        run(tmp_path / 'both', multiplex='hadamard', allocation={})
