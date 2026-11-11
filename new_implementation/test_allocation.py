"""Checks of adaptive allocation (Track 2): the allocation math, the split-half measurement, the wiring.

Each check compares against an independent truth: hand-solved allocations, a brute-force search, the
estimator's variance over seeds (what the split-half number claims to measure), accounting's FLOPs
(the per-draw costs), a fixed-allocation run (measuring changes nothing), and an uninterrupted run
(resume is exact).

Run: python -m pytest new_implementation/test_allocation.py
"""
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dust  # noqa: E402  (imports after the sys.path shim above)
from accounting import analytic_cost  # noqa: E402
from allocation import Allocator, apportion, draw_costs, neyman, unit_counts, unit_of, with_counts  # noqa: E402
from run_metrics import dust_update_cost  # noqa: E402
from toy import TOY_CONFIG, toy_data, toy_recipe  # noqa: E402

CPU = torch.device('cpu')
DATA = toy_data(steps=8)
BUDGET = 52  # reallocated draws in recipe(): writers 24 + hidden 24 + embedding 4


def recipe():
    """The toy recipe with every reallocated unit at two chunks or more (nrep = 2, so 4 draws)."""
    return toy_recipe(scale=2)


def parameter_of(unit):
    if unit == 'embedding':
        return 'transformer.wte.weight'
    group, layer = unit.split('.')
    return f"transformer.h.{layer}.{'attn.c_proj' if group == 'writers' else 'mlp.c_fc'}.weight"


def statistics(variances, draws=8):
    """What DUST.split_statistics would report for units with these per-draw variances."""
    return {parameter_of(unit): (variance * draws, draws) for unit, variance in variances.items()}


def flat(recipe_, value=1.0, **overrides):
    return {**{unit: value for unit in unit_counts(recipe_)}, **overrides}


def ones(units):
    return {unit: 1 for unit in units}


# ---- the allocation math -------------------------------------------------------------------------

def test_units_cover_exactly_the_reallocated_parameters():
    model = dust.make_model(TOY_CONFIG, CPU, 0)
    units = {unit_of(name) for name, _ in model.named_parameters()} - {None}
    assert units == set(unit_counts(recipe()))
    assert unit_of('transformer.h.2.attn.c_proj.weight') == unit_of('transformer.h.2.mlp.c_proj.weight') == 'writers.2'
    assert unit_of('transformer.h.1.attn.c_q.weight') is None and unit_of('lm_head.weight') is None
    assert unit_counts(with_counts(recipe(), unit_counts(recipe()))) == unit_counts(recipe())


def test_neyman_known_answers():
    assert neyman(dict(a=1.0, b=3.0), 40, 0, ones('ab')) == pytest.approx(dict(a=10, b=30))
    # Equal noise, b four times the cost: K is proportional to 1 / sqrt(cost), and a + 4 b = 60.
    assert neyman(dict(a=1.0, b=1.0), 60, 0, dict(a=1, b=4)) == pytest.approx(dict(a=20, b=10))
    # The floor pins b and c; a takes what is left, not its unconstrained 40 * 100 / 102.
    assert neyman(dict(a=100.0, b=1.0, c=1.0), 40, 4, ones('abc')) == pytest.approx(dict(a=32, b=4, c=4))
    assert neyman(dict(a=0.0, b=0.0), 40, 4, ones('ab')) == pytest.approx(dict(a=20, b=20))


def test_neyman_beats_random_allocations_of_the_same_budget():
    rng = random.Random(0)
    units = 'abcde'
    sigmas = {u: rng.uniform(0.1, 5) for u in units}
    costs = {u: rng.uniform(1, 4) for u in units}
    budget = 1000.0
    error = lambda draws: sum(sigmas[u] ** 2 / draws[u] for u in units)
    best = neyman(sigmas, budget, 0, costs)
    assert sum(costs[u] * best[u] for u in units) == pytest.approx(budget)
    for _ in range(200):
        other = {u: rng.uniform(0.1, 1) for u in units}
        scale = budget / sum(costs[u] * other[u] for u in units)
        assert error({u: scale * k for u, k in other.items()}) >= error(best) * (1 - 1e-9)
    # Negative control: allocating by variance (sigma squared) instead of sigma is measurably worse.
    wrong = neyman({u: s ** 2 for u, s in sigmas.items()}, budget, 0, costs)
    assert error(wrong) > error(best) * 1.01


def test_apportion_is_largest_remainder():
    # Round down to 6, 4, 2; the two leftover quanta go to the largest remainders (a: 1.5, b: 1.3).
    assert apportion(dict(a=7.5, b=5.3, c=3.2), 16, 2, 2, ones('abc')) == dict(a=8, b=6, c=2)
    # A target a hair under a multiple (floating point) is that multiple, not one quantum less.
    assert apportion(dict(a=4 - 1e-12, b=8.0), 12, 2, 4, ones('ab')) == dict(a=4, b=8)
    # Unequal costs: never above the budget, and nothing more fits afterwards.
    costs = dict(a=3.0, b=1.0)
    counts = apportion(dict(a=5.0, b=5.0), 20.0, 2, 2, costs)
    spent = sum(costs[u] * k for u, k in counts.items())
    assert spent <= 20 and 20 - spent < 2 * min(costs.values())


# ---- the policy ----------------------------------------------------------------------------------

def test_allocator_reaches_the_neyman_allocation():
    """Noise 9 on one unit, 1 on the other eight: their optimum 52/17 = 3.06 is under the floor of 4, so
    they are pinned (32 draws) and the noisy unit gets the remaining 20."""
    allocator = Allocator(recipe(), interval=1, ema=0.0, rate=1.0)
    allocator.observe(statistics(flat(recipe(), **{'writers.0': 81.0})))
    assert allocator.update(1)
    assert allocator.counts == flat(recipe(), 4, **{'writers.0': 20})
    assert allocator.sigmas()['writers.0'] == pytest.approx(9)
    settings = allocator.recipe(recipe())
    assert settings.writers == [20, 4, 4, 4] and settings.hidden == [4, 4, 4, 4] and settings.embedding == 4
    assert settings.hubs == recipe().hubs and settings.head == recipe().head and settings.local == recipe().local


def test_allocator_damping_interval_and_smoothing():
    signal = statistics(flat(recipe(), **{'writers.0': 81.0}))
    half = Allocator(recipe(), interval=1, ema=0.0, rate=0.5)
    half.observe(signal)
    half.update(1)
    # Half way from [8, 8, 4, 4] x 2 + 4 to the optimum above (20, the rest 4).
    assert half.counts == {'writers.0': 14, 'writers.1': 6, 'writers.2': 4, 'writers.3': 4,
                           'hidden.0': 6, 'hidden.1': 6, 'hidden.2': 4, 'hidden.3': 4, 'embedding': 4}

    slow = Allocator(recipe(), interval=2, ema=0.0, rate=1.0)
    slow.observe(signal)
    assert not slow.update(1) and slow.counts == unit_counts(recipe())
    assert slow.update(2)

    never = Allocator(recipe(), interval=0)
    never.observe(signal)
    assert not never.update(1) and not never.update(4)

    smooth = Allocator(recipe(), ema=0.5)
    smooth.observe(statistics({'hidden.0': 4.0}))
    smooth.observe(statistics({'hidden.0': 0.0}))
    assert smooth.variance['hidden.0'] == pytest.approx(2.0)
    assert not smooth.update(4), 'no reallocation until every unit has been measured'


def test_noise_is_per_draw_and_weighted_and_summed_over_a_unit():
    allocator = Allocator(recipe(), ema=0.0, weights={'transformer.wte.weight': 10.0})
    # The same per-draw variance measured from 8 or from 32 draws is the same noise.
    allocator.observe(statistics({'hidden.0': 3.0}, draws=8))
    few = allocator.variance['hidden.0']
    allocator.observe(statistics({'hidden.0': 3.0}, draws=32))
    assert allocator.variance['hidden.0'] == pytest.approx(few) == pytest.approx(3.0)
    # A learning-rate weight of 10 is 100 in variance.
    allocator.observe(statistics({'embedding': 2.0}))
    assert allocator.variance['embedding'] == pytest.approx(200.0)
    # The two writer projections of a block share draws: their variances add.
    allocator.observe({'transformer.h.1.attn.c_proj.weight': (8.0, 8), 'transformer.h.1.mlp.c_proj.weight': (16.0, 8),
                       'transformer.h.1.attn.c_q.weight': (99.0, 8)})
    assert allocator.variance['writers.1'] == pytest.approx(3.0)
    assert set(allocator.variance) == {'hidden.0', 'embedding', 'writers.1'}


def test_allocator_start_and_validation():
    with pytest.raises(ValueError, match='budget'):
        Allocator(toy_recipe())  # 26 draws cannot give 9 units 4 each
    with pytest.raises(ValueError, match='minimum'):
        Allocator(recipe(), minimum=2)  # one chunk cannot be split in two
    with pytest.raises(ValueError, match='minimum'):
        Allocator(recipe(), minimum=5)
    # Units under the floor are lifted to it at the start; the total is kept.
    lopsided = replace(recipe(), writers=[16, 16, 2, 2], hidden=[16, 16, 2, 2], embedding=2)
    counts = Allocator(lopsided).counts
    assert sum(counts.values()) == 74 and min(counts.values()) == 4
    assert all(count % 2 == 0 for count in counts.values())
    assert Allocator(recipe()).counts == unit_counts(recipe()), 'a feasible recipe starts unchanged'


def test_state_round_trip():
    allocator = Allocator(recipe(), interval=1, ema=0.0, rate=1.0)
    allocator.observe(statistics(flat(recipe(), **{'hidden.3': 25.0})))
    allocator.update(1)
    state = json.loads(json.dumps(allocator.state_dict()))  # survives JSON, so it is plain data
    restored = Allocator(recipe(), interval=1, ema=0.0, rate=1.0)
    restored.load_state_dict(state)
    assert restored.counts == allocator.counts and restored.variance == allocator.variance


@pytest.mark.parametrize('suffix', [True, False])
def test_draw_costs_are_what_accounting_charges(suffix):
    """Two more draws on a unit raise analytic_cost by twice that unit's draw cost."""
    costs = draw_costs(TOY_CONFIG, batch=8, suffix=suffix)
    base = analytic_cost(recipe(), TOY_CONFIG, batch=8, suffix=suffix).flops_total
    for unit, count in unit_counts(recipe()).items():
        more = with_counts(recipe(), {**unit_counts(recipe()), unit: count + 2})
        extra = analytic_cost(more, TOY_CONFIG, batch=8, suffix=suffix).flops_total - base
        assert extra == pytest.approx(2 * costs[unit])
    if suffix:
        assert costs['writers.0'] > costs['writers.3'], 'a deep draw reruns fewer blocks'
    else:
        assert costs['writers.0'] == costs['writers.3'] == costs['embedding']


def test_flop_budget_is_never_exceeded():
    costs = draw_costs(TOY_CONFIG)
    allocator = Allocator(recipe(), interval=1, ema=0.0, rate=1.0, costs=costs)
    spent = lambda: sum(costs[unit] * count for unit, count in allocator.counts.items())
    budget = spent()
    for step, noisy in enumerate(('writers.0', 'hidden.3', 'embedding'), start=1):
        allocator.observe(statistics(flat(recipe(), **{noisy: 400.0})))
        assert allocator.update(step)
        assert spent() <= budget * (1 + 1e-9)
        assert budget - spent() < 2 * min(costs.values()), 'no affordable quantum is left unspent'
        assert min(allocator.counts.values()) >= 4
        assert allocator.counts[noisy] == max(allocator.counts.values())


# ---- the measurement, on the real estimator ------------------------------------------------------

def held_out_batch():
    rows = DATA[1][:8]
    return rows[:, :-1].contiguous(), rows[:, 1:].contiguous()


def estimate(model, seed, measure):
    estimator = dust.DUST(model, recipe(), seed)
    estimator.measure_split = measure
    try:
        gradients, _, _ = estimator.estimate(*held_out_batch())
        return gradients, estimator.split_statistics()
    finally:
        estimator.close()


def test_measuring_leaves_the_estimate_bit_identical():
    model = dust.make_model(TOY_CONFIG, CPU, 0)
    plain, nothing = estimate(model, 7, measure=False)
    measured, split = estimate(model, 7, measure=True)
    assert nothing == {}
    assert all(torch.equal(plain[name], measured[name]) for name in plain)
    assert {unit_of(name) for name in split} == set(unit_counts(recipe()))
    assert all(square > 0 and draws == unit_counts(recipe())[unit_of(name)]
               for name, (square, draws) in split.items())


def test_split_noise_matches_the_variance_over_seeds():
    """The split-half number claims to be the per-draw variance v, with Var(estimate) = v / K. Check it
    against the variance of the estimate over independent seeds, times K, pooled over parameters."""
    model = dust.make_model(TOY_CONFIG, CPU, 0)
    seeds = range(10)
    runs = [estimate(model, seed, measure=True) for seed in seeds]
    names = list(runs[0][1])
    from_split = sum(sum(split[name][0] / split[name][1] for _, split in runs) / len(runs) for name in names)
    over_seeds = sum(
        split_draws * torch.stack([gradients[name] for gradients, _ in runs]).var(0).sum().item()
        for name, (_, split_draws) in runs[0][1].items()
    )
    assert 0.6 < from_split / over_seeds < 1.6
    # Negative control: without the per-draw rescale (the K above, 4 or 8 here) the two cannot agree.
    unscaled = sum(torch.stack([gradients[name] for gradients, _ in runs]).var(0).sum().item() for name in names)
    assert from_split / unscaled > 2


# ---- the training loop ---------------------------------------------------------------------------

def run(output, steps=6, **kwargs):
    training, validation, testing = DATA
    return dust.train(TOY_CONFIG, recipe(), training, validation, testing, output, steps=steps,
                      eval_interval=2, seed=42, device=CPU, echo=lambda _: None, **kwargs)


def records(output):
    return [json.loads(line) for line in (Path(output) / 'metrics.jsonl').read_text().splitlines()]


def final_state(output):
    return torch.load(Path(output) / 'state.pt', weights_only=False)


def same_weights(a, b):
    return all(torch.equal(a[name], b[name]) for name in a)


ADAPTIVE = dict(interval=2, rate=1.0)


def test_measure_only_run_is_the_fixed_allocation_run(tmp_path):
    run(tmp_path / 'fixed', steps=4, checkpoint_interval=4)
    run(tmp_path / 'measured', steps=4, checkpoint_interval=4, allocation=dict(interval=0))
    fixed, measured = final_state(tmp_path / 'fixed'), final_state(tmp_path / 'measured')
    assert same_weights(fixed['model'], measured['model'])
    assert torch.equal(fixed['generator'], measured['generator'])
    assert fixed['allocator_state'] is None
    assert set(measured['allocator_state']['variance']) == set(unit_counts(recipe()))


def test_adaptive_run_keeps_its_budget_and_is_charged_what_it_ran(tmp_path):
    result = run(tmp_path / 'run', steps=6, allocation=ADAPTIVE)
    rows = records(tmp_path / 'run')
    allocations = [row['allocation'] for row in rows if 'allocation' in row]
    assert len(allocations) >= 2, 'the allocation should move at least once in six steps'
    for allocation in allocations:
        counts = allocation['writers'] + allocation['hidden'] + [allocation['embedding']]
        assert sum(counts) == BUDGET and min(counts) >= 4 and all(count % 2 == 0 for count in counts)
        assert allocation['hubs'] == recipe().hubs and allocation['head'] == recipe().head
    assert all(row['population'] == rows[1]['population'] for row in rows[1:])
    # A decision made after step s is logged at s and first used at s + 1.
    decided = [row['step'] for row in rows if 'allocation_noise' in row]
    used = [row['step'] for row in rows if 'allocation' in row]
    assert all(step % 2 == 0 for step in decided)
    assert used == [1] + [step + 1 for step in decided if step < 6]

    # Cumulative FLOPs = the sum of each step's cost under the allocation in force at that step.
    flops, current = 0.0, None
    for row in rows[1:]:
        current = row.get('allocation', current)
        settings = dust.Recipe(**current, lr=0.25, momentum=0.95)
        flops += dust_update_cost(settings, TOY_CONFIG, batch=8)['flops']
        assert row['flops'] == pytest.approx(flops)
    assert result['best_val'] < result['history'][0]['val_loss'], 'it still learns'


def test_adaptive_resume_matches_an_uninterrupted_run(tmp_path):
    run(tmp_path / 'straight', steps=6, checkpoint_interval=3, allocation=ADAPTIVE)
    run(tmp_path / 'resumed', steps=3, checkpoint_interval=3, allocation=ADAPTIVE)
    run(tmp_path / 'resumed', steps=6, checkpoint_interval=3, allocation=ADAPTIVE, resume=True)
    straight, resumed = final_state(tmp_path / 'straight'), final_state(tmp_path / 'resumed')
    assert same_weights(straight['model'], resumed['model'])
    assert torch.equal(straight['generator'], resumed['generator'])
    assert straight['allocator_state'] == resumed['allocator_state']
    assert straight['settings'] == resumed['settings']
    pick = lambda rows: [(row['step'], row.get('allocation'), row['flops']) for row in rows]
    assert pick(records(tmp_path / 'resumed')) == pick(records(tmp_path / 'straight'))

    with pytest.raises(ValueError, match='allocation'):
        run(tmp_path / 'resumed', steps=6, allocation=dict(interval=1), resume=True)
    with pytest.raises(ValueError, match='allocation'):
        run(tmp_path / 'resumed', steps=6, resume=True)  # a fixed-allocation run cannot continue it
