"""Adaptive population allocation (Track 2, method B): spend draws where the estimate is noisiest.

The shipped Recipe fixes how many draws each part of the model gets. This module re-decides it during
training from a noise measurement that costs no extra forward pass.

The rule (Neyman allocation; tasks/lessons/literature-findings.md). If unit i has per-draw variance v_i,
its gradient estimate from K_i draws has error v_i / K_i. Minimizing the total error sum_i v_i / K_i
under a fixed budget sum_i c_i K_i = C gives

    K_i  proportional to  sigma_i / sqrt(c_i),        sigma_i = sqrt(v_i)

With c_i = 1 the budget is a draw count and K_i is proportional to sigma_i. A unit's importance needs no
separate weight: a large gradient already shows up as a large sigma_i.

The measurement (DUST.direct_errors, split=True). A unit's draw chunks are split into two halves that
are otherwise identical. D, the signed sum of the chunk contributions (+ one half, - the other), has
E|D|^2 = (paired draws) * v_i, so v_i = |D|^2 / paired draws. That is already per draw: it does not
shrink as K_i grows, so units with different K_i are comparable (comparing the variances of the
K_i-averaged estimates instead would be wrong by a factor K_i each).

What is reallocated: the draws of ``writers`` and ``hidden`` at every block and of ``embedding`` (one
"unit" each, named 'writers.3', 'hidden.0', 'embedding'). Not reallocated:
- ``hubs``: its estimate is not a weight gradient but the target the attention internals are scored
  against, so its noise has no free measurement in the same currency;
- ``head``, the attention internals and the scalars: cheap, and outside the population K.

Guards against the known failure modes:
- thrash: the variance is EMA-smoothed, the allocation is recomputed every ``interval`` steps, and each
  recompute moves only ``rate`` of the way to the new optimum;
- mid-run bias: an allocation is used from the step after the one that measured it;
- floor: every unit keeps ``minimum`` draws (two chunks per GPU, the least that can be split), and the
  optimum is re-solved over the remaining units;
- rounding: counts are multiples of world * nrep, by largest remainder, never above the budget.

No torch dependency: plain floats in, a Recipe out.
"""
import math
from dataclasses import replace

from accounting import block_flops, head_flops, window_sizes
from groups import layer_of, param_group

PER_BLOCK = ('writers', 'hidden')


def unit_of(parameter):
    """The allocation unit whose draws estimate ``parameter``, or None if its draws are not reallocated."""
    group = param_group(parameter)
    if group == 'embedding':
        return 'embedding'
    if group in PER_BLOCK:
        return f'{group}.{layer_of(parameter)}'
    return None


def unit_counts(recipe):
    """Draws per unit in a Recipe."""
    counts = {f'{group}.{layer}': count for group in PER_BLOCK
              for layer, count in enumerate(getattr(recipe, group))}
    counts['embedding'] = recipe.embedding
    return counts


def with_counts(recipe, counts):
    """``recipe`` with its reallocated draws replaced by ``counts``; everything else unchanged."""
    layers = range(len(recipe.writers))
    return replace(
        recipe, embedding=counts['embedding'],
        **{group: [counts[f'{group}.{layer}'] for layer in layers] for group in PER_BLOCK},
    )


def draw_costs(config, batch=8, suffix=True):
    """FLOPs of one direct draw per unit, as accounting.analytic_cost charges it.

    A draw at block l reruns blocks l onward (the whole stack under full replay) plus the head; an
    embedding draw always reruns everything.
    """
    tokens = batch * config.sequence_len
    blocks = [block_flops(tokens, window, config) for window in window_sizes(config)]
    head = head_flops(tokens, config)
    costs = {}
    for layer in range(config.n_layer):
        cost = sum(blocks[layer if suffix else 0:]) + head
        for group in PER_BLOCK:
            costs[f'{group}.{layer}'] = cost
    costs['embedding'] = sum(blocks) + head
    return costs


def neyman(sigmas, budget, minimum, costs):
    """Real-valued draws per unit minimizing sum sigma_i^2 / K_i with sum c_i K_i = budget, K_i >= minimum.

    Units whose optimum falls below ``minimum`` are pinned there and the rest re-solved on what is left.
    With no signal at all the free units share the remaining budget equally in draws.
    """
    pinned = set()
    while True:
        free = [unit for unit in sigmas if unit not in pinned]
        remaining = budget - minimum * sum(costs[unit] for unit in pinned)
        weight = {unit: sigmas[unit] / math.sqrt(costs[unit]) for unit in free}
        total = sum(weight[unit] * costs[unit] for unit in free)
        if total > 0:
            draws = {unit: remaining * weight[unit] / total for unit in free}
        else:
            draws = {unit: remaining / sum(costs[other] for other in free) for unit in free}
        low = [unit for unit in free if draws[unit] < minimum]
        if not low:
            return {unit: draws.get(unit, float(minimum)) for unit in sigmas}
        pinned.update(low)


def apportion(target, budget, quantum, minimum, costs):
    """Round ``target`` to multiples of ``quantum`` (each at least ``minimum``) without exceeding ``budget``.

    Largest-remainder (Hamilton) apportionment: round down, then hand the leftover budget out one
    quantum at a time to the units that lost the most to rounding. With equal costs the result sums to
    the budget exactly; with unequal costs it stops when no unit's next quantum fits.
    """
    counts = {unit: max(minimum, quantum * math.floor(draws / quantum + 1e-9))
              for unit, draws in target.items()}
    left = budget - sum(costs[unit] * count for unit, count in counts.items())
    order = sorted(counts, key=lambda unit: target[unit] - counts[unit], reverse=True)
    slack = 1e-9 * budget
    granted = True
    while granted:
        granted = False
        for unit in order:
            price = costs[unit] * quantum
            if price <= left + slack:
                counts[unit] += quantum
                left -= price
                granted = True
    return counts


class Allocator:
    """Holds the current draws per unit and re-decides them from measured noise.

    Per training step: ``observe(estimator.split_statistics())``, then ``update(step)``; when that
    returns True, run the next step with ``recipe(settings)``. ``state_dict`` goes in the checkpoint.

    ``recipe``: the starting Recipe; its reallocated draws (or their cost, with ``costs``) are the
    budget. Units below ``minimum`` are lifted to it at the start, paid for by the others in proportion.
    ``interval``: recompute every that many steps (0: never, measure only).
    ``ema``: weight of the old variance in the running average (0: use only the latest step).
    ``rate``: fraction of the way each recompute moves toward the Neyman optimum (1: jump to it).
    ``minimum``: floor per unit, a multiple of world * nrep and at least two chunks per GPU.
    ``weights``: parameter name -> factor on its noise (the learning rate, to weigh the noise as the
    update sees it); missing names weigh 1.
    ``costs``: unit -> cost of one draw (``draw_costs``), to fix a FLOP budget instead of a draw count.
    """

    def __init__(self, recipe, world=1, interval=4, ema=0.7, rate=0.5, minimum=None, weights=None, costs=None):
        self.quantum = world * recipe.nrep
        self.minimum = 2 * self.quantum if minimum is None else minimum
        if self.minimum % self.quantum or self.minimum < 2 * self.quantum:
            raise ValueError(f'minimum must be a multiple of world * nrep = {self.quantum} and at least twice it')
        if interval < 0 or not 0 <= ema < 1 or not 0 < rate <= 1:
            raise ValueError('need interval >= 0, 0 <= ema < 1 and 0 < rate <= 1')
        self.interval, self.ema, self.rate = interval, ema, rate
        self.weights = weights or {}

        counts = unit_counts(recipe)
        if any(count % self.quantum for count in counts.values()):
            raise ValueError(f'every draw count must be a multiple of world * nrep = {self.quantum}')
        self.costs = dict(costs) if costs is not None else {unit: 1 for unit in counts}
        if set(self.costs) != set(counts):
            raise ValueError('costs must name exactly the reallocated units')
        self.budget = sum(self.costs[unit] * count for unit, count in counts.items())
        if self.budget < self.minimum * sum(self.costs.values()):
            raise ValueError(f'The budget cannot give every unit the minimum of {self.minimum} draws')
        if min(counts.values()) < self.minimum:
            shares = {unit: count * math.sqrt(self.costs[unit]) for unit, count in counts.items()}
            counts = self._round(neyman(shares, self.budget, self.minimum, self.costs))
        self.counts = counts
        self.variance = {}  # unit -> running average of the weighted per-draw variance

    def _round(self, target):
        return apportion(target, self.budget, self.quantum, self.minimum, self.costs)

    def recipe(self, settings):
        """``settings`` carrying the current allocation."""
        return with_counts(settings, self.counts)

    def observe(self, statistics):
        """Fold one step's split-half measurement into the running per-draw variance.

        ``statistics``: parameter name -> (squared norm of the half difference, paired draws), both
        summed over GPUs (DUST.split_statistics). Parameters of one unit add up.
        """
        sample = {}
        for name, (square, draws) in statistics.items():
            unit = unit_of(name)
            if unit is None or not draws:
                continue
            sample[unit] = sample.get(unit, 0.0) + self.weights.get(name, 1.0) ** 2 * square / draws
        for unit, value in sample.items():
            if not math.isfinite(value):
                continue
            old = self.variance.get(unit)
            self.variance[unit] = value if old is None else self.ema * old + (1 - self.ema) * value

    def sigmas(self):
        """Per-draw noise (standard deviation) per measured unit, the quantity draws are proportional to."""
        return {unit: math.sqrt(value) for unit, value in self.variance.items()}

    def update(self, step):
        """After ``step``: recompute the allocation if it is due. True if any count changed."""
        if not self.interval or step % self.interval or set(self.variance) != set(self.counts):
            return False
        sigmas = self.sigmas()
        target = neyman({unit: sigmas[unit] for unit in self.counts}, self.budget, self.minimum, self.costs)
        blended = {unit: count + self.rate * (target[unit] - count) for unit, count in self.counts.items()}
        counts = self._round(blended)
        changed = counts != self.counts
        self.counts = counts
        return changed

    def state_dict(self):
        return dict(counts=dict(self.counts), variance=dict(self.variance))

    def load_state_dict(self, state):
        if set(state['counts']) != set(self.counts):
            raise ValueError('The checkpoint allocates different units than this run')
        self.counts = {unit: state['counts'][unit] for unit in self.counts}
        self.variance = dict(state['variance'])
