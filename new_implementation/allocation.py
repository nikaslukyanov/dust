"""Adaptive population allocation (Track 2, method B): spend draws where the estimate is least reliable.

The shipped Recipe fixes how many draws each part of the model gets. This module re-decides it during
training from a measurement that costs no extra forward pass. It implements the two policies the
proposal names, both fed by the same split of a unit's draws into two independent halves a and b
(tasks/lessons/literature-findings.md, literature/adaptive-allocation/bibliography.md):

``variance`` (Neyman allocation). If unit i has per-draw variance v_i, its estimate from K_i draws has
error v_i / K_i. Minimizing the total error sum_i v_i / K_i under a budget sum_i c_i K_i = C gives

    K_i  proportional to  sigma_i / sqrt(c_i),        sigma_i = sqrt(v_i)

A unit's importance needs no separate weight: a large gradient already shows up as a large sigma_i.
v_i comes from the halves' disagreement, E|a - b|^2 = 4 v_i / K_i.

``agreement`` (gradient agreement; the norm test). Equalize every unit's relative error
v_i / (K_i |g_i|^2), i.e. bring every unit to the same half-to-half cosine:

    K_i  proportional to  v_i / |g_i|^2

|g_i|^2 comes from the halves' inner product, E[a . b] = |g_i|^2. The expected cosine between the
halves is rho / (rho + 1) with rho = |g_i|^2 K_i / (2 v_i), so this is the cosine signal in a form
that can be averaged over steps. It is scale-free, where ``variance`` favours large gradients.

Both v_i and |g_i|^2 arrive already per draw (DUST.split_statistics), so units with different K_i are
comparable; comparing the variances of the K_i-averaged estimates would be wrong by a factor K_i each.

What is reallocated, one "unit" each: ``writers`` and ``hidden`` at every block ('writers.3',
'hidden.0') and ``embedding``. ``agreement`` also reallocates ``hubs`` ('hubs.2'): a hub's estimate is
not a weight gradient but the target the attention internals are scored against, so its noise is not
in the same units as the others' and only the scale-free policy can rank it. Head, the attention
internals and the scalars are never reallocated (cheap, and outside the population K).

Guards against the known failure modes:
- thrash: the measurements are EMA-smoothed, the allocation is recomputed every ``interval`` steps,
  and each recompute moves only ``rate`` of the way to the new target;
- mid-run bias: an allocation is used from the step after the one that measured it;
- floor: every unit keeps ``minimum`` draws (two chunks per GPU, the least that can be split, and the
  least that keeps a starved unit measurable), and the target is re-solved over the remaining units;
- rounding: counts are multiples of world * nrep, by largest remainder, never above the budget.

No torch dependency: plain floats in, a Recipe out.
"""
import math
from dataclasses import replace

from accounting import block_flops, head_flops, window_sizes
from groups import layer_of, param_group

POLICIES = ('variance', 'agreement')
PER_BLOCK = ('writers', 'hidden', 'hubs')


def unit_of(name):
    """The allocation unit of a parameter name or a hub site ('o@3'), or None if it has none."""
    if name.startswith('o@'):
        return f'hubs.{layer_of(name)}'
    group = param_group(name)
    if group == 'embedding':
        return 'embedding'
    if group in ('writers', 'hidden'):
        return f'{group}.{layer_of(name)}'
    return None


def unit_counts(recipe, hubs=False):
    """Draws per unit in a Recipe; ``hubs`` adds the attention-output units."""
    groups = PER_BLOCK if hubs else PER_BLOCK[:2]
    counts = {f'{group}.{layer}': count for group in groups for layer, count in enumerate(getattr(recipe, group))}
    counts['embedding'] = recipe.embedding
    return counts


def with_counts(recipe, counts):
    """``recipe`` with the draws of the units in ``counts`` replaced; everything else unchanged."""
    layers = range(len(recipe.writers))
    changed = {group: [counts[f'{group}.{layer}'] for layer in layers]
               for group in PER_BLOCK if f'{group}.0' in counts}
    return replace(recipe, embedding=counts['embedding'], **changed)


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


def proportional(shares, budget, minimum, costs):
    """Real-valued draws K_i proportional to ``shares`` with sum c_i K_i = budget and K_i >= minimum.

    Units whose share falls below ``minimum`` are pinned there and the rest re-solved on what is left.
    With every share zero the free units split the remaining budget equally in draws.
    """
    pinned = set()
    while True:
        free = [unit for unit in shares if unit not in pinned]
        remaining = budget - minimum * sum(costs[unit] for unit in pinned)
        total = sum(shares[unit] * costs[unit] for unit in free)
        if total > 0:
            draws = {unit: remaining * shares[unit] / total for unit in free}
        else:
            draws = {unit: remaining / sum(costs[other] for other in free) for unit in free}
        low = [unit for unit in free if draws[unit] < minimum]
        if not low:
            return {unit: draws.get(unit, float(minimum)) for unit in shares}
        pinned.update(low)


def neyman(sigmas, budget, minimum, costs):
    """Draws minimizing sum sigma_i^2 / K_i on the budget: K_i proportional to sigma_i / sqrt(c_i)."""
    shares = {unit: sigma / math.sqrt(costs[unit]) for unit, sigma in sigmas.items()}
    return proportional(shares, budget, minimum, costs)


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
    """Holds the current draws per unit and re-decides them from the split-half measurement.

    Per training step: ``observe(estimator.split_statistics())``, then ``update(step)``; when that
    returns True, run the next step with ``recipe(settings)``. ``state_dict`` goes in the checkpoint.

    ``recipe``: the starting Recipe; its reallocated draws (or their cost, with ``costs``) are the
    budget. Units below ``minimum`` are lifted to it at the start, paid for by the others in proportion.
    ``policy``: 'variance' or 'agreement' (see the module docstring).
    ``interval``: recompute every that many steps (0: never, measure only).
    ``ema``: weight of the old measurement in the running average (0: use only the latest step).
    ``rate``: fraction of the way each recompute moves toward the target (1: jump to it).
    ``minimum``: floor per unit, a multiple of world * nrep and at least two chunks per GPU.
    ``weights``: parameter name -> factor on its noise and signal (the learning rate, to weigh the noise
    as the update sees it); missing names weigh 1. Only 'variance' depends on it across units.
    ``costs``: unit -> cost of one draw (``draw_costs``), to fix a FLOP budget instead of a draw count.
    """

    def __init__(self, recipe, world=1, policy='variance', interval=4, ema=0.7, rate=0.5, minimum=None,
                 weights=None, costs=None):
        if policy not in POLICIES:
            raise ValueError(f'policy must be one of {POLICIES}')
        self.quantum = world * recipe.nrep
        self.minimum = 2 * self.quantum if minimum is None else minimum
        if self.minimum % self.quantum or self.minimum < 2 * self.quantum:
            raise ValueError(f'minimum must be a multiple of world * nrep = {self.quantum} and at least twice it')
        if interval < 0 or not 0 <= ema < 1 or not 0 < rate <= 1:
            raise ValueError('need interval >= 0, 0 <= ema < 1 and 0 < rate <= 1')
        self.policy, self.interval, self.ema, self.rate = policy, interval, ema, rate
        self.weights = weights or {}

        counts = unit_counts(recipe, hubs=policy == 'agreement')
        if any(count % self.quantum for count in counts.values()):
            raise ValueError(f'every draw count must be a multiple of world * nrep = {self.quantum}')
        if costs is not None and not set(counts) <= set(costs):
            raise ValueError('costs must name every reallocated unit')
        self.costs = {unit: costs[unit] if costs is not None else 1 for unit in counts}
        self.budget = sum(self.costs[unit] * count for unit, count in counts.items())
        self.ceiling = 1000 * sum(counts.values())  # relative noise of a unit whose signal is lost in noise
        if self.budget < self.minimum * sum(self.costs.values()):
            raise ValueError(f'The budget cannot give every unit the minimum of {self.minimum} draws')
        if min(counts.values()) < self.minimum:
            counts = self._round(proportional(counts, self.budget, self.minimum, self.costs))
        self.counts = counts
        self.variance = {}  # unit -> running average of the per-draw variance v
        self.signal = {}    # unit -> running average of the squared gradient norm |g|^2

    def _round(self, target):
        return apportion(target, self.budget, self.quantum, self.minimum, self.costs)

    def recipe(self, settings):
        """``settings`` carrying the current allocation."""
        return with_counts(settings, self.counts)

    def observe(self, statistics):
        """Fold one step's split-half measurement into the running averages.

        ``statistics``: name -> dict with ``variance`` (per draw) and ``signal`` (squared norm of the
        estimated quantity), from DUST.split_statistics. Parameters of one unit add up.
        """
        variance, signal = {}, {}
        for name, measured in statistics.items():
            unit = unit_of(name)
            if unit not in self.counts:
                continue
            weight = self.weights.get(name, 1.0) ** 2
            variance[unit] = variance.get(unit, 0.0) + weight * measured['variance']
            signal[unit] = signal.get(unit, 0.0) + weight * measured['signal']
        for average, sample in ((self.variance, variance), (self.signal, signal)):
            for unit, value in sample.items():
                if math.isfinite(value):
                    old = average.get(unit)
                    average[unit] = value if old is None else self.ema * old + (1 - self.ema) * value

    def sigmas(self):
        """Per-draw noise (standard deviation) per measured unit: what 'variance' allocates in proportion to."""
        return {unit: math.sqrt(max(value, 0.0)) for unit, value in self.variance.items()}

    def relative_noise(self):
        """v / |g|^2 per measured unit: what 'agreement' allocates in proportion to.

        It is the number of draws at which the unit's noise equals its signal. The signal estimate
        can come out zero or negative when it is lost in the noise; the ratio is then capped at 1000
        times the starting draw total ("needs far more than the whole budget") instead of dividing by
        nothing.
        """
        ratios = {}
        for unit, variance in self.variance.items():
            floor = variance / self.ceiling
            ratios[unit] = variance / max(self.signal.get(unit, 0.0), floor) if variance > 0 else 0.0
        return ratios

    def agreement(self):
        """Expected cosine between a unit's two half estimates at its current draws: rho / (rho + 1)."""
        return {unit: 1 / (1 + 2 * ratio / self.counts[unit])
                for unit, ratio in self.relative_noise().items() if unit in self.counts}

    def target(self):
        """Real-valued draws the policy asks for from the current averages."""
        if self.policy == 'variance':
            sigmas = self.sigmas()
            return neyman({unit: sigmas[unit] for unit in self.counts}, self.budget, self.minimum, self.costs)
        ratios = self.relative_noise()
        return proportional({unit: ratios[unit] for unit in self.counts}, self.budget, self.minimum, self.costs)

    def update(self, step):
        """After ``step``: recompute the allocation if it is due. True if any count changed."""
        if not self.interval or step % self.interval or not set(self.counts) <= set(self.variance):
            return False
        target = self.target()
        blended = {unit: count + self.rate * (target[unit] - count) for unit, count in self.counts.items()}
        counts = self._round(blended)
        changed = counts != self.counts
        self.counts = counts
        return changed

    def state_dict(self):
        return dict(counts=dict(self.counts), variance=dict(self.variance), signal=dict(self.signal))

    def load_state_dict(self, state):
        if set(state['counts']) != set(self.counts):
            raise ValueError('The checkpoint allocates different units than this run')
        self.counts = {unit: state['counts'][unit] for unit in self.counts}
        self.variance, self.signal = dict(state['variance']), dict(state['signal'])
