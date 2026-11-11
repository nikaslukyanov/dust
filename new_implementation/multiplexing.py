"""Structured perturbation multiplexing (Track 3, method C): the sign codes and their decoder.

Several module groups are perturbed in the same forward passes. Over a code sequence of M passes,
group g keeps one base noise field a_g and pass m applies it with sign C[m, g]: the perturbation is
C[m, g] * sigma * a_g. Each group's own loss change is then recovered by correlating the M observed
losses with its code column (literature/multiplexing/bibliography.md, "Method C math verification"):

    r_g  =  (1 / M) * sum_m C[m, g] * (L_m - L_0)

What the code has to do, and which choice does it:

- First order (the loss linear in each perturbation): orthogonal columns, C^T C = M I, separate the
  groups exactly. Hadamard columns are exactly orthogonal; i.i.d. Rademacher signs only on average.
- Baseline and self-curvature: a column that sums to zero rejects anything constant across the
  sequence -- the clean loss L_0 and each group's own second-order term. So the all-ones Hadamard
  column is never used. This is DUST's reward centering, done by the code: the centering group is the
  code sequence. ``centered`` applies the same centering to codes whose columns do not sum to zero.
- Cross-curvature between two groups g, h (second order): it reaches group k with weight
  T[k, g, h] = (1 / M) sum_m C[m, k] C[m, g] C[m, h]. For Hadamard columns the product of columns g
  and h is column g XOR h, so T is 1 when k = g XOR h and 0 otherwise: the cross term lands at full
  strength on one column. Using only columns whose index has an odd number of set bits makes
  g XOR h even, hence never a used column: every second-order cross term is cancelled exactly.
  That is the largest such ("sum-free") set, so G groups need M = the power of two with M / 2 >= G.

What the code cannot do: third-order terms survive, and a sequence still tests only one direction a_g
per group however long it is. See multiplex_experiment.py for what that costs in practice.
"""
import torch

KINDS = ('hadamard', 'rademacher')


def hadamard(order):
    """Sylvester Hadamard matrix of a power-of-two ``order``, entries +-1: H[m, j] = (-1)^popcount(m & j)."""
    if order < 1 or order & (order - 1):
        raise ValueError(f'A Sylvester Hadamard matrix needs a power-of-two order, got {order}')
    matrix = torch.ones(1, 1)
    while matrix.shape[0] < order:
        matrix = torch.cat([torch.cat([matrix, matrix], 1), torch.cat([matrix, -matrix], 1)])
    return matrix


def code_length(groups):
    """Passes in one code sequence for ``groups`` multiplexed groups: the power of two M with M / 2 >= groups."""
    if groups < 1:
        raise ValueError('Need at least one group')
    order = 2
    while order // 2 < groups:
        order *= 2
    return order


def sum_free_columns(groups):
    """Hadamard column indices for ``groups`` groups: the first ones with an odd number of set bits.

    Odd XOR odd is even, so no two of them multiply to a third (nor to the all-ones column 0).
    """
    order = code_length(groups)
    return [index for index in range(order) if bin(index).count('1') % 2][:groups]


def codebook(groups, kind='hadamard', noise=None, columns=None):
    """An [M, groups] matrix of +-1: row m holds every group's sign in pass m of a code sequence.

    ``hadamard``: the sum-free Hadamard columns (or the given ``columns``, for studying aliasing).
    ``rademacher``: independent random signs of the same length, drawn as the sign of ``noise(M, groups)``
    (standard normal), so they come from the estimator's seeded generator.
    """
    order = code_length(groups)
    if kind == 'hadamard':
        return hadamard(order)[:, sum_free_columns(groups) if columns is None else columns]
    if kind == 'rademacher':
        if noise is None:
            raise ValueError('Rademacher codes need a noise source')
        signs = noise(order, groups)
        return torch.where(signs >= 0, 1.0, -1.0).to(signs.device)
    raise ValueError(f'kind must be one of {KINDS}')


def centered(code):
    """Decode weights: each column minus its mean. Correlating losses with these equals correlating
    sequence-centered losses with the raw code. Unchanged for zero-sum (Hadamard) columns."""
    return code - code.mean(0, keepdim=True)


def decode(losses, code):
    """Each group's loss change from the ``losses`` of one code sequence ([M, ...]): (1 / M) C^T L, centered."""
    return torch.einsum('mg,m...->g...', centered(code).to(losses.dtype), losses) / code.shape[0]


def cross_talk(code):
    """(1 / M) C^T C: the identity for orthogonal codes; off-diagonals are first-order leakage."""
    return code.T @ code / code.shape[0]


def triple_products(code):
    """T[k, g, h] = (1 / M) sum_m C[m, k] C[m, g] C[m, h]: how much of the g-h cross-curvature reaches k."""
    return torch.einsum('mk,mg,mh->kgh', code, code, code) / code.shape[0]
