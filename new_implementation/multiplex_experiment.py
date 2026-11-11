"""Method C's first experiment: two groups that share a rerun, multiplexed vs evaluated separately.

The two groups are one block's writers (attn.c_proj + mlp.c_proj) and its hidden layer (mlp.c_fc):
DUST evaluates them in separate forward passes that restart at the same block, so sharing passes
between them is the cleanest case for multiplexing. Each method estimates both groups' gradients on
the same weights and batch over repeated noise seeds and is scored against backprop:

- ``separate``:   DUST as shipped. P draws per group, 2 P forward passes.
- ``separate/2``: the same with P / 2 draws per group: P forward passes, the compute-matched baseline.
- ``joint``:      both groups perturbed in every pass with fresh independent Gaussian noise (how DUST
                  already treats the two writer projections). P draws per group, P passes.
- ``hadamard``:   sign-coded passes, exactly orthogonal codes (multiplexing.py). P draws per group, P passes.
- ``rademacher``: sign-coded passes, random signs. P draws per group, P passes.

So ``separate`` matches the others in draws per group and noise scale (the proposal's comparison), and
``separate/2`` matches them in forward passes. Reported per group: ``cosine`` (mean over seeds of the
cosine to backprop), ``mean_cosine`` (cosine of the seed-averaged estimate: direction of the bias),
``rel_bias`` (|mean estimate - backprop| / |backprop|; its floor from finite seeds is about
sqrt(variance / seeds)), ``variance`` (summed over coordinates, relative to |backprop|^2) and
``rel_norm``.

Run on CPU in seconds: python new_implementation/multiplex_experiment.py [--layer 0 --passes 16 --seeds 16 --nrep 2]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradient_check as gc  # noqa: E402  (imports after the sys.path shim above)
from dust import DUST, Recipe, make_model  # noqa: E402

METHODS = ('separate', 'separate/2', 'joint', 'hadamard', 'rademacher')


def experiment_recipe(nrep):
    """A Recipe whose only job here is to set the chunk size: every count the smallest valid multiple."""
    return Recipe(writers=[nrep] * 4, hidden=[nrep] * 4, hubs=[nrep] * 4, embedding=nrep, head=16 * nrep,
                  local=dict(q=nrep, v=nrep, gate=nrep, k=nrep, ve=nrep), nrep=nrep, lr=0.25, momentum=0.95)


def block_groups(layer):
    prefix = f'transformer.h.{layer}.'
    return dict(writers=[prefix + 'attn.c_proj', prefix + 'mlp.c_proj'], hidden=[prefix + 'mlp.c_fc'])


@torch.no_grad()
def estimate(estimator, method, layer, passes, sigma=0.2):
    """One estimate of the block's writer and hidden gradients; needs estimator.capture(x, y) first.

    Returns (gradients by parameter name, summed over tokens; forward passes used).
    """
    groups = block_groups(layer)
    sites = groups['writers'] + groups['hidden']
    if method in ('separate', 'separate/2'):
        draws = passes if method == 'separate' else passes // 2
        errors = {**estimator.direct_errors(groups['writers'], draws, sigma, layer),
                  **estimator.direct_errors(groups['hidden'], draws, sigma, layer)}
        forwards = 2 * draws
    elif method == 'joint':
        errors, forwards = estimator.direct_errors(sites, passes, sigma, layer), passes
    else:
        coded = [(groups['writers'], sigma), (groups['hidden'], sigma)]
        errors, forwards = estimator.multiplexed_errors(coded, passes, layer, method), passes
    return {site + '.weight': estimator.reassemble(site, error) for site, error in errors.items()}, forwards


def flatten(gradients, sites):
    return torch.cat([gradients[site + '.weight'].reshape(-1) for site in sites]).double()


def compare(model, settings, x, y, layer=0, passes=16, seeds=range(16), methods=METHODS, sigma=0.2):
    """Score every method against backprop on (x, y). Returns {method: {forwards, draws_per_group, groups}}."""
    reference, _ = gc.reference_gradients(model, x, y)
    valid_tokens = (y != -1).sum()
    groups = block_groups(layer)
    truth = {name: flatten(reference, sites) for name, sites in groups.items()}
    estimator = DUST(model, settings)
    report = {}
    try:
        estimator.capture(x, y)
        for method in methods:
            samples = {name: [] for name in groups}
            for seed in seeds:
                gc.reseed(estimator, seed)
                gradients, forwards = estimate(estimator, method, layer, passes, sigma)
                for name, sites in groups.items():
                    samples[name].append(flatten(gradients, sites) / valid_tokens)
            scores = {}
            for name, rows in samples.items():
                stack, target = torch.stack(rows), truth[name]
                mean = stack.mean(0)
                cosine = torch.nn.functional.cosine_similarity(stack, target.unsqueeze(0), dim=1)
                scores[name] = dict(
                    cosine=cosine.mean().item(),
                    mean_cosine=torch.nn.functional.cosine_similarity(mean, target, dim=0).item(),
                    rel_bias=((mean - target).norm() / target.norm()).item(),
                    variance=(stack.var(0).sum() / target.square().sum()).item(),
                    rel_norm=(stack.norm(dim=1).mean() / target.norm()).item(),
                )
            draws = passes // 2 if method == 'separate/2' else passes
            report[method] = dict(forwards=forwards, draws_per_group=draws, groups=scores)
    finally:
        estimator.close()
    return report


def table(report):
    lines = [f"{'method':12s}{'forwards':>9s}{'draws/grp':>10s}  {'group':8s}{'cosine':>8s}{'mean_cos':>9s}"
             f"{'rel_bias':>9s}{'variance':>10s}{'rel_norm':>9s}"]
    for method, row in report.items():
        for name, s in row['groups'].items():
            lines.append(f"{method:12s}{row['forwards']:9d}{row['draws_per_group']:10d}  {name:8s}{s['cosine']:8.3f}"
                         f"{s['mean_cosine']:9.3f}{s['rel_bias']:9.2f}{s['variance']:10.1f}{s['rel_norm']:9.2f}")
    return '\n'.join(lines)


def main():
    from toy import TOY_CONFIG, toy_data

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--passes', type=int, default=16, help='P: a multiple of 8 and of 2 * nrep')
    parser.add_argument('--nrep', type=int, default=2, choices=(1, 2, 4, 8, 16),
                        help='draws per forward pass; DUST centers rewards across them (2 to 16 in the recipes)')
    parser.add_argument('--seeds', type=int, default=16)
    parser.add_argument('--sigma', type=float, default=0.2, help='noise scale (DUST uses 0.2 for these groups)')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    model = make_model(TOY_CONFIG, torch.device('cpu'), seed=0)
    rows = toy_data()[1][:8]
    report = compare(model, experiment_recipe(args.nrep), rows[:, :-1].contiguous(), rows[:, 1:].contiguous(), layer=args.layer,
                     passes=args.passes, seeds=range(args.seeds), sigma=args.sigma)
    print(json.dumps(report, indent=2) if args.json else table(report))


if __name__ == '__main__':
    main()
