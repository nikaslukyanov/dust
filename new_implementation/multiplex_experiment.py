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

GPU half (Track 3): the real 8x512 model under bf16 autocast, on a training batch, swept over sigma and
blocks, with the time per estimate:

    python new_implementation/multiplex_experiment.py --real --device cuda --tokens 1M \
        --layers 0 4 --sigmas 0.05 0.1 0.2 0.4 0.8 --nrep 8 --passes 16 --output runs/track3/nrep8.json

The bias of the decode is second order in sigma while the bf16 readout noise falls as sigma grows, so
a sigma where hadamard beats separate/2 (equal forward passes) is the case for a training campaign.
"""
import time
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


def experiment_recipe(nrep, n_layer=4):
    """A Recipe whose only job here is to set the chunk size: every count the smallest valid multiple."""
    return Recipe(writers=[nrep] * n_layer, hidden=[nrep] * n_layer, hubs=[nrep] * n_layer, embedding=nrep,
                  head=16 * nrep,
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


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def compare(model, settings, x, y, layer=0, passes=16, seeds=range(16), methods=METHODS, sigma=0.2,
            reference=None, precision='fp32', chunk=None):
    """Score every method against backprop on (x, y). Returns {method: {forwards, draws_per_group, seconds, groups}}.

    ``reference``: precomputed backprop gradients (reused across a sweep); otherwise computed here at
    ``precision`` with ``chunk`` sequences per backward pass (gradient_check.reference_gradients).
    ``seconds`` is the median wall time of one estimate, synchronized on CUDA.
    """
    if reference is None:
        reference, _ = gc.reference_gradients(model, x, y, precision=precision, chunk=chunk)
    device = x.device
    valid_tokens = (y != -1).sum()
    groups = block_groups(layer)
    truth = {name: flatten(reference, sites) for name, sites in groups.items()}
    estimator = DUST(model, settings)
    report = {}
    try:
        estimator.capture(x, y)
        for method in methods:
            samples = {name: [] for name in groups}
            seconds = []
            for seed in seeds:
                gc.reseed(estimator, seed)
                _sync(device)
                start = time.monotonic()
                gradients, forwards = estimate(estimator, method, layer, passes, sigma)
                _sync(device)
                seconds.append(time.monotonic() - start)
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
            seconds.sort()
            report[method] = dict(forwards=forwards, draws_per_group=draws,
                                  seconds=seconds[len(seconds) // 2], groups=scores)
    finally:
        estimator.close()
    return report


def table(report):
    lines = [f"{'method':12s}{'forwards':>9s}{'draws/grp':>10s}{'sec':>8s}  {'group':8s}{'cosine':>8s}{'mean_cos':>9s}"
             f"{'rel_bias':>9s}{'variance':>10s}{'rel_norm':>9s}"]
    for method, row in report.items():
        for name, s in row['groups'].items():
            lines.append(f"{method:12s}{row['forwards']:9d}{row['draws_per_group']:10d}{row['seconds']:8.3f}  "
                         f"{name:8s}{s['cosine']:8.3f}"
                         f"{s['mean_cosine']:9.3f}{s['rel_bias']:9.2f}{s['variance']:10.1f}{s['rel_norm']:9.2f}")
    return '\n'.join(lines)


def real_setup(args):
    """The paper's 8x512 model at init on one training batch of real data (as gpu_preflight does)."""
    from model import GPTConfig
    import dust

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda':
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')
    config = GPTConfig()
    rows = dust.load_sequences(args.data_dir / f'train_{args.tokens}.pt', args.batch).to(device)
    model = make_model(config, device, seed=args.model_seed)
    return model, config, rows, device


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
    real = parser.add_argument_group('real model (Track 3 GPU half)')
    real.add_argument('--real', action='store_true', help='The 8x512 model on real data instead of the toy.')
    real.add_argument('--device', help='cuda (default when available) or cpu.')
    real.add_argument('--tokens', default='1M', help='Which train_<tokens>.pt the batch comes from.')
    real.add_argument('--data-dir', type=Path, default=Path('data'))
    real.add_argument('--batch', type=int, default=8, help='Sequences in the batch (training uses 8).')
    real.add_argument('--model-seed', type=int, default=42)
    real.add_argument('--layers', type=int, nargs='+', help='Blocks to test (overrides --layer).')
    real.add_argument('--sigmas', type=float, nargs='+', help='Noise scales to sweep (overrides --sigma).')
    real.add_argument('--reference', choices=('fp32', 'match'), default='fp32',
                      help='Backprop reference precision: fp32 truth, or bf16 autocast matching DUST.')
    real.add_argument('--chunk', type=int, default=1, help='Sequences per backward pass of the reference.')
    real.add_argument('--output', type=Path, help='Write the whole sweep as JSON here.')
    args = parser.parse_args()

    if args.real:
        model, config, rows, device = real_setup(args)
        settings = experiment_recipe(args.nrep, config.n_layer)
    else:
        model, rows, device = make_model(TOY_CONFIG, torch.device('cpu'), seed=0), toy_data()[1][:8], torch.device('cpu')
        settings = experiment_recipe(args.nrep)
    x, y = rows[:, :-1].contiguous(), rows[:, 1:].contiguous()
    layers, sigmas = args.layers or [args.layer], args.sigmas or [args.sigma]

    if len(layers) == 1 and len(sigmas) == 1 and not args.real:  # the original single CPU comparison
        report = compare(model, settings, x, y, layer=layers[0], passes=args.passes, seeds=range(args.seeds),
                         sigma=sigmas[0])
        print(json.dumps(report, indent=2) if args.json else table(report))
        return

    reference, _ = gc.reference_gradients(model, x, y, precision=args.reference, chunk=args.chunk if args.real else None)
    sweep = dict(real=args.real, device=str(device), nrep=args.nrep, passes=args.passes, seeds=args.seeds,
                 batch=len(x), reference=args.reference, results=[])
    if device.type == 'cuda':
        sweep['gpu'] = torch.cuda.get_device_name(device)
    for layer in layers:
        for sigma in sigmas:
            report = compare(model, settings, x, y, layer=layer, passes=args.passes, seeds=range(args.seeds),
                             sigma=sigma, reference=reference)
            sweep['results'].append(dict(layer=layer, sigma=sigma, methods=report))
            print(f'\n== block {layer}, sigma {sigma}', flush=True)
            print(json.dumps(report, indent=2) if args.json else table(report), flush=True)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(sweep, indent=2) + '\n')


if __name__ == '__main__':
    main()
