"""Minimal DUST implementation: forward-only transformer training with SGD.

No autograd or backward pass is used. One update (DUST.step):
  1. A clean forward caches every layer's input and output and the per-token losses.
  2. Direct draws jitter a layer's output at every token, y + sigma * a with a ~ N(0, I), rerun the
     forward and reward each token's jitter with the loss reduction at that token. The
     reward-weighted noise, averaged over draws, is the estimated error at the layer's output.
     A draw at block l reruns only blocks l onward, from the residual stream the clean pass
     cached at block l (cached-suffix execution; --full-replay reruns the whole network instead).
  3. The attention internals (queries, keys, values, value-embedding gates, value embeddings) are
     credited through the estimated error at the attention output instead,
     recomputing only their block's attention.
  4. The head is jittered on the cached logits, one 256-column slab at a time.
  5. The outer product of each layer's estimated output error with its cached input, summed over
     tokens, is the layer's weight gradient, as in backprop. The 2L residual mixing scalars use
     two-sided weight-space ES. Then one SGD step.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import torch
from torch import nn
from torch.profiler import record_function

from model import GPT, GPTConfig, attention, apply_rotary_emb, has_ve, norm

# Training-time logging and checkpointing live in new_implementation/ (the team's additions).
sys.path.append(str(Path(__file__).resolve().parent / 'new_implementation'))
from allocation import Allocator, draw_costs  # noqa: E402
from checkpoint import checkpoint_path, load_state, save_state  # noqa: E402
from tracking import WandbLogger  # noqa: E402
from groups import layer_of  # noqa: E402
from multiplexing import KINDS, centered, code_length, codebook  # noqa: E402
from run_metrics import TrainingLog, dust_update_cost, gradient_probe  # noqa: E402


@dataclass
class Recipe:
    """Draws per update for each part of the model, summed over GPUs. Lists have one entry per block."""
    writers: list[int]      # draws per block for the attention and MLP output projections (c_proj)
    hidden: list[int]       # draws per block for the MLP's first layer (mlp.c_fc)
    hubs: list[int]         # draws per block for the attention output, before attn.c_proj
    embedding: int          # draws for the token embedding
    head: int               # draws for the output head (lm_head)
    local: dict[str, int]   # draws per block for each attention internal: q, v, gate, k, ve
    nrep: int               # draws run together in one forward pass
    lr: float
    momentum: float
    gamma: float = 0.98  # attention-internal credit decay per token of lag


DEFAULT_CONFIGS = {
    '1M': {
        'steps': 61,
        'eval_interval': 2,
        'optimizer': {
            256: dict(lr=0.25, momentum=0.95),
            1024: dict(lr=0.225, momentum=0.95),
            4096: dict(lr=0.225, momentum=0.98),
            16384: dict(lr=0.225, momentum=0.98),
        },
    },
    '10M': {
        'steps': 610,
        'eval_interval': 10,
        'optimizer': {
            256: dict(lr=0.175, momentum=0.95),
            1024: dict(lr=0.25, momentum=0.95),
            4096: dict(lr=0.25, momentum=0.95),
            16384: dict(lr=0.25, momentum=0.95),
        },
        # At 16,384 draws the estimator constants were tuned with (lr, momentum) at fixed compute:
        # twice the attention-output draws, paid by the writer streams, and credit decay 0.99.
        'allocation': {
            16384: dict(writers=[34, 28, 14, 8, 6, 4, 2, 2], hubs=[12, 10, 8, 4, 4, 4, 4, 4], gamma=0.99),
        },
    },
}

# Draws per forward pass (the batch is repeated this many times).
DRAW_CHUNK_SIZES = {256: 2, 1024: 8, 4096: 8, 16384: 16}


def recipe(tokens, population):
    """Build a saved config, population counts direct-loss draws only. """
    optimizer = DEFAULT_CONFIGS[tokens]['optimizer'][population]
    nrep = DRAW_CHUNK_SIZES[population]
    scale = population // 256
    override = DEFAULT_CONFIGS[tokens].get('allocation', {}).get(population, {})
    writers = [v * scale for v in override.get('writers', [38, 32, 16, 10, 6, 4, 2, 2])]
    hidden = writers.copy()
    hubs = [v * scale for v in override.get('hubs', [6, 6, 4, 2, 2, 2, 2, 2])]
    embedding, local = 10 * scale, [v * scale for v in [2, 20, 6, 12, 8]]

    return Recipe(
        writers, hidden, hubs, embedding, 4 * population,
        dict(zip(('q', 'v', 'gate', 'k', 've'), local)),
        nrep, **optimizer, gamma=override.get('gamma', 0.98),
    )


def average_ranks(tensor):
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        tensor.div_(torch.distributed.get_world_size())
    return tensor


class DUST:
    """
    The actual algorithm.

    Forward hooks on every Linear and Embedding capture its input and output during the clean pass and add the
    current jitter to its output during a draw. Pre-hooks on attn.c_proj add the virtual sites o@{layer}, the
    attention output before the projection. Each GPU draws different noise for the same batch.

    With suffix=True (baseline A) a direct draw at block l restarts the forward at block l from the
    clean pass's cached residual stream; suffix=False reruns the whole network (the public code).
    Both consume the same noise, so for a fixed seed they differ only by floating-point rounding.

    multiplex='hadamard' or 'rademacher' (method C) estimates each block's writers and hidden layer
    from shared, sign-coded forward passes instead of separate ones (multiplexed_errors): the same
    draws per group, half the forward passes for those two groups.
    """

    def __init__(self, model, settings, seed=42, rank=0, world=1, suffix=True, multiplex=None):
        self.model, self.settings, self.world, self.suffix = model, settings, world, suffix
        if multiplex is not None and multiplex not in KINDS:
            raise ValueError(f'multiplex must be None or one of {KINDS}')
        self.multiplex = multiplex
        self.device = next(model.parameters()).device
        self.dtype = torch.bfloat16 if self.device.type == 'cuda' else torch.float32
        self.generator = torch.Generator(self.device).manual_seed(seed + 3 + 7777 * rank)
        self.modules = {n: m for n, m in model.named_modules() if isinstance(m, (nn.Linear, nn.Embedding))}
        self.inputs, self.outputs, self.jitter = {}, {}, {}
        self.capturing = False
        # Split-half noise measurement for adaptive allocation (Track 2). Off by default. When on,
        # estimate() fills ``split`` at no extra forward pass; read it with split_statistics().
        self.measure_split, self.split = False, {}

        counts = settings.writers + settings.hidden + settings.hubs + [settings.embedding]
        counts += list(settings.local.values())
        if any(k % (world * settings.nrep) for k in counts) or 8 % world:
            raise ValueError('Population/chunk size is incompatible with this GPU count.')
        slabs = model.config.vocab_size // 256
        if model.config.vocab_size % 256 or settings.head % (world * settings.nrep * slabs):
            raise ValueError('Head draws must visit each 256-column slab equally on every GPU.')

        self.hooks = [m.register_forward_hook(self.output_hook(n)) for n, m in self.modules.items()]
        for layer in range(model.config.n_layer):
            module = self.modules[f'transformer.h.{layer}.attn.c_proj']
            self.hooks.append(module.register_forward_pre_hook(self.input_hook(f'o@{layer}')))

    def autocast(self):
        return torch.autocast('cuda', dtype=self.dtype) if self.device.type == 'cuda' else nullcontext()

    def output_hook(self, name):
        def hook(module, args, output):
            if self.capturing:
                self.inputs[name] = args[0].float() if args[0].is_floating_point() else args[0]
                self.outputs[name] = output
            if name in self.jitter:
                return output + self.jitter[name].to(output.dtype)

        return hook

    def input_hook(self, name):
        def hook(module, args):
            if name in self.jitter:
                return (args[0] + self.jitter[name].to(args[0].dtype),)

        return hook

    def close(self):
        for hook in self.hooks:
            hook.remove()

    def noise(self, *shape, dtype=None):
        return torch.randn(*shape, device=self.device, dtype=dtype or self.dtype, generator=self.generator)

    def losses(self, x, y, resume=None, streams=None):
        """Per-token losses. resume = (start, stream, x0) reruns only blocks start onward."""
        with self.autocast():
            if resume is None:
                loss = self.model(x, y, loss_reduction='none', streams=streams)
            else:
                loss = self.model.forward_from(*resume, x, y, loss_reduction='none')
            return loss.view_as(y).float()

    def capture(self, x, y):
        self.x, self.y = x, y
        self.batch, self.length = x.shape
        self.streams = []  # streams[l]: the clean residual stream entering block l; streams[0] is x0
        self.capturing = True
        try:
            self.clean_loss = self.losses(x, y, streams=self.streams)
        finally:
            self.capturing = False

    def direct_errors(self, sites, draws, sigma, start=None, split=False):
        """One-sided loss drops, centered across each chunk of draws.

        start is the first block the sites can change; with cached-suffix execution each draw reruns
        only blocks start onward. None (the token embedding) always reruns the whole network.

        split=True also measures the estimate's reliability for adaptive allocation, from the same
        draws: chunks are paired off into two independent halves A and B (summed separately, and turned
        into weight gradients except at the o@ sites). self.split[parameter or o@ site] gets
        (|A - B|^2, A . B, |A|^2, |B|^2) and the paired draw count; split_statistics() turns them into
        the per-draw variance, the squared gradient norm and the halves' cosine. The returned errors
        are unchanged.
        """
        n = self.settings.nrep
        x, y = self.x.repeat(n, 1), self.y.repeat(n, 1)
        resume = None
        if self.suffix and start is not None:
            resume = start, self.streams[start].repeat(n, 1, 1), self.streams[0].repeat(n, 1, 1)
        width = lambda s: self.model.config.n_embd if s.startswith('o@') else self.outputs[s].shape[-1]
        errors = {s: torch.zeros(self.batch, self.length, width(s), device=self.device) for s in sites}

        chunks = draws // n
        paired = 2 * (chunks // 2) if split else 0  # an odd last chunk is left out of the comparison
        halves = {s: [torch.zeros_like(errors[s]), torch.zeros_like(errors[s])] for s in sites} if paired else {}

        for chunk in range(chunks):
            noise = {s: self.noise(n * self.batch, self.length, width(s)) for s in sites}
            self.jitter = {s: sigma * a for s, a in noise.items()}
            try:
                perturbed_loss = self.losses(x, y, resume).view(n, self.batch, self.length)
            finally:
                self.jitter = {}

            reward = self.clean_loss.unsqueeze(0) - perturbed_loss
            if n > 1:
                reward = reward - reward.mean(0, keepdim=True)
            for site, a in noise.items():
                contribution = torch.einsum(
                    'nbt,nbtd->btd', reward.to(self.dtype),
                    a.view(n, self.batch, self.length, -1),
                ).float() / sigma
                errors[site] -= contribution
                if chunk < paired:
                    halves[site][chunk % 2] += contribution

        for site, (a, b) in halves.items():
            name = site
            if not site.startswith('o@'):
                a, b, name = self.reassemble(site, a), self.reassemble(site, b), site + '.weight'
            sums = torch.stack([(a - b).square().sum(), (a * b).sum(), a.square().sum(), b.square().sum()])
            self.split[name] = sums, paired * n
        return {s: error / draws for s, error in errors.items()}

    def multiplexed_errors(self, groups, passes, start, kind='hadamard'):
        """Estimate several groups' output errors from shared, sign-coded forward passes (method C).

        groups: [(sites, sigma), ...], all in block ``start`` or the one after it -- only groups that
        share their rerun save compute when multiplexed, and every pass reruns from ``start``.
        passes: forward evaluations in total. Each one perturbs every group, so each group gets
        ``passes`` draws for ``passes`` forward passes, where separate evaluation needs
        ``len(groups) * passes``.

        The passes run in code sequences of M = code_length(len(groups)). Within a sequence every
        site keeps one noise field a and pass m applies sigma * C[m, g] * a (multiplexing.codebook;
        'rademacher' redraws its signs for every sequence). A group's error is its code column
        correlated with the per-token loss drops, times a: the same estimator as direct_errors with
        the reward decoded across the sequence. The correlation uses the centered column, which is
        DUST's reward centering with the code sequence as the centering group.
        """
        n, order = self.settings.nrep, code_length(len(groups))
        if passes % order or passes % n or (order % n and n % order):
            raise ValueError(f'{passes} passes cannot be run as code sequences of {order} in chunks of {n}')
        layers = [layer_of(site) for sites, _ in groups for site in sites]
        if None in layers or min(layers) != start or max(layers) > start + 1:
            raise ValueError('Multiplexed groups must sit in the start block or the one after it')
        x, y = self.x.repeat(n, 1), self.y.repeat(n, 1)
        resume = None
        if self.suffix:
            resume = start, self.streams[start].repeat(n, 1, 1), self.streams[0].repeat(n, 1, 1)
        placed = {site: (g, sigma) for g, (sites, sigma) in enumerate(groups) for site in sites}
        width = lambda s: self.model.config.n_embd if s.startswith('o@') else self.outputs[s].shape[-1]
        errors = {s: torch.zeros(self.batch, self.length, width(s), device=self.device) for s in placed}
        fixed = codebook(len(groups), kind).to(self.device) if kind == 'hadamard' else None
        fields, codes = {}, {}  # per running code sequence: every site's noise field, and the code

        for forward in range(passes // n):
            index = range(forward * n, forward * n + n)
            for sequence in dict.fromkeys(p // order for p in index):
                if sequence not in fields:
                    fields[sequence] = {s: self.noise(self.batch, self.length, width(s)) for s in placed}
                    codes[sequence] = fixed if fixed is not None else codebook(len(groups), kind, self.noise)
            signs = torch.stack([codes[p // order][p % order] for p in index])                # [n, groups]
            weights = torch.stack([centered(codes[p // order])[p % order] for p in index])     # [n, groups]
            noise = {s: torch.stack([fields[p // order][s] for p in index]) for s in placed}   # [n, b, t, d]
            self.jitter = {
                s: (sigma * signs[:, g].view(n, 1, 1, 1).to(self.dtype) * noise[s]).view(n * self.batch, self.length, -1)
                for s, (g, sigma) in placed.items()
            }
            try:
                perturbed_loss = self.losses(x, y, resume).view(n, self.batch, self.length)
            finally:
                self.jitter = {}

            reward = self.clean_loss.unsqueeze(0) - perturbed_loss
            for s, (g, sigma) in placed.items():
                errors[s] -= torch.einsum(
                    'nbt,nbtd->btd', (reward * weights[:, g].view(n, 1, 1)).to(self.dtype), noise[s],
                ).float() / sigma
            for sequence in [q for q in fields if (q + 1) * order <= (forward + 1) * n]:
                del fields[sequence], codes[sequence]

        return {s: error / passes for s, error in errors.items()}

    def split_statistics(self):
        """The last estimate's split-half measurement, per parameter (or o@ hub site). Empty unless
        measure_split is on. One host sync; nothing is read back inside the draw loops.

        With the estimate g = S / K (S the sum over K draws) and halves A, B of p draws in total:
        - ``variance``: per-draw variance v, where Var(g) = v / K. From E|A - B|^2 = p v.
        - ``signal``: |E g|^2, the squared norm of what is being estimated. From E[a . b], a = A / (p/2).
        - ``cosine``: cosine between the two halves (the proposal's gradient agreement).
        - ``draws``: p, summed over GPUs.
        Both are in the estimate's raw units (summed over tokens, before the division by valid tokens).
        Across GPUs the sums are added, so every rank sees the same numbers; the cosine is then of the
        per-GPU halves, pooled.
        """
        if not self.split:
            return {}
        names = list(self.split)
        sums = torch.stack([self.split[name][0] for name in names])
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(sums, op=torch.distributed.ReduceOp.SUM)
        statistics = {}
        for name, (difference, dot, norm_a, norm_b) in zip(names, sums.tolist()):
            paired = self.split[name][1]  # per GPU
            statistics[name] = dict(
                variance=difference / (self.world * paired),
                signal=dot / (self.world * (paired / 2) ** 2),
                cosine=dot / math.sqrt(norm_a * norm_b) if norm_a * norm_b > 0 else 0.0,
                draws=self.world * paired,
            )
        return statistics

    def head_error(self, draws, sigma=0.05):
        """Perturb one vocabulary slab at a time; recompute CE from its changed logits.

        Each chunk jitters one 256-column slab of the cached logits, and the loss is recomputed from the
        cached sums of the other slabs. This is much cheaper than a forward pass, and each reward only
        reflects 256 jittered columns instead of all 4096.
        """
        n, b, t = self.settings.nrep, self.batch, self.length
        raw = self.outputs['lm_head']
        vocab, slab = self.model.config.vocab_size, 256
        slabs = vocab // slab
        logits = 15 * torch.tanh(raw[..., :vocab].float() / 15)
        slab_sums = logits.exp().view(b, t, slabs, slab).sum(-1)
        total = slab_sums.sum(-1)
        target = self.y.clamp(min=0)
        clean_target = logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        target = target.repeat(n, 1)
        error = torch.zeros_like(raw, dtype=torch.float32)

        for chunk in range(draws // n):
            block = chunk % slabs
            lo = block * slab
            a = self.noise(n * b, t, slab)
            changed = raw[..., lo:lo + slab].float().repeat(n, 1, 1) + sigma * a.float()
            changed = 15 * torch.tanh(changed / 15)
            partition = (total - slab_sums[..., block]).repeat(n, 1) + changed.exp().sum(-1)
            in_slab = (target >= lo) & (target < lo + slab)
            target_logit = torch.where(
                in_slab,
                changed.gather(
                    -1, (target - lo).clamp(0, slab - 1).unsqueeze(-1),
                ).squeeze(-1),
                clean_target.repeat(n, 1),
            )
            loss = (partition.log() - target_logit).view(n, b, t)
            reward = torch.where(self.y.unsqueeze(0) != -1, self.clean_loss.unsqueeze(0) - loss, 0)
            if b > 1:
                reward = reward - reward.mean(1, keepdim=True)
            error[..., lo:lo + slab] -= torch.einsum(
                'nbt,nbtd->btd', (reward / sigma).to(self.dtype), a.view(n, b, t, slab),
            ).float()

        return error / (draws // slabs)

    def local_errors(self, kind, draws, targets, sigma=0.05):
        """Score attention-output changes against estimated output errors (no derivatives).

        Each draw recomputes the block's attention output from the cached clean activations and scores the
        change per head against the output's estimated error (`targets`, computed by direct_errors on the 
        o@{layer} sites). Queries keep their own token's score; the other kinds sum the scores of later
        tokens, decayed by gamma per token.
        """
        cfg = self.model.config
        n, b, t, h, d = self.settings.nrep, self.batch, self.length, cfg.n_head, cfg.n_embd // cfg.n_head
        site_suffix = {'q': 'attn.c_q', 'k': 'attn.c_k', 'v': 'attn.c_v', 'gate': 'attn.ve_gate'}
        layers = [i for i in range(cfg.n_layer) if kind not in ('ve', 'gate') or has_ve(i, cfg.n_layer)]
        sites = {
            i: f'value_embeds.{i}' if kind == 've' else f'transformer.h.{i}.{site_suffix[kind]}'
            for i in layers
        }
        width = h if kind == 'gate' else cfg.n_embd
        errors = {i: torch.zeros(b, t, width, device=self.device) for i in layers}
        cos, sin = self.model.cos[:, :t].float(), self.model.sin[:, :t].float()

        def processed(raw):
            return norm(apply_rotary_emb(raw, cos, sin)).to(self.dtype)

        def repeat(x):
            return x.unsqueeze(0).expand(n, -1, -1, -1, -1).reshape(n * b, t, h, d)

        clean = {}
        for i in layers:
            prefix = f'transformer.h.{i}.attn.'
            q, k, v = [self.outputs[prefix + 'c_' + name].float().view(b, t, h, d) for name in ('q', 'k', 'v')]
            z = self.outputs[prefix + 've_gate'].float() if has_ve(i, cfg.n_layer) else None
            ve = self.outputs[f'value_embeds.{i}'].float().view(b, t, h, d) if z is not None else None
            values = v if z is None else v + (2 * torch.sigmoid(z)).unsqueeze(-1) * ve
            qp, kp = processed(q), processed(k)
            window = self.model.window_sizes[i][0]
            baseline = attention(qp, kp, values.to(self.dtype), window)
            positions = torch.arange(t, device=self.device)
            lag = positions[None, :] - positions[:, None]  # [perturbation position, output position]
            # credit[t, s]: how much of output s's score goes to the jitter at t, inside the causal window
            credit = ((lag >= 0) & (lag <= window)).float() * self.settings.gamma ** lag.clamp(min=0)
            clean[i] = q, k, z, ve, qp, kp, values, baseline, credit

        for _ in range(draws // n):
            for i in layers:
                a = self.noise(n, b, t, width, dtype=torch.float32)
                q, k, z, ve, qp, kp, values, baseline, credit = clean[i]
                window = self.model.window_sizes[i][0]
                divisor = sigma
                if kind in ('q', 'k'):
                    raw = q if kind == 'q' else k
                    perturbed = processed(
                        (raw.unsqueeze(0) + sigma * a.view(n, b, t, h, d)).reshape(n * b, t, h, d)
                    )
                    qa, ka = (perturbed, repeat(kp)) if kind == 'q' else (repeat(qp), perturbed)
                    delta = (
                        attention(qa, ka, repeat(values.to(self.dtype)), window).view(n, b, t, h, d)
                        - baseline
                    )
                else:
                    if kind == 'gate':
                        delta_gate = 2 * torch.sigmoid(z + sigma * a) - 2 * torch.sigmoid(z - sigma * a)
                        delta_value = delta_gate.unsqueeze(-1) * ve
                        divisor = 2 * sigma
                    else:
                        delta_value = sigma * a.view(n, b, t, h, d)
                        if kind == 've':
                            delta_value = delta_value * (2 * torch.sigmoid(z)).unsqueeze(-1)
                    delta = attention(
                        repeat(qp), repeat(kp),
                        delta_value.reshape(n * b, t, h, d).to(self.dtype), window,
                    ).view(n, b, t, h, d)

                score = torch.einsum('bthd,nbthd->nbth', targets[i].view(b, t, h, d).to(self.dtype), delta)
                if kind != 'q':
                    score = torch.einsum('nbsh,ts->nbth', score.float(), credit)
                estimate = torch.einsum('nbth,nbthd->bthd', score.float(), a.view(n, b, t, h, -1))
                errors[i] += estimate.reshape(b, t, width) / divisor

        return {sites[i]: errors[i] / draws for i in layers}

    def reassemble(self, site, error):
        """The exact local linear map: error times input, or scatter-add for an embedding. """
        x, module = self.inputs[site], self.modules[site]
        if isinstance(module, nn.Embedding):
            weight_error = torch.zeros_like(module.weight, dtype=torch.float32)
            return weight_error.index_add_(0, x.reshape(-1), error.reshape(-1, error.shape[-1]))
        return torch.einsum('btd,bti->di', error, x)

    def scalar_gradients(self, draws, sigma=0.03):
        """Two-sided weight-space ES for the 2L residual mixing scalars, rewarded by the summed token loss."""
        parameters = [self.model.resid_lambdas, self.model.x0_lambdas]
        original = [p.clone() for p in parameters]
        estimate = torch.zeros(2, self.model.config.n_layer, device=self.device)
        try:
            for _ in range(draws):
                a = self.noise(2, self.model.config.n_layer, dtype=torch.float32)
                losses = []
                for sign in (1, -1):
                    for p, clean, direction in zip(parameters, original, a):
                        p.copy_(clean + sign * sigma * direction)
                    losses.append(self.losses(self.x, self.y))
                difference = (losses[0] - losses[1])[self.y != -1].sum() / (2 * sigma)
                estimate += difference * a / draws
        finally:
            for p, clean in zip(parameters, original):
                p.copy_(clean)
        return dict(zip(('resid_lambdas', 'x0_lambdas'), estimate))

    @torch.no_grad()
    def estimate(self, x, y):
        """One clean pass plus the forward-only gradient estimate, without touching the weights.

        Returns (gradients, valid_tokens, clean_loss). ``gradients`` maps each parameter name to
        its estimate summed over tokens; dividing by ``valid_tokens`` gives the gradient of the
        mean loss, the same normalization the backprop baseline uses. Diagnostics and the adaptive
        allocator call this and inspect the estimate; apply() consumes it to take the SGD step.
        estimate() followed by apply() is identical to the old fused step().
        """
        # record_function labels name each phase in the profiler table and the Perfetto trace, so the
        # trace reads as direct / embedding / hubs / head / local / scalars (Lecture 3). They are inert
        # when no profiler is active, so they stay in the hot path.
        with record_function('dust/clean'):
            self.capture(x, y)
        r, world, split = self.settings, self.world, self.measure_split
        gradients, self.split = {}, {}
        with record_function('dust/direct'):
            for layer in range(self.model.config.n_layer):
                prefix = f'transformer.h.{layer}.'
                groups = [
                    ([prefix + 'attn.c_proj', prefix + 'mlp.c_proj'], r.writers[layer]),
                    ([prefix + 'mlp.c_fc'], r.hidden[layer]),
                ]
                if self.multiplex:  # method C: both groups ride the same coded passes
                    if r.writers[layer] != r.hidden[layer] or split:
                        raise ValueError('Multiplexing needs equal writer and hidden draws, and no split measurement')
                    coded = [(sites, 0.2) for sites, _ in groups]
                    found = self.multiplexed_errors(coded, r.writers[layer] // world, layer, self.multiplex)
                    for site, error in found.items():
                        gradients[site + '.weight'] = self.reassemble(site, error)
                    continue
                for sites, count in groups:
                    for site, error in self.direct_errors(sites, count // world, 0.2, layer, split).items():
                        gradients[site + '.weight'] = self.reassemble(site, error)

        with record_function('dust/embedding'):
            site = 'transformer.wte'
            error = self.direct_errors([site], r.embedding // world, 0.2, split=split)[site]
            gradients[site + '.weight'] = self.reassemble(site, error)

        # Attention outputs, each block alone. The deeper half moves the loss too little at 0.2 for a readable
        # reward, so it uses 0.4. The targets are averaged over GPUs so that every GPU scores the attention
        # internals against the estimate from the whole population.
        with record_function('dust/hubs'):
            targets = {}
            for layer in range(self.model.config.n_layer):
                site = f'o@{layer}'
                sigma = 0.2 if layer < self.model.config.n_layer // 2 else 0.4
                errors = self.direct_errors([site], r.hubs[layer] // world, sigma, layer, split)
                targets[layer] = average_ranks(errors[site])

        with record_function('dust/head'):
            gradients['lm_head.weight'] = self.reassemble('lm_head', self.head_error(r.head // world))
        with record_function('dust/local'):
            for kind, count in r.local.items():
                for site, error in self.local_errors(kind, count // world, targets).items():
                    gradients[site + '.weight'] = self.reassemble(site, error)
        with record_function('dust/scalars'):
            gradients.update(self.scalar_gradients(8 // world))

        valid_tokens = (y != -1).sum()
        return gradients, valid_tokens, self.clean_loss[y != -1].mean().item()

    @torch.no_grad()
    def apply(self, gradients, valid_tokens, optimizer):
        """Normalize the summed estimate to the mean-loss gradient, sync across ranks, take one SGD step."""
        optimizer.zero_grad(set_to_none=True)
        for name, parameter in self.model.named_parameters():
            gradient = average_ranks((gradients[name] / valid_tokens).to(parameter.dtype))
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f'Non-finite estimate for {name}')
            parameter.grad = gradient
        optimizer.step()

    @torch.no_grad()
    def step(self, x, y, optimizer):
        """One clean pass, forward-only estimate, then an SGD update (estimate followed by apply)."""
        gradients, valid_tokens, loss = self.estimate(x, y)
        self.apply(gradients, valid_tokens, optimizer)
        return loss


def load_sequences(path, count, sequence_len=2048):
    """Read a token matrix or chunked dataset, excluding padded rows."""
    data = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(data, dict):
        rows = [
            chunk.view(data['batch_size'], data['sequence_size'])[:valid]
            for chunk, valid in zip(data['chunks'], data['valid_counts'])
        ]
        data = torch.cat(rows)
    if not isinstance(data, torch.Tensor) or data.ndim != 2 or data.shape[1] != sequence_len + 1:
        raise ValueError(f'{path}: expected [sequences, {sequence_len + 1}] token IDs')
    if len(data) < count:
        raise ValueError(f'{path}: need {count} sequences, found {len(data)}')
    if data.is_floating_point() or data.is_complex() or data.dtype == torch.bool:
        raise ValueError(f'{path}: token IDs must have an integer dtype')
    data = data[:count].long()
    if data.min() < 0 or data.max() >= 4096:
        raise ValueError(f'{path}: token IDs must be in [0, 4096)')
    return data


@torch.no_grad()
def evaluate(model, rows, device, batch_size=32):
    total, tokens = 0.0, 0
    for batch in rows.split(batch_size):
        batch = batch.to(device)
        x, y = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
        context = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
        with context:
            loss = model(x, y)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite evaluation loss')
        total += loss.item() * y.numel()
        tokens += y.numel()
    return total / tokens


def make_model(config, device, seed):
    torch.manual_seed(seed)
    with torch.device('meta'):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    # init output projections at std 1.5 / sqrt(fan_in)
    # instead of the zeros init_weights sets.
    with torch.no_grad():
        for name, module in model.named_modules():
            if 'c_proj' in name and isinstance(module, nn.Linear):
                scale = 1.5 * math.sqrt(3 / module.weight.shape[1])
                nn.init.uniform_(module.weight, -scale, scale)
    return model.requires_grad_(False)


def make_optimizer(model, settings):
    groups = {}
    for name, parameter in model.named_parameters():
        lr = settings.lr
        if name == 'transformer.wte.weight':
            lr = 1000.0
        elif 'value_embeds' in name:
            lr = 0.3
        elif 'lambdas' in name:
            lr = 0.03
        groups.setdefault(lr, []).append(parameter)
    return torch.optim.SGD(
        [{'params': ps, 'lr': lr} for lr, ps in groups.items()],
        momentum=settings.momentum,
    )


def make_allocator(model, optimizer, config, settings, *, world=1, batch_size=8, suffix=True,
                   budget='draws', objective='update', **options):
    """The adaptive allocator for a run (new_implementation/allocation.py, Track 2).

    budget: 'draws' keeps the number of reallocated draws fixed; 'flops' keeps their analytic cost
    fixed instead, so an expensive shallow draw has to earn more than a cheap deep one.
    objective: 'update' weighs each parameter's noise by its learning rate, i.e. the noise the SGD
    step actually injects (the token embedding's rate is 1000, the blocks' about 0.25); 'gradient'
    weighs the raw gradient noise equally. It matters to the 'variance' policy only; 'agreement' is
    scale-free. The remaining ``options`` (policy, interval, ema, rate, minimum) go to Allocator.
    """
    if budget not in ('draws', 'flops') or objective not in ('update', 'gradient'):
        raise ValueError("budget must be 'draws' or 'flops', objective 'update' or 'gradient'")
    weights = None
    if objective == 'update':
        rates = {id(p): group['lr'] for group in optimizer.param_groups for p in group['params']}
        weights = {name: rates[id(p)] for name, p in model.named_parameters()}
    costs = draw_costs(config, batch_size, suffix) if budget == 'flops' else None
    return Allocator(settings, world=world, weights=weights, costs=costs, **options)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def data_hash(*tensors):
    """SHA-256 of token tensors, so a resumed run can check it sees the same data."""
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def snapshot(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def train(config, settings, training, validation, testing, output, *, steps, eval_interval, seed=42,
          device=torch.device('cpu'), rank=0, world=1, suffix=True, batch_size=8, checkpoint_interval=0,
          resume=False, probe_interval=0, probe_seeds=(0,), allocation=None, multiplex=None, metadata=None,
          wandb=None, echo=print):
    """Train with DUST and write the run to ``output``; returns result.json's contents (rank 0) or None.

    ``training`` rows are consumed in order, ``batch_size`` per step; every rank sees the same batch and
    only the perturbation population is split across ranks. Files written (rank 0 unless noted):

    - config.json: ``metadata`` (written once, on a fresh run);
    - metrics.jsonl: step 0's val loss, then one run_metrics record per step (cost, time, val loss on
      eval steps, the gradient probe on probe steps);
    - best.pt: weights at the best validation loss; result.json: best val, test at best val, history,
      timing summary;
    - state.pt (every rank, every ``checkpoint_interval`` steps and at the end): resumable full state.
      ``resume=True`` continues from it, refusing a different model, recipe, seed, world size or data,
      and rewinds metrics.jsonl to the checkpoint so a crashed tail is not duplicated.

    ``probe_interval`` > 0 scores the estimate against backprop on the first ``batch_size`` validation
    rows every that many steps (run_metrics.gradient_probe); it does not change the training run.

    ``wandb`` (rank 0): a dict of tracking.WandbLogger options (project, entity, group, ...) mirrors
    every record to Weights & Biases; resume continues the same W&B run.

    ``allocation`` (a dict of make_allocator options, ``{}`` for the defaults) turns on adaptive
    allocation (method B): ``settings`` is then only the starting allocation and the budget. Each
    step's noise is measured from its own draws, and the allocation it yields is used from the next
    step on. metrics.jsonl gets ``allocation`` on the first step run with a new allocation, and
    ``allocation_noise`` (per-unit per-draw noise) and ``allocation_agreement`` (per-unit expected
    half-to-half cosine) on the step that decided it. The measurement and the decision are inside the
    timed step, so their overhead is in ``train_seconds``; the decision alone is also totalled as
    ``allocation_seconds`` in result.json's timing.

    ``multiplex`` ('hadamard' or 'rademacher') turns on method C: each block's writers and hidden layer
    are estimated from shared sign-coded passes (DUST.multiplexed_errors). The logged cost charges the
    shared passes once and still counts both groups' draws. It cannot be combined with ``allocation``
    yet (that is variant D).
    """
    if allocation is not None and multiplex is not None:
        raise ValueError('Adaptive allocation with multiplexing (variant D) is not built yet')
    output = Path(output)
    state_file, metrics_file = output / 'state.pt', output / 'metrics.jsonl'
    fingerprint = dict(model=asdict(config), recipe=asdict(settings), seed=seed, world=world,
                       suffix=suffix, batch_size=batch_size, data_sha256=data_hash(training, validation),
                       allocation=allocation, multiplex=multiplex)

    model = make_model(config, device, seed)
    optimizer = make_optimizer(model, settings)
    allocator = None
    if allocation is not None:
        allocator = make_allocator(model, optimizer, config, settings, world=world, batch_size=batch_size,
                                   suffix=suffix, **allocation)
    estimator = DUST(model, allocator.recipe(settings) if allocator else settings, seed, rank, world,
                     suffix=suffix, multiplex=multiplex)
    estimator.measure_split = allocator is not None
    cost = partial(dust_update_cost, config=config, world=world, batch=batch_size, suffix=suffix,
                   multiplex=multiplex is not None)
    run_log = TrainingLog(cost, world=world)

    if resume:
        if not checkpoint_path(state_file, rank, world).is_file():
            raise FileNotFoundError(f'Nothing to resume: {checkpoint_path(state_file, rank, world)} does not exist')
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Choose an empty output directory: {output}')
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    if world > 1:
        torch.distributed.barrier()
    tracker = WandbLogger(output, metadata, resume=resume, **wandb) if wandb and rank == 0 else None

    def save_best():
        if rank == 0:
            torch.save(best_state, output / 'best.pt')

    def write(record):
        if rank == 0:
            echo(json.dumps(record))
            with metrics_file.open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            if tracker is not None:
                tracker.log(record)

    if resume:
        start, allocator_state, extra = load_state(state_file, model=model, optimizer=optimizer,
                                                   estimator=estimator, rank=rank, world=world)
        changed = sorted(k for k in fingerprint if extra['fingerprint'].get(k) != fingerprint[k])
        if changed:
            raise ValueError(f'Cannot resume {output}: the run differs from the checkpoint in {changed}')
        if allocator is not None:
            allocator.load_state_dict(allocator_state)
            estimator.settings = allocator.recipe(settings)
        best, best_step, best_state, history = extra['best'], extra['best_step'], extra['best_state'], extra['history']
        run_log.load_state_dict(extra['log'])
        if rank == 0:
            save_best()  # best.pt may hold a better step from after the checkpoint, which is about to rerun
            kept = [line for line in metrics_file.read_text().splitlines() if json.loads(line)['step'] <= start]
            metrics_file.write_text(''.join(line + '\n' for line in kept))
        echo(f'Resuming {output} at step {start} of {steps}, best val {best:.4f}')
    else:
        if rank == 0 and metadata is not None:
            (output / 'config.json').write_text(json.dumps(metadata, indent=2) + '\n')
        start = best_step = 0
        with run_log.timed('eval'):
            best = evaluate(model, validation, device)
        best_state = snapshot(model) if rank == 0 else None
        save_best()
        history = [dict(step=0, val_loss=best)]
        write(dict(step=0, val_loss=best, flops=0.0, forward_equivalents=0.0, draws=0,
                   train_seconds=0.0, gpu_hours=0.0))

    if probe_interval:
        rows = validation[:batch_size].to(device)
        probe_x, probe_y = rows[:, :-1].contiguous(), rows[:, 1:].contiguous()

    for step in range(start + 1, steps + 1):
        batch = training[(step - 1) * batch_size:step * batch_size].to(device)
        reallocated = False
        with run_log.step():
            loss = estimator.step(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), optimizer)
            if allocator is not None:
                with run_log.timed('allocation'):  # the decision's own cost; the measurement is in the step
                    allocator.observe(estimator.split_statistics())
                    reallocated = allocator.update(step)
        record = run_log.record(step, loss, estimator.settings)  # charged the allocation this step ran
        if reallocated:
            estimator.settings = allocator.recipe(estimator.settings)
            record['allocation_noise'] = allocator.sigmas()
            record['allocation_agreement'] = allocator.agreement()
        if step == 1 or step % eval_interval == 0 or step == steps:
            with run_log.timed('eval'):
                value = evaluate(model, validation, device)
            record['val_loss'] = value
            if value < best:
                best, best_step = value, step
                best_state = snapshot(model) if rank == 0 else None
                save_best()
        if probe_interval and (step % probe_interval == 0 or step == steps):
            with run_log.timed('probe'):
                record['probe'] = gradient_probe(estimator, probe_x, probe_y, probe_seeds, sync=world > 1)
        history.append(record)
        write(record)
        if checkpoint_interval and (step % checkpoint_interval == 0 or step == steps):
            with run_log.timed('checkpoint'):
                save_state(
                    state_file, model=model, optimizer=optimizer, estimator=estimator, step=step,
                    settings=asdict(estimator.settings), rank=rank, world=world,
                    allocator_state=allocator.state_dict() if allocator else None,
                    extra=dict(fingerprint=fingerprint, best=best, best_step=best_step, best_state=best_state,
                               history=history, log=run_log.state_dict()),
                )

    estimator.close()
    if rank != 0:
        return None
    model.load_state_dict(torch.load(output / 'best.pt', map_location=device, weights_only=True))
    result = dict(best_val=best, best_step=best_step, test_at_best_val=evaluate(model, testing, device),
                  history=history, timing=run_log.summary())
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    if tracker is not None:
        tracker.finish(result)
    echo(f"Best validation: {best:.4f} (step {best_step}); test: {result['test_at_best_val']:.4f}")
    return result


def wandb_arguments(parser):
    """--wandb PROJECT and friends; shared with baselines/backprop.py."""
    parser.add_argument('--wandb', metavar='PROJECT',
                        help='Also log to Weights & Biases under this project (run `wandb login` first).')
    parser.add_argument('--wandb-entity', help='W&B team or user the project lives under.')
    parser.add_argument('--wandb-group', help='W&B group, e.g. one per sweep or method.')
    parser.add_argument('--wandb-name', help='W&B run name (default: the output directory name).')
    parser.add_argument('--wandb-tags', nargs='*', help='W&B tags, e.g. A 1M K256.')


def wandb_options(args):
    if not args.wandb:
        return None
    return dict(project=args.wandb, entity=args.wandb_entity, group=args.wandb_group, name=args.wandb_name,
                tags=args.wandb_tags)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', choices=DEFAULT_CONFIGS, default='1M')
    parser.add_argument(
        '--population', type=int, choices=DRAW_CHUNK_SIZES, default=16384,
        help='Total direct-loss draws across GPUs (default: 16384); each value selects its tuned recipe.',
    )
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('runs/default'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--steps', type=int,
        help='Stop early for a smoke test; the training subset stays unchanged.',
    )
    parser.add_argument(
        '--full-replay', action='store_true',
        help='Rerun the whole network for every direct draw, as the public code does (upstream reference).',
    )
    parser.add_argument(
        '--toy', action='store_true',
        help='Tiny model, recipe and synthetic data (new_implementation/toy.py): a CPU smoke run in seconds. '
             'Ignores --tokens, --population and --data-dir.',
    )
    parser.add_argument('--device', help='cuda (default), cpu or mps. Only --toy runs are practical off CUDA.')
    parser.add_argument('--checkpoint-interval', type=int, default=0,
                        help='Save resumable state to OUTPUT/state.pt every N steps and at the end (0: never).')
    parser.add_argument('--resume', action='store_true',
                        help='Continue the run in --output from its state.pt (same settings and seed).')
    parser.add_argument('--probe-interval', type=int, default=0,
                        help='Every N steps, log the cosine of the estimate to backprop on a held-out batch '
                             '(costs one backprop + one estimate per seed; 0: never).')
    wandb_arguments(parser)
    parser.add_argument('--probe-seeds', type=int, default=1,
                        help='Estimates per probe; 2 or more also log estimator variance and bias.')
    parser.add_argument('--adaptive', action='store_true',
                        help='Adaptive allocation (method B): move the writer, hidden and embedding draws to '
                             'where the estimate is noisiest. The recipe is the starting point and the budget.')
    parser.add_argument('--multiplex', choices=KINDS,
                        help='Structured multiplexing (method C): estimate each block\'s writers and hidden layer '
                             'from shared sign-coded passes. hadamard: exactly orthogonal codes; rademacher: '
                             'random signs, the comparison. Needs writer draws that are multiples of 4.')
    parser.add_argument('--allocation-policy', choices=('variance', 'agreement'), default='variance',
                        help='With --adaptive: draws in proportion to per-draw noise (variance), or to equalize '
                             'the cosine between half-population estimates (agreement; also moves the hubs).')
    parser.add_argument('--allocation-interval', type=int, default=4,
                        help='With --adaptive: recompute the allocation every N steps (0: measure only).')
    parser.add_argument('--allocation-budget', choices=('draws', 'flops'), default='draws',
                        help='With --adaptive: hold the reallocated draw count fixed, or its analytic FLOPs.')
    parser.add_argument('--allocation-objective', choices=('update', 'gradient'), default='update',
                        help='With --adaptive: weigh noise by learning rate (update) or not (gradient).')
    args = parser.parse_args()

    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() or not args.toy else 'cpu'
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            parser.error('Training requires a CUDA GPU (use --toy for a CPU smoke run).')
        device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
        torch.cuda.set_device(device)
    else:
        if world > 1:
            parser.error('Multi-process runs need CUDA.')
        device = torch.device(args.device)
    torch.set_float32_matmul_precision('high')
    if world > 1:
        torch.distributed.init_process_group('nccl', device_id=device)
    if min(args.checkpoint_interval, args.probe_interval) < 0 or args.probe_seeds < 1:
        parser.error('--checkpoint-interval and --probe-interval must be >= 0, --probe-seeds >= 1')
    if args.allocation_interval < 0:
        parser.error('--allocation-interval must be >= 0')
    if args.adaptive and args.multiplex:
        parser.error('--adaptive with --multiplex (variant D) is not built yet')
    allocation = None
    if args.adaptive:
        allocation = dict(policy=args.allocation_policy, interval=args.allocation_interval,
                          budget=args.allocation_budget,
                          objective=args.allocation_objective)

    if args.toy:
        from toy import TOY_BATCH, TOY_CONFIG, TOY_EVAL_INTERVAL, TOY_STEPS, toy_data, toy_recipe
        # The toy recipe has too few draws to keep two chunks per unit (adaptive) or to fill a code
        # sequence of 4 passes (multiplexing); doubled, it has enough.
        settings = toy_recipe(scale=2 if args.adaptive or args.multiplex else 1)
        config, steps, interval = TOY_CONFIG, TOY_STEPS, TOY_EVAL_INTERVAL
        training, validation, testing = toy_data(TOY_STEPS, TOY_BATCH)
        data = dict(source='toy (new_implementation/toy.py)')
    else:
        settings = recipe(args.tokens, args.population)
        defaults = DEFAULT_CONFIGS[args.tokens]
        config, steps, interval = GPTConfig(), defaults['steps'], defaults['eval_interval']
        paths = dict(
            train=args.data_dir / f'train_{args.tokens}.pt',
            val=args.data_dir / 'val.pt',
            test=args.data_dir / 'test.pt',
        )
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            parser.error(
                f"Missing data: {', '.join(missing)}. Run python prepare_data.py "
                f"--tokens {args.tokens} --output {args.data_dir}"
            )
        training = load_sequences(paths['train'], steps * 8)
        validation = load_sequences(paths['val'], 544)
        testing = load_sequences(paths['test'], 544)
        data = dict(sha256={k: file_hash(v) for k, v in paths.items()}) if rank == 0 else {}
    if args.steps is not None and not 1 <= args.steps <= steps:
        parser.error(f'--steps must be between 1 and {steps}')
    order = torch.randperm(len(training), generator=torch.Generator().manual_seed(args.seed + 1))
    training = training[order]
    steps = args.steps or steps

    metadata = dict(
        tokens='toy' if args.toy else args.tokens,
        population=sum(settings.writers + settings.hidden + settings.hubs) + settings.embedding,
        seed=args.seed,
        world=world,
        steps=steps,
        training_tokens=steps * 8 * config.sequence_len,
        validation_sequences=len(validation),
        test_sequences=len(testing),
        model=asdict(config),
        recipe=asdict(settings),
        allocation=allocation,  # None: fixed; else the adaptive options, and `recipe` is the starting point
        multiplex=args.multiplex,
        reward_centering='per-token local draw-chunk mean; skipped at nrep=1; no rescaling',
        execution='full-replay' if args.full_replay else 'cached-suffix',
        device=str(device),
        checkpoint_interval=args.checkpoint_interval,
        probe_interval=args.probe_interval,
        probe_seeds=args.probe_seeds,
        source_sha256=file_hash(Path(__file__)),
        model_sha256=file_hash(Path(__file__).parent / 'model.py'),
        torch_version=torch.__version__,
        data=data,
    )
    if rank == 0:
        print(f"{metadata['tokens']}: {steps} steps, population={metadata['population']}, GPUs={world}, "
              f'device={device}', flush=True)
    train(
        config, settings, training, validation, testing, args.output,
        steps=steps, eval_interval=interval, seed=args.seed, device=device, rank=rank, world=world,
        suffix=not args.full_replay, checkpoint_interval=args.checkpoint_interval, resume=args.resume,
        probe_interval=args.probe_interval, probe_seeds=tuple(range(args.probe_seeds)),
        allocation=allocation, multiplex=args.multiplex, metadata=metadata,
        wandb=wandb_options(args), echo=lambda line: print(line, flush=True),
    )
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
