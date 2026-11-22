"""CPU checks for the fast path (model.set_fast) and antithetic draws (DUST(antithetic=True)).

The fast path must be the same math: on CPU its functions run eagerly, so the forward and the DUST
estimate must match the plain path to fp32 rounding, including FlexAttention's sliding-window mask.
Antithetic draws must keep the estimate's scale (the learning rate means the same thing) and, at
nrep 2, point closer to the backprop gradient than independent draws at the same forward passes.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import model as model_module  # noqa: E402
import gradient_check as gc  # noqa: E402
from dust import DUST, make_model  # noqa: E402
from model import GPTConfig  # noqa: E402
from multiplex_experiment import experiment_recipe  # noqa: E402

CONFIG = GPTConfig(sequence_len=256, vocab_size=256, n_layer=4, n_head=2, n_kv_head=2, n_embd=64)


@pytest.fixture
def fast():
    model_module.set_fast(True)
    yield
    model_module.set_fast(False)


def batch(rows=2, seed=0):
    generator = torch.Generator().manual_seed(seed)
    data = torch.randint(0, CONFIG.vocab_size, (rows, CONFIG.sequence_len + 1), generator=generator)
    return data[:, :-1].contiguous(), data[:, 1:].contiguous()


def test_windows_exercise_both_attention_paths():
    model = make_model(CONFIG, torch.device('cpu'), seed=0)
    windows = [w for w, _ in model.window_sizes]
    assert min(windows) < CONFIG.sequence_len - 1 <= max(windows)  # flex and causal both run


@pytest.mark.parametrize('window', [0, 1, 37, 128, 255, 256])
def test_fast_attention_matches_masked(window, fast):
    generator = torch.Generator().manual_seed(window)
    q, k, v = (torch.randn(2, CONFIG.sequence_len, 2, 32, generator=generator) for _ in range(3))
    model_module.set_fast(False)
    plain = model_module.attention(q, k, v, window)
    model_module.set_fast(True)
    fused = model_module.attention(q, k, v, window)
    assert torch.allclose(plain, fused, atol=1e-5, rtol=1e-5), (plain - fused).abs().max()


def test_fast_forward_matches_plain():
    model = make_model(CONFIG, torch.device('cpu'), seed=0)
    x, y = batch()
    with torch.no_grad():
        plain = model(x, y, loss_reduction='none')
        model_module.set_fast(True)
        try:
            fused = model(x, y, loss_reduction='none')
            fused_mean = model(x, y)
        finally:
            model_module.set_fast(False)
    assert torch.allclose(plain, fused, atol=1e-5, rtol=1e-5), (plain - fused).abs().max()
    assert torch.allclose(plain.mean(), fused_mean, atol=1e-6)


def test_fast_estimate_matches_plain():
    model = make_model(CONFIG, torch.device('cpu'), seed=0)
    settings = experiment_recipe(2, CONFIG.n_layer)
    x, y = batch()

    def estimate():
        estimator = DUST(model, settings, seed=3)
        try:
            gradients, valid, _ = estimator.estimate(x, y)
        finally:
            estimator.close()
        return {n: g / valid for n, g in gradients.items()}

    plain = estimate()
    model_module.set_fast(True)
    try:
        fused = estimate()
    finally:
        model_module.set_fast(False)
    for name in plain:
        scale = plain[name].norm().clamp(min=1e-12)
        assert ((plain[name] - fused[name]).norm() / scale) < 1e-3, name  # loss drops amplify fp32 rounding


def test_antithetic_needs_even_nrep():
    model = make_model(CONFIG, torch.device('cpu'), seed=0)
    with pytest.raises(ValueError):
        DUST(model, experiment_recipe(1, CONFIG.n_layer), antithetic=True)


def block_writer_estimates(antithetic, seeds=12, layer=1, draws=8, nrep=2):
    model = make_model(CONFIG, torch.device('cpu'), seed=0)
    x, y = batch()
    reference, _ = gc.reference_gradients(model, x, y)
    sites = [f'transformer.h.{layer}.attn.c_proj', f'transformer.h.{layer}.mlp.c_proj']
    truth = torch.cat([reference[s + '.weight'].reshape(-1) for s in sites]).double()
    estimator = DUST(model, experiment_recipe(nrep, CONFIG.n_layer), antithetic=antithetic)
    rows = []
    try:
        estimator.capture(x, y)
        for seed in range(seeds):
            gc.reseed(estimator, seed)
            errors = estimator.direct_errors(sites, draws, 0.2, layer)
            rows.append(torch.cat([estimator.reassemble(s, errors[s]).reshape(-1) for s in sites]).double()
                        / (y != -1).sum())
    finally:
        estimator.close()
    stack = torch.stack(rows)
    cosine = torch.nn.functional.cosine_similarity(stack, truth.unsqueeze(0), dim=1).mean().item()
    projection = (stack.mean(0) @ truth / truth.square().sum()).item()  # mean estimate along the truth
    return cosine, projection


def test_antithetic_keeps_scale_and_improves_direction():
    independent_cosine, independent_scale = block_writer_estimates(False)
    antithetic_cosine, antithetic_scale = block_writer_estimates(True)
    # Both target (1 - 1/nrep) = 0.5 of the gradient along its direction.
    assert 0.3 < independent_scale < 0.7 and 0.3 < antithetic_scale < 0.7, (independent_scale, antithetic_scale)
    assert antithetic_cosine > independent_cosine, (antithetic_cosine, independent_cosine)
