"""The transformer DUST trains: a decoder-only GPT with value embeddings and a learned residual mix."""
import functools
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


# Fast path (off by default, so the upstream reference stays reproducible): set_fast(True) or --fast.
# The same math with fewer, fused kernels:
# - full-window blocks call SDPA with is_causal=True (no explicit mask, so the flash kernel can run);
#   sliding-window blocks use FlexAttention with a causal + window block mask, which skips the
#   masked-out tiles instead of computing the dense [T, T] scores;
# - RMSNorm feeding a Linear, rotary + q/k norm, relu^2, the residual mix, and the soft-capped
#   cross-entropy are each one compiled kernel on CUDA. Under bf16 autocast the norms emit bf16
#   directly: autocast would round them to bf16 for the next matmul anyway, so only the separate
#   cast kernels disappear. On CPU the same functions run eagerly (identical math, for tests).
FAST = False


def set_fast(enabled=True):
    global FAST
    FAST = bool(enabled)
    if FAST:  # one compiled graph per distinct shape (batch 8 clean, 8 * nrep draws, eval, probe)
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)


def _autocast_dtype(x):
    kind = x.device.type
    return torch.get_autocast_dtype(kind) if torch.is_autocast_enabled(kind) else None


@functools.lru_cache(maxsize=None)
def _compiled(fn):
    return torch.compile(fn, dynamic=False)


def _run(fn, *args):
    """fn compiled on CUDA, eager elsewhere."""
    return (_compiled(fn) if args[0].is_cuda else fn)(*args)


def _norm_cast(x, dtype):
    y = F.rms_norm(x, (x.size(-1),))
    return y if dtype is None else y.to(dtype)


def norm_for_matmul(x):
    """norm(x), in the dtype the following matmul runs in under autocast."""
    if not FAST:
        return norm(x)
    dtype = _autocast_dtype(x)
    return _run(_norm_cast, x, dtype) if dtype is not None and x.is_cuda else _norm_cast(x, dtype)


def _qk_prep(q, k, cos, sin, dtype):
    q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
    return _norm_cast(q, dtype), _norm_cast(k, dtype)


def _relu2(h):
    return F.relu(h).square()


def _mix(x, x0, a, b):
    return a * x + b * x0


def _capped_ce(logits, targets, vocab, reduction):
    logits = logits[..., :vocab].float()
    logits = 15 * torch.tanh(logits / 15)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1,
                           reduction=reduction)


@functools.lru_cache(maxsize=None)
def _window_block_mask(length, window, device):
    from torch.nn.attention.flex_attention import create_block_mask

    def keep(b, h, q_idx, kv_idx):
        distance = q_idx - kv_idx
        return (distance >= 0) & (distance <= window)

    return create_block_mask(keep, B=None, H=None, Q_LEN=length, KV_LEN=length, device=device)


def _fast_attention(q, k, v, window):
    length = q.shape[1]
    dtype = _autocast_dtype(v) or v.dtype
    q, k, v = (t.transpose(1, 2).to(dtype) for t in (q, k, v))
    if window >= length - 1:  # the window covers every earlier key: plain causal attention
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    else:
        from torch.nn.attention.flex_attention import flex_attention
        mask = _window_block_mask(length, window, str(q.device))
        flex = _compiled(flex_attention) if q.is_cuda else flex_attention
        out = flex(q, k, v, block_mask=mask)
    return out.transpose(1, 2)


def attention(q, k, v, window):
    """Causal attention with a left window; tensors have shape [batch, token, head, dim].

    A query at position s sees keys at s - window to s.
    """
    if FAST:
        return _fast_attention(q, k, v, window)
    length = q.shape[1]
    positions = torch.arange(length, device=q.device)
    distance = positions[:, None] - positions[None, :]
    mask = (distance >= 0) & (distance <= window)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask,
    )
    return out.transpose(1, 2)


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 4096
    n_layer: int = 8
    n_head: int = 8
    n_kv_head: int = 8
    n_embd: int = 512
    # Per block, repeating: S = half-sequence window, L = full. The last block is always L.
    window_pattern: str = 'SSSL'


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Value embeddings on every other block, counting back from the last."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = (x[..., :d], x[..., d:])
    return torch.cat([x1 * cos + x2 * sin, x1 * -sin + x2 * cos], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # The value embedding is scaled per head by 2 * sigmoid(gate), read from the block input's first 32
        # channels.
        self.ve_gate_channels = 32
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer) else None
        )

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        if FAST:
            q, k = _run(_qk_prep, q, k, cos, sin, _autocast_dtype(q))
        else:
            q, k = (apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin))
            q, k = (norm(q), norm(k))
        y = attention(q, k, v, window_size[0])
        y = y.contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        h = self.c_fc(x)
        return self.c_proj(_run(_relu2, h) if FAST else F.relu(h).square())


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        # Pre-norm block. attn.c_proj and mlp.c_proj are the "writers" into the residual stream.
        x = x + self.attn(norm_for_matmul(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm_for_matmul(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        padded_vocab = (config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to * pad_vocab_size_to
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(padded_vocab, config.n_embd),
            'h': nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab, bias=False)  # untied from the embedding
        # The stream entering block i is resid_lambdas[i] * x + x0_lambdas[i] * x0, a learned mix of the
        # previous block's output and the normalized token embedding. These 2L scalars are trained
        # with weight-space ES (DUST.scalar_gradients).
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(padded_vocab, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = 3 ** 0.5 * self.config.n_embd ** (-0.5)
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)  # make_model re-initializes both c_proj weights
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        (self.cos, self.sin) = (cos, sin)
        # Token and value embeddings are stored in bfloat16 on the GPU
        if self.transformer.wte.weight.device.type == 'cuda':
            self.transformer.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)

    def _precompute_rotary(self, seq_len, head_dim, base=10000):
        device = self.transformer.wte.weight.device
        inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = (freqs.cos().bfloat16(), freqs.sin().bfloat16())
        return (cos[None, :, None, :], sin[None, :, None, :])

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_w, short_w = (config.sequence_len, config.sequence_len // 2)
        char_to_w = {'L': (long_w, 0), 'S': (short_w, 0)}
        sizes = [char_to_w[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        sizes[-1] = (long_w, 0)
        return sizes

    def forward(self, idx, targets=None, loss_reduction='mean', streams=None):
        x0 = norm(self.transformer.wte(idx))
        return self.forward_from(0, x0, x0, idx, targets, loss_reduction, streams)

    def forward_from(self, start, x, x0, idx, targets=None, loss_reduction='mean', streams=None):
        """Run blocks start onward, then the head, from the residual stream x entering block start.

        x is the stream before block start's residual mix (the previous block's output, or x0 for
        block 0), so forward_from(0, x0, x0, idx) is the full forward. If streams is a list, the stream
        entering each block run is appended to it; after a full forward streams[i] enters block i and
        streams[0] is x0, which is what DUST caches to rerun only the blocks a perturbation reaches.
        """
        B, T = idx.size()
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        for i in range(start, self.config.n_layer):
            block = self.transformer.h[i]
            if streams is not None:
                streams.append(x)
            if FAST:
                x = _run(_mix, x, x0, self.resid_lambdas[i], self.x0_lambdas[i])
            else:
                x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm_for_matmul(x)
        if FAST and targets is not None:
            return _run(_capped_ce, self.lm_head(x), targets, self.config.vocab_size, loss_reduction)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        logits = 15 * torch.tanh(logits / 15)  # soft cap, repeated in DUST.head_error
        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-1, reduction=loss_reduction,
            )
        return logits
