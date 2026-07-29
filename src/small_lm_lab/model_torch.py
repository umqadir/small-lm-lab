"""Decoder-only transformer in pure PyTorch, built from scratch.

Architecture: pre-norm RMSNorm, RoPE on q and k, causal self-attention via
scaled_dot_product_attention(is_causal=True), SwiGLU MLP, no biases anywhere,
tied input/output embeddings, a final RMSNorm before the lm head.

RoPE convention: half-rotation (the GPT-NeoX / Llama "rotate_half" convention).
The head dimension is split into a first half and a second half; element i in
the first half pairs with element i in the second half and both are rotated by
angle pos * theta^(-2i/head_dim). This is implemented identically in the MLX
twin via mx.fast.rope(traditional=False). See model_mlx.py.

Interpretability capture: forward(tokens, capture=...) accepts an optional
dict. When provided, each block writes its per-head attention output (the
value-weighted sum before the output projection), shaped
[batch, seq, n_heads, head_dim], into capture["attn_per_head"][layer_index].
The model is a plain nn.Module, so forward hooks also work; the capture dict is
provided because it gives clean per-head access without extra bookkeeping.

Attention weights: scaled_dot_product_attention never materializes the
post-softmax weight matrix, and the pre-registered prefix-matching score is a
mean of attention weights, so there has to be a second way to get them. Setting
capture[WANT_ATTN_WEIGHTS] = True switches the attention to an analysis path
that builds the causal weights explicitly, applies them to v itself, and writes
them to capture[ATTN_WEIGHTS_KEY][layer_index], shaped
[batch, n_heads, seq, seq]. The two paths compute the same function, and
tests/test_interp.py asserts they agree on the block output to tight tolerance,
because an analysis path that quietly computed something else would invalidate
every reading taken through it.

The fast path is what runs whenever the flag is absent, which is every training
step: training passes no capture dict at all. The analysis path costs a
materialized [batch, n_heads, seq, seq] tensor per layer and is only worth
paying for when the weights themselves are the measurement.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

# Capture-dict keys. WANT_ATTN_WEIGHTS is an input flag, the other two name
# outputs. They live here rather than as bare strings in the analysis code so a
# typo is a NameError instead of a silently empty capture.
WANT_ATTN_WEIGHTS = "want_attn_weights"
ATTN_WEIGHTS_KEY = "attn_weights"
ATTN_PER_HEAD_KEY = "attn_per_head"


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm computed in float32 for stability, cast back to input dtype."""
    dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (x.to(dtype)) * weight


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm(x, self.weight, self.eps)


def build_rope_cache(
    seq_len: int, head_dim: int, theta: float, device, dtype=torch.float32
):
    """Return (cos, sin), each [seq_len, head_dim], for the half-rotation form.

    inv_freq[i] = theta^(-2i/head_dim) for i in [0, head_dim/2). The angles are
    concatenated with themselves so cos/sin span the full head_dim, matching the
    rotate_half application below.
    """
    half = head_dim // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, half, device=device, dtype=torch.float32) * 2.0 / head_dim)
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # [seq_len, half]
    emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x shaped [batch, n_heads, seq, head_dim].

    cos and sin are [seq, head_dim]; broadcast over batch and head dims.
    """
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + rotate_half(x) * sin


def causal_attention_weights(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Post-softmax causal attention weights for q, k shaped [b, n_heads, t, d].

    Returns [b, n_heads, t, t], row-stochastic, with the strict upper triangle
    exactly zero. This is the same function scaled_dot_product_attention applies
    internally with is_causal=True: the same 1/sqrt(head_dim) scale, the same
    lower-triangular mask, the same softmax over the key axis. It exists because
    that call returns only the weighted values and there is no way to ask it for
    the weights.
    """
    t = q.shape[-2]
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    mask = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.attn_dim = cfg.attn_dim
        self.q_proj = nn.Linear(cfg.d_model, self.attn_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.attn_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.attn_dim, bias=False)
        self.o_proj = nn.Linear(self.attn_dim, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        capture: Optional[dict] = None,
        layer_idx: int = -1,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # [b, n_heads, t, head_dim]. The fast path is the only one training ever
        # takes, since training passes no capture dict.
        if capture is not None and capture.get(WANT_ATTN_WEIGHTS):
            weights = causal_attention_weights(q, k)
            attn = torch.matmul(weights, v)
            capture.setdefault(ATTN_WEIGHTS_KEY, {})[layer_idx] = weights.detach()
        else:
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Per-head output before the output projection: [b, t, n_heads, head_dim].
        attn_per_head = attn.transpose(1, 2).contiguous()
        if capture is not None:
            capture.setdefault(ATTN_PER_HEAD_KEY, {})[layer_idx] = attn_per_head.detach()

        out = attn_per_head.view(b, t, self.attn_dim)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)  # gate
        self.w3 = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)  # up
        self.w2 = nn.Linear(cfg.mlp_hidden, cfg.d_model, bias=False)  # down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rmsnorm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.rmsnorm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        capture: Optional[dict] = None,
        layer_idx: int = -1,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, capture, layer_idx)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.rmsnorm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # RoPE cache is not a parameter; register as a buffer so it moves with .to().
        cos, sin = build_rope_cache(cfg.context_len, cfg.head_dim, cfg.rope_theta, "cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scale output projections by 1/sqrt(2 * n_layers).
        scale = 1.0 / math.sqrt(2 * cfg.n_layers)
        for block in self.blocks:
            with torch.no_grad():
                block.attn.o_proj.weight.mul_(scale)
                block.mlp.w2.weight.mul_(scale)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, tokens: torch.Tensor, capture: Optional[dict] = None
    ) -> torch.Tensor:
        b, t = tokens.shape
        cos = self.rope_cos[:t]
        sin = self.rope_sin[:t]
        x = self.embed(tokens)
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, capture, i)
        x = self.final_norm(x)
        return self.lm_head(x)
