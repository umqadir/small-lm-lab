"""Decoder-only transformer in MLX, the exact twin of model_torch.py.

Architecture: pre-norm RMSNorm, RoPE on q and k, causal self-attention via
mx.fast.scaled_dot_product_attention with a causal mask, SwiGLU MLP, no biases
anywhere, tied input/output embeddings, a final RMSNorm before the lm head.

RoPE convention: half-rotation (the GPT-NeoX / Llama "rotate_half" convention).
This is mx.fast.rope(traditional=False). It was verified numerically to match
the torch apply_rope in model_torch.py to about 1e-7 on random inputs (see
tests/test_equivalence.py and convert.logit_max_abs_diff). The interleaved
GPT-J form (traditional=True) does NOT match and is deliberately not used.

Parameter names mirror model_torch.py exactly so weights transfer one to one:
  embed.weight
  blocks.{i}.attn_norm.weight
  blocks.{i}.attn.{q,k,v,o}_proj.weight
  blocks.{i}.mlp_norm.weight
  blocks.{i}.mlp.{w1,w3,w2}.weight
  final_norm.weight
MLX and torch Linear both store weight as [out, in] and compute x @ weight.T, so
projection weights copy across with no transpose. The lm head reuses the tied
embedding via Embedding.as_linear, so no separate lm_head weight exists when
tie_embeddings is True.
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.attn_dim = cfg.attn_dim
        self.scale = cfg.head_dim ** -0.5
        self.rope_theta = cfg.rope_theta
        self.q_proj = nn.Linear(cfg.d_model, self.attn_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.attn_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.attn_dim, bias=False)
        self.o_proj = nn.Linear(self.attn_dim, cfg.d_model, bias=False)

    def __call__(
        self, x: mx.array, capture: Optional[dict] = None, layer_idx: int = -1
    ) -> mx.array:
        b, t, _ = x.shape
        # [b, n_heads, t, head_dim]
        q = self.q_proj(x).reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # RoPE, half-rotation convention (matches torch rotate_half).
        q = mx.fast.rope(
            q, dims=self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=0
        )
        k = mx.fast.rope(
            k, dims=self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=0
        )

        # Causal mask. For square self-attention (T_q == T_kv) the lower-right
        # aligned "causal" mask is the standard lower-triangular causal mask.
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")

        # Per-head output before the output projection: [b, t, n_heads, head_dim].
        attn_per_head = attn.transpose(0, 2, 1, 3)
        if capture is not None:
            capture.setdefault("attn_per_head", {})[layer_idx] = attn_per_head

        out = attn_per_head.reshape(b, t, self.attn_dim)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)  # gate
        self.w3 = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)  # up
        self.w2 = nn.Linear(cfg.mlp_hidden, cfg.d_model, bias=False)  # down

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rmsnorm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.rmsnorm_eps)
        self.mlp = SwiGLU(cfg)

    def __call__(
        self, x: mx.array, capture: Optional[dict] = None, layer_idx: int = -1
    ) -> mx.array:
        x = x + self.attn(self.attn_norm(x), capture, layer_idx)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = [Block(cfg) for _ in range(cfg.n_layers)]
        self.final_norm = RMSNorm(cfg.d_model, cfg.rmsnorm_eps)
        if not cfg.tie_embeddings:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        """Init: normal std 0.02 for all Linear/Embedding weights, ones for norms.

        The two residual-output projections per block (attention o_proj and MLP
        w2 down projection) are additionally scaled by 1/sqrt(2*n_layers), the
        same scheme used in model_torch.py.
        """
        n_layers = self.cfg.n_layers
        residual_scale = 1.0 / math.sqrt(2 * n_layers)
        new_params: dict[str, mx.array] = {}
        for name, p in tree_flatten(self.parameters()):
            if name.endswith("_norm.weight"):
                val = mx.ones(p.shape)
            else:
                # Linear and Embedding weights.
                val = mx.random.normal(shape=p.shape) * 0.02
            new_params[name] = val
        for i in range(n_layers):
            o = f"blocks.{i}.attn.o_proj.weight"
            w2 = f"blocks.{i}.mlp.w2.weight"
            new_params[o] = new_params[o] * residual_scale
            new_params[w2] = new_params[w2] * residual_scale
        self.load_weights(list(new_params.items()))

    def __call__(
        self, tokens: mx.array, capture: Optional[dict] = None
    ) -> mx.array:
        x = self.embed(tokens)
        for i, block in enumerate(self.blocks):
            x = block(x, capture, i)
        x = self.final_norm(x)
        if self.cfg.tie_embeddings:
            return self.embed.as_linear(x)
        return self.lm_head(x)
