"""Equivalence tests for the torch and MLX transformer twins.

Covers exact parameter accounting in both frameworks, float32 logit equivalence
after sharing weights, and causal masking in both frameworks.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import mlx.core as mx
from mlx.utils import tree_flatten

from small_lm_lab.config import get_config, n_params, registered_configs
from small_lm_lab import model_torch, model_mlx, convert


def _torch_param_count(model: model_torch.TransformerLM) -> int:
    # parameters() deduplicates the tied embedding, matching n_params.
    return sum(p.numel() for p in model.parameters())


def _mlx_param_count(model: model_mlx.TransformerLM) -> int:
    return int(sum(a.size for _, a in tree_flatten(model.parameters())))


@pytest.mark.parametrize("name", sorted(registered_configs().keys()))
def test_param_counts_match_torch(name: str) -> None:
    cfg = get_config(name)
    model = model_torch.TransformerLM(cfg)
    assert _torch_param_count(model) == n_params(cfg)


@pytest.mark.parametrize("name", sorted(registered_configs().keys()))
def test_param_counts_match_mlx(name: str) -> None:
    cfg = get_config(name)
    model = model_mlx.TransformerLM(cfg)
    assert _mlx_param_count(model) == n_params(cfg)


def test_logit_equivalence_fp32() -> None:
    # Both models share weights via convert; float32 logits must agree closely.
    # Measured around 3e-6 max abs diff, comfortably under 2e-3.
    diff = convert.logit_max_abs_diff(get_config("size30m"), seed=0)
    assert diff < 2e-3, f"logit max abs diff {diff} exceeds 2e-3"


def _make_two_sequences(cfg, change_pos: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, cfg.vocab_size, size=(1, cfg.context_len), dtype=np.int64)
    other = base.copy()
    # Force a different token at change_pos.
    other[0, change_pos] = (base[0, change_pos] + 1) % cfg.vocab_size
    return base, other


def test_causality_torch() -> None:
    cfg = get_config("size30m")
    change_pos = cfg.context_len // 2
    base, other = _make_two_sequences(cfg, change_pos)
    model = model_torch.TransformerLM(cfg).to(torch.float32)
    model.eval()
    with torch.no_grad():
        l_base = model(torch.from_numpy(base)).to(torch.float32).numpy()
        l_other = model(torch.from_numpy(other)).to(torch.float32).numpy()
    # Positions strictly before the changed token must be identical.
    assert np.array_equal(l_base[:, :change_pos], l_other[:, :change_pos])
    # Sanity: the changed position itself must differ.
    assert not np.array_equal(l_base[:, change_pos], l_other[:, change_pos])


def test_causality_mlx() -> None:
    cfg = get_config("size30m")
    change_pos = cfg.context_len // 2
    base, other = _make_two_sequences(cfg, change_pos)
    model = model_mlx.TransformerLM(cfg)
    l_base = np.array(model(mx.array(base)), dtype=np.float32)
    l_other = np.array(model(mx.array(other)), dtype=np.float32)
    assert np.array_equal(l_base[:, :change_pos], l_other[:, :change_pos])
    assert not np.array_equal(l_base[:, change_pos], l_other[:, change_pos])
