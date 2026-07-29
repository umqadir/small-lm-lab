"""Weight conversion and equivalence checking between the torch and MLX twins.

The two implementations use identical parameter names and identical [out, in]
weight layouts, so conversion is a straight per-tensor copy with no transpose.
When tie_embeddings is True there is a single embedding matrix and no separate
lm_head weight, so the tied duplicate is dropped on the way to MLX and updated
through embed.weight on the way back to torch.
"""

from __future__ import annotations

import numpy as np
import torch
import mlx.core as mx
from mlx.utils import tree_flatten

from .config import ModelConfig
from . import model_torch
from . import model_mlx


def torch_to_mlx(torch_model: model_torch.TransformerLM) -> dict[str, mx.array]:
    """Return a flat dict {param_name: mx.array} for the MLX twin.

    The lm_head duplicate is skipped when embeddings are tied, since the MLX
    model realizes the head through embed.as_linear and has no lm_head weight.
    """
    tied = torch_model.cfg.tie_embeddings
    params: dict[str, mx.array] = {}
    for name, tensor in torch_model.state_dict().items():
        if tied and name == "lm_head.weight":
            continue
        arr = tensor.detach().to(torch.float32).cpu().numpy()
        params[name] = mx.array(arr)
    return params


def mlx_to_torch(
    mlx_model: model_mlx.TransformerLM, torch_model: model_torch.TransformerLM
) -> model_torch.TransformerLM:
    """Copy MLX parameters into an existing torch model in place.

    named_parameters deduplicates tied weights, so copying embed.weight also
    updates the tied lm_head. Returns the torch model for convenience.
    """
    torch_params = dict(torch_model.named_parameters())
    for name, arr in tree_flatten(mlx_model.parameters()):
        if name not in torch_params:
            raise KeyError(f"MLX param {name!r} has no match in the torch model")
        np_arr = np.array(arr, dtype=np.float32)
        target = torch_params[name]
        with torch.no_grad():
            target.copy_(torch.from_numpy(np_arr).to(target.dtype))
    return torch_model


def copy_torch_into_mlx(
    torch_model: model_torch.TransformerLM, mlx_model: model_mlx.TransformerLM
) -> model_mlx.TransformerLM:
    """Load the torch model's weights into an existing MLX model in place."""
    params = torch_to_mlx(torch_model)
    mlx_model.load_weights(list(params.items()))
    mx.eval(mlx_model.parameters())
    return mlx_model


def logit_max_abs_diff(cfg: ModelConfig, seed: int = 0) -> float:
    """Build both models, share weights torch->mlx, return max abs logit diff.

    Everything runs in float32. The torch side runs on CPU. The MLX side runs on
    whatever mx.default_device() is set to, which is the GPU unless the caller
    pins it, so this function does not by itself guarantee a GPU-free run. The
    test suite pins MLX to the CPU in conftest.py. The same random token batch of
    shape [2, cfg.context_len] is fed through both models and the maximum
    absolute difference over all logits is returned.
    """
    torch.manual_seed(seed)

    torch_model = model_torch.TransformerLM(cfg).to(torch.float32)
    torch_model.eval()

    mlx_model = model_mlx.TransformerLM(cfg)
    copy_torch_into_mlx(torch_model, mlx_model)

    rng = np.random.default_rng(seed)
    tokens_np = rng.integers(0, cfg.vocab_size, size=(2, cfg.context_len), dtype=np.int64)

    with torch.no_grad():
        torch_logits = torch_model(torch.from_numpy(tokens_np)).to(torch.float32).numpy()

    mlx_logits = mlx_model(mx.array(tokens_np))
    mx.eval(mlx_logits)
    mlx_logits_np = np.array(mlx_logits, dtype=np.float32)

    return float(np.max(np.abs(torch_logits - mlx_logits_np)))
