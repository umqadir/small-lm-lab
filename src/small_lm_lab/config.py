"""Model configuration and exact parameter accounting.

Architecture is fixed by the lab plan: decoder-only, pre-norm RMSNorm, RoPE,
SwiGLU MLP, no biases anywhere, tied input/output embeddings, head_dim 64.

The parameter count computed by n_params is the ground truth the framework
implementations are checked against in tests/test_equivalence.py.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


def default_mlp_hidden(d_model: int) -> int:
    """SwiGLU hidden size: 8/3 * d_model rounded up to a multiple of 64."""
    return int(math.ceil((8.0 / 3.0 * d_model) / 64.0) * 64)


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    context_len: int
    head_dim: int = 64
    rope_theta: float = 10000.0
    mlp_hidden: int = 0  # 0 means "use default_mlp_hidden(d_model)"
    rmsnorm_eps: float = 1e-5
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.mlp_hidden == 0:
            self.mlp_hidden = default_mlp_hidden(self.d_model)

    @property
    def attn_dim(self) -> int:
        """Total attention projection width (n_heads * head_dim)."""
        return self.n_heads * self.head_dim


def n_params(cfg: ModelConfig) -> int:
    """Exact trainable parameter count.

    Tied embeddings are counted once (the lm head shares the embedding matrix).
    No biases exist in the model. RMSNorm contributes one weight vector of size
    d_model per norm (two per layer plus one final norm).
    """
    d = cfg.d_model
    attn = cfg.attn_dim
    mh = cfg.mlp_hidden

    embedding = cfg.vocab_size * d

    # Per transformer block:
    #   2 RMSNorm weights: 2 * d
    #   attention q,k,v,o projections (no bias): 4 * d * attn
    #   SwiGLU W1(gate), W3(up), W2(down) (no bias): 3 * d * mh
    per_layer = 2 * d + 4 * d * attn + 3 * d * mh

    final_norm = d

    total = embedding + cfg.n_layers * per_layer + final_norm
    if not cfg.tie_embeddings:
        total += cfg.vocab_size * d
    return total


def save_config(cfg: ModelConfig, path: str | Path) -> None:
    """Write a config to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)


def load_config(path: str | Path) -> ModelConfig:
    """Read a config from a YAML file."""
    with Path(path).open("r") as f:
        data = yaml.safe_load(f)
    return ModelConfig(**data)


# Named configs for the size sweep. All share vocab 16384, context 512,
# head_dim 64. n_heads * head_dim == d_model in every case.
_REGISTRY: dict[str, ModelConfig] = {
    "size30m": ModelConfig(
        vocab_size=16384, d_model=512, n_layers=8, n_heads=8, context_len=512
    ),
    "size60m": ModelConfig(
        vocab_size=16384, d_model=576, n_layers=12, n_heads=9, context_len=512
    ),
    "size120m": ModelConfig(
        vocab_size=16384, d_model=768, n_layers=16, n_heads=12, context_len=512
    ),
}


def get_config(name: str) -> ModelConfig:
    """Return a copy of a named registered config."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown config {name!r}; known: {sorted(_REGISTRY)}")
    base = _REGISTRY[name]
    return ModelConfig(**asdict(base))


def registered_configs() -> dict[str, ModelConfig]:
    """Return fresh copies of all registered configs, keyed by name."""
    return {name: get_config(name) for name in _REGISTRY}


if __name__ == "__main__":
    for name, cfg in registered_configs().items():
        print(
            f"{name}: d_model={cfg.d_model} n_layers={cfg.n_layers} "
            f"n_heads={cfg.n_heads} head_dim={cfg.head_dim} "
            f"mlp_hidden={cfg.mlp_hidden} -> {n_params(cfg):,} params"
        )
