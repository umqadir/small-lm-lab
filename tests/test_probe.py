"""End-to-end test for the cheap induction probe, scripts/10_probe_copying.py.

The probe reads a checkpoint the pod wrote and appends one line to the probe
curve. It runs on CPU against a tiny random-init checkpoint and a tiny synthetic
val stream, so nothing here touches the corpus, the GPU, or the training lock.
The probe is a monitoring instrument and is not the registered evaluation, so
the test checks that it produces the right shape and finite numbers, not that
the numbers hit a pre-registered anchor.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from small_lm_lab import config, train
from small_lm_lab.config import ModelConfig
from small_lm_lab.model_torch import TransformerLM

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_SCRIPT = REPO_ROOT / "scripts" / "10_probe_copying.py"


def load_probe_script():
    """Import scripts/10_probe_copying.py by path. scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location("probe_script", PROBE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def probe_cfg() -> ModelConfig:
    """Small width, but the lab's real context: the ICL score reads positions
    450 to 500, so the context has to reach 512."""
    return ModelConfig(
        vocab_size=256, d_model=16, n_layers=2, n_heads=2, head_dim=8, context_len=512
    )


def write_checkpoint(cfg: ModelConfig, directory: Path, tokens: int) -> Path:
    """One random-init checkpoint in train.py's real format."""
    torch.manual_seed(0)
    model = TransformerLM(cfg)
    weights = {k: v.detach().float().numpy() for k, v in model.state_dict().items()}
    return train.save_checkpoint(
        directory, weights, tokens, step=0, target_tokens=tokens, meta={}
    )


def write_val_stream(root: Path, domain: str, n_tokens: int, seed: int = 0) -> None:
    """A synthetic val stream, uint16, real ids so nothing looks like a boundary."""
    rng = np.random.default_rng(seed)
    tokens = rng.integers(1, 256, size=n_tokens, dtype=np.uint16)
    tokens.tofile(root / f"{domain}_val.bin")


def test_probe_appends_a_line_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = probe_cfg()
    monkeypatch.setitem(config._REGISTRY, "probetest", cfg)
    ckpt_dir = tmp_path / "ckpts"
    checkpoint = write_checkpoint(cfg, ckpt_dir, tokens=512_000_000)
    # Enough tokens for several 512-length windows so the capped ICL has data.
    tok_root = tmp_path / "tok"
    tok_root.mkdir()
    write_val_stream(tok_root, "tinystories", n_tokens=8 * 512 + 10)

    out = tmp_path / "staged" / "probe_curve.jsonl"
    script = load_probe_script()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "10_probe_copying.py", "--config", "probetest",
            "--checkpoint", str(checkpoint), "--device", "cpu",
            "--domain", "tinystories", "--tokenized-root", str(tmp_path / "tok"),
            "--n", "8", "--max-windows", "4", "--n-resamples", "20",
            "--batch-size", "8", "--out", str(out),
        ],
    )
    script.main()

    lines = out.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert set(record) == {"checkpoint", "tokens", "copying", "icl", "utc"}
    assert record["checkpoint"] == str(checkpoint)
    # No --tokens was passed, so the token count is the one the checkpoint recorded.
    assert record["tokens"] == 512_000_000
    assert np.isfinite(record["copying"]) and np.isfinite(record["icl"])
    # A random-init model does not copy: exact match sits at or near zero.
    assert 0.0 <= record["copying"] < 0.2

    # A second call appends instead of truncating: the file is the growing curve.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "10_probe_copying.py", "--config", "probetest",
            "--checkpoint", str(checkpoint), "--tokens", "600e6", "--device", "cpu",
            "--domain", "tinystories", "--tokenized-root", str(tmp_path / "tok"),
            "--n", "8", "--max-windows", "4", "--n-resamples", "20",
            "--batch-size", "8", "--out", str(out),
        ],
    )
    script.main()
    lines = out.read_text().splitlines()
    assert len(lines) == 2
    # --tokens overrides the checkpoint's recorded count.
    assert json.loads(lines[1])["tokens"] == 600_000_000


def test_probe_needs_a_token_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint that recorded no token count and no --tokens flag has no x
    coordinate, so the probe refuses rather than writing a placeable-nowhere point."""
    cfg = probe_cfg()
    monkeypatch.setitem(config._REGISTRY, "probetest", cfg)
    ckpt_dir = tmp_path / "ckpts"
    # A bare state_dict carries no token count for the loader to find.
    torch.manual_seed(0)
    model = TransformerLM(cfg)
    bare = ckpt_dir / "bare.pt"
    ckpt_dir.mkdir(parents=True)
    torch.save(model.state_dict(), bare)
    write_val_stream(tmp_path, "tinystories", n_tokens=4 * 512 + 10)

    script = load_probe_script()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "10_probe_copying.py", "--config", "probetest",
            "--checkpoint", str(bare), "--device", "cpu",
            "--tokenized-root", str(tmp_path),
            "--n", "8", "--max-windows", "2", "--n-resamples", "20",
            "--batch-size", "8", "--out", str(tmp_path / "probe.jsonl"),
        ],
    )
    with pytest.raises(SystemExit, match="no token count"):
        script.main()
