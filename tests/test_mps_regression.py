"""Regression tests for the MPS float64 failure of 2026-07-21.

MPS tensors cannot be cast to float64: torch raises "Cannot convert a MPS
Tensor to float64 dtype as the MPS framework doesn't support float64". Every
eval and interp reduction used the idiom `.double().cpu()`, which casts on the
device and therefore detonates the first time the function runs on MPS. The
schedule's LR-scoring step was that first time, after three complete training
runs. Training never hit it because training is MLX; the eval battery is the
PyTorch path.

The fix is `.cpu().double()`: widening float32 to float64 is exact, so doing
it after the on-device float32 reduction is bit-identical, and where float64
is the accumulation dtype (bool means, the head-mean sums) the tensor moves to
the CPU first so the accumulation stays float64.

Two layers of defense here:

1. A static invariant, always on: no `.double(` in the package unless it is
   immediately preceded by `.cpu()`. This kills the whole class, including
   sites added later.
2. Runtime tests that drive the real public entry points on an actual MPS
   device and check they agree with the CPU path. These are opt-in via
   SMALL_LM_LAB_MPS_TESTS=1 because the default suite stays off the GPU while
   another job may hold the shared training lock (see tests/conftest.py). The
   variable is set only when the caller holds the lock or the machine is idle. The scheduler's smoke gate exercises the same paths through the
   real CLIs, so these tests are the fine-grained diagnosis layer, not the
   only guard.
"""

from __future__ import annotations

import copy
import math
import os
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from small_lm_lab import emergence, evaluate, interp
from small_lm_lab.config import ModelConfig
from small_lm_lab.model_torch import TransformerLM

SRC_DIR = Path(__file__).resolve().parents[1] / "src" / "small_lm_lab"
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"

MPS_ENABLED = (
    os.environ.get("SMALL_LM_LAB_MPS_TESTS") == "1"
    and torch.backends.mps.is_available()
)
needs_mps = pytest.mark.skipif(
    not MPS_ENABLED,
    reason="opt-in GPU test: set SMALL_LM_LAB_MPS_TESTS=1 on a machine whose "
    "MPS device is free (the default suite stays off the GPU so it can run "
    "while another job holds the training lock)",
)

VOCAB = 512
CONTEXT = 512  # the ICL windows are pre-registered at (50,100)/(450,500)


# ---------------------------------------------------------------------------
# Layer 1: the static invariant, always on.


def test_no_device_side_float64_casts() -> None:
    """Every `.double(` in the package must be written `.cpu().double(`.

    A float64 cast on a device tensor is exactly the bug class that stopped
    the schedule on 2026-07-21, and it stays invisible until the first MPS
    execution of that line. Whitespace is stripped so a fluent chain broken
    across lines still counts as adjacent.
    """
    offenders: list[str] = []
    for py in sorted(list(SRC_DIR.glob("*.py")) + list(SCRIPTS_DIR.glob("*.py"))):
        flat = re.sub(r"\s+", "", py.read_text())
        for occ in re.finditer(re.escape(".double("), flat):
            if flat[max(0, occ.start() - 6) : occ.start()] != ".cpu()":
                offenders.append(f"{py.name} (flattened offset {occ.start()})")
    assert not offenders, (
        ".double() reached without .cpu() first; MPS has no float64, so this "
        f"raises on the device: {offenders}"
    )


def test_no_combined_device_and_float64_to_calls() -> None:
    """No `.to(...)` may name a device and float64 in the same call.

    `.to(device="cpu", dtype=torch.float64)` reads as a move followed by a
    cast and is neither: torch casts on the SOURCE device, so on MPS it raises
    the same error `.double()` on a device tensor raises. The `.double(` check
    above does not see this spelling, and the emergence module's OV circuit
    carried it undetected until the first MPS run of that path. The move and
    the cast have to be two calls, `.cpu().to(torch.float64)`.
    """
    offenders: list[str] = []
    for py in sorted(list(SRC_DIR.glob("*.py")) + list(SCRIPTS_DIR.glob("*.py"))):
        flat = re.sub(r"\s+", "", py.read_text())
        for occ in re.finditer(r"\.to\(([^()]*)\)", flat):
            arguments = occ.group(1)
            names_device = "device=" in arguments
            names_float64 = "float64" in arguments or "torch.double" in arguments
            if names_device and names_float64:
                offenders.append(f"{py.name}: .to({arguments})")
    assert not offenders, (
        "a .to() call names both a device and float64; the cast happens on the "
        f"source device and MPS has no float64: {offenders}"
    )


# ---------------------------------------------------------------------------
# Layer 2: the broken entry points, executed on a real MPS device.


def tiny_cfg() -> ModelConfig:
    # Real context length so the pre-registered ICL windows fit; everything
    # else at its smallest legal value. n_heads * head_dim must equal d_model.
    return ModelConfig(
        vocab_size=VOCAB, d_model=64, n_layers=2, n_heads=1, context_len=CONTEXT
    )


@pytest.fixture(scope="module")
def models() -> tuple[torch.nn.Module, torch.nn.Module]:
    """The same tiny random-init model, one copy per device."""
    torch.manual_seed(7)
    cpu = TransformerLM(tiny_cfg()).to(torch.float32).eval()
    mps = copy.deepcopy(cpu).to("mps")
    return cpu, mps


@pytest.fixture(scope="module")
def stream_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("streams")
    rng = np.random.default_rng(3)
    for domain in evaluate.DOMAINS:
        tokens = rng.integers(1, VOCAB, size=6 * CONTEXT + 1, dtype=np.int64)
        tokens.astype(np.uint16).tofile(root / f"{domain}_val.bin")
    return root


def finite_floats(result) -> dict[str, float]:
    """Flatten a result dataclass to its finite floats, failing on non-finite."""
    out: dict[str, float] = {}

    def walk(prefix: str, node) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(f"{prefix}.{k}", v)
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(f"{prefix}[{i}]", v)
        elif isinstance(node, float):
            assert math.isfinite(node), f"{prefix} is {node}"
            out[prefix] = node

    walk("", asdict(result))
    return out


def assert_close(cpu_result, mps_result, rtol: float = 5e-3) -> None:
    a, b = finite_floats(cpu_result), finite_floats(mps_result)
    assert a.keys() == b.keys()
    for key in a:
        assert a[key] == pytest.approx(b[key], rel=rtol, abs=1e-4), key


@needs_mps
def test_perplexity_on_mps_matches_cpu(models, stream_root) -> None:
    cpu, mps = models
    kwargs = dict(batch_size=2, root=stream_root, n_resamples=50, seed=1)
    got_cpu = evaluate.test_perplexity(cpu, "tinystories", "val", "cpu", **kwargs)
    got_mps = evaluate.test_perplexity(mps, "tinystories", "val", "mps", **kwargs)
    assert_close(got_cpu, got_mps)


@needs_mps
def test_icl_score_on_mps_matches_cpu(models, stream_root) -> None:
    cpu, mps = models
    kwargs = dict(batch_size=2, root=stream_root, n_resamples=50, seed=1)
    got_cpu = evaluate.icl_score(cpu, "tinystories", "val", "cpu", **kwargs)
    got_mps = evaluate.icl_score(mps, "tinystories", "val", "mps", **kwargs)
    assert_close(got_cpu, got_mps)


@needs_mps
def test_copying_eval_on_mps_matches_cpu(models) -> None:
    cpu, mps = models
    kwargs = dict(n_sequences=8, repeat_len=16, seed=1, batch_size=4, n_resamples=50)
    got_cpu = evaluate.copying_eval(cpu, "cpu", **kwargs)
    got_mps = evaluate.copying_eval(mps, "mps", **kwargs)
    assert_close(got_cpu, got_mps)


@needs_mps
def test_sentence_logprobs_on_mps_match_cpu(models) -> None:
    cpu, mps = models
    rng = np.random.default_rng(5)
    sentences = [
        [int(t) for t in rng.integers(1, VOCAB, size=n)] for n in (4, 9, 6)
    ]
    got_cpu = evaluate.sum_sentence_logprobs(cpu, sentences, "cpu", batch_size=2)
    got_mps = evaluate.sum_sentence_logprobs(mps, sentences, "mps", batch_size=2)
    assert np.all(np.isfinite(got_mps))
    np.testing.assert_allclose(got_cpu, got_mps, rtol=5e-3, atol=1e-3)


@needs_mps
def test_copying_per_sequence_on_mps_matches_cpu(models) -> None:
    cpu, mps = models
    kwargs = dict(n_sequences=8, repeat_len=16, seed=1, batch_size=4)
    got_cpu = interp.copying_per_sequence(cpu, "cpu", **kwargs)
    got_mps = interp.copying_per_sequence(mps, "mps", **kwargs)
    assert np.all(np.isfinite(got_mps))
    np.testing.assert_allclose(got_cpu, got_mps, rtol=5e-3, atol=1e-3)


@needs_mps
def test_window_losses_on_mps_match_cpu(models, stream_root) -> None:
    cpu, mps = models
    kwargs = dict(batch_size=2, root=stream_root)
    got_cpu = interp.window_losses(cpu, "tinystories", "val", "cpu", **kwargs)
    got_mps = interp.window_losses(mps, "tinystories", "val", "mps", **kwargs)
    assert np.all(np.isfinite(got_mps))
    np.testing.assert_allclose(got_cpu, got_mps, rtol=5e-3, atol=1e-3)


@needs_mps
def test_head_output_means_on_mps_match_cpu(models) -> None:
    cpu, mps = models
    rng = np.random.default_rng(9)
    batches = [
        rng.integers(1, VOCAB, size=(2, 64), dtype=np.int64) for _ in range(3)
    ]
    got_cpu = interp.head_output_means(cpu, batches, "cpu")
    got_mps = interp.head_output_means(mps, batches, "mps")
    for layer, mean_cpu in got_cpu.per_layer.items():
        mean_mps = got_mps.per_layer[layer]
        assert torch.isfinite(mean_mps).all()
        torch.testing.assert_close(
            mean_cpu, mean_mps.cpu(), rtol=5e-3, atol=1e-4
        )


@needs_mps
def test_ov_scores_on_mps_match_cpu(models) -> None:
    """The weights-only OV readings, which are the path that raised on MPS.

    Both are computed in float64 on the CPU from weights that were widened
    exactly, so the two devices agree to float64 and not merely to float32.
    """
    cpu, mps = models
    ids = np.arange(1, 33, dtype=np.int64)
    np.testing.assert_allclose(
        emergence.ov_copying_score(cpu), emergence.ov_copying_score(mps), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        emergence.ov_self_logit_rank(cpu, ids),
        emergence.ov_self_logit_rank(mps, ids),
        rtol=0,
        atol=0,
    )


@needs_mps
def test_patching_scores_on_mps_match_cpu() -> None:
    torch.manual_seed(11)
    logits = torch.randn(4, 32, VOCAB)
    correct = torch.randint(0, VOCAB, (4, 8))
    target = torch.randint(0, VOCAB, (4, 8))
    scored = slice(16, 24)
    got_cpu = interp._patching_scores(logits, correct, target, scored)
    got_mps = interp._patching_scores(
        logits.to("mps"), correct.to("mps"), target.to("mps"), scored
    )
    assert np.all(np.isfinite(got_mps))
    np.testing.assert_allclose(got_cpu, got_mps, rtol=5e-3, atol=1e-3)
