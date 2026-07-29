"""Correctness tests for the emergence trajectory instrument, on CPU.

The pattern is the one tests/test_interp.py sets: every measurement is checked
against a number computed independently of the code under test. A stub whose
attention is exact induction must score 1 on the registered metric and 0 on both
off-by-one controls, a stub attending to the earlier occurrence itself must score
the mirror image of that, a stub attending uniformly must have normalized entropy
exactly 1, and the OV reduction must equal the eigenvalue score of the full
16384-by-16384 circuit instead of approximating it.

Two of these matter more than the rest.

test_attention_summary_reproduces_the_registered_score is the anchor. This module
reads the registered prefix-matching score off a forward that also produces six
other things, and if that reading drifted from
interp.prefix_matching_per_sequence by so much as a float, every trajectory
number would describe a metric nobody registered. It is asserted at 1e-12 and it
holds exactly.

test_the_ov_reduction_is_exact_not_approximate is the other. The 64-by-64
reduction is justified by an algebraic identity, not by an approximation
argument, so the test builds the full matrix on a small config and compares. If
that ever fails, the identity was misapplied and the score means nothing.

Every config here is tiny except the one real model in the anchor test. Nothing
touches the test split, the GPU, the training lock, or the real corpus.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest
import torch
import torch.nn as nn

from small_lm_lab import config, emergence, evaluate, interp, train
from small_lm_lab.config import ModelConfig
from small_lm_lab.model_torch import ATTN_WEIGHTS_KEY, WANT_ATTN_WEIGHTS, TransformerLM

REPO_ROOT = Path(__file__).resolve().parents[1]
EMERGENCE_SCRIPT = REPO_ROOT / "scripts" / "19_emergence.py"

DEVICE = "cpu"
VOCAB = 16384
CONTEXT = 512
REPEAT = interp.COPY_REPEAT_LEN

# The stubs below are exact patterns, so the only error left is float32 rounding
# on a sum of at most a few hundred terms.
EXACT = 1e-6


def small_cfg(n_layers: int = 2, n_heads: int = 3) -> ModelConfig:
    """A real model small enough to run a few forwards on CPU in a test."""
    return ModelConfig(
        vocab_size=VOCAB,
        d_model=n_heads * 8,
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=8,
        context_len=CONTEXT,
    )


def tiny_model(seed: int = 0, n_layers: int = 2, n_heads: int = 3) -> TransformerLM:
    torch.manual_seed(seed)
    model = TransformerLM(small_cfg(n_layers, n_heads))
    model.eval()
    return model


class AttentionPatternStub(nn.Module):
    """A model that is nothing but a prescribed attention pattern.

    attention_summary reads post-softmax attention weights out of the capture
    dict and nothing else, so a stub that writes a chosen weight matrix there
    pins all seven metrics to a pattern whose scores are known by hand. Building
    the stub out of real weight matrices would mean asserting against a pattern
    that had to be measured first, which is not an anchor.

    The patterns, all row stochastic and all causal:

      induction     query repeat_len + i attends entirely to key i + 1, the
                    token following the earlier occurrence of the token it
                    holds. What the registered score is meant to find.
      token_match   query repeat_len + i attends entirely to key i, the earlier
                    occurrence ITSELF. A head doing this is matching tokens, not
                    inducting, and the registered score has to give it 0. This
                    is the pattern an off-by-one in the score would call an
                    induction head.
      prev_token    every query attends entirely to the position before it. The
                    induction precursor.
      uniform       every query spreads its attention over its whole causal
                    prefix.

    Queries in the first copy have no earlier occurrence to find, so under the
    two matching patterns they attend to themselves, which the matching scores
    never read.
    """

    def __init__(self, cfg: ModelConfig, pattern: str, repeat_len: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.pattern = pattern
        self.repeat_len = repeat_len

    def weights(self, t: int) -> torch.Tensor:
        r = self.repeat_len
        w = torch.zeros(self.cfg.n_heads, t, t, dtype=torch.float32)
        if self.pattern == "induction":
            for q in range(t):
                w[:, q, q - r + 1 if q >= r else q] = 1.0
        elif self.pattern == "token_match":
            for q in range(t):
                w[:, q, q - r if q >= r else q] = 1.0
        elif self.pattern == "prev_token":
            for q in range(t):
                w[:, q, max(q - 1, 0)] = 1.0
        elif self.pattern == "uniform":
            for q in range(t):
                w[:, q, : q + 1] = 1.0 / (q + 1)
        else:
            raise ValueError(f"unknown pattern {self.pattern!r}")
        return w

    def forward(self, tokens: torch.Tensor, capture: dict | None = None) -> torch.Tensor:
        b, t = tokens.shape
        if capture is not None and capture.get(WANT_ATTN_WEIGHTS):
            w = self.weights(t).unsqueeze(0).expand(b, -1, -1, -1)
            for layer in range(self.cfg.n_layers):
                capture.setdefault(ATTN_WEIGHTS_KEY, {})[layer] = w
        return torch.zeros(b, t, self.cfg.vocab_size, dtype=torch.float32)


def stub_summary(pattern: str, repeat_len: int = 16) -> emergence.AttentionSummary:
    cfg = ModelConfig(
        vocab_size=256,
        d_model=16,
        n_layers=2,
        n_heads=2,
        head_dim=8,
        context_len=2 * repeat_len,
    )
    stub = AttentionPatternStub(cfg, pattern, repeat_len)
    return emergence.attention_summary(
        stub, DEVICE, n_sequences=4, repeat_len=repeat_len, seed=1, batch_size=2
    )


def means(summary: emergence.AttentionSummary, metric: str) -> np.ndarray:
    return getattr(summary, metric).mean(axis=0)


# ----------------------------------------------------------------------------
# the registered score, read through this module
# ----------------------------------------------------------------------------

def test_attention_summary_reproduces_the_registered_score() -> None:
    """The anchor. The prefix-matching array this module reads off its own
    forward has to BE interp's, not a near miss: same sequences from the shared
    make_copy_sequences, same (query, key) pairs from the shared
    prefix_matching_indices, same float32 gather widened the same way. Asserted
    at 1e-12, which is far tighter than any drift a real bug would produce."""
    model = tiny_model(seed=0)
    kwargs = dict(n_sequences=6, repeat_len=32, seed=1, batch_size=3)
    summary = emergence.attention_summary(model, DEVICE, **kwargs)
    reference = interp.prefix_matching_per_sequence(model, DEVICE, **kwargs)

    assert summary.prefix_matching.shape == reference.shape
    assert np.abs(summary.prefix_matching - reference).max() < 1e-12
    # And the mean of it is the registered per-head score.
    assert summary.prefix_matching.mean(axis=0) == pytest.approx(
        interp.prefix_matching_score(model, DEVICE, **kwargs), abs=1e-12
    )


def test_the_three_pair_metrics_do_not_share_a_denominator() -> None:
    """offset_plus_two drops one query the other two keep, because its key at
    the last query would be the first token of the second copy. Recording the
    counts is what stops a reader assuming otherwise."""
    summary = stub_summary("uniform", repeat_len=16)
    assert summary.n_query_positions["prefix_matching"] == 15
    assert summary.n_query_positions["token_matching"] == 15
    assert summary.n_query_positions["offset_plus_two"] == 14
    for metric in ("prev_token", "self_token", "first_token", "entropy"):
        assert summary.n_query_positions[metric] == 31

    offsets = emergence.attention_offsets(16)
    prefix_q, prefix_k = offsets["prefix_matching"]
    token_q, token_k = offsets["token_matching"]
    plus_q, plus_k = offsets["offset_plus_two"]
    # The controls sit on either side of the registered key, at the same queries.
    assert torch.equal(token_q, prefix_q)
    assert torch.equal(token_k, prefix_k - 1)
    assert torch.equal(plus_q, prefix_q[:-1])
    assert torch.equal(plus_k, prefix_k[:-1] + 1)
    # Every pair is causal, and no key reaches into the second copy.
    for q, k in ((prefix_q, prefix_k), (token_q, token_k), (plus_q, plus_k)):
        assert bool((k <= q).all())
        assert int(k.max()) < 16


# ----------------------------------------------------------------------------
# positive controls on synthetic attention
# ----------------------------------------------------------------------------

def test_exact_induction_stub_scores_one_and_the_controls_zero() -> None:
    """The positive control. A head attending exactly to the token after the
    earlier occurrence is what the registered score is meant to find, so it must
    score 1, and both off-by-one controls must score 0. Anything else means the
    score and its controls are not reading the cells they claim to."""
    summary = stub_summary("induction")
    assert means(summary, "prefix_matching") == pytest.approx(1.0, abs=EXACT)
    assert means(summary, "token_matching") == pytest.approx(0.0, abs=EXACT)
    assert means(summary, "offset_plus_two") == pytest.approx(0.0, abs=EXACT)
    # A point mass has no entropy.
    assert means(summary, "entropy") == pytest.approx(0.0, abs=EXACT)


def test_token_matching_stub_is_the_mirror_image() -> None:
    """The off-by-one this exists to catch. A head attending to the earlier
    occurrence itself is matching tokens instead of inducting from them. If the
    registered score read key i instead of key i + 1 it would call this head an
    induction head, and this pair of assertions is what would notice."""
    summary = stub_summary("token_match")
    assert means(summary, "token_matching") == pytest.approx(1.0, abs=EXACT)
    assert means(summary, "prefix_matching") == pytest.approx(0.0, abs=EXACT)
    assert means(summary, "offset_plus_two") == pytest.approx(0.0, abs=EXACT)


def test_previous_token_stub_scores_one_and_the_rest_near_zero() -> None:
    """The induction precursor, isolated. A head attending only to j - 1 scores
    1 on prev_token, 0 on self and on all three pair metrics, and 1/(t - 1) on
    first_token, which is not zero because at j = 1 the previous token IS the
    first token. That residue is the arithmetic and not a tolerance."""
    repeat_len = 16
    summary = stub_summary("prev_token", repeat_len=repeat_len)
    seq_len = 2 * repeat_len
    assert means(summary, "prev_token") == pytest.approx(1.0, abs=EXACT)
    assert means(summary, "self_token") == pytest.approx(0.0, abs=EXACT)
    assert means(summary, "prefix_matching") == pytest.approx(0.0, abs=EXACT)
    assert means(summary, "token_matching") == pytest.approx(0.0, abs=EXACT)
    assert means(summary, "offset_plus_two") == pytest.approx(0.0, abs=EXACT)
    assert means(summary, "entropy") == pytest.approx(0.0, abs=EXACT)
    assert means(summary, "first_token") == pytest.approx(1.0 / (seq_len - 1), abs=EXACT)


def test_uniform_stub_scores_full_entropy_and_analytic_chance() -> None:
    """Chance is not zero and it is not a mystery. A head spreading attention
    over its whole causal prefix pays 1/(q + 1) to each key it can see, so the
    registered score is the mean of that over the scored queries, and its rows
    are exactly as wide as causality permits so the normalized entropy is 1.
    Asserting closed forms catches an off-by-one in the query range that a
    "close to zero" assertion would sail past."""
    repeat_len = 16
    summary = stub_summary("uniform", repeat_len=repeat_len)
    chance = float(np.mean([1.0 / (repeat_len + i + 1) for i in range(repeat_len - 1)]))

    assert means(summary, "entropy") == pytest.approx(1.0, rel=1e-6)
    assert means(summary, "prefix_matching") == pytest.approx(chance, rel=1e-6)
    # The module's own closed form is the same closed form.
    assert emergence.prefix_matching_chance(repeat_len) == pytest.approx(chance, rel=1e-12)
    # And at the registered repeat length it is the number the prediction
    # document states, which is where the 0.2 threshold's margin comes from.
    assert emergence.prefix_matching_chance(REPEAT) == pytest.approx(0.005412, abs=1e-6)


# ----------------------------------------------------------------------------
# the OV circuit
# ----------------------------------------------------------------------------

def ov_cfg(n_layers: int = 2, n_heads: int = 2) -> ModelConfig:
    """Small enough that the full V by V circuit can actually be built."""
    return ModelConfig(
        vocab_size=24,
        d_model=8,
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=4,
        context_len=16,
    )


def full_circuit_score(model: TransformerLM) -> np.ndarray:
    """The score computed on the whole V by V circuit, built explicitly.

    Deliberately written out from the algebra instead of by calling anything in
    emergence.py, so that it is an independent computation and not the same code
    twice. This is only affordable because the config is tiny; at the lab's
    16384 vocabulary the matrix would be 2 GB per head in float64.
    """
    cfg = model.cfg
    embed = model.embed.weight.detach().double().numpy()
    final = model.final_norm.weight.detach().double().numpy()
    head_dim = int(cfg.head_dim)
    out = np.empty((int(cfg.n_layers), int(cfg.n_heads)), dtype=np.float64)
    for layer, block in enumerate(model.blocks):
        attn = block.attn_norm.weight.detach().double().numpy()
        v = block.attn.v_proj.weight.detach().double().numpy()
        o = block.attn.o_proj.weight.detach().double().numpy()
        for head in range(int(cfg.n_heads)):
            lo, hi = head * head_dim, (head + 1) * head_dim
            p = v[lo:hi, :].T
            q = o[:, lo:hi].T
            full = (embed * attn[None, :]) @ p @ q @ (final[:, None] * embed.T)
            assert full.shape == (int(cfg.vocab_size), int(cfg.vocab_size))
            eigenvalues = np.linalg.eigvals(full)
            out[layer, head] = eigenvalues.real.sum() / np.abs(eigenvalues).sum()
    return out


def test_the_ov_reduction_is_exact_not_approximate() -> None:
    """The 64-by-64 reduction is justified by an identity, not by an
    approximation argument: the nonzero eigenvalues of X Y are exactly those of
    Y X, and the V - head_dim zero eigenvalues of the full matrix contribute 0
    to both the numerator and the denominator of this ratio. So the two scores
    are not close, they are the same number, and the tolerance here is float64
    noise on an eigenvalue solve instead of a margin for error."""
    torch.manual_seed(3)
    model = TransformerLM(ov_cfg())
    model.eval()
    reduced = emergence.ov_copying_score(model)
    full = full_circuit_score(model)
    assert reduced.shape == (model.cfg.n_layers, model.cfg.n_heads)
    assert np.abs(reduced - full).max() < 1e-9


def orthonormal(rows: int, cols: int, seed: int) -> torch.Tensor:
    """A [rows, cols] matrix with orthonormal columns. Needs rows >= cols."""
    generator = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(rows, cols, generator=generator))
    return q


def copying_head_model(sign: float) -> TransformerLM:
    """A model whose one head's OV path is sign times the identity map.

    Built and not found, so the answer is known in advance.
    With E's columns orthonormal, E^T E is the identity, so the reduced circuit
    is Q diag(w_final) E^T E diag(w_attn) P, which with both norm weights at
    their initial ones and P = A, Q = sign A^T for A with orthonormal columns,
    collapses to sign times the identity on head_dim dimensions. Every
    eigenvalue is then sign, and the score is exactly sign: +1 for a head that
    copies what it attends to, -1 for one that maps every token away from
    itself.
    """
    cfg = ModelConfig(
        vocab_size=16, d_model=8, n_layers=1, n_heads=1, head_dim=4, context_len=8
    )
    torch.manual_seed(0)
    model = TransformerLM(cfg)
    model.eval()
    a = orthonormal(cfg.d_model, cfg.head_dim, seed=5)
    with torch.no_grad():
        model.embed.weight.copy_(orthonormal(cfg.vocab_size, cfg.d_model, seed=7))
        model.final_norm.weight.fill_(1.0)
        model.blocks[0].attn_norm.weight.fill_(1.0)
        model.blocks[0].attn.v_proj.weight.copy_(a.T)
        model.blocks[0].attn.o_proj.weight.copy_(sign * a)
    return model


def test_ov_copying_score_is_plus_one_for_a_copying_head() -> None:
    score = emergence.ov_copying_score(copying_head_model(1.0))
    assert score.shape == (1, 1)
    assert float(score[0, 0]) == pytest.approx(1.0, abs=1e-5)


def test_ov_copying_score_is_minus_one_for_an_anti_copying_head() -> None:
    score = emergence.ov_copying_score(copying_head_model(-1.0))
    assert float(score[0, 0]) == pytest.approx(-1.0, abs=1e-5)


def test_ov_copying_score_is_homogeneous_of_degree_zero() -> None:
    """Why the input-dependent RMS scalar can be dropped. The score is a ratio
    of two quantities both homogeneous of degree 1 in the circuit, so scaling
    the circuit by any positive constant leaves it exactly where it was. A
    negative constant flips every eigenvalue's real part while leaving every
    magnitude alone, so it flips the sign and nothing else. If either of these
    failed, dropping the RMS scalar would be an approximation instead of an
    identity and the score would depend on the input after all.

    The constant is a power of two so that scaling a float32 parameter is exact
    and only the exponent moves. At 7.5 the scaled weights are the float32
    rounding of 7.5 times the originals rather than 7.5 times them, and the
    residual measures that rounding at about 3e-08 rather than measuring the
    homogeneity this test is about."""
    torch.manual_seed(11)
    model = TransformerLM(ov_cfg())
    model.eval()
    base = emergence.ov_copying_score(model)

    with torch.no_grad():
        for block in model.blocks:
            block.attn.o_proj.weight.mul_(8.0)
    scaled = emergence.ov_copying_score(model)
    assert np.abs(scaled - base).max() < 1e-12

    with torch.no_grad():
        for block in model.blocks:
            block.attn.o_proj.weight.mul_(-1.0)
    flipped = emergence.ov_copying_score(model)
    assert np.abs(flipped + base).max() < 1e-12


def test_ov_copying_score_is_nan_for_a_dead_head() -> None:
    """A head whose eigenvalue magnitudes sum to zero has no score at all, and
    reporting 0.0 there would read as a head precisely balanced between copying
    and anti-copying, which is a much stronger claim than "this head does
    nothing"."""
    torch.manual_seed(2)
    model = TransformerLM(ov_cfg(n_layers=1, n_heads=2))
    model.eval()
    with torch.no_grad():
        model.blocks[0].attn.v_proj.weight[0:4, :].zero_()
    score = emergence.ov_copying_score(model)
    assert np.isnan(score[0, 0])
    assert not np.isnan(score[0, 1])


def test_ov_self_logit_rank_is_zero_for_a_copying_head() -> None:
    """The legible companion, on a head built to copy. With the OV path the
    identity on an orthonormal embedding, the restricted circuit is the identity
    on the sampled ids, so every token's own logit is the only nonzero entry in
    its row and its rank is 0."""
    model = copying_head_model(1.0)
    ranks = emergence.ov_self_logit_rank(model, np.arange(1, 9))
    assert ranks.shape == (1, 1)
    assert float(ranks[0, 0]) == 0.0


def test_ov_self_logit_rank_refuses_ids_outside_the_vocabulary() -> None:
    model = copying_head_model(1.0)
    with pytest.raises(ValueError, match="token ids must lie in"):
        emergence.ov_self_logit_rank(model, [0, 999])
    with pytest.raises(ValueError, match="at least one token id"):
        emergence.ov_self_logit_rank(model, [])


# ----------------------------------------------------------------------------
# the negative control
# ----------------------------------------------------------------------------

def test_random_init_shows_no_induction_at_all() -> None:
    """P8 of docs/PROTOCOL.md, on a model shaped like
    size30m and shrunk in every dimension that does not matter here: 8 layers
    and 8 heads, so the maximum is taken over 64 heads as it will be on the real
    run, with the width and the vocabulary cut for speed.

    An untrained model has no induction heads by construction, so the maximum
    prefix-matching score has to sit at chance and no head may cross the
    registered 0.2. If either fired on random weights the metric would be
    measuring something other than induction and no trained number from it would
    be admissible."""
    cfg = ModelConfig(
        vocab_size=512, d_model=64, n_layers=8, n_heads=8, head_dim=8, context_len=128
    )
    torch.manual_seed(0)
    model = TransformerLM(cfg)
    model.eval()

    summary = emergence.attention_summary(
        model, DEVICE, n_sequences=8, repeat_len=64, seed=1, batch_size=4
    )
    scores = summary.prefix_matching.mean(axis=0)
    assert scores.shape == (8, 8)
    assert float(scores.max()) < 0.05
    assert interp.identify_induction_heads(scores) == []
    # Not merely below the threshold: sitting on the analytic chance level,
    # which is the stronger statement and the one that says the geometry rather
    # than the weights is producing the number.
    chance = emergence.prefix_matching_chance(64)
    assert float(scores.max()) < 2.0 * chance
    # An untrained head attends nearly uniformly, so its rows are nearly as wide
    # as causality allows.
    assert float(summary.entropy.mean()) > 0.99


# ----------------------------------------------------------------------------
# the trajectory
# ----------------------------------------------------------------------------

def synthetic_point(
    tokens: Optional[int],
    max_prefix: float,
    max_prev: float = 0.0,
    clears_chance: bool = False,
    icl: Optional[dict[str, float]] = None,
    induction_heads: tuple[tuple[int, int], ...] = (),
    per_sequence: Optional[np.ndarray] = None,
    n_sequences: int = 4,
    seed: int = 1,
) -> emergence.EmergencePoint:
    """A grid point with chosen summary numbers and no measurement behind them.

    For the trajectory-level readings, which are functions of the per-point
    numbers alone. Building them from real forwards would make these tests slow
    and would test the wrong thing: what is under test here is the reduction of
    a list of points to a trajectory, not the measurement of a point.
    """
    grid = np.zeros((1, 2), dtype=np.float64)
    scores = grid.copy()
    scores[0, 0] = max_prefix
    prev = grid.copy()
    prev[0, 0] = max_prev
    heads = emergence.HeadScores(
        prefix_matching=scores,
        prefix_matching_ci_low=scores,
        prefix_matching_ci_high=scores,
        token_matching=grid.copy(),
        offset_plus_two=grid.copy(),
        prev_token=prev,
        self_token=grid.copy(),
        first_token=grid.copy(),
        entropy=grid.copy(),
        ov_copying_score=grid.copy(),
        ov_self_logit_rank=grid.copy(),
    )
    return emergence.EmergencePoint(
        tokens=tokens,
        checkpoint=None if tokens is None else f"tokens_{tokens}.pt",
        heads=heads,
        prefix_matching_per_sequence=per_sequence,
        max_prefix_matching=max_prefix,
        max_prefix_matching_head=(0, 0),
        induction_heads=induction_heads,
        induction_threshold=interp.INDUCTION_THRESHOLD,
        max_prev_token=max_prev,
        max_prev_token_head=(0, 0),
        prefix_matching_chance=0.005412,
        copying=emergence.CopyingPoint(
            exact_match=0.5 if clears_chance else 0.0,
            exact_match_ci_low=0.4 if clears_chance else 0.0,
            exact_match_ci_high=0.6 if clears_chance else 1e-4,
            loss_delta=-1.0,
            loss_delta_ci_low=-1.1,
            loss_delta_ci_high=-0.9,
            mean_first_copy_loss=5.0,
            mean_second_copy_loss=4.0,
            chance=6.104e-05,
            clears_chance=clears_chance,
            n_sequences=512,
            repeat_len=REPEAT,
        ),
        icl={
            domain: emergence.IclPoint(
                domain=domain,
                split="val",
                icl_score=score,
                ci_low=score - 0.01,
                ci_high=score + 0.01,
                n_windows=100,
                max_windows=None,
                icl_windows_capped=False,
                mean_early_loss=5.0,
                mean_late_loss=5.0 + score,
            )
            for domain, score in (icl or {}).items()
        },
        n_sequences=n_sequences,
        repeat_len=REPEAT,
        n_query_positions={"prefix_matching": REPEAT - 1},
        n_self_logit_tokens=0,
        self_logit_token_source=emergence.SELF_LOGIT_SOURCE_NONE,
        bootstrap=emergence.BootstrapProvenance(
            n_resamples=10, alpha=0.05, sequence_seed=seed, resample_seed=seed
        ),
    )


def test_trajectory_summary_reproduces_the_registered_phase_change() -> None:
    """The registered interval is interp.phase_change_interval's and nothing
    here recomputes it. So the summary's answer must be that function's answer
    on the same trajectory, field for field, and the token axis it was read from
    must be the one the points carry."""
    trajectory = {1_000: 0.02, 2_000: 0.05, 4_000: 0.09, 8_000: 0.45, 16_000: 0.80}
    points = [synthetic_point(t, s) for t, s in trajectory.items()]
    summary = emergence.trajectory_summary(points, n_resamples=5)

    expected = interp.phase_change_interval(trajectory)
    assert summary["phase_change"]["ran"] is True
    assert summary["phase_change"]["interval"] == expected
    assert expected.crossed is True
    assert expected.start_tokens == 4_000
    assert expected.end_tokens == 8_000
    # No grid point lies strictly inside, which is what P3 reads.
    assert summary["phase_change"]["n_grid_points_between"] == 0
    assert summary["induction_threshold"] == interp.INDUCTION_THRESHOLD
    assert summary["phase_bounds"] == [interp.PHASE_LOW, interp.PHASE_HIGH]


def test_trajectory_summary_counts_the_grid_points_inside_the_crossing() -> None:
    trajectory = {1: 0.05, 2: 0.11, 3: 0.12, 4: 0.13, 5: 0.9}
    points = [synthetic_point(t, s) for t, s in trajectory.items()]
    summary = emergence.trajectory_summary(points, n_resamples=5)
    assert summary["phase_change"]["interval"].start_tokens == 1
    assert summary["phase_change"]["interval"].end_tokens == 5
    assert summary["phase_change"]["n_grid_points_between"] == 3


def test_trajectory_summary_finds_the_steepest_icl_step_and_the_firsts() -> None:
    points = [
        synthetic_point(1, 0.01, max_prev=0.05, icl={"fineweb_edu": -0.10}),
        synthetic_point(2, 0.02, max_prev=0.40, icl={"fineweb_edu": -0.15}),
        synthetic_point(4, 0.50, max_prev=0.60, clears_chance=True,
                        icl={"fineweb_edu": -0.60}),
        synthetic_point(8, 0.70, max_prev=0.65, clears_chance=True,
                        icl={"fineweb_edu": -0.62}),
    ]
    summary = emergence.trajectory_summary(points, n_resamples=5)
    steepest = summary["steepest_icl_step"]["fineweb_edu"]
    assert (steepest["from_tokens"], steepest["to_tokens"]) == (2, 4)
    assert steepest["delta"] == pytest.approx(-0.45)
    assert summary["first_copying_above_chance"] == 4
    # The previous-token head is above the level at 2, before the crossing.
    assert summary["first_prev_token_above"] == 2
    assert summary["prev_token_level"] == interp.PHASE_HIGH


def test_a_rising_icl_trajectory_reports_its_least_bad_step_with_its_sign() -> None:
    """A non-negative steepest delta says the score never fell anywhere, which
    is a result about the trajectory and not a missing value."""
    points = [
        synthetic_point(1, 0.01, icl={"tinystories": -0.30}),
        synthetic_point(2, 0.01, icl={"tinystories": -0.20}),
        synthetic_point(4, 0.01, icl={"tinystories": -0.05}),
    ]
    summary = emergence.trajectory_summary(points, n_resamples=5)
    steepest = summary["steepest_icl_step"]["tinystories"]
    assert steepest["delta"] > 0.0
    assert (steepest["from_tokens"], steepest["to_tokens"]) == (1, 2)


def test_the_random_init_control_sits_on_no_point_of_the_token_axis() -> None:
    """A model that has taken no optimizer steps is not the first checkpoint of
    the trajectory, it is the control the trajectory is read against, so it is
    excluded from every trajectory-level reading rather than sorted to zero."""
    points = [
        synthetic_point(None, 0.9, max_prev=0.9, clears_chance=True),
        synthetic_point(1_000, 0.02),
        synthetic_point(2_000, 0.90),
    ]
    summary = emergence.trajectory_summary(points, n_resamples=5)
    assert summary["n_points"] == 3
    assert summary["n_points_on_token_axis"] == 2
    assert summary["n_points_without_tokens"] == 1
    assert [p["tokens"] for p in summary["phase_change"]["trajectory"]] == [1_000, 2_000]
    assert summary["first_copying_above_chance"] is None
    assert summary["first_prev_token_above"] is None


def test_a_single_placed_point_reports_no_phase_change_rather_than_one() -> None:
    summary = emergence.trajectory_summary([synthetic_point(1_000, 0.9)], n_resamples=5)
    assert summary["phase_change"]["ran"] is False
    assert summary["phase_change"]["interval"] is None
    assert "at least 2" in summary["phase_change"]["note"]


# ----------------------------------------------------------------------------
# the crossing-time interval and head churn, amendment of 2026-07-26
# ----------------------------------------------------------------------------

def constant_array(value: float, n_sequences: int = 8) -> np.ndarray:
    """[n_sequences, 1, 1] with every sequence carrying the same score.

    Every resample of identical values gives the identical mean, so a trajectory
    built from these has a crossing time that is known exactly rather than up to
    bootstrap noise. That is what makes the assertion below a construction and
    not a measurement.
    """
    return np.full((n_sequences, 1, 1), value, dtype=np.float64)


def test_crossing_time_is_exact_when_every_sequence_agrees() -> None:
    """Known by construction. With no variation between sequences there is no
    variation between resamples, so the median and both bounds must be the one
    checkpoint that crosses, the no-crossing share must be exactly 0, and the
    interval must agree with the point estimate the existing rule gives."""
    values = {10: 0.01, 20: 0.05, 30: 0.90, 40: 0.95}
    points = [
        synthetic_point(t, v, per_sequence=constant_array(v))
        for t, v in values.items()
    ]
    crossing = emergence.crossing_time_interval(points, n_resamples=64, seed=1)

    assert crossing.reported is True
    assert crossing.median_tokens == 30
    assert crossing.ci_low_tokens == 30
    assert crossing.ci_high_tokens == 30
    assert crossing.no_crossing_share == 0.0
    assert crossing.level == interp.PHASE_HIGH
    assert crossing.n_sequences == 8
    assert crossing.n_checkpoints == 4
    # Alongside the point estimate, not instead of it.
    assert crossing.point_tokens == interp.phase_change_interval(values).end_tokens == 30


def test_crossing_time_spans_grid_points_when_the_sequences_disagree() -> None:
    """The reason the interval exists. The middle checkpoint crosses only in
    resamples that happened to draw enough of the high-scoring sequences, so the
    crossing time is sometimes that checkpoint and sometimes the next one, and
    the interval has to show both. A per-checkpoint resample could not produce
    this, because the whole point is that one draw of sequences decides the
    whole trajectory."""
    n = 8
    early = np.zeros((n, 1, 1))
    middle = np.zeros((n, 1, 1))
    middle[:3] = 1.0  # observed mean 0.375, crosses only when 3 or more are drawn
    late = np.ones((n, 1, 1))
    points = [
        synthetic_point(10, 0.0, per_sequence=early),
        synthetic_point(20, 0.375, per_sequence=middle),
        synthetic_point(30, 1.0, per_sequence=late),
    ]
    crossing = emergence.crossing_time_interval(points, n_resamples=400, seed=1)

    assert crossing.reported is True
    assert crossing.no_crossing_share == 0.0
    assert crossing.ci_low_tokens == 20
    assert crossing.ci_high_tokens == 30
    assert crossing.median_tokens in (20, 30)
    assert crossing.point_tokens == 20


def test_crossing_time_is_withheld_when_most_resamples_never_cross() -> None:
    """Ranked, not dropped. A resample that never crossed is a crossing time
    later than every checkpoint measured, so when more than half of them fail
    the median is not a token count at all and is reported as absent with the
    share attached, rather than being computed over the minority that did."""
    points = [
        synthetic_point(t, 0.01, per_sequence=constant_array(0.01))
        for t in (10, 20, 30)
    ]
    crossing = emergence.crossing_time_interval(points, n_resamples=32, seed=1)
    assert crossing.reported is False
    assert crossing.median_tokens is None
    assert crossing.ci_low_tokens is None
    assert crossing.ci_high_tokens is None
    assert crossing.no_crossing_share == 1.0
    assert "never exceeded" in crossing.note
    assert crossing.point_tokens is None


def test_crossing_time_refuses_points_measured_on_different_sequences() -> None:
    """The invariant the whole construction rests on. Pairing resample index 7
    across checkpoints only means anything if index 7 is the same copying
    sequence at every checkpoint, which holds when every point was measured with
    the same count, repeat length and seed. Nothing in the arrays reveals a
    mismatch, so it is checked against the provenance the points carry."""
    points = [
        synthetic_point(10, 0.01, per_sequence=constant_array(0.01), seed=1),
        synthetic_point(20, 0.90, per_sequence=constant_array(0.90), seed=2),
    ]
    with pytest.raises(ValueError, match="same copying sequences"):
        emergence.crossing_time_interval(points, n_resamples=8, seed=1)


def test_crossing_time_is_withheld_when_no_arrays_were_retained() -> None:
    points = [synthetic_point(10, 0.01), synthetic_point(20, 0.90)]
    crossing = emergence.crossing_time_interval(points, n_resamples=8, seed=1)
    assert crossing.reported is False
    assert "retained no per-sequence" in crossing.note


def test_head_churn_counts_heads_entering_and_leaving_the_registered_set() -> None:
    """Registered by the 2026-07-26 amendment. A set that forms and holds and a
    set that reshuffles every checkpoint are different findings, and the maximum
    score alone cannot tell them apart."""
    points = [
        synthetic_point(10, 0.05, induction_heads=()),
        synthetic_point(20, 0.40, induction_heads=((1, 2),)),
        synthetic_point(30, 0.50, induction_heads=((1, 2), (3, 4))),
        synthetic_point(40, 0.55, induction_heads=((3, 4), (5, 6))),
    ]
    churn = emergence.head_churn(points)
    assert [c["from_tokens"] for c in churn] == [10, 20, 30]
    assert churn[0] == {
        "from_tokens": 10, "to_tokens": 20,
        "n_before": 0, "n_after": 1, "n_entered": 1, "n_left": 0,
        "entered": [[1, 2]], "left": [],
    }
    assert churn[1]["n_entered"] == 1 and churn[1]["n_left"] == 0
    assert churn[2]["entered"] == [[5, 6]]
    assert churn[2]["left"] == [[1, 2]]
    # And it reaches the summary.
    summary = emergence.trajectory_summary(points, n_resamples=5)
    assert summary["head_churn"] == churn


# ----------------------------------------------------------------------------
# the CLI
# ----------------------------------------------------------------------------

def cli_cfg() -> ModelConfig:
    """A config sized for the CLI tests, which are about plumbing.

    The CLI runs the pre-registered 256 prefix-matching sequences and 512
    copying sequences whatever it is pointed at, because those counts are not
    flags. So the model has to be the cheap part: a 256 vocabulary and a context
    of exactly two copies of the 128-token copying block.
    """
    return ModelConfig(
        vocab_size=256,
        d_model=16,
        n_layers=2,
        n_heads=2,
        head_dim=8,
        context_len=2 * REPEAT,
    )


def load_emergence_script():
    """Import scripts/19_emergence.py by path. scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location("emergence_script", EMERGENCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def strict_loads(text: str) -> Any:
    """json.loads that refuses the tokens the JSON grammar does not contain.

    json.dump writes NaN, Infinity and -Infinity as bare tokens and json.loads
    reads them straight back, so a result file can be malformed for its whole
    life without a Python-only pipeline noticing. This module writes NaN
    routinely, since a dead head's OV score is NaN and so is every self-logit
    rank when no token ids were available, so the rule is on the main path here.
    """

    def reject(token: str) -> Any:
        raise ValueError(f"not valid JSON: bare {token} token")

    return json.loads(text, parse_constant=reject)


def write_checkpoints(cfg: ModelConfig, directory: Path, token_counts: list[int]) -> None:
    """Real checkpoints in the real format, written by train.py's own saver."""
    for i, tokens in enumerate(token_counts):
        torch.manual_seed(i)
        model = TransformerLM(cfg)
        weights = {k: v.detach().float().numpy() for k, v in model.state_dict().items()}
        train.save_checkpoint(
            directory, weights, tokens, step=i, target_tokens=tokens, meta={}
        )


def test_cli_sweeps_a_checkpoint_dir_into_a_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI against checkpoints train.py wrote, end to end, with no corpus.

    Two things this catches that the library tests cannot. The checkpoint loader
    is imported from scripts/06_evaluate.py through scripts/08_interp.py rather
    than copied, so a drift between the saved format and the loaded one shows
    here. And the results are numpy all the way down until they are written, so
    a value json.dump refuses shows here too.
    """
    cfg = cli_cfg()
    monkeypatch.setitem(config._REGISTRY, "clitest", cfg)
    ckpt_dir = tmp_path / "ckpts"
    write_checkpoints(cfg, ckpt_dir, [1_000_000, 2_000_000])

    script = load_emergence_script()
    out = tmp_path / "emergence.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "19_emergence.py", "--config", "clitest",
            "--checkpoint-dir", str(ckpt_dir),
            "--device", "cpu", "--out", str(out),
            "--skip-icl",  # no corpus is mounted
            "--n-resamples", "20", "--n-freq-tokens", "16",
            "--weights-batch-size", "32", "--batch-size", "32",
        ],
    )
    script.main()

    results = json.loads(out.read_text())
    metadata = results["metadata"]
    assert metadata["n_checkpoints"] == 2
    assert metadata["skip_icl"] is True
    assert metadata["random_init"] is False
    # The registered constants reach the file, so nobody has to open the source
    # to learn what the numbers were measured against.
    assert metadata["induction_threshold"] == interp.INDUCTION_THRESHOLD
    assert metadata["phase_bounds"] == [interp.PHASE_LOW, interp.PHASE_HIGH]
    assert metadata["prev_token_level"] == emergence.PREV_TOKEN_LEVEL
    assert metadata["prefix_matching_n_sequences"] == interp.PREFIX_MATCHING_N_SEQUENCES
    assert metadata["copying_n_sequences"] == evaluate.COPY_N_SEQUENCES
    # No corpus, so the self-logit ids came from the seeded fallback draw.
    assert metadata["self_logit_token_source"] == emergence.SELF_LOGIT_SOURCE_UNIFORM

    points = results["points"]
    assert [p["tokens"] for p in points] == [1_000_000, 2_000_000]
    assert np.asarray(points[0]["heads"]["prefix_matching"]).shape == (
        cfg.n_layers,
        cfg.n_heads,
    )
    for metric in emergence.ATTENTION_METRICS:
        assert metric in points[0]["heads"]
    for field in ("ov_copying_score", "ov_self_logit_rank"):
        assert np.asarray(points[0]["heads"][field]).shape == (
            cfg.n_layers,
            cfg.n_heads,
        )
    assert points[0]["icl"] == {}
    assert points[0]["copying"]["clears_chance"] is False
    assert points[0]["copying"]["chance"] == interp.copy_chance_accuracy(cfg.vocab_size)
    assert points[0]["induction_heads"] == []
    assert points[0]["n_self_logit_tokens"] == 16

    # The bulk array is in the sidecar and not in the JSON.
    assert "prefix_matching_per_sequence" not in points[0]
    sidecar = Path(metadata["prefix_matching_per_sequence_path"])
    assert sidecar.exists()
    with np.load(sidecar) as arrays:
        assert sorted(arrays) == ["tokens_000001000000", "tokens_000002000000"]
        stored = arrays["tokens_000001000000"]
        assert stored.shape == (
            interp.PREFIX_MATCHING_N_SEQUENCES,
            cfg.n_layers,
            cfg.n_heads,
        )
        # And the mean of what was stored is the per-head score in the JSON.
        assert stored.mean(axis=0) == pytest.approx(
            np.asarray(points[0]["heads"]["prefix_matching"]), abs=1e-12
        )
    assert points[0]["prefix_matching_per_sequence_key"] == "tokens_000001000000"
    assert metadata["prefix_matching_per_sequence_keys"] == [
        "tokens_000001000000",
        "tokens_000002000000",
    ]

    summary = results["trajectory_summary"]
    # Untrained models, so no crossing. That is a reported result.
    assert summary["phase_change"]["ran"] is True
    assert summary["phase_change"]["interval"]["crossed"] is False
    assert summary["crossing_time"]["reported"] is False
    assert summary["crossing_time"]["no_crossing_share"] == 1.0
    assert summary["first_copying_above_chance"] is None
    assert len(summary["head_churn"]) == 1
    assert summary["head_churn"][0]["n_entered"] == 0

    # The file a spec-conformant parser sees is the file Python sees.
    assert strict_loads(out.read_text()) == results


def test_cli_random_init_writes_a_control_with_no_token_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P8's negative control, through the same instrument. It carries tokens
    null because a model that has taken no optimizer steps sits at no point of
    the token axis, and it is therefore excluded from the trajectory readings
    rather than treated as the first checkpoint."""
    cfg = cli_cfg()
    monkeypatch.setitem(config._REGISTRY, "clitest", cfg)
    script = load_emergence_script()
    out = tmp_path / "control.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "19_emergence.py", "--config", "clitest", "--random-init",
            "--init-seed", "3", "--device", "cpu", "--out", str(out),
            "--skip-icl", "--n-resamples", "20", "--n-freq-tokens", "0",
            "--weights-batch-size", "32", "--batch-size", "32",
        ],
    )
    script.main()

    results = strict_loads(out.read_text())
    assert results["metadata"]["random_init"] is True
    assert results["metadata"]["init_seed"] == 3
    assert results["metadata"]["checkpoint"] is None

    point = results["points"][0]
    assert point["tokens"] is None
    assert point["checkpoint"] is None
    assert point["induction_heads"] == []
    assert point["max_prefix_matching"] < interp.INDUCTION_THRESHOLD
    assert point["copying"]["clears_chance"] is False
    # No token ids were asked for, so every self-logit rank is null rather than
    # a number computed against an unstated sample.
    assert point["n_self_logit_tokens"] == 0
    assert point["self_logit_token_source"] == emergence.SELF_LOGIT_SOURCE_NONE
    assert all(all(v is None for v in row) for row in point["heads"]["ov_self_logit_rank"])
    # It still keeps its array, under a name rather than a token count.
    assert point["prefix_matching_per_sequence_key"] == "random_init"

    summary = results["trajectory_summary"]
    assert summary["n_points_on_token_axis"] == 0
    assert summary["phase_change"]["ran"] is False
    assert summary["crossing_time"]["reported"] is False


def test_cli_finds_checkpoints_and_ignores_the_resume_state(tmp_path: Path) -> None:
    """find_checkpoints is 08_interp's, imported rather than copied, so this
    pins the reuse: a lexical sort is chronological because train.py zero pads,
    and resume_state.pt is not a checkpoint."""
    script = load_emergence_script()
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    for tokens in (4_000_000, 1_000_000, 2_000_000):
        (ckpt_dir / f"{train.CHECKPOINT_PREFIX}{tokens:012d}.pt").write_bytes(b"")
    (ckpt_dir / "resume_state.pt").write_bytes(b"")

    found = script.interp_script().find_checkpoints(ckpt_dir)
    assert [p.name for p in found] == [
        "tokens_000001000000.pt",
        "tokens_000002000000.pt",
        "tokens_000004000000.pt",
    ]
