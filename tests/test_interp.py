"""Correctness tests for the Phase B instrument, on CPU.

Every measurement here is checked against a number computed independently of the
code under test: a stub whose attention pattern is exact induction by
construction must score 1, a stub attending uniformly must score the mean of
1/(q + 1) over the scored queries, patching a head with its own activations must
change nothing at all, and the per-unit loops must reproduce evaluate.py's point
estimates to the last bit. Those anchors are what would catch a real bug, so
they are asserted tightly.

The one that matters most is test_analysis_path_matches_the_fast_path. The
prefix-matching score reads attention weights, and the weights only exist on the
analysis path. If that path computed something other than what training and
evaluation compute, every Phase B number would be a measurement of a model that
does not exist, and nothing else in this file would notice.

Nothing here touches the test split, the GPU, or the training lock. The token
streams are tiny synthetic files written into a tmp directory.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from small_lm_lab import config, evaluate, interp, train
from small_lm_lab.config import ModelConfig
from small_lm_lab.evaluate import copying_eval, icl_score, make_copy_sequences
from small_lm_lab.interp import (
    HeadMeans,
    activation_patching,
    build_ablation_context,
    copying_per_sequence,
    correct_continuation_tokens,
    head_output_means,
    identify_induction_heads,
    induction_target_tokens,
    iter_batches,
    matched_control_heads,
    mean_ablate_heads,
    patch_heads,
    phase_change_interval,
    prefix_matching_indices,
    prefix_matching_score,
    window_losses,
)
from small_lm_lab.model_torch import (
    ATTN_PER_HEAD_KEY,
    ATTN_WEIGHTS_KEY,
    WANT_ATTN_WEIGHTS,
    TransformerLM,
)

compute_perplexity = evaluate.test_perplexity

REPO_ROOT = Path(__file__).resolve().parents[1]
INTERP_SCRIPT = REPO_ROOT / "scripts" / "08_interp.py"

DEVICE = "cpu"
VOCAB = 16384
CONTEXT = 512
REPEAT = interp.COPY_REPEAT_LEN
FAST_RESAMPLES = 50

# What "unchanged" means for the shift-invariance tests. The cancellation is
# exact in real arithmetic and the residue here is float32 rounding on logits
# carrying a constant of order 10: about 1e-6 an element, and averaging over the
# scored positions does not grow it. So this tolerance is ~10x the noise floor
# and ~1e6 x smaller than the constant being cancelled, which is the gap that
# makes the assertion mean something.
SHIFT_TOL = 1e-5


def small_cfg(n_layers: int = 2, n_heads: int = 3) -> ModelConfig:
    """A real model small enough to run a few forwards on CPU in a test.

    The vocabulary and context are the lab's own, since the copying task draws
    against the vocabulary and the in-context windows sit at positions 450 to
    500. Only width and depth shrink.
    """
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


def random_tokens(batch: int, length: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.integers(1, VOCAB, size=(batch, length), dtype=np.int64))


class AttentionPatternStub(nn.Module):
    """A model that is nothing but a prescribed attention pattern.

    prefix_matching_score reads post-softmax attention weights out of the
    capture dict and nothing else, so a stub that writes a chosen weight matrix
    there pins the metric to a pattern whose score is known by hand. Building
    the stub out of real weight matrices instead would mean asserting against a
    pattern that had to be measured first, which is not an anchor.
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
            # Query r + i attends entirely to key i + 1: the token right after
            # the earlier occurrence of the token it is holding. Queries in the
            # first copy have no earlier occurrence to find, so they attend to
            # themselves, which the score never reads.
            for q in range(t):
                w[:, q, q - r + 1 if q >= r else q] = 1.0
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


class PositionShiftWrapper(nn.Module):
    """The same model, with an arbitrary per-position constant on its logits.

    Softmax is shift invariant: add a constant to every logit at a position and
    every token's probability at that position is exactly where it was. So this
    wrapper is not an approximation of the model it wraps, it is that model.
    Every quantity that is a property of the model rather than of the arbitrary
    constant has to read the same through it, and the pre-registered patching
    metric is one of those. That is what the amendment of 2026-07-17 bought.

    shift is [context_len]. vary_with_input scales it by a function of the
    tokens, so two runs over two different prefixes carry two different
    constants. That is the case the amendment is actually about: nothing forces
    the clean run and the corrupted run to agree on the constant, so a raw logit
    differenced across them mixes it in, while a difference taken inside one
    forward cannot.

    Everything the interventions reach for is delegated to the wrapped model, so
    the hooks land on the real blocks and the patched forward is the real
    patched forward plus a constant.
    """

    def __init__(
        self, inner: TransformerLM, shift: torch.Tensor, vary_with_input: bool = False
    ) -> None:
        super().__init__()
        self.inner = inner
        self.vary_with_input = vary_with_input
        self.register_buffer("shift", shift)

    @property
    def cfg(self) -> ModelConfig:
        return self.inner.cfg

    @property
    def blocks(self) -> nn.ModuleList:
        return self.inner.blocks

    def forward(self, tokens: torch.Tensor, capture: dict | None = None) -> torch.Tensor:
        logits = self.inner(tokens, capture=capture)
        t = logits.shape[1]
        constant = self.shift[:t].view(1, -1, 1)
        if self.vary_with_input:
            # Deterministic in the tokens and nothing else, so a rerun repeats
            # it, and different between the clean and the corrupted prefixes.
            scale = 1.0 + (tokens.sum(dim=1) % 5).float()
            constant = constant * scale.view(-1, 1, 1)
        return logits + constant


def arbitrary_shift(seed: int = 11) -> torch.Tensor:
    """A per-position constant with no readable structure. [context]."""
    generator = torch.Generator().manual_seed(seed)
    ramp = torch.linspace(-4.0, 7.0, CONTEXT)
    return ramp + torch.randn(CONTEXT, generator=generator)


def write_stream(root: Path, domain: str, split: str, tokens: np.ndarray) -> None:
    root.mkdir(parents=True, exist_ok=True)
    tokens.astype(np.uint16).tofile(root / f"{domain}_{split}.bin")


def random_stream(n_windows: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(10, VOCAB, size=n_windows * CONTEXT + 1, dtype=np.int64)


# ----------------------------------------------------------------------------
# the analysis attention path
# ----------------------------------------------------------------------------

def test_analysis_path_matches_the_fast_path() -> None:
    """The anchor the whole phase hangs from.

    scaled_dot_product_attention does not return attention weights, so the
    prefix-matching score is read off a second implementation. If that
    implementation is not the same function, every Phase B number describes a
    model nobody trained. So: same weights, same fixed batch, both paths, and
    the outputs have to agree.
    """
    model = tiny_model(seed=0, n_layers=3, n_heads=4)
    x = random_tokens(4, 64, seed=1)

    with torch.no_grad():
        fast = model(x)
        capture: dict = {WANT_ATTN_WEIGHTS: True}
        analysis = model(x, capture=capture)

    assert torch.allclose(fast, analysis, atol=1e-5, rtol=1e-5)
    # Tighter than the tolerance above on the scale of the logits themselves.
    assert (fast - analysis).abs().max().item() < 1e-4 * fast.abs().max().item()
    assert sorted(capture[ATTN_WEIGHTS_KEY]) == [0, 1, 2]


def test_analysis_weights_are_causal_and_row_stochastic() -> None:
    model = tiny_model(seed=0)
    x = random_tokens(2, 32, seed=2)
    capture: dict = {WANT_ATTN_WEIGHTS: True}
    with torch.no_grad():
        model(x, capture=capture)

    for weights in capture[ATTN_WEIGHTS_KEY].values():
        assert weights.shape == (2, model.cfg.n_heads, 32, 32)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(2, model.cfg.n_heads, 32))
        # Nothing may attend to its future.
        upper = torch.triu(torch.ones(32, 32, dtype=torch.bool), diagonal=1)
        assert weights[:, :, upper].abs().max().item() == 0.0


def test_the_fast_path_is_untouched_without_the_flag() -> None:
    """A capture dict without the flag still takes scaled_dot_product_attention
    and still yields the per-head outputs, so the interventions below do not
    pay for weights they never read."""
    model = tiny_model(seed=0)
    x = random_tokens(2, 16, seed=3)
    capture: dict = {}
    with torch.no_grad():
        model(x, capture=capture)
    assert ATTN_WEIGHTS_KEY not in capture
    assert sorted(capture[ATTN_PER_HEAD_KEY]) == [0, 1]


# ----------------------------------------------------------------------------
# prefix matching
# ----------------------------------------------------------------------------

def test_prefix_matching_indices_read_the_pre_registered_pairs() -> None:
    queries, keys = prefix_matching_indices(5)
    # Query 5 + i holds the token at position i; the key is i + 1.
    assert queries.tolist() == [5, 6, 7, 8]
    assert keys.tolist() == [1, 2, 3, 4]
    # Every pair is causal, and the last query of the sequence is dropped.
    assert bool((keys <= queries).all())
    assert queries.max().item() == 2 * 5 - 2


def test_exact_induction_stub_scores_one() -> None:
    """A head attending exactly to the token after the previous occurrence of
    the token it holds is what the score is meant to find, so it has to score
    1. Anything less means the score is reading the wrong cell."""
    cfg = small_cfg()
    stub = AttentionPatternStub(cfg, "induction", REPEAT)
    scores = prefix_matching_score(
        stub, DEVICE, n_sequences=8, repeat_len=REPEAT, seed=1, batch_size=4
    )
    assert scores.shape == (cfg.n_layers, cfg.n_heads)
    assert scores == pytest.approx(np.ones_like(scores), abs=1e-6)


def test_uniform_stub_scores_chance() -> None:
    """Chance is not zero and it is not a mystery: a head spreading attention
    over its whole causal prefix pays 1/(q + 1) to each key it can see, so its
    score is the mean of that over the scored queries. Asserting the closed form
    catches an off-by-one in the query range that a "close to zero" assertion
    would sail past."""
    cfg = small_cfg()
    stub = AttentionPatternStub(cfg, "uniform", REPEAT)
    scores = prefix_matching_score(
        stub, DEVICE, n_sequences=8, repeat_len=REPEAT, seed=1, batch_size=4
    )
    chance = float(np.mean([1.0 / (REPEAT + i + 1) for i in range(REPEAT - 1)]))

    assert scores == pytest.approx(np.full_like(scores, chance), rel=1e-6)
    # And chance sits far below the pre-registered threshold, in both
    # directions: the threshold is not measuring noise, and the induction stub
    # above is not scoring high for a reason that geometry alone would give it.
    assert chance < 0.01
    assert chance * 30 < interp.INDUCTION_THRESHOLD


def test_prefix_matching_runs_on_the_real_model_and_scores_near_chance() -> None:
    """An untrained model has no induction heads, by construction. This is the
    control the trained numbers are read against, and it also checks that the
    real forward, the real capture and the real gather fit together."""
    model = tiny_model(seed=0)
    scores = prefix_matching_score(
        model, DEVICE, n_sequences=4, repeat_len=32, seed=1, batch_size=2
    )
    assert scores.shape == (model.cfg.n_layers, model.cfg.n_heads)
    assert float(scores.max()) < interp.INDUCTION_THRESHOLD
    assert identify_induction_heads(scores) == []


def test_identify_induction_heads_is_strict_about_the_threshold() -> None:
    scores = np.array([[0.0, 0.2, 0.21], [0.19999, 0.5, 1.0]])
    # Above means above. 0.2 exactly is not an induction head.
    assert identify_induction_heads(scores) == [(0, 2), (1, 1), (1, 2)]
    assert identify_induction_heads(scores, threshold=0.5) == [(1, 2)]
    assert identify_induction_heads(np.zeros((2, 2))) == []


# ----------------------------------------------------------------------------
# matched controls
# ----------------------------------------------------------------------------

def test_matched_control_heads_match_count_and_layer() -> None:
    scores = np.zeros((4, 6))
    induction = [(1, 0), (1, 3), (2, 5)]
    controls = matched_control_heads(scores, induction, seed=0)

    assert len(controls) == len(induction)
    # Two from layer 1, one from layer 2, none from anywhere else.
    assert sorted(l for l, _ in controls) == [1, 1, 2]
    assert not set(controls) & set(induction)
    assert len(set(controls)) == len(controls)


def test_matched_control_heads_never_return_an_induction_head() -> None:
    """Swept across seeds, because a control set that only usually avoids the
    induction heads would corrupt the comparison silently and rarely."""
    scores = np.zeros((2, 8))
    induction = [(0, 0), (0, 1), (0, 2)]
    for seed in range(50):
        controls = matched_control_heads(scores, induction, seed)
        assert len(controls) == 3
        assert len(set(controls)) == 3
        assert not set(controls) & set(induction)
        assert all(layer == 0 for layer, _ in controls)


def test_matched_control_heads_refuse_an_impossible_layer() -> None:
    scores = np.zeros((2, 3))
    induction = [(0, 0), (0, 1), (0, 2)]
    with pytest.raises(ValueError, match="matched control set"):
        matched_control_heads(scores, induction, seed=0)


def test_matched_control_heads_are_deterministic_given_seed() -> None:
    scores = np.zeros((3, 8))
    induction = [(0, 0), (1, 1), (2, 2)]
    assert matched_control_heads(scores, induction, 7) == matched_control_heads(
        scores, induction, 7
    )


def test_matched_control_heads_may_draw_a_head_just_under_the_threshold() -> None:
    """Eligibility is "not an induction head", which is the pre-registration's
    own definition. A head at 0.19 is a legal control. Excluding near misses
    would stack the control set with heads chosen for being inert, which is the
    bias the control exists to rule out."""
    scores = np.array([[0.9, 0.19, 0.0]])
    induction = identify_induction_heads(scores)
    assert induction == [(0, 0)]
    drawn = {matched_control_heads(scores, induction, seed)[0] for seed in range(40)}
    assert drawn == {(0, 1), (0, 2)}


def test_empty_induction_set_gives_an_empty_control_set() -> None:
    assert matched_control_heads(np.zeros((2, 2)), [], seed=0) == []


# ----------------------------------------------------------------------------
# interventions
# ----------------------------------------------------------------------------

def test_mean_ablation_changes_the_forward_and_is_not_zeroing() -> None:
    """Two claims in one, both pre-registered. The ablation has to bite, and it
    has to bite with the mean rather than with zero: the pre-registration
    replaces a head output with its mean "rather than zero, since zero is off
    distribution", and a mean ablation that happened to equal zero ablation
    would be that forbidden intervention wearing the right name."""
    model = tiny_model(seed=0)
    seqs = make_copy_sequences(8, 32, VOCAB, seed=1)[:, :-1]
    x = torch.from_numpy(seqs)
    heads = [(0, 1), (1, 2)]

    means = head_output_means(model, iter_batches(seqs, 4), DEVICE)
    zeros = HeadMeans(
        per_layer={l: torch.zeros_like(v) for l, v in means.per_layer.items()},
        per_position=means.per_position,
        n_sequences=means.n_sequences,
        seq_len=means.seq_len,
    )

    with torch.no_grad():
        base = model(x)
        with mean_ablate_heads(model, heads, means):
            ablated = model(x)
        with mean_ablate_heads(model, heads, zeros):
            zeroed = model(x)
        after = model(x)

    # The mean is a real quantity, not a rounding of zero.
    assert means.per_layer[0][:, 1, :].abs().mean().item() > 1e-4
    assert (ablated - base).abs().max().item() > 1e-4
    assert (ablated - zeroed).abs().max().item() > 1e-4
    # And the hooks come off: the intervention cannot outlive its context.
    assert torch.equal(after, base)


def test_mean_ablation_only_touches_the_named_heads() -> None:
    model = tiny_model(seed=0)
    seqs = make_copy_sequences(4, 16, VOCAB, seed=1)[:, :-1]
    x = torch.from_numpy(seqs)
    means = head_output_means(model, iter_batches(seqs, 2), DEVICE)

    with torch.no_grad():
        capture: dict = {}
        model(x, capture=capture)
        base_heads = {l: v.clone() for l, v in capture[ATTN_PER_HEAD_KEY].items()}

        with mean_ablate_heads(model, [(0, 2)], means):
            capture = {}
            model(x, capture=capture)

    # Worth pinning: the capture records the head output before the hook
    # rewrites it, so layer 0's capture is identical under ablation. Anyone
    # reading attention outputs during an intervention is reading pre-ablation
    # values, by design and by test.
    assert torch.equal(capture[ATTN_PER_HEAD_KEY][0], base_heads[0])
    # Layer 1 is downstream of the ablation, so its input moved and so must it.
    assert not torch.equal(capture[ATTN_PER_HEAD_KEY][1], base_heads[1])


def test_head_means_shapes_follow_the_per_position_choice() -> None:
    model = tiny_model(seed=0)
    seqs = make_copy_sequences(6, 16, VOCAB, seed=1)[:, :-1]
    n_heads, head_dim = model.cfg.n_heads, model.cfg.head_dim

    per_position = head_output_means(model, iter_batches(seqs, 3), DEVICE)
    assert per_position.per_layer[0].shape == (seqs.shape[1], n_heads, head_dim)
    assert per_position.n_sequences == 6

    globally = head_output_means(
        model, iter_batches(seqs, 3), DEVICE, per_position=False
    )
    assert globally.per_layer[0].shape == (1, n_heads, head_dim)
    # The global mean is the per-position mean averaged over positions, exactly.
    assert torch.allclose(
        globally.per_layer[0][0], per_position.per_layer[0].mean(dim=0), atol=1e-5
    )


def test_head_means_average_the_batches_they_are_given() -> None:
    """One batch of six or three batches of two must give the same mean, or the
    accumulator is weighting batches instead of sequences."""
    model = tiny_model(seed=0)
    seqs = make_copy_sequences(6, 16, VOCAB, seed=1)[:, :-1]
    one = head_output_means(model, iter_batches(seqs, 6), DEVICE)
    many = head_output_means(model, iter_batches(seqs, 2), DEVICE)
    for layer in one.per_layer:
        assert torch.allclose(one.per_layer[layer], many.per_layer[layer], atol=1e-5)


def test_patching_a_head_with_its_own_clean_activations_is_a_no_op() -> None:
    """The indexing test. A patch that writes a head's own output back into its
    own slot must leave the forward bit for bit unchanged. Get the head axis
    wrong, or the batch or position axis, and this is the assertion that fails
    while a recovery curve computed from the same bug would still look
    plausible."""
    model = tiny_model(seed=0, n_layers=3, n_heads=4)
    x = random_tokens(3, 24, seed=5)

    with torch.no_grad():
        capture: dict = {}
        clean = model(x, capture=capture)
        per_head = capture[ATTN_PER_HEAD_KEY]

        # Every head of every layer, patched with itself.
        patches = {
            (layer, head): per_head[layer][:, :, head, :]
            for layer in per_head
            for head in range(model.cfg.n_heads)
        }
        with patch_heads(model, patches):
            all_patched = model(x)

        # And one head alone, which is what activation_patching actually does.
        with patch_heads(model, {(1, 2): per_head[1][:, :, 2, :]}):
            one_patched = model(x)

    assert torch.equal(all_patched, clean)
    assert torch.equal(one_patched, clean)


def test_patching_another_head_activation_does_change_the_forward() -> None:
    """The companion to the no-op: a patch that should bite has to bite, or the
    no-op above would pass on a patch mechanism that does nothing at all."""
    model = tiny_model(seed=0)
    x = random_tokens(3, 24, seed=5)
    other = random_tokens(3, 24, seed=6)

    with torch.no_grad():
        capture: dict = {}
        model(other, capture=capture)
        donor = capture[ATTN_PER_HEAD_KEY][0][:, :, 1, :]
        clean = model(x)
        with patch_heads(model, {(0, 1): donor}):
            patched = model(x)

    assert (patched - clean).abs().max().item() > 1e-4


def test_patching_rejects_a_wrongly_shaped_patch() -> None:
    model = tiny_model(seed=0)
    x = random_tokens(2, 16, seed=7)
    with pytest.raises(ValueError, match="expected"):
        with patch_heads(model, {(0, 0): torch.zeros(2, 16, 999)}):
            model(x)


# ----------------------------------------------------------------------------
# per-unit measurements reproduce evaluate.py
# ----------------------------------------------------------------------------

def test_copying_per_sequence_reproduces_copying_eval() -> None:
    """The reuse anchor. The pre-registered copying task is built once, in
    evaluate.make_copy_sequences, and scored on the slices
    evaluate.copy_scoring_slices names. Averaging the per-sequence columns must
    land exactly on copying_eval's point estimates, or this module is measuring
    a second copying task that merely resembles the first."""
    model = tiny_model(seed=0)
    kwargs = dict(n_sequences=16, repeat_len=32, seed=3, batch_size=8)

    rows = copying_per_sequence(model, DEVICE, **kwargs)
    reference = copying_eval(model, DEVICE, n_resamples=FAST_RESAMPLES, **kwargs)

    assert rows.shape == (16, 3)
    assert float(rows[:, 0].mean()) == pytest.approx(reference.exact_match, rel=1e-12)
    assert float(rows[:, 1].mean()) == pytest.approx(
        reference.mean_second_copy_loss, rel=1e-12
    )
    assert float(rows[:, 2].mean()) == pytest.approx(
        reference.mean_first_copy_loss, rel=1e-12
    )
    assert float(rows[:, 1].mean() - rows[:, 2].mean()) == pytest.approx(
        reference.loss_delta, rel=1e-12
    )


def test_window_losses_reproduce_perplexity_and_icl(tmp_path: Path) -> None:
    """The same anchor for the text metrics: the columns must feed evaluate's
    own statistics and come back with evaluate's own numbers."""
    write_stream(tmp_path, "tinystories", "val", random_stream(6, seed=0))
    model = tiny_model(seed=0)

    rows = window_losses(
        model, "tinystories", "val", DEVICE, batch_size=4, root=tmp_path
    )
    ppl = compute_perplexity(
        model, "tinystories", "val", DEVICE, batch_size=4, root=tmp_path,
        n_resamples=FAST_RESAMPLES,
    )
    icl = icl_score(
        model, "tinystories", "val", DEVICE, batch_size=4, root=tmp_path,
        n_resamples=FAST_RESAMPLES,
    )

    assert rows.shape == (6, 4)
    assert evaluate._weighted_perplexity(rows[:, 0:2]) == pytest.approx(
        ppl.perplexity, rel=1e-12
    )
    assert evaluate._icl_statistic(rows[:, 2:4]) == pytest.approx(
        icl.icl_score, rel=1e-12
    )


def test_window_losses_can_be_capped_for_cost(tmp_path: Path) -> None:
    write_stream(tmp_path, "tinystories", "val", random_stream(6, seed=0))
    model = tiny_model(seed=0)
    full = window_losses(model, "tinystories", "val", DEVICE, batch_size=4, root=tmp_path)
    capped = window_losses(
        model, "tinystories", "val", DEVICE, batch_size=4, root=tmp_path, max_windows=3
    )
    assert capped.shape == (3, 4)
    # The cap keeps the first windows the exhaustive tiling produces, in order.
    assert np.allclose(capped, full[:3])


# ----------------------------------------------------------------------------
# ablation effect
# ----------------------------------------------------------------------------

def test_ablation_effect_reports_a_paired_delta_on_copying() -> None:
    model = tiny_model(seed=0)
    context = build_ablation_context(
        model,
        DEVICE,
        domains=(),  # copying only; no corpus needed
        n_sequences=16,
        repeat_len=32,
        seed=3,
        batch_size=8,
    )
    effect = interp.ablation_effect(
        model,
        [(0, 1)],
        DEVICE,
        context=context,
        batch_size=8,
        n_resamples=FAST_RESAMPLES,
    )

    assert effect.heads == ((0, 1),)
    match = effect.copying_exact_match
    assert match.n_units == 16
    # The delta is exactly the two point estimates differenced, and its interval
    # brackets it.
    assert match.delta.point == pytest.approx(match.ablated - match.baseline, rel=1e-12)
    assert match.delta.ci_low <= match.delta.point <= match.delta.ci_high
    assert effect.copying_loss_delta.delta.point == pytest.approx(
        effect.copying_loss_delta.ablated - effect.copying_loss_delta.baseline, rel=1e-9
    )
    assert effect.perplexity == {} and effect.icl == {}


def test_a_supplied_context_wins_over_the_distribution_arguments() -> None:
    """A context carries the baseline. Arguments describing a different
    evaluation distribution cannot be honored alongside it, because the delta
    would then be a difference against a baseline measured on something else.
    The context wins, and this pins it rather than leaving it to be discovered."""
    model = tiny_model(seed=0)
    context = build_ablation_context(
        model, DEVICE, domains=(), n_sequences=16, repeat_len=32, seed=3, batch_size=8
    )
    effect = interp.ablation_effect(
        model,
        [(0, 1)],
        DEVICE,
        context=context,
        n_sequences=999,  # ignored
        repeat_len=999,  # ignored, and would not fit the context anyway
        seed=999,  # ignored
        n_resamples=FAST_RESAMPLES,
    )
    assert effect.n_sequences == 16
    assert effect.repeat_len == 32
    assert effect.seed == 3
    assert effect.copying_exact_match.n_units == 16


def test_ablating_nothing_leaves_every_metric_untouched() -> None:
    """An empty head set must be exactly the baseline, on every metric. If a
    zero-head ablation moves a number, the harness itself is perturbing the
    model and every measured effect size is contaminated by it."""
    model = tiny_model(seed=0)
    effect = interp.ablation_effect(
        model,
        [],
        DEVICE,
        domains=(),
        n_sequences=8,
        repeat_len=32,
        seed=3,
        batch_size=8,
        n_resamples=FAST_RESAMPLES,
    )
    assert effect.copying_exact_match.delta.point == 0.0
    assert effect.copying_exact_match.delta.ci_low == 0.0
    assert effect.copying_exact_match.delta.ci_high == 0.0
    assert effect.copying_loss_delta.delta.point == pytest.approx(0.0, abs=1e-12)
    assert effect.copying_exact_match.ablated == effect.copying_exact_match.baseline


def test_ablation_effect_covers_the_text_metrics(tmp_path: Path) -> None:
    write_stream(tmp_path, "tinystories", "val", random_stream(4, seed=0))
    model = tiny_model(seed=0)
    effect = interp.ablation_effect(
        model,
        [(1, 0)],
        DEVICE,
        domains=("tinystories",),
        n_sequences=8,
        repeat_len=32,
        seed=3,
        batch_size=4,
        root=tmp_path,
        n_resamples=FAST_RESAMPLES,
    )

    assert set(effect.perplexity) == {"tinystories"}
    assert set(effect.icl) == {"tinystories"}
    assert effect.n_windows == {"tinystories": 4}
    ppl = effect.perplexity["tinystories"]
    assert ppl.delta.point == pytest.approx(ppl.ablated - ppl.baseline, rel=1e-9)
    assert ppl.baseline > 0.0
    icl = effect.icl["tinystories"]
    assert icl.delta.point == pytest.approx(icl.ablated - icl.baseline, rel=1e-9)


# ----------------------------------------------------------------------------
# activation patching
# ----------------------------------------------------------------------------

def test_corrupted_sequences_replace_only_the_earlier_occurrence() -> None:
    clean = make_copy_sequences(4, 16, VOCAB, seed=1)
    corrupt = interp.corrupted_copy_sequences(clean, 16, VOCAB, corrupt_seed=2)

    assert corrupt.shape == clean.shape
    # The second copy, which carries the correct continuations, is untouched.
    assert np.array_equal(corrupt[:, 16:], clean[:, 16:])
    # The first copy, the earlier occurrence, is gone.
    assert not np.array_equal(corrupt[:, :16], clean[:, :16])
    # And the replacement is drawn from the same distribution: real vocabulary,
    # never the eot id.
    assert corrupt.min() > evaluate.EOT_ID
    assert corrupt.max() < VOCAB


def test_the_two_tokens_of_the_metric_are_fixed_by_the_construction() -> None:
    """The pre-registered logit difference names two tokens, and both have to
    fall out of the construction rather than out of a model output, or the
    metric has a free parameter in it.

    Query repeat_len + i holds token A[i]. The correct continuation is A[i + 1].
    The token the corrupted prefix would induct to sits at key i + 1, which the
    corruption rewrote, so it is A'[i + 1]. That key is the same cell
    prefix_matching_indices scores, which is what ties the metric to the head
    score rather than to a second, similar-looking rule.
    """
    r = 16
    clean = make_copy_sequences(4, r, VOCAB, seed=1)
    corrupt = interp.corrupted_copy_sequences(clean, r, VOCAB, corrupt_seed=2)

    correct = correct_continuation_tokens(clean, r)
    target = induction_target_tokens(corrupt, r)
    assert correct.shape == (4, r - 1)
    assert target.shape == (4, r - 1)

    queries, keys = prefix_matching_indices(r)
    for i in range(r - 1):
        # The correct continuation is what the model is scored on predicting at
        # prediction index repeat_len + i, which is what y holds there.
        assert np.array_equal(correct[:, i], clean[:, r + i + 1])
        # And it is A[i + 1], the same token the first copy already carried.
        assert np.array_equal(correct[:, i], clean[:, i + 1])
        # The inducted-to token is the corrupted token at the key the
        # prefix-matching score reads for this query.
        assert int(queries[i]) == r + i
        assert np.array_equal(target[:, i], corrupt[:, int(keys[i])])
        assert int(keys[i]) == i + 1

    # The corruption leaves the second copy alone, so the correct continuation is
    # the same token in either run. Only the token inducted to moved.
    assert np.array_equal(correct_continuation_tokens(corrupt, r), correct)
    assert not np.array_equal(target, correct)
    # The scored prediction indices the metric reads are evaluate's own.
    _, scored = evaluate.copy_scoring_slices(r)
    assert clean[:, 1:][:, scored].shape == correct.shape
    assert np.array_equal(clean[:, 1:][:, scored], correct)


def test_the_token_helpers_refuse_a_sequence_that_is_not_a_double_copy() -> None:
    with pytest.raises(ValueError, match="copying sequences"):
        correct_continuation_tokens(np.zeros((2, 31), dtype=np.int64), 16)
    with pytest.raises(ValueError, match="copying sequences"):
        induction_target_tokens(np.zeros((2, 31), dtype=np.int64), 16)


def test_activation_patching_reports_every_named_head() -> None:
    model = tiny_model(seed=0)
    heads = [(0, 0), (1, 2)]
    result = activation_patching(
        model,
        DEVICE,
        heads=heads,
        n_sequences=8,
        repeat_len=32,
        seed=3,
        batch_size=4,
        n_resamples=FAST_RESAMPLES,
    )

    assert [(h.layer, h.head) for h in result.heads] == heads
    assert result.corrupt_seed == 4
    # The primary metric, and the result says so in words rather than leaving it
    # to be inferred from a field name.
    assert result.metric == interp.PATCHING_METRIC
    assert "logit(correct continuation)" in result.metric
    assert result.gap.point == pytest.approx(
        result.clean_logit_diff - result.corrupted_logit_diff
    )
    # The gap carries its own interval, because it is the denominator every
    # recovered_fraction divides by and a reader cannot judge a fraction without
    # knowing whether the thing under it is distinguishable from zero.
    assert result.gap.ci_low <= result.gap.point <= result.gap.ci_high
    assert result.gap_excludes_zero == (
        result.gap.ci_low > 0.0 or result.gap.ci_high < 0.0
    )
    for head in result.heads:
        assert head.recovered_logit_diff.point == pytest.approx(
            head.patched_logit_diff - result.corrupted_logit_diff, rel=1e-9
        )
        assert head.recovered_logit_diff.ci_low <= head.recovered_logit_diff.point
        assert head.recovered_logit_diff.point <= head.recovered_logit_diff.ci_high

    # The secondary raw logit is reported, and it is a different number: if it
    # were the same one, the amendment would have changed nothing.
    assert result.secondary_metric == interp.PATCHING_SECONDARY_METRIC
    assert result.correct_logit_gap == pytest.approx(
        result.clean_correct_logit - result.corrupted_correct_logit
    )
    assert result.clean_logit_diff != result.clean_correct_logit
    for head in result.heads:
        assert head.recovered_correct_logit.point == pytest.approx(
            head.patched_correct_logit - result.corrupted_correct_logit, rel=1e-9
        )


def test_patching_the_clean_run_with_its_own_heads_recovers_the_whole_gap() -> None:
    """The no-op anchor, read through the new metric.

    tests above pin that patching a head with its own clean activations leaves
    the forward bit for bit alone. This is the same fact stated in the units the
    claim is reported in: if the patched run is the clean run, the patch
    recovered the clean run, so recovered_fraction is 1. Not approximately 1.
    The forward is identical, so the numerator and the denominator are the same
    float, and the ratio is exactly 1.0.

    The patched run here is the clean forward, which is what makes the fraction
    exactly 1. activation_patching splices into the corrupted forward instead,
    where patching every head could not reproduce the clean run anyway, since
    the residual stream still carries the corrupted prefix's embeddings.
    """
    r = 32
    model = tiny_model(seed=0, n_layers=2, n_heads=3)
    clean_seqs = make_copy_sequences(8, r, VOCAB, seed=3)
    corrupt_seqs = interp.corrupted_copy_sequences(clean_seqs, r, VOCAB, corrupt_seed=4)
    correct = torch.from_numpy(correct_continuation_tokens(clean_seqs, r))
    target = torch.from_numpy(induction_target_tokens(corrupt_seqs, r))
    _, scored = evaluate.copy_scoring_slices(r)

    with torch.no_grad():
        capture: dict = {}
        clean_logits = model(torch.from_numpy(clean_seqs[:, :-1]), capture=capture)
        per_head = capture[ATTN_PER_HEAD_KEY]
        corrupt_logits = model(torch.from_numpy(corrupt_seqs[:, :-1]))

        # Every head of the clean run, patched back into the clean run.
        patches = {
            (layer, head): per_head[layer][:, :, head, :]
            for layer in per_head
            for head in range(model.cfg.n_heads)
        }
        with patch_heads(model, patches):
            patched_logits = model(torch.from_numpy(clean_seqs[:, :-1]))

    assert torch.equal(patched_logits, clean_logits)

    clean = interp._patching_scores(clean_logits, correct, target, scored)
    corrupt = interp._patching_scores(corrupt_logits, correct, target, scored)
    patched = interp._patching_scores(patched_logits, correct, target, scored)

    # Columns 0, 1, 2: the patched, corrupted and clean runs' logit differences.
    values = np.column_stack([patched[:, 0], corrupt[:, 0], clean[:, 0]])
    assert interp._recovered_fraction(values) == 1.0
    assert interp._recovered_logit_diff(values) == pytest.approx(
        float(clean[:, 0].mean() - corrupt[:, 0].mean()), rel=1e-12
    )
    # And the gap it is a fraction of is a real quantity, not a rounding of zero,
    # so the 1.0 above is not the NaN guard's neighborhood.
    assert abs(float(clean[:, 0].mean() - corrupt[:, 0].mean())) > interp.GAP_FLOOR


def test_recovered_fraction_is_nan_when_the_gap_vanishes() -> None:
    """The guard survives the amendment, and it guards exactly what it guarded
    before: an exactly degenerate denominator. It is not a conditioning check.
    The amendment made the gap well defined, not large, and on a model that has
    not learned induction the gap is near zero on purpose, several orders of
    magnitude above this floor, and the fraction it produces is meaningless
    without being NaN. See interp._recovered_fraction."""
    # Columns 0, 1, 2: patched, corrupted, clean. Clean equals corrupted, so the
    # denominator is zero.
    values = np.array([[0.5, 0.2, 0.2], [0.7, 0.4, 0.4]])
    assert np.isnan(interp._recovered_fraction(values))
    # A gap under the floor is the same case.
    values = np.array([[0.5, 0.2, 0.2 + 1e-15]])
    assert np.isnan(interp._recovered_fraction(values))
    # And a gap over it is not.
    values = np.array([[0.5, 0.2, 0.4]])
    assert interp._recovered_fraction(values) == pytest.approx(1.5)


def test_recovered_fraction_is_withheld_unless_the_gap_clears_zero() -> None:
    """The withholding guard, both branches, on a real random-init model.

    A random-init model has no induction heads by construction, so there is no
    clean-minus-corrupted gap for a patch to recover: the gap is noise and its
    interval straddles zero. Divide by it and out comes a number like 0.86,
    which reads as "this head recovered most of the behavior" and means
    nothing. GAP_FLOOR does not catch this, and cannot: the gap here is around
    1e-4, eight orders of magnitude above it. So the interval catches it, and
    the fraction is NaN.

    What is not withheld is the part that is still a quantity.
    recovered_logit_diff is the same recovery unnormalized and is well defined
    whatever the gap does, so it stays a real number with a real interval, and
    the pre-registered induction-versus-control comparison can be read off it.

    The second half drives the open branch through alpha. That is not a claim
    that this model's gap is real. The gate is defined against the interval and
    alpha is what sets the interval, so widening alpha until the interval clears
    zero brings the fraction back, equal to the ratio computed by hand.
    """
    model = tiny_model(seed=0, n_layers=2, n_heads=3)
    kwargs = dict(
        heads=[(0, 0), (1, 2)],
        n_sequences=8,
        repeat_len=32,
        seed=3,
        batch_size=4,
        n_resamples=FAST_RESAMPLES,
        # The primary gate is held open throughout, so this test is about the
        # gap tests and nothing else. Without this every status below would be
        # withheld_no_copying_reading and the gap gate would never be reached.
        copying_exact_match=copying_that_clears_chance(),
    )

    # Closed: the pre-registered alpha, and a gap that does not clear zero.
    gated = activation_patching(model, DEVICE, **kwargs)
    assert gated.gap_excludes_zero is False
    assert gated.gap.ci_low < 0.0 < gated.gap.ci_high
    # Far above the degenerate-denominator floor, which is the whole point: the
    # floor was never going to catch this one.
    assert abs(gated.gap.point) > 1e6 * interp.GAP_FLOOR
    for head in gated.heads:
        assert head.recovered_fraction is None
        # And it says WHY it is absent, so a reader never has to guess whether a
        # missing fraction was withheld or simply never produced.
        assert head.recovered_fraction_status == interp.FRACTION_WITHHELD_GAP_INCLUDES_ZERO
        # The unnormalized recovery survives, so nothing is actually lost.
        assert not np.isnan(head.recovered_logit_diff.point)
        assert head.recovered_logit_diff.ci_low <= head.recovered_logit_diff.point

    # Open: a wide enough alpha that the interval clears zero.
    ungated = activation_patching(model, DEVICE, alpha=0.9999, **kwargs)
    assert ungated.gap_excludes_zero is True
    assert ungated.gap.point == pytest.approx(gated.gap.point, rel=1e-12)
    for head in ungated.heads:
        assert head.recovered_fraction is not None
        assert head.recovered_fraction_status == interp.FRACTION_REPORTED
        assert head.recovered_fraction.point == pytest.approx(
            head.recovered_logit_diff.point / ungated.gap.point, rel=1e-9
        )

    # The invariant the two fields hang on, on every head of both runs: the
    # status says "reported" exactly when there is something to report.
    for head in list(gated.heads) + list(ungated.heads):
        assert head.recovered_fraction_status in interp.FRACTION_STATUSES
        assert (head.recovered_fraction_status == interp.FRACTION_REPORTED) == (
            head.recovered_fraction is not None
        )


def test_a_non_copying_model_withholds_the_fraction_even_when_the_gap_clears_zero() -> None:
    """The 47.5 percent case, pinned.

    This is the exact configuration the third amendment was written about: a
    random-init model, which contains no induction whatever, whose gap interval
    nonetheless clears zero. Under the old design that combination printed a
    confident recovered_fraction. It is not a hypothetical, it is roughly half
    of random-init models at the pre-registered sequence count, and it is what
    made the gap interval the wrong instrument.

    The copying positive control is what closes it, and this test is arranged so
    that only the copying control can be doing the work: alpha is opened wide
    enough that the gap gate says go, and the assertion is that the fraction is
    withheld anyway. If someone removes the copying gate, gap_excludes_zero is
    still True here and this test goes red.
    """
    model = tiny_model(seed=0, n_layers=2, n_heads=3)
    kwargs = dict(
        heads=[(0, 0), (1, 2)],
        n_sequences=8,
        repeat_len=32,
        seed=3,
        batch_size=4,
        n_resamples=FAST_RESAMPLES,
        alpha=0.9999,  # force the gap gate open, as it opens by itself at n=256
    )

    result = activation_patching(
        model, DEVICE, copying_exact_match=copying_at_chance(), **kwargs
    )
    # The demoted gate says go.
    assert result.gap_excludes_zero is True
    # The primary gate says no, and it wins.
    assert result.copying_clears_chance is False
    assert result.copying_chance == pytest.approx(1.0 / (VOCAB - 1))
    for head in result.heads:
        assert head.recovered_fraction is None
        assert (
            head.recovered_fraction_status
            == interp.FRACTION_WITHHELD_NO_COPYING_BEHAVIOR
        )
        # And the recovery itself is still reported, so the negative result is
        # legible rather than a blank.
        assert not np.isnan(head.recovered_logit_diff.point)

    # The control is the only thing changed, and it flips the outcome: proof
    # that the copying reading is what withheld the fraction above, not some
    # other difference between these two runs.
    copying = activation_patching(
        model, DEVICE, copying_exact_match=copying_that_clears_chance(), **kwargs
    )
    assert copying.copying_clears_chance is True
    for head in copying.heads:
        assert head.recovered_fraction is not None
        assert head.recovered_fraction_status == interp.FRACTION_REPORTED

    # The reading that decided is carried on the result, so the gate can be
    # audited from the output rather than taken on trust.
    assert result.copying_exact_match == copying_at_chance()
    assert copying.copying_exact_match == copying_that_clears_chance()


def test_the_fraction_fails_closed_when_no_copying_reading_is_supplied() -> None:
    """A gate with no reading is not an open gate.

    "Nobody measured whether this model copies" and "this model copies" are
    different facts, and only one of them licenses a ratio. So the default
    withholds, under its own status rather than borrowing the one that means the
    model was measured and found not to copy.
    """
    model = tiny_model(seed=0, n_layers=2, n_heads=3)
    result = activation_patching(
        model,
        DEVICE,
        heads=[(0, 0)],
        n_sequences=8,
        repeat_len=32,
        seed=3,
        batch_size=4,
        n_resamples=FAST_RESAMPLES,
        alpha=0.9999,  # the gap gate is open, so only the missing reading withholds
    )
    assert result.gap_excludes_zero is True
    assert result.copying_exact_match is None
    assert result.copying_clears_chance is False
    for head in result.heads:
        assert head.recovered_fraction is None
        assert (
            head.recovered_fraction_status
            == interp.FRACTION_WITHHELD_NO_COPYING_READING
        )


def test_copying_clears_chance_reads_the_interval_against_the_construction() -> None:
    """Chance is 1/(vocab-1) because the targets are uniform over the non-eot
    vocabulary. It is read off the construction, so there is no threshold here
    that anyone chose and could later move."""
    assert interp.copy_chance_accuracy(16384) == pytest.approx(1.0 / 16383)
    chance = interp.copy_chance_accuracy(VOCAB)

    # The whole interval must clear chance, not just the point estimate: a point
    # above chance with an interval spanning it is exactly the reading that
    # cannot distinguish copying from not copying.
    assert not interp.copying_clears_chance(
        interp.Estimate(0.5, chance / 2, 0.9), VOCAB
    )
    assert not interp.copying_clears_chance(interp.Estimate(chance, 0.0, 1.0), VOCAB)
    assert interp.copying_clears_chance(interp.Estimate(0.5, 0.4, 0.6), VOCAB)
    # A model producing no exact matches at all is the random-init case.
    assert not interp.copying_clears_chance(interp.Estimate(0.0, 0.0, 0.0), VOCAB)


def test_the_patching_metric_is_shift_invariant() -> None:
    """The test that proves the amendment's fix did what the amendment says.

    A logit is defined only up to a per-position additive constant, since
    softmax is shift invariant. So a model with an arbitrary constant added to
    every logit at every position is the same model, and the pre-registered
    metric, being a difference of two logits at one position of one forward,
    cannot notice the constant. Every primary number must come back identical.

    The secondary raw logit does notice, which is asserted too: it is what makes
    this test non-vacuous. A wrapper that added nothing would pass the
    invariance assertions for the wrong reason.
    """
    model = tiny_model(seed=0, n_layers=2, n_heads=3)
    shifted = PositionShiftWrapper(model, arbitrary_shift())
    kwargs = dict(
        heads=[(0, 0), (1, 2)],
        n_sequences=8,
        repeat_len=32,
        seed=3,
        batch_size=4,
        n_resamples=FAST_RESAMPLES,
    )

    plain = activation_patching(model, DEVICE, **kwargs)
    with_shift = activation_patching(shifted, DEVICE, **kwargs)

    assert with_shift.clean_logit_diff == pytest.approx(
        plain.clean_logit_diff, abs=SHIFT_TOL
    )
    assert with_shift.corrupted_logit_diff == pytest.approx(
        plain.corrupted_logit_diff, abs=SHIFT_TOL
    )
    assert with_shift.gap.point == pytest.approx(plain.gap.point, abs=SHIFT_TOL)
    assert with_shift.gap.ci_low == pytest.approx(plain.gap.ci_low, abs=SHIFT_TOL)
    assert with_shift.gap.ci_high == pytest.approx(plain.gap.ci_high, abs=SHIFT_TOL)
    # The verdict the gate reaches is a property of the model, so a constant the
    # model is invariant to must not change it.
    assert with_shift.gap_excludes_zero == plain.gap_excludes_zero
    for before, after in zip(plain.heads, with_shift.heads):
        assert (after.layer, after.head) == (before.layer, before.head)
        assert after.patched_logit_diff == pytest.approx(
            before.patched_logit_diff, abs=SHIFT_TOL
        )
        assert after.recovered_logit_diff.point == pytest.approx(
            before.recovered_logit_diff.point, abs=SHIFT_TOL
        )
        # The intervals too, so the invariance is a property of the measurement
        # and not only of its point estimate.
        assert after.recovered_logit_diff.ci_low == pytest.approx(
            before.recovered_logit_diff.ci_low, abs=SHIFT_TOL
        )
        assert after.recovered_logit_diff.ci_high == pytest.approx(
            before.recovered_logit_diff.ci_high, abs=SHIFT_TOL
        )
        # These are random-init models, so the gap does not clear zero and the
        # fraction is withheld on both sides. The invariance that matters here
        # is that an arbitrary constant cannot flip that verdict: the gate is a
        # property of the model too. The fraction's numeric invariance is pinned
        # in test_recovered_fraction_is_withheld_unless_the_gap_clears_zero,
        # where the gate is open.
        assert before.recovered_fraction is None
        assert after.recovered_fraction is None
        assert after.recovered_fraction_status == before.recovered_fraction_status

    # The constant is real and large, and the raw logit reports it. This is the
    # number the amendment took off the headline.
    assert abs(with_shift.clean_correct_logit - plain.clean_correct_logit) > 1.0


def test_a_run_dependent_shift_moves_the_raw_logit_gap_but_not_the_metric() -> None:
    """The sharp version of the invariance test, and the amendment's actual
    argument.

    Nothing forces two forwards over two different prefixes to land on the same
    per-position constant. This wrapper makes that concrete: the constant is
    scaled by a function of the tokens, so the clean run and the corrupted run
    carry different ones while the model, in every sense softmax can see, is
    unchanged.

    The raw logit gap moves, because it differences across the two runs and so
    mixes the two constants in. The pre-registered logit difference does not,
    because it is taken inside one forward before anything is compared across
    runs. That is exactly the distinction the amendment drew.
    """
    model = tiny_model(seed=0, n_layers=2, n_heads=3)
    shifted = PositionShiftWrapper(model, arbitrary_shift(), vary_with_input=True)
    kwargs = dict(
        heads=[(0, 0), (1, 2)],
        n_sequences=8,
        repeat_len=32,
        seed=3,
        batch_size=4,
        n_resamples=FAST_RESAMPLES,
    )

    plain = activation_patching(model, DEVICE, **kwargs)
    with_shift = activation_patching(shifted, DEVICE, **kwargs)

    # The metric is untouched.
    assert with_shift.clean_logit_diff == pytest.approx(
        plain.clean_logit_diff, abs=SHIFT_TOL
    )
    assert with_shift.corrupted_logit_diff == pytest.approx(
        plain.corrupted_logit_diff, abs=SHIFT_TOL
    )
    assert with_shift.gap.point == pytest.approx(plain.gap.point, abs=SHIFT_TOL)
    assert with_shift.gap.ci_low == pytest.approx(plain.gap.ci_low, abs=SHIFT_TOL)
    assert with_shift.gap.ci_high == pytest.approx(plain.gap.ci_high, abs=SHIFT_TOL)
    # The verdict the gate reaches is a property of the model, so a constant the
    # model is invariant to must not change it.
    assert with_shift.gap_excludes_zero == plain.gap_excludes_zero
    for before, after in zip(plain.heads, with_shift.heads):
        assert after.recovered_logit_diff.point == pytest.approx(
            before.recovered_logit_diff.point, abs=SHIFT_TOL
        )
        assert before.recovered_fraction is None
        assert after.recovered_fraction is None
        assert after.recovered_fraction_status == before.recovered_fraction_status

    # The raw gap is not. It moved by more than the whole quantity it is meant
    # to measure, which is what "silently mixes in any shift" cashes out to.
    assert abs(with_shift.correct_logit_gap - plain.correct_logit_gap) > 1.0
    assert abs(with_shift.correct_logit_gap - plain.correct_logit_gap) > abs(
        plain.correct_logit_gap
    )


def test_activation_patching_refuses_a_corrupt_seed_that_is_not_a_corruption() -> None:
    model = tiny_model(seed=0)
    with pytest.raises(ValueError, match="corrupt_seed"):
        activation_patching(
            model, DEVICE, heads=[(0, 0)], n_sequences=2, repeat_len=8, seed=3,
            corrupt_seed=3,
        )


def test_activation_patching_defaults_to_every_head() -> None:
    model = tiny_model(seed=0, n_layers=2, n_heads=3)
    result = activation_patching(
        model,
        DEVICE,
        heads=None,
        n_sequences=4,
        repeat_len=16,
        seed=3,
        batch_size=4,
        n_resamples=FAST_RESAMPLES,
    )
    assert [(h.layer, h.head) for h in result.heads] == [
        (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)
    ]


# ----------------------------------------------------------------------------
# phase change
# ----------------------------------------------------------------------------

def test_phase_change_interval_brackets_the_crossing() -> None:
    scores = {
        1_000_000: 0.01,
        2_000_000: 0.02,
        4_000_000: 0.05,
        8_000_000: 0.45,  # first checkpoint above 0.3
        16_000_000: 0.60,
    }
    out = phase_change_interval(scores)
    assert out.crossed
    assert out.start_tokens == 4_000_000  # last checkpoint below 0.1
    assert out.end_tokens == 8_000_000
    assert out.width_tokens == 4_000_000
    assert out.start_score == 0.05
    assert out.end_score == 0.45


def test_phase_change_ignores_a_later_crossing() -> None:
    """The end is the first checkpoint above 0.3 and the start is the last one
    below 0.1 before it, so a score that wanders back down later cannot move the
    interval."""
    scores = [(1, 0.05), (2, 0.4), (4, 0.05), (8, 0.9)]
    out = phase_change_interval(scores)
    assert out.crossed
    assert (out.start_tokens, out.end_tokens) == (1, 2)


def test_phase_change_reports_no_crossing_rather_than_inventing_one() -> None:
    out = phase_change_interval({1: 0.01, 2: 0.05, 4: 0.2})
    assert not out.crossed
    assert out.start_tokens is None and out.end_tokens is None
    assert "never exceeds 0.3" in out.note


def test_phase_change_reports_an_unbracketed_crossing() -> None:
    """Already above 0.1 at the first checkpoint means the crossing happened
    before anyone was watching. That is a gap in the schedule, not an interval,
    and the pre-registration says to re-run with extra checkpoints."""
    out = phase_change_interval({1: 0.15, 2: 0.5})
    assert not out.crossed
    assert out.start_tokens is None
    assert out.end_tokens == 2
    assert "not bracketed" in out.note


def test_phase_change_needs_a_trajectory() -> None:
    out = phase_change_interval({1: 0.9})
    assert not out.crossed
    assert "at least 2" in out.note


def test_phase_change_sorts_by_token_count() -> None:
    """Checkpoints arriving out of order must not reorder the trajectory."""
    ordered = phase_change_interval([(1, 0.01), (2, 0.05), (4, 0.5)])
    shuffled = phase_change_interval([(4, 0.5), (1, 0.01), (2, 0.05)])
    assert ordered == shuffled


# ----------------------------------------------------------------------------
# the CLI
# ----------------------------------------------------------------------------

def cli_cfg() -> ModelConfig:
    """A config sized for the CLI tests, which are about plumbing.

    The CLI runs the pre-registered 256 sequences and the pre-registered 512
    copying sequences whatever it is pointed at, because those counts are not
    flags. So the model has to be the cheap part: a 256 vocabulary and a 256
    context, which is two copies of the 128-token copying block and nothing
    more. Every other test in this file runs the lab's real vocabulary
    and context, since the measurements there depend on them.
    """
    return ModelConfig(
        vocab_size=256,
        d_model=16,
        n_layers=2,
        n_heads=2,
        head_dim=8,
        context_len=2 * REPEAT,
    )


def load_interp_script():
    """Import scripts/08_interp.py by path. scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location("interp_script", INTERP_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copying_result(exact_match: float) -> evaluate.CopyingResult:
    """A CopyingResult carrying a chosen exact-match reading.

    For CLI tests that need the copying positive control to say something
    definite. The interval is tight around the point, so the reading is
    unambiguous in whichever direction the caller wanted.
    """
    return evaluate.CopyingResult(
        loss_delta=-1.0,
        loss_delta_ci_low=-1.1,
        loss_delta_ci_high=-0.9,
        exact_match=exact_match,
        exact_match_ci_low=exact_match * 0.95,
        exact_match_ci_high=min(1.0, exact_match * 1.05),
        n_sequences=512,
        repeat_len=REPEAT,
        seed=1,
        mean_first_copy_loss=5.0,
        mean_second_copy_loss=4.0,
    )


def copying_that_clears_chance() -> interp.Estimate:
    """A copying reading that is unambiguously real behavior.

    For isolating the other gates: with the primary gate open, whatever happens
    next is the gap tests talking.
    """
    return interp.Estimate(point=0.9, ci_low=0.85, ci_high=0.95)


def copying_at_chance() -> interp.Estimate:
    """A copying reading indistinguishable from not copying at all.

    Chance on the copying task is 1/(vocab-1), about 6.1e-05, and a model that
    is not copying produces almost no exact matches, so the bootstrap's lower
    bound sits at zero. That is what this is.
    """
    return interp.Estimate(
        point=interp.copy_chance_accuracy(VOCAB), ci_low=0.0, ci_high=2e-4
    )


def strict_loads(text: str) -> Any:
    """json.loads that refuses the tokens the JSON grammar does not contain.

    json.dump writes NaN, Infinity and -Infinity as bare tokens and json.loads
    reads them straight back, so a result file can be malformed for its whole
    life without a Python-only pipeline ever noticing, and only break when
    something else tries to read it. parse_constant is called for exactly those
    three tokens and nothing else, so raising from it is the cheapest stand-in
    for the spec-conformant parser every other language ships.
    """

    def reject(token: str) -> Any:
        raise ValueError(f"not valid JSON: bare {token} token")

    return json.loads(text, parse_constant=reject)


def test_the_strict_parser_check_can_actually_fail() -> None:
    """The guard test below is worth nothing if strict_loads accepts anything.

    So: this is what the file looked like before the non-finite rule. Python
    round-trips it happily, which is why the bug could survive, and a parser
    that follows the spec rejects it. If this test ever passes without
    raising, the strict checks elsewhere in this file have stopped testing
    anything.
    """
    text = json.dumps({"recovered_fraction": float("nan")})
    assert "NaN" in text
    # Python reads its own invalid output back without complaint.
    assert np.isnan(json.loads(text)["recovered_fraction"])
    with pytest.raises(ValueError, match="not valid JSON"):
        strict_loads(text)


def test_to_jsonable_maps_non_finite_floats_to_null() -> None:
    """Every non-finite float, wherever it is buried, and whatever numpy type it
    arrived as. NaN is the routine case for a withheld fraction, not a corner,
    so this is on the main path of every result file."""
    script = load_interp_script()
    doc = script.to_jsonable(
        {
            "nan": float("nan"),
            "inf": float("inf"),
            "neg_inf": float("-inf"),
            "finite": 1.5,
            "numpy_nan": np.float64("nan"),
            "numpy_finite": np.float64(2.5),
            "nested": [float("nan"), {"x": float("inf")}],
            "array": np.array([1.0, float("nan")]),
            "estimate": interp.Estimate(float("nan"), float("nan"), float("nan")),
        }
    )
    assert doc["nan"] is None
    assert doc["inf"] is None
    assert doc["neg_inf"] is None
    assert doc["numpy_nan"] is None
    assert doc["nested"] == [None, {"x": None}]
    assert doc["array"] == [1.0, None]
    assert doc["estimate"] == {"point": None, "ci_low": None, "ci_high": None}
    # Finite values are untouched: the rule removes what cannot be written, not
    # what is merely small.
    assert doc["finite"] == 1.5
    assert doc["numpy_finite"] == 2.5
    # And the whole thing survives a parser that follows the spec.
    assert strict_loads(json.dumps(doc)) == doc


def test_every_fraction_status_serializes_as_a_null_and_a_reason() -> None:
    """All four statuses, including the ones a random-init model never reaches.

    The two are paired: null says a number is absent, the status says why, and
    the two must never disagree. A reader who sees null alone cannot tell
    withheld from missing, which is the confusion the status exists to remove.
    """
    script = load_interp_script()
    assert set(interp.FRACTION_STATUSES) == {
        "reported",
        "withheld_no_copying_behavior",
        "withheld_no_copying_reading",
        "withheld_gap_includes_zero",
        "withheld_degenerate_gap",
        "not_computed",
    }
    for status in interp.FRACTION_STATUSES:
        reported = status == interp.FRACTION_REPORTED
        head = interp.HeadPatch(
            layer=0,
            head=1,
            patched_logit_diff=0.5,
            recovered_logit_diff=interp.Estimate(0.1, 0.0, 0.2),
            recovered_fraction=interp.Estimate(0.4, 0.3, 0.5) if reported else None,
            recovered_fraction_status=status,
            patched_correct_logit=1.0,
            recovered_correct_logit=interp.Estimate(0.1, 0.0, 0.2),
        )
        doc = strict_loads(json.dumps(script.to_jsonable(head)))
        assert doc["recovered_fraction_status"] == status
        assert (doc["recovered_fraction"] is not None) == reported
        # The recovery itself is reported in every case, which is what makes
        # withholding the ratio cost nothing.
        assert doc["recovered_logit_diff"]["point"] == 0.1


def write_checkpoints(cfg: ModelConfig, directory: Path, token_counts: list[int]) -> None:
    """Real checkpoints in the real format, written by train.py's own saver.

    Not a hand-rolled dict: the point of going through save_checkpoint is that
    the CLI has to load what training actually writes, bf16 storage, tied head
    and all.
    """
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
    """The CLI against checkpoints train.py wrote, end to end.

    Two things this catches that the library tests cannot. The checkpoint loader
    is imported from scripts/06_evaluate.py rather than copied, so this is where
    a drift between the saved format and the loaded one would show. And the
    results are numpy all the way down until they are written, so this is where
    a value json.dump refuses would show.

    identify_induction_heads is stubbed because a random-init model has no
    induction heads, by construction, and without one the causal branches would
    never run. The stub only decides which heads are intervened on; every number
    downstream of it is the real measurement.
    """
    cfg = cli_cfg()
    monkeypatch.setitem(config._REGISTRY, "clitest", cfg)
    ckpt_dir = tmp_path / "ckpts"
    write_checkpoints(cfg, ckpt_dir, [1_000_000, 2_000_000, 4_000_000])

    script = load_interp_script()
    monkeypatch.setattr(
        script.interp, "identify_induction_heads", lambda s, threshold=0.2: [(0, 1)]
    )
    out = tmp_path / "interp.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "08_interp.py", "--config", "clitest", "--checkpoint-dir", str(ckpt_dir),
            "--analysis", "all", "--device", "cpu", "--out", str(out),
            "--final-only",  # the causal battery on the trained model alone
            "--domains",  # copying only, so no corpus is needed
            "--patch-sequences", "4", "--n-resamples", "20",
            "--weights-batch-size", "32", "--batch-size", "32",
        ],
    )
    script.main()

    results = json.loads(out.read_text())
    assert results["metadata"]["n_checkpoints"] == 3
    assert results["metadata"]["induction_threshold"] == 0.2
    assert results["metadata"]["causal_selection"] == "final_only"
    assert results["metadata"]["n_causal_checkpoints"] == 1

    trajectory = results["prefix_matching_trajectory"]
    assert [e["tokens"] for e in trajectory] == [1_000_000, 2_000_000, 4_000_000]
    assert np.asarray(trajectory[0]["scores"]).shape == (cfg.n_layers, cfg.n_heads)

    # No crossing, because these are untrained models. That is a reported result.
    assert results["phase_change"]["ran"]
    assert results["phase_change"]["interval"]["crossed"] is False

    # The causal analyses run on the last checkpoint, which is the trained model.
    assert results["causal_checkpoint"]["tokens"] == 4_000_000

    ablation = results["ablation"]
    assert ablation["ran"]
    assert ablation["induction_heads"] == [[0, 1]]
    assert len(ablation["control_heads"]) == 1
    # Matched: same layer, and not the induction head itself.
    assert ablation["control_heads"][0][0] == 0
    assert ablation["control_heads"][0] != [0, 1]
    assert "delta" in ablation["induction_joint"]["copying_exact_match"]
    assert "delta" in ablation["control_joint"]["copying_exact_match"]
    # Amendment item 1: the headline is the per-position mean, and it is not a
    # flag. The sensitivity check is absent because it was not asked for.
    assert ablation["primary_mean_ablation"] == "per_position"
    assert ablation["mean_ablation"] == "per_position"
    assert ablation["per_position_means"] is True
    assert ablation["sensitivity_global_mean"] is None
    assert results["metadata"]["mean_ablation_primary"] == "per_position"
    assert results["metadata"]["global_mean_sensitivity"] is False

    patching = results["patching"]
    patched = patching["result"]["heads"]
    assert {(h["layer"], h["head"]) for h in patched} == {(0, 1)} | {
        tuple(ablation["control_heads"][0])
    }
    # Amendment item 2: the file says which metric the claim is read from rather
    # than leaving a reader to guess from the field names, and both families of
    # number are actually present under the names it advertises.
    assert patching["metric"]["primary"] == script.interp.PATCHING_METRIC
    assert patching["metric"]["secondary"] == script.interp.PATCHING_SECONDARY_METRIC
    assert patching["result"]["metric"] == script.interp.PATCHING_METRIC
    assert results["metadata"]["patching_metric"] == script.interp.PATCHING_METRIC
    for field in ("clean_logit_diff", "corrupted_logit_diff", "gap"):
        assert field in patching["result"]
    for field in ("patched_logit_diff", "recovered_logit_diff", "recovered_fraction"):
        assert field in patched[0]
    for field in ("patched_correct_logit", "recovered_correct_logit"):
        assert field in patched[0]
    # The gap reaches the file as a point with an interval, and the verdict that
    # gates every fraction reaches it too, so a reader looking at a NaN fraction
    # can see why it is NaN without rerunning anything.
    assert set(patching["result"]["gap"]) == {"point", "ci_low", "ci_high"}
    assert patching["result"]["gap_excludes_zero"] in (True, False)
    assert "fraction_gate" in patching["metric"]

    # The file a spec-conformant parser sees is the file Python sees.
    assert strict_loads(out.read_text()) == results

    # What the CLI owes is that the two fields agree and say something legal,
    # whichever way the gate fell. The verdict itself is deliberately not pinned
    # here: this run uses 4 sequences for cost, a bootstrap over 4 units has
    # almost no resolution, and which side of zero its interval lands on is
    # luck rather than a fact about the model. The gate's actual behavior is
    # pinned against the real gate in
    # test_recovered_fraction_is_withheld_unless_the_gap_clears_zero.
    assert patching["result"]["gap_excludes_zero"] in (True, False)
    for head in patched:
        status = head["recovered_fraction_status"]
        assert status in script.interp.FRACTION_STATUSES
        assert (head["recovered_fraction"] is not None) == (
            status == script.interp.FRACTION_REPORTED
        )
        # The unnormalized recovery is a number either way, which is what makes
        # a withheld ratio cost nothing.
        assert head["recovered_logit_diff"]["point"] is not None

    # This run is not the declared escalation, and says so.
    power = patching["power"]
    assert power["escalated"] is False
    assert power["n_sequences"] == 4  # what --patch-sequences asked for
    assert power["pre_registered_n_sequences"] == script.interp.PATCHING_N_SEQUENCES
    assert power["escalated_n_sequences"] == 256 * 4
    assert results["metadata"]["patch_escalated"] is False
    assert results["metadata"]["patch_sequences"] == 4


def test_cli_all_checkpoints_runs_the_causal_battery_at_every_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bug fix: the causal battery runs across checkpoints, not on the final
    one alone. --all-checkpoints puts it at every swept anchor and the results
    come back keyed by checkpoint, with the token count carried so a reader can
    place each on the training axis.
    """
    cfg = cli_cfg()
    monkeypatch.setitem(config._REGISTRY, "clitest", cfg)
    ckpt_dir = tmp_path / "ckpts"
    write_checkpoints(cfg, ckpt_dir, [1_000_000, 2_000_000])

    script = load_interp_script()
    monkeypatch.setattr(
        script.interp, "identify_induction_heads", lambda s, threshold=0.2: [(0, 1)]
    )
    out = tmp_path / "interp.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "08_interp.py", "--config", "clitest", "--checkpoint-dir", str(ckpt_dir),
            "--analysis", "ablation", "--device", "cpu", "--out", str(out),
            "--all-checkpoints", "--domains", "--n-resamples", "20",
            "--weights-batch-size", "32", "--batch-size", "32",
        ],
    )
    script.main()

    results = json.loads(out.read_text())
    assert results["metadata"]["causal_selection"] == "all"
    assert results["metadata"]["n_causal_checkpoints"] == 2
    # The singular key is absent: several anchors produce the list form.
    assert "ablation" not in results
    by_ckpt = results["ablation_by_checkpoint"]
    assert [e["tokens"] for e in by_ckpt] == [1_000_000, 2_000_000]
    for entry in by_ckpt:
        assert entry["ablation"]["ran"]
        assert entry["ablation"]["induction_heads"] == [[0, 1]]
    assert [e["tokens"] for e in results["causal_checkpoints"]] == [1_000_000, 2_000_000]


def test_cli_checkpoints_selects_by_token_count_and_by_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--checkpoints names anchors explicitly, as token counts or as paths, and
    only those run. This is how the amendment's rule-chosen anchors (pre, mid,
    post, final) are pointed at once the probe curve has picked them."""
    cfg = cli_cfg()
    monkeypatch.setitem(config._REGISTRY, "clitest", cfg)
    ckpt_dir = tmp_path / "ckpts"
    write_checkpoints(cfg, ckpt_dir, [1_000_000, 2_000_000, 4_000_000])
    four_m = next(p for p in ckpt_dir.glob("tokens_*.pt") if "004000000" in p.name)

    script = load_interp_script()
    monkeypatch.setattr(
        script.interp, "identify_induction_heads", lambda s, threshold=0.2: [(0, 1)]
    )
    out = tmp_path / "interp.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "08_interp.py", "--config", "clitest", "--checkpoint-dir", str(ckpt_dir),
            "--analysis", "ablation", "--device", "cpu", "--out", str(out),
            "--checkpoints", "2000000", str(four_m),
            "--domains", "--n-resamples", "20",
            "--weights-batch-size", "32", "--batch-size", "32",
        ],
    )
    script.main()

    results = json.loads(out.read_text())
    assert results["metadata"]["causal_selection"] == "explicit"
    assert results["metadata"]["n_causal_checkpoints"] == 2
    assert [e["tokens"] for e in results["ablation_by_checkpoint"]] == [
        2_000_000,
        4_000_000,
    ]


def test_cli_causal_over_a_dir_requires_a_selection_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The heart of the not-silent rule: an old invocation that relied on the
    final-only default now errors and points at the three explicit choices,
    rather than quietly keeping the behavior the registration calls a bug. It
    fails before writing anything."""
    cfg = cli_cfg()
    monkeypatch.setitem(config._REGISTRY, "clitest", cfg)
    ckpt_dir = tmp_path / "ckpts"
    write_checkpoints(cfg, ckpt_dir, [1_000_000, 2_000_000])
    out = tmp_path / "interp.json"

    script = load_interp_script()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "08_interp.py", "--config", "clitest", "--checkpoint-dir", str(ckpt_dir),
            "--analysis", "ablation", "--device", "cpu", "--out", str(out),
            "--domains", "--n-resamples", "20",
        ],
    )
    with pytest.raises(SystemExit, match="must say which checkpoints"):
        script.main()
    assert not out.exists()


def test_cli_prefix_only_over_a_dir_needs_no_selection_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-causal analysis has nothing to select, so the flag is not required.
    The selection rule is about the causal battery, not the whole script."""
    cfg = cli_cfg()
    monkeypatch.setitem(config._REGISTRY, "clitest", cfg)
    ckpt_dir = tmp_path / "ckpts"
    write_checkpoints(cfg, ckpt_dir, [1_000_000, 2_000_000])
    out = tmp_path / "interp.json"

    script = load_interp_script()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "08_interp.py", "--config", "clitest", "--checkpoint-dir", str(ckpt_dir),
            "--analysis", "prefix_matching", "--device", "cpu", "--out", str(out),
            "--weights-batch-size", "32",
        ],
    )
    script.main()
    results = json.loads(out.read_text())
    assert len(results["prefix_matching_trajectory"]) == 2
    # No causal analysis ran and no selection flag was given, so there is no
    # selection to record. The point is that main() did not refuse.
    assert results["metadata"]["causal_selection"] is None


def test_cli_rejects_multi_checkpoint_flags_with_a_single_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = cli_cfg()
    monkeypatch.setitem(config._REGISTRY, "clitest", cfg)
    write_checkpoints(cfg, tmp_path / "ckpts", [1_000_000])
    checkpoint = next((tmp_path / "ckpts").glob("tokens_*.pt"))
    out = tmp_path / "interp.json"

    script = load_interp_script()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "08_interp.py", "--config", "clitest", "--checkpoint", str(checkpoint),
            "--analysis", "ablation", "--device", "cpu", "--out", str(out),
            "--all-checkpoints", "--domains",
        ],
    )
    with pytest.raises(SystemExit, match="single --checkpoint"):
        script.main()
    assert not out.exists()


def test_cli_checkpoints_rejects_an_unknown_token_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = cli_cfg()
    monkeypatch.setitem(config._REGISTRY, "clitest", cfg)
    ckpt_dir = tmp_path / "ckpts"
    write_checkpoints(cfg, ckpt_dir, [1_000_000, 2_000_000])
    out = tmp_path / "interp.json"

    script = load_interp_script()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "08_interp.py", "--config", "clitest", "--checkpoint-dir", str(ckpt_dir),
            "--analysis", "ablation", "--device", "cpu", "--out", str(out),
            "--checkpoints", "999", "--domains",
        ],
    )
    with pytest.raises(SystemExit, match="not found"):
        script.main()
    assert not out.exists()


def test_cli_emits_the_global_mean_sensitivity_alongside_the_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Amendment item 1, at the CLI.

    Per position is the headline and the global mean is a sensitivity check
    reported alongside it, so the flag has to add the second one rather than
    swap the first. Both reach the file, and the file says which is which. The
    headline is unflaggable because it was fixed in advance, and a reading that
    can be selected at the command line is a reading that can be selected after
    seeing the numbers.
    """
    cfg = cli_cfg()
    monkeypatch.setitem(config._REGISTRY, "clitest", cfg)
    write_checkpoints(cfg, tmp_path / "ckpts", [1_000_000])
    checkpoint = next((tmp_path / "ckpts").glob("tokens_*.pt"))

    script = load_interp_script()
    monkeypatch.setattr(
        script.interp, "identify_induction_heads", lambda s, threshold=0.2: [(0, 1)]
    )
    out = tmp_path / "interp.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "08_interp.py", "--config", "clitest", "--checkpoint", str(checkpoint),
            "--analysis", "ablation", "--device", "cpu", "--out", str(out),
            "--domains",  # copying only, so no corpus is needed
            "--global-mean-ablation",
            "--n-resamples", "20", "--weights-batch-size", "32", "--batch-size", "32",
        ],
    )
    script.main()

    ablation = json.loads(out.read_text())["ablation"]
    # The headline did not move.
    assert ablation["primary_mean_ablation"] == "per_position"
    assert ablation["mean_ablation"] == "per_position"
    assert ablation["per_position_means"] is True

    sensitivity = ablation["sensitivity_global_mean"]
    assert sensitivity["mean_ablation"] == "global"
    assert sensitivity["per_position_means"] is False
    # It carries the same conditions, so it is a sensitivity check on the claim
    # rather than a different, smaller report.
    for key in ("induction_joint", "control_joint", "induction_singles", "control_singles"):
        assert key in sensitivity
    assert sensitivity["induction_joint"]["per_position_means"] is False
    assert ablation["induction_joint"]["per_position_means"] is True
    # Two means are two ablations, so they are two numbers. If these agreed, the
    # flag would be reporting the headline twice under two names.
    assert (
        sensitivity["induction_joint"]["copying_loss_delta"]["ablated"]
        != ablation["induction_joint"]["copying_loss_delta"]["ablated"]
    )
    assert json.loads(out.read_text())["metadata"]["global_mean_sensitivity"] is True


def test_cli_records_the_declared_escalation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second amendment's mechanics, and only the mechanics.

    The escalation is a conditional the pre-registration states in words: it is
    legitimate only where the model demonstrably copies and the gap still
    straddles zero, and copying accuracy is the positive control that decides.
    That is a judgment on a reading, so nothing here makes it. What the script
    owes is that the state is recordable and the rerun is one flag: the count
    used and whether the run was the declared escalation both reach the file, so
    an escalated run cannot be mistaken later for a routine one, and a routine
    one cannot be dressed up as escalated.
    """
    cfg = cli_cfg()
    monkeypatch.setitem(config._REGISTRY, "clitest", cfg)
    write_checkpoints(cfg, tmp_path / "ckpts", [1_000_000])
    checkpoint = next((tmp_path / "ckpts").glob("tokens_*.pt"))

    script = load_interp_script()
    monkeypatch.setattr(
        script.interp, "identify_induction_heads", lambda s, threshold=0.2: [(0, 1)]
    )
    # The escalation is reachable only once the copying gate is open, and these
    # are random-init checkpoints that copy at chance. So the positive control is
    # stubbed to a model that demonstrably copies: this test is about the
    # mechanics of recording an escalation, and the refusal when the gate is shut
    # is test_cli_refuses_to_escalate_while_the_copying_gate_is_shut.
    monkeypatch.setattr(script.evaluate, "copying_eval", lambda *a, **k: copying_result(0.9))
    # The escalated count is the pre-registered 1024 and is not a flag value, so
    # the run cannot be cheap. Shrink the constant itself for the test: what is
    # under test is that the declared count is used and recorded, not what the
    # number is, and the real number is asserted against interp's constant in
    # test_cli_sweeps_a_checkpoint_dir_into_a_trajectory.
    monkeypatch.setattr(script.interp, "PATCHING_ESCALATED_N_SEQUENCES", 8)
    out = tmp_path / "interp.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "08_interp.py", "--config", "clitest", "--checkpoint", str(checkpoint),
            "--analysis", "patching", "--device", "cpu", "--out", str(out),
            "--domains", "--patch-escalate",
            "--patch-sequences", "4",  # overridden by --patch-escalate
            "--n-resamples", "20", "--weights-batch-size", "32", "--batch-size", "32",
        ],
    )
    script.main()

    results = strict_loads(out.read_text())
    power = results["patching"]["power"]
    assert power["escalated"] is True
    # --patch-escalate wins over --patch-sequences rather than quietly losing to it.
    assert power["n_sequences"] == 8
    assert results["patching"]["result"]["n_sequences"] == 8
    assert results["metadata"]["patch_escalated"] is True
    assert results["metadata"]["patch_sequences"] == 8
    # The rule the escalation answers to travels with the number, so nobody has
    # to reconstruct from the count alone why a rerun happened.
    assert "copying" in power["rule"]
    assert "once" in power["rule"]


def test_cli_refuses_to_escalate_while_the_copying_gate_is_shut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Third amendment, point 3: on a non-copying model the escalation is not
    just pointless, it is harmful.

    More sequences resolve a meaningless gap more sharply, so escalating on a
    model that does not copy buys a more confident version of a number that
    should not be printed at all. These are random-init checkpoints, which copy
    at chance, so the script refuses rather than quietly running at the lower
    count: an escalation that silently did not happen would be worse than an
    error, because the results file would look like a routine run.
    """
    cfg = cli_cfg()
    monkeypatch.setitem(config._REGISTRY, "clitest", cfg)
    write_checkpoints(cfg, tmp_path / "ckpts", [1_000_000])
    checkpoint = next((tmp_path / "ckpts").glob("tokens_*.pt"))

    script = load_interp_script()
    monkeypatch.setattr(
        script.interp, "identify_induction_heads", lambda s, threshold=0.2: [(0, 1)]
    )
    monkeypatch.setattr(script.interp, "PATCHING_ESCALATED_N_SEQUENCES", 8)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "08_interp.py", "--config", "clitest", "--checkpoint", str(checkpoint),
            "--analysis", "patching", "--device", "cpu",
            "--out", str(tmp_path / "x.json"),
            "--domains", "--patch-escalate",
            "--n-resamples", "20", "--weights-batch-size", "32", "--batch-size", "32",
        ],
    )
    with pytest.raises(SystemExit, match="unreachable while the copying positive"):
        script.main()
    # And it refused before writing anything, so no half-run file is left behind
    # to be read later as a completed escalation.
    assert not (tmp_path / "x.json").exists()


def test_cli_refuses_phase_change_on_a_single_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A phase change is a trajectory. One checkpoint cannot make one, and the
    script says so rather than reporting an interval built from a single point."""
    cfg = cli_cfg()
    monkeypatch.setitem(config._REGISTRY, "clitest", cfg)
    write_checkpoints(cfg, tmp_path / "ckpts", [1_000_000])
    checkpoint = next((tmp_path / "ckpts").glob("tokens_*.pt"))

    script = load_interp_script()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "08_interp.py", "--config", "clitest", "--checkpoint", str(checkpoint),
            "--analysis", "phase_change", "--device", "cpu",
            "--out", str(tmp_path / "x.json"),
        ],
    )
    with pytest.raises(SystemExit, match="--checkpoint-dir"):
        script.main()


def test_cli_finds_checkpoints_and_ignores_the_resume_state(tmp_path: Path) -> None:
    """resume_state.pt carries optimizer state and is overwritten in place. It
    is not a pre-registered checkpoint and must never enter a trajectory."""
    cfg = cli_cfg()
    ckpt_dir = tmp_path / "ckpts"
    write_checkpoints(cfg, ckpt_dir, [2_000_000, 1_000_000])
    (ckpt_dir / train.RESUME_STATE_NAME).write_bytes(b"not a checkpoint")

    script = load_interp_script()
    found = script.find_checkpoints(ckpt_dir)
    assert [p.name for p in found] == [
        "tokens_000001000000.pt",
        "tokens_000002000000.pt",
    ]
    with pytest.raises(FileNotFoundError):
        script.find_checkpoints(tmp_path / "empty")
