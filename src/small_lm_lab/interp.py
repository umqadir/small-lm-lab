"""Phase B of the pre-registration, and nothing else.

docs/PROTOCOL.md, "Phase B protocol", fixes every definition here: the
prefix-matching score, the 0.2 threshold that calls a head an induction head,
the 0.1-to-0.3 phase-change interval, mean ablation against matched control
heads, and activation patching on the copying task. None of it is tunable from
the outside, for the same reason the BLiMP paradigm list in evaluate.py is
hardcoded: a threshold that can be passed in is a threshold that can be chosen
after seeing the distribution. What is exposed instead is cost (batch sizes,
resample counts) and provenance (seeds), which cannot move a number.

The amendment of 2026-07-17 closed the two readings this module had to guess at,
and both are now fixed, not optional:

  mean ablation is per position, the mean over sequences at each position
    instead of one constant vector per head. head_output_means defaults to it
    and the pre-registered comparison runs on it. The global-mean variant is
    still reachable, as the sensitivity check the amendment calls it, never as
    the headline.
  activation patching is scored by a logit difference, correct continuation
    against the token the corrupted prefix would induct to, not by a raw logit.
    A logit is defined only up to a per-position additive constant, so a raw
    logit differenced across two runs mixes that constant in. See
    activation_patching.

The pre-registration also fixes what is not here. No t-SNE, no saliency maps, no
single-neuron anecdotes. Nothing in this module produces a picture.

Shared with evaluate.py, not rebuilt:

  make_copy_sequences   the copying task itself, repeated uniform random tokens
                        drawn above the eot id. The prefix-matching score, the
                        ablation copying numbers and the patching runs all read
                        their sequences from it, so "repeated random tokens" is
                        one construction instead of three that could drift.
  copy_scoring_slices   which prediction indices count as the first and second
                        copy.
  _weighted_perplexity, _icl_statistic, _copy_loss_delta
                        the statistics themselves, so an ablation delta is a
                        difference of the pre-registered quantity and not of a
                        near-miss recomputed here.
  bootstrap_ci          the one percentile bootstrap, 1000 resamples.

Ablation and patching deltas are measured with a paired bootstrap: the same
evaluation sequences are resampled on both sides of the difference, because the
pre-registered claim compares intervals on the damage done, and an unpaired
interval on a difference of two aggregates would be wider than the evidence
warrants. That is why the per-unit loops below exist rather than two calls to
evaluate.copying_eval: those functions aggregate internally and hand back a
point with its own interval, which cannot be paired after the fact. The per-unit
loops are anchored to them in tests/test_interp.py, which asserts they reproduce
evaluate's point estimates exactly.

Interventions are forward pre-hooks on each block's o_proj, whose input is the
per-head attention output flattened over heads. Nothing in the model is edited,
so an intervention cannot survive the context manager that installed it and
cannot reach a training forward.

Reading attention weights needs the model's analysis path, since
scaled_dot_product_attention never materializes them. See model_torch.py.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from .bootstrap import DEFAULT_ALPHA, DEFAULT_N_RESAMPLES, bootstrap_ci
from .data import DEFAULT_TOKENIZED_ROOT, iter_eval_split
from .evaluate import (
    COPY_N_SEQUENCES,
    CopyingResult,
    COPY_REPEAT_LEN,
    DEFAULT_SEED,
    DOMAINS,
    ICL_EARLY,
    ICL_LATE,
    _copy_loss_delta,
    _forward_nll,
    _icl_statistic,
    _ready,
    _weighted_perplexity,
    copy_scoring_slices,
    make_copy_sequences,
)
from .model_torch import ATTN_PER_HEAD_KEY, ATTN_WEIGHTS_KEY, WANT_ATTN_WEIGHTS

# Pre-registration, "Phase B protocol". Not arguments, not configurable.
PREFIX_MATCHING_N_SEQUENCES = 256
INDUCTION_THRESHOLD = 0.2
PHASE_LOW = 0.1
PHASE_HIGH = 0.3

# Pre-registration amendment of 2026-07-17, item 2. Named here, and carried on
# every PatchingResult, so a result file states which number the claim is read
# from, instead of leaving a reader to infer it from a field name.
PATCHING_METRIC = (
    "logit(correct continuation) - logit(token the corrupted prefix would induct "
    "to), averaged over the second copy's scored positions"
)
PATCHING_SECONDARY_METRIC = (
    "raw logit on the correct continuation, averaged over the same positions. "
    "Not shift invariant, so it is reported but nothing is read from it"
)

# Under this, the clean-minus-corrupted gap is float noise and not a
# denominator, and recovered_fraction is withheld instead of reported as a ratio.
GAP_FLOOR = 1e-12

# Pre-registration amendment, item 4: patching uses 256 sequences, matching the
# prefix-matching count. Named separately from PREFIX_MATCHING_N_SEQUENCES even
# though it is the same number, because they are the same number by decision
# rather than by definition, and the second amendment moves one of them.
PATCHING_N_SEQUENCES = PREFIX_MATCHING_N_SEQUENCES

# Second amendment of 2026-07-17. The one escalation, declared in advance so it
# cannot be tuned later: where the model demonstrably copies but the gap still
# straddles zero, patching is re-run once at this count. Not automatic. The rule
# turns on a copying-accuracy reading, which is a judgment, and a threshold that
# moves until it produces a number is not a threshold.
PATCHING_ESCALATED_N_SEQUENCES = 1024

# The vocabulary recovered_fraction_status draws from. A withheld fraction has
# to say which kind of absent it is: "no number here" and "no number here
# because the denominator could not be told from zero" are different facts, and
# a reader must not have to infer which one a null meant.
FRACTION_REPORTED = "reported"
FRACTION_WITHHELD_NO_COPYING_BEHAVIOR = "withheld_no_copying_behavior"
FRACTION_WITHHELD_NO_COPYING_READING = "withheld_no_copying_reading"
FRACTION_WITHHELD_GAP_INCLUDES_ZERO = "withheld_gap_includes_zero"
FRACTION_WITHHELD_DEGENERATE_GAP = "withheld_degenerate_gap"
FRACTION_NOT_COMPUTED = "not_computed"
FRACTION_STATUSES = (
    FRACTION_REPORTED,
    FRACTION_WITHHELD_NO_COPYING_BEHAVIOR,
    FRACTION_WITHHELD_NO_COPYING_READING,
    FRACTION_WITHHELD_GAP_INCLUDES_ZERO,
    FRACTION_WITHHELD_DEGENERATE_GAP,
    FRACTION_NOT_COMPUTED,
)

# Cost only. An attention-weight forward holds n_layers tensors of
# [batch, n_heads, seq, seq]: at size120m, 256-token copying sequences and a
# batch of 8, that is 16 x 12 x 8 x 256 x 256 x 4 bytes, about 400 MB. Hence a
# smaller default here than the batch of 16 evaluate.py uses for loss-only work.
DEFAULT_WEIGHTS_BATCH_SIZE = 8
DEFAULT_BATCH_SIZE = 16

Head = tuple[int, int]


@dataclass(frozen=True)
class Estimate:
    """A point estimate and its percentile bootstrap interval."""

    point: float
    ci_low: float
    ci_high: float


def _estimate(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_resamples: int,
    seed: int,
    alpha: float,
) -> Estimate:
    point, lo, hi = bootstrap_ci(
        values, statistic, n_resamples=n_resamples, seed=seed, alpha=alpha
    )
    return Estimate(point=point, ci_low=lo, ci_high=hi)


def iter_batches(sequences: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    """Split [n, t] token ids into batches of rows, in order."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    for start in range(0, sequences.shape[0], batch_size):
        yield sequences[start : start + batch_size]


def head_geometry(model: torch.nn.Module) -> tuple[int, int]:
    """(n_heads, head_dim) as the model's own config states them."""
    return int(model.cfg.n_heads), int(model.cfg.head_dim)


# ----------------------------------------------------------------------------
# prefix matching
# ----------------------------------------------------------------------------

def prefix_matching_indices(repeat_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(query positions, key positions) the prefix-matching score averages over.

    The sequence is a block of random tokens repeated twice: position i and
    position repeat_len + i hold the same token. A token at its second
    occurrence sits at query repeat_len + i; the earlier occurrence of that
    token is position i; the token immediately following that earlier occurrence
    is at key i + 1. So the score reads weight[repeat_len + i, i + 1].

    i runs 0 .. repeat_len - 2. The last query of the sequence is dropped: for
    i = repeat_len - 1 the key would be repeat_len, which is the first token of
    the second copy instead of a continuation of the first, and that query
    position predicts nothing inside the sequence anyway. Every remaining pair
    is causal, since i + 1 <= repeat_len + i always.
    """
    if repeat_len < 2:
        raise ValueError(f"repeat_len must be at least 2, got {repeat_len}")
    queries = torch.arange(repeat_len, 2 * repeat_len - 1)
    keys = torch.arange(1, repeat_len)
    return queries, keys


@torch.no_grad()
def prefix_matching_per_sequence(
    model: torch.nn.Module,
    device: str | torch.device,
    n_sequences: int = PREFIX_MATCHING_N_SEQUENCES,
    repeat_len: int = COPY_REPEAT_LEN,
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_WEIGHTS_BATCH_SIZE,
) -> np.ndarray:
    """[n_sequences, n_layers, n_heads] of per-sequence prefix-matching scores.

    One sequence's score for one head is the mean over the query positions above
    of the attention weight it pays to the matching key. Kept per sequence so a
    bootstrap over sequences has units to resample; prefix_matching_score is the
    mean of this over sequences, which is the pre-registered quantity.
    """
    _ready(model, device)
    cfg = model.cfg
    seq_len = 2 * repeat_len
    if seq_len > int(cfg.context_len):
        raise ValueError(
            f"two copies of {repeat_len} tokens exceed the model context "
            f"{cfg.context_len}"
        )

    seqs = make_copy_sequences(n_sequences, repeat_len, int(cfg.vocab_size), seed)
    queries, keys = prefix_matching_indices(repeat_len)

    per_sequence: list[np.ndarray] = []
    for chunk in iter_batches(seqs, batch_size):
        x = torch.from_numpy(np.ascontiguousarray(chunk)).to(device)
        capture: dict = {WANT_ATTN_WEIGHTS: True}
        model(x, capture=capture)
        weights = capture[ATTN_WEIGHTS_KEY]
        q = queries.to(x.device)
        k = keys.to(x.device)
        # [n_layers, batch, n_heads]: gather the (query, key) pairs, then average
        # over them. Advanced indexing pairs q and k elementwise.
        layer_scores = [
            weights[layer][:, :, q, k].float().mean(dim=-1).cpu().numpy()
            for layer in sorted(weights)
        ]
        per_sequence.append(np.stack(layer_scores, axis=1))

    if not per_sequence:
        raise ValueError("no sequences to score")
    return np.concatenate(per_sequence, axis=0).astype(np.float64)


def prefix_matching_score(
    model: torch.nn.Module,
    device: str | torch.device,
    n_sequences: int = PREFIX_MATCHING_N_SEQUENCES,
    repeat_len: int = COPY_REPEAT_LEN,
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_WEIGHTS_BATCH_SIZE,
) -> np.ndarray:
    """[n_layers, n_heads] prefix-matching scores, the pre-registered induction
    head score.

    On repeated-random-token sequences, the mean attention weight paid by a
    token at its second occurrence to the token immediately following the
    earlier occurrence of that token. Averaged over query positions and over
    n_sequences sequences, per head, per layer. The pre-registration fixes
    n_sequences at 256 and the copying task at repeat_len 128.

    Chance is small and depends on the geometry, not on the model: a head
    spreading its attention uniformly over the causal prefix pays 1/(q + 1) to
    each key it can see, so it scores the mean of 1/(repeat_len + i + 1) over
    the scored queries, about 0.0054 at repeat_len 128. A head that has learned
    exact induction scores near 1. The 0.2 threshold sits between the two by a
    wide margin either way.

    One place the construction is not perfectly clean, worth knowing when
    reading a number: token ids are drawn uniformly at random, so a block of 128
    of them contains a repeated id fairly often, and for those queries the
    phrase "the earlier occurrence" names more than one position. The score
    reads the occurrence the construction put there (position i for query
    repeat_len + i) and ignores the coincidental one, so a head that splits its
    attention across both scores slightly below what it deserves. The bias is
    small and it is downward, so it cannot manufacture an induction head.
    """
    return prefix_matching_per_sequence(
        model,
        device,
        n_sequences=n_sequences,
        repeat_len=repeat_len,
        seed=seed,
        batch_size=batch_size,
    ).mean(axis=0)


def prefix_matching_intervals(
    per_sequence: np.ndarray,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) arrays shaped [n_layers, n_heads], bootstrapped over sequences."""
    n, n_layers, n_heads = per_sequence.shape
    lo = np.empty((n_layers, n_heads), dtype=np.float64)
    hi = np.empty((n_layers, n_heads), dtype=np.float64)
    for layer in range(n_layers):
        for head in range(n_heads):
            _, lo[layer, head], hi[layer, head] = bootstrap_ci(
                per_sequence[:, layer, head],
                lambda v: float(v.mean()),
                n_resamples=n_resamples,
                seed=seed,
                alpha=alpha,
            )
    return lo, hi


def identify_induction_heads(
    scores: np.ndarray, threshold: float = INDUCTION_THRESHOLD
) -> list[Head]:
    """(layer, head) pairs scoring above the pre-registered threshold.

    Above means strictly above, as the pre-registration words it: "a
    pre-registered threshold of prefix-matching score above 0.2". The default is
    that 0.2 and there is no reason to pass anything else; the argument exists
    so a test can drive the boundary.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"scores must be [n_layers, n_heads], got {scores.shape}")
    layers, heads = np.nonzero(scores > threshold)
    return [(int(l), int(h)) for l, h in zip(layers, heads)]


def matched_control_heads(
    scores: np.ndarray, induction_heads: Sequence[Head], seed: int
) -> list[Head]:
    """An equal number of non-induction heads, drawn from the same layers.

    This is the comparison the whole phase rests on, so the rule is written out
    rather than left to be inferred:

      1. Count the induction heads each layer contributes. Call it k for that
         layer. A layer contributing none contributes no controls.
      2. From that same layer, draw k heads uniformly without replacement from
         the heads that are not on the induction list.

    So the control set matches the induction set on size and on the distribution
    over layers, head for head. Matching on layer matters because depth is not
    incidental: induction heads sit in particular layers, and a control drawn
    from layer 0 against an induction head in layer 6 would compare two
    different positions in the residual stream instead of two heads.

    Eligibility is "not on the induction list", which is the
    pre-registration's non-induction head. A head scoring 0.19 is eligible. It
    is tempting to exclude near-threshold heads on the grounds that they might
    be induction heads that missed the cut, but doing so would stack the control
    set with heads chosen for being inert, which is precisely the bias the
    control exists to rule out. The 0.2 line was fixed in advance; it is honored
    on both sides of the comparison.

    Deterministic given seed. Raises when a layer cannot supply enough controls,
    rather than borrowing from a neighboring layer, since a silently unmatched
    control set would answer a different question than the one asked.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"scores must be [n_layers, n_heads], got {scores.shape}")
    n_layers, n_heads = scores.shape

    induction = set()
    for layer, head in induction_heads:
        if not (0 <= layer < n_layers and 0 <= head < n_heads):
            raise ValueError(
                f"induction head ({layer}, {head}) is outside a "
                f"{n_layers} x {n_heads} score grid"
            )
        induction.add((int(layer), int(head)))

    wanted: dict[int, int] = {}
    for layer, _ in sorted(induction):
        wanted[layer] = wanted.get(layer, 0) + 1

    rng = np.random.default_rng(seed)
    controls: list[Head] = []
    for layer in sorted(wanted):
        k = wanted[layer]
        eligible = [h for h in range(n_heads) if (layer, h) not in induction]
        if len(eligible) < k:
            raise ValueError(
                f"layer {layer} holds {k} induction heads and only "
                f"{len(eligible)} non-induction heads, so it cannot supply a "
                "matched control set. Matched controls must come from the same "
                "layer, so this is reported rather than worked around."
            )
        chosen = rng.choice(np.asarray(eligible), size=k, replace=False)
        controls.extend((layer, int(h)) for h in sorted(chosen))
    return controls


# ----------------------------------------------------------------------------
# interventions
# ----------------------------------------------------------------------------

@contextmanager
def head_intervention(
    model: torch.nn.Module, fn: Callable[[int, torch.Tensor], torch.Tensor]
) -> Iterator[None]:
    """Rewrite every layer's per-head attention output for the duration.

    fn(layer_idx, per_head) takes and returns [batch, seq, n_heads, head_dim],
    the value-weighted sum each head produces before the out-projection mixes
    the heads together. That tensor is the last point at which a head is still a
    separable object, which is why interventions land here.

    Installed as a forward pre-hook on each block's o_proj, whose input is that
    same tensor flattened over heads: head h owns channels
    [h * head_dim, (h + 1) * head_dim). The model itself is untouched, and the
    hooks come off on the way out even if the body raises, so no intervention
    can outlive its context or reach a training forward.
    """
    n_heads, head_dim = head_geometry(model)
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(layer_idx: int):
        def hook(module: torch.nn.Module, args: tuple):
            (x,) = args
            b, t, _ = x.shape
            per_head = fn(layer_idx, x.reshape(b, t, n_heads, head_dim))
            return (per_head.reshape(b, t, n_heads * head_dim),)

        return hook

    try:
        for i, block in enumerate(model.blocks):
            handles.append(block.attn.o_proj.register_forward_pre_hook(make_hook(i)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _group_by_layer(heads: Sequence[Head]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for layer, head in heads:
        out.setdefault(int(layer), []).append(int(head))
    return out


@dataclass(frozen=True)
class HeadMeans:
    """Per-head mean attention output over an evaluation distribution.

    per_layer[layer] is [seq_len, n_heads, head_dim] when per_position is True
    and [1, n_heads, head_dim] when it is False.
    """

    per_layer: dict[int, torch.Tensor]
    per_position: bool
    n_sequences: int
    seq_len: int

    def for_layer(
        self, layer: int, t: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        mean = self.per_layer[layer]
        if self.per_position:
            if t > mean.shape[0]:
                raise ValueError(
                    f"per-position means cover {mean.shape[0]} positions but the "
                    f"forward is {t} long. Means must be computed on the same "
                    "distribution they are applied to."
                )
            mean = mean[:t]
        return mean.to(device=device, dtype=dtype)


@torch.no_grad()
def head_output_means(
    model: torch.nn.Module,
    batches: Iterable[np.ndarray],
    device: str | torch.device,
    per_position: bool = True,
) -> HeadMeans:
    """Mean per-head attention output over the batches handed in.

    The pre-registration ablates a head by "replacing the head output with its
    mean over the evaluation distribution instead of zero, since zero is off
    distribution". So the mean belongs to the distribution the metric is
    measured on: copying means come from copying sequences, perplexity means
    from that split's windows. It is never a global constant reused across
    tasks.

    per_position was the one genuine reading ambiguity in the sentence above,
    and the amendment of 2026-07-17, item 1, settled it as the per-position
    reading. What the two mean:

      True, the default, and the pre-registered headline. Average over sequences
        at each position, keeping the mean's dependence on position. The
        evaluation distribution is a distribution over sequences, all of them
        the same length and position-aligned here, so this is the mean of the
        head's output under that distribution, position by position. It is the
        reading that honors the stated reason for using a mean at all: at
        position p, a constant averaged across all positions is itself off
        distribution for any head whose output moves with position. And an
        induction head is defined by a content-dependent computation, which a
        per-position mean removes while leaving the positional baseline intact,
        so it is the contrast the claim is about.
      False. Average over sequences and positions both, giving one vector per
        head, which is the more common plain-mean-ablation convention. The
        amendment keeps it as a sensitivity check reported alongside, never as
        the headline.

    The choice matters for the control heads specifically: a head whose output
    is largely a function of position survives a per-position mean nearly
    intact, and is destroyed by a global one.
    """
    _ready(model, device)
    totals: dict[int, torch.Tensor] = {}
    n_sequences = 0
    seq_len = 0

    for chunk in batches:
        x = torch.from_numpy(np.ascontiguousarray(chunk)).to(device)
        if seq_len and int(x.shape[1]) != seq_len:
            raise ValueError(
                f"batches are ragged: saw length {seq_len} and then {x.shape[1]}. "
                "A mean over positions needs one aligned length."
            )
        seq_len = int(x.shape[1])
        capture: dict = {}
        model(x, capture=capture)
        for layer, per_head in capture[ATTN_PER_HEAD_KEY].items():
            # [batch, seq, n_heads, head_dim] summed over the batch. cpu() first
            # because MPS has no float64; the accumulation across batches stays
            # float64 on the CPU, and for_layer() casts back to the consumer's
            # device and dtype at use time.
            summed = per_head.cpu().double().sum(dim=0)
            if not per_position:
                summed = summed.sum(dim=0, keepdim=True)
            totals[layer] = summed if layer not in totals else totals[layer] + summed
        n_sequences += int(x.shape[0])

    if not totals:
        raise ValueError("no batches to average over")

    divisor = float(n_sequences if per_position else n_sequences * seq_len)
    return HeadMeans(
        per_layer={l: (v / divisor).to(torch.float32) for l, v in totals.items()},
        per_position=per_position,
        n_sequences=n_sequences,
        seq_len=seq_len,
    )


@contextmanager
def mean_ablate_heads(
    model: torch.nn.Module, heads: Sequence[Head], means: HeadMeans
) -> Iterator[None]:
    """Replace the named heads' outputs with their mean, for the duration.

    Mean, not zero: the pre-registration is explicit that zero is off
    distribution and that the replacement is the head's mean over the evaluation
    distribution. Each head gets its own mean; heads not named are untouched;
    every other layer runs unmodified.

    What this destroys is the head's dependence on the input. What it leaves is
    whatever the head contributes on average, so the downstream layers still
    receive something of the right scale from the right place. That is the whole
    point of the choice.
    """
    by_layer = _group_by_layer(heads)

    def fn(layer_idx: int, per_head: torch.Tensor) -> torch.Tensor:
        targets = by_layer.get(layer_idx)
        if not targets:
            return per_head
        out = per_head.clone()
        mean = means.for_layer(layer_idx, per_head.shape[1], per_head.device, per_head.dtype)
        for head in targets:
            out[:, :, head, :] = mean[:, head, :]
        return out

    with head_intervention(model, fn):
        yield


@contextmanager
def patch_heads(
    model: torch.nn.Module, patches: Mapping[Head, torch.Tensor]
) -> Iterator[None]:
    """Overwrite the named heads' outputs with the given activations.

    patches[(layer, head)] is [batch, seq, head_dim], the output that head
    produced on some other run. Everything else runs on the current input, so
    the forward is the corrupted run with one head's worth of the clean run
    spliced in.

    Patching a head with its own activations from the same run must be exactly a
    no-op. tests/test_interp.py asserts it, because an off-by-one in the head
    axis would still produce a plausible-looking recovery curve.
    """
    by_layer = _group_by_layer(list(patches))

    def fn(layer_idx: int, per_head: torch.Tensor) -> torch.Tensor:
        targets = by_layer.get(layer_idx)
        if not targets:
            return per_head
        out = per_head.clone()
        for head in targets:
            value = patches[(layer_idx, head)]
            if value.shape != (per_head.shape[0], per_head.shape[1], per_head.shape[3]):
                raise ValueError(
                    f"patch for head ({layer_idx}, {head}) is {tuple(value.shape)}, "
                    f"expected {(per_head.shape[0], per_head.shape[1], per_head.shape[3])}"
                )
            out[:, :, head, :] = value.to(device=per_head.device, dtype=per_head.dtype)
        return out

    with head_intervention(model, fn):
        yield


# ----------------------------------------------------------------------------
# per-unit measurements, paired to evaluate.py's statistics
# ----------------------------------------------------------------------------

@torch.no_grad()
def copying_per_sequence(
    model: torch.nn.Module,
    device: str | torch.device,
    n_sequences: int = COPY_N_SEQUENCES,
    repeat_len: int = COPY_REPEAT_LEN,
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """[n_sequences, 3]: exact-match rate, second-copy loss, first-copy loss.

    The same sequences, the same scored slices and the same two quantities as
    evaluate.copying_eval, kept per sequence instead of aggregated so a delta
    can be bootstrapped paired. tests/test_interp.py asserts that averaging
    these columns reproduces copying_eval's point estimates exactly, which is
    what keeps this from drifting into a second, subtly different copying task.
    """
    _ready(model, device)
    cfg = model.cfg
    if 2 * repeat_len > int(cfg.context_len):
        raise ValueError(
            f"two copies of {repeat_len} tokens exceed the model context "
            f"{cfg.context_len}"
        )
    seqs = make_copy_sequences(n_sequences, repeat_len, int(cfg.vocab_size), seed)
    first, second = copy_scoring_slices(repeat_len)

    rows: list[np.ndarray] = []
    for chunk in iter_batches(seqs, batch_size):
        arr = torch.from_numpy(np.ascontiguousarray(chunk)).to(device)
        x, y = arr[:, :-1], arr[:, 1:]
        logits, nll = _forward_nll(model, x, y)
        # cpu() first: MPS has no float64. Bool to float64 is exact, and the
        # mean then accumulates in float64 on the CPU as intended.
        match = (
            (logits[:, second, :].argmax(dim=-1) == y[:, second])
            .cpu()
            .double()
            .mean(dim=1)
        )
        rows.append(
            np.stack(
                [
                    match.cpu().numpy(),
                    nll[:, second].mean(dim=1).cpu().double().numpy(),
                    nll[:, first].mean(dim=1).cpu().double().numpy(),
                ],
                axis=1,
            )
        )
    return np.concatenate(rows, axis=0)


def _eval_batches(
    domain: str,
    split: str,
    context_len: int,
    batch_size: int,
    root: Path | str,
    max_windows: Optional[int],
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """iter_eval_split, optionally stopped early.

    max_windows is a cost control and nothing else. It does not change any
    definition: the windows it keeps are the first windows the pre-registered
    exhaustive tiling produces, in order, and the count actually used is
    recorded in the result so a truncated ablation sweep can never be read as a
    full one. The headline perplexity in evaluate.py never truncates.
    """
    taken = 0
    for x, y in iter_eval_split(domain, split, context_len, batch_size, root):
        if max_windows is not None:
            room = max_windows - taken
            if room <= 0:
                return
            if room < x.shape[0]:
                x, y = x[:room], y[:room]
        taken += x.shape[0]
        yield x, y


@torch.no_grad()
def window_losses(
    model: torch.nn.Module,
    domain: str,
    split: str,
    device: str | torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
    root: Path | str = DEFAULT_TOKENIZED_ROOT,
    max_windows: Optional[int] = None,
) -> np.ndarray:
    """[n_windows, 4]: mean nll, token count, late-window mean, early-window mean.

    Columns 0 and 1 feed evaluate._weighted_perplexity, columns 2 and 3 feed
    evaluate._icl_statistic. Both statistics are imported rather than restated,
    so a perplexity delta here is a delta of the pre-registered perplexity.
    tests/test_interp.py asserts the reproduction against evaluate.test_perplexity
    and evaluate.icl_score.
    """
    _ready(model, device)
    context_len = int(model.cfg.context_len)
    if context_len < ICL_LATE[1]:
        raise ValueError(
            f"context {context_len} is too short for the pre-registered late "
            f"window {ICL_LATE}"
        )

    rows: list[np.ndarray] = []
    for x_np, y_np in _eval_batches(
        domain, split, context_len, batch_size, root, max_windows
    ):
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
        _, nll = _forward_nll(model, x, y)
        rows.append(
            np.stack(
                [
                    nll.mean(dim=1).cpu().double().numpy(),
                    np.full(nll.shape[0], nll.shape[1], dtype=np.float64),
                    nll[:, ICL_LATE[0] : ICL_LATE[1]].mean(dim=1).cpu().double().numpy(),
                    nll[:, ICL_EARLY[0] : ICL_EARLY[1]].mean(dim=1).cpu().double().numpy(),
                ],
                axis=1,
            )
        )
    if not rows:
        raise ValueError(f"no windows for domain {domain!r} split {split!r}")
    return np.concatenate(rows, axis=0)


# ----------------------------------------------------------------------------
# ablation
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Comparison:
    """One metric under baseline and under ablation, with a paired delta."""

    baseline: float
    ablated: float
    delta: Estimate
    n_units: int


@dataclass(frozen=True)
class DomainBaseline:
    means: HeadMeans
    windows: np.ndarray


@dataclass(frozen=True)
class AblationContext:
    """Everything that does not depend on which heads are ablated.

    The means and the baseline runs are properties of the evaluation
    distribution, not of the intervention, so they are computed once and reused
    across every condition: each induction head singly, all of them jointly, and
    the matched controls. Recomputing them per condition would cost a full pass
    over the corpus per head and, worse, would let the baseline drift between
    the conditions being compared.
    """

    copy_baseline: np.ndarray
    copy_means: HeadMeans
    domains: dict[str, DomainBaseline]
    split: str
    n_sequences: int
    repeat_len: int
    seed: int
    per_position_means: bool
    max_windows: Optional[int]
    # Carried so the ablated run is read the same way the baseline was. A
    # paired bootstrap over windows is only paired if both sides saw the same
    # windows, which means the same root and the same batching.
    root: str
    batch_size: int


def build_ablation_context(
    model: torch.nn.Module,
    device: str | torch.device,
    split: str = "val",
    domains: Sequence[str] = DOMAINS,
    n_sequences: int = COPY_N_SEQUENCES,
    repeat_len: int = COPY_REPEAT_LEN,
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    root: Path | str = DEFAULT_TOKENIZED_ROOT,
    per_position_means: bool = True,
    max_windows: Optional[int] = None,
) -> AblationContext:
    """Baselines and per-distribution head means, computed once.

    domains may be empty, which measures copying only. That is the one metric of
    the three that needs no corpus, so it is also the one an ablation can be
    checked on without the tokenized streams mounted.
    """
    cfg = model.cfg
    seqs = make_copy_sequences(n_sequences, repeat_len, int(cfg.vocab_size), seed)
    copy_inputs = seqs[:, :-1]  # what the model is actually fed
    copy_means = head_output_means(
        model, iter_batches(copy_inputs, batch_size), device, per_position_means
    )
    copy_baseline = copying_per_sequence(
        model,
        device,
        n_sequences=n_sequences,
        repeat_len=repeat_len,
        seed=seed,
        batch_size=batch_size,
    )

    per_domain: dict[str, DomainBaseline] = {}
    for domain in domains:
        means = head_output_means(
            model,
            (
                x
                for x, _ in _eval_batches(
                    domain,
                    split,
                    int(cfg.context_len),
                    batch_size,
                    root,
                    max_windows,
                )
            ),
            device,
            per_position_means,
        )
        windows = window_losses(
            model,
            domain,
            split,
            device,
            batch_size=batch_size,
            root=root,
            max_windows=max_windows,
        )
        per_domain[domain] = DomainBaseline(means=means, windows=windows)

    return AblationContext(
        copy_baseline=copy_baseline,
        copy_means=copy_means,
        domains=per_domain,
        split=split,
        n_sequences=n_sequences,
        repeat_len=repeat_len,
        seed=seed,
        per_position_means=per_position_means,
        max_windows=max_windows,
        root=str(root),
        batch_size=batch_size,
    )


@dataclass(frozen=True)
class AblationEffect:
    """What ablating one set of heads did to the pre-registered metrics."""

    heads: tuple[Head, ...]
    copying_exact_match: Comparison
    copying_loss_delta: Comparison
    perplexity: dict[str, Comparison]
    icl: dict[str, Comparison]
    split: str
    n_sequences: int
    repeat_len: int
    seed: int
    per_position_means: bool
    n_windows: dict[str, int]


def _delta_exact_match(values: np.ndarray) -> float:
    """Ablated exact match minus baseline. Columns: base acc, ablated acc."""
    return float(values[:, 1].mean() - values[:, 0].mean())


def _delta_copy_loss_delta(values: np.ndarray) -> float:
    """Change in the pre-registered copying loss delta.

    Columns: baseline second, baseline first, ablated second, ablated first.
    """
    return _copy_loss_delta(values[:, 2:4]) - _copy_loss_delta(values[:, 0:2])


def _delta_perplexity(values: np.ndarray) -> float:
    """Ablated perplexity minus baseline.

    Columns: baseline nll, baseline count, ablated nll, ablated count.
    """
    return _weighted_perplexity(values[:, 2:4]) - _weighted_perplexity(values[:, 0:2])


def _delta_icl(values: np.ndarray) -> float:
    """Ablated in-context score minus baseline.

    Columns: baseline late, baseline early, ablated late, ablated early.
    """
    return _icl_statistic(values[:, 2:4]) - _icl_statistic(values[:, 0:2])


def ablation_effect(
    model: torch.nn.Module,
    heads: Sequence[Head],
    device: str | torch.device,
    context: Optional[AblationContext] = None,
    split: str = "val",
    domains: Sequence[str] = DOMAINS,
    n_sequences: int = COPY_N_SEQUENCES,
    repeat_len: int = COPY_REPEAT_LEN,
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    root: Path | str = DEFAULT_TOKENIZED_ROOT,
    per_position_means: bool = True,
    max_windows: Optional[int] = None,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
) -> AblationEffect:
    """Mean-ablate `heads` and measure the damage, with paired intervals.

    Three pre-registered metrics: copying exact match, the in-context learning
    score, and test perplexity, plus the copying loss delta that the copying
    section names alongside exact match. Each is reported at baseline, under
    ablation, and as a delta with a bootstrap interval over evaluation
    sequences.

    The delta's interval is paired: a resample draws sequence indices once and
    reads the baseline and the ablated value of those same sequences. Both runs
    see the identical sequences, so the pairing is real and not assumed, and
    the interval measures the variability of the difference instead of the sum
    of two independent variabilities. The pre-registered claim compares
    intervals on the damage, so it is the delta's interval that has to be right.

    Sign conventions, since three of these metrics point in different
    directions: a negative exact-match delta means ablation hurt copying. A
    positive copying-loss delta and a positive perplexity delta mean the same
    thing. The in-context learning score is itself negative when the model uses
    context, so a positive delta there means ablation cost the model context.

    Passing `context` reuses one set of baselines and means across conditions,
    which is what the CLI does. Leaving it None builds one for this call, so the
    function stands alone. A supplied context wins over every argument below it
    that describes the evaluation distribution: the sequences, the seed, the
    split, the domains, the root, the batching and the choice of mean. It has
    to, since those arguments are what the baseline it carries was measured
    under, and a delta against a baseline measured under something else is not
    a delta.
    """
    if context is None:
        context = build_ablation_context(
            model,
            device,
            split=split,
            domains=domains,
            n_sequences=n_sequences,
            repeat_len=repeat_len,
            seed=seed,
            batch_size=batch_size,
            root=root,
            per_position_means=per_position_means,
            max_windows=max_windows,
        )

    heads = [(int(l), int(h)) for l, h in heads]

    with mean_ablate_heads(model, heads, context.copy_means):
        ablated_copy = copying_per_sequence(
            model,
            device,
            n_sequences=context.n_sequences,
            repeat_len=context.repeat_len,
            seed=context.seed,
            batch_size=context.batch_size,
        )

    base = context.copy_baseline
    acc_values = np.stack([base[:, 0], ablated_copy[:, 0]], axis=1)
    exact_match = Comparison(
        baseline=float(base[:, 0].mean()),
        ablated=float(ablated_copy[:, 0].mean()),
        delta=_estimate(acc_values, _delta_exact_match, n_resamples, context.seed, alpha),
        n_units=int(base.shape[0]),
    )
    loss_values = np.concatenate([base[:, 1:3], ablated_copy[:, 1:3]], axis=1)
    loss_delta = Comparison(
        baseline=_copy_loss_delta(base[:, 1:3]),
        ablated=_copy_loss_delta(ablated_copy[:, 1:3]),
        delta=_estimate(
            loss_values, _delta_copy_loss_delta, n_resamples, context.seed, alpha
        ),
        n_units=int(base.shape[0]),
    )

    perplexity: dict[str, Comparison] = {}
    icl: dict[str, Comparison] = {}
    n_windows: dict[str, int] = {}
    for domain, baseline in context.domains.items():
        with mean_ablate_heads(model, heads, baseline.means):
            ablated_windows = window_losses(
                model,
                domain,
                context.split,
                device,
                batch_size=context.batch_size,
                root=context.root,
                max_windows=context.max_windows,
            )
        if ablated_windows.shape[0] != baseline.windows.shape[0]:
            raise ValueError(
                f"domain {domain!r} yielded {ablated_windows.shape[0]} windows under "
                f"ablation and {baseline.windows.shape[0]} at baseline; the paired "
                "bootstrap needs the same windows on both sides"
            )
        ppl_values = np.concatenate(
            [baseline.windows[:, 0:2], ablated_windows[:, 0:2]], axis=1
        )
        perplexity[domain] = Comparison(
            baseline=_weighted_perplexity(baseline.windows[:, 0:2]),
            ablated=_weighted_perplexity(ablated_windows[:, 0:2]),
            delta=_estimate(
                ppl_values, _delta_perplexity, n_resamples, context.seed, alpha
            ),
            n_units=int(baseline.windows.shape[0]),
        )
        icl_values = np.concatenate(
            [baseline.windows[:, 2:4], ablated_windows[:, 2:4]], axis=1
        )
        icl[domain] = Comparison(
            baseline=_icl_statistic(baseline.windows[:, 2:4]),
            ablated=_icl_statistic(ablated_windows[:, 2:4]),
            delta=_estimate(icl_values, _delta_icl, n_resamples, context.seed, alpha),
            n_units=int(baseline.windows.shape[0]),
        )
        n_windows[domain] = int(baseline.windows.shape[0])

    return AblationEffect(
        heads=tuple(heads),
        copying_exact_match=exact_match,
        copying_loss_delta=loss_delta,
        perplexity=perplexity,
        icl=icl,
        split=context.split,
        n_sequences=context.n_sequences,
        repeat_len=context.repeat_len,
        seed=context.seed,
        per_position_means=context.per_position_means,
        n_windows=n_windows,
    )


# ----------------------------------------------------------------------------
# activation patching
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class HeadPatch:
    """What splicing one head's clean output into the corrupted run bought.

    The logit_diff fields are the pre-registered metric and are what the claim
    is read from. The correct_logit fields are the raw logit on the correct
    continuation: secondary, reported because it costs nothing beyond a gather
    that has already happened, and not shift invariant, which is why the
    amendment took it off the headline. See activation_patching.

    recovered_fraction is None whenever it is withheld, and
    recovered_fraction_status says which of the reasons in FRACTION_STATUSES
    applies. The two travel together and the invariant is exact: the status is
    FRACTION_REPORTED if and only if the fraction is not None. None rather than
    an Estimate full of NaN is deliberate. NaN propagates silently through
    arithmetic and prints as a number-shaped thing; None does not, so a caller
    that has not thought about the withheld case fails at it instead of
    publishing it.
    """

    layer: int
    head: int
    patched_logit_diff: float
    recovered_logit_diff: Estimate
    recovered_fraction: Optional[Estimate]
    recovered_fraction_status: str
    patched_correct_logit: float
    recovered_correct_logit: Estimate


@dataclass(frozen=True)
class PatchingResult:
    """The three runs' scores and what each head recovered.

    metric and secondary_metric are carried rather than left implicit, so a
    result file says in words which of these numbers the pre-registered claim
    is read from.
    """

    metric: str
    secondary_metric: str
    clean_logit_diff: float
    corrupted_logit_diff: float
    gap: Estimate
    gap_excludes_zero: bool
    # The positive control the fraction is gated on, carried so the gate can be
    # audited from the result alone rather than trusted.
    copying_exact_match: Optional[Estimate]
    copying_chance: float
    copying_clears_chance: bool
    clean_correct_logit: float
    corrupted_correct_logit: float
    correct_logit_gap: float
    heads: tuple[HeadPatch, ...]
    n_sequences: int
    repeat_len: int
    seed: int
    corrupt_seed: int


def corrupted_copy_sequences(
    clean: np.ndarray, repeat_len: int, vocab_size: int, corrupt_seed: int
) -> np.ndarray:
    """The clean sequences with the earlier occurrence replaced.

    Clean is [A, A]. This returns [A', A]: the first copy becomes an independent
    draw of random tokens, the second copy is untouched. So the correct
    continuation at every scored position is the same token in both runs, and
    the only thing the corruption removes is the earlier occurrence an induction
    head would have to find. A', like A, comes from make_copy_sequences, so both
    halves are drawn from the same distribution and the corruption is not also a
    change of distribution.

    A' is also what the metric differences against: an induction head reading
    this prefix lands on a token of A' instead of one of A, and that token is
    the second half of the pre-registered logit difference. See
    induction_target_tokens.
    """
    other = make_copy_sequences(clean.shape[0], repeat_len, vocab_size, corrupt_seed)
    return np.concatenate([other[:, :repeat_len], clean[:, repeat_len:]], axis=1)


def _check_copy_shape(seqs: np.ndarray, repeat_len: int) -> None:
    if seqs.ndim != 2 or seqs.shape[1] != 2 * repeat_len:
        raise ValueError(
            f"expected [n, {2 * repeat_len}] copying sequences, got {seqs.shape}"
        )


def correct_continuation_tokens(clean: np.ndarray, repeat_len: int) -> np.ndarray:
    """[n, repeat_len - 1]: the correct next token at each scored position.

    One half of the pre-registered logit difference. The scored positions are
    the second slice copy_scoring_slices names, prediction indices repeat_len ..
    2 * repeat_len - 2. Prediction index repeat_len + i predicts sequence
    position repeat_len + i + 1, which in a sequence [A, A] holds A[i + 1]. So
    column i is clean[:, repeat_len + i + 1].

    The corruption rewrites only the first copy, so these are the same tokens in
    the corrupted run. Corrupting the first copy alone leaves the correct answer
    in place and moves only the evidence for it.
    """
    _check_copy_shape(clean, repeat_len)
    return clean[:, repeat_len + 1 : 2 * repeat_len]


def induction_target_tokens(corrupt: np.ndarray, repeat_len: int) -> np.ndarray:
    """[n, repeat_len - 1]: the token the corrupted prefix would induct to.

    The other half of the pre-registered logit difference, and like the first it
    is fixed by the construction and not picked after seeing an output.

    An induction head at query repeat_len + i reads key i + 1, the token
    following the earlier occurrence of the token it is holding. That is the
    same cell prefix_matching_indices scores, so the induction head score and
    this function name the same position by the same rule, which is what makes
    the difference the contrast the corruption actually creates. In the
    corrupted run key i + 1 holds the corrupted first copy's token, so at scored
    position repeat_len + i the token inducted to is corrupt[:, i + 1].

    One coincidence worth knowing about when reading a number, and it is the one
    item 3 of the amendment already covers: the corrupted draw can land on the
    clean token, and then the two halves of the difference are one token and
    that position contributes exactly 0 instead of a signed number. At 16,383
    ids that is about one scored position in 16,000, it is not conditioned on
    anything the model did, and it pulls clean and corrupted alike toward 0, so
    it cannot manufacture a recovery.
    """
    _check_copy_shape(corrupt, repeat_len)
    return corrupt[:, 1:repeat_len]


def _to_device(arr: np.ndarray, device: str | torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device)


def _gathered_logits(
    logits: torch.Tensor, tokens: torch.Tensor, scored: slice
) -> torch.Tensor:
    """[batch, n_scored]: the logits on `tokens` at the scored prediction indices."""
    at_scored = logits[:, scored, :].float()
    return at_scored.gather(2, tokens.unsqueeze(-1)).squeeze(-1)


def _patching_scores(
    logits: torch.Tensor,
    correct: torch.Tensor,
    induction_target: torch.Tensor,
    scored: slice,
) -> np.ndarray:
    """[batch, 2]: the pre-registered logit difference, then the raw logit.

    Column 0 is the metric: logit(correct continuation) minus logit(the token
    the corrupted prefix would induct to), both read at the same prediction
    index of the same forward and averaged over the scored positions. Reading
    both at the same position is the whole of the shift invariance: whatever
    additive constant that position carries is added to both and cancels in the
    subtraction, before anything is compared across runs.

    Column 1 is the raw logit on the correct continuation, the secondary number,
    which is column 0's first term on its own and so costs nothing extra.
    """
    correct_logit = _gathered_logits(logits, correct, scored)
    target_logit = _gathered_logits(logits, induction_target, scored)
    return np.stack(
        [
            (correct_logit - target_logit).mean(dim=1).cpu().double().numpy(),
            correct_logit.mean(dim=1).cpu().double().numpy(),
        ],
        axis=1,
    )


def _recovered_logit_diff(values: np.ndarray) -> float:
    """Patched minus corrupted, in logit differences.

    Columns 0, 1: the patched and corrupted runs' logit differences.
    """
    return float(values[:, 0].mean() - values[:, 1].mean())


def _gap(values: np.ndarray) -> float:
    """Clean minus corrupted, in logit differences.

    Columns 0, 1: the clean and corrupted runs' logit differences. This is the
    denominator recovered_fraction divides by, and the quantity a patch is
    trying to buy back.
    """
    return float(values[:, 0].mean() - values[:, 1].mean())


def _excludes_zero(estimate: Estimate) -> bool:
    """Whether a bootstrap interval lies entirely on one side of zero."""
    return estimate.ci_low > 0.0 or estimate.ci_high < 0.0


def copy_chance_accuracy(vocab_size: int) -> float:
    """Exact-match accuracy on the copying task for a model that is not copying.

    The task draws tokens uniformly from the real vocabulary excluding the eot
    id, so a prediction that ignores the context lands on the target with
    probability 1 / (vocab_size - 1): about 6.1e-05 at the lab's 16384. This is
    read off the construction, not chosen, which is what lets it sit in a gate.
    """
    if vocab_size < 2:
        raise ValueError(f"vocabulary must hold at least one non-eot id, got {vocab_size}")
    return 1.0 / (vocab_size - 1)


def copying_positive_control(result: CopyingResult) -> Estimate:
    """The copying positive control, taken from the battery's own result.

    An adapter and deliberately nothing more. The number that gates the fraction
    has to be the pre-registered copying accuracy that evaluate.copying_eval
    already reports, with the interval it already computed, rather than a second
    reading taken privately by the patching code. One measurement, named once,
    read here.
    """
    return Estimate(
        point=result.exact_match,
        ci_low=result.exact_match_ci_low,
        ci_high=result.exact_match_ci_high,
    )


def copying_clears_chance(copying_exact_match: Estimate, vocab_size: int) -> bool:
    """Whether the copying interval lies entirely above chance.

    The third amendment's phrase is "the interval on copying exact-match
    accuracy lies entirely above chance", and this is that: the interval's lower
    bound sits above copy_chance_accuracy. Both numbers come from the
    construction, so there is no threshold here that anyone picked.

    Worth being exact about what this is and is not, given the amendment exists
    because a significance test was caught standing in for a magnitude question.
    In form this is still an interval compared against a reference value, so the
    form is not what saves it. The quantity is. The gap is continuous and every
    random-init model has a small systematic one, so more sequences resolve it
    and the test opens: measured, 30 percent of models at 64 sequences and 47.5
    percent at 256, rising. Exact-match accuracy against a uniform random target
    is discrete and chance is about 6.1e-05, so a model that is not copying
    produces almost no exact matches at all and the lower bound of a bootstrap
    over sequences sits at zero. Adding sequences cannot manufacture matches
    that are not there, so the bound stays at zero rather than climbing.
    Measured the same way on the same 40 random-init models, this gate opened on
    none of them, at 64, at 256 and at 1024 sequences alike, with the median
    lower bound exactly 0.0 at every count.

    That is evidence, not a proof. At some large enough n a model sitting
    systematically above chance would eventually clear it. Nothing observed here
    is such a model: the median point estimate wanders around chance rather than
    above it, 1.3e-04 at 256 sequences and 3.2e-05 at 1024. The pre-registered
    copying count is COPY_N_SEQUENCES, which those two bracket.
    """
    return copying_exact_match.ci_low > copy_chance_accuracy(vocab_size)


def _recovered_fraction(values: np.ndarray) -> float:
    """The share of the clean-minus-corrupted gap the patch bought back.

    Columns 0, 1, 2: the patched, corrupted and clean runs' logit differences.
    0 is a patch that bought nothing, 1 is a patch that recovered the clean run.

    NaN when the gap falls under GAP_FLOOR, because the gap is the denominator
    and a denominator made of float noise gives a ratio made of float noise with
    a confident-looking number printed on it.

    The guard survives the amendment, and so does the caveat around it. The
    amendment could be read as having retired the guard, and it has not. What the logit difference fixes is that the gap
    is now a well-defined quantity: a raw logit differenced across two runs
    carried whatever the two runs' per-position constants happened to do, and
    could be moved to any value at all by a shift the model is invariant to.
    What it does not fix is the size of the gap. The gap is large only when the
    model has actually learned induction, which is the thing under test, so on a
    model that has not the ratio is noise over noise whichever metric it is read
    in. Measured on random-init models at 64 sequences: the gap sits near 1e-4
    with a paired bootstrap standard error of the same order, which is eight
    orders of magnitude above GAP_FLOOR. The guard does not fire and the
    fraction that comes out is a plausible-looking number with nothing behind
    it.

    So GAP_FLOOR catches an exactly degenerate denominator, which is all it
    claims to catch. recovered_fraction is readable only where the gap is
    distinguishable from zero, and that is a judgment made from the gap and its
    spread, not from this guard.

    That judgment is made, and it is made in activation_patching rather than
    here: it reports this fraction as NaN for every head unless the gap's own
    bootstrap interval excludes zero. The two guards are different instruments
    for different failures and both are needed. This one is arithmetic, it sees
    one resample at a time, and it fires on a denominator that is not a number.
    The interval test is statistical, it is a property of the whole run, and it
    fires on a denominator that is a number but not a distinguishable one. This
    function cannot make the second call: it is the statistic the bootstrap
    resamples, so it runs once per resample and has no interval to consult, and
    a resample is not where the question is asked anyway.
    """
    gap = float(values[:, 2].mean() - values[:, 1].mean())
    if abs(gap) < GAP_FLOOR:
        return float("nan")
    return float(values[:, 0].mean() - values[:, 1].mean()) / gap


def _recovered_correct_logit(values: np.ndarray) -> float:
    """The secondary number: patched minus corrupted, in raw logits.

    Columns 3, 4: the patched and corrupted runs' raw correct-continuation
    logits.
    """
    return float(values[:, 3].mean() - values[:, 4].mean())


@torch.no_grad()
def activation_patching(
    model: torch.nn.Module,
    device: str | torch.device,
    heads: Optional[Sequence[Head]] = None,
    n_sequences: int = PATCHING_N_SEQUENCES,
    repeat_len: int = COPY_REPEAT_LEN,
    seed: int = DEFAULT_SEED,
    corrupt_seed: Optional[int] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    copying_exact_match: Optional[Estimate] = None,
) -> PatchingResult:
    """Patch clean head outputs into a corrupted copying run, head by head.

    The pre-registered design. Clean run: a block of random tokens repeated
    twice, so the second copy is predictable by finding the earlier occurrence.
    Corrupted run: the earlier occurrence replaced by different random tokens,
    so it is not. For each head, the clean run's output for that head alone is
    spliced into the corrupted run and the run is scored again.

    The metric, fixed by the pre-registration amendment of 2026-07-17, item 2:

        logit(correct continuation)
            - logit(the token the corrupted prefix would induct to)

    read at each of the second copy's scored prediction indices, the same
    positions evaluate.copying_eval scores via the shared copy_scoring_slices,
    and averaged over them. correct_continuation_tokens and
    induction_target_tokens name the two tokens; both fall out of the
    construction, so neither is chosen after seeing an output.

    Why a difference instead of the correct token's logit alone. Softmax is
    shift invariant: add a constant to every logit at a position and every
    probability at that position is exactly where it was. A logit is therefore
    defined only up to a per-position additive constant, and nothing forces two
    forwards over two different prefixes to land on the same constant. Reading
    one token's logit and differencing it across runs mixes the quantity of
    interest with whatever the constants did. Both halves of the difference
    above are read at the same position of the same forward, so the constant is
    gone before anything is compared across runs.
    tests/test_interp.py adds an arbitrary per-position constant through a
    wrapper and asserts these numbers do not move.

    Which token to difference against is the part that had to be pre-registered,
    since a wrong token picked after seeing outputs would be a free parameter.
    It is the token an induction head reading the corrupted prefix would land
    on: query repeat_len + i, key i + 1, the cell the corruption rewrote and the
    same cell the prefix-matching score reads. So the difference is exactly the
    contrast the corruption creates.

    What comes back, per run and per head:
      clean_logit_diff, corrupted_logit_diff   the metric on each run. gap is
                          their difference, with its own interval, and it is
                          what a patch can recover.
      recovered_logit_diff                     patched minus corrupted.
      recovered_fraction                       that, over gap: 0 is a patch that
                          bought nothing, 1 is a patch that recovered the clean
                          run. NaN unless gap_excludes_zero, and NaN when the
                          gap falls under GAP_FLOOR.

    Each carries a percentile bootstrap interval over sequences, paired: a
    resample draws sequence indices once and reads that sequence's clean,
    corrupted and patched values together.

    On the fraction and when it is withheld. Two gates, and they are not equals.

    The primary gate is the copying positive control, which the third amendment
    put here. copying_exact_match is the pre-registered copying exact-match
    accuracy with its interval, measured by the caller and passed in, and the
    fraction is withheld unless that interval lies entirely above chance. The
    reasoning is not statistical, it is prior to the statistics: if the model
    does not copy then there is no in-context copying to recover, and no
    arithmetic on the gap can conjure any. It is a magnitude criterion in
    behavioral units a reader can check for themselves.

    It is passed in rather than measured here on purpose. The gate is then
    auditable from the result, which carries the reading it was given, and the
    instrument cannot quietly take a second copying measurement that drifts from
    the pre-registered one in evaluate.copying_eval. copying_positive_control
    adapts that battery's result into the Estimate this wants. Passing nothing
    withholds the fraction under FRACTION_WITHHELD_NO_COPYING_READING: a gate
    with no reading fails closed, because "nobody checked" and "the model does
    not copy" are different facts and neither is license to print a ratio.

    The gap interval test is retained but demoted, and the third amendment is
    the reason. It is necessary, not sufficient. What it can do is catch a
    denominator that is degenerate or unresolvable; what it cannot do is
    establish that a fraction means anything, and it must not be described as
    though it can. It asks whether the gap differs from zero, and that is not
    the question. A random-init model's gap is not noise, it is small but
    systematic, a fixed property of the weights: it converges while the standard
    error falls as one over root n, so the ratio grows with sequences, about 0.6
    at 8, 1.7 at 64, 2.7 at 256, 5.3 at 1024. Across 40 random-init models this
    test alone opened on 47 percent of them at the pre-registered 256, and
    alpha at 0.01 only took that to 35 percent, because the effect it resolves
    is real. Left as the only gate it would let the instrument print a confident
    0.86 for a model the adjudication rule separately calls a negative result.

    So the order is: no copying reading, or copying at chance, withholds
    whatever the gap does. Only past that do the gap tests get a say.

    Withheld means recovered_fraction is None and recovered_fraction_status
    names which of FRACTION_STATUSES applies. The status is FRACTION_REPORTED if
    and only if the fraction is not None, so the two cannot disagree.

    Neither gate can move the claim. Both are properties of the run rather than
    of any head: the copying reading is of the model, the gap of the clean and
    corrupted runs. Each silences every head's fraction or none, and cannot
    silence a control head's while sparing an induction head's.

    Nothing is lost by withholding. recovered_logit_diff is the same recovery
    without the normalization, it is well defined whatever the gates decide, it
    keeps its interval, and the pre-registered induction-versus-control
    comparison reads straight off it.

    What a withheld fraction on a trained model means is the second amendment's
    subject. A withheld fraction is a negative result when the model does not
    copy, and an underpowered measurement when it does. Since the third
    amendment the first of those is decided here rather than left to the
    writeup: copying at chance withholds, under
    FRACTION_WITHHELD_NO_COPYING_BEHAVIOR, so the instrument and the
    adjudication rule can no longer contradict each other.

    The escalation is the remaining case: the model copies, so the gate is open,
    and the gap still does not clear zero. Only then is patching re-run once at
    PATCHING_ESCALATED_N_SEQUENCES. It stays a judgment made by a person and
    passed in through n_sequences; nothing here triggers it, and
    scripts/08_interp.py records the count used and whether the run was the
    declared escalation, and refuses to escalate while the copying gate is shut.
    That refusal matters: on a model that does not copy, more sequences resolve
    a meaningless gap more sharply, so escalating there makes the output worse.

    The raw logit on the correct continuation is reported alongside, as
    clean_correct_logit, corrupted_correct_logit, correct_logit_gap and the
    per-head patched_correct_logit and recovered_correct_logit. It is the first
    term of the metric on its own, so it is free, and it is what an earlier
    reading of the protocol reported. It is secondary and no result is read from
    it, for the reason above.

    heads defaults to every head in the model. Passing the induction heads and
    their matched controls is the cheap version of the same comparison, since
    that is the pair the pre-registered expectation is about.
    """
    _ready(model, device)
    cfg = model.cfg
    n_layers, n_heads = int(cfg.n_layers), int(cfg.n_heads)
    if 2 * repeat_len > int(cfg.context_len):
        raise ValueError(
            f"two copies of {repeat_len} tokens exceed the model context "
            f"{cfg.context_len}"
        )
    if corrupt_seed is None:
        corrupt_seed = seed + 1
    if corrupt_seed == seed:
        raise ValueError(
            "corrupt_seed must differ from seed, or the corrupted run is the "
            "clean run and every patch recovers everything"
        )
    if heads is None:
        heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    heads = [(int(l), int(h)) for l, h in heads]

    clean_seqs = make_copy_sequences(n_sequences, repeat_len, int(cfg.vocab_size), seed)
    corrupt_seqs = corrupted_copy_sequences(
        clean_seqs, repeat_len, int(cfg.vocab_size), corrupt_seed
    )
    # The two tokens of the metric, both fixed by the construction and both the
    # same for all three runs. The correct continuation is read off the clean
    # sequences and the inducted-to token off the corrupted ones, which is what
    # the amendment names; the corruption leaves the second copy alone, so the
    # correct continuation is the same token in either.
    correct_tokens = correct_continuation_tokens(clean_seqs, repeat_len)
    target_tokens = induction_target_tokens(corrupt_seqs, repeat_len)
    _, scored = copy_scoring_slices(repeat_len)

    clean_values: list[np.ndarray] = []
    corrupt_values: list[np.ndarray] = []
    patched_values: dict[Head, list[np.ndarray]] = {head: [] for head in heads}

    for start in range(0, n_sequences, batch_size):
        stop = start + batch_size
        clean_x = _to_device(clean_seqs[start:stop, :-1], device)
        corrupt_x = _to_device(corrupt_seqs[start:stop, :-1], device)
        correct = _to_device(correct_tokens[start:stop], device)
        target = _to_device(target_tokens[start:stop], device)

        capture: dict = {}
        clean_logits = model(clean_x, capture=capture)
        clean_per_head = capture[ATTN_PER_HEAD_KEY]
        clean_values.append(_patching_scores(clean_logits, correct, target, scored))

        corrupt_logits = model(corrupt_x)
        corrupt_values.append(_patching_scores(corrupt_logits, correct, target, scored))

        for layer, head in heads:
            patch = {(layer, head): clean_per_head[layer][:, :, head, :]}
            with patch_heads(model, patch):
                patched_logits = model(corrupt_x)
            patched_values[(layer, head)].append(
                _patching_scores(patched_logits, correct, target, scored)
            )

    clean = np.concatenate(clean_values)
    corrupt = np.concatenate(corrupt_values)

    # The gap is the denominator every recovered_fraction divides by, and it is
    # a property of the clean and corrupted runs alone, so it is estimated once
    # here rather than per head. Its interval decides whether a fraction is a
    # quantity at all: see the comment on the gate below.
    gap = _estimate(
        np.column_stack([clean[:, 0], corrupt[:, 0]]), _gap, n_resamples, seed, alpha
    )
    gap_excludes_zero = _excludes_zero(gap)

    # The verdicts, reached once, because all of them belong to the run and none
    # to any head.
    #
    # The copying positive control comes first because it is the primary gate
    # and because it is the only one of these that asks the question that
    # matters. If the model does not copy there is no in-context copying to
    # recover, and no arithmetic on the gap can change that. The gap tests
    # follow as necessary-but-not-sufficient conditions: they can only catch a
    # denominator that is degenerate or unresolvable, which is all they were
    # ever able to do. Among them the degenerate test runs first, since where
    # both fire it is the more specific diagnosis.
    clears_chance = (
        copying_exact_match is not None
        and copying_clears_chance(copying_exact_match, int(cfg.vocab_size))
    )
    if copying_exact_match is None:
        withheld: Optional[str] = FRACTION_WITHHELD_NO_COPYING_READING
    elif not clears_chance:
        withheld = FRACTION_WITHHELD_NO_COPYING_BEHAVIOR
    elif abs(gap.point) < GAP_FLOOR:
        withheld = FRACTION_WITHHELD_DEGENERATE_GAP
    elif not gap_excludes_zero:
        withheld = FRACTION_WITHHELD_GAP_INCLUDES_ZERO
    else:
        withheld = None

    results: list[HeadPatch] = []
    for layer, head in heads:
        patched = np.concatenate(patched_values[(layer, head)])
        # Columns 0 to 2 are the metric on the patched, corrupted and clean runs;
        # columns 3 to 5 the secondary raw logit in the same order. One array, so
        # a resample draws a sequence and reads all six of its numbers, which is
        # what makes the interval on a difference a paired one.
        values = np.column_stack(
            [
                patched[:, 0],
                corrupt[:, 0],
                clean[:, 0],
                patched[:, 1],
                corrupt[:, 1],
                clean[:, 1],
            ]
        )
        # The gate, pre-registered as item 5 of the first amendment. A fraction
        # of a gap that cannot be told from zero is not a small number or an
        # uncertain one, it is a ratio of noise over noise that prints as a
        # finding: on a random-init model it returns values like 0.86 with
        # nothing whatever behind them. So it is withheld, for every head, with
        # the reason attached and the gap and its interval next to it.
        #
        # Nothing is lost. recovered_logit_diff is the same recovery
        # unnormalized, it is well defined whatever the gap does, it keeps its
        # interval, and the pre-registered induction-versus-control comparison
        # reads straight off it.
        #
        # The gate cannot bias the claim either way, since the gap depends only
        # on the clean and corrupted runs: it silences every head's fraction or
        # none, and cannot silence a control head's while sparing an induction
        # head's.
        if withheld is not None:
            recovered_fraction: Optional[Estimate] = None
            fraction_status = withheld
        else:
            recovered_fraction = _estimate(
                values, _recovered_fraction, n_resamples, seed, alpha
            )
            # The gap cleared zero, so every resample's denominator should have
            # too and the point should be finite. If it is not, something the
            # two guards above do not model has happened, and the label is that
            # no fraction was obtained, not a confident NaN.
            if not math.isfinite(recovered_fraction.point):
                recovered_fraction = None
                fraction_status = FRACTION_NOT_COMPUTED
            else:
                fraction_status = FRACTION_REPORTED
        results.append(
            HeadPatch(
                layer=layer,
                head=head,
                patched_logit_diff=float(patched[:, 0].mean()),
                recovered_logit_diff=_estimate(
                    values, _recovered_logit_diff, n_resamples, seed, alpha
                ),
                recovered_fraction=recovered_fraction,
                recovered_fraction_status=fraction_status,
                patched_correct_logit=float(patched[:, 1].mean()),
                recovered_correct_logit=_estimate(
                    values, _recovered_correct_logit, n_resamples, seed, alpha
                ),
            )
        )

    return PatchingResult(
        metric=PATCHING_METRIC,
        secondary_metric=PATCHING_SECONDARY_METRIC,
        clean_logit_diff=float(clean[:, 0].mean()),
        corrupted_logit_diff=float(corrupt[:, 0].mean()),
        gap=gap,
        gap_excludes_zero=gap_excludes_zero,
        copying_exact_match=copying_exact_match,
        copying_chance=copy_chance_accuracy(int(cfg.vocab_size)),
        copying_clears_chance=clears_chance,
        clean_correct_logit=float(clean[:, 1].mean()),
        corrupted_correct_logit=float(corrupt[:, 1].mean()),
        correct_logit_gap=float(clean[:, 1].mean() - corrupt[:, 1].mean()),
        heads=tuple(results),
        n_sequences=int(n_sequences),
        repeat_len=int(repeat_len),
        seed=int(seed),
        corrupt_seed=int(corrupt_seed),
    )


# ----------------------------------------------------------------------------
# phase change
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class PhaseChange:
    """The pre-registered phase-change interval, or why there is not one."""

    crossed: bool
    start_tokens: Optional[int]
    end_tokens: Optional[int]
    start_score: Optional[float]
    end_score: Optional[float]
    width_tokens: Optional[int]
    note: str


def phase_change_interval(
    per_checkpoint_scores: Mapping[int, float] | Sequence[tuple[int, float]],
) -> PhaseChange:
    """The token interval over which the max prefix-matching score crosses.

    The pre-registration: "the token interval over which the maximum
    prefix-matching score across heads goes from below 0.1 to above 0.3". So,
    over checkpoints ordered by token count: the end is the first checkpoint
    scoring above 0.3, and the start is the last checkpoint before it scoring
    below 0.1. The interval is the token counts of those two checkpoints, and
    its width is what says whether the emergence was abrupt or gradual.

    Both bounds are checkpoints, not interpolations, so the interval is only
    ever as tight as the checkpoint schedule. That is why the schedule is log
    spaced and dense early, and why the pre-registration commits to re-running
    with extra checkpoints if the crossing lands inside a wide gap.

    Three ways there is no interval, each reported rather than papered over:
    the max never goes above 0.3; it is already at or above 0.1 at the first
    checkpoint, so the crossing is not bracketed; or fewer than two checkpoints
    were supplied. crossed is False in each case, and note says which.
    """
    if isinstance(per_checkpoint_scores, Mapping):
        points = sorted((int(k), float(v)) for k, v in per_checkpoint_scores.items())
    else:
        points = sorted((int(k), float(v)) for k, v in per_checkpoint_scores)

    if len(points) < 2:
        return PhaseChange(
            crossed=False,
            start_tokens=None,
            end_tokens=None,
            start_score=None,
            end_score=None,
            width_tokens=None,
            note=f"{len(points)} checkpoints supplied; an interval needs at least 2",
        )

    above = [i for i, (_, score) in enumerate(points) if score > PHASE_HIGH]
    if not above:
        best = max(score for _, score in points)
        return PhaseChange(
            crossed=False,
            start_tokens=None,
            end_tokens=None,
            start_score=None,
            end_score=None,
            width_tokens=None,
            note=(
                f"the max prefix-matching score never exceeds {PHASE_HIGH}; it "
                f"peaks at {best:.4f}. No phase change to report."
            ),
        )

    end = above[0]
    below = [i for i in range(end) if points[i][1] < PHASE_LOW]
    if not below:
        return PhaseChange(
            crossed=False,
            start_tokens=None,
            end_tokens=points[end][0],
            start_score=None,
            end_score=points[end][1],
            width_tokens=None,
            note=(
                f"the max prefix-matching score is already {points[0][1]:.4f} at "
                f"the first checkpoint ({points[0][0]:,} tokens), which is not "
                f"below {PHASE_LOW}, so the crossing is not bracketed. Earlier "
                "checkpoints would be needed to bound it."
            ),
        )

    start = below[-1]
    return PhaseChange(
        crossed=True,
        start_tokens=points[start][0],
        end_tokens=points[end][0],
        start_score=points[start][1],
        end_score=points[end][1],
        width_tokens=points[end][0] - points[start][0],
        note=(
            f"the max prefix-matching score goes from {points[start][1]:.4f} at "
            f"{points[start][0]:,} tokens to {points[end][1]:.4f} at "
            f"{points[end][0]:,} tokens"
        ),
    )
