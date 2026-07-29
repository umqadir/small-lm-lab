"""When induction heads form, measured one checkpoint at a time.

docs/PROTOCOL.md, "Phase B protocol", already fixes what an induction
head is: the prefix-matching score, the 0.2 threshold, and the 0.1-to-0.3 phase
change read off a trajectory of checkpoints. interp.py implements that and this
module does not reimplement any of it. Every registered quantity here is
imported from interp or evaluate and called, never restated, for the same reason
interp.py hardcodes its own thresholds: a constant that can be redeclared is a
constant that can be redeclared to taste after seeing the distribution.

What this module adds is a cheaper instrument and a denser grid. The registered
battery in scripts/08_interp.py runs ablation and patching, which cost a forward
per head per condition, so it runs at a handful of anchors. The question "at
which token count did the head form" needs the opposite trade: no interventions
at all, one attention-weights forward per checkpoint, and every checkpoint on
the 34-point schedule. That is what measure_checkpoint is, and trajectory_summary
turns a list of those points into the fields
docs/PROTOCOL.md P1 to P7 are scored against.

The supplementary diagnostics, which that document names in advance and from
which no registered claim is read:

  token_matching, offset_plus_two   the two off-by-one controls. Same query
      positions as the registered score, key i and key i + 2 instead of key
      i + 1. A head concentrated on key i is matching the token itself rather
      than inducting from it. These exist because an indexing error in the
      registered score would produce a plausible number and nothing else in the
      battery would notice.
  prev_token, self_token, first_token, entropy
      what the rest of the heads are doing while one of them becomes an
      induction head. prev_token is the induction precursor: induction needs a
      previous-token head composing with a matching head, so a previous-token
      head appearing first is a mechanism claim with a date on it.
  ov_copying_score      the Olsson and Elhage OV-circuit statistic, read from
      the weights alone with no data at all. It says whether the head's value
      and output path copies what it attends to, which is the other half of the
      circuit the attention pattern only shows one half of.
  ov_self_logit_rank    a legible companion to the above, in units of "how many
      sampled tokens does this head boost more than the token itself".

Cost. attention_summary takes one attention-weights forward over the copying
sequences and reads all seven per-head quantities off it. Reading them in seven
passes would be seven times the cost of the expensive part, since the expensive
part is materializing [batch, n_heads, seq, seq] per layer, not reducing it. The
weights for one batch are reduced layer by layer and dropped as they are
consumed, so peak memory is one layer of one batch above the forward itself.
Accumulation is float64 on the CPU, as everywhere else in the lab, because MPS
has no float64 and a cast on the device raises.

The amendment of 2026-07-26 adds three things to this, all of them additive:

  head_churn, under "Analyses registered as planned". The count of heads
      entering and leaving the set above the registered 0.2 between consecutive
      grid checkpoints. A set that is stable once it forms and a set that
      reshuffles every checkpoint are different findings, and the maximum score
      alone cannot tell them apart.
  crossing_time_interval, under "Uncertainty on the crossing time". The
      phase-change statistic is a maximum over 64 per-head values each
      estimated from 256 sequences, and a
      maximum over noisy values is biased upward. So every point retains its
      whole [n_sequences, n_layers, n_heads] prefix-matching array, not just the
      per-head mean, and crossing_time_interval resamples sequence indices once
      per resample and recomputes the maximum at every checkpoint from those
      same sequences. That is what makes the resulting distribution a
      distribution over crossing times instead of 34 unrelated intervals, and
      it is only possible because the same sequences were drawn at every
      checkpoint. crossing_time_interval asserts that instead of assuming it.
      The upward bias is disclosed and not corrected, since a bias-corrected
      maximum is a different statistic from the one the registration named.

The one number this module introduces that is not read from the construction is
PREV_TOKEN_LEVEL. See its comment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from .bootstrap import DEFAULT_ALPHA, DEFAULT_N_RESAMPLES, bootstrap_ci
from .data import DEFAULT_TOKENIZED_ROOT, open_split
from .evaluate import (
    COPY_N_SEQUENCES,
    COPY_REPEAT_LEN,
    DEFAULT_SEED,
    DOMAINS,
    EOT_ID,
    _icl_statistic,
    _ready,
    copying_eval,
    make_copy_sequences,
)
from .interp import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_WEIGHTS_BATCH_SIZE,
    INDUCTION_THRESHOLD,
    PHASE_HIGH,
    PHASE_LOW,
    PREFIX_MATCHING_N_SEQUENCES,
    Head,
    copy_chance_accuracy,
    copying_clears_chance,
    copying_positive_control,
    identify_induction_heads,
    iter_batches,
    phase_change_interval,
    prefix_matching_indices,
    prefix_matching_intervals,
    window_losses,
)
from .model_torch import ATTN_WEIGHTS_KEY, WANT_ATTN_WEIGHTS

# The seven per-head quantities one attention forward yields, in the order they
# are documented above. Named once so a caller cannot ask for a metric that is
# spelled almost right.
ATTENTION_METRICS = (
    "prefix_matching",
    "token_matching",
    "offset_plus_two",
    "prev_token",
    "self_token",
    "first_token",
    "entropy",
)

# The level P4 of docs/PROTOCOL.md reads the previous-token
# score against. It is the same number as interp.PHASE_HIGH and it is assigned
# from it instead of typed again, on the same reasoning interp.py gives for
# PATCHING_N_SEQUENCES: the two are the same number by decision and not by
# definition, and writing the digits twice would let them drift apart silently.
#
# It is a module constant and not an argument, so no invocation can move it. No
# registered claim is read from it: the previous-token score is a supplementary
# diagnostic, declared as such in the prediction document before any checkpoint
# had been measured.
PREV_TOKEN_LEVEL = PHASE_HIGH

# Cost only, and recorded in every result that uses it. The in-context learning
# score over an unbounded split is a full pass over the corpus per checkpoint,
# which is affordable once and not 34 times. The headline in-context number
# still comes from evaluate.icl_score with no cap at all; see measure_checkpoint.
DEFAULT_ICL_MAX_WINDOWS = 512

# Cost only. How many token ids ov_self_logit_rank restricts its circuit to, and
# how many tokens of a stream are counted to find them.
DEFAULT_N_FREQ_TOKENS = 512
DEFAULT_FREQ_SAMPLE_TOKENS = 20_000_000

# Where ov_self_logit_rank's token ids came from. Carried on every point,
# because "the 512 most frequent ids in the corpus" and "512 ids drawn at
# random" are different samples and a rank read on one does not mean what the
# same rank read on the other means.
SELF_LOGIT_SOURCE_NONE = "none"
SELF_LOGIT_SOURCE_CORPUS = "corpus_frequency"
SELF_LOGIT_SOURCE_UNIFORM = "uniform_sample"
SELF_LOGIT_SOURCES = (
    SELF_LOGIT_SOURCE_NONE,
    SELF_LOGIT_SOURCE_CORPUS,
    SELF_LOGIT_SOURCE_UNIFORM,
)


# ----------------------------------------------------------------------------
# the attention summary
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class AttentionSummary:
    """Seven per-head attention readings, per sequence, from one forward.

    Every array is [n_sequences, n_layers, n_heads]. Per sequence instead of
    averaged, so a bootstrap over sequences has units to resample, which is the
    same shape and the same reason interp.prefix_matching_per_sequence keeps.

    n_query_positions says how many query positions each metric averaged over.
    The three that read a (query, key) pair do NOT share a denominator, and a
    reader comparing prefix_matching against offset_plus_two has to know that
    the second one dropped a query the first one kept. Recording the counts is
    cheaper than trusting anyone to remember.
    """

    prefix_matching: np.ndarray
    token_matching: np.ndarray
    offset_plus_two: np.ndarray
    prev_token: np.ndarray
    self_token: np.ndarray
    first_token: np.ndarray
    entropy: np.ndarray
    n_sequences: int
    repeat_len: int
    seed: int
    n_query_positions: dict[str, int]


@dataclass(frozen=True)
class _Indices:
    """The gather positions of one attention summary, already on the device."""

    prefix_q: torch.Tensor
    prefix_k: torch.Tensor
    token_q: torch.Tensor
    token_k: torch.Tensor
    offset_q: torch.Tensor
    offset_k: torch.Tensor
    rows: torch.Tensor
    log_rows: torch.Tensor


def attention_offsets(repeat_len: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """The (query, key) pairs of the registered score and its two controls.

    All three start from interp.prefix_matching_indices, so the registered pair
    is not re-derived here and the controls cannot drift off it by an index.
    Writing q = repeat_len + i for the second occurrence of the token at
    position i:

      prefix_matching   key i + 1, the token following the earlier occurrence.
                        The registered score. i runs 0 .. repeat_len - 2.
      token_matching    key i, the earlier occurrence itself. Same queries, so
                        the two are directly comparable per head.
      offset_plus_two   key i + 2, one past the registered key. i runs only
                        0 .. repeat_len - 3 here. At i = repeat_len - 2 the key
                        would be repeat_len, which is a legal causal index but
                        is the first token of the second copy instead of
                        anything two after the earlier occurrence in the first.
                        Keeping it would mix a different quantity into the
                        control at one position out of repeat_len - 1. That
                        exclusion is why the three metrics have two different
                        denominators, which is why the denominators are
                        recorded.
    """
    if repeat_len < 3:
        raise ValueError(
            f"repeat_len must be at least 3 for the offset controls, got {repeat_len}"
        )
    prefix_q, prefix_k = prefix_matching_indices(repeat_len)
    keep = repeat_len - 2
    return {
        "prefix_matching": (prefix_q, prefix_k),
        "token_matching": (prefix_q, prefix_k - 1),
        "offset_plus_two": (prefix_q[:keep], prefix_k[:keep] + 1),
    }


def prefix_matching_chance(repeat_len: int) -> float:
    """The prefix-matching score of a head that spreads attention uniformly.

    Read off the geometry and nothing else: a head attending uniformly over its
    causal prefix pays 1 / (q + 1) to every key it can see, so its score is the
    mean of that over the scored queries. About 0.005412 at the registered
    repeat length of 128, which is where the 0.2 threshold's margin comes from.
    Derived from the same prefix_matching_indices the score reads, so it cannot
    describe a different set of queries than the score does.
    """
    queries, _ = prefix_matching_indices(repeat_len)
    return float(np.mean(1.0 / (queries.numpy().astype(np.float64) + 1.0)))


def _pair_mean(w: torch.Tensor, queries: torch.Tensor, keys: torch.Tensor) -> np.ndarray:
    """[batch, n_heads]: mean weight over the (query, key) pairs, elementwise.

    The same three operations in the same order as
    interp.prefix_matching_per_sequence, deliberately: float32 gather, float32
    mean over the pairs, then the widening to float64 on the CPU. Widening
    float32 to float64 is exact, so this reproduces that function bit for bit
    and not approximately, and tests/test_emergence.py asserts it does.
    """
    return w[:, :, queries, keys].float().mean(dim=-1).cpu().double().numpy()


def _row_entropy(w: torch.Tensor, log_rows: torch.Tensor) -> np.ndarray:
    """[batch, n_heads]: mean normalized Shannon entropy of the attention rows.

    Natural log, and each row divided by log(j + 1), the entropy of a uniform
    distribution over the j + 1 keys row j is allowed to see. So 1 is a head
    spreading attention as widely as causality permits and 0 is a head paying
    everything to one key, at every row alike, which is what makes the number
    comparable across positions and across sequence lengths.

    Row 0 is skipped: it has one legal key, its entropy is 0 and its normalizer
    is log(1) = 0, so it is 0/0 and not a degenerate case needing a
    convention.

    Zero weights are handled by clamping the log's argument instead of by
    masking. A weight of exactly 0 multiplied by log(tiny) is exactly 0 in
    floating point, so this is the same number as dropping the term, without a
    mask the size of the weight matrix.
    """
    wf = w.float()
    logw = torch.log(wf.clamp_min(torch.finfo(torch.float32).tiny))
    h = -(wf * logw).sum(dim=-1)  # [batch, n_heads, seq]
    return (h[:, :, 1:] / log_rows).mean(dim=-1).cpu().double().numpy()


def _reduce_layer(w: torch.Tensor, idx: _Indices) -> dict[str, np.ndarray]:
    """Every per-head reading for one layer of one batch, as [batch, n_heads]."""
    rows = idx.rows
    return {
        "prefix_matching": _pair_mean(w, idx.prefix_q, idx.prefix_k),
        "token_matching": _pair_mean(w, idx.token_q, idx.token_k),
        "offset_plus_two": _pair_mean(w, idx.offset_q, idx.offset_k),
        "prev_token": w[:, :, rows, rows - 1].float().mean(dim=-1).cpu().double().numpy(),
        "self_token": w[:, :, rows, rows].float().mean(dim=-1).cpu().double().numpy(),
        "first_token": w[:, :, rows, 0].float().mean(dim=-1).cpu().double().numpy(),
        "entropy": _row_entropy(w, idx.log_rows),
    }


@torch.no_grad()
def attention_summary(
    model: torch.nn.Module,
    device: str | torch.device,
    n_sequences: int = PREFIX_MATCHING_N_SEQUENCES,
    repeat_len: int = COPY_REPEAT_LEN,
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_WEIGHTS_BATCH_SIZE,
) -> AttentionSummary:
    """Seven per-head attention readings from one attention-weights forward.

    The sequences come from evaluate.make_copy_sequences, the same construction
    the registered prefix-matching score, the ablation copying numbers and the
    patching runs all read, instead of a private draw that looks the same. The
    query and key positions come from attention_offsets, which is built on
    interp.prefix_matching_indices. So prefix_matching here is not a
    reimplementation of the registered score, it is the registered score read
    off a forward that also happens to yield six other things, and
    tests/test_emergence.py pins it to interp.prefix_matching_per_sequence at
    1e-12 on a real model.

    Memory. The analysis path materializes [batch, n_heads, seq, seq] per layer,
    which at size30m with 256-token copying sequences and a batch of 8 is about
    4 MB a layer and 34 MB for the model. That is per batch, and the reduction
    to [batch, n_heads] happens layer by layer with each layer's weights dropped
    as it is consumed, so nothing here grows with n_sequences. Holding the
    weights for all 256 sequences would be a gigabyte for no reason.
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
    offsets = attention_offsets(repeat_len)
    rows = torch.arange(1, seq_len)
    idx = _Indices(
        prefix_q=offsets["prefix_matching"][0].to(device),
        prefix_k=offsets["prefix_matching"][1].to(device),
        token_q=offsets["token_matching"][0].to(device),
        token_k=offsets["token_matching"][1].to(device),
        offset_q=offsets["offset_plus_two"][0].to(device),
        offset_k=offsets["offset_plus_two"][1].to(device),
        rows=rows.to(device),
        # log(j + 1) for j = 1 .. seq_len - 1, the uniform-over-causal-prefix
        # entropy each row is normalized against.
        log_rows=torch.log(rows.to(torch.float32) + 1.0).to(device),
    )

    chunks: dict[str, list[np.ndarray]] = {name: [] for name in ATTENTION_METRICS}
    for chunk in iter_batches(seqs, batch_size):
        x = torch.from_numpy(np.ascontiguousarray(chunk)).to(device)
        capture: dict = {WANT_ATTN_WEIGHTS: True}
        model(x, capture=capture)
        weights = capture[ATTN_WEIGHTS_KEY]
        per_layer: dict[str, list[np.ndarray]] = {name: [] for name in ATTENTION_METRICS}
        for layer in sorted(weights):
            # pop, not read: the dict is the only remaining reference once the
            # forward's frame is gone, so this is what frees a layer's weights
            # before the next layer's reductions allocate their temporaries.
            w = weights.pop(layer)
            for name, value in _reduce_layer(w, idx).items():
                per_layer[name].append(value)
            del w
        del weights, capture
        for name in ATTENTION_METRICS:
            chunks[name].append(np.stack(per_layer[name], axis=1))

    if not chunks[ATTENTION_METRICS[0]]:
        raise ValueError("no sequences to score")

    stacked = {
        name: np.concatenate(chunks[name], axis=0).astype(np.float64)
        for name in ATTENTION_METRICS
    }
    return AttentionSummary(
        n_sequences=int(n_sequences),
        repeat_len=int(repeat_len),
        seed=int(seed),
        n_query_positions={
            "prefix_matching": int(idx.prefix_q.numel()),
            "token_matching": int(idx.token_q.numel()),
            "offset_plus_two": int(idx.offset_q.numel()),
            "prev_token": int(seq_len - 1),
            "self_token": int(seq_len - 1),
            "first_token": int(seq_len - 1),
            "entropy": int(seq_len - 1),
        },
        **stacked,
    )


# ----------------------------------------------------------------------------
# the OV circuit, from weights alone
# ----------------------------------------------------------------------------

def _ov_parts(model: torch.nn.Module) -> tuple[np.ndarray, np.ndarray]:
    """(E, w_final) in float64 on the CPU.

    float64 because eigenvalues of a product of five matrices are not something
    to read in float32, and CPU because MPS has no float64 at all.
    """
    embed = model.embed.weight.detach().to(device="cpu", dtype=torch.float64).numpy()
    final = model.final_norm.weight.detach().to(device="cpu", dtype=torch.float64).numpy()
    return embed, final


def _head_slices(
    block: torch.nn.Module, head: int, head_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """(P, Q) for one head: [d, head_dim] and [head_dim, d], in float64.

    The layouts are this repo's, checked against model_torch.py instead of
    assumed. v_proj is Linear(d_model, attn_dim) so its weight is
    [n_heads * head_dim, d_model] and head h owns rows
    [h * head_dim, (h + 1) * head_dim). o_proj is Linear(attn_dim, d_model) so
    its weight is [d_model, n_heads * head_dim] and head h owns the matching
    COLUMNS. Both are transposed here because the circuit below is written for
    row vectors, which is the direction a token embedding actually travels.
    """
    lo, hi = head * head_dim, (head + 1) * head_dim
    v = block.attn.v_proj.weight.detach().to(device="cpu", dtype=torch.float64).numpy()
    o = block.attn.o_proj.weight.detach().to(device="cpu", dtype=torch.float64).numpy()
    return v[lo:hi, :].T, o[:, lo:hi].T


@torch.no_grad()
def ov_copying_score(model: torch.nn.Module) -> np.ndarray:
    """[n_layers, n_heads]: the OV-circuit copying score, from weights only.

    The statistic is Olsson and Elhage's: the sum of the real parts of the full
    OV circuit's eigenvalues over the sum of their MAGNITUDES. It is +1 when the
    circuit maps every token to a positive multiple of itself, which is copying,
    and -1 when it maps every token away from itself. No data is involved, so
    this reads the head's mechanism instead of its behavior on any particular
    input.

    The circuit, as a map from a source token id to logits, with the two
    diagonal RMSNorm weights folded in:

        M = E diag(w_attn_l) P_lh Q_lh diag(w_final) E^T

    where E is the embedding [V, d] (the lm head is tied to it), w_attn_l is the
    layer's pre-attention RMSNorm weight, w_final the final RMSNorm weight,
    P_lh = v_proj rows for the head, transposed, [d, head_dim], and
    Q_lh = o_proj columns for the head, transposed, [head_dim, d].

    Two things are dropped and both are safe, for different reasons.

    The input-dependent RMS scalar. RMSNorm is x -> (x / rms(x)) * w, and only
    the diagonal w is kept. The discarded factor is a positive scalar, so it
    turns M into c M for some c > 0. This score is a ratio of two quantities
    that are both homogeneous of degree 1 in M, hence homogeneous of degree 0
    overall: c M has eigenvalues c times M's, both sums scale by c, and the
    ratio is identical. So dropping it changes nothing about this statistic. It
    would change a raw logit, which is why nothing here reports one.

    The V - head_dim zero eigenvalues. M is 16384 x 16384 at this vocabulary and
    is never built. It factors as X Y with X = E diag(w_attn) P [V, head_dim]
    and Y = Q diag(w_final) E^T [head_dim, V], so its rank is at most head_dim,
    which is 64. The nonzero eigenvalues of X Y are exactly the eigenvalues of
    Y X, which is head_dim by head_dim, and the remaining V - head_dim
    eigenvalues of M are 0. A zero eigenvalue contributes 0 to the numerator and
    0 to the denominator. So the score computed on the 64 by 64 reduction is
    exactly the score on the 16384 by 16384 matrix, not an approximation of it,
    and tests/test_emergence.py builds the full matrix on a small config and
    asserts the two agree.

    Y X = Q diag(w_final) E^T E diag(w_attn) P, so the only large product is
    E^T E, which is [d, d], does not depend on the layer or the head, and is
    computed once here and reused across all of them.

    A head whose eigenvalue magnitudes sum to exactly 0 gets NaN, not 0.0. The
    score is undefined there, since the ratio has no denominator, and a dead
    head reported as 0.0 would read as a head that is precisely balanced between
    copying and anti-copying, which is a different and much stronger claim than
    "this head does nothing at all".
    """
    cfg = model.cfg
    n_layers, n_heads = int(cfg.n_layers), int(cfg.n_heads)
    head_dim = int(cfg.head_dim)

    embed, w_final = _ov_parts(model)
    gram = embed.T @ embed  # E^T E, [d, d], layer and head independent
    final_gram = w_final[:, None] * gram  # diag(w_final) E^T E

    out = np.empty((n_layers, n_heads), dtype=np.float64)
    for layer, block in enumerate(model.blocks):
        w_attn = (
            block.attn_norm.weight.detach()
            .to(device="cpu", dtype=torch.float64)
            .numpy()
        )
        core = final_gram * w_attn[None, :]  # diag(w_final) E^T E diag(w_attn)
        for head in range(n_heads):
            p, q = _head_slices(block, head, head_dim)
            reduced = q @ core @ p  # [head_dim, head_dim]
            eigenvalues = np.linalg.eigvals(reduced)
            magnitude = float(np.abs(eigenvalues).sum())
            out[layer, head] = (
                float("nan")
                if magnitude == 0.0
                else float(eigenvalues.real.sum()) / magnitude
            )
    return out


@torch.no_grad()
def ov_self_logit_rank(
    model: torch.nn.Module, token_ids: Sequence[int] | np.ndarray
) -> np.ndarray:
    """[n_layers, n_heads]: median rank of a token's own logit in its own OV row.

    The same circuit as ov_copying_score, restricted on both sides to the token
    ids handed in, which the caller draws (the most frequent ids in the corpus,
    where the corpus is available). For each id, its row of that restricted
    matrix is ranked and the position of the diagonal entry is recorded: 0 means
    the head boosts the token itself above every other sampled token, and n - 1
    means it boosts it below all of them. The median over ids is reported.

    Secondary, and legible in a way the eigenvalue ratio is not: "this head
    ranks a token's own logit first among 512 frequent tokens" is a sentence,
    where "its OV eigenvalues have real parts summing to 0.7 of their
    magnitudes" is a statistic. No registered claim is read from either.

    The restriction is what keeps this cheap: nothing larger than
    [len(ids), len(ids)] is ever built, where the unrestricted circuit is
    16384 x 16384. It also changes the quantity, and deliberately so. A rank
    against 512 frequent tokens is a rank against 512 frequent tokens, not
    against the vocabulary, and the two are not interchangeable.

    Ties do not count against the diagonal: the rank is the number of entries
    strictly greater. A head with a constant row therefore scores 0 instead of
    len(ids) - 1, which is the reading that does not turn an uninformative head
    into a confidently anti-copying one.
    """
    ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    if ids.size == 0:
        raise ValueError("ov_self_logit_rank needs at least one token id")
    cfg = model.cfg
    vocab_size = int(cfg.vocab_size)
    if ids.min() < 0 or ids.max() >= vocab_size:
        raise ValueError(
            f"token ids must lie in [0, {vocab_size}), got "
            f"[{int(ids.min())}, {int(ids.max())}]"
        )

    n_layers, n_heads = int(cfg.n_layers), int(cfg.n_heads)
    head_dim = int(cfg.head_dim)
    embed, w_final = _ov_parts(model)
    rows = embed[ids, :]  # [n_ids, d]
    right_base = (rows * w_final[None, :]).T  # diag(w_final) E_ids^T, [d, n_ids]

    out = np.empty((n_layers, n_heads), dtype=np.float64)
    for layer, block in enumerate(model.blocks):
        w_attn = (
            block.attn_norm.weight.detach()
            .to(device="cpu", dtype=torch.float64)
            .numpy()
        )
        left_base = rows * w_attn[None, :]  # E_ids diag(w_attn), [n_ids, d]
        for head in range(n_heads):
            p, q = _head_slices(block, head, head_dim)
            restricted = (left_base @ p) @ (q @ right_base)  # [n_ids, n_ids]
            diagonal = np.diag(restricted)
            ranks = (restricted > diagonal[:, None]).sum(axis=1)
            out[layer, head] = float(np.median(ranks))
    return out


def frequent_token_ids(
    n: int,
    root: Path | str = DEFAULT_TOKENIZED_ROOT,
    domains: Sequence[str] = DOMAINS,
    split: str = "val",
    max_tokens: int = DEFAULT_FREQ_SAMPLE_TOKENS,
    vocab_size: Optional[int] = None,
) -> np.ndarray:
    """The n most frequent token ids over a bounded prefix of the named streams.

    For ov_self_logit_rank, which needs a sample of ids and is more informative
    on ids the model has actually seen many times than on a uniform draw over a
    vocabulary whose tail is nearly unused. Counting is capped at max_tokens per
    domain because the ordering of the head of the distribution is settled long
    before a stream is exhausted, and the whole point of this module is that it
    runs 34 times.

    The eot id is excluded, matching the copying task's own draw, since it is a
    document boundary instead of a word and it would otherwise dominate.
    Raises when a stream is missing, so a caller can fall back explicitly rather
    than silently ranking against whatever happened to be countable.
    """
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    if not domains:
        raise ValueError("no domains to count over")

    counts: Optional[np.ndarray] = None
    for domain in domains:
        tokens = open_split(domain, split, root)
        prefix = np.asarray(tokens[: int(max_tokens)], dtype=np.int64)
        if prefix.size == 0:
            raise ValueError(f"stream for domain {domain!r} split {split!r} is empty")
        length = vocab_size if vocab_size is not None else int(prefix.max()) + 1
        counted = np.bincount(prefix, minlength=int(length))
        if counts is None:
            counts = counted
        elif counted.size != counts.size:
            size = max(counted.size, counts.size)
            counts = np.pad(counts, (0, size - counts.size)) + np.pad(
                counted, (0, size - counted.size)
            )
        else:
            counts = counts + counted
    assert counts is not None

    counts = counts.astype(np.int64)
    counts[EOT_ID] = -1
    order = np.argsort(-counts, kind="stable")
    present = int((counts > 0).sum())
    return order[: min(int(n), present)].astype(np.int64)


# ----------------------------------------------------------------------------
# one grid point
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class HeadScores:
    """Every per-head reading of one checkpoint. All arrays [n_layers, n_heads].

    prefix_matching is the registered score and carries the only interval here,
    since it is the only one of these a registered claim is read from. The rest
    are point estimates: they are supplementary diagnostics, and a bootstrap per
    head per metric per checkpoint would multiply the cost of the dense grid by
    the number of metrics for numbers nothing is decided on.
    """

    prefix_matching: np.ndarray
    prefix_matching_ci_low: np.ndarray
    prefix_matching_ci_high: np.ndarray
    token_matching: np.ndarray
    offset_plus_two: np.ndarray
    prev_token: np.ndarray
    self_token: np.ndarray
    first_token: np.ndarray
    entropy: np.ndarray
    ov_copying_score: np.ndarray
    ov_self_logit_rank: np.ndarray


@dataclass(frozen=True)
class CopyingPoint:
    """evaluate.copying_eval's two numbers, with the chance level beside them.

    chance comes from interp.copy_chance_accuracy, which reads it off the
    construction of the task instead of choosing it, and clears_chance is
    interp.copying_clears_chance's verdict on the interval. Both are carried so
    that a reader can check the verdict against the numbers it was reached from
    instead of trusting it, and so that P6 of the prediction document is a
    field lookup instead of an arithmetic exercise.
    """

    exact_match: float
    exact_match_ci_low: float
    exact_match_ci_high: float
    loss_delta: float
    loss_delta_ci_low: float
    loss_delta_ci_high: float
    mean_first_copy_loss: float
    mean_second_copy_loss: float
    chance: float
    clears_chance: bool
    n_sequences: int
    repeat_len: int


@dataclass(frozen=True)
class IclPoint:
    """The in-context learning score on one domain, on the dense grid.

    icl_score is evaluate._icl_statistic over interp.window_losses, bootstrapped
    the same way evaluate.icl_score bootstraps it, so this is that statistic and
    not a near miss. What differs is max_windows, a cost cap, and the two fields
    that record it: n_windows is what was actually read and icl_windows_capped
    says whether the cap bit. With max_windows None the two functions compute
    the same number on the same windows.

    icl_windows_capped is true when a cap was set and the count reached it. A
    split holding exactly max_windows windows would also report true, which
    overstates and does not understate the truncation, and is the direction an
    ambiguous flag should err in.
    """

    domain: str
    split: str
    icl_score: float
    ci_low: float
    ci_high: float
    n_windows: int
    max_windows: Optional[int]
    icl_windows_capped: bool
    mean_early_loss: float
    mean_late_loss: float


@dataclass(frozen=True)
class BootstrapProvenance:
    """What every interval on this point was computed with.

    sequence_seed and resample_seed hold the same value, because one seed drives
    the copying draws and every resample, which is what scripts/08_interp.py
    does too. They are two fields because they are two different jobs and a
    later run that wants to vary one without the other should not have to guess
    which of them a single number meant.
    """

    n_resamples: int
    alpha: float
    sequence_seed: int
    resample_seed: int


@dataclass(frozen=True)
class EmergencePoint:
    """One checkpoint on the emergence trajectory.

    tokens is the token count the checkpoint recorded and is None for a
    random-init control, which sits on no point of the token axis at all rather
    than at zero: a model that has seen no tokens has also taken no optimizer
    steps and is not the beginning of the trajectory, it is the negative
    control the trajectory is read against.
    """

    tokens: Optional[int]
    checkpoint: Optional[str]
    heads: HeadScores
    # The raw [n_sequences, n_layers, n_heads] prefix-matching array, retained by
    # the 2026-07-26 amendment, "Uncertainty on the crossing time", so that the
    # crossing time can carry an interval. heads.prefix_matching is this
    # averaged over its first axis, and
    # nothing else in this record can be recovered from the mean alone: a
    # bootstrap that has to resample the same sequences at every checkpoint needs
    # the per-sequence values at every checkpoint. 131 KB at the registered 256
    # sequences and an 8 by 8 head grid, so it is too large for the result JSON
    # and is written to an .npz sidecar by scripts/19_emergence.py. Optional
    # only so that a synthetic point built for a trajectory-level test does not
    # have to carry one.
    prefix_matching_per_sequence: Optional[np.ndarray]
    max_prefix_matching: float
    max_prefix_matching_head: Head
    induction_heads: tuple[Head, ...]
    induction_threshold: float
    max_prev_token: float
    max_prev_token_head: Head
    prefix_matching_chance: float
    copying: CopyingPoint
    icl: dict[str, IclPoint]
    n_sequences: int
    repeat_len: int
    n_query_positions: dict[str, int]
    n_self_logit_tokens: int
    self_logit_token_source: str
    bootstrap: BootstrapProvenance


def _argmax_head(scores: np.ndarray) -> Head:
    layer, head = np.unravel_index(int(np.argmax(scores)), scores.shape)
    return (int(layer), int(head))


def measure_checkpoint(
    model: torch.nn.Module,
    device: str | torch.device,
    tokens: Optional[int] = None,
    checkpoint: Optional[Path | str] = None,
    n_sequences: int = PREFIX_MATCHING_N_SEQUENCES,
    repeat_len: int = COPY_REPEAT_LEN,
    seed: int = DEFAULT_SEED,
    weights_batch_size: int = DEFAULT_WEIGHTS_BATCH_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    copy_n_sequences: int = COPY_N_SEQUENCES,
    domains: Sequence[str] = DOMAINS,
    split: str = "val",
    root: Path | str = DEFAULT_TOKENIZED_ROOT,
    max_windows: Optional[int] = DEFAULT_ICL_MAX_WINDOWS,
    self_logit_token_ids: Optional[Sequence[int] | np.ndarray] = None,
    self_logit_token_source: str = SELF_LOGIT_SOURCE_NONE,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
) -> EmergencePoint:
    """Everything one grid point of the emergence trajectory carries.

    Three measurements, in increasing order of cost: the OV circuit, which reads
    weights only and touches no data; the attention summary and the copying
    battery, which run on the synthetic copying sequences and need no corpus;
    and the in-context learning score, which needs the tokenized streams
    mounted. Passing an empty `domains` skips the third, which is what makes the
    first two runnable on a machine that holds the checkpoints and not the
    corpus.

    The in-context number here is the dense-grid instrument, not the headline.
    evaluate.icl_score over the whole split is the pre-registered reading and it
    is what scripts/06_evaluate.py reports. This one is the same statistic on
    the same windows in the same order, capped at max_windows for cost, with the
    cap and the realized window count recorded on every IclPoint so a capped
    reading can never be mistaken for the uncapped one.

    induction_heads comes from interp.identify_induction_heads with no threshold
    argument, so it is the registered 0.2 and there is no second copy of that
    number anywhere in this module.
    """
    summary = attention_summary(
        model,
        device,
        n_sequences=n_sequences,
        repeat_len=repeat_len,
        seed=seed,
        batch_size=weights_batch_size,
    )
    scores = summary.prefix_matching.mean(axis=0)
    ci_low, ci_high = prefix_matching_intervals(
        summary.prefix_matching, n_resamples=n_resamples, seed=seed, alpha=alpha
    )

    prev_token = summary.prev_token.mean(axis=0)
    if self_logit_token_ids is None:
        self_rank = np.full_like(scores, np.nan)
        n_self_tokens = 0
        source = SELF_LOGIT_SOURCE_NONE
    else:
        ids = np.asarray(self_logit_token_ids, dtype=np.int64).reshape(-1)
        self_rank = ov_self_logit_rank(model, ids)
        n_self_tokens = int(ids.size)
        source = self_logit_token_source

    heads = HeadScores(
        prefix_matching=scores,
        prefix_matching_ci_low=ci_low,
        prefix_matching_ci_high=ci_high,
        token_matching=summary.token_matching.mean(axis=0),
        offset_plus_two=summary.offset_plus_two.mean(axis=0),
        prev_token=prev_token,
        self_token=summary.self_token.mean(axis=0),
        first_token=summary.first_token.mean(axis=0),
        entropy=summary.entropy.mean(axis=0),
        ov_copying_score=ov_copying_score(model),
        ov_self_logit_rank=self_rank,
    )

    copying = copying_eval(
        model,
        device,
        n_sequences=copy_n_sequences,
        repeat_len=repeat_len,
        seed=seed,
        batch_size=batch_size,
        n_resamples=n_resamples,
        alpha=alpha,
    )
    vocab_size = int(model.cfg.vocab_size)
    copying_point = CopyingPoint(
        exact_match=copying.exact_match,
        exact_match_ci_low=copying.exact_match_ci_low,
        exact_match_ci_high=copying.exact_match_ci_high,
        loss_delta=copying.loss_delta,
        loss_delta_ci_low=copying.loss_delta_ci_low,
        loss_delta_ci_high=copying.loss_delta_ci_high,
        mean_first_copy_loss=copying.mean_first_copy_loss,
        mean_second_copy_loss=copying.mean_second_copy_loss,
        chance=copy_chance_accuracy(vocab_size),
        clears_chance=copying_clears_chance(
            copying_positive_control(copying), vocab_size
        ),
        n_sequences=copying.n_sequences,
        repeat_len=copying.repeat_len,
    )

    icl: dict[str, IclPoint] = {}
    for domain in domains:
        windows = window_losses(
            model,
            domain,
            split,
            device,
            batch_size=batch_size,
            root=root,
            max_windows=max_windows,
        )
        # Columns 2 and 3 are the late and early window means, which is
        # what evaluate.icl_score bootstraps, so this is that statistic.
        values = windows[:, 2:4]
        point, lo, hi = bootstrap_ci(
            values, _icl_statistic, n_resamples=n_resamples, seed=seed, alpha=alpha
        )
        n_windows = int(values.shape[0])
        icl[domain] = IclPoint(
            domain=domain,
            split=split,
            icl_score=point,
            ci_low=lo,
            ci_high=hi,
            n_windows=n_windows,
            max_windows=max_windows,
            icl_windows_capped=bool(
                max_windows is not None and n_windows >= int(max_windows)
            ),
            mean_early_loss=float(values[:, 1].mean()),
            mean_late_loss=float(values[:, 0].mean()),
        )

    return EmergencePoint(
        tokens=None if tokens is None else int(tokens),
        checkpoint=None if checkpoint is None else str(checkpoint),
        heads=heads,
        prefix_matching_per_sequence=summary.prefix_matching,
        max_prefix_matching=float(scores.max()),
        max_prefix_matching_head=_argmax_head(scores),
        induction_heads=tuple(identify_induction_heads(scores)),
        induction_threshold=float(INDUCTION_THRESHOLD),
        max_prev_token=float(prev_token.max()),
        max_prev_token_head=_argmax_head(prev_token),
        prefix_matching_chance=prefix_matching_chance(repeat_len),
        copying=copying_point,
        icl=icl,
        n_sequences=int(n_sequences),
        repeat_len=int(repeat_len),
        n_query_positions=dict(summary.n_query_positions),
        n_self_logit_tokens=n_self_tokens,
        self_logit_token_source=source,
        bootstrap=BootstrapProvenance(
            n_resamples=int(n_resamples),
            alpha=float(alpha),
            sequence_seed=int(seed),
            resample_seed=int(seed),
        ),
    )


# ----------------------------------------------------------------------------
# the trajectory
# ----------------------------------------------------------------------------

def _on_the_token_axis(points: Sequence[EmergencePoint]) -> list[EmergencePoint]:
    """The points that carry a token count, oldest first.

    A random-init control carries none and is dropped here instead of sorted to
    the front, because it is not the first point of the trajectory. Everything
    below reads a token axis and a point with no place on it cannot be read.
    """
    placed = [p for p in points if p.tokens is not None]
    return sorted(placed, key=lambda p: int(p.tokens))


def _steepest_icl_step(
    points: Sequence[EmergencePoint], domain: str
) -> Optional[dict[str, Any]]:
    """The grid step over which the in-context score fell furthest.

    delta is the later score minus the earlier one, so the step wanted is the
    most negative delta and a NON-negative delta reported here means the score
    never fell anywhere on the grid. Reported with its sign rather than
    suppressed, since "the steepest step was an increase" is a result about the
    trajectory and hiding it behind a None would lose it.
    """
    scored = [p for p in points if domain in p.icl]
    if len(scored) < 2:
        return None
    best: Optional[dict[str, Any]] = None
    for earlier, later in zip(scored, scored[1:]):
        delta = later.icl[domain].icl_score - earlier.icl[domain].icl_score
        if best is None or delta < best["delta"]:
            best = {
                "from_tokens": int(earlier.tokens),
                "to_tokens": int(later.tokens),
                "from_score": float(earlier.icl[domain].icl_score),
                "to_score": float(later.icl[domain].icl_score),
                "delta": float(delta),
            }
    return best


def per_sequence_key(point: EmergencePoint) -> str:
    """The .npz key one point's retained prefix-matching array is stored under.

    Zero padded to twelve digits and prefixed the same way train.py names its
    checkpoint files, so a key in the sidecar and a filename in the checkpoint
    directory are the same string and a reader does not have to map between two
    conventions. A point with no token count is the random-init control, which
    sits on no point of the token axis and gets a name rather than a number.
    """
    if point.tokens is None:
        return "random_init"
    return f"tokens_{int(point.tokens):012d}"


@dataclass(frozen=True)
class CrossingTime:
    """The crossing time with a bootstrap interval.

    Registered by the 2026-07-26 amendment, "Uncertainty on the crossing time".

    point_tokens is what the existing rule gives, which is
    interp.phase_change_interval's end_tokens: the first grid checkpoint whose
    observed maximum prefix-matching score exceeds 0.3. It is carried because
    the amendment asks for the interval alongside the point estimate, not
    instead of it, and because a median over resamples and a value read off the
    observed data are two different numbers that should not be confused.

    median_tokens, ci_low_tokens and ci_high_tokens are grid checkpoints and
    never interpolations, for the same reason interp.phase_change_interval's
    bounds are: the crossing time is the token count of a checkpoint that was
    actually measured, and a percentile that landed between two of them would
    name a checkpoint that does not exist. So the percentiles are taken by
    nearest rank on the checkpoint's position in the grid.

    no_crossing_share is the share of resamples in which no checkpoint's
    resampled maximum ever exceeded 0.3. Those resamples are ranked, not
    dropped: a resample that never crossed is a crossing time later than every
    checkpoint measured, so it sorts above all of them. Dropping them would
    condition the interval on having crossed and make it narrower the less
    evidence there was. Ranking them is also what makes the reporting rule below
    exact rather than arbitrary: the median is finite if and only if fewer than
    half the resamples failed to cross, which is why that is the cut.

    reported is False, with median and both bounds None and note saying why,
    when no_crossing_share is above 0.5. A bound that lands on a resample that
    never crossed is also reported as None, with the note naming which one, so
    a one-sided interval is visible as one rather than silently truncated to the
    last checkpoint on the grid.
    """

    reported: bool
    point_tokens: Optional[int]
    median_tokens: Optional[int]
    ci_low_tokens: Optional[int]
    ci_high_tokens: Optional[int]
    no_crossing_share: float
    level: float
    n_resamples: int
    alpha: float
    seed: int
    n_sequences: int
    n_checkpoints: int
    note: str


def _withheld_crossing(note: str, n_resamples: int, alpha: float, seed: int) -> CrossingTime:
    return CrossingTime(
        reported=False,
        point_tokens=None,
        median_tokens=None,
        ci_low_tokens=None,
        ci_high_tokens=None,
        no_crossing_share=float("nan"),
        level=PHASE_HIGH,
        n_resamples=int(n_resamples),
        alpha=float(alpha),
        seed=int(seed),
        n_sequences=0,
        n_checkpoints=0,
        note=note,
    )


def _check_shared_sequences(points: Sequence[EmergencePoint]) -> np.ndarray:
    """Stack the retained per-sequence arrays, refusing an unshared draw.

    The whole construction rests on one thing: resample index 7 has to mean the
    same copying sequence at every checkpoint. It does when every checkpoint
    called evaluate.make_copy_sequences with the same count, the same repeat
    length and the same seed, which is what a single run of
    scripts/19_emergence.py does, and it does not when two points came from two
    invocations with different seeds. Nothing in the arrays themselves reveals
    the difference, so it is checked against the provenance every point already
    carries rather than assumed. Mixing two draws would not error anywhere
    downstream; it would just quietly pair unrelated sequences and produce an
    interval that is too wide by an amount nobody could estimate.

    Returns [n_checkpoints, n_sequences, n_layers, n_heads].
    """
    first = points[0]
    for point in points:
        if point.prefix_matching_per_sequence is None:
            raise ValueError(
                f"the point at {point.tokens} tokens retained no per-sequence "
                "prefix-matching array, so the crossing time cannot be "
                "resampled over shared sequences"
            )
        if (
            point.n_sequences != first.n_sequences
            or point.repeat_len != first.repeat_len
            or point.bootstrap.sequence_seed != first.bootstrap.sequence_seed
        ):
            raise ValueError(
                "every checkpoint must have been measured on the same copying "
                "sequences for the crossing-time bootstrap to pair them: the "
                f"point at {first.tokens} tokens used n_sequences "
                f"{first.n_sequences}, repeat_len {first.repeat_len}, seed "
                f"{first.bootstrap.sequence_seed} and the point at "
                f"{point.tokens} tokens used {point.n_sequences}, "
                f"{point.repeat_len}, {point.bootstrap.sequence_seed}"
            )
        if (
            point.prefix_matching_per_sequence.shape
            != first.prefix_matching_per_sequence.shape
        ):
            raise ValueError(
                "the retained prefix-matching arrays differ in shape: "
                f"{first.prefix_matching_per_sequence.shape} at {first.tokens} "
                f"tokens and {point.prefix_matching_per_sequence.shape} at "
                f"{point.tokens} tokens"
            )
    return np.stack([p.prefix_matching_per_sequence for p in points], axis=0)


def crossing_time_interval(
    points: Sequence[EmergencePoint],
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> CrossingTime:
    """A bootstrap interval on the token count at which the crossing happened.

    Amendment of 2026-07-26, "Uncertainty on the crossing time". Within each
    resample the sequence indices are drawn once, the per-head means and their
    maximum are recomputed at every
    checkpoint from those same sequences, and the first checkpoint whose
    resampled maximum exceeds interp.PHASE_HIGH is recorded. The reported
    crossing time is the median of that distribution with a percentile interval.

    Drawing the indices once per resample rather than once per checkpoint is the
    whole design. The quantity of interest is a property of the trajectory, so a
    resample has to perturb the trajectory, not each of its points
    independently. Resampling per checkpoint would let sampling noise move
    checkpoints in opposite directions within one resample, which would produce
    a crossing-time distribution wider than the evidence and, worse, would
    describe a trajectory no draw of sequences could ever have produced.

    Nothing here is a new threshold. The level crossed is interp.PHASE_HIGH, the
    same 0.3 interp.phase_change_interval reads, imported and not restated. The
    0.5 share above which the interval is withheld is not a threshold on any
    measured quantity: it is the exact point at which a median over the ranked
    distribution stops being finite, so it is arithmetic rather than a choice.
    """
    placed = _on_the_token_axis(points)
    if len(placed) < 2:
        return _withheld_crossing(
            f"{len(placed)} checkpoints carry a token count; a crossing time "
            "needs at least 2",
            n_resamples,
            alpha,
            seed,
        )
    if any(p.prefix_matching_per_sequence is None for p in placed):
        return _withheld_crossing(
            "at least one checkpoint retained no per-sequence prefix-matching "
            "array, so sequence indices cannot be shared across checkpoints",
            n_resamples,
            alpha,
            seed,
        )

    stacked = _check_shared_sequences(placed)
    n_checkpoints, n_sequences = stacked.shape[0], stacked.shape[1]
    token_axis = [int(p.tokens) for p in placed]
    observed = phase_change_interval(
        {int(p.tokens): p.max_prefix_matching for p in placed}
    )

    # A resample that never crossed is ranked above every checkpoint rather than
    # dropped, so this sentinel is one past the last grid position.
    never = n_checkpoints
    rng = np.random.default_rng(seed)
    crossings = np.empty(int(n_resamples), dtype=np.int64)
    for r in range(int(n_resamples)):
        idx = rng.integers(0, n_sequences, size=n_sequences)
        maxima = stacked[:, idx].mean(axis=1).reshape(n_checkpoints, -1).max(axis=1)
        above = np.nonzero(maxima > PHASE_HIGH)[0]
        crossings[r] = int(above[0]) if above.size else never

    no_crossing_share = float((crossings == never).mean())
    # Nearest rank, because the statistic is a position in the grid and a
    # percentile that interpolated between two positions would name a checkpoint
    # that was never measured.
    median = int(np.percentile(crossings, 50.0, method="nearest"))
    low = int(np.percentile(crossings, 100.0 * (alpha / 2.0), method="nearest"))
    high = int(np.percentile(crossings, 100.0 * (1.0 - alpha / 2.0), method="nearest"))

    def at(position: int) -> Optional[int]:
        return None if position >= never else token_axis[position]

    if no_crossing_share > 0.5:
        withheld = _withheld_crossing(
            f"{no_crossing_share:.3f} of resamples never exceeded {PHASE_HIGH} at "
            "any checkpoint on the grid, so more than half the distribution lies "
            "beyond the last checkpoint measured and its median is not a token "
            "count. Reported as a share rather than by dropping those resamples, "
            "which would condition the interval on having crossed.",
            n_resamples,
            alpha,
            seed,
        )
        return CrossingTime(
            reported=False,
            point_tokens=observed.end_tokens,
            median_tokens=None,
            ci_low_tokens=None,
            ci_high_tokens=None,
            no_crossing_share=no_crossing_share,
            level=PHASE_HIGH,
            n_resamples=int(n_resamples),
            alpha=float(alpha),
            seed=int(seed),
            n_sequences=int(n_sequences),
            n_checkpoints=int(n_checkpoints),
            note=withheld.note,
        )

    unbounded = [
        name
        for name, position in (("ci_high", high), ("ci_low", low))
        if position >= never
    ]
    note = (
        f"median over {n_resamples} resamples of the first grid checkpoint whose "
        f"resampled maximum prefix-matching score exceeds {PHASE_HIGH}, with the "
        "same sequence indices applied at every checkpoint within a resample. "
        "The maximum over heads is biased upward and this interval does not "
        "correct it."
    )
    if unbounded:
        note += (
            " " + " and ".join(sorted(unbounded)) + " is null because that "
            "percentile lands on a resample that never crossed, so the interval "
            "is one sided rather than closed at the last checkpoint."
        )
    return CrossingTime(
        reported=True,
        point_tokens=observed.end_tokens,
        median_tokens=at(median),
        ci_low_tokens=at(low),
        ci_high_tokens=at(high),
        no_crossing_share=no_crossing_share,
        level=PHASE_HIGH,
        n_resamples=int(n_resamples),
        alpha=float(alpha),
        seed=int(seed),
        n_sequences=int(n_sequences),
        n_checkpoints=int(n_checkpoints),
        note=note,
    )


def head_churn(points: Sequence[EmergencePoint]) -> list[dict[str, Any]]:
    """Heads entering and leaving the registered induction set, step by step.

    Amendment of 2026-07-26, "Analyses registered as planned". The set is the one
    interp.identify_induction_heads named at each checkpoint, so the 0.2 is the
    registered one and is applied once, at measurement time, rather than again
    here.

    Registered because the maximum score cannot distinguish a set that forms and
    then holds from one that reshuffles at every checkpoint, and those are
    different claims about what formed. A head that crosses the threshold and
    stays is a head that was built; a set whose membership turns over every step
    is a threshold cutting through a cloud of heads sitting near 0.2, which is a
    result about the threshold rather than about the model.

    Both directions are reported, and the heads themselves and not only the
    counts, since which head left is the question a count immediately raises.
    """
    placed = _on_the_token_axis(points)
    churn: list[dict[str, Any]] = []
    for earlier, later in zip(placed, placed[1:]):
        before = {tuple(int(i) for i in h) for h in earlier.induction_heads}
        after = {tuple(int(i) for i in h) for h in later.induction_heads}
        entered = sorted(after - before)
        left = sorted(before - after)
        churn.append(
            {
                "from_tokens": int(earlier.tokens),
                "to_tokens": int(later.tokens),
                "n_before": len(before),
                "n_after": len(after),
                "n_entered": len(entered),
                "n_left": len(left),
                "entered": [list(h) for h in entered],
                "left": [list(h) for h in left],
            }
        )
    return churn


def trajectory_summary(
    points: Sequence[EmergencePoint],
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """The trajectory-level readings, from a list of grid points.

    The fields docs/PROTOCOL.md scores P1 to P7 against,
    plus the two the 2026-07-26 amendment registers:

      phase_change                the registered interval. Delegated whole to
          interp.phase_change_interval, which owns the 0.1 and the 0.3 and the
          three ways there is no interval. Nothing is recomputed here; the
          mapping from token count to max prefix-matching score is handed over
          and the answer is carried through. n_grid_points_between counts the
          checkpoints strictly inside the interval, which is what P3's "at most
          3 grid steps" is read from, and it is a count rather than a threshold.
      steepest_icl_step           per domain, for P5.
      first_copying_above_chance  for P6. The verdict per point is
          interp.copying_clears_chance's, taken at measurement time, so this is
          a search over verdicts rather than a second comparison against chance.
      first_prev_token_above      for P4, against PREV_TOKEN_LEVEL.
      crossing_time               an interval on the crossing
          time itself. n_resamples, seed and alpha reach only this: every other
          reading here is a lookup over point estimates already computed.
      head_churn                  heads entering and leaving the registered set.

    Points with no token count, meaning random-init controls, are excluded from
    every one of these, since all of them read a token axis.
    """
    placed = _on_the_token_axis(points)
    n_unplaced = len(points) - len(placed)

    if len(placed) < 2:
        phase: dict[str, Any] = {
            "ran": False,
            "note": (
                f"{len(placed)} checkpoints carry a token count; a phase-change "
                "interval needs at least 2. Points with no token count, which "
                "are random-init controls, sit on no token axis and are excluded."
            ),
            "trajectory": [
                {"tokens": int(p.tokens), "max_score": p.max_prefix_matching}
                for p in placed
            ],
            "interval": None,
            "n_grid_points_between": None,
        }
    else:
        interval = phase_change_interval(
            {int(p.tokens): p.max_prefix_matching for p in placed}
        )
        if interval.crossed:
            between = sum(
                1
                for p in placed
                if interval.start_tokens < int(p.tokens) < interval.end_tokens
            )
        else:
            between = None
        phase = {
            "ran": True,
            "note": "",
            "trajectory": [
                {"tokens": int(p.tokens), "max_score": p.max_prefix_matching}
                for p in placed
            ],
            "interval": interval,
            "n_grid_points_between": between,
        }

    domains = sorted({domain for p in placed for domain in p.icl})
    steepest = {domain: _steepest_icl_step(placed, domain) for domain in domains}

    first_copying = next(
        (int(p.tokens) for p in placed if p.copying.clears_chance), None
    )
    first_prev = next(
        (int(p.tokens) for p in placed if p.max_prev_token > PREV_TOKEN_LEVEL), None
    )

    return {
        "n_points": len(points),
        "n_points_on_token_axis": len(placed),
        "n_points_without_tokens": n_unplaced,
        "phase_change": phase,
        "crossing_time": crossing_time_interval(
            placed, n_resamples=n_resamples, seed=seed, alpha=alpha
        ),
        "head_churn": head_churn(placed),
        "steepest_icl_step": steepest,
        "first_copying_above_chance": first_copying,
        "first_prev_token_above": first_prev,
        "prev_token_level": PREV_TOKEN_LEVEL,
        "induction_threshold": INDUCTION_THRESHOLD,
        "phase_bounds": [PHASE_LOW, PHASE_HIGH],
    }
