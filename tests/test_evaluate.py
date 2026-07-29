"""Correctness tests for the evaluation battery, on CPU.

The tests here are built around stub models whose exact loss is known in
closed form, so each measurement is checked against an independently computed
number rather than against itself. A uniform model must score perplexity equal
to the vocabulary size; an oracle that has perfectly solved copying must score
exact match 1.0; a model that ignores position must score an in-context
learning score of 0. Those anchors are what would catch a real bug, so they are
asserted tightly.

Nothing here touches the test split, the GPU, or the training lock. The token
streams are tiny synthetic files written into a tmp directory, so the tests do
not depend on the external volume being mounted.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from small_lm_lab import evaluate
from small_lm_lab.bootstrap import bootstrap_ci
from small_lm_lab.config import ModelConfig
from small_lm_lab.evaluate import (
    BLIMP_PARADIGMS,
    EOT_ID,
    blimp_accuracy,
    copying_eval,
    icl_score,
    sum_sentence_logprobs,
)

# Imported under another name on purpose: pytest would collect anything called
# test_* in this module's namespace and try to run the measurement itself.
compute_perplexity = evaluate.test_perplexity

DEVICE = "cpu"
VOCAB = 16384
CONTEXT = 512
# Small enough to stay fast, large enough that the bootstrap has something to do.
FAST_RESAMPLES = 100


def tiny_cfg(vocab_size: int = VOCAB, context_len: int = CONTEXT) -> ModelConfig:
    """A config carrying the real vocabulary and context. The stub models below
    ignore the width and depth, so those stay at their smallest legal values."""
    return ModelConfig(
        vocab_size=vocab_size,
        d_model=64,
        n_layers=1,
        n_heads=1,
        context_len=context_len,
    )


class UniformModel(nn.Module):
    """Constant equal logits, so every token costs exactly log(vocab_size)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, t = tokens.shape
        return torch.zeros(b, t, self.cfg.vocab_size, dtype=torch.float32,
                           device=tokens.device)


class FavorTokenModel(nn.Module):
    """One token gets a fixed logit boost, always, regardless of context or
    position. Every per-token log probability is then one of two known values."""

    def __init__(self, cfg: ModelConfig, token: int, boost: float) -> None:
        super().__init__()
        self.cfg = cfg
        self.token = token
        self.boost = boost

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, t = tokens.shape
        logits = torch.zeros(b, t, self.cfg.vocab_size, dtype=torch.float32,
                             device=tokens.device)
        logits[:, :, self.token] = self.boost
        return logits

    def logprobs(self) -> tuple[float, float]:
        """(favored, other) per-token log probability, computed by hand."""
        z = math.log(math.exp(self.boost) + self.cfg.vocab_size - 1)
        return self.boost - z, -z


class CopyOracleModel(nn.Module):
    """Solves copying perfectly, using only the tokens to its left.

    At prediction index i it must produce the token at sequence position i + 1.
    Once i + 1 is strictly inside the second copy, that token equals the one at
    position i + 1 - repeat_len, which sits in the input at that same index. So
    the oracle is a legal causal computation and not a model peeking at its own
    targets.

    It stays uniform at index repeat_len - 1, which is the prediction of the
    first token of the second copy. Nothing in the sequence marks where the
    repeat begins, so no real model can have that one, and an oracle that got it
    anyway would hide an off-by-one in the scored range.
    """

    def __init__(self, cfg: ModelConfig, repeat_len: int, boost: float = 30.0) -> None:
        super().__init__()
        self.cfg = cfg
        self.repeat_len = repeat_len
        self.boost = boost

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, t = tokens.shape
        r = self.repeat_len
        logits = torch.zeros(b, t, self.cfg.vocab_size, dtype=torch.float32,
                             device=tokens.device)
        if t > r:
            # Index i takes its answer from input index i + 1 - r, for i >= r.
            src = tokens[:, 1 : t - r + 1]
            logits[:, r:, :].scatter_(2, src.unsqueeze(-1), self.boost)
        return logits


def write_stream(root: Path, domain: str, split: str, tokens: np.ndarray) -> None:
    root.mkdir(parents=True, exist_ok=True)
    tokens.astype(np.uint16).tofile(root / f"{domain}_{split}.bin")


def random_stream(n_windows: int, seed: int = 0, low: int = 10) -> np.ndarray:
    """A stream that tiles into exactly n_windows windows of context tokens.

    Ids start at `low` so specific small ids can be reserved by a test.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(low, VOCAB, size=n_windows * CONTEXT + 1, dtype=np.int64)


# Bootstrap.


def test_bootstrap_recovers_known_mean_and_brackets_it() -> None:
    true_mean = 5.0
    rng = np.random.default_rng(0)
    sample = rng.normal(loc=true_mean, scale=2.0, size=2000)

    point, lo, hi = bootstrap_ci(sample, lambda v: float(v.mean()), seed=0)

    # The point estimate is the statistic on the observed data, exactly.
    assert point == pytest.approx(float(sample.mean()), rel=1e-12)
    assert lo < true_mean < hi
    # The interval should be close to the normal-theory one, not wildly wide.
    half_width = 1.96 * sample.std(ddof=1) / math.sqrt(len(sample))
    assert (hi - lo) == pytest.approx(2 * half_width, rel=0.15)


def test_bootstrap_is_deterministic_given_seed() -> None:
    values = np.random.default_rng(1).normal(size=200)
    a = bootstrap_ci(values, lambda v: float(v.mean()), n_resamples=200, seed=7)
    b = bootstrap_ci(values, lambda v: float(v.mean()), n_resamples=200, seed=7)
    c = bootstrap_ci(values, lambda v: float(v.mean()), n_resamples=200, seed=8)
    assert a == b
    assert a[1:] != c[1:]


def test_bootstrap_resamples_rows_of_a_2d_array() -> None:
    """Rows must stay intact under resampling: a statistic reading two columns
    of the same row must never see them paired across different rows."""
    values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    seen: list[bool] = []

    def statistic(v: np.ndarray) -> float:
        seen.append(bool(np.all(v[:, 1] == v[:, 0] * 10.0)))
        return float(v[:, 0].mean())

    bootstrap_ci(values, statistic, n_resamples=50, seed=0)
    assert all(seen)


def test_bootstrap_rejects_empty_and_bad_alpha() -> None:
    with pytest.raises(ValueError):
        bootstrap_ci(np.array([]), lambda v: 0.0)
    with pytest.raises(ValueError):
        bootstrap_ci(np.array([1.0, 2.0]), lambda v: 0.0, alpha=1.5)


# Perplexity.


def test_uniform_model_perplexity_equals_vocab_size(tmp_path: Path) -> None:
    """The key anchor. A model with equal logits assigns every token
    probability 1/vocab_size, so the token-level perplexity is exactly the
    vocabulary size. Any error in the NLL, the averaging, or the exp shows up
    here as a number that is not 16384."""
    write_stream(tmp_path, "tinystories", "val", random_stream(6, seed=0))
    cfg = tiny_cfg()
    model = UniformModel(cfg)

    result = compute_perplexity(
        model, "tinystories", "val", DEVICE, batch_size=2, root=tmp_path,
        n_resamples=FAST_RESAMPLES, seed=0,
    )

    assert result.perplexity == pytest.approx(float(VOCAB), rel=1e-4)
    assert result.n_windows == 6
    assert result.n_tokens == 6 * CONTEXT
    # Every window scores identically, so the bootstrap cannot move the number.
    assert result.ci_low == pytest.approx(float(VOCAB), rel=1e-4)
    assert result.ci_high == pytest.approx(float(VOCAB), rel=1e-4)


def test_perplexity_weights_windows_by_token_count() -> None:
    """Perplexity is exp of the mean token NLL, not the mean of per-window
    perplexities. The two differ whenever windows carry different token counts,
    and this pins the implementation to the token-weighted definition."""
    values = np.array([[1.0, 100.0], [3.0, 300.0]])

    got = evaluate._weighted_perplexity(values)

    weighted_mean_nll = (1.0 * 100 + 3.0 * 300) / 400.0  # 2.5
    assert got == pytest.approx(math.exp(weighted_mean_nll))
    # The unweighted alternative would be exp(2.0). It must not be that.
    assert got != pytest.approx(math.exp(2.0))


def test_perplexity_batch_size_does_not_change_the_number(tmp_path: Path) -> None:
    write_stream(tmp_path, "tinystories", "val", random_stream(5, seed=3))
    model = FavorTokenModel(tiny_cfg(), token=11, boost=3.0)
    kwargs = dict(root=tmp_path, n_resamples=10, seed=0)

    a = compute_perplexity(model, "tinystories", "val", DEVICE, batch_size=1, **kwargs)
    b = compute_perplexity(model, "tinystories", "val", DEVICE, batch_size=4, **kwargs)

    assert a.perplexity == pytest.approx(b.perplexity, rel=1e-6)
    assert a.n_windows == b.n_windows == 5


# BLiMP.


def test_blimp_refuses_unknown_paradigm() -> None:
    model = UniformModel(tiny_cfg())
    with pytest.raises(ValueError, match="unknown BLiMP paradigm"):
        blimp_accuracy(model, "not_a_real_paradigm", DEVICE)


def test_preregistered_constants_are_what_the_document_says() -> None:
    """The pre-registration is the contract, so the numbers it fixes are pinned
    here as literals, transcribed from the registration. Nothing else in
    this file may read these constants to build its own expectations, or a
    change to a definition would silently move the tests with it."""
    assert BLIMP_PARADIGMS == (
        "anaphor_number_agreement",
        "anaphor_gender_agreement",
        "determiner_noun_agreement_1",
        "determiner_noun_agreement_2",
        "regular_plural_subject_verb_agreement_1",
        "irregular_plural_subject_verb_agreement_1",
        "irregular_past_participle_verbs",
        "animate_subject_trans",
        "npi_present_1",
        "wh_questions_object_gap",
    )
    assert len(set(BLIMP_PARADIGMS)) == 10
    # Section 3: positions 450 to 500 minus positions 50 to 100.
    assert evaluate.ICL_EARLY == (50, 100)
    assert evaluate.ICL_LATE == (450, 500)
    # Section 4: 512 sequences, length 128 per repeat.
    assert evaluate.COPY_N_SEQUENCES == 512
    assert evaluate.COPY_REPEAT_LEN == 128
    # Section 2 and the loader: the separator is id 0 and the two domains are
    # reported separately.
    assert EOT_ID == 0
    assert evaluate.DOMAINS == ("tinystories", "fineweb_edu")


def test_sentence_scoring_is_a_sum_not_a_mean() -> None:
    """Scores must be summed over the sentence, so length is penalized.

    Constructed so the two disagree: sentence A has the better per-token log
    probability but the worse total. A mean would pick A; the pre-registration
    says sum, so the implementation must pick B.
    """
    cfg = tiny_cfg()
    favored = 123
    model = FavorTokenModel(cfg, token=favored, boost=5.0)
    lp_favored, lp_other = model.logprobs()

    a = [EOT_ID] + [favored] * 10
    b = [EOT_ID] + [7, 9]
    scores = sum_sentence_logprobs(model, [a, b], DEVICE)

    # Each score equals the hand-computed sum over the real tokens.
    assert scores[0] == pytest.approx(10 * lp_favored, rel=1e-5)
    assert scores[1] == pytest.approx(2 * lp_other, rel=1e-5)
    # A wins on the mean, B wins on the sum, and the code follows the sum.
    assert lp_favored > lp_other
    assert 10 * lp_favored < 2 * lp_other
    assert scores[0] < scores[1]


def test_sentence_scoring_masks_padding() -> None:
    """A short sentence batched next to a long one must score the same as it
    does alone, which is only true if the padding is masked out of the sum."""
    cfg = tiny_cfg()
    model = FavorTokenModel(cfg, token=123, boost=5.0)
    short = [EOT_ID, 7, 9]
    long = [EOT_ID] + [123] * 20

    together = sum_sentence_logprobs(model, [short, long], DEVICE, batch_size=8)
    alone = sum_sentence_logprobs(model, [short], DEVICE, batch_size=8)

    assert together[0] == pytest.approx(alone[0], rel=1e-6)


def test_blimp_counts_the_item_correct_when_good_beats_bad(monkeypatch) -> None:
    """Direction check. The stub is context independent, so every extra token
    only subtracts from a sum: the strictly shorter sentence always scores
    higher. Pairs are built with the short sentence as the good one, so a
    correct implementation scores 1.0 and a reversed comparison scores 0.0.
    """
    cfg = tiny_cfg()
    model = FavorTokenModel(cfg, token=123, boost=5.0)
    short = ["The cat sat.", "Susan revealed herself.", "A boy ran home."]
    long = [s + " Then many additional ordinary words followed after it." for s in short]

    pairs = list(zip(short, long))
    monkeypatch.setattr(
        evaluate, "_load_blimp_pairs", lambda paradigm, hf_home=None: pairs
    )
    good = blimp_accuracy(
        model, "npi_present_1", DEVICE, n_resamples=FAST_RESAMPLES, seed=0
    )
    assert good.accuracy == 1.0
    assert good.n_items == 3

    swapped = list(zip(long, short))
    monkeypatch.setattr(
        evaluate, "_load_blimp_pairs", lambda paradigm, hf_home=None: swapped
    )
    bad = blimp_accuracy(
        model, "npi_present_1", DEVICE, n_resamples=FAST_RESAMPLES, seed=0
    )
    assert bad.accuracy == 0.0


def test_blimp_prepends_the_eot_bos_and_scores_every_real_token(monkeypatch) -> None:
    """The <|endoftext|> id has to be prepended before scoring.

    Without it the first real token of every sentence becomes context instead of
    a prediction, so it silently drops out of the sum and the first token of a
    sentence is never scored at all. This captures the sequences that reach the
    scorer and checks them against the tokenizer directly.
    """
    model = UniformModel(tiny_cfg())
    sentences = ["The cat sat.", "Susan revealed herself."]
    pairs = [(sentences[0], sentences[1])]
    captured: list[list[int]] = []

    def capture(model_, token_lists, device, batch_size=32):
        captured.extend([list(t) for t in token_lists])
        return np.zeros(len(token_lists))

    monkeypatch.setattr(
        evaluate, "_load_blimp_pairs", lambda paradigm, hf_home=None: pairs
    )
    monkeypatch.setattr(evaluate, "sum_sentence_logprobs", capture)
    blimp_accuracy(model, "npi_present_1", DEVICE, n_resamples=10, seed=0)

    tokenizer = evaluate.load_tokenizer(str(evaluate.TOKENIZER_PATH))
    assert len(captured) == 2
    for sentence, tokens in zip(sentences, captured):
        expected = tokenizer.encode(sentence).ids
        assert tokens[0] == EOT_ID, "the BOS must lead every scored sequence"
        assert tokens == [EOT_ID] + expected
        # Every real token is scored: prediction indices 0 .. len(tokens) - 2.
        assert len(tokens) - 1 == len(expected)


def test_blimp_uniform_model_scores_chance_on_length_matched_pairs(monkeypatch) -> None:
    """A uniform model scores every sentence of equal token length identically,
    so no item can be counted correct under a strict comparison."""
    model = UniformModel(tiny_cfg())
    pairs = [("the cat", "the dog"), ("a boy", "a girl")]
    monkeypatch.setattr(
        evaluate, "_load_blimp_pairs", lambda paradigm, hf_home=None: pairs
    )

    result = blimp_accuracy(
        model, "anaphor_number_agreement", DEVICE, n_resamples=10, seed=0
    )
    assert result.accuracy == 0.0


# In-context learning score.


def test_icl_is_zero_for_a_position_independent_model(tmp_path: Path) -> None:
    """A model whose output does not depend on position pays the same loss late
    as early, so the score is exactly 0."""
    write_stream(tmp_path, "fineweb_edu", "val", random_stream(6, seed=1))
    model = UniformModel(tiny_cfg())

    result = icl_score(
        model, "fineweb_edu", "val", DEVICE, batch_size=2, root=tmp_path,
        n_resamples=FAST_RESAMPLES, seed=0,
    )

    assert result.icl_score == pytest.approx(0.0, abs=1e-6)
    assert result.mean_early_loss == pytest.approx(math.log(VOCAB), rel=1e-4)
    assert result.mean_late_loss == pytest.approx(math.log(VOCAB), rel=1e-4)
    assert result.n_windows == 6


def test_icl_is_negative_when_late_positions_are_the_predictable_ones(
    tmp_path: Path,
) -> None:
    """Sign check, and a check that the scored positions are the pre-registered
    ones. The stream is built so the tokens the late window scores are always
    the one token the stub predicts confidently, while the early window scores
    random tokens. Context use must come out negative.
    """
    special = 7
    stream = random_stream(6, seed=2, low=10)  # ids never collide with `special`
    offsets = np.arange(len(stream)) % CONTEXT
    # Literal on purpose: prediction index i scores stream position i + 1 within
    # its window, so the pre-registered late window of 450 to 500 scores offsets
    # 451 to 500 inside each 512 block. Reading ICL_LATE here instead would move
    # this fixture along with any change to the definition and assert nothing.
    stream[(offsets >= 451) & (offsets <= 500)] = special
    write_stream(tmp_path, "tinystories", "val", stream)

    # A boost of 10 keeps every quantity here in the range float32 resolves
    # well. A much larger boost would drive the favored token's loss to ~1e-5,
    # which is the difference of two numbers near the boost itself and so is
    # mostly rounding error at float32, making the assertion below a test of
    # the arithmetic rather than of the measurement.
    model = FavorTokenModel(tiny_cfg(), token=special, boost=10.0)
    result = icl_score(
        model, "tinystories", "val", DEVICE, batch_size=2, root=tmp_path,
        n_resamples=FAST_RESAMPLES, seed=0,
    )

    lp_favored, lp_other = model.logprobs()
    assert result.mean_late_loss == pytest.approx(-lp_favored, rel=1e-3)
    assert result.mean_early_loss == pytest.approx(-lp_other, rel=1e-3)
    assert result.icl_score == pytest.approx(lp_other - lp_favored, rel=1e-3)
    assert result.icl_score < -5.0
    assert result.ci_high < 0.0


# Copying.


def test_copying_oracle_scores_perfect_exact_match() -> None:
    """A model that has perfectly solved copying must score exact match 1.0 and
    a strongly negative loss delta."""
    repeat_len = 16
    cfg = tiny_cfg()
    model = CopyOracleModel(cfg, repeat_len=repeat_len)

    result = copying_eval(
        model, DEVICE, n_sequences=32, repeat_len=repeat_len, seed=0,
        batch_size=8, n_resamples=FAST_RESAMPLES,
    )

    assert result.exact_match == 1.0
    assert result.exact_match_ci_low == 1.0
    assert result.exact_match_ci_high == 1.0
    # The first copy is unpredictable, the second is free.
    assert result.mean_first_copy_loss == pytest.approx(math.log(VOCAB), rel=1e-3)
    assert result.mean_second_copy_loss == pytest.approx(0.0, abs=1e-3)
    assert result.loss_delta == pytest.approx(-math.log(VOCAB), rel=1e-3)
    assert result.loss_delta_ci_high < 0.0


def test_copying_uniform_model_scores_chance() -> None:
    """A model that ignores the earlier copy pays the same loss on both copies
    and never gets the second copy right."""
    repeat_len = 16
    model = UniformModel(tiny_cfg())

    result = copying_eval(
        model, DEVICE, n_sequences=32, repeat_len=repeat_len, seed=0,
        batch_size=8, n_resamples=FAST_RESAMPLES,
    )

    assert result.exact_match < 0.01
    assert result.loss_delta == pytest.approx(0.0, abs=1e-5)
    assert result.mean_first_copy_loss == pytest.approx(math.log(VOCAB), rel=1e-4)
    assert result.mean_second_copy_loss == pytest.approx(math.log(VOCAB), rel=1e-4)


def test_copying_is_deterministic_given_seed() -> None:
    model = UniformModel(tiny_cfg())
    kwargs = dict(n_sequences=16, repeat_len=8, batch_size=4, n_resamples=10)

    a = copying_eval(model, DEVICE, seed=3, **kwargs)
    b = copying_eval(model, DEVICE, seed=3, **kwargs)
    c = copying_eval(model, DEVICE, seed=4, **kwargs)

    assert a == b
    assert a.mean_first_copy_loss == pytest.approx(c.mean_first_copy_loss, abs=1e-6)


def test_copying_sequences_repeat_and_exclude_the_eot_id() -> None:
    """The sequences the eval actually builds, not a restatement of them.

    The eot id has to stay out: it is the document separator, and a model that
    has learned to predict it at boundaries would score copying partly on that
    rather than on the induction mechanism the task is meant to isolate.
    """
    # A two token vocabulary leaves exactly one legal id once the eot is
    # excluded, so this is decisive rather than probabilistic: at the real vocab
    # size a draw that wrongly included id 0 would produce one only about 3
    # percent of the time and the check would pass by luck.
    narrow = evaluate.make_copy_sequences(
        n_sequences=64, repeat_len=8, vocab_size=2, seed=3
    )
    assert set(np.unique(narrow).tolist()) == {1}

    seqs = evaluate.make_copy_sequences(
        n_sequences=64, repeat_len=8, vocab_size=VOCAB, seed=3
    )

    assert seqs.shape == (64, 16)
    assert seqs.min() > EOT_ID
    assert seqs.max() < VOCAB
    # The second copy is the first, exactly.
    np.testing.assert_array_equal(seqs[:, :8], seqs[:, 8:])
    # Deterministic given the seed, and the seed actually does something.
    other = evaluate.make_copy_sequences(64, 8, VOCAB, seed=3)
    np.testing.assert_array_equal(seqs, other)
    assert not np.array_equal(seqs, evaluate.make_copy_sequences(64, 8, VOCAB, seed=4))


def test_copying_scores_equal_counts_on_both_copies() -> None:
    """Both copies must contribute the same number of positions, or the delta
    would partly measure how many tokens each side was scored on. This reads the
    slices the eval actually uses, not a local restatement of them."""
    repeat_len = evaluate.COPY_REPEAT_LEN
    first, second = evaluate.copy_scoring_slices(repeat_len)
    n = 2 * repeat_len - 1  # length of the prediction axis

    first_idx = range(*first.indices(n))
    second_idx = range(*second.indices(n))

    assert len(first_idx) == len(second_idx) == repeat_len - 1
    # The unpredictable first token of each copy is excluded from both.
    assert 0 not in [i + 1 for i in first_idx]
    assert repeat_len not in [i + 1 for i in second_idx]
    # The two ranges must not overlap.
    assert set(first_idx).isdisjoint(second_idx)


def test_copying_rejects_a_repeat_longer_than_the_context() -> None:
    model = UniformModel(tiny_cfg(context_len=64))
    with pytest.raises(ValueError, match="exceed the model context"):
        copying_eval(model, DEVICE, n_sequences=2, repeat_len=64, n_resamples=10)
