"""The four pre-registered measurements, and nothing else.

docs/PROTOCOL.md fixes what is measured and how. This module implements
exactly that list: held-out perplexity per domain, the ten named BLiMP
paradigms, the in-context learning score, and synthetic copying. No metric here
is optional, none was added afterwards, and the definitions are not
parameterized in ways that would let them be tuned after seeing results. The
paradigm list is hardcoded and unknown names are refused, because the point of
naming the ten in advance is that the set cannot drift toward whatever the
models happen to do well on.

Everything is PyTorch. Training may end up in MLX, but evaluation and
interpretability run in PyTorch either way, converting checkpoints if MLX wins
the throughput pilot. See convert.py.

Uncertainty is the shared percentile bootstrap in bootstrap.py: 1000 resamples,
percentile interval, resampling the unit the pre-registration names (windows for
perplexity and for the in-context score, items for BLiMP, sequences for
copying).

Losses are read out in float32 regardless of the dtype the weights are stored
in, since checkpoints are saved in bf16 and a bf16 cross entropy would put
visible noise into a headline number.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .bootstrap import DEFAULT_ALPHA, DEFAULT_N_RESAMPLES, bootstrap_ci
from .data import DEFAULT_TOKENIZED_ROOT, iter_eval_split
from .paths import HF_CACHE

REPO_ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_PATH = REPO_ROOT / "data" / "tokenizer" / "tokenizer.json"
DEFAULT_HF_HOME = HF_CACHE

# The tokenizer's only special token. Doubles as the BOS for BLiMP scoring and
# as the id excluded from the copying task's random draws.
EOT_ID = 0

DEFAULT_SEED = 1

# The two domains, reported separately and never pooled.
DOMAINS = ("tinystories", "fineweb_edu")

# Pre-registration section 2, in the order listed there. Hardcoded on purpose.
BLIMP_PARADIGMS = (
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

# Pre-registration section 3: mean loss at positions 450 to 500 minus mean loss
# at positions 50 to 100. Half-open, 50 scored positions each.
ICL_EARLY = (50, 100)
ICL_LATE = (450, 500)

# Pre-registration section 4.
COPY_N_SEQUENCES = 512
COPY_REPEAT_LEN = 128

DEFAULT_BATCH_SIZE = 16
DEFAULT_BLIMP_BATCH_SIZE = 32


@dataclass(frozen=True)
class PerplexityResult:
    domain: str
    split: str
    perplexity: float
    ci_low: float
    ci_high: float
    n_windows: int
    n_tokens: int


@dataclass(frozen=True)
class BlimpResult:
    paradigm: str
    accuracy: float
    ci_low: float
    ci_high: float
    n_items: int


@dataclass(frozen=True)
class IclResult:
    domain: str
    split: str
    icl_score: float
    ci_low: float
    ci_high: float
    n_windows: int
    mean_early_loss: float
    mean_late_loss: float


@dataclass(frozen=True)
class CopyingResult:
    loss_delta: float
    loss_delta_ci_low: float
    loss_delta_ci_high: float
    exact_match: float
    exact_match_ci_low: float
    exact_match_ci_high: float
    n_sequences: int
    repeat_len: int
    seed: int
    mean_first_copy_loss: float
    mean_second_copy_loss: float


def _ready(model: torch.nn.Module, device: str | torch.device) -> torch.nn.Module:
    """Put the model on the device in eval mode. Idempotent."""
    model.to(device)
    model.eval()
    return model


def _forward_nll(
    model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (logits, per-token NLL) for one batch. NLL is [batch, seq]."""
    logits = model(x)
    b, t, v = logits.shape
    nll = F.cross_entropy(
        logits.reshape(b * t, v).float(), y.reshape(b * t), reduction="none"
    )
    return logits, nll.view(b, t)


def _weighted_perplexity(values: np.ndarray) -> float:
    """exp of the token-weighted mean NLL. Column 0 is a per-window mean NLL,
    column 1 that window's token count.

    Weighting by token count is what makes this a token-level number rather than
    an average of window-level numbers. The two coincide only when every window
    holds the same token count, which is true here because the loader drops the
    final partial window, but the weighting is written out anyway so a resample
    of windows can never silently become an unweighted average.
    """
    mean_nll = values[:, 0]
    counts = values[:, 1]
    total_tokens = counts.sum()
    if total_tokens <= 0:
        raise ValueError("no tokens to score")
    return float(np.exp((mean_nll * counts).sum() / total_tokens))


@torch.no_grad()
def test_perplexity(
    model: torch.nn.Module,
    domain: str,
    split: str,
    device: str | torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
    root: Path | str = DEFAULT_TOKENIZED_ROOT,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> PerplexityResult:
    """Token-level perplexity over exhaustive non-overlapping windows.

    One domain at a time, never pooled across domains. Windows tile the stream
    exactly once, so no token is scored twice and none is weighted twice.
    Perplexity is exp of the mean token NLL over the whole split, not the mean
    of per-window perplexities. Bootstrap resamples windows.
    """
    _ready(model, device)
    context_len = int(model.cfg.context_len)

    means: list[np.ndarray] = []
    counts: list[np.ndarray] = []
    for x_np, y_np in iter_eval_split(domain, split, context_len, batch_size, root):
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
        _, nll = _forward_nll(model, x, y)
        # cpu() before double(), here and everywhere: MPS has no float64, so a
        # cast on the device raises. Widening float32 to float64 is exact, so
        # doing it after the on-device float32 reduction changes nothing.
        means.append(nll.mean(dim=1).cpu().double().numpy())
        counts.append(np.full(nll.shape[0], nll.shape[1], dtype=np.float64))

    if not means:
        raise ValueError(f"no windows for domain {domain!r} split {split!r}")

    values = np.stack([np.concatenate(means), np.concatenate(counts)], axis=1)
    point, lo, hi = bootstrap_ci(
        values, _weighted_perplexity, n_resamples=n_resamples, seed=seed, alpha=alpha
    )
    return PerplexityResult(
        domain=domain,
        split=split,
        perplexity=point,
        ci_low=lo,
        ci_high=hi,
        n_windows=int(values.shape[0]),
        n_tokens=int(values[:, 1].sum()),
    )


@lru_cache(maxsize=4)
def load_tokenizer(path: str = str(TOKENIZER_PATH)):
    """Load the lab tokenizer once per path and cache it.

    Cached because BLiMP scores 2000 sentences per paradigm across ten
    paradigms, and rebuilding the tokenizer per item would dominate the run.
    """
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))


def _load_blimp_pairs(
    paradigm: str, hf_home: Path | str = DEFAULT_HF_HOME
) -> list[tuple[str, str]]:
    """Return [(good, bad)] for one paradigm from the local HF cache."""
    os.environ.setdefault("HF_HOME", str(hf_home))
    from datasets import load_dataset

    ds = load_dataset("nyu-mll/blimp", paradigm, split="train")
    return list(zip(ds["sentence_good"], ds["sentence_bad"]))


@torch.no_grad()
def sum_sentence_logprobs(
    model: torch.nn.Module,
    token_lists: Sequence[Sequence[int]],
    device: str | torch.device,
    batch_size: int = DEFAULT_BLIMP_BATCH_SIZE,
) -> np.ndarray:
    """Total log probability of each token sequence, summed over its real tokens.

    Each list is expected to already carry its BOS at index 0. The token at
    index 0 is context and is not scored; every token after it is. The score is
    a SUM over the sentence, not a mean, so a longer sentence is penalized for
    its length. That is the pre-registered choice.

    Sequences are padded to the longest in each batch and the padding is masked
    out of the sum, so batching cannot change a score.
    """
    _ready(model, device)
    context_len = int(model.cfg.context_len)
    n = len(token_lists)
    out = np.empty(n, dtype=np.float64)

    for start in range(0, n, batch_size):
        chunk = [list(t) for t in token_lists[start : start + batch_size]]
        lengths = [len(t) for t in chunk]
        if min(lengths) < 2:
            raise ValueError("every sequence needs a BOS plus at least one real token")
        max_len = max(lengths)
        if max_len > context_len:
            raise ValueError(
                f"sequence of {max_len} tokens exceeds the model context {context_len}"
            )

        padded = np.zeros((len(chunk), max_len), dtype=np.int64)
        for i, toks in enumerate(chunk):
            padded[i, : len(toks)] = toks
        arr = torch.from_numpy(padded).to(device)

        _, nll = _forward_nll(model, arr[:, :-1], arr[:, 1:])
        # Prediction index i scores token i+1, so sequence of length L has real
        # targets at indices 0 .. L-2.
        pos = torch.arange(max_len - 1, device=nll.device)[None, :]
        lens = torch.tensor(lengths, device=nll.device)[:, None]
        mask = (pos < (lens - 1)).to(nll.dtype)
        totals = -(nll * mask).sum(dim=1)
        out[start : start + len(chunk)] = totals.cpu().double().numpy()

    return out


def blimp_accuracy(
    model: torch.nn.Module,
    paradigm: str,
    device: str | torch.device,
    batch_size: int = DEFAULT_BLIMP_BATCH_SIZE,
    hf_home: Path | str = DEFAULT_HF_HOME,
    tokenizer_path: Path | str = TOKENIZER_PATH,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> BlimpResult:
    """Accuracy on one pre-registered BLiMP paradigm, with a bootstrap over items.

    Each minimal pair is scored by summing token log probabilities over the full
    sentence, with the <|endoftext|> id prepended as a BOS so the first real
    token is predicted from something rather than from nothing. The item counts
    as correct when the grammatical sentence scores strictly higher. Chance is
    0.50.
    """
    if paradigm not in BLIMP_PARADIGMS:
        raise ValueError(
            f"unknown BLiMP paradigm {paradigm!r}. The pre-registered set is fixed: "
            + ", ".join(BLIMP_PARADIGMS)
        )

    pairs = _load_blimp_pairs(paradigm, hf_home)
    if not pairs:
        raise ValueError(f"paradigm {paradigm!r} loaded zero items")

    tokenizer = load_tokenizer(str(tokenizer_path))
    sentences = [p[0] for p in pairs] + [p[1] for p in pairs]
    # One batched call, and the tokenizer itself is cached across paradigms.
    encoded = tokenizer.encode_batch(sentences)
    token_lists = [[EOT_ID] + e.ids for e in encoded]

    scores = sum_sentence_logprobs(model, token_lists, device, batch_size)
    n = len(pairs)
    correct = (scores[:n] > scores[n:]).astype(np.float64)

    point, lo, hi = bootstrap_ci(
        correct,
        lambda v: float(v.mean()),
        n_resamples=n_resamples,
        seed=seed,
        alpha=alpha,
    )
    return BlimpResult(
        paradigm=paradigm, accuracy=point, ci_low=lo, ci_high=hi, n_items=n
    )


def _icl_statistic(values: np.ndarray) -> float:
    """Mean late loss minus mean early loss. Column 0 late, column 1 early."""
    return float(values[:, 0].mean() - values[:, 1].mean())


@torch.no_grad()
def icl_score(
    model: torch.nn.Module,
    domain: str,
    split: str,
    device: str | torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
    root: Path | str = DEFAULT_TOKENIZED_ROOT,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> IclResult:
    """Mean token loss late in the context minus mean token loss early in it.

    Positions 450 to 500 minus positions 50 to 100, on non-overlapping windows
    of the given split. More negative means the model is getting more out of a
    longer context. Bootstrap resamples windows.

    Every window contributes the same number of positions to each side, so the
    difference of means over windows is the same quantity as the difference of
    means over tokens.
    """
    _ready(model, device)
    context_len = int(model.cfg.context_len)
    if context_len < ICL_LATE[1]:
        raise ValueError(
            f"context {context_len} is too short for the pre-registered late window "
            f"{ICL_LATE}"
        )

    late_means: list[np.ndarray] = []
    early_means: list[np.ndarray] = []
    for x_np, y_np in iter_eval_split(domain, split, context_len, batch_size, root):
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
        _, nll = _forward_nll(model, x, y)
        late_means.append(nll[:, ICL_LATE[0] : ICL_LATE[1]].mean(dim=1).cpu().double().numpy())
        early_means.append(
            nll[:, ICL_EARLY[0] : ICL_EARLY[1]].mean(dim=1).cpu().double().numpy()
        )

    if not late_means:
        raise ValueError(f"no windows for domain {domain!r} split {split!r}")

    values = np.stack([np.concatenate(late_means), np.concatenate(early_means)], axis=1)
    point, lo, hi = bootstrap_ci(
        values, _icl_statistic, n_resamples=n_resamples, seed=seed, alpha=alpha
    )
    return IclResult(
        domain=domain,
        split=split,
        icl_score=point,
        ci_low=lo,
        ci_high=hi,
        n_windows=int(values.shape[0]),
        mean_early_loss=float(values[:, 1].mean()),
        mean_late_loss=float(values[:, 0].mean()),
    )


def _copy_loss_delta(values: np.ndarray) -> float:
    """Second copy loss minus first copy loss. Column 0 second, column 1 first."""
    return float(values[:, 0].mean() - values[:, 1].mean())


def make_copy_sequences(
    n_sequences: int, repeat_len: int, vocab_size: int, seed: int
) -> np.ndarray:
    """[n_sequences, 2 * repeat_len] of a random block repeated twice.

    Ids are uniform over the real vocabulary excluding <|endoftext|>, so nothing
    in a sequence looks like a document boundary. Deterministic given seed.
    """
    if repeat_len < 2:
        raise ValueError(f"repeat_len must be at least 2, got {repeat_len}")
    if vocab_size < 2:
        raise ValueError("vocabulary must hold at least one non-eot id")
    rng = np.random.default_rng(seed)
    half = rng.integers(
        EOT_ID + 1, vocab_size, size=(n_sequences, repeat_len), dtype=np.int64
    )
    return np.concatenate([half, half], axis=1)


def copy_scoring_slices(repeat_len: int) -> tuple[slice, slice]:
    """(first, second) scored prediction indices for the copying task.

    Prediction index i scores the token at sequence position i + 1, which is why
    both slices sit one below the positions they name. The first token of each
    copy is excluded: position 0 has no context, and position repeat_len is the
    start of the repeat, which nothing in the sequence marks. That leaves
    repeat_len - 1 scored positions on each side.
    """
    first = slice(0, repeat_len - 1)  # positions 1 .. repeat_len - 1
    second = slice(repeat_len, 2 * repeat_len - 1)  # positions repeat_len+1 .. 2r-1
    return first, second


@torch.no_grad()
def copying_eval(
    model: torch.nn.Module,
    device: str | torch.device,
    n_sequences: int = COPY_N_SEQUENCES,
    repeat_len: int = COPY_REPEAT_LEN,
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
) -> CopyingResult:
    """Copying of a repeated block of uniformly random tokens.

    Each sequence is repeat_len uniformly random token ids repeated twice, so
    2 * repeat_len tokens inside the model's context. Ids are drawn from the
    real vocabulary excluding <|endoftext|>, so nothing in the sequence looks
    like a document boundary. Random ids make the second copy unlearnable except
    by attending to the earlier occurrence, which isolates the
    induction mechanism from ordinary language modeling.

    Two numbers, both pre-registered:
      loss_delta, mean token loss on the second copy minus the first.
      exact_match, how often the argmax over the second copy is the right token.

    Scored positions: the first token of each copy is excluded from both. The
    first token of the second copy is not predictable, since nothing marks where
    the repeat begins, and the first token of the first copy has no context at
    all. Excluding one from each keeps the two sides symmetric at
    repeat_len - 1 positions each, so the difference is not an artifact of
    scoring different numbers of positions.

    Deterministic given seed.
    """
    _ready(model, device)
    cfg = model.cfg
    context_len = int(cfg.context_len)
    vocab_size = int(cfg.vocab_size)
    if 2 * repeat_len > context_len:
        raise ValueError(
            f"two copies of {repeat_len} tokens exceed the model context {context_len}"
        )

    seqs = make_copy_sequences(n_sequences, repeat_len, vocab_size, seed)
    first, second = copy_scoring_slices(repeat_len)

    first_means: list[np.ndarray] = []
    second_means: list[np.ndarray] = []
    matches: list[np.ndarray] = []
    for start in range(0, n_sequences, batch_size):
        chunk = seqs[start : start + batch_size]
        arr = torch.from_numpy(chunk).to(device)
        x, y = arr[:, :-1], arr[:, 1:]
        logits, nll = _forward_nll(model, x, y)
        first_means.append(nll[:, first].mean(dim=1).cpu().double().numpy())
        second_means.append(nll[:, second].mean(dim=1).cpu().double().numpy())
        preds = logits[:, second, :].argmax(dim=-1)
        matches.append(
            # cpu() first: MPS has no float64. Bool to float64 is exact, and the
            # mean then accumulates in float64 on the CPU as intended.
            (preds == y[:, second]).cpu().double().mean(dim=1).numpy()
        )

    values = np.stack(
        [np.concatenate(second_means), np.concatenate(first_means)], axis=1
    )
    delta, delta_lo, delta_hi = bootstrap_ci(
        values, _copy_loss_delta, n_resamples=n_resamples, seed=seed, alpha=alpha
    )
    acc_values = np.concatenate(matches)
    acc, acc_lo, acc_hi = bootstrap_ci(
        acc_values,
        lambda v: float(v.mean()),
        n_resamples=n_resamples,
        seed=seed,
        alpha=alpha,
    )
    return CopyingResult(
        loss_delta=delta,
        loss_delta_ci_low=delta_lo,
        loss_delta_ci_high=delta_hi,
        exact_match=acc,
        exact_match_ci_low=acc_lo,
        exact_match_ci_high=acc_hi,
        n_sequences=int(n_sequences),
        repeat_len=int(repeat_len),
        seed=int(seed),
        mean_first_copy_loss=float(values[:, 1].mean()),
        mean_second_copy_loss=float(values[:, 0].mean()),
    )
