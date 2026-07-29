"""Batch loading over the tokenized uint16 streams.

Two access patterns, matching the two things the lab does with data:

  MixtureBatcher: random training batches drawn from several domains at a
    fixed mix ratio. Each batch element picks its domain independently by
    weight, then a random window inside that domain's stream. Given a seed the
    entire sequence of batches is reproducible, which matters because the
    scaling comparison across model sizes is only clean if every run sees the
    same token order.

  iter_eval_windows: deterministic, exhaustive, non-overlapping windows over a
    val or test stream, for perplexity. Non-overlapping matters because
    overlapping windows would score some tokens repeatedly and weight them more
    heavily in the average.

Arrays come back as numpy int64. The framework decision (MLX or PyTorch-MPS) is
still open, and both accept numpy, so nothing here imports either one.

Windows may cross document boundaries. That is intended: documents are joined by
the <|endoftext|> separator, and learning to predict the separator and then a
fresh document start is part of the modeling task. The train/val/test split is
by document, so a window that crosses a boundary inside val still contains only
val documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .paths import TOKENIZED_ROOT

DEFAULT_TOKENIZED_ROOT = TOKENIZED_ROOT
DTYPE = np.uint16


# Additional train shards written by a corpus expansion sit alongside the base
# train stream as {domain}_train_ext_{k}.bin, so the base train/val/test streams
# are never rewritten. The index is zero padded so a lexical sort is also the
# order the shards were written.
TRAIN_EXT_GLOB = "{domain}_train_ext_*.bin"


def split_path(domain: str, split: str, root: Path | str = DEFAULT_TOKENIZED_ROOT) -> Path:
    return Path(root) / f"{domain}_{split}.bin"


def train_ext_paths(
    domain: str, root: Path | str = DEFAULT_TOKENIZED_ROOT
) -> list[Path]:
    """Additional train shards for a domain, in write order (oldest first)."""
    return sorted(Path(root).glob(TRAIN_EXT_GLOB.format(domain=domain)))


def open_split(
    domain: str, split: str, root: Path | str = DEFAULT_TOKENIZED_ROOT
) -> np.memmap:
    """Memmap one token stream read-only.

    Memmap rather than read: the train streams are hundreds of MB to a few GB
    and only a few windows of each are touched per batch, so the page cache
    handles residency and process memory stays flat.
    """
    path = split_path(domain, split, root)
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run scripts/04_tokenize.py first")
    return np.memmap(path, dtype=DTYPE, mode="r")


class ConcatTokens:
    """Read-only virtual concatenation of several uint16 token memmaps.

    A corpus expansion adds train shards alongside the base train stream rather
    than rewriting it. This presents the base plus those extra shards as one
    contiguous stream, so MixtureBatcher can draw a training window from anywhere
    in the expanded corpus without any shard being read into memory: the parts
    stay memmapped and only the requested window is materialized.

    Only the two operations the batcher uses are supported: reading the total
    length through .shape, and slicing one contiguous window. A window that falls
    inside a single shard is returned as that shard's memmap view; the rare
    window that spans a shard boundary is stitched from both. Crossing a shard
    boundary is the same kind of event as crossing a document boundary inside a
    single stream, which the lab already treats as part of the modeling task, so
    the seam needs no special handling beyond being read correctly.
    """

    def __init__(self, parts: Sequence[np.memmap]) -> None:
        self._parts = list(parts)
        if not self._parts:
            raise ValueError("ConcatTokens needs at least one part")
        self._lengths = np.array([int(p.shape[0]) for p in self._parts], dtype=np.int64)
        self._offsets = np.concatenate([[0], np.cumsum(self._lengths)]).astype(np.int64)
        self.shape = (int(self._offsets[-1]),)
        self.dtype = self._parts[0].dtype

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key: slice) -> np.ndarray:
        if not isinstance(key, slice):
            raise TypeError("ConcatTokens supports only contiguous slice indexing")
        start, stop, step = key.indices(self.shape[0])
        if step != 1:
            raise ValueError("ConcatTokens supports only contiguous (step 1) slices")
        if stop <= start:
            return np.empty(0, dtype=self.dtype)
        first = int(np.searchsorted(self._offsets, start, side="right") - 1)
        last = int(np.searchsorted(self._offsets, stop - 1, side="right") - 1)
        if first == last:
            base = int(self._offsets[first])
            return self._parts[first][start - base : stop - base]
        chunks: list[np.ndarray] = []
        for part_idx in range(first, last + 1):
            part_base = int(self._offsets[part_idx])
            part_end = int(self._offsets[part_idx + 1])
            lo = max(start, part_base) - part_base
            hi = min(stop, part_end) - part_base
            chunks.append(np.asarray(self._parts[part_idx][lo:hi]))
        return np.concatenate(chunks)


def open_train_tokens(
    domain: str, root: Path | str = DEFAULT_TOKENIZED_ROOT
) -> np.memmap | ConcatTokens:
    """The domain's full training stream: the base shard plus any extra shards.

    With no extra shards this is exactly the base memmap open_split returns, so a
    domain that was never expanded reads identically to before. With extra shards
    present it is a ConcatTokens over the base and the extras, in write order.
    """
    base = open_split(domain, "train", root)
    exts = train_ext_paths(domain, root)
    if not exts:
        return base
    parts = [base] + [np.memmap(p, dtype=DTYPE, mode="r") for p in exts]
    return ConcatTokens(parts)


@dataclass(frozen=True)
class DomainSource:
    """One domain's stream and its share of each batch.

    tokens is a memmap for a single-shard stream (val, test, or an unexpanded
    train stream) or a ConcatTokens over the base plus extra shards for an
    expanded train stream. Both expose .shape and contiguous slicing, which is
    all MixtureBatcher asks of them.
    """

    name: str
    weight: float
    tokens: "np.memmap | ConcatTokens"

    @property
    def n_tokens(self) -> int:
        return int(self.tokens.shape[0])


def load_sources(
    weights: dict[str, float],
    split: str = "train",
    root: Path | str = DEFAULT_TOKENIZED_ROOT,
) -> list[DomainSource]:
    """Memmap one stream per domain and attach normalized mix weights."""
    if not weights:
        raise ValueError("weights must name at least one domain")
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError(f"weights must sum to a positive number, got {total}")
    sources: list[DomainSource] = []
    for name, w in weights.items():
        # Only the train stream picks up expansion shards. val and test are the
        # closed evaluation splits and are always a single stream, never expanded,
        # so a corpus expansion cannot touch what perplexity is scored on.
        if split == "train":
            tokens = open_train_tokens(name, root)
        else:
            tokens = open_split(name, split, root)
        sources.append(DomainSource(name=name, weight=w / total, tokens=tokens))
    return sources


class MixtureBatcher:
    """Random (x, y) next-token batches drawn from a weighted domain mixture.

    Deterministic given seed: the same seed replays the same domain choices and
    the same window offsets, in the same order.
    """

    def __init__(
        self,
        sources: Sequence[DomainSource],
        batch_size: int,
        context_len: int,
        seed: int,
    ) -> None:
        if not sources:
            raise ValueError("need at least one DomainSource")
        for s in sources:
            # A window needs context_len inputs plus the one shifted target.
            if s.n_tokens < context_len + 1:
                raise ValueError(
                    f"domain {s.name!r} has {s.n_tokens} tokens, "
                    f"needs at least {context_len + 1}"
                )
        self.sources = list(sources)
        self.batch_size = batch_size
        self.context_len = context_len
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._probs = np.array([s.weight for s in self.sources], dtype=np.float64)
        self._probs = self._probs / self._probs.sum()
        # Highest valid start index, inclusive, per domain.
        self._max_start = np.array(
            [s.n_tokens - context_len - 1 for s in self.sources], dtype=np.int64
        )

    def reset(self) -> None:
        """Rewind the stream to the batch this batcher started at."""
        self._rng = np.random.default_rng(self.seed)

    def next_batch(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (x, y), each int64 of shape (batch_size, context_len)."""
        t = self.context_len
        picks = self._rng.choice(len(self.sources), size=self.batch_size, p=self._probs)
        # One draw per element in [0, max_start], then scaled per domain, so the
        # RNG consumes a fixed amount regardless of which domains came up.
        unit = self._rng.random(self.batch_size)
        starts = (unit * (self._max_start[picks] + 1)).astype(np.int64)
        starts = np.minimum(starts, self._max_start[picks])

        x = np.empty((self.batch_size, t), dtype=np.int64)
        y = np.empty((self.batch_size, t), dtype=np.int64)
        for row, (domain_idx, start) in enumerate(zip(picks, starts)):
            tokens = self.sources[domain_idx].tokens
            window = np.asarray(tokens[start : start + t + 1], dtype=np.int64)
            x[row] = window[:-1]
            y[row] = window[1:]
        return x, y

    def domain_counts(self, n_batches: int) -> dict[str, int]:
        """Element counts per domain over n_batches, without disturbing state.

        Used to confirm the realized mix matches the configured weights.
        """
        rng = np.random.default_rng(self.seed)
        counts = {s.name: 0 for s in self.sources}
        for _ in range(n_batches):
            picks = rng.choice(len(self.sources), size=self.batch_size, p=self._probs)
            rng.random(self.batch_size)
            for i in picks:
                counts[self.sources[i].name] += 1
        return counts


def n_eval_windows(n_tokens: int, context_len: int) -> int:
    """Number of non-overlapping windows a stream yields."""
    return max((n_tokens - 1) // context_len, 0)


def iter_eval_windows(
    tokens: np.memmap,
    context_len: int,
    batch_size: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield non-overlapping (x, y) batches covering a stream exactly once.

    Deterministic and exhaustive. Windows advance by context_len so no token is
    predicted twice. The final partial window is dropped, so every batch is the
    same shape; at 5M tokens and context 512 that discards at most 511 tokens.
    The last batch may hold fewer than batch_size rows.
    """
    t = context_len
    total = n_eval_windows(int(tokens.shape[0]), t)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for w in range(total):
        start = w * t
        window = np.asarray(tokens[start : start + t + 1], dtype=np.int64)
        xs.append(window[:-1])
        ys.append(window[1:])
        if len(xs) == batch_size:
            yield np.stack(xs), np.stack(ys)
            xs, ys = [], []
    if xs:
        yield np.stack(xs), np.stack(ys)


def iter_eval_split(
    domain: str,
    split: str,
    context_len: int,
    batch_size: int,
    root: Path | str = DEFAULT_TOKENIZED_ROOT,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """iter_eval_windows over a named domain and split."""
    yield from iter_eval_windows(open_split(domain, split, root), context_len, batch_size)
