"""How much of the 16,384-id vocabulary the training corpus actually trains.

Written 2026-07-26 to back a number quoted in docs/PROTOCOL.md
before that prediction could be checked against any model.

The question it answers is specific and it bears on whether the registered copying
probe can measure what it claims. The probe draws token ids uniformly from the
vocabulary excluding the eot id, and a repeated block of those ids is what the
prefix-matching score and the copying accuracy are read from. Natural text is not
uniform: the tokenizer was trained on this mix, so a handful of ids carry most of
the mass and the tail is rare. If the tail were rare enough to leave its embeddings
near their initialization, the probe would be asking the model to match tokens it
has barely seen, and a model with a working induction head could score at chance on
it. That would be a false negative produced by the instrument and not a fact about
the model.

The measurement is a corpus property, not a model property, so it is available
before any checkpoint is read and it cannot be tuned by anything a model does.

What is counted: a prefix of each domain's train stream, not the whole stream. The
web train stream alone is 2.8e9 tokens once the extension is included, and reading
it end to end costs several minutes of disk for a distributional fact that a large
prefix already fixes. The counted prefix is recorded in the output, so the figure
is reproducible and its basis is legible. Ids are uint16 on disk and the vocabulary
is 16,384, so a bincount at that width is exact.

Usage:
  python scripts/20_vocab_coverage.py --out analysis/vocab_coverage.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from small_lm_lab.data import DEFAULT_TOKENIZED_ROOT
from small_lm_lab.evaluate import DOMAINS, EOT_ID
from small_lm_lab.paths import portable_path

# Read the memmap in chunks instead of materializing a 2.8e9-element int64 array.
CHUNK_TOKENS = 20_000_000

# The counts a reader is likely to care about, fixed here and not chosen after
# seeing the histogram.
OCCURRENCE_THRESHOLDS = (1, 10, 100, 1_000, 10_000)
MASS_RANKS = (100, 500, 1_000, 4_000, 8_000)


def count_stream(path: Path, vocab_size: int, max_tokens: int) -> tuple[np.ndarray, int]:
    """Token id counts over the first max_tokens ids of a .bin stream."""
    arr = np.memmap(path, dtype=np.uint16, mode="r")
    n = int(min(max_tokens, arr.shape[0]))
    counts = np.zeros(vocab_size, dtype=np.int64)
    for start in range(0, n, CHUNK_TOKENS):
        chunk = np.asarray(arr[start : min(start + CHUNK_TOKENS, n)], dtype=np.int64)
        counts += np.bincount(chunk, minlength=vocab_size)
    return counts, n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tokenized-root", type=Path, default=DEFAULT_TOKENIZED_ROOT)
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument(
        "--max-tokens-per-domain",
        type=int,
        default=200_000_000,
        help="prefix of each domain's train stream to count; recorded in the output",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    counts = np.zeros(args.vocab_size, dtype=np.int64)
    per_domain: dict[str, dict] = {}
    for domain in DOMAINS:
        path = Path(args.tokenized_root) / f"{domain}_train.bin"
        sub, counted = count_stream(path, args.vocab_size, args.max_tokens_per_domain)
        counts += sub
        per_domain[domain] = {
            "path": portable_path(path),
            "tokens_counted": counted,
            "distinct_ids": int((sub > 0).sum()),
        }
        print(f"{domain}: counted {counted:,} tokens, {int((sub > 0).sum()):,} distinct ids")

    # The uniform draw the copying probe makes: every id except eot.
    drawable = np.ones(args.vocab_size, dtype=bool)
    drawable[EOT_ID] = False
    n_drawable = int(drawable.sum())

    occurrence = {
        str(t): int((counts >= t).sum()) for t in OCCURRENCE_THRESHOLDS
    }
    # The number that decides whether the probe is asking about trained embeddings:
    # the share of a uniform draw landing on an id the corpus barely contains.
    uniform_on_rare = {
        str(t): float((drawable & (counts < t)).sum() / n_drawable)
        for t in OCCURRENCE_THRESHOLDS
    }

    mass = counts[drawable].astype(np.float64)
    mass /= mass.sum()
    cumulative = np.cumsum(np.sort(mass)[::-1])
    concentration = {str(k): float(cumulative[k - 1]) for k in MASS_RANKS}

    result = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tokenized_root": portable_path(args.tokenized_root),
            "vocab_size": args.vocab_size,
            "eot_id": EOT_ID,
            "drawable_ids": n_drawable,
            "max_tokens_per_domain": args.max_tokens_per_domain,
            "note": (
                "counts are over a prefix of each domain's train stream, not the "
                "whole stream. tokens_counted per domain records the prefix used."
            ),
        },
        "per_domain": per_domain,
        "tokens_counted_total": sum(d["tokens_counted"] for d in per_domain.values()),
        "distinct_ids_seen": int((counts > 0).sum()),
        "ids_seen_at_least": occurrence,
        "uniform_draw_share_on_ids_seen_under": uniform_on_rare,
        "corpus_mass_carried_by_top_k_ids": concentration,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
