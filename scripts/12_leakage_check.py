"""Token-level n-gram leakage check for new train shards against val and test.

The corpus expansion adds web train shards. Before training on them, the same
check the base corpus passed has to pass for the new shards: no held-out
document leaks into training. This is the token-level 200-gram check, run for the
new train shards against the existing val and test streams, which are never
touched by an expansion.

Method, matching the check the base corpus passed. For each holdout split it
samples n-grams (default length 200) at seeded random positions and searches the
train shards for an exact token match. A held-out n-gram found verbatim in train
is a leak. The search is exact: candidate positions are found by the n-gram's
first token, then the full window is compared, so nothing is a hash collision.

A positive control travels with every run. The same sampled holdout n-grams are
searched in their own split, where they must be found, so a clean train result
(zero leaks) is only trusted when the control confirms the search actually finds
a present n-gram. A failed control is reported as a failure, because a search
that finds nothing proves nothing.

Exits nonzero if any leak is found or if the positive control fails, so it can
gate a run from a shell script.

Usage:
  # the new extension shards for a domain against its val and test
  python scripts/12_leakage_check.py --domain fineweb_edu --extension
  # explicit shard files
  python scripts/12_leakage_check.py --domain fineweb_edu \
      --train-shards /path/fineweb_edu_train_ext_000.bin
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from small_lm_lab.data import DEFAULT_TOKENIZED_ROOT, open_split, train_ext_paths

DTYPE = np.uint16
DEFAULT_NGRAM = 200
DEFAULT_SAMPLES = 128
DEFAULT_SEED = 20260717
HOLDOUT_SPLITS = ("val", "test")


def sample_ngrams(
    tokens: np.ndarray, n: int, k: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """k n-grams of length n at seeded random start positions."""
    length = int(tokens.shape[0])
    if length < n:
        raise ValueError(f"stream of {length} tokens is shorter than the {n}-gram")
    max_start = length - n
    starts = rng.integers(0, max_start + 1, size=k)
    return [np.asarray(tokens[s : s + n], dtype=np.int64) for s in starts]


def ngram_in_stream(ngram: np.ndarray, stream: np.ndarray) -> bool:
    """Whether ngram appears verbatim in stream. Exact, not a hash match.

    Candidate positions are where the stream's token equals the n-gram's first
    token; each candidate is then compared in full, so a returned True is a real
    verbatim match and not a collision.
    """
    n = int(ngram.shape[0])
    length = int(stream.shape[0])
    if length < n:
        return False
    head = np.asarray(stream[: length - n + 1], dtype=np.int64)
    candidates = np.nonzero(head == int(ngram[0]))[0]
    for c in candidates:
        if np.array_equal(np.asarray(stream[c : c + n], dtype=np.int64), ngram):
            return True
    return False


def ngram_in_any(ngram: np.ndarray, streams: list[np.ndarray]) -> bool:
    return any(ngram_in_stream(ngram, s) for s in streams)


@dataclass
class SplitReport:
    split: str
    n_samples: int
    n_leaks: int
    positive_control_passed: bool
    leak_positions: list[int] = field(default_factory=list)


@dataclass
class LeakageReport:
    domain: str
    ngram: int
    train_shards: list[str]
    splits: list[SplitReport]

    @property
    def passed(self) -> bool:
        return all(
            s.n_leaks == 0 and s.positive_control_passed for s in self.splits
        )


def check_leakage(
    train_streams: list[np.ndarray],
    holdout_streams: dict[str, np.ndarray],
    domain: str,
    train_shard_names: list[str],
    n: int = DEFAULT_NGRAM,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> LeakageReport:
    """Run the sampled n-gram check for one domain.

    For each holdout split, sample n-grams, count how many appear verbatim in the
    train streams (leaks), and confirm the same n-grams are found in their own
    split (the positive control).
    """
    splits: list[SplitReport] = []
    for split_index, (split, stream) in enumerate(holdout_streams.items()):
        # Seeded from the split's fixed position, not Python's salted hash(), so
        # the sampled n-grams are reproducible across processes and runs.
        rng = np.random.default_rng([seed, split_index])
        grams = sample_ngrams(stream, n, samples, rng)
        leak_positions = [i for i, g in enumerate(grams) if ngram_in_any(g, train_streams)]
        control = all(ngram_in_stream(g, stream) for g in grams)
        splits.append(
            SplitReport(
                split=split,
                n_samples=len(grams),
                n_leaks=len(leak_positions),
                positive_control_passed=control,
                leak_positions=leak_positions,
            )
        )
    return LeakageReport(
        domain=domain, ngram=n, train_shards=train_shard_names, splits=splits
    )


def format_report(report: LeakageReport) -> str:
    lines = [
        f"{report.ngram}-gram leakage check for domain {report.domain!r}",
        f"  train shards: {report.train_shards}",
    ]
    for s in report.splits:
        control = "PASS" if s.positive_control_passed else "FAIL (search found nothing)"
        verdict = "clean" if s.n_leaks == 0 else f"LEAK ({s.n_leaks} of {s.n_samples})"
        lines.append(
            f"  {s.split}: {s.n_leaks}/{s.n_samples} sampled n-grams found in "
            f"train -> {verdict}; positive control {control}"
        )
    lines.append(f"  verdict: {'PASS' if report.passed else 'FAIL'}")
    return "\n".join(lines)


def resolve_train_shards(args: argparse.Namespace) -> list[Path]:
    if args.train_shards:
        return [Path(p) for p in args.train_shards]
    if args.extension:
        paths = train_ext_paths(args.domain, args.tokenized_root)
        if not paths:
            raise SystemExit(
                f"no {args.domain}_train_ext_*.bin shards under "
                f"{args.tokenized_root}; nothing to check"
            )
        return paths
    raise SystemExit(
        "name the train shards to check: --extension (the new "
        f"{args.domain}_train_ext_*.bin shards) or --train-shards PATH ..."
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--domain", default="fineweb_edu", help="domain whose val and test to check against")
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--extension",
        action="store_true",
        help="check the new {domain}_train_ext_*.bin shards",
    )
    src.add_argument(
        "--train-shards",
        nargs="+",
        default=None,
        help="explicit train shard .bin files to check",
    )
    p.add_argument("--ngram", type=int, default=DEFAULT_NGRAM)
    p.add_argument("--samples", type=int, default=DEFAULT_SAMPLES, help="n-grams sampled per split")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--tokenized-root", type=Path, default=DEFAULT_TOKENIZED_ROOT)
    return p


def main() -> None:
    args = build_parser().parse_args()
    shard_paths = resolve_train_shards(args)
    train_streams = [np.memmap(p, dtype=DTYPE, mode="r") for p in shard_paths]
    holdout_streams = {
        split: open_split(args.domain, split, args.tokenized_root)
        for split in HOLDOUT_SPLITS
    }
    report = check_leakage(
        train_streams,
        holdout_streams,
        domain=args.domain,
        train_shard_names=[p.name for p in shard_paths],
        n=args.ngram,
        samples=args.samples,
        seed=args.seed,
    )
    print(format_report(report))
    raise SystemExit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
