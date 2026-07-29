"""Decontaminate the web corpus extension against the closed holdout, at the
document level, in token space.

Why this exists. The staged deep run expands the FineWeb-Edu train stream by
continuing the sample-10BT stream past the original prefix (scripts/02_download.py
--extra-web-tokens). The 200-gram leakage check (scripts/12_leakage_check.py)
fired on that extension: sample-10BT contains near-duplicate documents of the
held-out val and test documents, and continuing the stream past the original
slice pulls those duplicates into training. The original slice was clean only
because the holdout documents were excluded from exactly that prefix at the
document level; the continuation has no such exclusion.

What it does. It tokenizes the fineweb_edu_ext raw documents into the same
train-ext shards scripts/04_tokenize.py --extra-web would write, but drops any
document that shares a 200-token n-gram with the fineweb_edu val or test stream.
A contaminated tokenized artifact is never written to disk: filtering happens as
the shards are built, so the shards that land are already clean.

Why it passes the gate by construction. The registered leakage gate samples
200-grams from the holdout and searches the train shards for a verbatim match.
This filter removes every extension document that contains any holdout 200-gram,
so no holdout 200-gram survives in the train-ext stream, so any 200-gram the gate
samples from the holdout is absent from train. The gate therefore passes for any
sample the gate happens to draw, not just on average. The only windows the
per-document filter does not see are ones that would straddle a boundary between
two surviving documents; a full hash-scan of the written shards checks those too
and asserts zero holdout 200-grams remain anywhere.

Why token space rather than raw text. The gate is defined on token n-grams, so
the decontamination is defined on token n-grams as well: a document is dropped
iff its tokenized form contains a holdout 200-token-gram. Working in the same
space the gate works in is what makes the guarantee exact rather than approximate.

Method detail. Holdout 200-grams are compared by a 64-bit polynomial hash rather
than stored verbatim (ten million 200-token windows would be four gigabytes of
ids). A hash collision can only cause a document to be dropped that did not need
to be, never the reverse, so it costs at most a negligible amount of clean data
and cannot let a leak through. With about ten million holdout n-grams and a few
billion extension n-grams the expected number of false-positive windows is far
below one, and the registered gate is the final word regardless.

This script replaces scripts/04_tokenize.py --extra-web for the extension: run
this instead, then run scripts/12_leakage_check.py --domain fineweb_edu
--extension to record the registered PASS.

Usage:
  python scripts/13_filter_ext.py
  python scripts/13_filter_ext.py --workers 8 --ngram 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
from tokenizers import Tokenizer

from small_lm_lab.data import open_split, train_ext_paths
from small_lm_lab.paths import RAW_ROOT, TOKENIZED_ROOT

REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = REPO_ROOT / "data" / "tokenizer" / "tokenizer.json"

DTYPE = np.uint16
EOT_TOKEN = "<|endoftext|>"

# The domain whose val and test the extension is checked against, the raw domain
# the extension documents come from, and the domain the train-ext shards belong
# to. Matches scripts/04_tokenize.py so the loader and the gate read the output
# with no change.
EXT_RAW_DOMAIN = "fineweb_edu_ext"
EXT_OUT_DOMAIN = "fineweb_edu"
EXT_META_NAME = "fineweb_edu_train_ext_meta.json"
DECON_REPORT_NAME = "fineweb_edu_train_ext_decon_report.json"
HOLDOUT_HASH_NAME = "fineweb_edu_holdout_200gram_hashes.npy"

NGRAM = 200
HOLDOUT_SPLITS = ("val", "test")

# Roll a new shard at ~512M tokens, identical to scripts/04_tokenize.py.
SHARD_MAX_TOKENS = 512_000_000
DOCS_PER_BATCH = 1000
N_WORKERS = max(mp.cpu_count() - 2, 1)

# Chunk length for hashing the long contiguous streams (holdout, and the
# written shards during self-verification). The per-window materialization is
# (chunk, ngram) uint64, so this bounds peak memory to a few hundred MB.
HASH_CHUNK_TOKENS = 262_144

# Polynomial-hash base. Odd, so it is coprime to 2**64; all arithmetic is uint64
# and wraps mod 2**64 by numpy's overflow semantics.
HASH_BASE = np.uint64(1_000_000_007)


# ----------------------------------------------------------------------------
# n-gram hashing
# ----------------------------------------------------------------------------

def ngram_powers(n: int) -> np.ndarray:
    """The weights [base**0, base**1, ..., base**(n-1)] as uint64 (mod 2**64)."""
    powers = np.empty(n, dtype=np.uint64)
    powers[0] = np.uint64(1)
    # The wraparound at 2**64 is the modulus, not an error, so it is ignored
    # deliberately rather than left to warn on every call.
    with np.errstate(over="ignore"):
        for i in range(1, n):
            powers[i] = powers[i - 1] * HASH_BASE
    return powers


def ngram_hashes(tokens: np.ndarray, n: int, powers: Optional[np.ndarray] = None) -> np.ndarray:
    """64-bit hashes of every length-n window in tokens, in position order.

    A fixed polynomial hash: window (t0..t_{n-1}) maps to sum_j (t_j + 1) *
    base**j mod 2**64. Token ids are shifted by one so the separator id 0 never
    zeroes out a term. Returns an empty array when tokens is shorter than n,
    since a stream with no length-n window has no n-gram to match.
    """
    length = int(tokens.shape[0])
    if length < n:
        return np.empty(0, dtype=np.uint64)
    if powers is None:
        powers = ngram_powers(n)
    shifted = tokens.astype(np.uint64) + np.uint64(1)
    windows = np.lib.stride_tricks.sliding_window_view(shifted, n)
    # (windows * powers) wraps mod 2**64; the row sum accumulates in uint64. The
    # wraparound is the modulus, so overflow is ignored deliberately.
    with np.errstate(over="ignore"):
        return (windows * powers).sum(axis=1)


def iter_ngram_hashes_chunked(
    stream: "np.ndarray",
    n: int,
    powers: Optional[np.ndarray] = None,
    chunk_tokens: int = HASH_CHUNK_TOKENS,
) -> Iterator[np.ndarray]:
    """ngram_hashes over a long (possibly memmapped) stream, in bounded chunks.

    Consecutive chunks overlap by n-1 tokens so that every window is emitted
    exactly once, including windows that would fall across a chunk boundary. The
    concatenation of the yielded arrays equals ngram_hashes(stream, n).
    """
    if powers is None:
        powers = ngram_powers(n)
    length = int(stream.shape[0])
    if length < n:
        return
    step = max(chunk_tokens, n)
    start = 0
    while start + n <= length:
        end = min(start + step, length)
        sub = np.asarray(stream[start:end])
        yield ngram_hashes(sub, n, powers)
        if end == length:
            break
        start = end - (n - 1)


def contains_holdout_ngram(
    tokens: np.ndarray, holdout_sorted: np.ndarray, n: int, powers: np.ndarray
) -> bool:
    """Whether any length-n window of tokens hashes into the sorted holdout set."""
    h = ngram_hashes(tokens, n, powers)
    if h.size == 0 or holdout_sorted.size == 0:
        return False
    idx = np.searchsorted(holdout_sorted, h)
    idx = np.minimum(idx, holdout_sorted.size - 1)
    return bool((holdout_sorted[idx] == h).any())


def build_holdout_hashes(
    domain: str,
    n: int,
    root: Path,
    splits: tuple[str, ...] = HOLDOUT_SPLITS,
) -> np.ndarray:
    """Sorted, unique 64-bit hashes of every holdout n-gram for a domain.

    Built from the closed val and test streams, which a corpus expansion never
    touches. Sorted so membership is a searchsorted, unique so it is compact.
    """
    powers = ngram_powers(n)
    parts: list[np.ndarray] = []
    for split in splits:
        stream = open_split(domain, split, root)
        for h in iter_ngram_hashes_chunked(stream, n, powers):
            parts.append(h)
    if not parts:
        return np.empty(0, dtype=np.uint64)
    return np.unique(np.concatenate(parts))


# ----------------------------------------------------------------------------
# tokenize + filter workers
# ----------------------------------------------------------------------------

_WORKER_TOKENIZER: Optional[Tokenizer] = None
_WORKER_EOT_ID: int = 0
_WORKER_HOLDOUT: Optional[np.ndarray] = None
_WORKER_POWERS: Optional[np.ndarray] = None
_WORKER_N: int = NGRAM


def _worker_init(tokenizer_path: str, eot_id: int, holdout_hash_path: str, n: int) -> None:
    global _WORKER_TOKENIZER, _WORKER_EOT_ID, _WORKER_HOLDOUT, _WORKER_POWERS, _WORKER_N
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _WORKER_TOKENIZER = Tokenizer.from_file(tokenizer_path)
    _WORKER_EOT_ID = eot_id
    # Memmapped so the workers share one copy of the holdout hashes rather than
    # each unpickling its own.
    _WORKER_HOLDOUT = np.load(holdout_hash_path, mmap_mode="r")
    _WORKER_N = n
    _WORKER_POWERS = ngram_powers(n)


def _encode_and_filter_batch(texts: list[str]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Encode a batch and keep only documents with no holdout n-gram.

    Returns the concatenated token ids of the kept documents, each terminated by
    the separator exactly as the base train stream is, plus counts
    (docs_in, docs_dropped, tokens_kept, tokens_dropped) where token counts
    include the per-document separator.
    """
    assert _WORKER_TOKENIZER is not None and _WORKER_HOLDOUT is not None
    assert _WORKER_POWERS is not None
    encodings = _WORKER_TOKENIZER.encode_batch(texts)
    kept: list[np.ndarray] = []
    docs_in = len(texts)
    docs_dropped = 0
    tokens_kept = 0
    tokens_dropped = 0
    sep = np.array([_WORKER_EOT_ID], dtype=DTYPE)
    for enc in encodings:
        ids = np.asarray(enc.ids, dtype=DTYPE)
        if contains_holdout_ngram(ids, _WORKER_HOLDOUT, _WORKER_N, _WORKER_POWERS):
            docs_dropped += 1
            tokens_dropped += ids.size + 1
            continue
        kept.append(ids)
        kept.append(sep)
        tokens_kept += ids.size + 1
    out = np.concatenate(kept) if kept else np.empty(0, dtype=DTYPE)
    return out, (docs_in, docs_dropped, tokens_kept, tokens_dropped)


# ----------------------------------------------------------------------------
# shard writing (matches scripts/04_tokenize.py output format)
# ----------------------------------------------------------------------------

def ext_shard_path(out_domain: str, index: int, root: Path) -> Path:
    return root / f"{out_domain}_train_ext_{index:03d}.bin"


def stream_encoded_to_shards(
    encoded: Iterable[np.ndarray],
    out_domain: str,
    root: Path,
    shard_max_tokens: int = SHARD_MAX_TOKENS,
) -> tuple[int, list[dict]]:
    """Write already-encoded token arrays into rolling train-ext shard files.

    Identical rollover rule and file naming to scripts/04_tokenize.py, so the
    loader and the gate cannot tell a decontaminated extension from an
    unfiltered one by its shape. Empty arrays (a whole batch dropped) contribute
    nothing and never open a shard on their own.
    """
    shard_idx = 0
    shard_tokens = 0
    total = 0
    path = ext_shard_path(out_domain, shard_idx, root)
    fh = path.open("wb")
    shards: list[dict] = [{"file": path.name, "tokens": 0}]
    try:
        for arr in encoded:
            if arr.size == 0:
                continue
            if shard_tokens > 0 and shard_tokens + int(arr.size) > shard_max_tokens:
                fh.close()
                shards[-1]["tokens"] = shard_tokens
                shard_idx += 1
                shard_tokens = 0
                path = ext_shard_path(out_domain, shard_idx, root)
                fh = path.open("wb")
                shards.append({"file": path.name, "tokens": 0})
            arr.tofile(fh)
            shard_tokens += int(arr.size)
            total += int(arr.size)
    finally:
        fh.close()
    shards[-1]["tokens"] = shard_tokens
    return total, shards


# ----------------------------------------------------------------------------
# raw document iteration
# ----------------------------------------------------------------------------

def load_raw_manifest(domain: str, raw_root: Path) -> dict:
    path = raw_root / domain / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; run scripts/02_download.py --extra-web-tokens first"
        )
    with path.open() as f:
        return json.load(f)


def iter_raw_docs(domain: str, raw_root: Path) -> Iterator[str]:
    manifest = load_raw_manifest(domain, raw_root)
    for shard_name in manifest["shards"]:
        with (raw_root / domain / shard_name).open("r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)["text"]


def batched(iterator: Iterator[str], size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for item in iterator:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


# ----------------------------------------------------------------------------
# self-verification
# ----------------------------------------------------------------------------

def count_holdout_ngrams_in_shards(
    shard_paths: list[Path], holdout_sorted: np.ndarray, n: int
) -> int:
    """Count holdout n-grams present anywhere in the written shards.

    Scans the full concatenated train-ext stream, so it also sees windows that
    straddle a boundary between two surviving documents, which the per-document
    filter does not. Should be zero.
    """
    powers = ngram_powers(n)
    leaks = 0
    for path in shard_paths:
        stream = np.memmap(path, dtype=DTYPE, mode="r")
        for h in iter_ngram_hashes_chunked(stream, n, powers):
            if h.size == 0:
                continue
            idx = np.searchsorted(holdout_sorted, h)
            idx = np.minimum(idx, holdout_sorted.size - 1)
            leaks += int((holdout_sorted[idx] == h).sum())
    return leaks


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------

def filter_extension(
    tokenizer: Tokenizer,
    eot_id: int,
    n: int = NGRAM,
    workers: int = N_WORKERS,
    raw_root: Path = RAW_ROOT,
    tokenized_root: Path = TOKENIZED_ROOT,
    shard_max_tokens: int = SHARD_MAX_TOKENS,
    verify: bool = True,
) -> dict:
    """Tokenize and decontaminate the extension into train-ext shards.

    Refuses to run if extension shards already exist, so a rerun cannot silently
    clobber or double-append. Writes the shards, the extension meta (same shape
    as scripts/04_tokenize.py, plus a decontamination block), a standalone
    decontamination report, and the cached holdout hash table.
    """
    tokenized_root.mkdir(parents=True, exist_ok=True)

    existing = train_ext_paths(EXT_OUT_DOMAIN, tokenized_root)
    if existing:
        raise SystemExit(
            f"extension shards already exist ({[p.name for p in existing]}); "
            "remove them or move them aside before re-filtering the extension"
        )

    manifest = load_raw_manifest(EXT_RAW_DOMAIN, raw_root)
    print(f"[run ] {EXT_RAW_DOMAIN}: {manifest['documents']:,} docs -> filtered train-ext shards")

    print(f"       building holdout {n}-gram hash table from {EXT_OUT_DOMAIN} val + test")
    holdout_sorted = build_holdout_hashes(EXT_OUT_DOMAIN, n, tokenized_root)
    hash_path = tokenized_root / HOLDOUT_HASH_NAME
    np.save(hash_path, holdout_sorted)
    print(f"       {holdout_sorted.size:,} unique holdout {n}-grams -> {hash_path.name}")

    batches = batched(iter_raw_docs(EXT_RAW_DOMAIN, raw_root), DOCS_PER_BATCH)

    totals = {"docs_in": 0, "docs_dropped": 0, "tokens_kept": 0, "tokens_dropped": 0}

    def encoded_arrays(source) -> Iterator[np.ndarray]:
        for arr, (d_in, d_drop, t_keep, t_drop) in source:
            totals["docs_in"] += d_in
            totals["docs_dropped"] += d_drop
            totals["tokens_kept"] += t_keep
            totals["tokens_dropped"] += t_drop
            if totals["docs_in"] % 500_000 < DOCS_PER_BATCH:
                print(
                    f"       {totals['docs_in']:,} docs seen, "
                    f"{totals['docs_dropped']:,} dropped, "
                    f"{totals['tokens_kept']/1e6:.0f} M tokens kept",
                    flush=True,
                )
            yield arr

    pool: Optional[mp.pool.Pool] = None
    if workers and workers > 1:
        pool = mp.Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(str(TOKENIZER_PATH), eot_id, str(hash_path), n),
        )
        source = pool.imap(_encode_and_filter_batch, batches, chunksize=1)
    else:
        # Single process: initialize the module globals the worker fn reads.
        _worker_init(str(TOKENIZER_PATH), eot_id, str(hash_path), n)
        source = map(_encode_and_filter_batch, batches)

    try:
        total, shards = stream_encoded_to_shards(
            encoded_arrays(source), EXT_OUT_DOMAIN, tokenized_root, shard_max_tokens
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    docs_kept = totals["docs_in"] - totals["docs_dropped"]
    drop_rate = totals["docs_dropped"] / max(totals["docs_in"], 1)

    shard_paths = [tokenized_root / s["file"] for s in shards]
    verified_leaks: Optional[int] = None
    if verify:
        print("       self-verifying: scanning shards for any residual holdout n-gram")
        verified_leaks = count_holdout_ngrams_in_shards(shard_paths, holdout_sorted, n)
        if verified_leaks != 0:
            raise SystemExit(
                f"self-verification found {verified_leaks} holdout {n}-grams in the "
                "filtered shards; decontamination did not converge, not writing meta"
            )

    decon = {
        "method": (
            "document-level, token-space: drop any extension document whose "
            f"tokenized form contains a {n}-token n-gram present in the "
            f"{EXT_OUT_DOMAIN} val or test stream"
        ),
        "ngram": n,
        "holdout_splits": list(HOLDOUT_SPLITS),
        "holdout_ngrams": int(holdout_sorted.size),
        "documents_in": totals["docs_in"],
        "documents_dropped": totals["docs_dropped"],
        "documents_kept": docs_kept,
        "drop_rate": drop_rate,
        "tokens_kept": totals["tokens_kept"],
        "tokens_dropped": totals["tokens_dropped"],
        "hash": "64-bit polynomial (base 1000000007, mod 2**64), id-shifted by 1",
        "collision_note": (
            "a hash collision can only drop a clean document, never admit a leak; "
            "the registered gate (scripts/12_leakage_check.py) is the final check"
        ),
        "self_verified_residual_ngrams": verified_leaks,
    }

    ext_meta = {
        "created": datetime.now(timezone.utc).isoformat(),
        "extends_domain": EXT_OUT_DOMAIN,
        "raw_domain": EXT_RAW_DOMAIN,
        "split": "train",
        "tokens": total,
        "shards": shards,
        "shard_max_tokens": shard_max_tokens,
        "tokenizer_sha256": hashlib.sha256(TOKENIZER_PATH.read_bytes()).hexdigest(),
        "provenance": manifest["provenance"],
        "decontamination": decon,
        "note": (
            "Additional train-only web shards, decontaminated at the document "
            "level against the closed val and test. The base train/val/test "
            "streams and meta.json are unchanged. Produced by "
            "scripts/13_filter_ext.py (which replaces scripts/04_tokenize.py "
            "--extra-web for the extension). Run scripts/12_leakage_check.py "
            "--domain fineweb_edu --extension to record the registered PASS."
        ),
    }
    with (tokenized_root / EXT_META_NAME).open("w") as f:
        json.dump(ext_meta, f, indent=2)
    with (tokenized_root / DECON_REPORT_NAME).open("w") as f:
        json.dump(decon, f, indent=2)

    print(f"       kept {docs_kept:,}/{totals['docs_in']:,} docs "
          f"({100 * drop_rate:.3f}% dropped), {total:,} tokens across {len(shards)} shards")
    for shard in shards:
        print(f"         {shard['file']}: {shard['tokens']:,} tokens")
    if verify:
        print(f"       self-verification: {verified_leaks} residual holdout {n}-grams (want 0)")
    print(f"       wrote {tokenized_root / EXT_META_NAME}")
    print(f"       wrote {tokenized_root / DECON_REPORT_NAME}")
    return ext_meta


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ngram", type=int, default=NGRAM, help="n-gram length for the match")
    p.add_argument("--workers", type=int, default=N_WORKERS, help="tokenize/filter processes")
    p.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the full-scan self-verification of the written shards",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"missing {TOKENIZER_PATH}; run scripts/03_train_tokenizer.py first"
        )
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    vocab_size = tokenizer.get_vocab_size()
    eot_id = tokenizer.token_to_id(EOT_TOKEN)
    if eot_id is None:
        raise SystemExit(f"tokenizer has no {EOT_TOKEN} token")
    assert vocab_size < 65536, f"vocab {vocab_size} does not fit uint16"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print(f"tokenizer: {TOKENIZER_PATH} (vocab {vocab_size}, {EOT_TOKEN} id {eot_id})")
    print(f"output:    {TOKENIZED_ROOT}")
    print(f"workers:   {args.workers}")
    print(f"ngram:     {args.ngram}\n")

    filter_extension(
        tokenizer,
        eot_id,
        n=args.ngram,
        workers=args.workers,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
