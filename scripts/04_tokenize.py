"""Tokenize the raw corpora into flat uint16 token streams on the bulk root.

Output per domain: {domain}_train.bin, {domain}_val.bin, {domain}_test.bin,
each a flat uint16 array of token ids with documents joined by the
<|endoftext|> token (id 0). uint16 is exact here because the vocabulary is
16384, well under 65536; the script asserts this rather than trusting it.

Split protocol, and the reason for it:

  Splits are made BY document, never by cutting a token stream at a random
  offset. If the stream were cut by position, the same story or web page could
  appear on both sides of the cut, and held-out perplexity would be measuring
  memorization of text the model had already trained on. So the holdout
  documents are chosen and removed before the training stream is built, and a
  document lands in exactly one split.

  Mechanically: a seeded RNG marks a pool of candidate holdout documents per
  domain, generously sized. Candidates are tokenized first, filling val to
  VAL_TOKENS and then test to TEST_TOKENS, both counted in real tokens rather
  than estimated. Candidates left over once both are full are released back to
  train, so no document is wasted and none is shared. Everything not consumed
  by val or test is then streamed into train.

Throughput: tokenization is CPU bound and parallel over N_WORKERS processes,
fed in bounded batches so memory stays flat regardless of corpus size. The
tokenizer is loaded once per worker. Nothing here touches the GPU.

Corpus expansion (--extra-web): tokenizes the fineweb_edu_ext raw domain (fetched
by scripts/02_download.py --extra-web-tokens) into additional train shards named
fineweb_edu_train_ext_{k}.bin, written alongside the existing streams. It never
rewrites the base train/val/test bins or meta.json: the extension is all train
(val and test were fixed from the base corpus and stay closed), it records its
own meta file, and the loader stitches the extra shards into the train mix. Run
the leakage check (scripts/12_leakage_check.py) on the new shards before using
them.
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

from small_lm_lab.paths import DATA_ROOT, RAW_ROOT, TOKENIZED_ROOT

REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = REPO_ROOT / "data" / "tokenizer" / "tokenizer.json"
META_PATH = TOKENIZED_ROOT / "meta.json"

DOMAINS = ["tinystories", "fineweb_edu"]
DTYPE = np.uint16
EOT_TOKEN = "<|endoftext|>"

VAL_TOKENS = 5_000_000
TEST_TOKENS = 5_000_000
HOLDOUT_SEED = 20260717

# Corpus expansion. The extension is fineweb_edu_ext raw docs tokenized into
# train-only shards for the fineweb_edu domain.
EXT_RAW_DOMAIN = "fineweb_edu_ext"
EXT_OUT_DOMAIN = "fineweb_edu"
EXT_META_NAME = "fineweb_edu_train_ext_meta.json"
# Roll a new shard at ~512M tokens, about 1GB of uint16, so no single file is
# unwieldy and a failed run loses at most one shard.
SHARD_MAX_TOKENS = 512_000_000

# Oversize the candidate pool so val and test fill even if the documents drawn
# are shorter than the manifest average. Leftovers return to train.
CANDIDATE_SAFETY = 2.5
CHARS_PER_TOKEN_ESTIMATE = 4.0

# 10 physical cores; leave one for the feeder process and the OS.
N_WORKERS = max(mp.cpu_count() - 2, 1)
DOCS_PER_BATCH = 1000
HOLDOUT_BATCH_DOCS = 200

_WORKER_TOKENIZER: Optional[Tokenizer] = None
_WORKER_EOT_ID: int = 0


def _worker_init(tokenizer_path: str, eot_id: int) -> None:
    global _WORKER_TOKENIZER, _WORKER_EOT_ID
    # Each worker owns one tokenizer; disable internal threading so the process
    # pool is the only source of parallelism.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _WORKER_TOKENIZER = Tokenizer.from_file(tokenizer_path)
    _WORKER_EOT_ID = eot_id


def _encode_batch(texts: list[str]) -> np.ndarray:
    """Encode a batch of documents, appending the separator after each."""
    assert _WORKER_TOKENIZER is not None
    encodings = _WORKER_TOKENIZER.encode_batch(texts)
    out: list[int] = []
    for enc in encodings:
        out.extend(enc.ids)
        out.append(_WORKER_EOT_ID)
    return np.asarray(out, dtype=DTYPE)


def load_manifest(domain: str) -> dict:
    path = RAW_ROOT / domain / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run scripts/02_download.py first")
    with path.open() as f:
        return json.load(f)


def iter_domain_docs(domain: str) -> Iterator[str]:
    """Yield every document of a domain in shard order."""
    manifest = load_manifest(domain)
    for shard_name in manifest["shards"]:
        with (RAW_ROOT / domain / shard_name).open("r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)["text"]


def choose_candidates(domain: str) -> set[int]:
    """Pick the holdout candidate document indices for a domain.

    Seeded and therefore reproducible. Sized to comfortably cover
    VAL_TOKENS + TEST_TOKENS even if the drawn documents are short.
    """
    manifest = load_manifest(domain)
    total_docs = manifest["documents"]
    avg_doc_chars = manifest["chars"] / max(total_docs, 1)
    avg_doc_tokens = max(avg_doc_chars / CHARS_PER_TOKEN_ESTIMATE, 1.0)

    needed_tokens = (VAL_TOKENS + TEST_TOKENS) * CANDIDATE_SAFETY
    n_candidates = int(min(needed_tokens / avg_doc_tokens, total_docs * 0.5))
    n_candidates = max(n_candidates, 1)

    # Seeded from the domain's fixed position in DOMAINS. Python's hash() is
    # salted per process, so it must not be used for anything reproducible.
    rng = np.random.default_rng([HOLDOUT_SEED, DOMAINS.index(domain)])
    picked = rng.choice(total_docs, size=n_candidates, replace=False)
    return set(int(i) for i in picked)


def batched(iterator: Iterator[str], size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for item in iterator:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


_LOCAL_TOKENIZER: Optional[Tokenizer] = None
_LOCAL_EOT_ID: int = 0


def _encode_batch_local(texts: list[str]) -> np.ndarray:
    """Single-process encode, used for the small bounded holdout streams."""
    assert _LOCAL_TOKENIZER is not None
    encodings = _LOCAL_TOKENIZER.encode_batch(texts)
    out: list[int] = []
    for enc in encodings:
        out.extend(enc.ids)
        out.append(_LOCAL_EOT_ID)
    return np.asarray(out, dtype=DTYPE)


def build_holdout(
    domain: str, candidates: set[int]
) -> tuple[dict[str, int], set[int]]:
    """Tokenize candidate documents into val then test, in that order.

    Returns the token counts written and the set of document indices actually
    consumed. Candidates not consumed are released back to train by the caller.
    """
    counts: dict[str, int] = {}
    consumed: set[int] = set()

    targets = [("val", VAL_TOKENS), ("test", TEST_TOKENS)]
    target_idx = 0
    written = 0
    fh = (TOKENIZED_ROOT / f"{domain}_val.bin").open("wb")
    split_name = "val"

    batch: list[str] = []
    batch_idx: list[int] = []

    def flush() -> None:
        nonlocal written, fh, split_name, target_idx, batch, batch_idx
        if not batch:
            return
        arr = _encode_batch_local(batch)
        arr.tofile(fh)
        written += int(arr.size)
        consumed.update(batch_idx)
        batch = []
        batch_idx = []

    for i, text in enumerate(iter_domain_docs(domain)):
        if i not in candidates:
            continue
        if target_idx >= len(targets):
            break
        batch.append(text)
        batch_idx.append(i)
        if len(batch) >= HOLDOUT_BATCH_DOCS:
            flush()
            limit = targets[target_idx][1]
            if written >= limit:
                fh.close()
                counts[split_name] = written
                print(f"       {domain} {split_name}: {written:,} tokens", flush=True)
                target_idx += 1
                written = 0
                if target_idx < len(targets):
                    split_name = targets[target_idx][0]
                    fh = (TOKENIZED_ROOT / f"{domain}_{split_name}.bin").open("wb")

    if target_idx < len(targets):
        flush()
        fh.close()
        counts[split_name] = written
        print(f"       {domain} {split_name}: {written:,} tokens", flush=True)
        target_idx += 1
        # If a target was never filled, the remaining splits are empty files.
        while target_idx < len(targets):
            name = targets[target_idx][0]
            (TOKENIZED_ROOT / f"{domain}_{name}.bin").open("wb").close()
            counts[name] = 0
            print(f"       {domain} {name}: 0 tokens (candidate pool exhausted)", flush=True)
            target_idx += 1
    else:
        if not fh.closed:
            fh.close()

    return counts, consumed


def build_train(pool: mp.pool.Pool, domain: str, consumed: set[int]) -> int:
    """Tokenize every document not consumed by val or test into train."""
    out_path = TOKENIZED_ROOT / f"{domain}_train.bin"

    def train_docs() -> Iterator[str]:
        for i, text in enumerate(iter_domain_docs(domain)):
            if i in consumed:
                continue
            yield text

    written = 0
    with out_path.open("wb") as fh:
        batches = batched(train_docs(), DOCS_PER_BATCH)
        next_report = 50_000_000
        for arr in pool.imap(_encode_batch, batches, chunksize=1):
            arr.tofile(fh)
            written += int(arr.size)
            if written >= next_report:
                print(f"       {domain} train: {written/1e6:.0f} M tokens", flush=True)
                next_report += 50_000_000
    print(f"       {domain} train: {written:,} tokens", flush=True)
    return written


def ext_shard_path(out_domain: str, index: int) -> Path:
    # Zero padded so a lexical listing is the write order the loader reads in.
    return TOKENIZED_ROOT / f"{out_domain}_train_ext_{index:03d}.bin"


def stream_encoded_to_shards(
    encoded: Iterable[np.ndarray],
    out_domain: str,
    shard_max_tokens: int = SHARD_MAX_TOKENS,
) -> tuple[int, list[dict]]:
    """Write already-encoded token arrays into rolling train-ext shard files.

    Rolls to a new shard once the current one would pass shard_max_tokens, so
    each shard is at most one encoded batch over the target. Returns the total
    token count and one record per shard. The caller guarantees the encoded
    arrays already carry the per-document EOT separators, exactly as the base
    train stream does.
    """
    shard_idx = 0
    shard_tokens = 0
    total = 0
    path = ext_shard_path(out_domain, shard_idx)
    fh = path.open("wb")
    shards: list[dict] = [{"file": path.name, "tokens": 0}]
    try:
        for arr in encoded:
            if shard_tokens > 0 and shard_tokens + int(arr.size) > shard_max_tokens:
                fh.close()
                shards[-1]["tokens"] = shard_tokens
                shard_idx += 1
                shard_tokens = 0
                path = ext_shard_path(out_domain, shard_idx)
                fh = path.open("wb")
                shards.append({"file": path.name, "tokens": 0})
            arr.tofile(fh)
            shard_tokens += int(arr.size)
            total += int(arr.size)
    finally:
        fh.close()
    shards[-1]["tokens"] = shard_tokens
    return total, shards


def tokenize_extension(
    tokenizer: Tokenizer,
    eot_id: int,
    pool: Optional[mp.pool.Pool] = None,
    shard_max_tokens: int = SHARD_MAX_TOKENS,
) -> dict:
    """Tokenize the fineweb_edu_ext raw domain into train-ext shards.

    All extension documents go to train. val and test were fixed from the base
    corpus and are left closed and untouched, and meta.json is not rewritten: the
    extension records its own meta file. Refuses to run if extension shards
    already exist, so a rerun cannot silently clobber or double-append.
    """
    global _LOCAL_TOKENIZER, _LOCAL_EOT_ID
    _LOCAL_TOKENIZER = tokenizer
    _LOCAL_EOT_ID = eot_id

    existing = sorted(TOKENIZED_ROOT.glob(f"{EXT_OUT_DOMAIN}_train_ext_*.bin"))
    if existing:
        raise SystemExit(
            f"extension shards already exist ({[p.name for p in existing]}); "
            "remove them or move them aside before re-tokenizing the extension"
        )

    manifest = load_manifest(EXT_RAW_DOMAIN)
    print(f"[run ] {EXT_RAW_DOMAIN}: {manifest['documents']:,} docs -> train-ext shards")

    batches = batched(iter_domain_docs(EXT_RAW_DOMAIN), DOCS_PER_BATCH)
    if pool is not None:
        encoded = pool.imap(_encode_batch, batches, chunksize=1)
    else:
        encoded = map(_encode_batch_local, batches)

    total, shards = stream_encoded_to_shards(encoded, EXT_OUT_DOMAIN, shard_max_tokens)

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
        "note": (
            "Additional train-only web shards. The base train/val/test streams and "
            "meta.json are unchanged. Run scripts/12_leakage_check.py on these "
            "shards against the existing val and test before training on them."
        ),
    }
    with (TOKENIZED_ROOT / EXT_META_NAME).open("w") as f:
        json.dump(ext_meta, f, indent=2)

    print(f"       extension: {total:,} tokens across {len(shards)} shards")
    for shard in shards:
        print(f"         {shard['file']}: {shard['tokens']:,} tokens")
    print(f"       wrote {TOKENIZED_ROOT / EXT_META_NAME}")
    return ext_meta


def run_extension() -> None:
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"missing {TOKENIZER_PATH}; run scripts/03_train_tokenizer.py first"
        )
    TOKENIZED_ROOT.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    vocab_size = tokenizer.get_vocab_size()
    eot_id = tokenizer.token_to_id(EOT_TOKEN)
    if eot_id is None:
        raise SystemExit(f"tokenizer has no {EOT_TOKEN} token")
    assert vocab_size < 65536, f"vocab {vocab_size} does not fit uint16"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print(f"tokenizer: {TOKENIZER_PATH} (vocab {vocab_size}, {EOT_TOKEN} id {eot_id})")
    print(f"output:    {TOKENIZED_ROOT}")
    print(f"workers:   {N_WORKERS}\n")

    pool = mp.Pool(
        processes=N_WORKERS,
        initializer=_worker_init,
        initargs=(str(TOKENIZER_PATH), eot_id),
    )
    try:
        tokenize_extension(tokenizer, eot_id, pool=pool)
    finally:
        pool.close()
        pool.join()


def main() -> None:
    global _LOCAL_TOKENIZER, _LOCAL_EOT_ID

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--extra-web",
        action="store_true",
        help=(
            "tokenize the fineweb_edu_ext raw domain into additional "
            "fineweb_edu_train_ext_*.bin shards, leaving the base train/val/test "
            "streams and meta.json untouched"
        ),
    )
    args = parser.parse_args()
    if args.extra_web:
        run_extension()
        return

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"missing {TOKENIZER_PATH}; run scripts/03_train_tokenizer.py first"
        )
    TOKENIZED_ROOT.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    vocab_size = tokenizer.get_vocab_size()
    eot_id = tokenizer.token_to_id(EOT_TOKEN)
    if eot_id is None:
        raise SystemExit(f"tokenizer has no {EOT_TOKEN} token")
    # uint16 storage is only exact while the vocabulary fits in 16 bits.
    assert vocab_size < 65536, f"vocab {vocab_size} does not fit uint16"
    assert np.dtype(DTYPE).itemsize == 2

    _LOCAL_TOKENIZER = tokenizer
    _LOCAL_EOT_ID = eot_id

    tokenizer_sha = hashlib.sha256(TOKENIZER_PATH.read_bytes()).hexdigest()
    print(f"tokenizer: {TOKENIZER_PATH}")
    print(f"  vocab {vocab_size}, {EOT_TOKEN} id {eot_id}, sha256 {tokenizer_sha[:16]}...")
    print(f"output:    {TOKENIZED_ROOT}")
    print(f"workers:   {N_WORKERS}\n")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    domains_meta: dict[str, dict] = {}
    pool = mp.Pool(
        processes=N_WORKERS,
        initializer=_worker_init,
        initargs=(str(TOKENIZER_PATH), eot_id),
    )
    try:
        for domain in DOMAINS:
            manifest = load_manifest(domain)
            print(f"[run ] {domain}: {manifest['documents']:,} docs")

            candidates = choose_candidates(domain)
            print(f"       holdout candidate pool: {len(candidates):,} docs")

            counts, consumed = build_holdout(domain, candidates)
            released = len(candidates) - len(consumed)
            print(
                f"       holdout consumed {len(consumed):,} docs; "
                f"{released:,} candidates released back to train"
            )

            train_tokens = build_train(pool, domain, consumed)

            domains_meta[domain] = {
                "documents_total": manifest["documents"],
                "documents_holdout": len(consumed),
                "documents_train": manifest["documents"] - len(consumed),
                "chars_raw": manifest["chars"],
                "tokens": {
                    "train": train_tokens,
                    "val": counts.get("val", 0),
                    "test": counts.get("test", 0),
                },
                "chars_per_token": manifest["chars"]
                / max(train_tokens + counts.get("val", 0) + counts.get("test", 0), 1),
                "provenance": manifest["provenance"],
            }
    finally:
        pool.close()
        pool.join()

    meta = {
        "created": datetime.now(timezone.utc).isoformat(),
        "tokenizer": {
            "path": str(TOKENIZER_PATH.relative_to(REPO_ROOT)),
            "sha256": tokenizer_sha,
            "vocab_size": vocab_size,
            "eot_token": EOT_TOKEN,
            "eot_id": eot_id,
        },
        "dtype": "uint16",
        "context_len_target": 512,
        "split_protocol": {
            "unit": "document",
            "seed": HOLDOUT_SEED,
            "val_tokens_target": VAL_TOKENS,
            "test_tokens_target": TEST_TOKENS,
            "note": (
                "Holdout documents are selected and tokenized before the train "
                "stream is built. A document belongs to exactly one split, so no "
                "text is shared between train and val or test."
            ),
        },
        "domains": domains_meta,
    }
    with META_PATH.open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'domain':<14} {'train':>14} {'val':>10} {'test':>10} {'chars/token':>12}")
    print("-" * 66)
    grand = 0
    for domain, m in domains_meta.items():
        t = m["tokens"]
        grand += t["train"] + t["val"] + t["test"]
        print(
            f"{domain:<14} {t['train']:>14,} {t['val']:>10,} {t['test']:>10,} "
            f"{m['chars_per_token']:>12.3f}"
        )
    print("-" * 66)
    print(f"total tokens: {grand:,}")
    print(f"\nWrote {META_PATH}")


if __name__ == "__main__":
    main()
