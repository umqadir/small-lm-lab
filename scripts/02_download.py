"""Download the raw corpora to the bulk artifact root as jsonl shards.

Two domains, both license-verified and non-gated:

  tinystories: roneneldan/TinyStories, CDLA-Sharing-1.0. The V2 GPT-4 variant
    is used specifically (TinyStoriesV2-GPT4-train.txt and
    TinyStoriesV2-GPT4-valid.txt), which is the GPT-4-generated rewrite of the
    original GPT-3.5 corpus. Those files are plain text with documents separated
    by a line containing the literal marker <|endoftext|>. They are not exposed
    as a datasets config (the parquet config in data/ is the V1 corpus), so they
    are fetched with hf_hub_download from the same dataset repo. If they ever
    become unavailable, FALLBACK_TO_PARQUET loads the standard dataset config
    instead and the manifest records which path was taken.

  fineweb_edu: HuggingFaceFW/fineweb-edu, config sample-10BT, ODC-By 1.0.
    Streamed and stopped early. About 150M tokens are needed, so the stream
    pulls roughly FINEWEB_CHAR_TARGET characters (about 4 chars per token) and
    stops.
    The full 10BT sample is never downloaded.

Both domains are normalized to the same on-disk form: jsonl shards of
{"text": ...} records, one document per line, capped at SHARD_MAX_BYTES.

Resumable: a domain with a complete manifest.json is skipped entirely. A domain
interrupted partway through is restarted from scratch (partial shards are
removed), since a half-written stream has no reliable resume point.

Bulk output goes only to the bulk artifact root. The HuggingFace cache is
redirected there too so that no bulk data lands on the boot disk.

Corpus expansion (--extra-web-tokens N): the staged deep run needs more web data
than the original 168M-token slice, so web data never repeats within a run of
several billion tokens. The FineWeb-Edu stream is deterministic in dataset
order, and the original download is a prefix of it, so an expansion continues
that stream: it skips exactly the documents the original prefix consumed and
collects the next N tokens' worth into a separate raw domain (fineweb_edu_ext),
leaving the original fineweb_edu raw shards and manifest untouched. TinyStories
cannot be expanded (its whole corpus is already downloaded), so only web is.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from small_lm_lab.paths import DATA_ROOT, HF_CACHE, RAW_ROOT

# Redirect the HuggingFace cache to the bulk root before datasets/huggingface_hub
# load.
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE / "datasets"))

from datasets import load_dataset  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

TINYSTORIES_REPO = "roneneldan/TinyStories"
TINYSTORIES_LICENSE = "CDLA-Sharing-1.0"
TINYSTORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories"
# The V2 GPT-4 files. Both are pooled into one document pool; the train/val/test
# split is made by document in 04_tokenize.py, uniformly across domains.
TINYSTORIES_V2_FILES = [
    "TinyStoriesV2-GPT4-train.txt",
    "TinyStoriesV2-GPT4-valid.txt",
]
TINYSTORIES_DOC_SEPARATOR = "<|endoftext|>"
FALLBACK_TO_PARQUET = True

FINEWEB_REPO = "HuggingFaceFW/fineweb-edu"
FINEWEB_CONFIG = "sample-10BT"
FINEWEB_LICENSE = "ODC-By 1.0"
FINEWEB_URL = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu"
# About 4 chars per token, target ~150M tokens, so ~600M chars. The target is
# 750M, which leaves headroom for a worse chars-per-token ratio than estimated.
FINEWEB_CHAR_TARGET = 750_000_000

SHARD_MAX_BYTES = 256 * 1024 * 1024
PROGRESS_EVERY = 200_000

# Corpus expansion. The extension is a separate raw domain so the original
# fineweb_edu raw shards are never rewritten.
FINEWEB_EXT_DOMAIN = "fineweb_edu_ext"
# Measured chars per token for FineWeb-Edu (docs/DATA.md), with headroom, so a token
# target is turned into a character target the stream can stop on. Tokenization
# in 04_tokenize.py reports the exact realized token count.
FINEWEB_CHARS_PER_TOKEN = 4.21
EXTRA_CHAR_HEADROOM = 1.05
# A smoke run proves the path on a few MB rather than billions of tokens.
SMOKE_CHAR_TARGET = 4 * 1024 * 1024


@dataclass
class ShardWriter:
    """Writes {"text": ...} jsonl records, rolling over at SHARD_MAX_BYTES."""

    out_dir: Path
    max_bytes: int = SHARD_MAX_BYTES

    def __post_init__(self) -> None:
        self.shard_index = 0
        self.shard_bytes = 0
        self.total_bytes = 0
        self.total_docs = 0
        self.total_chars = 0
        self.paths: list[Path] = []
        self._fh = None
        self._open_shard()

    def _open_shard(self) -> None:
        path = self.out_dir / f"shard_{self.shard_index:05d}.jsonl"
        self._fh = path.open("w", encoding="utf-8")
        self.paths.append(path)
        self.shard_bytes = 0

    def write(self, text: str) -> None:
        line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
        encoded = len(line.encode("utf-8"))
        if self.shard_bytes > 0 and self.shard_bytes + encoded > self.max_bytes:
            self._fh.close()
            self.shard_index += 1
            self._open_shard()
        self._fh.write(line)
        self.shard_bytes += encoded
        self.total_bytes += encoded
        self.total_docs += 1
        self.total_chars += len(text)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def manifest_path(domain: str) -> Path:
    return RAW_ROOT / domain / "manifest.json"


def is_complete(domain: str) -> bool:
    return manifest_path(domain).exists()


def reset_domain(domain: str) -> None:
    """Clear any partial output so an interrupted domain restarts cleanly."""
    out_dir = RAW_ROOT / domain
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def write_manifest(domain: str, writer: ShardWriter, provenance: dict) -> dict:
    manifest = {
        "domain": domain,
        "documents": writer.total_docs,
        "bytes_jsonl": writer.total_bytes,
        "chars": writer.total_chars,
        "shards": [p.name for p in writer.paths],
        "provenance": provenance,
    }
    with manifest_path(domain).open("w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def iter_tinystories_docs(txt_path: Path) -> Iterator[str]:
    """Yield documents from a TinyStories V2 txt file.

    Documents are separated by a line holding the <|endoftext|> marker. The
    marker itself is not part of any document; 04_tokenize.py re-inserts the
    separator as a real token id.
    """
    buffer: list[str] = []
    with txt_path.open("r", encoding="utf-8", errors="strict") as f:
        for line in f:
            if line.strip() == TINYSTORIES_DOC_SEPARATOR:
                text = "".join(buffer).strip()
                if text:
                    yield text
                buffer = []
            else:
                buffer.append(line)
    text = "".join(buffer).strip()
    if text:
        yield text


def download_tinystories() -> dict:
    domain = "tinystories"
    if is_complete(domain):
        with manifest_path(domain).open() as f:
            manifest = json.load(f)
        print(f"[skip] {domain}: manifest present, {manifest['documents']:,} docs")
        return manifest

    print(f"[run ] {domain}: fetching V2 GPT-4 files from {TINYSTORIES_REPO}")
    reset_domain(domain)

    local_files: list[Path] = []
    source_mode = "v2_gpt4_txt"
    try:
        for name in TINYSTORIES_V2_FILES:
            path = Path(
                hf_hub_download(
                    repo_id=TINYSTORIES_REPO,
                    filename=name,
                    repo_type="dataset",
                    cache_dir=str(HF_CACHE / "hub"),
                )
            )
            print(f"       fetched {name}: {path.stat().st_size / 1e9:.2f} GB")
            local_files.append(path)
    except Exception as exc:  # noqa: BLE001 - fall back and record the reason
        if not FALLBACK_TO_PARQUET:
            raise
        print(f"       V2 txt files unavailable ({type(exc).__name__}: {exc})")
        print("       falling back to the standard parquet dataset config")
        source_mode = "parquet_config_fallback"
        local_files = []

    writer = ShardWriter(RAW_ROOT / domain)
    if source_mode == "v2_gpt4_txt":
        for path in local_files:
            for i, text in enumerate(iter_tinystories_docs(path)):
                writer.write(text)
                if writer.total_docs % PROGRESS_EVERY == 0:
                    print(f"       {writer.total_docs:,} docs, {writer.total_chars/1e9:.2f} G chars")
    else:
        dataset = load_dataset(TINYSTORIES_REPO, split="train")
        for row in dataset:
            text = (row["text"] or "").strip()
            if text:
                writer.write(text)
                if writer.total_docs % PROGRESS_EVERY == 0:
                    print(f"       {writer.total_docs:,} docs, {writer.total_chars/1e9:.2f} G chars")
    writer.close()

    provenance = {
        "repo": TINYSTORIES_REPO,
        "url": TINYSTORIES_URL,
        "license": TINYSTORIES_LICENSE,
        "source_mode": source_mode,
        "files": [p.name for p in local_files] if local_files else [f"{TINYSTORIES_REPO}:train"],
        "gated": False,
        "note": (
            "V2 GPT-4 variant. The shipped train and valid txt files are pooled into "
            "one document pool; the split used by this lab is made by document in "
            "04_tokenize.py."
        ),
    }
    manifest = write_manifest(domain, writer, provenance)
    print(
        f"       done: {manifest['documents']:,} docs, "
        f"{manifest['chars']:,} chars, {manifest['bytes_jsonl']:,} jsonl bytes, "
        f"{len(manifest['shards'])} shards"
    )
    return manifest


def download_fineweb_edu() -> dict:
    domain = "fineweb_edu"
    if is_complete(domain):
        with manifest_path(domain).open() as f:
            manifest = json.load(f)
        print(f"[skip] {domain}: manifest present, {manifest['documents']:,} docs")
        return manifest

    print(
        f"[run ] {domain}: streaming {FINEWEB_REPO} {FINEWEB_CONFIG} "
        f"until {FINEWEB_CHAR_TARGET:,} chars"
    )
    reset_domain(domain)

    dataset = load_dataset(FINEWEB_REPO, FINEWEB_CONFIG, split="train", streaming=True)
    writer = ShardWriter(RAW_ROOT / domain)
    for row in dataset:
        text = (row["text"] or "").strip()
        if not text:
            continue
        writer.write(text)
        if writer.total_docs % 50_000 == 0:
            pct = 100.0 * writer.total_chars / FINEWEB_CHAR_TARGET
            print(
                f"       {writer.total_docs:,} docs, "
                f"{writer.total_chars/1e6:.0f} M chars ({pct:.0f}% of target)"
            )
        if writer.total_chars >= FINEWEB_CHAR_TARGET:
            break
    writer.close()

    provenance = {
        "repo": FINEWEB_REPO,
        "url": FINEWEB_URL,
        "license": FINEWEB_LICENSE,
        "config": FINEWEB_CONFIG,
        "split": "train",
        "streaming": True,
        "char_target": FINEWEB_CHAR_TARGET,
        "gated": False,
        "note": (
            "Streamed in dataset order and stopped once the character target was "
            "reached. This is a prefix of the sample-10BT stream, not a random "
            "sample of it, and not the full sample."
        ),
    }
    manifest = write_manifest(domain, writer, provenance)
    print(
        f"       done: {manifest['documents']:,} docs, "
        f"{manifest['chars']:,} chars, {manifest['bytes_jsonl']:,} jsonl bytes, "
        f"{len(manifest['shards'])} shards"
    )
    return manifest


def collect_extension_docs(
    texts: Iterable[str], skip_docs: int, char_target: int
) -> Iterator[str]:
    """Skip the prefix the original download consumed, then yield the next docs.

    The original download wrote skip_docs non-empty documents and stopped, so the
    extension applies the identical non-empty filter, skips exactly that many, and
    yields the documents that follow until char_target characters are collected.
    Skipping by the prefix's document count rather than by characters is what
    makes the extension pick up exactly where the prefix stopped: no document is
    both in the prefix and in the extension, and none is missed between them.
    """
    skipped = 0
    collected = 0
    for raw in texts:
        text = (raw or "").strip()
        if not text:
            continue
        if skipped < skip_docs:
            skipped += 1
            continue
        yield text
        collected += len(text)
        if collected >= char_target:
            return


def download_fineweb_edu_extension(extra_web_tokens: int, smoke: bool = False) -> dict:
    domain = FINEWEB_EXT_DOMAIN
    if is_complete(domain):
        with manifest_path(domain).open() as f:
            manifest = json.load(f)
        print(f"[skip] {domain}: manifest present, {manifest['documents']:,} docs")
        return manifest

    # Where the original prefix stopped, in documents. The extension continues
    # from exactly there so web data never repeats within a run.
    base_manifest_path = manifest_path("fineweb_edu")
    if not base_manifest_path.exists():
        raise SystemExit(
            f"missing {base_manifest_path}; download the base fineweb_edu slice "
            "before extending it, so the extension knows where the prefix ended"
        )
    with base_manifest_path.open() as f:
        base_manifest = json.load(f)
    skip_docs = int(base_manifest["documents"])

    char_target = (
        SMOKE_CHAR_TARGET
        if smoke
        else int(extra_web_tokens * FINEWEB_CHARS_PER_TOKEN * EXTRA_CHAR_HEADROOM)
    )
    print(
        f"[run ] {domain}: skipping the {skip_docs:,}-document prefix, then "
        f"streaming {char_target:,} more chars"
        + (" (smoke)" if smoke else "")
    )
    reset_domain(domain)

    dataset = load_dataset(FINEWEB_REPO, FINEWEB_CONFIG, split="train", streaming=True)
    writer = ShardWriter(RAW_ROOT / domain)
    texts = (row["text"] for row in dataset)
    for text in collect_extension_docs(texts, skip_docs, char_target):
        writer.write(text)
        if writer.total_docs % 50_000 == 0:
            pct = 100.0 * writer.total_chars / char_target
            print(
                f"       {writer.total_docs:,} docs, "
                f"{writer.total_chars/1e6:.0f} M chars ({pct:.0f}% of target)"
            )
    writer.close()

    provenance = {
        "repo": FINEWEB_REPO,
        "url": FINEWEB_URL,
        "license": FINEWEB_LICENSE,
        "config": FINEWEB_CONFIG,
        "split": "train",
        "streaming": True,
        "extends": "fineweb_edu",
        "skip_documents": skip_docs,
        "char_target": char_target,
        "extra_web_tokens_requested": None if smoke else int(extra_web_tokens),
        "smoke": bool(smoke),
        "gated": False,
        "note": (
            "Continuation of the sample-10BT stream past the original prefix. The "
            "first skip_documents documents (the prefix the base download "
            "consumed) are skipped, so no document repeats between fineweb_edu and "
            "fineweb_edu_ext. Tokenized into train-only shards by "
            "scripts/04_tokenize.py --extra-web; val and test are not touched."
        ),
    }
    manifest = write_manifest(domain, writer, provenance)
    print(
        f"       done: {manifest['documents']:,} docs, "
        f"{manifest['chars']:,} chars, {manifest['bytes_jsonl']:,} jsonl bytes, "
        f"{len(manifest['shards'])} shards"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--extra-web-tokens",
        type=float,
        default=None,
        help=(
            "corpus expansion: fetch roughly this many additional FineWeb-Edu "
            "tokens beyond the original slice, continuing the stream past the "
            "prefix into the fineweb_edu_ext raw domain. Leaves the base "
            "fineweb_edu raw shards untouched. Without this flag the script "
            "downloads the two base domains as before"
        ),
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help=(
            f"with --extra-web-tokens, cap the extension at {SMOKE_CHAR_TARGET:,} "
            "chars to prove the path on a few MB rather than billions of tokens"
        ),
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.extra_web_tokens is not None or args.smoke:
        RAW_ROOT.mkdir(parents=True, exist_ok=True)
        HF_CACHE.mkdir(parents=True, exist_ok=True)
        print(f"raw root:  {RAW_ROOT}")
        print(f"hf cache:  {HF_CACHE}\n")
        manifest = download_fineweb_edu_extension(
            int(args.extra_web_tokens or 0), smoke=args.smoke
        )
        print(
            f"\nExtension raw ready: {manifest['documents']:,} docs, "
            f"{manifest['chars']:,} chars. Tokenize with "
            "scripts/04_tokenize.py --extra-web."
        )
        return

    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    print(f"raw root:  {RAW_ROOT}")
    print(f"hf cache:  {HF_CACHE}\n")

    manifests = [download_tinystories(), download_fineweb_edu()]

    print("\n" + f"{'domain':<14} {'documents':>12} {'chars':>16} {'jsonl bytes':>16} {'shards':>7}")
    print("-" * 70)
    total_docs = total_chars = total_bytes = 0
    for m in manifests:
        print(
            f"{m['domain']:<14} {m['documents']:>12,} {m['chars']:>16,} "
            f"{m['bytes_jsonl']:>16,} {len(m['shards']):>7}"
        )
        total_docs += m["documents"]
        total_chars += m["chars"]
        total_bytes += m["bytes_jsonl"]
    print("-" * 70)
    print(f"{'total':<14} {total_docs:>12,} {total_chars:>16,} {total_bytes:>16,}")
    print(f"\nEstimated tokens at 4 chars/token: {total_chars/4/1e6:.0f} M (verified in 04)")


if __name__ == "__main__":
    main()
