"""Tests for corpus-expansion download, tokenize, and leakage-check helpers.

No network and no real corpus: the download extension is tested on a synthetic
document iterable, the tokenize extension on a tiny synthetic raw domain, and the
leakage check on tiny synthetic shards. All CPU, no lock.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DTYPE = np.uint16


def load_script(name: str, module_name: str):
    """Import a numbered script by path, registered so its dataclasses build."""
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------
# 02_download: the extension continues the stream past the prefix, no repeats
# ----------------------------------------------------------------------------

def test_extension_skips_the_prefix_and_collects_the_next_docs() -> None:
    download = load_script("02_download.py", "download_script")
    docs = [f"document number {i} with unique words {i}{i}{i}" for i in range(20)]

    # The base prefix was the first 5 non-empty docs. The extension skips exactly
    # those, so it starts at doc 5 and never repeats a prefix document. Docs are
    # stripped on the way through, as the base download strips them.
    skip_docs = 5
    char_target = 1  # tiny: one doc is enough to pass the target
    got = list(download.collect_extension_docs(docs, skip_docs, char_target))
    assert got[0] == docs[5].strip()
    stripped_prefix = {d.strip() for d in docs[:skip_docs]}
    assert all(d not in stripped_prefix for d in got)


def test_extension_applies_the_same_empty_filter_as_the_base() -> None:
    """The base download counts only non-empty docs, so the extension must skip
    non-empty docs, or empty rows would shift the boundary and cause a repeat."""
    download = load_script("02_download.py", "download_script")
    docs = ["a" * 50, "", "  ", "b" * 50, "", "c" * 50, "d" * 50]
    # Two non-empty docs make the prefix (a, b). Skipping must land on c, not be
    # thrown off by the empty rows in between.
    got = list(download.collect_extension_docs(docs, skip_docs=2, char_target=1))
    assert got[0] == "c" * 50


# ----------------------------------------------------------------------------
# 04_tokenize: encoded arrays roll into train-ext shards
# ----------------------------------------------------------------------------

def test_stream_encoded_to_shards_rolls_and_concatenates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenize = load_script("04_tokenize.py", "tokenize_script")
    monkeypatch.setattr(tokenize, "TOKENIZED_ROOT", tmp_path)

    # Five encoded batches; a small shard cap forces a rollover.
    batches = [np.arange(i * 10, i * 10 + 10, dtype=DTYPE) for i in range(5)]
    total, shards = tokenize.stream_encoded_to_shards(
        (b for b in batches), out_domain="fineweb_edu", shard_max_tokens=25
    )
    assert total == 50
    # 25-token cap over 10-token batches: a shard rolls once the next batch would
    # pass 25, so shards fill to 20 then roll -> 20, 20, 10.
    assert [s["tokens"] for s in shards] == [20, 20, 10]
    names = [s["file"] for s in shards]
    assert names == [
        "fineweb_edu_train_ext_000.bin",
        "fineweb_edu_train_ext_001.bin",
        "fineweb_edu_train_ext_002.bin",
    ]

    # Concatenating the shards back reproduces the encoded stream exactly.
    written = np.concatenate([np.fromfile(tmp_path / n, dtype=DTYPE) for n in names])
    assert np.array_equal(written, np.concatenate(batches))


def test_tokenize_extension_writes_shards_and_meta_without_touching_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The extension tokenizes fineweb_edu_ext raw docs into train-ext shards,
    writes its own meta, and leaves the base bins and meta.json alone."""
    from tokenizers import Tokenizer

    tokenize = load_script("04_tokenize.py", "tokenize_script2")
    raw_root = tmp_path / "raw"
    tok_root = tmp_path / "tokenized"
    tok_root.mkdir(parents=True)
    monkeypatch.setattr(tokenize, "RAW_ROOT", raw_root)
    monkeypatch.setattr(tokenize, "TOKENIZED_ROOT", tok_root)

    # A base train bin and meta.json that must survive untouched.
    base_train = tok_root / "fineweb_edu_train.bin"
    np.arange(100, 200, dtype=DTYPE).tofile(base_train)
    base_sha_before = base_train.read_bytes()
    meta_json = tok_root / "meta.json"
    meta_json.write_text(json.dumps({"sentinel": True}))

    # A tiny fineweb_edu_ext raw domain: one shard of two documents.
    ext_dir = raw_root / "fineweb_edu_ext"
    ext_dir.mkdir(parents=True)
    (ext_dir / "shard_00000.jsonl").write_text(
        json.dumps({"text": "the quick brown fox"}) + "\n"
        + json.dumps({"text": "jumps over the lazy dog"}) + "\n"
    )
    (ext_dir / "manifest.json").write_text(
        json.dumps({"documents": 2, "shards": ["shard_00000.jsonl"], "provenance": {"extends": "fineweb_edu"}})
    )

    tokenizer = Tokenizer.from_file(str(tokenize.TOKENIZER_PATH))
    eot_id = tokenizer.token_to_id(tokenize.EOT_TOKEN)
    ext_meta = tokenize.tokenize_extension(tokenizer, eot_id, pool=None, shard_max_tokens=10**9)

    # The extension wrote at least one shard, and the meta records the total.
    shard_files = sorted(tok_root.glob("fineweb_edu_train_ext_*.bin"))
    assert shard_files, "no extension shard written"
    assert ext_meta["tokens"] > 0
    assert (tok_root / tokenize.EXT_META_NAME).exists()

    # Each document is EOT-terminated, so the number of EOT ids equals the
    # document count, exactly like the base train stream.
    tokens = np.concatenate([np.fromfile(p, dtype=DTYPE) for p in shard_files])
    assert int((tokens == eot_id).sum()) == 2

    # The base bin and meta.json are byte-for-byte unchanged.
    assert base_train.read_bytes() == base_sha_before
    assert json.loads(meta_json.read_text()) == {"sentinel": True}


def test_tokenize_extension_refuses_to_clobber_existing_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tokenizers import Tokenizer

    tokenize = load_script("04_tokenize.py", "tokenize_script3")
    raw_root = tmp_path / "raw"
    tok_root = tmp_path / "tokenized"
    tok_root.mkdir(parents=True)
    (raw_root / "fineweb_edu_ext").mkdir(parents=True)
    monkeypatch.setattr(tokenize, "RAW_ROOT", raw_root)
    monkeypatch.setattr(tokenize, "TOKENIZED_ROOT", tok_root)
    (tok_root / "fineweb_edu_train_ext_000.bin").write_bytes(b"\x00\x00")

    tokenizer = Tokenizer.from_file(str(tokenize.TOKENIZER_PATH))
    eot_id = tokenizer.token_to_id(tokenize.EOT_TOKEN)
    with pytest.raises(SystemExit, match="already exist"):
        tokenize.tokenize_extension(tokenizer, eot_id, pool=None)


# ----------------------------------------------------------------------------
# 12_leakage_check: new train shards against existing val/test
# ----------------------------------------------------------------------------

def write_bin(path: Path, tokens: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens.astype(DTYPE).tofile(path)


def test_leakage_check_clean_when_train_is_disjoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    leak = load_script("12_leakage_check.py", "leak_script")
    rng = np.random.default_rng(0)
    val = rng.integers(1, 5000, size=2000, dtype=DTYPE)
    test = rng.integers(5000, 9000, size=2000, dtype=DTYPE)
    # Train ids are in a range that shares no value with val or test, so no
    # window can possibly overlap.
    train = rng.integers(9000, 12000, size=4000, dtype=DTYPE)

    report = leak.check_leakage(
        [train.astype(np.int64)],
        {"val": val.astype(np.int64), "test": test.astype(np.int64)},
        domain="fineweb_edu",
        train_shard_names=["ext_000"],
        n=8,
        samples=32,
    )
    assert report.passed
    for s in report.splits:
        assert s.n_leaks == 0
        # The positive control found every sampled n-gram in its own split.
        assert s.positive_control_passed


def test_leakage_check_flags_an_injected_leak(tmp_path: Path) -> None:
    leak = load_script("12_leakage_check.py", "leak_script2")
    rng = np.random.default_rng(1)
    val = rng.integers(1, 5000, size=2000, dtype=DTYPE).astype(np.int64)
    test = rng.integers(5000, 9000, size=2000, dtype=DTYPE).astype(np.int64)
    # Splice the entire val stream verbatim into train, between disjoint random
    # regions. Every sampled val n-gram is then present in train, so the leak is
    # caught whichever positions the sampler happens to draw. test stays disjoint.
    pad_a = rng.integers(9000, 12000, size=1000, dtype=DTYPE).astype(np.int64)
    pad_b = rng.integers(9000, 12000, size=1000, dtype=DTYPE).astype(np.int64)
    train = np.concatenate([pad_a, val, pad_b])

    report = leak.check_leakage(
        [train], {"val": val, "test": test},
        domain="fineweb_edu", train_shard_names=["ext_000"], n=8, samples=32,
    )
    assert not report.passed
    val_report = next(s for s in report.splits if s.split == "val")
    test_report = next(s for s in report.splits if s.split == "test")
    assert val_report.n_leaks == val_report.n_samples  # every sample leaked
    assert test_report.n_leaks == 0  # test is disjoint


def test_leakage_positive_control_catches_a_broken_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the search machinery cannot find a present n-gram, a zero-leak result
    is meaningless. The control is what turns that into a reported failure."""
    leak = load_script("12_leakage_check.py", "leak_script3")
    monkeypatch.setattr(leak, "ngram_in_stream", lambda ngram, stream: False)
    rng = np.random.default_rng(2)
    val = rng.integers(1, 5000, size=1000, dtype=DTYPE).astype(np.int64)
    report = leak.check_leakage(
        [rng.integers(9000, 12000, size=1000, dtype=DTYPE).astype(np.int64)],
        {"val": val},
        domain="fineweb_edu", train_shard_names=["ext_000"], n=8, samples=16,
    )
    assert not report.passed
    assert not report.splits[0].positive_control_passed


def test_ngram_search_is_exact_not_a_prefix_match() -> None:
    leak = load_script("12_leakage_check.py", "leak_script4")
    stream = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
    assert leak.ngram_in_stream(np.array([3, 4, 5], dtype=np.int64), stream)
    # Same first token, different tail: not a match.
    assert not leak.ngram_in_stream(np.array([3, 9, 9], dtype=np.int64), stream)
    # Absent entirely.
    assert not leak.ngram_in_stream(np.array([100, 101], dtype=np.int64), stream)
