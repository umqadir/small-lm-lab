"""Tests for the document-level, token-space corpus decontamination.

No network and no real corpus. The n-gram hashing is tested on synthetic arrays,
and the end-to-end filter on a tiny synthetic holdout plus a tiny raw extension:
a document that repeats a holdout document verbatim must be dropped, a disjoint
document must survive, and the registered leakage check must then pass on the
written shards. All CPU, no lock.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
DTYPE = np.uint16


def load_script(name: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------
# n-gram hashing
# ----------------------------------------------------------------------------

def test_ngram_hashes_equal_windows_get_equal_hashes() -> None:
    filt = load_script("13_filter_ext.py", "filt_script_a")
    # Two identical length-4 windows at different positions must hash the same,
    # and a window that differs in one token must (almost surely) not.
    tokens = np.array([5, 6, 7, 8, 99, 5, 6, 7, 8], dtype=DTYPE)
    h = filt.ngram_hashes(tokens, 4)
    assert h[0] == h[5]  # [5,6,7,8] appears at 0 and at 5
    assert h[0] != h[1]  # [5,6,7,8] vs [6,7,8,99]


def test_ngram_hashes_empty_when_shorter_than_n() -> None:
    filt = load_script("13_filter_ext.py", "filt_script_b")
    assert filt.ngram_hashes(np.array([1, 2, 3], dtype=DTYPE), 4).size == 0


def test_chunked_hashing_equals_unchunked() -> None:
    filt = load_script("13_filter_ext.py", "filt_script_c")
    rng = np.random.default_rng(0)
    stream = rng.integers(0, 16384, size=5000, dtype=DTYPE)
    n = 8
    full = filt.ngram_hashes(stream, n)
    # A small chunk forces many overlaps; the concatenation must still equal the
    # unchunked hashes exactly, every window emitted once and only once.
    chunked = np.concatenate(
        list(filt.iter_ngram_hashes_chunked(stream, n, chunk_tokens=100))
    )
    assert chunked.shape == full.shape
    assert np.array_equal(chunked, full)


def test_contains_holdout_ngram() -> None:
    filt = load_script("13_filter_ext.py", "filt_script_d")
    n = 5
    holdout_stream = np.array([10, 11, 12, 13, 14, 15, 16], dtype=DTYPE)
    holdout_sorted = np.unique(filt.ngram_hashes(holdout_stream, n))
    powers = filt.ngram_powers(n)
    # A document that contains [12,13,14,15,16] (a holdout window) is flagged.
    leaky = np.array([1, 2, 12, 13, 14, 15, 16, 3], dtype=DTYPE)
    assert filt.contains_holdout_ngram(leaky, holdout_sorted, n, powers)
    # A disjoint document is not.
    clean = np.array([1, 2, 3, 4, 5, 6, 7], dtype=DTYPE)
    assert not filt.contains_holdout_ngram(clean, holdout_sorted, n, powers)
    # A document shorter than n has no n-gram, so it cannot contain a holdout one.
    short = np.array([12, 13, 14], dtype=DTYPE)
    assert not filt.contains_holdout_ngram(short, holdout_sorted, n, powers)


# ----------------------------------------------------------------------------
# end-to-end filter on a tiny synthetic corpus
# ----------------------------------------------------------------------------

def _tok():
    filt = load_script("13_filter_ext.py", "filt_e2e_mod")
    tokenizer = Tokenizer.from_file(str(filt.TOKENIZER_PATH))
    eot_id = tokenizer.token_to_id(filt.EOT_TOKEN)
    return filt, tokenizer, eot_id


def _write_bin(path: Path, tokens: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens.astype(DTYPE).tofile(path)


def test_filter_drops_the_leaky_doc_keeps_the_clean_and_passes_the_gate(
    tmp_path: Path,
) -> None:
    filt, tokenizer, eot_id = _tok()
    tok_root = tmp_path / "tokenized"
    raw_root = tmp_path / "raw"
    tok_root.mkdir(parents=True)

    # The two held-out documents, as text, tokenized straight into the closed
    # val and test bins. A leaky extension doc repeats the val doc verbatim, so
    # its tokenization contains the val token n-grams and it must be dropped.
    val_text = (
        "the lighthouse keeper counted seven ships before the fog rolled in "
        "across the northern bay that cold grey autumn morning"
    )
    test_text = (
        "quantum entanglement links two distant particles so that measuring one "
        "instantly constrains the state of the other however far apart they are"
    )
    val_ids = np.asarray(tokenizer.encode(val_text).ids, dtype=DTYPE)
    test_ids = np.asarray(tokenizer.encode(test_text).ids, dtype=DTYPE)
    _write_bin(tok_root / "fineweb_edu_val.bin", val_ids)
    _write_bin(tok_root / "fineweb_edu_test.bin", test_ids)
    base_val_bytes = (tok_root / "fineweb_edu_val.bin").read_bytes()
    base_test_bytes = (tok_root / "fineweb_edu_test.bin").read_bytes()

    # A tiny raw extension: two clean docs, one verbatim copy of the val doc
    # (the leak), and one short clean doc.
    ext_dir = raw_root / "fineweb_edu_ext"
    ext_dir.mkdir(parents=True)
    clean_a = "penguins huddle together on the antarctic ice to share warmth through the polar night"
    clean_b = "the baker proofed the sourdough overnight then scored the loaf and slid it into the oven"
    short_clean = "a brief note"
    docs = [clean_a, val_text, clean_b, short_clean]  # doc index 1 is the leak
    with (ext_dir / "shard_00000.jsonl").open("w") as f:
        for d in docs:
            f.write(json.dumps({"text": d}) + "\n")
    (ext_dir / "manifest.json").write_text(
        json.dumps(
            {
                "documents": len(docs),
                "shards": ["shard_00000.jsonl"],
                "provenance": {"extends": "fineweb_edu", "repo": "synthetic"},
            }
        )
    )

    # A small n so a sentence-length overlap is enough to flag the leak.
    n = 8
    ext_meta = filt.filter_extension(
        tokenizer,
        eot_id,
        n=n,
        workers=1,
        raw_root=raw_root,
        tokenized_root=tok_root,
        shard_max_tokens=10**9,
        verify=True,
    )

    decon = ext_meta["decontamination"]
    assert decon["documents_in"] == 4
    assert decon["documents_dropped"] >= 1  # the verbatim val copy
    assert decon["self_verified_residual_ngrams"] == 0

    # The verbatim val copy is gone: no shard contains the full val id sequence.
    shard_files = sorted(tok_root.glob("fineweb_edu_train_ext_*.bin"))
    assert shard_files, "no extension shard written"
    train = np.concatenate([np.fromfile(p, dtype=DTYPE) for p in shard_files])
    assert not filt.contains_holdout_ngram(
        train, np.unique(filt.ngram_hashes(val_ids, n)), n, filt.ngram_powers(n)
    )
    # The clean documents survived: kept count is in-minus-dropped, at least the
    # three non-leak docs (the short one included).
    assert decon["documents_kept"] == 4 - decon["documents_dropped"]
    assert decon["documents_kept"] >= 3

    # The closed val and test bins were not touched.
    assert (tok_root / "fineweb_edu_val.bin").read_bytes() == base_val_bytes
    assert (tok_root / "fineweb_edu_test.bin").read_bytes() == base_test_bytes

    # The registered leakage gate passes on the written shards.
    leak = load_script("12_leakage_check.py", "leak_for_filt")
    report = leak.check_leakage(
        [train.astype(np.int64)],
        {
            "val": val_ids.astype(np.int64),
            "test": test_ids.astype(np.int64),
        },
        domain="fineweb_edu",
        train_shard_names=[p.name for p in shard_files],
        n=n,
        samples=16,
    )
    assert report.passed
    for s in report.splits:
        assert s.n_leaks == 0
        assert s.positive_control_passed


def test_filter_refuses_to_clobber_existing_shards(tmp_path: Path) -> None:
    filt, tokenizer, eot_id = _tok()
    tok_root = tmp_path / "tokenized"
    raw_root = tmp_path / "raw"
    tok_root.mkdir(parents=True)
    (raw_root / "fineweb_edu_ext").mkdir(parents=True)
    _write_bin(tok_root / "fineweb_edu_val.bin", np.arange(50, dtype=DTYPE))
    _write_bin(tok_root / "fineweb_edu_test.bin", np.arange(50, dtype=DTYPE))
    (tok_root / "fineweb_edu_train_ext_000.bin").write_bytes(b"\x00\x00")

    with pytest.raises(SystemExit, match="already exist"):
        filt.filter_extension(
            tokenizer, eot_id, n=8, workers=1, raw_root=raw_root, tokenized_root=tok_root
        )
