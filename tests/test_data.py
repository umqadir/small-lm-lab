"""Tests for the loader's corpus-expansion support.

A corpus expansion adds train shards alongside the base train stream. The loader
has to stitch them into one training stream while leaving val and test, the
closed evaluation splits, exactly one stream each. Nothing here touches the real
corpus: tiny synthetic uint16 shards are written into a tmp directory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from small_lm_lab.data import (
    ConcatTokens,
    DomainSource,
    MixtureBatcher,
    load_sources,
    open_split,
    open_train_tokens,
    train_ext_paths,
)

DTYPE = np.uint16


def write_bin(path: Path, tokens: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens.astype(DTYPE).tofile(path)


def memmap(path: Path) -> np.memmap:
    return np.memmap(path, dtype=DTYPE, mode="r")


# ----------------------------------------------------------------------------
# ConcatTokens
# ----------------------------------------------------------------------------

def test_concat_tokens_matches_a_plain_concatenation(tmp_path: Path) -> None:
    a = np.arange(0, 100, dtype=DTYPE)
    b = np.arange(1000, 1050, dtype=DTYPE)
    c = np.arange(5000, 5030, dtype=DTYPE)
    write_bin(tmp_path / "a.bin", a)
    write_bin(tmp_path / "b.bin", b)
    write_bin(tmp_path / "c.bin", c)
    reference = np.concatenate([a, b, c])

    concat = ConcatTokens([memmap(tmp_path / "a.bin"), memmap(tmp_path / "b.bin"), memmap(tmp_path / "c.bin")])
    assert concat.shape[0] == len(reference) == 180
    assert len(concat) == 180

    # The whole thing, and a scattering of windows that fall inside one shard,
    # span exactly one boundary, and span two boundaries at once.
    assert np.array_equal(np.asarray(concat[:]), reference)
    for lo, hi in [(0, 10), (95, 105), (98, 152), (90, 175), (149, 180), (179, 180)]:
        assert np.array_equal(np.asarray(concat[lo:hi]), reference[lo:hi]), (lo, hi)


def test_concat_tokens_within_a_shard_is_a_view_not_a_copy(tmp_path: Path) -> None:
    """A window inside one shard is returned as that shard's memmap slice, so the
    common case reads nothing extra into memory."""
    write_bin(tmp_path / "a.bin", np.arange(0, 100, dtype=DTYPE))
    write_bin(tmp_path / "b.bin", np.arange(1000, 1100, dtype=DTYPE))
    concat = ConcatTokens([memmap(tmp_path / "a.bin"), memmap(tmp_path / "b.bin")])
    window = concat[10:20]
    assert isinstance(window, np.memmap)
    boundary = concat[95:105]  # spans the seam, so it must be materialized
    assert not isinstance(boundary, np.memmap)


def test_concat_tokens_rejects_non_contiguous_indexing(tmp_path: Path) -> None:
    write_bin(tmp_path / "a.bin", np.arange(0, 10, dtype=DTYPE))
    concat = ConcatTokens([memmap(tmp_path / "a.bin")])
    with pytest.raises(ValueError, match="contiguous"):
        _ = concat[0:10:2]
    with pytest.raises(TypeError):
        _ = concat[5]


# ----------------------------------------------------------------------------
# open_train_tokens / load_sources
# ----------------------------------------------------------------------------

def test_no_extension_reads_the_bare_base_stream(tmp_path: Path) -> None:
    """A domain that was never expanded reads exactly as before: the bare base
    memmap, so an existing on-disk run is bit-identical."""
    base = np.arange(0, 200, dtype=DTYPE)
    write_bin(tmp_path / "fineweb_edu_train.bin", base)
    assert train_ext_paths("fineweb_edu", tmp_path) == []
    stream = open_train_tokens("fineweb_edu", tmp_path)
    assert isinstance(stream, np.memmap)
    assert np.array_equal(np.asarray(stream), base)


def test_extension_shards_extend_the_train_stream(tmp_path: Path) -> None:
    base = np.arange(0, 200, dtype=DTYPE)
    ext0 = np.arange(9000, 9100, dtype=DTYPE)
    ext1 = np.arange(9100, 9160, dtype=DTYPE)
    write_bin(tmp_path / "fineweb_edu_train.bin", base)
    write_bin(tmp_path / "fineweb_edu_train_ext_00.bin", ext0)
    write_bin(tmp_path / "fineweb_edu_train_ext_01.bin", ext1)

    # In write order (lexical is chronological because of the zero padding).
    assert [p.name for p in train_ext_paths("fineweb_edu", tmp_path)] == [
        "fineweb_edu_train_ext_00.bin",
        "fineweb_edu_train_ext_01.bin",
    ]
    stream = open_train_tokens("fineweb_edu", tmp_path)
    assert isinstance(stream, ConcatTokens)
    assert np.array_equal(np.asarray(stream[:]), np.concatenate([base, ext0, ext1]))


def test_load_sources_expands_train_but_never_val_or_test(tmp_path: Path) -> None:
    """The expansion reaches the train mix and nothing else. val and test are the
    closed evaluation splits, so their streams stay a single file each even when
    train has extra shards."""
    write_bin(tmp_path / "fineweb_edu_train.bin", np.arange(0, 200, dtype=DTYPE))
    write_bin(tmp_path / "fineweb_edu_train_ext_00.bin", np.arange(9000, 9500, dtype=DTYPE))
    write_bin(tmp_path / "fineweb_edu_val.bin", np.arange(0, 120, dtype=DTYPE))
    write_bin(tmp_path / "fineweb_edu_test.bin", np.arange(0, 130, dtype=DTYPE))
    write_bin(tmp_path / "tinystories_train.bin", np.arange(0, 300, dtype=DTYPE))
    write_bin(tmp_path / "tinystories_val.bin", np.arange(0, 110, dtype=DTYPE))

    mix = {"tinystories": 0.7, "fineweb_edu": 0.3}
    train = {s.name: s for s in load_sources(mix, split="train", root=tmp_path)}
    # fineweb_edu train grew by the extension; tinystories (no extension) did not.
    assert train["fineweb_edu"].n_tokens == 200 + 500
    assert isinstance(train["fineweb_edu"].tokens, ConcatTokens)
    assert train["tinystories"].n_tokens == 300
    assert isinstance(train["tinystories"].tokens, np.memmap)

    val = {s.name: s for s in load_sources(mix, split="val", root=tmp_path)}
    # val ignores the extension entirely: still the single val stream.
    assert val["fineweb_edu"].n_tokens == 120
    assert isinstance(val["fineweb_edu"].tokens, np.memmap)


def test_batcher_draws_windows_from_the_extension_region(tmp_path: Path) -> None:
    """A window can land in the extension shards, so the extra web tokens are
    actually trained on rather than merely present on disk.

    Base tokens are all 1 and extension tokens all 7, each region long enough
    that most windows fall cleanly inside one, so a run of 7s in a drawn window
    can only have come from the extension.
    """
    base = np.ones(4000, dtype=DTYPE)
    ext = np.full(4000, 7, dtype=DTYPE)
    write_bin(tmp_path / "fineweb_edu_train.bin", base)
    write_bin(tmp_path / "fineweb_edu_train_ext_00.bin", ext)

    stream = open_train_tokens("fineweb_edu", tmp_path)
    source = DomainSource(name="fineweb_edu", weight=1.0, tokens=stream)
    batcher = MixtureBatcher([source], batch_size=16, context_len=8, seed=0)

    saw_base = saw_ext = False
    for _ in range(50):
        x, y = batcher.next_batch()
        # Every window is a valid next-token pair over ids that exist in the
        # stream, whichever region it came from.
        assert x.shape == (16, 8) and y.shape == (16, 8)
        assert set(np.unique(x)).issubset({1, 7})
        for row in x:
            if np.all(row == 1):
                saw_base = True
            if np.all(row == 7):
                saw_ext = True
    assert saw_base, "no window came from the base stream"
    assert saw_ext, "no window came from the extension shards"
