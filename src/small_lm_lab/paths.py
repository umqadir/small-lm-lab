"""Where the bulk artifacts live, written down once.

The token streams, the checkpoints, and the run logs are far too large for the
repository, so they sit outside it. They have already moved once: the first home
was an external disk that read at well under 40 MB/s and left training waiting on
data and not on the GPU, and it could be unplugged mid-run. Moving them meant
editing a hardcoded path in every module and script that touched them, which is
work that has nothing to do with the research and is easy to do incompletely.

So the root is one constant here and everything else is derived from it. Moving
the artifacts again is a one-line change, and a run that needs a different root
for one invocation can set SMALL_LM_LAB_BULK_ROOT in the environment instead of
editing anything. Modules keep their own public constant names and assign them
from these, so callers and tests are unaffected.

The default root is machine independent: XDG_DATA_HOME if the environment sets
it, otherwise ~/.local/share. It is a user-writable location on Linux, macOS and
Windows, so a clone runs without editing this file. Point
SMALL_LM_LAB_BULK_ROOT at a fast disk with roughly 20 GB free for a full corpus
and checkpoint schedule.

Recorded paths are written through portable_path() so that run metadata and
generated documents carry $SMALL_LM_LAB_BULK_ROOT instead of one machine's home
directory, which keeps committed outputs readable on any machine.
"""

from __future__ import annotations

import os
from pathlib import Path

BULK_ROOT_ENV = "SMALL_LM_LAB_BULK_ROOT"


def _default_bulk_root() -> Path:
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "small-lm-lab"


BULK_ROOT = Path(os.environ.get(BULK_ROOT_ENV, _default_bulk_root()))

DATA_ROOT = BULK_ROOT / "data"
TOKENIZED_ROOT = DATA_ROOT / "tokenized"
RAW_ROOT = DATA_ROOT / "raw"
CHECKPOINT_ROOT = BULK_ROOT / "checkpoints"
RUN_ROOT = BULK_ROOT / "runs"
HF_CACHE = DATA_ROOT / "hf_cache"


def portable_path(path: str | os.PathLike[str] | None) -> str | None:
    """Render a path with the bulk root replaced by its environment variable.

    A path under the bulk root becomes "$SMALL_LM_LAB_BULK_ROOT/checkpoints/..."
    so that a recorded path identifies the artifact rather than the machine.
    Anything outside the bulk root is returned unchanged. None passes through,
    since metadata fields use it to mean "no checkpoint".
    """
    if path is None:
        return None
    p = Path(path)
    try:
        relative = p.resolve().relative_to(BULK_ROOT.resolve())
    except (ValueError, OSError):
        return str(p)
    return f"${BULK_ROOT_ENV}/{relative.as_posix()}"
