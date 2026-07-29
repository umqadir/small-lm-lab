"""Decide whether a Stage A run can be safely resumed, and refuse rather than guess.

The failure this exists to prevent is silent and total: if a supervisor restarts after
a power cut, finds no usable resume state, and quietly launches a fresh run, it
overwrites an in-progress run's log and restarts from token zero. The checkpoint grid
is the experiment for an induction-head emergence study, so that is not a lost
afternoon, it is a destroyed result that still looks like a healthy run.

So the contract is deliberately asymmetric:

  - a genuinely empty checkpoint directory  -> fresh is safe (exit 10)
  - a valid resume_state.pt                 -> resume (exit 0)
  - anything else                           -> refuse (exit 20)

"Anything else" includes checkpoints present with no resume state, a resume state that
will not load, and a resume state that disagrees with the run it claims to belong to.
Refusing leaves a stopped run with a loud log, which a human fixes in minutes. Guessing
wrong destroys weeks of GPU time and is not detectable after the fact.

Usage:
  python scripts/17_validate_resume.py --checkpoint-dir <dir> \
      --run-name size30m_staged_seed1 --framework torch --seed 1 \
      [--tokens-per-step 16384] [--check-checkpoints]
Exit codes: 0 resume, 10 fresh, 20 refuse.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXIT_RESUME = 0
EXIT_FRESH = 10
EXIT_REFUSE = 20

REQUIRED_KEYS = {
    "weights_fp32",
    "optimizer",
    "stream_state",
    "step",
    "tokens",
    "framework",
    "run_name",
    "seed",
}


def say(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True, type=Path)
    p.add_argument("--run-name", required=True)
    p.add_argument("--framework", default="torch")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--tokens-per-step", type=int, default=16384)
    p.add_argument(
        "--check-checkpoints",
        action="store_true",
        help="also load every tokens_*.pt to catch a partial write",
    )
    args = p.parse_args()

    ckpt_dir: Path = args.checkpoint_dir
    resume = ckpt_dir / "resume_state.pt"
    grid = sorted(ckpt_dir.glob("tokens_*.pt")) if ckpt_dir.is_dir() else []

    if not ckpt_dir.is_dir() or (not grid and not resume.exists()):
        say(f"FRESH: no prior run at {ckpt_dir}")
        return EXIT_FRESH

    # A run exists in some form. From here, fresh is never an acceptable answer.
    if not resume.exists():
        say(
            f"REFUSE: {len(grid)} checkpoint(s) present but resume_state.pt is missing "
            f"at {resume}. A fresh start would overwrite an in-progress run. "
            f"Human decision required."
        )
        return EXIT_REFUSE

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        say(f"REFUSE: cannot import torch to validate resume state: {exc}")
        return EXIT_REFUSE

    try:
        obj = torch.load(resume, map_location="cpu", weights_only=False)
    except Exception as exc:
        say(f"REFUSE: resume_state.pt will not load ({type(exc).__name__}: {exc})")
        return EXIT_REFUSE

    if not isinstance(obj, dict):
        say(f"REFUSE: resume_state.pt is {type(obj).__name__}, expected dict")
        return EXIT_REFUSE

    missing = REQUIRED_KEYS - set(obj)
    if missing:
        say(f"REFUSE: resume_state.pt missing keys {sorted(missing)}")
        return EXIT_REFUSE

    # Identity: a resume file from a different run must never be silently adopted.
    for field, want in (
        ("run_name", args.run_name),
        ("framework", args.framework),
        ("seed", args.seed),
    ):
        got = obj[field]
        if got != want:
            say(f"REFUSE: resume_state.pt {field}={got!r}, expected {want!r}")
            return EXIT_REFUSE

    step, tokens = int(obj["step"]), int(obj["tokens"])
    if step < 0 or tokens <= 0:
        say(f"REFUSE: implausible step={step} tokens={tokens}")
        return EXIT_REFUSE

    # tokens must be consistent with step at the registered tokens-per-step.
    expected = (step + 1) * args.tokens_per_step
    if tokens != expected:
        say(
            f"REFUSE: tokens={tokens:,} inconsistent with step={step:,} at "
            f"{args.tokens_per_step:,} tok/step (expected {expected:,})"
        )
        return EXIT_REFUSE

    # The resume point must not sit behind the newest grid checkpoint, which would
    # mean the grid contains work the resume state cannot account for.
    if grid:
        newest = max(int(g.stem.split("_")[-1]) for g in grid)
        if newest > tokens:
            say(
                f"REFUSE: newest grid checkpoint is {newest:,} tokens but resume state "
                f"is only at {tokens:,}; the grid is ahead of the resume point"
            )
            return EXIT_REFUSE

    if args.check_checkpoints:
        bad = []
        for g in grid:
            try:
                torch.load(g, map_location="cpu", weights_only=False)
            except Exception as exc:
                bad.append(f"{g.name} ({type(exc).__name__})")
        if bad:
            say(f"REFUSE: {len(bad)} checkpoint(s) will not load: {bad}")
            return EXIT_REFUSE
        say(f"  all {len(grid)} grid checkpoints load cleanly")

    say(
        f"RESUME: step={step:,} tokens={tokens:,} run={obj['run_name']} "
        f"framework={obj['framework']} seed={obj['seed']} grid={len(grid)} checkpoints"
    )
    return EXIT_RESUME


if __name__ == "__main__":
    sys.exit(main())
