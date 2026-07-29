"""Fail-fast gate: run every downstream pipeline stage against a tiny model.

Motivation, recorded so the cost is never argued away: on 2026-07-21 the
schedule spent three training runs' worth of GPU-hours, then died in two
seconds on the first execution of the LR-scoring path, because eval code cast
a float64 on an MPS tensor and MPS has no float64. Training is MLX and never
touches that code; the eval battery is PyTorch. Any step that has never
executed on its real device is a loaded gun sitting behind hours of training.
This gate fires every such step in miniature, on the same device, in minutes.

What it runs, all through the real command-line entry points the schedule
itself invokes, against a random-init size30m checkpoint written to a scratch
directory and synthetic validation streams that mimic the corpus format:

  1. scripts/06_evaluate.py --suite perplexity   (the LR-scoring command)
  2. scripts/06_evaluate.py --suite icl
  3. scripts/06_evaluate.py --suite copying
  4. scripts/06_evaluate.py --suite blimp        (offline, warm HF cache)
  5. scripts/08_interp.py   --analysis all       (small sequence counts)

plus one in-process call of evaluate.sum_sentence_logprobs on the gate device,
because that function is the blimp scorer's core and deserves a direct check
even if the cache ever goes missing.

Exit 0 on pass, 1 on any failure, with the failing command's tail printed.
The scheduler (scripts/09_run_schedule.py) runs this once per repo commit
state before doing anything else, and refuses to proceed if it fails.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from small_lm_lab import evaluate  # noqa: E402
from small_lm_lab.config import get_config  # noqa: E402
from small_lm_lab.model_torch import TransformerLM  # noqa: E402

PYTHON = sys.executable
N_WINDOWS = 6  # per domain; enough for a bootstrap to have rows, small enough for seconds


def build_scratch(scratch: Path, device: str) -> Path:
    """A random-init size30m checkpoint plus synthetic val streams."""
    cfg = get_config("size30m")
    torch.manual_seed(0)
    model = TransformerLM(cfg)
    state = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
    ckpt = scratch / "smoke_size30m.pt"
    torch.save({"model": state, "tokens": 0}, ckpt)

    rng = np.random.default_rng(0)
    for domain in evaluate.DOMAINS:
        tokens = rng.integers(
            1, cfg.vocab_size, size=N_WINDOWS * cfg.context_len + 1, dtype=np.int64
        )
        tokens.astype(np.uint16).tofile(scratch / f"{domain}_val.bin")
    return ckpt


def run(cmd: list[str], env: dict, timeout: int = 900) -> tuple[int, str, float]:
    t0 = time.monotonic()
    proc = subprocess.run(
        cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout
    )
    took = time.monotonic() - t0
    return proc.returncode, (proc.stdout + proc.stderr), took


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--device", default="mps", help="the device the schedule will use")
    p.add_argument("--keep", action="store_true", help="keep the scratch directory")
    args = p.parse_args()

    scratch = Path(tempfile.mkdtemp(prefix="small_lm_lab_smoke_"))
    started = time.monotonic()
    failures = 0
    try:
        ckpt = build_scratch(scratch, args.device)

        # The blimp stage must not reach for the network: a warm cache is part
        # of what the gate certifies, and a silent download would hide a hole.
        env = dict(os.environ)
        env["HF_HUB_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"

        eval_common = [
            PYTHON, str(REPO / "scripts" / "06_evaluate.py"),
            "--checkpoint", str(ckpt),
            "--config", "size30m",
            "--device", args.device,
            "--split", "val",
            "--tokenized-root", str(scratch),
            "--n-resamples", "100",
        ]
        stages: list[tuple[str, list[str]]] = [
            (
                f"evaluate --suite {suite}",
                eval_common + ["--suite", suite, "--out", str(scratch / f"eval_{suite}.json")],
            )
            for suite in ("perplexity", "icl", "copying", "blimp")
        ]
        stages.append((
            "interp --analysis all",
            [
                PYTHON, str(REPO / "scripts" / "08_interp.py"),
                "--checkpoint", str(ckpt),
                "--config", "size30m",
                "--device", args.device,
                "--split", "val",
                "--tokenized-root", str(scratch),
                "--n-resamples", "100",
                "--patch-sequences", "8",
                "--max-windows", "8",
                "--batch-size", "8",
                "--weights-batch-size", "4",
                "--out", str(scratch / "interp.json"),
            ],
        ))

        for name, cmd in stages:
            rc, output, took = run(cmd, env)
            if rc == 0:
                print(f"smoke PASS  {name}  ({took:.1f}s)", flush=True)
            else:
                failures += 1
                tail = "\n".join(output.strip().splitlines()[-15:])
                print(f"smoke FAIL  {name}  exit {rc}  ({took:.1f}s)\n{tail}", flush=True)

        # Direct check of the blimp scorer core on the gate device.
        try:
            cfg = get_config("size30m")
            torch.manual_seed(0)
            model = TransformerLM(cfg).to(torch.float32).to(args.device).eval()
            rng = np.random.default_rng(1)
            sentences = [
                [int(t) for t in rng.integers(1, cfg.vocab_size, size=n)]
                for n in (5, 11, 7)
            ]
            got = evaluate.sum_sentence_logprobs(model, sentences, args.device, batch_size=2)
            assert np.all(np.isfinite(got)), f"non-finite sentence logprobs: {got}"
            print("smoke PASS  sum_sentence_logprobs in-process", flush=True)
        except Exception as exc:  # noqa: BLE001 - the gate reports, never raises
            failures += 1
            print(f"smoke FAIL  sum_sentence_logprobs in-process: {exc!r}", flush=True)
    finally:
        if args.keep:
            print(f"scratch kept at {scratch}", flush=True)
        else:
            shutil.rmtree(scratch, ignore_errors=True)

    total = time.monotonic() - started
    verdict = "PASS" if failures == 0 else f"FAIL ({failures} stage(s))"
    print(f"smoke gate {verdict} on device {args.device} in {total:.1f}s", flush=True)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
