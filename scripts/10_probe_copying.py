"""Cheap induction-signature probe for one checkpoint, run on the laptop.

The staged deep run streams checkpoints off the pod as it trains. This probe
reads one of them and records two numbers that rise as induction forms: the
synthetic copying score and the in-context learning score. It is a monitoring
instrument, not the pre-registered evaluation. It runs at a reduced sequence
count and a capped window count so it finishes in minutes on CPU, and its output
never enters a result. Every reported number still comes from the registered
battery in scripts/06_evaluate.py and scripts/08_interp.py.

The two numbers, both taken from src/small_lm_lab/evaluate.py so the probe tracks
the same quantities the battery reports, only cheaper:

  copying  the copying task's exact-match accuracy at a reduced sequence count
           (default 64 vs the registered 512). Random tokens repeated twice make
           the second copy unlearnable except by attending to the earlier
           occurrence, so this is near zero before induction forms and rises as
           it does. This is the transition signal the amendment's trigger and
           stop rules read.
  icl      the in-context learning score, mean loss late in the context minus
           early in it, over a capped number of held-out windows (default 128).
           A more negative score means the model is using context. Computed
           through interp.window_losses, which caps the windows without changing
           the pre-registered position windows, and evaluate's own ICL statistic.

One JSON line is appended per call to analysis/staged/probe_curve.jsonl:
  {"checkpoint", "tokens", "copying", "icl", "utc"}

so the growing file is the probe curve the interp anchors are chosen from. The
token count comes from --tokens, or from the token count the checkpoint recorded
if the flag is omitted.

Usage:
  python scripts/10_probe_copying.py --config size30m \
      --checkpoint /path/to/tokens_000512000000.pt
  python scripts/10_probe_copying.py --config size30m \
      --checkpoint CKPT --tokens 512e6 --domain fineweb_edu --n 64
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from small_lm_lab import evaluate, interp
from small_lm_lab.config import get_config
from small_lm_lab.data import DEFAULT_TOKENIZED_ROOT
from small_lm_lab.evaluate import DOMAINS

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = REPO_ROOT / "scripts" / "06_evaluate.py"
DEFAULT_OUT = REPO_ROOT / "analysis" / "staged" / "probe_curve.jsonl"

# Reduced counts: this is a monitoring probe, not the registered evaluation.
PROBE_N_SEQUENCES = 64
PROBE_MAX_WINDOWS = 128
PROBE_N_RESAMPLES = 200


def load_eval_script():
    """Import scripts/06_evaluate.py by path, for its checkpoint loader.

    The same loader training and the battery use, imported rather than copied so
    the probe reads exactly what the pod wrote: strict=True, bf16 storage cast up
    to float32, the tied head handled. scripts/ is not a package.
    """
    spec = importlib.util.spec_from_file_location("eval_script", EVAL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True, help="named config, for example size30m")
    p.add_argument(
        "--checkpoint", type=Path, required=True, help="path to a saved checkpoint"
    )
    p.add_argument(
        "--tokens",
        type=float,
        default=None,
        help=(
            "token count for this checkpoint, for the curve's x axis. Defaults to "
            "the token count the checkpoint recorded"
        ),
    )
    p.add_argument("--device", default="cpu", help="cpu, mps, or cuda")
    p.add_argument(
        "--domain",
        default="fineweb_edu",
        choices=list(DOMAINS),
        help=(
            "domain whose held-out split the ICL score is read from; the "
            "2026-07-26 amendment registers fineweb_edu as the default"
        ),
    )
    p.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="test is the closed split and is only read when named; the probe "
        "uses val",
    )
    p.add_argument(
        "--n",
        type=int,
        default=PROBE_N_SEQUENCES,
        help=f"copying sequences, reduced for cost. Default {PROBE_N_SEQUENCES}, "
        f"vs the registered {evaluate.COPY_N_SEQUENCES}",
    )
    p.add_argument(
        "--max-windows",
        type=int,
        default=PROBE_MAX_WINDOWS,
        help=f"ICL windows, capped for cost. Default {PROBE_MAX_WINDOWS}. The "
        "registered ICL score reads the whole split and is never capped",
    )
    p.add_argument(
        "--n-resamples",
        type=int,
        default=PROBE_N_RESAMPLES,
        help="bootstrap resamples behind the copying point estimate; the probe "
        "records only the point, so this is a cost knob",
    )
    p.add_argument("--seed", type=int, default=evaluate.DEFAULT_SEED)
    p.add_argument("--batch-size", type=int, default=evaluate.DEFAULT_BATCH_SIZE)
    p.add_argument("--tokenized-root", type=Path, default=DEFAULT_TOKENIZED_ROOT)
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="JSONL file to append one probe line to",
    )
    return p


def probe(args: argparse.Namespace) -> dict:
    """Compute the two probe numbers for one checkpoint and return the line."""
    cfg = get_config(args.config)
    eval_script = load_eval_script()
    model, ckpt_tokens = eval_script.load_model(cfg, args.checkpoint, args.device)

    tokens = int(args.tokens) if args.tokens is not None else ckpt_tokens
    if tokens is None:
        raise SystemExit(
            f"no token count for {args.checkpoint}: it recorded none and --tokens "
            "was not given, so the probe point could not be placed on the curve"
        )

    # Copying: the registered function at a reduced sequence count. The point
    # estimate is the induction signal; the interval it also computes is dropped.
    copying_result = evaluate.copying_eval(
        model,
        args.device,
        n_sequences=args.n,
        seed=args.seed,
        batch_size=args.batch_size,
        n_resamples=args.n_resamples,
    )

    # ICL: the pre-registered late-minus-early statistic over a capped number of
    # windows. window_losses keeps the registered ICL positions and only limits
    # how many windows are read, and evaluate._icl_statistic is the same reducer
    # the battery uses, so this is the registered ICL score on fewer windows.
    windows = interp.window_losses(
        model,
        args.domain,
        args.split,
        args.device,
        batch_size=args.batch_size,
        root=args.tokenized_root,
        max_windows=args.max_windows,
    )
    icl = evaluate._icl_statistic(windows[:, [2, 3]])

    return {
        "checkpoint": str(args.checkpoint),
        "tokens": int(tokens),
        "copying": float(copying_result.exact_match),
        "icl": float(icl),
        "utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    args = build_parser().parse_args()
    line = probe(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as f:
        f.write(json.dumps(line) + "\n")
    print(
        f"probe {Path(line['checkpoint']).name}: tokens {line['tokens']:,}, "
        f"copying {line['copying']:.4f}, icl {line['icl']:.4f} -> {args.out}"
    )


if __name__ == "__main__":
    main()
