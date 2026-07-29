"""Run the pre-registered evaluation battery against one checkpoint.

Writes a single JSON holding every pre-registered number, its bootstrap
interval, and enough metadata to say what produced it: the checkpoint, the
config, the token count if the checkpoint recorded one, the device, the seed,
and a timestamp. Re-runnable: the same arguments against the same checkpoint
reproduce the same file, because every bootstrap is seeded.

The test splits stay closed until the final evaluation, which runs once. So
--split defaults to val and test is only read when asked for by name. Nothing
here writes to the token streams or to the training lock.

Usage:
  python scripts/06_evaluate.py --config size30m --checkpoint PATH --out OUT.json
  python scripts/06_evaluate.py --config size30m --random-init --out OUT.json

--random-init is the control: an untrained model at the same config. Perplexity
should sit at roughly the vocabulary size, copying exact match at roughly zero,
and the in-context score at roughly zero. It is what the trained numbers are
read against.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch

from small_lm_lab import evaluate
from small_lm_lab.config import ModelConfig, get_config, n_params
from small_lm_lab.data import DEFAULT_TOKENIZED_ROOT
from small_lm_lab.model_torch import TransformerLM
from small_lm_lab.paths import portable_path

SUITES = ("all", "perplexity", "blimp", "icl", "copying")

# The training code is not written yet, so the checkpoint layout is not fixed.
# These are the shapes worth accepting without guessing: a bare state_dict, or a
# dict carrying one under a conventional key alongside bookkeeping.
STATE_KEYS = ("model", "model_state_dict", "state_dict", "weights")
TOKEN_KEYS = ("tokens", "tokens_seen", "token_count", "n_tokens", "total_tokens")


def _find_state_dict(obj: dict) -> dict:
    for key in STATE_KEYS:
        inner = obj.get(key)
        if isinstance(inner, dict):
            return inner
    return obj


def _find_token_count(obj: dict) -> Optional[int]:
    for key in TOKEN_KEYS:
        value = obj.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float) and float(value).is_integer():
            return int(value)
    return None


def load_model(
    cfg: ModelConfig, checkpoint: Optional[Path], device: str
) -> tuple[TransformerLM, Optional[int]]:
    """Build the model, load weights if given, return it with the token count.

    Weights are cast to float32 for evaluation even when the checkpoint stores
    bf16, so a headline number is never limited by the storage dtype.
    """
    model = TransformerLM(cfg)
    tokens: Optional[int] = None

    if checkpoint is not None:
        obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(obj, dict):
            raise ValueError(
                f"{checkpoint} does not hold a dict; expected a state_dict or a "
                "checkpoint dict carrying one"
            )
        state = _find_state_dict(obj)
        tokens = _find_token_count(obj)
        # torch.compile wraps parameter names; strip the wrapper if present.
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)

    model.to(torch.float32)
    model.to(device)
    model.eval()
    return model, tokens


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=Path, help="path to a saved checkpoint")
    src.add_argument(
        "--random-init",
        action="store_true",
        help="evaluate an untrained model at this config, as a control",
    )
    p.add_argument("--config", required=True, help="named config, for example size30m")
    p.add_argument("--device", default="cpu", help="cpu, mps, or cuda")
    p.add_argument("--out", type=Path, required=True, help="path for the JSON result")
    p.add_argument("--suite", choices=SUITES, default="all")
    p.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="test is the closed split and is only read when named",
    )
    p.add_argument("--seed", type=int, default=evaluate.DEFAULT_SEED)
    p.add_argument("--batch-size", type=int, default=evaluate.DEFAULT_BATCH_SIZE)
    p.add_argument(
        "--n-resamples", type=int, default=evaluate.DEFAULT_N_RESAMPLES,
        help="bootstrap resamples; the pre-registered value is 1000",
    )
    p.add_argument("--tokenized-root", type=Path, default=DEFAULT_TOKENIZED_ROOT)
    p.add_argument("--hf-home", type=Path, default=evaluate.DEFAULT_HF_HOME)
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = get_config(args.config)
    model, tokens = load_model(cfg, args.checkpoint, args.device)

    run_all = args.suite == "all"
    results: dict[str, Any] = {
        "metadata": {
            "checkpoint": portable_path(args.checkpoint),
            "random_init": bool(args.random_init),
            "config": args.config,
            "n_params": n_params(cfg),
            "tokens": tokens,
            "device": args.device,
            "seed": args.seed,
            "split": args.split,
            "suite": args.suite,
            "n_resamples": args.n_resamples,
            "batch_size": args.batch_size,
            "tokenized_root": portable_path(args.tokenized_root),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "torch_version": torch.__version__,
        }
    }

    if run_all or args.suite == "perplexity":
        results["perplexity"] = {
            domain: asdict(
                evaluate.test_perplexity(
                    model,
                    domain,
                    args.split,
                    args.device,
                    batch_size=args.batch_size,
                    root=args.tokenized_root,
                    n_resamples=args.n_resamples,
                    seed=args.seed,
                )
            )
            for domain in evaluate.DOMAINS
        }

    if run_all or args.suite == "blimp":
        results["blimp"] = {
            paradigm: asdict(
                evaluate.blimp_accuracy(
                    model,
                    paradigm,
                    args.device,
                    hf_home=args.hf_home,
                    n_resamples=args.n_resamples,
                    seed=args.seed,
                )
            )
            for paradigm in evaluate.BLIMP_PARADIGMS
        }

    if run_all or args.suite == "icl":
        results["icl"] = {
            domain: asdict(
                evaluate.icl_score(
                    model,
                    domain,
                    args.split,
                    args.device,
                    batch_size=args.batch_size,
                    root=args.tokenized_root,
                    n_resamples=args.n_resamples,
                    seed=args.seed,
                )
            )
            for domain in evaluate.DOMAINS
        }

    if run_all or args.suite == "copying":
        results["copying"] = asdict(
            evaluate.copying_eval(
                model,
                args.device,
                seed=args.seed,
                batch_size=args.batch_size,
                n_resamples=args.n_resamples,
            )
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(results, f, indent=2, sort_keys=False)
        f.write("\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
