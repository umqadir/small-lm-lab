"""Per-domain validation loss at every checkpoint on the grid.

Two things need this curve and neither is served by the training log.

The abruptness criterion registered in the 2026-07-26 amendment reads F, the
fraction of the run's total FineWeb-Edu validation-loss improvement that falls
inside the crossing interval, at four grid checkpoints. The verdict is a
function of that number, so it has to come from a measurement over the same
checkpoints the crossing was located on and not from a log line emitted by a
training process on another machine.

The loss curve reported beside the emergence curve needs the same quantity at
every grid point. A training log records the loss of whatever validation batch
the loop happened to draw at that step, which is a different estimator at every
point; this is one fixed set of windows scored by every checkpoint, so a
difference between two points is a difference in the model and not in the draw.

The statistic is the token-weighted mean negative log likelihood over
non-overlapping windows tiling the split, which is the log of
evaluate._weighted_perplexity's answer on the same windows. Both are imported
rather than restated. scripts/06_evaluate.py remains the headline perplexity and
never truncates; this one caps the window count for cost, records the cap and
the realized count on every point, and applies the same cap at every checkpoint
so the curve is comparable along its own axis.

Usage:
  python scripts/21_val_curve.py --config size30m --checkpoint-dir DIR \
      --out OUT.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from small_lm_lab import evaluate, interp
from small_lm_lab.bootstrap import DEFAULT_ALPHA, DEFAULT_N_RESAMPLES, bootstrap_ci
from small_lm_lab.config import get_config, n_params
from small_lm_lab.data import DEFAULT_TOKENIZED_ROOT
from small_lm_lab.evaluate import DOMAINS, _weighted_perplexity
from small_lm_lab.paths import portable_path

REPO_ROOT = Path(__file__).resolve().parents[1]
INTERP_SCRIPT = REPO_ROOT / "scripts" / "08_interp.py"

# Cost only. 2,048 windows of 512 tokens is 1,048,576 tokens per domain per
# checkpoint, against a validation split of roughly 9,800 windows. The same
# windows are scored at every checkpoint, in the order the exhaustive tiling
# produces them, so the sampling error is shared across the curve and cancels
# from every difference read off it.
DEFAULT_MAX_WINDOWS = 2048


def load_scripts():
    """scripts/08_interp.py, for find_checkpoints and the checkpoint loader.

    The same reason scripts/19_emergence.py imports them by path: a second copy
    of the checkpoint loader is a second thing to keep in step with train.py.
    """
    spec = importlib.util.spec_from_file_location("interp_script", INTERP_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.load_eval_script()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--config", required=True, help="named config, for example size30m")
    p.add_argument("--device", default="cpu", help="cpu, mps, or cuda")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="test is the closed split and is only read when named",
    )
    p.add_argument("--domains", nargs="*", default=list(DOMAINS), choices=list(DOMAINS))
    p.add_argument("--batch-size", type=int, default=32, help="cost only")
    p.add_argument(
        "--max-windows",
        type=int,
        default=DEFAULT_MAX_WINDOWS,
        help=(
            "windows scored per domain per checkpoint, the first windows of the "
            "exhaustive tiling. Recorded on every point. Negative for no cap"
        ),
    )
    p.add_argument("--n-resamples", type=int, default=DEFAULT_N_RESAMPLES)
    p.add_argument("--seed", type=int, default=evaluate.DEFAULT_SEED)
    p.add_argument("--tokenized-root", type=Path, default=DEFAULT_TOKENIZED_ROOT)
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = get_config(args.config)
    script, eval_script = load_scripts()
    checkpoints = list(script.find_checkpoints(args.checkpoint_dir))
    max_windows = None if args.max_windows < 0 else int(args.max_windows)

    points: list[dict[str, Any]] = []
    for path in checkpoints:
        model, tokens = eval_script.load_model(cfg, path, args.device)
        record: dict[str, Any] = {
            "tokens": None if tokens is None else int(tokens),
            "checkpoint": portable_path(path),
            "domains": {},
        }
        for domain in args.domains:
            windows = interp.window_losses(
                model,
                domain,
                args.split,
                args.device,
                batch_size=args.batch_size,
                root=args.tokenized_root,
                max_windows=max_windows,
            )
            values = windows[:, :2]
            point, lo, hi = bootstrap_ci(
                values,
                _weighted_perplexity,
                n_resamples=args.n_resamples,
                seed=args.seed,
                alpha=DEFAULT_ALPHA,
            )
            n_windows = int(values.shape[0])
            record["domains"][domain] = {
                "domain": domain,
                "split": args.split,
                # The loss the criterion reads. Token weighted, so it is the mean
                # over tokens and not the mean of per-window means.
                "mean_nll": float(
                    (values[:, 0] * values[:, 1]).sum() / values[:, 1].sum()
                ),
                "perplexity": point,
                "perplexity_ci_low": lo,
                "perplexity_ci_high": hi,
                "n_windows": n_windows,
                "n_tokens": int(values[:, 1].sum()),
                "max_windows": max_windows,
                "windows_capped": bool(
                    max_windows is not None and n_windows >= max_windows
                ),
            }
        points.append(record)
        summary = " ".join(
            f"{d} nll {record['domains'][d]['mean_nll']:.4f}" for d in args.domains
        )
        print(f"{path.name}: tokens {record['tokens']}, {summary}", flush=True)

    result = {
        "metadata": {
            "checkpoint_dir": portable_path(args.checkpoint_dir),
            "n_checkpoints": len(checkpoints),
            "config": args.config,
            "n_params": n_params(cfg),
            "device": args.device,
            "split": args.split,
            "domains": list(args.domains),
            "batch_size": args.batch_size,
            "max_windows": max_windows,
            "n_resamples": args.n_resamples,
            "seed": args.seed,
            "context_len": int(cfg.context_len),
            "tokenized_root": portable_path(args.tokenized_root),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "statistic": (
                "token-weighted mean negative log likelihood over the first "
                "max_windows non-overlapping windows of the split, and its "
                "exponential. Per domain, never pooled. The same windows in the "
                "same order at every checkpoint."
            ),
        },
        "points": points,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
