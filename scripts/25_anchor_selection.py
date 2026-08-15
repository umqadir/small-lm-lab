"""Measure the registered pre-transition band and select causal anchors.

The band uses ten independent sequence seeds at the first four Stage A
checkpoints.  Anchor thresholds and the three-point plateau rule are fixed in
docs/PROTOCOL.md.  This script turns those rules into a reproducible JSON file
and emits the de-duplicated anchor-plus-neighbour token list accepted by
scripts/08_interp.py --checkpoints.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from small_lm_lab import interp
from small_lm_lab.config import get_config, n_params

REPO_ROOT = Path(__file__).resolve().parents[1]
INTERP_SCRIPT = REPO_ROOT / "scripts" / "08_interp.py"
N_BAND_SEEDS = 10
N_BAND_CHECKPOINTS = 4


def load_interp_script():
    spec = importlib.util.spec_from_file_location("interp_script", INTERP_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_anchors(
    tokens: list[int], scores: list[float], mu: float, sigma: float
) -> dict:
    """Apply the registered plateau and 10/50/90-percent anchor rules."""
    if len(tokens) != len(scores) or len(tokens) < 3:
        raise ValueError("tokens and scores need the same length of at least three")
    tolerance = 3.0 * sigma
    band_upper = mu + tolerance
    rise_confirmation = next(
        (
            i
            for i in range(2, len(scores))
            if all(value > band_upper for value in scores[i - 2 : i + 1])
        ),
        None,
    )
    if rise_confirmation is None:
        raise ValueError("trajectory has no registered three-checkpoint rise")
    plateau_index = None
    plateau_value = None
    for i in range(rise_confirmation, len(scores)):
        running_max = max(scores[: i + 1])
        if all(running_max - value < tolerance for value in scores[i - 2 : i + 1]):
            plateau_index = i
            plateau_value = running_max
            break
    if plateau_index is None or plateau_value is None:
        raise ValueError("trajectory has no registered three-checkpoint plateau")

    levels = {
        "pre": mu + 0.10 * (plateau_value - mu),
        "mid": mu + 0.50 * (plateau_value - mu),
        "post": mu + 0.90 * (plateau_value - mu),
    }
    below = [i for i, value in enumerate(scores) if value < levels["pre"]]
    if not below:
        raise ValueError("trajectory has no checkpoint below the 10-percent level")
    anchor_indices = {
        "pre": below[-1],
        "mid": min(range(len(scores)), key=lambda i: abs(scores[i] - levels["mid"])),
        "post": next(i for i, value in enumerate(scores) if value >= levels["post"]),
        "final": len(scores) - 1,
    }
    sensitivity = set(anchor_indices.values())
    for index in list(sensitivity):
        sensitivity.update((max(0, index - 1), min(len(scores) - 1, index + 1)))
    return {
        "status": "selected",
        "rise": {
            "confirmation_index": rise_confirmation,
            "confirmation_tokens": tokens[rise_confirmation],
            "band_upper": band_upper,
        },
        "plateau": {
            "confirmation_index": plateau_index,
            "confirmation_tokens": tokens[plateau_index],
            "value": plateau_value,
            "tolerance": tolerance,
        },
        "levels": levels,
        "anchors": {
            name: {
                "index": index,
                "tokens": tokens[index],
                "score": scores[index],
            }
            for name, index in anchor_indices.items()
        },
        "causal_checkpoint_tokens": [tokens[i] for i in sorted(sensitivity)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--config", default="size30m")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--weights-batch-size", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    helper = load_interp_script()
    paths = helper.find_checkpoints(args.checkpoint_dir)
    if len(paths) < N_BAND_CHECKPOINTS:
        raise SystemExit(
            f"need the first {N_BAND_CHECKPOINTS} checkpoints, found {len(paths)}"
        )
    config = get_config(args.config)
    readings = []
    for path in paths[:N_BAND_CHECKPOINTS]:
        model, token_count = helper.load_eval_script().load_model(
            config, path, args.device
        )
        for seed in range(N_BAND_SEEDS):
            per_sequence = interp.prefix_matching_per_sequence(
                model,
                args.device,
                seed=seed,
                batch_size=args.weights_batch_size,
            )
            readings.append(
                {
                    "checkpoint": str(path),
                    "tokens": int(token_count),
                    "sequence_seed": seed,
                    "max_prefix_matching": float(per_sequence.mean(axis=0).max()),
                }
            )
        print(f"measured band at {path.name}", flush=True)

    values = np.asarray([item["max_prefix_matching"] for item in readings])
    mu = float(values.mean())
    sigma = float(values.std(ddof=1))
    with args.trajectory.open() as stream:
        trajectory = json.load(stream)
    points = [p for p in trajectory["points"] if p.get("tokens") is not None]
    tokens = [int(p["tokens"]) for p in points]
    scores = [float(p["max_prefix_matching"]) for p in points]
    try:
        selection = select_anchors(tokens, scores, mu, sigma)
    except ValueError as exc:
        selection = {
            "status": "unavailable",
            "reason": str(exc),
            "causal_checkpoint_tokens": [],
        }
    result = {
        "metadata": {
            "config": args.config,
            "n_params": n_params(config),
            "checkpoint_dir": str(args.checkpoint_dir),
            "trajectory": str(args.trajectory),
            "device": args.device,
            "n_band_seeds": N_BAND_SEEDS,
            "n_band_checkpoints": N_BAND_CHECKPOINTS,
            "n_sequences_per_reading": interp.PREFIX_MATCHING_N_SEQUENCES,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "band": {
            "mu": mu,
            "sigma": sigma,
            "upper": mu + 3.0 * sigma,
            "readings": readings,
        },
        "selection": selection,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(" ".join(str(t) for t in selection["causal_checkpoint_tokens"]))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
