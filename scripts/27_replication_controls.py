"""Summarize the registered second seed and depth-control trajectories."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

REQUIRED = ("headline", "replication_seed2", "depth1", "attention_only_2layer")


def parse_run(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {value!r}")
    return name, Path(path)


def summarize(path: Path) -> dict:
    with path.open() as stream:
        result = json.load(stream)
    points = [point for point in result["points"] if point.get("tokens") is not None]
    phase = result["trajectory_summary"]["phase_change"]
    interval = phase.get("interval")
    final = points[-1]
    return {
        "path": str(path),
        "config": result["metadata"]["config"],
        "n_params": result["metadata"]["n_params"],
        "n_sequences": result["metadata"]["n_sequences_used"],
        "final_tokens": int(final["tokens"]),
        "final_max_prefix_matching": float(final["max_prefix_matching"]),
        "final_induction_heads": final["induction_heads"],
        "phase_change": interval,
    }


def replication_verdict(headline: dict, replication: dict) -> dict:
    first = headline["phase_change"]
    second = replication["phase_change"]
    if not first or not first.get("crossed"):
        return {"verdict": "not_scored", "reason": "headline crossing is absent"}
    if not second or not second.get("crossed"):
        return {
            "verdict": "censored",
            "reason": (
                "the second seed did not cross within its registered replication "
                "horizon, so a crossing-location ratio is not identified"
            ),
            "headline_end_tokens": first["end_tokens"],
            "replication_horizon_tokens": replication["final_tokens"],
        }
    ends = [int(first["end_tokens"]), int(second["end_tokens"])]
    ratio = max(ends) / min(ends)
    return {
        "verdict": "location_uncertain" if ratio > 2.0 else "replicated_within_2x",
        "headline_end_tokens": ends[0],
        "replication_end_tokens": ends[1],
        "crossing_ratio": ratio,
        "registered_uncertainty_threshold": 2.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = dict(args.run)
    missing = sorted(set(REQUIRED) - set(paths))
    if missing:
        raise SystemExit(f"missing --run entries for {missing}")
    runs = {name: summarize(paths[name]) for name in REQUIRED}
    result = {
        "metadata": {
            "registered_controls": (
                "docs/PROTOCOL.md amendment 2026-07-26 section 10"
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "runs": runs,
        "replication": replication_verdict(
            runs["headline"], runs["replication_seed2"]
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(result["replication"]["verdict"])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
