"""Score the eight pre-registered predictions against the measured trajectory.

docs/PROTOCOL.md states P1 to P8 and, for each, the field
of the trajectory it is read from and the condition under which it fails. This
script performs those lookups and writes the verdicts, so that a scorecard in a
document is a copy of a pipeline output rather than a reading someone took by
eye. Nothing here restates a threshold that document does not state, and nothing
here is adjustable: there are no flags that move a criterion.

Two readings need a rule the prediction document leaves implicit, and both are
recorded on the output next to the verdict they affect rather than settled
silently.

  P1 and P6 name round token counts, 200,000,000, and no checkpoint sits on one.
  A checkpoint is saved at the first optimizer step at or past its grid target,
  so the 200M grid point records 200,015,872 tokens. The comparison is made
  against the grid TARGET from configs/stage_a_checkpoints.json, matched to the
  measured checkpoints by position, since "at or below 200M tokens" in a
  document about a pre-registered grid names a grid point and not an arithmetic
  bound that point misses by 0.008 percent.

  P5's second clause says the steepest in-context step falls "within one grid
  step" of the crossing. A step joins two adjacent grid positions, so the rule
  applied is that either endpoint of the steepest step lies within one grid
  position of the crossing checkpoint. Both endpoints are written out.

  P7 says the head's OV copying score is above 0.5 "at the checkpoint at or
  before its own crossing". That is read as the crossing checkpoint or the one
  grid step before it. The looser reading, any earlier checkpoint at all, is
  computed and written out beside the verdict rather than adopted: before the
  transition the OV score of an untrained head wanders over most of its range,
  so a transient early excursion above 0.5 is not the OV circuit having turned
  copying, and a criterion that such an excursion satisfies cannot fail.

P8 is scored only when --control is given, since it is a property of a
random-init model and not of the trajectory. Without it P8 is reported as not
scored, never as held.

Usage:
  python scripts/24_score_predictions.py --trajectory TRAJ.json \
      --control CONTROL.json --grid configs/stage_a_checkpoints.json \
      --out OUT.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

HELD = "held"
FAILED = "failed"
NOT_SCORED = "not scored"

# The two token counts docs/PROTOCOL.md names, and the
# level P4 reads the previous-token score against. Transcribed from that
# document, which is the only place they exist.
P1_DEADLINE_TOKENS = 200_000_000
P6_DEADLINE_TOKENS = 200_000_000
P5_MIN_ICL_DROP = 0.05
P5_DOMAIN = "fineweb_edu"
P7_OV_LEVEL = 0.5
P8_MAX_PREFIX_MATCHING = 0.05


def load(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def placed(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (p for p in points if p.get("tokens") is not None), key=lambda p: p["tokens"]
    )


def grid_targets(points: list[dict[str, Any]], grid: Optional[list[int]]) -> dict[int, int]:
    """{measured token count: its registered grid target}.

    Matched by position, which is exact when the run wrote one checkpoint per
    grid entry, and refused otherwise rather than guessed at.
    """
    if grid is None:
        return {int(p["tokens"]): int(p["tokens"]) for p in points}
    if len(grid) != len(points):
        raise ValueError(
            f"{len(points)} checkpoints against a {len(grid)}-point registered "
            "grid; the two cannot be matched by position"
        )
    return {int(p["tokens"]): int(t) for p, t in zip(points, sorted(grid))}


def score(
    trajectory: dict[str, Any],
    control: Optional[dict[str, Any]],
    grid: Optional[list[int]],
) -> list[dict[str, Any]]:
    points = placed(trajectory["points"])
    summary = trajectory["trajectory_summary"]
    phase = summary["phase_change"]
    interval = phase.get("interval")
    crossed = bool(interval and interval.get("crossed"))
    targets = grid_targets(points, grid)
    positions = {int(p["tokens"]): i for i, p in enumerate(points)}
    by_tokens = {int(p["tokens"]): p for p in points}
    final = points[-1]

    results: list[dict[str, Any]] = []

    # P1. The phase-change interval closes at or below 200,000,000 tokens.
    end_tokens = interval["end_tokens"] if crossed else None
    results.append(
        {
            "id": "P1",
            "primary": True,
            "statement": "the phase-change interval closes at or below 200,000,000 tokens",
            "criterion": (
                "fails if phase_change.interval.end_tokens exceeds 200,000,000, "
                "and outright if crossed is false at 2e9"
            ),
            "crossed": crossed,
            "end_tokens": end_tokens,
            "end_grid_target": targets.get(end_tokens) if crossed else None,
            "deadline_tokens": P1_DEADLINE_TOKENS,
            "point_prediction_tokens": 48_000_000,
            "point_prediction_hit": (
                crossed and targets.get(end_tokens) == 48_000_000
            ),
            "verdict": (
                HELD
                if crossed and targets[end_tokens] <= P1_DEADLINE_TOKENS
                else FAILED
            ),
        }
    )

    # P2. At least one head is an induction head at the final checkpoint.
    heads = [tuple(h) for h in final["induction_heads"]]
    layers = sorted({h[0] for h in heads})
    results.append(
        {
            "id": "P2",
            "primary": False,
            "statement": (
                "at least one head is above the registered 0.2 at the final "
                "checkpoint"
            ),
            "criterion": "fails if induction_heads is empty at the final checkpoint",
            "final_tokens": int(final["tokens"]),
            "induction_heads": [list(h) for h in heads],
            "n_induction_heads": len(heads),
            "layers_occupied": layers,
            "point_prediction": "between 1 and 4 heads, in layers 2 through 6",
            "point_prediction_hit": (
                1 <= len(heads) <= 4 and all(2 <= layer <= 6 for layer in layers)
            ),
            "verdict": HELD if heads else FAILED,
        }
    )

    # P3. The emergence is abrupt on the grid: at most 3 checkpoints lie
    # strictly inside the interval.
    between = phase.get("n_grid_points_between")
    results.append(
        {
            "id": "P3",
            "primary": False,
            "statement": "at most 3 grid checkpoints lie strictly inside the crossing",
            "criterion": "fails if more than 3 intervening grid points",
            "n_grid_points_between": between,
            "max_allowed": 3,
            "verdict": HELD if crossed and between is not None and between <= 3 else FAILED,
        }
    )

    # P4. A previous-token head precedes the induction head.
    level = summary["prev_token_level"]
    low = trajectory["metadata"]["phase_bounds"][0]
    below = [p for p in points if p["max_prefix_matching"] < low]
    anchor = below[-1] if below else None
    earlier = (
        [p for p in points if int(p["tokens"]) <= int(anchor["tokens"])]
        if anchor is not None
        else []
    )
    first_prev = summary["first_prev_token_above"]
    results.append(
        {
            "id": "P4",
            "primary": False,
            "statement": (
                "a previous-token head is already above the level at the last "
                "checkpoint whose max prefix-matching score is still below 0.1"
            ),
            "criterion": (
                "fails if no head exceeds the previous-token level at or before "
                "that checkpoint"
            ),
            "prev_token_level": level,
            "anchor_tokens": None if anchor is None else int(anchor["tokens"]),
            "anchor_max_prefix_matching": (
                None if anchor is None else anchor["max_prefix_matching"]
            ),
            "anchor_max_prev_token": None if anchor is None else anchor["max_prev_token"],
            "anchor_max_prev_token_head": (
                None if anchor is None else anchor["max_prev_token_head"]
            ),
            "first_prev_token_above_tokens": first_prev,
            "verdict": (
                HELD
                if anchor is not None
                and any(p["max_prev_token"] > level for p in earlier)
                else FAILED
            ),
        }
    )

    # P5. In-context learning moves with the heads, not before them.
    icl_at = {
        int(p["tokens"]): p["icl"][P5_DOMAIN]["icl_score"]
        for p in points
        if P5_DOMAIN in p.get("icl", {})
    }
    steepest = summary["steepest_icl_step"].get(P5_DOMAIN)
    if crossed and interval["start_tokens"] in icl_at and end_tokens in icl_at:
        drop = icl_at[end_tokens] - icl_at[interval["start_tokens"]]
    else:
        drop = None
    if crossed and steepest is not None:
        end_position = positions[end_tokens]
        near = min(
            abs(positions[steepest["from_tokens"]] - end_position),
            abs(positions[steepest["to_tokens"]] - end_position),
        )
    else:
        near = None
    results.append(
        {
            "id": "P5",
            "primary": False,
            "statement": (
                "the in-context learning score on fineweb_edu falls by at least "
                "0.05 nats across the crossing, and its steepest step lands "
                "within one grid step of the crossing"
            ),
            "criterion": (
                "fails if the drop over the interval is under 0.05 nats, or if "
                "the steepest step lands more than one grid point away"
            ),
            "domain": P5_DOMAIN,
            "icl_at_start": icl_at.get(interval["start_tokens"]) if crossed else None,
            "icl_at_end": icl_at.get(end_tokens) if crossed else None,
            "drop_over_interval": drop,
            "min_drop_required": -P5_MIN_ICL_DROP,
            "drop_clause_holds": drop is not None and drop <= -P5_MIN_ICL_DROP,
            "steepest_step": steepest,
            "steepest_step_grid_distance": near,
            "locality_clause_holds": near is not None and near <= 1,
            "verdict": (
                HELD
                if drop is not None
                and drop <= -P5_MIN_ICL_DROP
                and near is not None
                and near <= 1
                else FAILED
            ),
        }
    )

    # P6. Copying behavior clears chance at or below 200,000,000 tokens.
    first_copy = summary["first_copying_above_chance"]
    at_copy = by_tokens.get(first_copy) if first_copy is not None else None
    results.append(
        {
            "id": "P6",
            "primary": False,
            "statement": (
                "copying exact-match accuracy clears chance, lower bound above "
                "chance, at or below 200,000,000 tokens"
            ),
            "criterion": (
                "fails if the copying interval has not cleared chance by 200M tokens"
            ),
            "first_copying_above_chance_tokens": first_copy,
            "first_copying_above_chance_grid_target": (
                targets.get(first_copy) if first_copy is not None else None
            ),
            "exact_match_there": None if at_copy is None else at_copy["copying"]["exact_match"],
            "exact_match_ci_there": (
                None
                if at_copy is None
                else [
                    at_copy["copying"]["exact_match_ci_low"],
                    at_copy["copying"]["exact_match_ci_high"],
                ]
            ),
            "chance": None if at_copy is None else at_copy["copying"]["chance"],
            "deadline_tokens": P6_DEADLINE_TOKENS,
            "verdict": (
                HELD
                if first_copy is not None
                and targets[first_copy] <= P6_DEADLINE_TOKENS
                else FAILED
            ),
        }
    )

    # P7. The OV circuit turns copying at or before the QK circuit turns matching.
    if crossed:
        crossing_point = by_tokens[end_tokens]
        scores = np.asarray(crossing_point["heads"]["prefix_matching"], dtype=np.float64)
        ov_at = np.asarray(crossing_point["heads"]["ov_copying_score"], dtype=np.float64)
        high = trajectory["metadata"]["phase_bounds"][1]
        crossing_heads = [
            (int(layer), int(head))
            for layer, head in zip(*np.nonzero(scores > high))
        ]
        primary = tuple(int(i) for i in crossing_point["max_prefix_matching_head"])
        end_position = positions[end_tokens]
        before = points[end_position - 1] if end_position > 0 else None

        def ov_history(head: tuple[int, int]) -> list[dict[str, Any]]:
            return [
                {
                    "tokens": int(p["tokens"]),
                    "ov_copying_score": float(
                        np.asarray(p["heads"]["ov_copying_score"], dtype=np.float64)[
                            head[0], head[1]
                        ]
                    ),
                    "prefix_matching": float(
                        np.asarray(p["heads"]["prefix_matching"], dtype=np.float64)[
                            head[0], head[1]
                        ]
                    ),
                }
                for p in points
                if int(p["tokens"]) <= end_tokens
            ]

        def adjacent(head: tuple[int, int]) -> bool:
            """Above the level at the crossing checkpoint or the one before it."""
            here = float(ov_at[head[0], head[1]])
            if here > P7_OV_LEVEL:
                return True
            if before is None:
                return False
            prior = float(
                np.asarray(before["heads"]["ov_copying_score"], dtype=np.float64)[
                    head[0], head[1]
                ]
            )
            return prior > P7_OV_LEVEL

        history = ov_history(primary)
        ever = [h for h in history if h["ov_copying_score"] > P7_OV_LEVEL]
        p7 = {
            "crossing_tokens": end_tokens,
            "primary_head": list(primary),
            "primary_head_prefix_matching": float(scores[primary[0], primary[1]]),
            "primary_head_ov_copying_score_at_crossing": float(
                ov_at[primary[0], primary[1]]
            ),
            "primary_head_ov_copying_score_one_grid_step_earlier": (
                None
                if before is None
                else float(
                    np.asarray(before["heads"]["ov_copying_score"], dtype=np.float64)[
                        primary[0], primary[1]
                    ]
                )
            ),
            "primary_head_ov_history_to_crossing": history,
            "heads_above_the_high_bound_at_crossing": [list(h) for h in crossing_heads],
            "their_prefix_matching": [float(scores[l, h]) for l, h in crossing_heads],
            "their_ov_copying_score": [float(ov_at[l, h]) for l, h in crossing_heads],
            "holds_for_any_crossing_head": any(adjacent(h) for h in crossing_heads),
            "ov_level": P7_OV_LEVEL,
            "verdict": HELD if adjacent(primary) else FAILED,
            "reading": (
                "the OV score is read at the crossing checkpoint and at the one "
                "grid step before it, which is what 'the checkpoint at or before "
                "its own crossing' designates. The whole history to the crossing "
                "is carried so the alternative reading is checkable."
            ),
            "alternative_reading_any_earlier_checkpoint": {
                "holds": bool(ever),
                "first_tokens_above_level": ever[0]["tokens"] if ever else None,
                "note": (
                    "true if the score exceeded the level at ANY earlier "
                    "checkpoint. Reported and not adopted: before the transition "
                    "this score wanders over most of its range at every head, so "
                    "a transient early excursion is not the OV circuit having "
                    "turned copying, and a verdict read off one would be "
                    "unfalsifiable."
                ),
            },
        }
    else:
        p7 = {"crossing_tokens": None, "ov_level": P7_OV_LEVEL, "verdict": FAILED}
    p7.update(
        {
            "id": "P7",
            "primary": False,
            "statement": (
                "the head that first crosses prefix-matching 0.3 already has an "
                "OV copying score above 0.5 at or before its own crossing"
            ),
            "criterion": (
                "fails if the copying score crosses 0.5 strictly after the "
                "prefix-matching crossing for that head"
            ),
        }
    )
    results.append(p7)

    # P8. The negative control.
    if control is None:
        results.append(
            {
                "id": "P8",
                "primary": False,
                "statement": "a random-init model shows none of this",
                "criterion": (
                    "fails if max prefix-matching reaches 0.05, or any head "
                    "reaches the registered 0.2, or the copying interval clears "
                    "chance"
                ),
                "verdict": NOT_SCORED,
                "note": "no --control trajectory supplied",
            }
        )
    else:
        point = control["points"][0]
        results.append(
            {
                "id": "P8",
                "primary": False,
                "statement": "a random-init model shows none of this",
                "criterion": (
                    "fails if max prefix-matching reaches 0.05, or any head "
                    "reaches the registered 0.2, or the copying interval clears "
                    "chance"
                ),
                "init_seed": control["metadata"].get("init_seed"),
                "n_sequences_used": control["metadata"].get("n_sequences_used"),
                "max_prefix_matching": point["max_prefix_matching"],
                "prefix_matching_chance": point["prefix_matching_chance"],
                "n_induction_heads": len(point["induction_heads"]),
                "copying_exact_match": point["copying"]["exact_match"],
                "copying_ci": [
                    point["copying"]["exact_match_ci_low"],
                    point["copying"]["exact_match_ci_high"],
                ],
                "copying_chance": point["copying"]["chance"],
                "copying_clears_chance": point["copying"]["clears_chance"],
                "verdict": (
                    HELD
                    if point["max_prefix_matching"] < P8_MAX_PREFIX_MATCHING
                    and not point["induction_heads"]
                    and not point["copying"]["clears_chance"]
                    else FAILED
                ),
            }
        )
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--control", type=Path, help="a --random-init trajectory, for P8")
    p.add_argument("--grid", type=Path, help="configs/stage_a_checkpoints.json")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    trajectory = load(args.trajectory)
    control = load(args.control) if args.control else None
    grid = load(args.grid) if args.grid else None

    meta = trajectory["metadata"]
    registered = bool(meta.get("at_registered_n_sequences"))
    results = score(trajectory, control, grid)

    result = {
        "metadata": {
            "trajectory": str(args.trajectory),
            "control": None if args.control is None else str(args.control),
            "grid": None if args.grid is None else str(args.grid),
            "trajectory_timestamp": meta.get("timestamp"),
            "n_sequences_used": meta.get("n_sequences_used"),
            "at_registered_n_sequences": registered,
            "reading_status": meta.get("reading_status"),
            "predictions": "docs/PROTOCOL.md",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "n_held": sum(1 for r in results if r["verdict"] == HELD),
        "n_failed": sum(1 for r in results if r["verdict"] == FAILED),
        "n_not_scored": sum(1 for r in results if r["verdict"] == NOT_SCORED),
        "predictions": results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    for r in results:
        print(f"{r['id']}: {r['verdict']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
