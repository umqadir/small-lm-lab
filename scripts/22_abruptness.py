"""Apply the registered abruptness criterion to a measured trajectory.

The 2026-07-26 amendment to docs/PROTOCOL.md fixes three quantities and
a three-way verdict: G, the grid intervals the crossing spans; W, its width in
decades of tokens; F, the share of the run's total FineWeb-Edu validation-loss
improvement that falls inside it. ABRUPT is G <= 3 and F <= 0.10, GRADUAL is
G >= 7 or F >= 0.30, and anything else is INDETERMINATE AT THIS RESOLUTION.

This script is the adjudication and owns no arithmetic. It reads the two
artifacts, hands them to emergence.abruptness_verdict, and writes what comes
back. Everything it needs was already measured: the trajectory JSON from
scripts/19_emergence.py supplies the maximum prefix-matching score per
checkpoint and the crossing interval, and the curve JSON from
scripts/21_val_curve.py supplies the validation loss per checkpoint. Neither is
recomputed here, and no threshold is a flag.

Section 4 of the same amendment registers the loss-bump detector, which reads
the same validation curve, so it is run here too and written per domain beside
the verdict. Olsson et al. report the induction phase change alongside a visible
bump in the training loss; the detector is what turns the presence or absence of
one in this run into a number rather than an impression of a figure.

A run whose trajectory was measured below the registered sequence count is
refused. The verdict is a registered claim, the registration fixes 256
sequences, and scripts/19_emergence.py records on every file whether it met
that. --allow-monitoring runs it anyway and stamps the output as a monitoring
reading, for the case where a coarse look is wanted before the real sweep
finishes.

Usage:
  python scripts/22_abruptness.py --trajectory TRAJ.json --curve CURVE.json \
      --out OUT.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from small_lm_lab import emergence


def read_trajectory(path: Path) -> tuple[dict[int, float], dict]:
    """{token count: max prefix-matching score} and the file's metadata."""
    with path.open() as f:
        data = json.load(f)
    scores = {
        int(p["tokens"]): float(p["max_prefix_matching"])
        for p in data["points"]
        if p.get("tokens") is not None
    }
    return scores, data.get("metadata", {})


def read_curve(path: Path) -> tuple[dict[str, dict[int, float]], dict]:
    """{domain: {token count: validation loss}} and the file's metadata."""
    with path.open() as f:
        data = json.load(f)
    curves: dict[str, dict[int, float]] = {}
    for point in data["points"]:
        if point.get("tokens") is None:
            continue
        for domain, record in point["domains"].items():
            curves.setdefault(domain, {})[int(point["tokens"])] = float(
                record["mean_nll"]
            )
    return curves, data.get("metadata", {})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--curve", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--allow-monitoring",
        action="store_true",
        help=(
            "adjudicate a trajectory measured below the registered sequence "
            "count. The output is stamped as a monitoring reading and no "
            "registered claim may be read from it"
        ),
    )
    args = p.parse_args()

    scores, trajectory_meta = read_trajectory(args.trajectory)
    curves, curve_meta = read_curve(args.curve)
    if emergence.ABRUPTNESS_LOSS_DOMAIN not in curves:
        raise SystemExit(
            f"{args.curve} carries no {emergence.ABRUPTNESS_LOSS_DOMAIN} loss, "
            "which is the domain the comparator F is read on"
        )
    losses = curves[emergence.ABRUPTNESS_LOSS_DOMAIN]

    registered = bool(trajectory_meta.get("at_registered_n_sequences"))
    if not registered and not args.allow_monitoring:
        raise SystemExit(
            f"{args.trajectory} was measured at "
            f"{trajectory_meta.get('n_sequences_used')} sequences, not the "
            f"registered {trajectory_meta.get('prefix_matching_n_sequences')}. "
            "The abruptness verdict is a registered claim. Pass "
            "--allow-monitoring to adjudicate it anyway, as a monitoring read."
        )

    verdict = emergence.abruptness_verdict(scores, losses)
    result = {
        "metadata": {
            "trajectory": str(args.trajectory),
            "curve": str(args.curve),
            "trajectory_timestamp": trajectory_meta.get("timestamp"),
            "curve_timestamp": curve_meta.get("timestamp"),
            "n_sequences_used": trajectory_meta.get("n_sequences_used"),
            "at_registered_n_sequences": registered,
            "reading_status": (
                "registered"
                if registered
                else "MONITORING, reduced sequence count, not a reportable figure"
            ),
            "loss_statistic": curve_meta.get("statistic"),
            "loss_max_windows": curve_meta.get("max_windows"),
            "loss_split": curve_meta.get("split"),
            "criterion": (
                "docs/PROTOCOL.md, amendment 2026-07-26, section 1. "
                "ABRUPT: G <= 3 and F <= 0.10. GRADUAL: G >= 7 or F >= 0.30. "
                "Anything else: INDETERMINATE AT THIS RESOLUTION."
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "abruptness": asdict(verdict),
        # Registered by the same amendment, section 4. Reported per domain and
        # located against the crossing interval above, whatever it shows.
        "loss_bump": {
            domain: asdict(emergence.loss_bump(curve, domain=domain))
            for domain, curve in sorted(curves.items())
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(verdict.verdict)
    print(verdict.note)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
