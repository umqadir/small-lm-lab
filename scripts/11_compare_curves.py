"""Compare two training-log curves at matching steps, for the cross-backend gate.

The staged run trains on a rented RTX 4090 through MLX's CUDA backend, a
different machine and backend from the one every prior run and every completed
ablation used. Before the run starts, the pre-registered validation gate has to
show the new backend reproduces a known run. One leg of that gate is a same-seed
rerun of the banked abl_lr1.2e-3 ablation, held to a tolerance committed in
advance: the loss must match at the end of the window and on average across it.

This script implements that comparison against the repo's own training-log
format (the JSONL that train.py writes, one record per logged step with a
"step" and a "loss"). It reads both logs as last-record-wins per step, since a
resumed run can log a step more than once, then over a step window compares:

  end delta   |loss difference| at the final step both logs share inside the
              window. The pre-registered bound is 0.02.
  mean delta  the mean of |loss difference| over every shared step in the
              window. The pre-registered bound is 0.015.

It exits nonzero, with a report either way, if either bound is exceeded, so it
can gate the run from a shell script. The tolerances are pre-committed
(amendment 2026-07-21) and are the script's defaults; they are not tuned to a
result.

The reference for the real gate is the banked ablation log, committed at
analysis/reference/abl_lr1.2e-3/train_log.jsonl (REFERENCE_ABL_LR12E3 below) so
the comparison can be rerun from a clone. The candidate is the new backend's
train_log.

Usage:
  python scripts/11_compare_curves.py \
      --candidate /path/to/new/backend/rerun/train_log.jsonl \
      --step-start 400 --step-end 500 --tol-end 0.02 --tol-mean 0.015
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

# The banked reference for the real gate: the abl_lr1.2e-3 ablation, committed in
# the repository so the comparison reruns from a clone with no bulk data present.
REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ABL_LR12E3 = REPO_ROOT / "analysis" / "reference" / "abl_lr1.2e-3" / "train_log.jsonl"

# Pre-committed tolerances (amendment 2026-07-21).
DEFAULT_STEP_START = 400
DEFAULT_STEP_END = 500
DEFAULT_TOL_END = 0.02
DEFAULT_TOL_MEAN = 0.015


def read_losses(path: Path) -> dict[int, float]:
    """{step -> loss} from a train_log.jsonl, last record per step winning.

    A killed-and-resumed run appends its replayed steps, so a step can appear
    more than once; the later record is the one that actually ran, matching how
    train.py documents its own log. Records without a step or a loss are skipped.
    """
    losses: dict[int, float] = {}
    with Path(path).open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "step" not in record or "loss" not in record:
                continue
            losses[int(record["step"])] = float(record["loss"])
    if not losses:
        raise SystemExit(f"{path} holds no usable (step, loss) records")
    return losses


@dataclass(frozen=True)
class CurveComparison:
    step_start: int
    step_end: int
    tol_end: float
    tol_mean: float
    common_steps: list[int]
    end_step: int
    end_delta: float
    mean_delta: float
    passed: bool


def compare_logs(
    reference: Path,
    candidate: Path,
    step_start: int,
    step_end: int,
    tol_end: float,
    tol_mean: float,
) -> CurveComparison:
    """Compare two logs over [step_start, step_end], both bounds inclusive."""
    ref = read_losses(reference)
    cand = read_losses(candidate)

    common = sorted(
        s for s in (ref.keys() & cand.keys()) if step_start <= s <= step_end
    )
    if not common:
        raise SystemExit(
            f"no step in [{step_start}, {step_end}] is present in both logs, so "
            "there is nothing to compare. Check the step window and that both "
            "runs logged at the same cadence."
        )

    abs_deltas = [abs(ref[s] - cand[s]) for s in common]
    end_step = common[-1]
    end_delta = abs(ref[end_step] - cand[end_step])
    mean_delta = sum(abs_deltas) / len(abs_deltas)
    passed = (
        math.isfinite(end_delta)
        and math.isfinite(mean_delta)
        and end_delta <= tol_end
        and mean_delta <= tol_mean
    )
    return CurveComparison(
        step_start=step_start,
        step_end=step_end,
        tol_end=tol_end,
        tol_mean=tol_mean,
        common_steps=common,
        end_step=end_step,
        end_delta=end_delta,
        mean_delta=mean_delta,
        passed=passed,
    )


def format_report(cmp: CurveComparison, reference: Path, candidate: Path) -> str:
    end_ok = cmp.end_delta <= cmp.tol_end
    mean_ok = cmp.mean_delta <= cmp.tol_mean
    lines = [
        f"cross-backend curve comparison over steps "
        f"[{cmp.step_start}, {cmp.step_end}]",
        f"  reference: {reference}",
        f"  candidate: {candidate}",
        f"  {len(cmp.common_steps)} shared steps in window "
        f"(first {cmp.common_steps[0]}, last {cmp.end_step})",
        f"  end delta  |ref - cand| at step {cmp.end_step}: {cmp.end_delta:.6f} "
        f"(tol {cmp.tol_end}) -> {'PASS' if end_ok else 'FAIL'}",
        f"  mean delta over the window:              {cmp.mean_delta:.6f} "
        f"(tol {cmp.tol_mean}) -> {'PASS' if mean_ok else 'FAIL'}",
        f"  verdict: {'PASS' if cmp.passed else 'FAIL'}",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--reference",
        type=Path,
        default=REFERENCE_ABL_LR12E3,
        help=(
            "reference train_log.jsonl. Defaults to the banked abl_lr1.2e-3 log, "
            "the run the cross-backend gate checks the RTX 4090 rerun against"
        ),
    )
    p.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="candidate train_log.jsonl, for example the RTX 4090 rerun's log",
    )
    p.add_argument("--step-start", type=int, default=DEFAULT_STEP_START)
    p.add_argument("--step-end", type=int, default=DEFAULT_STEP_END)
    p.add_argument(
        "--tol-end",
        type=float,
        default=DEFAULT_TOL_END,
        help=f"max |loss delta| at the final shared step. Pre-committed "
        f"{DEFAULT_TOL_END}",
    )
    p.add_argument(
        "--tol-mean",
        type=float,
        default=DEFAULT_TOL_MEAN,
        help=f"max mean |loss delta| over the window. Pre-committed "
        f"{DEFAULT_TOL_MEAN}",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    cmp = compare_logs(
        args.reference,
        args.candidate,
        args.step_start,
        args.step_end,
        args.tol_end,
        args.tol_mean,
    )
    print(format_report(cmp, args.reference, args.candidate))
    raise SystemExit(0 if cmp.passed else 1)


if __name__ == "__main__":
    main()
