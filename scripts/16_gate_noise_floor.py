"""Measure the noise floor the cross-backend gate is applied on top of.

The pre-committed gate (scripts/11_compare_curves.py, tol-end 0.02, tol-mean 0.015)
compares a candidate training curve against the banked MLX reference and failed for
the torch-CUDA leg: end delta 0.061059, mean delta 0.043621.

That number is only interpretable against a scale, and the gate never established
one. torch and MLX seed their weights from different PRNGs, so a torch run and an
MLX run start from different points and their curves differ for reasons that have
nothing to do with backend fidelity. The pod's passing leg was MLX-vs-MLX at the
same seed, where the initial weights are identical and only device arithmetic
differs, which is why it reached 0.001119. The tolerance was calibrated there.

This script supplies the missing scale by comparing runs that differ only by seed:

  within-framework  torch seed i vs torch seed j, every pair. Backend, precision,
                    data and code are fixed, so this is pure initialization noise.
                    It is the floor below which NO comparison of two differently
                    seeded runs can go, whatever backend they are on.

  cross-framework   torch seed i vs the MLX reference, every seed. This is what the
                    gate actually measures.

The adjudication is a comparison of distributions, not of two numbers:

  If cross sits inside the spread of within, the gate cannot separate "different
  framework" from "different seed". It is mis-specified for a cross-framework
  comparison and its tolerance is unreachable by construction.

  If cross sits entirely above within, the frameworks diverge beyond
  initialization noise and that needs a cause before any run is launched.

  If they overlap but cross trends higher, say ambiguous and report it as such.
  Do not round an overlapping distribution into a verdict.

Usage:
  python scripts/16_gate_noise_floor.py --seed-logs <dir-of-train_logs>

The reference defaults to the committed banked log at
analysis/reference/abl_lr1.2e-3/train_log.jsonl; --reference overrides it.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ABL_LR12E3 = REPO_ROOT / "analysis" / "reference" / "abl_lr1.2e-3" / "train_log.jsonl"
DEFAULT_STEP_START = 400
DEFAULT_STEP_END = 500
TOL_END = 0.02
TOL_MEAN = 0.015
# The measured torch-vs-MLX gate result this analysis is explaining.
OBSERVED_GATE_END = 0.061059
OBSERVED_GATE_MEAN = 0.043621


def read_losses(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "step" in rec and "loss" in rec:
                out[int(rec["step"])] = float(rec["loss"])
    return out


def deltas(a: dict[int, float], b: dict[int, float], lo: int, hi: int):
    shared = sorted(s for s in (a.keys() & b.keys()) if lo <= s <= hi)
    if not shared:
        return None
    per = [abs(a[s] - b[s]) for s in shared]
    return {
        "n": len(shared),
        "end": abs(a[shared[-1]] - b[shared[-1]]),
        "mean": statistics.mean(per),
        "last_step": shared[-1],
    }


def describe(name: str, vals: list[float]) -> str:
    if not vals:
        return f"  {name}: no data"
    lo, hi = min(vals), max(vals)
    med = statistics.median(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else float("nan")
    return (
        f"  {name}: n={len(vals):<3} min={lo:.6f}  median={med:.6f}  "
        f"max={hi:.6f}  sd={sd:.6f}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed-logs", required=True, type=Path,
                   help="directory holding seedctl_s*_train_log.jsonl")
    p.add_argument("--reference", type=Path, default=REFERENCE_ABL_LR12E3)
    p.add_argument("--step-start", type=int, default=DEFAULT_STEP_START)
    p.add_argument("--step-end", type=int, default=DEFAULT_STEP_END)
    args = p.parse_args()

    logs = sorted(args.seed_logs.glob("seedctl_s*_train_log.jsonl"))
    if len(logs) < 2:
        raise SystemExit(f"need at least 2 seed logs, found {len(logs)} in {args.seed_logs}")
    runs = {pth.name.split("_")[1]: read_losses(pth) for pth in logs}
    ref = read_losses(args.reference)

    lo, hi = args.step_start, args.step_end
    print(f"noise-floor analysis over steps [{lo}, {hi}]")
    print(f"  reference (MLX): {args.reference}")
    print(f"  torch seeds:     {', '.join(sorted(runs))}  ({len(runs)} runs)")
    print()

    print("within-framework (torch vs torch, differs only by seed = initialization noise)")
    within_end, within_mean = [], []
    for a, b in itertools.combinations(sorted(runs), 2):
        d = deltas(runs[a], runs[b], lo, hi)
        if not d:
            continue
        within_end.append(d["end"])
        within_mean.append(d["mean"])
        print(f"  {a} vs {b}: end={d['end']:.6f}  mean={d['mean']:.6f}  (n={d['n']})")
    print()
    print(describe("within end ", within_end))
    print(describe("within mean", within_mean))
    print()

    print("cross-framework (torch vs the banked MLX reference = what the gate measures)")
    cross_end, cross_mean = [], []
    for s in sorted(runs):
        d = deltas(runs[s], ref, lo, hi)
        if not d:
            continue
        cross_end.append(d["end"])
        cross_mean.append(d["mean"])
        print(f"  {s} vs MLX: end={d['end']:.6f}  mean={d['mean']:.6f}  (n={d['n']})")
    print()
    print(describe("cross end ", cross_end))
    print(describe("cross mean", cross_mean))
    print()

    print("adjudication")
    print(f"  pre-committed tolerances:      end {TOL_END}, mean {TOL_MEAN}")
    print(f"  observed failing gate result:  end {OBSERVED_GATE_END}, mean {OBSERVED_GATE_MEAN}")
    if within_end:
        print(f"  within-framework end spread:   {min(within_end):.6f} .. {max(within_end):.6f}")
        n_over = sum(1 for v in within_end if v > TOL_END)
        print(f"  within-framework pairs that would THEMSELVES fail the end tolerance: "
              f"{n_over}/{len(within_end)}")
        n_over_m = sum(1 for v in within_mean if v > TOL_MEAN)
        print(f"  within-framework pairs that would THEMSELVES fail the mean tolerance: "
              f"{n_over_m}/{len(within_mean)}")
    if within_end and cross_end:
        overlap = min(cross_end) <= max(within_end)
        print()
        if overlap and statistics.median(cross_end) <= max(within_end):
            print("  reading: cross-framework deltas fall inside the within-framework")
            print("  spread. The gate cannot separate a different framework from a")
            print("  different seed, so its tolerance is unreachable by construction")
            print("  and the gate is MIS-specified for a cross-framework comparison.")
        elif not overlap:
            print("  reading: cross-framework deltas sit entirely above the")
            print("  within-framework spread. The divergence is real and needs a cause.")
        else:
            print("  reading: distributions overlap but cross trends higher. AMBIGUOUS.")
            print("  Report as ambiguous; do not round this into a verdict.")
    print()
    print("  note: a tolerance for any replacement gate must be derived from the")
    print("  within-framework spread above and written down before that gate runs.")


if __name__ == "__main__":
    main()
