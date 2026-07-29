"""Run the pre-registered Phase B interpretability protocol against checkpoints.

Writes a single JSON holding every Phase B number, its bootstrap interval, and
enough metadata to say what produced it: the checkpoint, the config, the token
count the checkpoint recorded, the device, the seeds, and a timestamp.
Re-runnable: the same arguments against the same checkpoints reproduce the same
file, because every bootstrap and every random draw is seeded.

Four analyses, all fixed by docs/PROTOCOL.md:

  prefix_matching  the induction head score per layer and head, with the heads
                   above the pre-registered 0.2 threshold named.
  ablation         mean-ablate each induction head singly, all of them jointly,
                   and an equal number of matched control heads from the same
                   layers. Copying accuracy, in-context learning score and
                   perplexity, each with a paired interval on the damage. The
                   mean is per position, which the amendment fixed as the
                   headline; --global-mean-ablation adds the global-mean variant
                   alongside as the sensitivity check.
  patching         clean head outputs spliced into a corrupted copying run,
                   scored by the pre-registered logit difference: the correct
                   continuation against the token the corrupted prefix would
                   induct to. The raw logit on the correct continuation rides
                   along as a labeled secondary number. The recovered
                   fraction is withheld, as null with a status saying why,
                   unless the copying positive control clears chance and the
                   clean-minus-corrupted gap resolves; --patch-escalate is the
                   one pre-registered rerun at a higher sequence count, and is
                   refused while the copying gate is shut.
  phase_change     the token interval over which the max prefix-matching score
                   goes from below 0.1 to above 0.3. Needs a trajectory, so it
                   needs --checkpoint-dir.

The 0.2 threshold, the 0.1 and 0.3 bounds, the 256 sequences behind the score,
the copying task, the per-position mean the ablation headline runs on and the
patching metric are not flags. They were fixed before any checkpoint existed and
they are read from small_lm_lab.interp, so no invocation of this script can move
one.

--checkpoint-dir sweeps every tokens_*.pt in a run directory and produces the
per-checkpoint prefix-matching trajectory the phase change is read from.

The causal analyses (ablation and patching) run at the interp anchors, which the
staged-run amendment fixes as several checkpoints across the emergence window
instead of the final one alone. Which checkpoints they run on is chosen
explicitly, never silently: over a --checkpoint-dir a causal analysis requires
one of --all-checkpoints (every swept checkpoint), --checkpoints (an explicit
list of checkpoint paths or token counts), or --final-only (the trained model
alone, which was the previous default and is now spelled out). A single
--checkpoint runs the causal battery on that checkpoint. Running the causal
battery only on the final checkpoint contradicted the registration's
"induction-head formation across checkpoints", so the correction is treated as a
bug fix and not a deviation.

The test splits stay closed until the final evaluation, so --split defaults to
val and test is only read when asked for by name. Nothing here writes to the
token streams or to the training lock.

Usage:
  python scripts/08_interp.py --config size30m --checkpoint PATH \
      --analysis prefix_matching --out OUT.json
  python scripts/08_interp.py --config size30m --checkpoint-dir DIR \
      --analysis all --all-checkpoints --out OUT.json
  python scripts/08_interp.py --config size30m --checkpoint-dir DIR \
      --analysis all --checkpoints 128000000 2000000000 --out OUT.json
  python scripts/08_interp.py --config size30m --checkpoint-dir DIR \
      --analysis all --final-only --out OUT.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from small_lm_lab import interp
from small_lm_lab.config import get_config, n_params
from small_lm_lab.data import DEFAULT_TOKENIZED_ROOT
from small_lm_lab import evaluate
from small_lm_lab.evaluate import DOMAINS
from small_lm_lab.train import CHECKPOINT_PREFIX

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = REPO_ROOT / "scripts" / "06_evaluate.py"

ANALYSES = ("prefix_matching", "ablation", "patching", "phase_change", "all")


def load_eval_script():
    """Import scripts/06_evaluate.py by path, for its checkpoint loader.

    The loader is not something to keep a second copy of. It already knows the
    checkpoint layout, strips a torch.compile prefix, loads with strict=True and
    casts bf16 storage up to float32 for evaluation. A copy here would be one
    more thing to keep in step with train.py. scripts/ is not a package, hence
    the path import; tests/test_train.py reaches for the same loader the same
    way.
    """
    spec = importlib.util.spec_from_file_location("eval_script", EVAL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_checkpoints(checkpoint_dir: Path) -> list[Path]:
    """Every pre-registered checkpoint in a run directory, oldest first.

    train.py zero pads the token count into the filename precisely so a lexical
    sort is a chronological one. resume_state.pt is not a checkpoint and does
    not match the prefix.
    """
    paths = sorted(Path(checkpoint_dir).glob(f"{CHECKPOINT_PREFIX}*.pt"))
    if not paths:
        raise FileNotFoundError(
            f"no {CHECKPOINT_PREFIX}*.pt checkpoints under {checkpoint_dir}"
        )
    return paths


def to_jsonable(obj: Any) -> Any:
    """Dataclasses, numpy scalars and arrays, and tuple keys into plain JSON.

    Non-finite floats become null. json.dump would otherwise write the bare
    tokens NaN, Infinity and -Infinity, which Python reads back happily and the
    JSON spec does not allow: a strict parser rejects the file outright and jq
    quietly rewrites the value. A result file that only its author's language
    can read is not a result file, and one that changes value depending on who
    reads it is worse.

    Withheld numbers are the common case, not a corner: recovered_fraction is
    withheld on any run whose clean-minus-corrupted gap does not clear zero. So
    null is load-bearing here, and null alone would say only that a number is
    absent, not why. That is what the status fields alongside are for; see
    patching_report. This function's job is narrower, and it is to make sure the
    file parses.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        # Recurse instead of returning: a numpy float can be NaN too.
        return to_jsonable(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def prefix_matching_report(
    model: torch.nn.Module, args: argparse.Namespace
) -> dict[str, Any]:
    """The induction head score for one checkpoint, with per-head intervals."""
    per_sequence = interp.prefix_matching_per_sequence(
        model,
        args.device,
        seed=args.seed,
        batch_size=args.weights_batch_size,
    )
    scores = per_sequence.mean(axis=0)
    lo, hi = interp.prefix_matching_intervals(
        per_sequence, n_resamples=args.n_resamples, seed=args.seed
    )
    induction = interp.identify_induction_heads(scores)
    return {
        "threshold": interp.INDUCTION_THRESHOLD,
        "n_sequences": interp.PREFIX_MATCHING_N_SEQUENCES,
        "repeat_len": interp.COPY_REPEAT_LEN,
        "scores": to_jsonable(scores),
        "ci_low": to_jsonable(lo),
        "ci_high": to_jsonable(hi),
        "max_score": float(scores.max()),
        "argmax": [int(i) for i in np.unravel_index(int(scores.argmax()), scores.shape)],
        "induction_heads": [list(h) for h in induction],
    }


def ablation_report(
    model: torch.nn.Module,
    args: argparse.Namespace,
    induction_heads: list[tuple[int, int]],
    scores: np.ndarray,
) -> dict[str, Any]:
    """Every pre-registered ablation condition against one shared baseline.

    The headline runs per-position means. The amendment of 2026-07-17, item 1,
    fixed that reading, so it is not a flag: the whole reason the threshold and
    the paradigm list are hardcoded applies here too. --global-mean-ablation
    adds the global-mean variant the same amendment names as a sensitivity
    check, under its own key and marked as not the headline.
    """
    if not induction_heads:
        return {
            "ran": False,
            "note": (
                "no head scores above the pre-registered threshold of "
                f"{interp.INDUCTION_THRESHOLD}, so there is nothing to ablate and "
                "nothing to match a control against. Reported as a negative "
                "result for the phase rather than by lowering the threshold."
            ),
        }

    controls = interp.matched_control_heads(scores, induction_heads, args.control_seed)

    def conditions(per_position: bool) -> dict[str, Any]:
        # One context per choice of mean, built once and read by every condition
        # under it, so every delta is measured against the same baseline on the
        # same sequences. The two choices of mean are two baselines and cannot
        # share one.
        context = interp.build_ablation_context(
            model,
            args.device,
            split=args.split,
            domains=args.domains,
            seed=args.seed,
            batch_size=args.batch_size,
            root=args.tokenized_root,
            per_position_means=per_position,
            max_windows=args.max_windows,
        )

        def run(heads: list[tuple[int, int]]) -> dict[str, Any]:
            return to_jsonable(
                interp.ablation_effect(
                    model,
                    heads,
                    args.device,
                    context=context,
                    n_resamples=args.n_resamples,
                )
            )

        return {
            "mean_ablation": "per_position" if per_position else "global",
            "per_position_means": per_position,
            # The pre-registered claim: these two, compared on copying accuracy.
            "induction_joint": run(induction_heads),
            "control_joint": run(controls),
            "induction_singles": [run([head]) for head in induction_heads],
            "control_singles": [run([head]) for head in controls],
        }

    report: dict[str, Any] = {
        "ran": True,
        "control_seed": args.control_seed,
        "induction_heads": [list(h) for h in induction_heads],
        "control_heads": [list(h) for h in controls],
        "control_head_scores": [float(scores[l, h]) for l, h in controls],
        "primary_mean_ablation": "per_position",
        "note": (
            "the pre-registration amendment of 2026-07-17, item 1, fixes mean "
            "ablation as per position. Those are the numbers at this level and "
            "they are the headline. sensitivity_global_mean holds the "
            "global-mean variant the same amendment names as a sensitivity "
            "check; it is null unless --global-mean-ablation was passed."
        ),
    }
    report.update(conditions(per_position=True))
    report["sensitivity_global_mean"] = (
        conditions(per_position=False) if args.global_mean_sensitivity else None
    )
    return report


def patching_report(
    model: torch.nn.Module,
    args: argparse.Namespace,
    induction_heads: list[tuple[int, int]],
    scores: np.ndarray,
) -> dict[str, Any]:
    """Activation patching over the heads the claim is about, or over all.

    The result carries two families of numbers and only one of them is the
    result, so the metric block below says which in words instead of leaving it
    to be inferred from a field name.
    """
    heads: Optional[list[tuple[int, int]]] = None
    controls: list[tuple[int, int]] = []
    if not args.patch_all_heads:
        if not induction_heads:
            return {
                "ran": False,
                "note": (
                    "no head scores above the pre-registered threshold, so there "
                    "is no induction-versus-control comparison to patch. Rerun "
                    "with --patch-all-heads to sweep every head anyway."
                ),
            }
        controls = interp.matched_control_heads(
            scores, induction_heads, args.control_seed
        )
        heads = list(induction_heads) + list(controls)

    # The primary gate's reading, measured once here rather than inside the
    # instrument, so the number that decides is the pre-registered copying
    # accuracy from the battery and is recorded next to the decision it drove.
    copying = evaluate.copying_eval(
        model,
        args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        n_resamples=args.n_resamples,
    )
    control = interp.copying_positive_control(copying)
    clears = interp.copying_clears_chance(control, int(model.cfg.vocab_size))

    # Third amendment, point 3: escalation is reachable only once the copying
    # gate is open. On a model that does not copy, more sequences resolve a
    # meaningless gap more sharply and make the output worse, so this refuses
    # rather than quietly running at the lower count.
    if args.patch_escalate and not clears:
        raise SystemExit(
            "--patch-escalate is unreachable while the copying positive control "
            f"is at chance: exact-match {control.point:.6g} with interval "
            f"[{control.ci_low:.6g}, {control.ci_high:.6g}] against chance "
            f"{interp.copy_chance_accuracy(int(model.cfg.vocab_size)):.6g}. The "
            "escalation exists for a model that demonstrably copies but whose "
            "gap will not resolve. Where the model does not copy, the recovered "
            "fraction is withheld at any sequence count and more sequences only "
            "sharpen a meaningless gap, so this is a negative result to report "
            "rather than a measurement to buy more of."
        )

    n_sequences = (
        interp.PATCHING_ESCALATED_N_SEQUENCES
        if args.patch_escalate
        else args.patch_sequences
    )
    result = interp.activation_patching(
        model,
        args.device,
        heads=heads,
        n_sequences=n_sequences,
        seed=args.seed,
        batch_size=args.batch_size,
        n_resamples=args.n_resamples,
        copying_exact_match=control,
    )
    return {
        "ran": True,
        "all_heads": bool(args.patch_all_heads),
        "power": {
            "n_sequences": int(n_sequences),
            "pre_registered_n_sequences": interp.PATCHING_N_SEQUENCES,
            "escalated": bool(args.patch_escalate),
            "escalated_n_sequences": interp.PATCHING_ESCALATED_N_SEQUENCES,
            "copying_gate_open": bool(clears),
            "escalation_reachable": bool(clears),
            "rule": (
                "the second pre-registration amendment of 2026-07-17. A withheld "
                "recovered_fraction means two different things and copying "
                "exact-match accuracy is the positive control that separates "
                "them. If copying is at or near chance there is no in-context "
                "copying to explain and a withheld fraction is a genuine "
                "negative result, reported as one. If the copying interval lies "
                "entirely above chance the behavior exists and a withheld fraction "
                "means the measurement is underpowered, and only then is patching re-run "
                "once at escalated_n_sequences via --patch-escalate. Still "
                "straddling zero at that count while the model demonstrably "
                "copies is reported as a limit of this instrument at this scale, "
                "explicitly not as evidence against induction, and there is no "
                "further escalation. The deciding copying reading is "
                "ablation.induction_joint.copying_exact_match.baseline here, or "
                "scripts/06_evaluate.py."
            ),
            "note": (
                "escalated records whether this run was the declared escalation, "
                "not merely whether n_sequences happens to equal "
                "escalated_n_sequences. The escalation turns on a judgment about "
                "a copying reading and nothing in this script makes it: it is "
                "asked for explicitly or it did not happen."
            ),
        },
        "metric": {
            "primary": interp.PATCHING_METRIC,
            "primary_fields": [
                "result.clean_logit_diff",
                "result.corrupted_logit_diff",
                "result.gap (point estimate and bootstrap interval)",
                "result.gap_excludes_zero",
                "result.heads[].patched_logit_diff",
                "result.heads[].recovered_logit_diff",
                "result.heads[].recovered_fraction",
            ],
            "fraction_gate": (
                "pre-registration amendment item 5. recovered_fraction is null "
                "for every head unless result.gap_excludes_zero, that is unless "
                "the clean-minus-corrupted gap's bootstrap interval clears zero. "
                "The fraction divides by that gap, and a ratio of a gap that is "
                "not distinguishable from zero is noise over noise that reads as "
                "a finding: on a model with no induction heads it returns values "
                "near 1 with nothing behind them. recovered_logit_diff is the "
                "same recovery unnormalized, is reported either way with its "
                "interval, and the induction-versus-control comparison can be "
                "read from it directly. The gap is a property of the clean and "
                "corrupted runs alone, so this gate applies to every head at "
                "once and cannot favor one head over another."
            ),
            "fraction_status_values": list(interp.FRACTION_STATUSES),
            "fraction_status_note": (
                "every head carries recovered_fraction_status. A null "
                "recovered_fraction says a number is absent; the status says "
                "why, so absent-because-withheld is never confused with "
                "absent-because-missing. The status is "
                f"{interp.FRACTION_REPORTED!r} if and only if "
                "recovered_fraction is non-null."
            ),
            "primary_gate": (
                "the copying positive control, third amendment. The fraction is "
                "withheld unless the interval on copying exact-match accuracy "
                "lies entirely above chance, because a model that does not copy "
                "has no in-context copying to recover and no arithmetic on the "
                "gap can supply any. The reading that decided is recorded at "
                "result.copying_exact_match against result.copying_chance, so "
                "the gate can be checked and not merely trusted. Passing no "
                "reading fails closed."
            ),
            "gap_gate_limit": (
                "the gap interval test is necessary but NOT sufficient and is "
                "not evidence that any fraction is meaningful. It tests whether "
                "the gap differs from zero, which is not the question. A "
                "random-init model's gap is small but systematic rather than "
                "noise, so more sequences resolve it better: measured on tiny "
                "random-init models this test alone opened on 47 percent of "
                "them at 256 sequences, and escalating to 1024 opens it more, "
                "not less. It is kept only for what it can do, which is catch a "
                "degenerate or unresolvable denominator. gap_excludes_zero true "
                "means the denominator was resolvable, NOT that the recovery is "
                "meaningful."
            ),
            "secondary": interp.PATCHING_SECONDARY_METRIC,
            "secondary_fields": [
                "result.clean_correct_logit",
                "result.corrupted_correct_logit",
                "result.correct_logit_gap",
                "result.heads[].patched_correct_logit",
                "result.heads[].recovered_correct_logit",
            ],
            "note": (
                "the pre-registration amendment of 2026-07-17, item 2, fixes the "
                "logit difference as the metric the causal claim is read from. "
                "The raw logit on the correct continuation is the first term of "
                "that difference on its own, so it is free to report, and it is "
                "reported only as a secondary number: a logit is defined up to a "
                "per-position additive constant, so differencing a raw logit "
                "across two runs mixes that constant in. No result is read from "
                "the secondary fields."
            ),
        },
        "induction_heads": [list(h) for h in induction_heads],
        "control_heads": [list(h) for h in controls],
        "result": to_jsonable(result),
    }


def validate_causal_selection(args: argparse.Namespace, want_causal: bool) -> None:
    """Refuse an ambiguous causal-checkpoint selection, before any work is done.

    The rule the amendment fixes: a causal analysis over a --checkpoint-dir has
    to say which checkpoints it runs on. The previous implicit final-only default
    is now --final-only, so an old invocation errors loudly instead of quietly
    keeping the behavior the registration calls a bug. A single --checkpoint is
    unambiguous and needs no flag; the multi-checkpoint flags then make no sense
    and are refused so a reader is never misled about what was run.
    """
    single = args.checkpoint is not None
    if single:
        if args.all_checkpoints or args.causal_checkpoints:
            raise SystemExit(
                "--all-checkpoints and --checkpoints select from a "
                "--checkpoint-dir sweep. With a single --checkpoint the causal "
                "battery runs on that checkpoint; pass --checkpoint-dir to select "
                "among several."
            )
        return
    if want_causal and not (
        args.all_checkpoints or args.causal_checkpoints or args.final_only
    ):
        raise SystemExit(
            "a causal analysis over --checkpoint-dir must say which checkpoints "
            "to run it on. Pass --all-checkpoints (every swept checkpoint, the "
            "interp anchors across training), --checkpoints <paths or token "
            "counts>, or --final-only (the trained model alone, which was the "
            "previous default and is now explicit)."
        )


def select_causal_checkpoints(
    args: argparse.Namespace, trajectory: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The trajectory entries the causal battery runs on, in trajectory order.

    Each returned entry already carries the checkpoint path, its token count, its
    prefix-matching scores and the induction heads named at the pre-registered
    threshold, so the causal analyses do not recompute any of it.
    """
    if args.checkpoint is not None or args.final_only:
        # A single checkpoint, or the trained model alone.
        return [trajectory[-1]]
    if args.all_checkpoints:
        return list(trajectory)

    # An explicit list of paths or token counts, resolved against the sweep.
    selected: list[dict[str, Any]] = []
    for item in args.causal_checkpoints:
        selected.append(_match_causal_item(item, trajectory))
    return selected


def _match_causal_item(item: str, trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    candidate = Path(item)
    if candidate.exists():
        target = candidate.resolve()
        for entry in trajectory:
            if Path(entry["checkpoint"]).resolve() == target:
                return entry
        raise SystemExit(
            f"--checkpoints path {item} is not among the checkpoints swept from "
            "--checkpoint-dir"
        )
    try:
        tokens = int(item)
    except ValueError:
        raise SystemExit(
            f"--checkpoints entry {item!r} is neither an existing checkpoint path "
            "nor an integer token count"
        )
    for entry in trajectory:
        if entry["tokens"] == tokens:
            return entry
    available = sorted(e["tokens"] for e in trajectory if e["tokens"] is not None)
    raise SystemExit(
        f"--checkpoints token count {tokens} not found among the swept "
        f"checkpoints; available token counts: {available}"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=Path, help="path to one saved checkpoint")
    src.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="a run's checkpoint directory; every checkpoint is swept",
    )
    causal = p.add_mutually_exclusive_group(required=False)
    causal.add_argument(
        "--all-checkpoints",
        action="store_true",
        help=(
            "run the causal battery (ablation, patching) at every swept "
            "checkpoint of --checkpoint-dir, the interp anchors across the "
            "emergence window. Required, with --checkpoints or --final-only, "
            "whenever a causal analysis runs over a --checkpoint-dir"
        ),
    )
    causal.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        dest="causal_checkpoints",
        help=(
            "run the causal battery at these checkpoints only, given as "
            "checkpoint file paths or as integer token counts that must match a "
            "swept checkpoint. Only meaningful with --checkpoint-dir"
        ),
    )
    causal.add_argument(
        "--final-only",
        action="store_true",
        help=(
            "run the causal battery on the final checkpoint alone. This was the "
            "previous implicit default over a --checkpoint-dir and is now spelled "
            "out, so an old invocation errors rather than silently changing "
            "meaning"
        ),
    )
    p.add_argument("--config", required=True, help="named config, for example size30m")
    p.add_argument("--device", default="cpu", help="cpu, mps, or cuda")
    p.add_argument("--out", type=Path, required=True, help="path for the JSON result")
    p.add_argument("--analysis", choices=ANALYSES, default="all")
    p.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="test is the closed split and is only read when named",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=interp.DEFAULT_SEED,
        help="seeds the copying draws and every bootstrap",
    )
    p.add_argument(
        "--control-seed",
        type=int,
        default=interp.DEFAULT_SEED,
        help="seeds the matched control head draw, and only that",
    )
    p.add_argument(
        "--batch-size", type=int, default=interp.DEFAULT_BATCH_SIZE, help="cost only"
    )
    p.add_argument(
        "--weights-batch-size",
        type=int,
        default=interp.DEFAULT_WEIGHTS_BATCH_SIZE,
        help="cost only; attention weights are the memory-heavy forward",
    )
    p.add_argument(
        "--n-resamples",
        type=int,
        default=interp.DEFAULT_N_RESAMPLES,
        help="bootstrap resamples; the pre-registered value is 1000",
    )
    p.add_argument(
        "--patch-sequences",
        type=int,
        default=interp.PATCHING_N_SEQUENCES,
        help=(
            "sequences behind the patching run. The amendment fixes the "
            f"pre-registered count at {interp.PATCHING_N_SEQUENCES}; this is a "
            "cost knob for exploratory runs and is recorded in the output"
        ),
    )
    p.add_argument(
        "--patch-escalate",
        action="store_true",
        help=(
            "the one pre-registered escalation: re-run patching at "
            f"{interp.PATCHING_ESCALATED_N_SEQUENCES} sequences and record the run "
            "as the declared escalation. Legitimate only where the model "
            "demonstrably copies and the gap still straddles zero, which is a "
            "judgment on a copying reading that this script does not make. "
            "Overrides --patch-sequences"
        ),
    )
    p.add_argument(
        "--patch-all-heads",
        action="store_true",
        help="patch every head instead of the induction heads and their controls",
    )
    p.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help=(
            "cap the windows the ablation's perplexity and in-context deltas read. "
            "Cost control for a many-condition sweep; the count used is recorded"
        ),
    )
    p.add_argument(
        "--domains",
        nargs="*",
        default=list(DOMAINS),
        choices=list(DOMAINS),
        help="domains for the ablation's perplexity and in-context deltas",
    )
    p.add_argument(
        "--global-mean-ablation",
        dest="global_mean_sensitivity",
        action="store_true",
        help=(
            "additionally run the global-mean ablation, one mean per head "
            "averaged over positions too, and report it under "
            "sensitivity_global_mean. The headline stays per position, which the "
            "pre-registration amendment fixed and which is therefore not a flag"
        ),
    )
    p.add_argument("--tokenized-root", type=Path, default=DEFAULT_TOKENIZED_ROOT)
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = get_config(args.config)
    eval_script = load_eval_script()

    if args.checkpoint_dir is not None:
        checkpoints = find_checkpoints(args.checkpoint_dir)
    else:
        checkpoints = [args.checkpoint]

    run_all = args.analysis == "all"
    want_prefix = run_all or args.analysis == "prefix_matching"
    want_phase = run_all or args.analysis == "phase_change"
    want_ablation = run_all or args.analysis == "ablation"
    want_patching = run_all or args.analysis == "patching"

    if args.analysis == "phase_change" and args.checkpoint_dir is None:
        raise SystemExit(
            "phase_change reads a trajectory across checkpoints. Pass "
            "--checkpoint-dir, not --checkpoint."
        )

    # Refuse an ambiguous causal-checkpoint selection before any model loads or
    # any prefix-matching sweep runs, so a bad invocation fails in a second
    # instead of after minutes of work.
    validate_causal_selection(args, want_ablation or want_patching)

    results: dict[str, Any] = {
        "metadata": {
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "checkpoint_dir": str(args.checkpoint_dir) if args.checkpoint_dir else None,
            "n_checkpoints": len(checkpoints),
            "config": args.config,
            "n_params": n_params(cfg),
            "device": args.device,
            "seed": args.seed,
            "control_seed": args.control_seed,
            "split": args.split,
            "analysis": args.analysis,
            "n_resamples": args.n_resamples,
            "batch_size": args.batch_size,
            "induction_threshold": interp.INDUCTION_THRESHOLD,
            "phase_bounds": [interp.PHASE_LOW, interp.PHASE_HIGH],
            "mean_ablation_primary": "per_position",
            "causal_selection": (
                "single_checkpoint"
                if args.checkpoint is not None
                else "all"
                if args.all_checkpoints
                else "explicit"
                if args.causal_checkpoints
                else "final_only"
                if args.final_only
                else None
            ),
            "global_mean_sensitivity": args.global_mean_sensitivity,
            "patching_metric": interp.PATCHING_METRIC,
            "patching_secondary_metric": interp.PATCHING_SECONDARY_METRIC,
            "patch_sequences": (
                interp.PATCHING_ESCALATED_N_SEQUENCES
                if args.patch_escalate
                else args.patch_sequences
            ),
            "patch_escalated": bool(args.patch_escalate),
            "max_windows": args.max_windows,
            "tokenized_root": str(args.tokenized_root),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "torch_version": torch.__version__,
        }
    }

    # The prefix-matching sweep is what the phase change is read from, and it is
    # also what names the induction heads the causal analyses intervene on. So it
    # runs whenever anything downstream of it is wanted.
    trajectory: list[dict[str, Any]] = []
    if want_prefix or want_phase or want_ablation or want_patching:
        for path in checkpoints:
            model, tokens = eval_script.load_model(cfg, path, args.device)
            entry = {"checkpoint": str(path), "tokens": tokens}
            entry.update(prefix_matching_report(model, args))
            trajectory.append(entry)
            print(
                f"{path.name}: tokens {tokens}, max prefix-matching score "
                f"{entry['max_score']:.4f}, {len(entry['induction_heads'])} "
                f"induction heads",
                flush=True,
            )

    if want_prefix:
        if args.checkpoint_dir is not None:
            results["prefix_matching_trajectory"] = trajectory
        else:
            results["prefix_matching"] = trajectory[0]

    if want_phase:
        if args.checkpoint_dir is None:
            results["phase_change"] = {
                "ran": False,
                "note": (
                    "phase_change reads a trajectory across checkpoints and one "
                    "checkpoint was given. Rerun with --checkpoint-dir."
                ),
            }
        elif any(e["tokens"] is None for e in trajectory):
            results["phase_change"] = {
                "ran": False,
                "note": (
                    "at least one checkpoint records no token count, so the "
                    "trajectory cannot be placed on a token axis."
                ),
            }
        else:
            results["phase_change"] = {
                "ran": True,
                "trajectory": [
                    {"tokens": e["tokens"], "max_score": e["max_score"]}
                    for e in trajectory
                ],
                "interval": to_jsonable(
                    interp.phase_change_interval(
                        {e["tokens"]: e["max_score"] for e in trajectory}
                    )
                ),
            }

    if want_ablation or want_patching:
        # The registered causal battery runs at the interp anchors, which the
        # selection flags name. One anchor (a single --checkpoint or --final-only)
        # keeps the original singular result shape so old readers still parse;
        # several anchors produce a list keyed by checkpoint.
        selected = select_causal_checkpoints(args, trajectory)
        results["metadata"]["n_causal_checkpoints"] = len(selected)
        singular = args.checkpoint is not None or args.final_only

        def causal_for(entry: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            path = Path(entry["checkpoint"])
            model, tokens = eval_script.load_model(cfg, path, args.device)
            scores = np.asarray(entry["scores"], dtype=np.float64)
            induction = [(int(l), int(h)) for l, h in entry["induction_heads"]]
            reports: dict[str, Any] = {}
            if want_ablation:
                reports["ablation"] = ablation_report(model, args, induction, scores)
            if want_patching:
                reports["patching"] = patching_report(model, args, induction, scores)
            meta = {"path": str(path), "tokens": tokens}
            return meta, reports

        if singular:
            meta, reports = causal_for(selected[0])
            results["causal_checkpoint"] = meta
            results.update(reports)
        else:
            causal_meta: list[dict[str, Any]] = []
            ablation_by_ckpt: list[dict[str, Any]] = []
            patching_by_ckpt: list[dict[str, Any]] = []
            for entry in selected:
                meta, reports = causal_for(entry)
                causal_meta.append(meta)
                if want_ablation:
                    ablation_by_ckpt.append(
                        {"checkpoint": meta["path"], "tokens": meta["tokens"],
                         "ablation": reports["ablation"]}
                    )
                if want_patching:
                    patching_by_ckpt.append(
                        {"checkpoint": meta["path"], "tokens": meta["tokens"],
                         "patching": reports["patching"]}
                    )
                print(
                    f"causal battery at {Path(meta['path']).name}: "
                    f"tokens {meta['tokens']}",
                    flush=True,
                )
            results["causal_checkpoints"] = causal_meta
            if want_ablation:
                results["ablation_by_checkpoint"] = ablation_by_ckpt
            if want_patching:
                results["patching_by_checkpoint"] = patching_by_ckpt

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(results, f, indent=2, sort_keys=False)
        f.write("\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
