"""Measure the induction-head emergence trajectory over a checkpoint grid.

Writes a single JSON holding one record per checkpoint plus the trajectory-level
summary, with enough metadata to say what produced it: the checkpoints, the
config, the token count each checkpoint recorded, the device, the seeds, the
cost caps and a timestamp. Re-runnable: the same arguments against the same
checkpoints reproduce the same file, because every draw and every bootstrap is
seeded.

What this is for. scripts/08_interp.py runs the registered causal battery, which
costs a forward per head per condition and therefore runs at a handful of
anchors. This script runs the cheap half of Phase B at every checkpoint on the
34-point grid, so the question it answers is when, not whether: at which token
count the prefix-matching score crosses, whether a previous-token head was there
first, when copying behavior clears chance, and where the in-context learning
score falls fastest. The fields it writes are the ones
docs/PROTOCOL.md predictions P1 to P8 are scored against.

Per checkpoint, from small_lm_lab.emergence:

  prefix_matching   the registered induction-head score per layer and head,
                    with per-head bootstrap intervals and the heads above the
                    registered 0.2 named. Identical to what 08_interp reports,
                    because it is the same code reading the same forward.
  attention         the two off-by-one controls (attention to the earlier
                    occurrence itself, and to one position past the registered
                    key), plus previous-token, self, first-token and normalized
                    entropy per head. Supplementary diagnostics, declared as
                    such in the prediction document before any of them was run.
  ov                the OV-circuit copying score per head, from weights only,
                    and the median self-logit rank over a sample of token ids.
  copying           evaluate.copying_eval, with the chance level read off the
                    construction and the registered clears-chance verdict.
  icl               the in-context learning score per domain, the same statistic
                    evaluate.icl_score reports, capped at --max-windows for cost
                    with the cap and the realized window count recorded.

Two files come out, not one. The JSON holds every number. Beside it, named after
it, an .npz holds each checkpoint's whole [n_sequences, n_layers, n_heads]
prefix-matching array, which the 2026-07-26 amendment requires be
retained so the crossing time can carry a bootstrap interval. It is a sidecar
rather than a JSON field because it is 131 KB per checkpoint at the registered
settings, which would bury the result file in sixteen thousand floats per point
for numbers no human reads directly. The sidecar's path and each point's key
into it are recorded in the JSON, so the pair travels together.

The 0.2 threshold, the 0.1 and 0.3 phase-change bounds, the 256 sequences, the
copying task and the in-context windows are not flags. They were fixed before
any checkpoint existed and they are read from small_lm_lab.interp and
small_lm_lab.evaluate, so no invocation of this script can move one. What is
exposed is cost (batch sizes, resample counts, the window cap) and provenance
(seeds), neither of which can move a number that a claim is read from.

--random-init is the negative control, P8: an untrained model at the same config
and a chosen init seed, measured through the identical instrument, written with
tokens null because a model that has taken no optimizer steps sits at no point
of the token axis. If the trained numbers are to mean anything this one has to
come back at chance.

--skip-icl drops the in-context block, which is the only measurement here that
needs the tokenized streams mounted. Everything else runs from the checkpoint
alone.

Usage:
  python scripts/19_emergence.py --config size30m --checkpoint-dir DIR \
      --out OUT.json
  python scripts/19_emergence.py --config size30m --checkpoint PATH \
      --skip-icl --out OUT.json
  python scripts/19_emergence.py --config size30m --random-init --out OUT.json
"""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import json

import numpy as np
import torch

from small_lm_lab import emergence, evaluate, interp
from small_lm_lab.config import get_config, n_params
from small_lm_lab.data import DEFAULT_TOKENIZED_ROOT
from small_lm_lab.evaluate import DOMAINS, EOT_ID
from small_lm_lab.paths import portable_path

REPO_ROOT = Path(__file__).resolve().parents[1]
INTERP_SCRIPT = REPO_ROOT / "scripts" / "08_interp.py"


@lru_cache(maxsize=1)
def interp_script():
    """Import scripts/08_interp.py by path, for the three things it already owns.

    find_checkpoints knows that train.py zero pads the token count so a lexical
    sort is a chronological one and that resume_state.pt is not a checkpoint.
    to_jsonable knows that a non-finite float has to become null or the result
    file is one a spec-conformant parser rejects. load_eval_script reaches
    scripts/06_evaluate.py for the checkpoint loader, which knows the checkpoint
    layout, strips a torch.compile prefix and casts bf16 storage up to float32.

    None of those takes a second copy: a second copy of the checkpoint loader
    is a second thing to keep in step with train.py, and a second copy of
    to_jsonable is a result file that is valid JSON in one script and not in the
    other. scripts/ is not a package, hence the path import.
    """
    spec = importlib.util.spec_from_file_location("interp_script", INTERP_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_self_logit_tokens(
    args: argparse.Namespace, vocab_size: int
) -> tuple[Optional[np.ndarray], str]:
    """The token ids ov_self_logit_rank ranks against, and where they came from.

    Preferred source is the corpus: the most frequent ids over a bounded prefix
    of the streams, since a rank against ids the model has seen many times says
    more than a rank against a vocabulary whose tail is nearly unused. That
    needs the streams mounted, which --skip-icl says they are not, and which can
    fail anyway on a machine holding only checkpoints.

    The fallback is a uniform draw over the non-eot ids, seeded by
    --control-seed. It is a different sample answering a slightly different
    question, so the source is recorded rather than left to be inferred from
    whether the corpus happened to be present. Nothing registered is read from
    this diagnostic either way.
    """
    n = int(args.n_freq_tokens)
    if n < 1:
        return None, emergence.SELF_LOGIT_SOURCE_NONE

    if not args.skip_icl and args.domains:
        try:
            ids = emergence.frequent_token_ids(
                n,
                root=args.tokenized_root,
                domains=args.domains,
                split=args.split,
                vocab_size=vocab_size,
            )
            return ids, emergence.SELF_LOGIT_SOURCE_CORPUS
        except (FileNotFoundError, OSError, ValueError) as exc:
            print(
                "token frequencies unavailable, falling back to a seeded uniform "
                f"draw over the vocabulary: {exc}",
                flush=True,
            )

    rng = np.random.default_rng(args.control_seed)
    pool = np.arange(EOT_ID + 1, vocab_size, dtype=np.int64)
    ids = rng.choice(pool, size=min(n, pool.size), replace=False)
    return np.sort(ids), emergence.SELF_LOGIT_SOURCE_UNIFORM


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="a run's checkpoint directory; every checkpoint is swept, oldest first",
    )
    src.add_argument("--checkpoint", type=Path, help="path to one saved checkpoint")
    src.add_argument(
        "--random-init",
        action="store_true",
        help=(
            "the negative control: an untrained model at this config and "
            "--init-seed, measured through the same instrument and recorded with "
            "tokens null"
        ),
    )
    p.add_argument("--config", required=True, help="named config, for example size30m")
    p.add_argument("--device", default="cpu", help="cpu, mps, or cuda")
    p.add_argument("--out", type=Path, required=True, help="path for the JSON result")
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
        help=(
            "seeds the fallback uniform token draw behind the self-logit rank, "
            "and only that. This script runs no matched-control comparison; the "
            "flag keeps its meaning as the seed of the draws that are not the "
            "registered ones, as in scripts/08_interp.py"
        ),
    )
    p.add_argument(
        "--init-seed",
        type=int,
        default=0,
        help="weights the --random-init control; ignored otherwise",
    )
    p.add_argument(
        "--n-sequences",
        type=int,
        default=interp.PREFIX_MATCHING_N_SEQUENCES,
        help=(
            "sequences behind the prefix-matching and attention summaries. The "
            f"pre-registered count is {interp.PREFIX_MATCHING_N_SEQUENCES} and is the "
            "default. Passing anything else makes the run a MONITORING read: the "
            "count used and the fact that it is not the registered one are both "
            "recorded in the output, exactly as --patch-sequences is handled in "
            "scripts/08_interp.py. Unlike the thresholds, this cannot move a "
            "definition, only the precision of the estimate, which is why it is a "
            "knob at all. No registered claim may be read from a run where "
            "metadata.at_registered_n_sequences is false"
        ),
    )
    p.add_argument(
        "--n-resamples",
        type=int,
        default=interp.DEFAULT_N_RESAMPLES,
        help="bootstrap resamples; the pre-registered value is 1000",
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
        "--max-windows",
        type=int,
        default=emergence.DEFAULT_ICL_MAX_WINDOWS,
        help=(
            "cap the windows the in-context score reads per domain. Cost control "
            "for a 34-point grid; the cap and the realized count are recorded on "
            "every point, and the uncapped headline number comes from "
            "scripts/06_evaluate.py. Pass a negative value for no cap"
        ),
    )
    p.add_argument(
        "--domains",
        nargs="*",
        default=list(DOMAINS),
        choices=list(DOMAINS),
        help="domains for the in-context score",
    )
    p.add_argument(
        "--skip-icl",
        action="store_true",
        help=(
            "skip the in-context score, the only measurement here that needs the "
            "tokenized streams mounted"
        ),
    )
    p.add_argument(
        "--n-freq-tokens",
        type=int,
        default=emergence.DEFAULT_N_FREQ_TOKENS,
        help=(
            "how many token ids the secondary self-logit rank is measured "
            "against. 0 skips it"
        ),
    )
    p.add_argument("--tokenized-root", type=Path, default=DEFAULT_TOKENIZED_ROOT)
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = get_config(args.config)
    script = interp_script()
    eval_script = script.load_eval_script()
    to_jsonable = script.to_jsonable

    if args.checkpoint_dir is not None:
        checkpoints: list[Optional[Path]] = list(
            script.find_checkpoints(args.checkpoint_dir)
        )
    elif args.checkpoint is not None:
        checkpoints = [args.checkpoint]
    else:
        checkpoints = [None]

    domains = [] if args.skip_icl else list(args.domains)
    max_windows = None if args.max_windows is not None and args.max_windows < 0 else args.max_windows
    token_ids, token_source = resolve_self_logit_tokens(args, int(cfg.vocab_size))

    results: dict[str, Any] = {
        "metadata": {
            "checkpoint": portable_path(args.checkpoint),
            "checkpoint_dir": portable_path(args.checkpoint_dir),
            "random_init": bool(args.random_init),
            "init_seed": args.init_seed if args.random_init else None,
            "n_checkpoints": len(checkpoints),
            "config": args.config,
            "n_params": n_params(cfg),
            "device": args.device,
            "seed": args.seed,
            "control_seed": args.control_seed,
            "split": args.split,
            "domains": domains,
            "skip_icl": bool(args.skip_icl),
            "n_resamples": args.n_resamples,
            "batch_size": args.batch_size,
            "weights_batch_size": args.weights_batch_size,
            "max_windows": max_windows,
            "n_freq_tokens": args.n_freq_tokens,
            "self_logit_token_source": token_source,
            # Every registered constant this run was measured under, written
            # into the file so a reader never has to open the source to learn
            # which numbers the claims were read against.
            "induction_threshold": interp.INDUCTION_THRESHOLD,
            "phase_bounds": [interp.PHASE_LOW, interp.PHASE_HIGH],
            "prev_token_level": emergence.PREV_TOKEN_LEVEL,
            "prefix_matching_n_sequences": interp.PREFIX_MATCHING_N_SEQUENCES,
            # What this run actually used, next to what the registration fixes,
            # so the two can never be confused by a reader who has only the file.
            "n_sequences_used": args.n_sequences,
            "at_registered_n_sequences": (
                args.n_sequences == interp.PREFIX_MATCHING_N_SEQUENCES
            ),
            "reading_status": (
                "registered"
                if args.n_sequences == interp.PREFIX_MATCHING_N_SEQUENCES
                else "MONITORING, reduced sequence count, not a reportable figure"
            ),
            "copying_n_sequences": evaluate.COPY_N_SEQUENCES,
            "repeat_len": evaluate.COPY_REPEAT_LEN,
            "prefix_matching_chance": emergence.prefix_matching_chance(
                evaluate.COPY_REPEAT_LEN
            ),
            "copying_chance": interp.copy_chance_accuracy(int(cfg.vocab_size)),
            "icl_windows": {"early": list(evaluate.ICL_EARLY), "late": list(evaluate.ICL_LATE)},
            "tokenized_root": portable_path(args.tokenized_root),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "torch_version": torch.__version__,
        }
    }

    points = []
    for path in checkpoints:
        if path is None:
            # The control's weights are the measurement, so the draw that makes
            # them is seeded here rather than left to whatever state torch is in.
            torch.manual_seed(args.init_seed)
        model, tokens = eval_script.load_model(cfg, path, args.device)
        point = emergence.measure_checkpoint(
            model,
            args.device,
            tokens=tokens,
            checkpoint=portable_path(path),
            n_sequences=args.n_sequences,
            seed=args.seed,
            weights_batch_size=args.weights_batch_size,
            batch_size=args.batch_size,
            domains=domains,
            split=args.split,
            root=args.tokenized_root,
            max_windows=max_windows,
            self_logit_token_ids=token_ids,
            self_logit_token_source=token_source,
            n_resamples=args.n_resamples,
        )
        points.append(point)
        print(
            f"{path.name if path is not None else 'random-init'}: "
            f"tokens {point.tokens}, "
            f"max prefix-matching {point.max_prefix_matching:.4f}, "
            f"max prev-token {point.max_prev_token:.4f}, "
            f"copying exact match {point.copying.exact_match:.6g}",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # The retained per-sequence arrays go to the sidecar and are taken out of
    # the record before it is serialized. asdict first and pop second, rather
    # than to_jsonable and pop, so the array is never walked into a nested list
    # of sixteen thousand floats on its way to being discarded.
    sidecar = args.out.with_name(args.out.stem + ".prefix_matching.npz")
    arrays: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    for point in points:
        key = emergence.per_sequence_key(point)
        record = asdict(point)
        retained = record.pop("prefix_matching_per_sequence")
        if retained is not None:
            arrays[key] = np.asarray(retained)
        record = to_jsonable(record)
        record["prefix_matching_per_sequence_key"] = key if retained is not None else None
        records.append(record)
    np.savez_compressed(sidecar, **arrays)

    results["metadata"]["prefix_matching_per_sequence_path"] = portable_path(sidecar)
    results["metadata"]["prefix_matching_per_sequence_keys"] = sorted(arrays)
    results["points"] = records
    results["trajectory_summary"] = to_jsonable(
        emergence.trajectory_summary(
            points, n_resamples=args.n_resamples, seed=args.seed
        )
    )

    with args.out.open("w") as f:
        json.dump(results, f, indent=2, sort_keys=False)
        f.write("\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
