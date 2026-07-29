"""Train one model to the token budget, in whichever framework the pilot picked.

The pre-registered settings are defaults in small_lm_lab.train and are not
exposed here as flags. What this script takes is what a run is allowed to vary:
the size, the framework, the peak learning rate, the budget, the seed, the mix,
and where things land.

Usage:
  python scripts/07_train.py --size size30m --framework torch --lr 1.2e-3 \
      --tokens 500e6 --run-name size30m_lr1.2e-3_seed1
  python scripts/07_train.py --size size30m --framework torch --lr 1.2e-3 \
      --tokens 500e6 --run-name size30m_lr1.2e-3_seed1 --resume

The banner prints the param count, tokens per step, and total steps before the
first step. The wall-clock projection cannot be printed there, because it is a
measurement: it appears a few steps in, once there is a throughput to project
from.

Training holds the shared GPU lock. This script does not take it and does not
release it. On a machine where more than one job can reach the same GPU, the
caller acquires the mutex at $ML_TRAIN_LOCK (default /tmp/ml-train.lock) before
running anything here on the GPU. On a machine with one job the lock is
unnecessary and nothing here checks for it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from small_lm_lab.config import n_params
from small_lm_lab.data import DEFAULT_TOKENIZED_ROOT
from small_lm_lab.train import (
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_LR_SCHEDULE,
    DEFAULT_MIX,
    DEFAULT_RUN_ROOT,
    DEFAULT_SEED,
    DEFAULT_TORCH_PRECISION,
    DEFAULT_WARMUP_TOKENS,
    LR_SCHEDULES,
    PREREG_TOKENS_PER_STEP,
    TORCH_PRECISIONS,
    WARMUP_FRACTION,
    TrainConfig,
    train,
    warmup_steps_for,
    warmup_steps_for_tokens,
)

SIZES = ("size30m", "size60m", "size120m")
FRAMEWORKS = ("torch", "mlx")


def load_checkpoint_schedule(path: Path) -> tuple[int, ...]:
    """Read a checkpoint schedule override: a JSON list of token counts.

    This is how the staged run overrides the built-in pre-registered schedule,
    and how mid-run densification is done: stop the run, extend this list with
    new future counts, and resume. A resume with an extended (superset) schedule
    does not rewrite the checkpoints already on disk, because the trainer only
    fires a checkpoint at steps it actually reaches, and a resumed run starts
    past every already-written step. Already-written counts in the list are
    therefore honored as history and the new future counts are honored going
    forward.
    """
    with Path(path).open("r") as f:
        raw = json.load(f)
    if not isinstance(raw, list) or not raw:
        raise SystemExit(
            f"checkpoint schedule {path} must be a non-empty JSON list of token "
            f"counts, got {type(raw).__name__}"
        )
    targets: list[int] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise SystemExit(
                f"checkpoint schedule {path} holds a non-numeric entry {item!r}"
            )
        value = int(item)
        if value <= 0:
            raise SystemExit(
                f"checkpoint schedule {path} holds a non-positive token count {item!r}"
            )
        targets.append(value)
    # Sorted and de-duplicated: the trainer maps each target to a step and keeps
    # the first target per step, so a deterministic order makes the mapping
    # reproducible whatever order the file listed the counts in.
    return tuple(sorted(set(targets)))


def parse_mix(text: str) -> dict[str, float]:
    """Parse "tinystories=0.7,fineweb_edu=0.3" into a weight dict."""
    mix: dict[str, float] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise argparse.ArgumentTypeError(
                f"bad mix entry {part!r}; expected domain=weight"
            )
        name, _, weight = part.partition("=")
        try:
            mix[name.strip()] = float(weight)
        except ValueError:
            raise argparse.ArgumentTypeError(f"bad weight in {part!r}")
    if not mix:
        raise argparse.ArgumentTypeError("mix names no domains")
    return mix


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--size", choices=SIZES, required=True)
    p.add_argument("--framework", choices=FRAMEWORKS, required=True)
    p.add_argument("--lr", type=float, required=True, help="peak learning rate")
    p.add_argument(
        "--tokens",
        type=float,
        required=True,
        help="token budget B, for example 500e6",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--run-name", required=True)
    p.add_argument(
        "--micro-batch",
        type=int,
        default=None,
        help=(
            "rows per forward/backward. Must divide the batch size of 32. "
            "Defaults to the full batch, meaning no accumulation. Purely a "
            "memory setting: the optimizer sees the same 32 x 512 batch and "
            "takes one step either way. size120m trains at 16."
        ),
    )
    p.add_argument(
        "--device",
        default="mps",
        help="torch: cpu, mps, cuda. mlx: cpu or gpu.",
    )
    p.add_argument(
        "--torch-precision",
        choices=TORCH_PRECISIONS,
        default=DEFAULT_TORCH_PRECISION,
        help=(
            "torch compute precision. fp32 is the registered precision the MLX "
            "reference trained in and is what the CUDA staged run uses; auto/bf16 "
            "run bf16 autocast on an accelerator (float32 on CPU). Ignored by mlx, "
            "which has no autocast and always runs float32."
        ),
    )
    p.add_argument("--resume", action="store_true", help="continue from resume_state.pt")
    p.add_argument(
        "--stop-after-steps",
        type=int,
        default=None,
        help=(
            "stop after this many steps while keeping the schedule set for "
            "--tokens, so each step runs at the rate the full run would. This is "
            "how the cross-backend gate does a 500-step same-seed rerun of "
            "abl_lr1.2e-3: --tokens 40e6 --stop-after-steps 500 reproduces the "
            "original run's first 500 steps exactly"
        ),
    )
    p.add_argument(
        "--mix",
        type=parse_mix,
        default=dict(DEFAULT_MIX),
        help='domain weights, for example "tinystories=0.7,fineweb_edu=0.3"',
    )
    p.add_argument(
        "--warmup-fraction",
        type=float,
        default=WARMUP_FRACTION,
        help="cosine warmup, pre-registered at 0.02; the ablation also runs 0",
    )
    p.add_argument(
        "--lr-schedule",
        choices=LR_SCHEDULES,
        default=DEFAULT_LR_SCHEDULE,
        help=(
            "cosine (pre-registered Phase A, the default) or constant (the staged "
            "deep run: linear warmup over --warmup-tokens then flat at the peak "
            "rate with no decay, because the run's stop token is outcome-"
            "contingent and a cosine-to-horizon schedule would be incoherent)"
        ),
    )
    p.add_argument(
        "--warmup-tokens",
        type=float,
        default=DEFAULT_WARMUP_TOKENS,
        help=(
            "absolute warmup length in tokens for --lr-schedule constant; ignored "
            f"by cosine. Default {DEFAULT_WARMUP_TOKENS:,}"
        ),
    )
    p.add_argument(
        "--checkpoint-schedule",
        type=Path,
        default=None,
        help=(
            "path to a JSON list of token counts that overrides the built-in "
            "pre-registered checkpoint schedule. This is how the staged run "
            "checkpoints on its own grid and how densification is done: stop, "
            "extend this list with new future counts, and --resume. A resume with "
            "an extended superset list does not rewrite checkpoints already on "
            "disk and honors the new future counts"
        ),
    )
    p.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    p.add_argument("--out-root", type=Path, default=DEFAULT_RUN_ROOT)
    p.add_argument("--tokenized-root", type=Path, default=DEFAULT_TOKENIZED_ROOT)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--val-every", type=int, default=200)
    p.add_argument("--val-batches", type=int, default=16)
    p.add_argument(
        "--resume-state-every",
        type=int,
        default=None,
        help=(
            "steps between resume_state.pt writes; 0 disables. Default 500. This is "
            "the only thing that bounds how much work a kill can destroy, so it must "
            "be shorter than the shortest interval in which the run can be stopped. "
            "A supervised run that yields the GPU to an interactive session should set "
            "this well below its yield interval: a launch that is always killed "
            "before it can persist makes no net progress at all, however long it runs"
        ),
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    checkpoint_targets = (
        load_checkpoint_schedule(args.checkpoint_schedule)
        if args.checkpoint_schedule is not None
        else None
    )

    # TrainConfig refuses a micro-batch that does not divide the batch evenly.
    # Surfaced as a clean exit rather than a traceback, matching the tokens per
    # step refusal below.
    try:
        cfg = TrainConfig(
            run_name=args.run_name,
            size=args.size,
            framework=args.framework,
            device=args.device,
            torch_precision=args.torch_precision,
            peak_lr=args.lr,
            total_tokens=int(args.tokens),
            warmup_fraction=args.warmup_fraction,
            lr_schedule=args.lr_schedule,
            warmup_tokens=int(args.warmup_tokens),
            seed=args.seed,
            micro_batch_size=args.micro_batch,
            mix=args.mix,
            stop_after_steps=args.stop_after_steps,
            checkpoint_token_targets=checkpoint_targets,
            checkpoint_root=args.checkpoint_root,
            out_root=args.out_root,
            tokenized_root=args.tokenized_root,
            log_every=args.log_every,
            val_every=args.val_every,
            val_batches=args.val_batches,
            **(
                {}
                if args.resume_state_every is None
                else {"resume_state_every": args.resume_state_every}
            ),
        )
    except ValueError as exc:
        raise SystemExit(f"refusing to run: {exc}")

    model_cfg = cfg.resolved_model_config()
    if cfg.tokens_per_step != PREREG_TOKENS_PER_STEP:
        raise SystemExit(
            f"tokens per step is {cfg.tokens_per_step}, but the pre-registration "
            f"fixes it at {PREREG_TOKENS_PER_STEP} (32 x 512) for every size. "
            "Refusing to run."
        )

    if cfg.lr_schedule == "constant":
        warmup = warmup_steps_for_tokens(
            cfg.warmup_tokens, cfg.tokens_per_step, cfg.total_steps
        )
        schedule_str = (
            f"  peak lr {cfg.peak_lr:.3e}, CONSTANT after "
            f"{cfg.warmup_tokens:,}-token warmup, no decay\n"
        )
    else:
        warmup = warmup_steps_for(cfg.total_steps, cfg.warmup_fraction)
        schedule_str = f"  peak lr {cfg.peak_lr:.3e}, cosine to {cfg.peak_lr * 0.1:.3e}\n"
    if checkpoint_targets is not None:
        schedule_str += (
            f"  checkpoint schedule override: {len(checkpoint_targets)} token "
            f"counts from {args.checkpoint_schedule}\n"
        )
    mix_str = ", ".join(f"{k} {v:.2f}" for k, v in cfg.mix.items())
    if cfg.accum_steps == 1:
        accum_str = (
            f"  micro-batch {cfg.resolved_micro_batch_size} (the full batch), "
            f"{cfg.accum_steps} accumulation step, no accumulation\n"
        )
    else:
        accum_str = (
            f"  micro-batch {cfg.resolved_micro_batch_size}, "
            f"{cfg.accum_steps} accumulation steps of "
            f"{cfg.resolved_micro_batch_size} x {model_cfg.context_len} = "
            f"{cfg.resolved_micro_batch_size * model_cfg.context_len:,} tokens each, "
            f"one optimizer step per {cfg.tokens_per_step:,} tokens\n"
        )
    precision_str = (
        f", torch precision {cfg.torch_precision}" if cfg.framework == "torch" else ""
    )
    print(
        f"run {cfg.run_name}\n"
        f"  {cfg.size} on {cfg.framework}/{cfg.device}, seed {cfg.seed}{precision_str}\n"
        f"  {n_params(model_cfg):,} params, d_model {model_cfg.d_model}, "
        f"{model_cfg.n_layers} layers, context {model_cfg.context_len}\n"
        f"  {cfg.tokens_per_step:,} tokens per step "
        f"({cfg.batch_size} x {model_cfg.context_len})\n"
        f"{accum_str}"
        f"  {cfg.total_steps:,} steps to {cfg.total_tokens:,} tokens, "
        f"{warmup:,} warmup steps\n"
        f"{schedule_str}"
        f"  mix: {mix_str}\n"
        f"  checkpoints -> {cfg.checkpoint_dir}\n"
        f"  logs -> {cfg.run_dir}",
        flush=True,
    )

    summary = train(cfg, resume=args.resume)
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
