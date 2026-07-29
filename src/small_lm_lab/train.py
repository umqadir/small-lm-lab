"""The training loop, written once and shared by both frameworks.

The framework decision (PyTorch-MPS or MLX) is settled by a throughput pilot, not
by this module. So everything that is not the inner optimizer step lives in
train() and is written exactly once: data loading, the learning rate schedule,
the checkpoint schedule, the JSONL log, the run manifest, validation, resume,
and timing. The two frameworks differ only inside an adapter that owns one
model, one optimizer, and one step.

The adapter contract, kept deliberately small:

  param_count()                -> int
  train_step(x, y, lr)         -> (loss, grad_norm), both float
  val_loss(x, y)               -> float
  get_weights_for_checkpoint() -> dict[str, np.ndarray], float32
  load_weights(dict)           -> None
  optimizer_state()            -> a picklable dict, for resume only
  load_optimizer_state(dict)   -> None
  peak_memory_bytes()          -> float

train_step returns the gradient norm alongside the loss because the JSONL log
records it, and the norm is only available inside the step where clipping
happens. That is the one place the contract is wider than "loss only".

Gradient accumulation. The pre-registration fixes 16,384 tokens per step at
every size and requires the optimizer to see that same batch regardless, so
accumulation here changes memory and nothing else. train_step still receives one
full batch_size batch and still performs exactly one optimizer step; an adapter
built with a micro_batch_size splits that batch by rows internally. Three
properties make the split an identity, not an approximation:

  Loss scaling. Every micro-batch holds the same number of tokens, which is what
    the divisibility check at startup buys. The mean over the full batch is then
    the plain mean of the micro-batch means, so each micro-batch loss is scaled
    by 1/accum_steps before its backward. The accumulated gradient is the
    gradient of the full-batch mean, not accum_steps times it, and the effective
    learning rate does not move with accum_steps.
  Clipping. The global-norm clip is applied once, to the accumulated gradient,
    immediately before the optimizer step. Clipping each micro-batch instead is
    a different algorithm: it rescales each piece by its own norm, so the summed
    direction differs from the clipped full-batch direction.
  Token order. The batcher is still constructed at batch_size and still yields
    whole batches, so the stream consumes the same RNG draws in the same order
    whatever micro_batch_size is set to. Splitting a materialized batch by rows,
    instead of asking the batcher for smaller ones, is what keeps a run at
    micro-batch 16 comparable to the same run at micro-batch 32 and keeps
    data.py untouched.

Checkpoint format. Weight dicts are keyed by the PyTorch state_dict names in
model_torch.py, in every framework. That naming is the canonical form because
evaluation and interpretability run in PyTorch whichever framework trains, and
because scripts/06_evaluate.py loads with strict=True and therefore needs every
key it expects and no others. Two consequences:

  - A torch state_dict lists lm_head.weight even when it is tied to the
    embedding, so a checkpoint must carry it. The MLX model realizes its head
    through embed.as_linear and has no such parameter, so the MLX adapter emits
    the tied duplicate on the way out and drops it on the way in. This mirrors
    convert.py, which is the verified reference for torch<->MLX transfer.
  - Weights land on disk as bf16 torch tensors, per the pre-registration.
    numpy has no bf16, so the adapters hand over float32 numpy and the shared
    saver does the single cast. The adapters never learn the storage dtype.

Resume. The pre-registration keeps optimizer state for the most recent
checkpoint only. That is honored by splitting the two artifacts apart:

  tokens_{n}.pt     the pre-registered checkpoint. bf16 weights, no optimizer
                    state, one per scheduled token count, never rewritten.
  resume_state.pt   float32 weights, optimizer state, step, token count, and
                    the data stream position. Exactly one exists at a time
                    because it is overwritten in place, so no old checkpoint
                    carries optimizer state.

Resuming from float32 instead of from the bf16 checkpoint is what makes a
resumed run continue the same trajectory instead of one perturbed by a rounding
of every weight.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

import numpy as np
import torch

from .config import ModelConfig, get_config, n_params
from .data import DEFAULT_TOKENIZED_ROOT, DomainSource, MixtureBatcher, load_sources
from .paths import CHECKPOINT_ROOT, RUN_ROOT, portable_path

REPO_ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_PATH = REPO_ROOT / "data" / "tokenizer" / "tokenizer.json"

DEFAULT_CHECKPOINT_ROOT = CHECKPOINT_ROOT
DEFAULT_RUN_ROOT = RUN_ROOT

# Pre-registration, "Model and training, fixed now". These are not tunable.
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95
ADAM_EPS = 1e-8
WEIGHT_DECAY = 0.1
GRAD_CLIP_NORM = 1.0
WARMUP_FRACTION = 0.02
LR_FINAL_FRACTION = 0.10

# Learning-rate schedules the trainer knows. "cosine" is the pre-registered
# Phase A schedule and is the default, unchanged. "constant" is the staged deep
# run's schedule (amendment 2026-07-21): linear warmup over an absolute token
# count, then flat at peak_lr with no decay. A cosine-to-horizon schedule is
# incoherent for an outcome-contingent run whose stop token is not known in
# advance, so the staged run holds the rate constant to keep checkpoints
# comparable across training.
LR_SCHEDULES = ("cosine", "constant")
DEFAULT_LR_SCHEDULE = "cosine"
# Warmup for the constant schedule is measured in tokens, not in a fraction of a
# horizon that does not exist. 4M tokens is ~244 steps at 16,384 tokens/step.
DEFAULT_WARMUP_TOKENS = 4_000_000
BATCH_SIZE = 32
PREREG_TOKENS_PER_STEP = 16384  # 32 x 512, held constant across sizes.
DEFAULT_MIX: dict[str, float] = {"tinystories": 0.70, "fineweb_edu": 0.30}
DEFAULT_SEED = 1

# Torch compute precision. The registered precision is float32: the MLX
# reference and every completed ablation trained in float32, and bf16 training
# was rejected as a different numeric regime that the learning-rate decision
# would not carry into. "fp32" forces
# plain float32 on any device; "auto" and "bf16" keep bf16 autocast on an
# accelerator (float32 on CPU always). Autocast keeps a float32 master copy of
# the weights, so "bf16" is a compute-precision choice, not bf16 parameters.
# Only the torch adapter reads this; MLX has no autocast and always runs float32.
TORCH_PRECISIONS = ("auto", "fp32", "bf16")
DEFAULT_TORCH_PRECISION = "auto"

# Pre-registration, "Checkpoint schedule, fixed now". The budget B is appended
# at run time, since it is a measurement and not a choice.
PREREG_CHECKPOINT_TOKENS: tuple[int, ...] = (
    1_000_000,
    2_000_000,
    4_000_000,
    8_000_000,
    16_000_000,
    24_000_000,
    32_000_000,
    48_000_000,
    64_000_000,
    96_000_000,
    128_000_000,
    160_000_000,
    200_000_000,
    250_000_000,
    300_000_000,
    350_000_000,
    400_000_000,
    450_000_000,
)

# The validation slice is drawn with a seed of its own, not the run seed, so
# every run and every size is scored on the identical held-out batches. VAL
# only. The test splits stay closed until the final evaluation.
VAL_SEED = 20260717
VAL_SPLIT = "val"

CHECKPOINT_PREFIX = "tokens_"
RESUME_STATE_NAME = "resume_state.pt"
LOG_NAME = "train_log.jsonl"
MANIFEST_NAME = "manifest.json"


@dataclass
class TrainConfig:
    """Everything one run needs. Defaults are the pre-registered values."""

    run_name: str
    size: str = "size30m"
    framework: str = "torch"
    device: str = "mps"
    # Torch compute precision, see TORCH_PRECISIONS. Only the torch adapter reads
    # it. Default "auto" preserves the prior behavior (bf16 autocast on an
    # accelerator, float32 on CPU); "fp32" forces the registered float32 on any
    # device, which is what the CUDA staged run uses to match the MLX reference.
    torch_precision: str = DEFAULT_TORCH_PRECISION
    peak_lr: float = 1.2e-3
    total_tokens: int = 500_000_000  # the budget B
    warmup_fraction: float = WARMUP_FRACTION
    # Learning-rate schedule. "cosine" is the pre-registered Phase A schedule and
    # the default; "constant" is the staged deep run's schedule, warming up over
    # warmup_tokens then holding peak_lr flat with no decay. See LR_SCHEDULES.
    lr_schedule: str = DEFAULT_LR_SCHEDULE
    # Absolute warmup length for the constant schedule, in tokens. Ignored by the
    # cosine schedule, which warms up over warmup_fraction of the horizon.
    warmup_tokens: int = DEFAULT_WARMUP_TOKENS
    seed: int = DEFAULT_SEED
    batch_size: int = BATCH_SIZE
    # Rows per forward/backward. None means the whole batch at once, which is
    # the un-accumulated step and the default. Setting it smaller splits the
    # step into batch_size // micro_batch_size accumulated micro-batches for
    # the same optimizer update. It is a memory knob, not a training one: see
    # the module docstring for why the two paths are the same arithmetic.
    micro_batch_size: Optional[int] = None
    mix: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_MIX))

    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT
    out_root: Path = DEFAULT_RUN_ROOT
    tokenized_root: Path = DEFAULT_TOKENIZED_ROOT

    # Cadences. None of these touch a pre-registered quantity.
    log_every: int = 10
    val_every: int = 200
    val_batches: int = 16
    resume_state_every: int = 500
    projection_after: int = 20

    # Stop after this many steps while keeping the schedule set for
    # total_tokens, so the learning rate at each step is the one the full run
    # would have used. That is what the pre-registered size120m stability check
    # needs, since it watches the first 2 percent of steps of the real run
    # rather than a short run of its own. Also how the resume test simulates a
    # kill. None means run to the budget.
    stop_after_steps: Optional[int] = None

    # Test and ablation hooks. model_config overrides the named size, which the
    # tests use to run a model small enough for CPU. checkpoint_token_targets
    # overrides the pre-registered schedule, which the tests use to fire a
    # checkpoint inside a short run.
    model_config: Optional[ModelConfig] = None
    checkpoint_token_targets: Optional[tuple[int, ...]] = None
    verbose: bool = True

    def __post_init__(self) -> None:
        # Refuses at construction rather than at the first step, so a run that
        # is not going to be equivalent never starts and never writes a
        # checkpoint or a manifest claiming it was.
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}")
        if self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(
                f"unknown lr_schedule {self.lr_schedule!r}; expected one of "
                f"{LR_SCHEDULES}"
            )
        if self.torch_precision not in TORCH_PRECISIONS:
            raise ValueError(
                f"unknown torch_precision {self.torch_precision!r}; expected one "
                f"of {TORCH_PRECISIONS}"
            )
        if self.lr_schedule == "constant" and self.warmup_tokens < 0:
            raise ValueError(
                f"warmup_tokens must be non-negative, got {self.warmup_tokens}"
            )
        # A negative cadence would silently disable resume writes, because the write
        # site guards on `> 0`. Disabling persistence is a legitimate choice but it
        # has to be made deliberately as 0, never arrived at by a typo: a run that
        # never persists loses everything to the first kill.
        if self.resume_state_every < 0:
            raise ValueError(
                f"resume_state_every must be non-negative, got "
                f"{self.resume_state_every}; use 0 to disable resume writes on purpose"
            )
        if self.micro_batch_size is None:
            return
        micro = int(self.micro_batch_size)
        if micro < 1:
            raise ValueError(
                f"micro_batch_size must be at least 1, got {micro}"
            )
        if self.batch_size % micro != 0:
            divisors = [
                d for d in range(1, self.batch_size + 1) if self.batch_size % d == 0
            ]
            raise ValueError(
                f"micro_batch_size {micro} does not divide batch_size "
                f"{self.batch_size} evenly: {self.batch_size} % {micro} = "
                f"{self.batch_size % micro}. Gradient accumulation only reproduces "
                f"the un-accumulated step when every micro-batch carries the same "
                f"number of tokens, because the full-batch mean is then the plain "
                f"mean of the micro-batch means. A ragged split would silently "
                f"reweight the last micro-batch, so it is refused rather than run. "
                f"Valid micro_batch_size values for batch_size {self.batch_size}: "
                f"{divisors}."
            )

    def resolved_model_config(self) -> ModelConfig:
        if self.model_config is not None:
            return self.model_config
        return get_config(self.size)

    @property
    def resolved_micro_batch_size(self) -> int:
        """Rows per forward/backward. Equal to batch_size when not accumulating."""
        if self.micro_batch_size is None:
            return self.batch_size
        return int(self.micro_batch_size)

    @property
    def accum_steps(self) -> int:
        """Micro-batches per optimizer step. 1 when not accumulating."""
        return self.batch_size // self.resolved_micro_batch_size

    @property
    def tokens_per_step(self) -> int:
        # Deliberately keyed to batch_size, not to micro_batch_size. The
        # optimizer sees one batch_size batch per step whatever the split, so
        # the pre-registered 16,384 holds across every accumulation setting.
        return self.batch_size * self.resolved_model_config().context_len

    @property
    def total_steps(self) -> int:
        return max(1, math.ceil(self.total_tokens / self.tokens_per_step))

    @property
    def run_dir(self) -> Path:
        return Path(self.out_root) / self.run_name

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.checkpoint_root) / self.run_name


# ----------------------------------------------------------------------------
# schedule
# ----------------------------------------------------------------------------

def warmup_steps_for(total_steps: int, warmup_fraction: float) -> int:
    """Number of warmup steps. Rounds to nearest, and never eats the whole run."""
    if warmup_fraction <= 0.0:
        return 0
    return min(int(round(warmup_fraction * total_steps)), max(total_steps - 1, 0))


def lr_at_step(
    step: int, total_steps: int, peak_lr: float, warmup_fraction: float
) -> float:
    """Learning rate for a zero-indexed step: linear warmup, then cosine decay.

    Pre-registered: cosine decay to 10 percent of peak, warmup 2 percent of
    total steps. Three properties the schedule is required to have, all of them
    tested:

      the last warmup step runs at exactly peak_lr;
      the first post-warmup step also runs at exactly peak_lr, so the schedule
        is continuous across the boundary instead of dropping a step;
      the final step runs at exactly 10 percent of peak.

    Warmup counts from 1/warmup_steps up to 1, so no step runs at a learning
    rate of zero. With warmup_fraction 0 the cosine covers the whole run and
    step 0 starts at peak, which is the pre-registered no-warmup ablation.
    """
    if total_steps < 1:
        raise ValueError(f"total_steps must be at least 1, got {total_steps}")
    if not 0 <= step < total_steps:
        raise ValueError(f"step {step} outside [0, {total_steps})")

    final_lr = peak_lr * LR_FINAL_FRACTION
    warmup = warmup_steps_for(total_steps, warmup_fraction)

    if step < warmup:
        return peak_lr * (step + 1) / warmup

    decay_steps = total_steps - 1 - warmup
    if decay_steps <= 0:
        # A run with no room to decay ends at peak. Only reachable in tests.
        return peak_lr
    progress = (step - warmup) / decay_steps
    return final_lr + 0.5 * (peak_lr - final_lr) * (1.0 + math.cos(math.pi * progress))


def warmup_steps_for_tokens(
    warmup_tokens: int, tokens_per_step: int, total_steps: int
) -> int:
    """Warmup step count for the constant schedule, from an absolute token budget.

    The constant schedule warms up over a fixed number of tokens, not a fraction
    of a horizon, because the staged run has no fixed horizon to take a fraction
    of. The step count is the token budget divided by the tokens per step,
    rounded to nearest so a budget that does not land on a step boundary still
    warms up for about the right number of tokens. A positive token budget always
    buys at least one ramp step, and the warmup never eats the whole run: it is
    capped at total_steps - 1 so at least one step runs flat at peak_lr.
    """
    if warmup_tokens <= 0:
        return 0
    steps = int(round(warmup_tokens / tokens_per_step))
    steps = max(steps, 1)
    return min(steps, max(total_steps - 1, 0))


def lr_at_step_constant(
    step: int,
    total_steps: int,
    peak_lr: float,
    warmup_tokens: int,
    tokens_per_step: int,
) -> float:
    """Learning rate for a zero-indexed step under the constant schedule.

    Linear warmup over warmup_tokens tokens, then flat at peak_lr indefinitely,
    with no decay. The properties, all tested:

      warmup counts from 1/warmup_steps up to 1, so no step runs at zero and the
        last warmup step runs at exactly peak_lr;
      every step at or after the warmup boundary runs at exactly peak_lr, so the
        schedule is flat with no horizon-dependent decay;
      the warmup never eats the whole run: at least one step runs flat at peak.

    Unlike the cosine schedule the rate here does not depend on total_steps
    except through the warmup cap, which is what keeps checkpoints at different
    token counts comparable when the run's stop token is decided by an outcome
    rather than fixed in advance.
    """
    if total_steps < 1:
        raise ValueError(f"total_steps must be at least 1, got {total_steps}")
    if not 0 <= step < total_steps:
        raise ValueError(f"step {step} outside [0, {total_steps})")

    warmup = warmup_steps_for_tokens(warmup_tokens, tokens_per_step, total_steps)
    if step < warmup:
        return peak_lr * (step + 1) / warmup
    return peak_lr


def lr_for_config(cfg: "TrainConfig", step: int, total_steps: int) -> float:
    """The learning rate for a step, dispatched on cfg.lr_schedule.

    The one place the trainer reads a learning rate. Cosine is the pre-registered
    Phase A path and is left bit-identical; constant is the staged deep run's
    path. Kept as a function instead of inlined so both paths are exercised the
    same way the trainer exercises them.
    """
    if cfg.lr_schedule == "constant":
        return lr_at_step_constant(
            step, total_steps, cfg.peak_lr, cfg.warmup_tokens, cfg.tokens_per_step
        )
    return lr_at_step(step, total_steps, cfg.peak_lr, cfg.warmup_fraction)


def checkpoint_steps(
    total_steps: int,
    tokens_per_step: int,
    targets: Sequence[int] = PREREG_CHECKPOINT_TOKENS,
) -> dict[int, int]:
    """Map {zero-indexed step -> target token count} for the checkpoint schedule.

    A step boundary rarely lands exactly on a round target, since steps advance
    by tokens_per_step. Each target fires at the first step whose cumulative
    token count reaches it. The final step always fires, and that one is the
    checkpoint at B. Targets past the end of the run are dropped, so a short run
    does not try to checkpoint at 450M.
    """
    final_tokens = total_steps * tokens_per_step
    out: dict[int, int] = {}
    for target in targets:
        if target <= 0 or target > final_tokens:
            continue
        step = math.ceil(target / tokens_per_step) - 1
        out.setdefault(step, target)
    out[total_steps - 1] = final_tokens
    return out


# ----------------------------------------------------------------------------
# weight decay grouping
# ----------------------------------------------------------------------------

def should_decay(name: str) -> bool:
    """Whether a parameter takes weight decay.

    Pre-registered: decay applies to matrices, not to RMSNorm weights and not to
    the embedding. The RMSNorm weights are the only 1-D parameters in the model
    and the embedding is the only matrix excluded by name, so the rule reads
    directly off the name. lm_head.weight is tied to embed.weight, and
    named_parameters deduplicates the tie, so the head is never offered here and
    could not be decayed by accident through the back door.
    """
    if name == "embed.weight" or name == "lm_head.weight":
        return False
    if name.endswith("_norm.weight"):
        return False
    return True


# ----------------------------------------------------------------------------
# adapters
# ----------------------------------------------------------------------------

def resolve_accumulation(
    batch_rows: int, micro_batch_size: Optional[int]
) -> tuple[int, int]:
    """(micro_batch_size, accum_steps) for a batch of batch_rows rows.

    None means no accumulation, which is one micro-batch of the whole batch.
    The divisibility check is repeated here rather than trusted from
    TrainConfig, because the adapters are constructed directly in tests and by
    the pilot scripts, and an uneven split is wrong wherever it comes from.
    """
    micro = batch_rows if micro_batch_size is None else int(micro_batch_size)
    if micro < 1:
        raise ValueError(f"micro_batch_size must be at least 1, got {micro}")
    if batch_rows % micro != 0:
        raise ValueError(
            f"micro_batch_size {micro} does not divide a batch of {batch_rows} "
            f"rows evenly: {batch_rows} % {micro} = {batch_rows % micro}. Every "
            "micro-batch must carry the same number of tokens or the accumulated "
            "gradient is not the full-batch gradient."
        )
    return micro, batch_rows // micro


def resolve_torch_autocast_dtype(device_type: str, torch_precision: str):
    """Autocast dtype for the torch adapter. None means plain float32.

    Kept a pure function of (device_type, precision) so the precision policy can
    be tested without constructing a model on an accelerator, which would put
    the GPU-free test suite on a GPU another run may hold. The policy:

      float32 always on CPU, because autocast on CPU is a slow no-win and the CPU
        smoke tests want plain float32 determinism;
      float32 on any device when precision is "fp32", which is how the CUDA
        staged run holds the registered float32 the MLX reference trained in;
      bf16 autocast on an accelerator for "auto" and "bf16", the prior behavior.
    """
    if torch_precision not in TORCH_PRECISIONS:
        raise ValueError(
            f"unknown torch_precision {torch_precision!r}; expected one of "
            f"{TORCH_PRECISIONS}"
        )
    if device_type == "cpu" or torch_precision == "fp32":
        return None
    return torch.bfloat16


def disable_tf32() -> dict:
    """Force true IEEE float32 matmuls, never TF32. Returns what was actually set.

    Suppressing bf16 autocast is NOT enough to get the registered float32 on an
    NVIDIA accelerator. On Ampere and later, a float32 matmul can be executed in
    TF32, which keeps float32 range but carries 10 explicit mantissa bits instead
    of 23, a relative error near 1e-3 against 1e-7 for true float32. Nothing in the model or
    the optimizer would report this: the tensors stay torch.float32 and only the
    arithmetic behind them is coarser, so it is invisible except as a training
    curve that quietly fails to reproduce its reference.

    The MLX leg of this study ran with MLX_ENABLE_TF32=0 and proved it with a
    float64-reference canary, so the completed ablations and the LR decision are
    all true-float32 numbers. Leaving the torch path on a framework default would
    silently train the staged run in a different numeric regime from the reference
    it is supposed to continue.

    Set explicitly rather than trusted to a default, because these defaults have
    moved across torch releases and differ between the matmul and cuDNN paths.
    Each knob is guarded: the names have changed over versions and a missing one
    must not break CPU or non-NVIDIA runs. scripts/15_tf32_canary.py measures the
    result rather than assuming it.
    """
    applied: dict = {}
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        applied["cuda.matmul.allow_tf32"] = False
    except Exception as exc:  # pragma: no cover - version dependent
        applied["cuda.matmul.allow_tf32"] = f"unavailable: {exc}"
    try:
        torch.backends.cudnn.allow_tf32 = False
        applied["cudnn.allow_tf32"] = False
    except Exception as exc:  # pragma: no cover - version dependent
        applied["cudnn.allow_tf32"] = f"unavailable: {exc}"
    try:
        # Public, stable spelling of the same intent. "highest" is true float32.
        torch.set_float32_matmul_precision("highest")
        applied["float32_matmul_precision"] = "highest"
    except Exception as exc:  # pragma: no cover - version dependent
        applied["float32_matmul_precision"] = f"unavailable: {exc}"
    # Newer torch exposes a per-backend fp32 precision knob; "ieee" is true fp32.
    for path in ("cuda.matmul", "cudnn.conv"):
        try:
            head, tail = path.split(".")
            mod = getattr(getattr(torch.backends, head), tail)
            if hasattr(mod, "fp32_precision"):
                mod.fp32_precision = "ieee"
                applied[f"{path}.fp32_precision"] = "ieee"
        except Exception as exc:  # pragma: no cover - version dependent
            applied[f"{path}.fp32_precision"] = f"unavailable: {exc}"
    return applied


class Adapter(Protocol):
    def param_count(self) -> int: ...
    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float) -> tuple[float, float]: ...
    def val_loss(self, x: np.ndarray, y: np.ndarray) -> float: ...
    def get_weights_for_checkpoint(self) -> dict[str, np.ndarray]: ...
    def load_weights(self, weights: dict[str, np.ndarray]) -> None: ...
    def optimizer_state(self) -> dict[str, Any]: ...
    def load_optimizer_state(self, state: dict[str, Any]) -> None: ...
    def peak_memory_bytes(self) -> float: ...
    def framework_version(self) -> str: ...


class TorchAdapter:
    """PyTorch. Parameters stay float32. Compute precision is set by
    torch_precision (see TORCH_PRECISIONS): "fp32" runs the forward and backward
    in plain float32 on any device, which is the registered precision the CUDA
    staged run uses; "auto" and "bf16" run them under bf16 autocast on an
    accelerator, which keeps a float32 master copy of the weights instead of
    rounding them. Cross entropy is taken in float32 in every case, so the logged
    loss is never a bf16 readout.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        device: str,
        seed: int,
        peak_lr: float,
        micro_batch_size: Optional[int] = None,
        torch_precision: str = DEFAULT_TORCH_PRECISION,
    ) -> None:
        import torch.nn.functional as F

        from . import model_torch

        self._F = F
        torch.manual_seed(seed)
        self.micro_batch_size = micro_batch_size
        self.device = torch.device(device)
        self.model = model_torch.TransformerLM(cfg).to(self.device)
        self.model.train()

        decay = [p for n, p in self.model.named_parameters() if should_decay(n)]
        no_decay = [p for n, p in self.model.named_parameters() if not should_decay(n)]
        self.opt = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": WEIGHT_DECAY},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=peak_lr,
            betas=(ADAM_BETA1, ADAM_BETA2),
            eps=ADAM_EPS,
        )
        # None means plain float32. Resolved by policy from the device and the
        # requested precision: float32 on CPU always, float32 on any device for
        # "fp32", bf16 autocast on an accelerator otherwise.
        self.torch_precision = torch_precision
        self.autocast_dtype = resolve_torch_autocast_dtype(
            self.device.type, torch_precision
        )
        # Registered precision is float32. On an NVIDIA accelerator that also means
        # refusing TF32, which autocast suppression alone does not do. Recorded so
        # the run manifest carries proof of the numeric regime it trained in.
        self.tf32_policy = (
            disable_tf32() if self.autocast_dtype is None else {"skipped": "autocast"}
        )

    def param_count(self) -> int:
        # parameters() deduplicates the tied embedding, matching config.n_params.
        return sum(p.numel() for p in self.model.parameters())

    def framework_version(self) -> str:
        return str(torch.__version__)

    def _batch(self, x: np.ndarray, y: np.ndarray):
        return (
            torch.from_numpy(x).to(self.device),
            torch.from_numpy(y).to(self.device),
        )

    def _loss(self, xt: torch.Tensor, yt: torch.Tensor) -> torch.Tensor:
        logits = self.model(xt)
        v = logits.shape[-1]
        return self._F.cross_entropy(logits.reshape(-1, v).float(), yt.reshape(-1))

    def _forward_loss(self, xt: torch.Tensor, yt: torch.Tensor) -> torch.Tensor:
        if self.autocast_dtype is not None:
            with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype):
                return self._loss(xt, yt)
        return self._loss(xt, yt)

    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float) -> tuple[float, float]:
        micro, accum_steps = resolve_accumulation(x.shape[0], self.micro_batch_size)
        for group in self.opt.param_groups:
            group["lr"] = lr
        self.model.train()
        self.opt.zero_grad(set_to_none=True)

        total_loss = 0.0
        for i in range(accum_steps):
            lo, hi = i * micro, (i + 1) * micro
            xt, yt = self._batch(x[lo:hi], y[lo:hi])
            loss = self._forward_loss(xt, yt)
            # _loss already means over this micro-batch's tokens, and every
            # micro-batch holds the same count, so dividing by accum_steps and
            # summing gives exactly the full-batch mean. Scaling before backward
            # rather than after is what keeps the accumulated .grad equal to the
            # gradient of that mean; without it the effective learning rate
            # would scale with accum_steps.
            scaled = loss / accum_steps
            scaled.backward()
            total_loss += float(scaled.detach().cpu())

        # Once, on the accumulated gradient, exactly as the un-accumulated path
        # clips the full-batch gradient. Returns the pre-clip global norm.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), GRAD_CLIP_NORM
        )
        self.opt.step()
        return total_loss, float(grad_norm.detach().cpu())

    @torch.no_grad()
    def val_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        self.model.eval()
        xt, yt = self._batch(x, y)
        loss = self._loss(xt, yt)
        self.model.train()
        return float(loss.detach().cpu())

    def get_weights_for_checkpoint(self) -> dict[str, np.ndarray]:
        # state_dict carries lm_head.weight even when tied, which is
        # what the strict=True loader in scripts/06_evaluate.py expects.
        return {
            k: v.detach().to(torch.float32).cpu().numpy()
            for k, v in self.model.state_dict().items()
        }

    def load_weights(self, weights: dict[str, np.ndarray]) -> None:
        state = {k: torch.from_numpy(np.asarray(v)).to(torch.float32) for k, v in weights.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device)

    def optimizer_state(self) -> dict[str, Any]:
        return self.opt.state_dict()

    def load_optimizer_state(self, state: dict[str, Any]) -> None:
        self.opt.load_state_dict(state)

    def peak_memory_bytes(self) -> float:
        if self.device.type == "mps":
            return float(torch.mps.driver_allocated_memory())
        if self.device.type == "cuda":
            return float(torch.cuda.max_memory_allocated())
        import psutil

        return float(psutil.Process().memory_info().rss)


class MlxAdapter:
    """MLX. Parameters stay float32, matching the torch adapter, so the pilot
    compares two frameworks doing the same arithmetic instead of one of them
    quietly training in bf16. MLX has no autocast, so a bf16 compute variant
    would mean bf16 parameters, which is a separate decision and not this
    module's to make.

    Two places where MLX's stock AdamW would silently disagree with torch's, and
    what is done about each:

      Weight decay. mlx.optimizers.AdamW has no parameter groups and decays
        every array it is handed, which would decay the RMSNorm weights and the
        embedding against the pre-registration. So the optimizer here is plain
        Adam and the decoupled decay is applied by name to the matrices only.
        The update it performs, param * (1 - lr * wd) before the Adam step, is
        the same expression MLX's own AdamW uses and the same one torch uses.

      Bias correction. mlx.optimizers.Adam defaults to bias_correction=False,
        while torch's AdamW always corrects. Left alone that difference would
        make the two frameworks train differently from identical settings, which
        would wreck the comparison the pilot exists to make. It is set to True
        here, which reduces MLX's update to torch's exactly.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        device: str,
        seed: int,
        peak_lr: float,
        micro_batch_size: Optional[int] = None,
    ) -> None:
        import mlx.core as mx
        import mlx.nn as mlx_nn
        import mlx.optimizers as mlx_optim
        from mlx.utils import tree_flatten, tree_map, tree_unflatten

        from . import model_mlx

        self.mx = mx
        self.mlx_nn = mlx_nn
        self.mlx_optim = mlx_optim
        self._tree_flatten = tree_flatten
        self._tree_map = tree_map
        self._tree_unflatten = tree_unflatten
        self.micro_batch_size = micro_batch_size

        # MLX selects its device globally. Honoring cfg.device here is what lets
        # the CPU tests keep off the GPU.
        if device == "cpu":
            mx.set_default_device(mx.cpu)
        elif device in ("gpu", "mps"):
            mx.set_default_device(mx.gpu)
        else:
            raise ValueError(f"mlx supports cpu or gpu, got {device!r}")

        mx.random.seed(seed)
        self.cfg = cfg
        self.model = model_mlx.TransformerLM(cfg)
        mx.eval(self.model.parameters())

        self.opt = mlx_optim.Adam(
            learning_rate=peak_lr,
            betas=[ADAM_BETA1, ADAM_BETA2],
            eps=ADAM_EPS,
            bias_correction=True,
        )
        # Differentiates the scaled loss rather than scaling the gradient
        # afterwards, which is what torch's loss.backward() on a divided loss
        # does, so the two adapters stay the same arithmetic under accumulation
        # as well as without it.
        self._loss_and_grad = mlx_nn.value_and_grad(self.model, self._scaled_loss_fn)
        self._decay_names = {
            name for name, _ in tree_flatten(self.model.parameters()) if should_decay(name)
        }
        mx.reset_peak_memory()

    def _loss_fn(self, model, x, y):
        logits = model(x)
        v = logits.shape[-1]
        flat = logits.reshape(-1, v).astype(self.mx.float32)
        return self.mx.mean(
            self.mlx_nn.losses.cross_entropy(flat, y.reshape(-1), reduction="none")
        )

    def _scaled_loss_fn(self, model, x, y, scale: float):
        return self._loss_fn(model, x, y) * scale

    def param_count(self) -> int:
        return int(sum(a.size for _, a in self._tree_flatten(self.model.parameters())))

    def framework_version(self) -> str:
        return str(package_version("mlx"))

    def _apply_weight_decay(self, lr: float) -> None:
        if WEIGHT_DECAY == 0.0:
            return
        factor = 1.0 - lr * WEIGHT_DECAY
        decayed = [
            (name, p * factor if name in self._decay_names else p)
            for name, p in self._tree_flatten(self.model.parameters())
        ]
        self.model.update(self._tree_unflatten(decayed))

    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float) -> tuple[float, float]:
        mx = self.mx
        micro, accum_steps = resolve_accumulation(x.shape[0], self.micro_batch_size)
        scale = 1.0 / accum_steps

        accum = None
        total_loss = 0.0
        for i in range(accum_steps):
            lo, hi = i * micro, (i + 1) * micro
            xa, ya = mx.array(x[lo:hi]), mx.array(y[lo:hi])
            loss, grads = self._loss_and_grad(self.model, xa, ya, scale)
            accum = (
                grads
                if accum is None
                else self._tree_map(lambda a, g: a + g, accum, grads)
            )
            # MLX is lazy, so without this the graph for every micro-batch stays
            # unevaluated until the optimizer step, holding every micro-batch's
            # activations live at once and handing back exactly the memory the
            # split exists to save. Forcing the accumulator here keeps the live
            # set at one micro-batch of activations plus one gradient tree, and
            # the graph does not grow with accum_steps.
            mx.eval(accum, loss)
            total_loss += float(loss)

        # Once, on the accumulated gradient. Returns the pre-clip norm, as
        # torch's clip_grad_norm_ does, and scales by
        # min(max_norm / (norm + 1e-6), 1), which is torch's expression too.
        grads, grad_norm = self.mlx_optim.clip_grad_norm(accum, GRAD_CLIP_NORM)
        self.opt.learning_rate = lr
        self._apply_weight_decay(lr)
        self.opt.update(self.model, grads)
        mx.eval(self.model.parameters(), self.opt.state, grad_norm)
        return total_loss, float(grad_norm)

    def val_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        mx = self.mx
        loss = self._loss_fn(self.model, mx.array(x), mx.array(y))
        mx.eval(loss)
        return float(loss)

    def get_weights_for_checkpoint(self) -> dict[str, np.ndarray]:
        out = {
            name: np.array(arr, dtype=np.float32)
            for name, arr in self._tree_flatten(self.model.parameters())
        }
        # The MLX model has no lm_head parameter when embeddings are tied; the
        # head is embed.as_linear. A torch state_dict lists the tie explicitly,
        # so emit it here or the strict=True load on the eval side fails.
        if self.cfg.tie_embeddings:
            out["lm_head.weight"] = out["embed.weight"].copy()
        return out

    def load_weights(self, weights: dict[str, np.ndarray]) -> None:
        mx = self.mx
        items = [
            (name, mx.array(np.asarray(arr, dtype=np.float32)))
            for name, arr in weights.items()
            if not (self.cfg.tie_embeddings and name == "lm_head.weight")
        ]
        self.model.load_weights(items)
        mx.eval(self.model.parameters())

    def optimizer_state(self) -> dict[str, Any]:
        """opt.state as plain numpy, so the resume file holds nothing MLX specific.

        The tree carries the per-parameter moments plus the step counter and the
        learning rate. Every leaf is kept, array or not, and each is tagged with
        whether it was an mx.array, so the restore rebuilds the tree exactly.
        Dropping a leaf here would silently reset the Adam step counter, and the
        bias correction reads that counter.
        """
        flat: list[tuple[str, Any, bool]] = []
        for name, value in self._tree_flatten(self.opt.state):
            if isinstance(value, self.mx.array):
                flat.append((name, np.array(value), True))
            else:
                flat.append((name, value, False))
        return {"flat": flat}

    def load_optimizer_state(self, state: dict[str, Any]) -> None:
        mx = self.mx
        items = [
            (name, mx.array(value) if was_array else value)
            for name, value, was_array in state["flat"]
        ]
        self.opt.state = self._tree_unflatten(items)
        mx.eval(self.opt.state)

    def peak_memory_bytes(self) -> float:
        return float(self.mx.get_peak_memory())


def make_adapter(
    framework: str,
    cfg: ModelConfig,
    device: str,
    seed: int,
    peak_lr: float,
    micro_batch_size: Optional[int] = None,
    torch_precision: str = DEFAULT_TORCH_PRECISION,
):
    if framework == "torch":
        return TorchAdapter(
            cfg, device, seed, peak_lr, micro_batch_size, torch_precision
        )
    if framework == "mlx":
        # MLX has no autocast and always trains in float32, so torch_precision
        # is not applicable and is deliberately not passed on.
        return MlxAdapter(cfg, device, seed, peak_lr, micro_batch_size)
    raise ValueError(f"unknown framework {framework!r}; expected torch or mlx")


# ----------------------------------------------------------------------------
# data stream position
# ----------------------------------------------------------------------------

def batcher_stream_state(batcher: MixtureBatcher) -> dict[str, Any]:
    """The batcher's exact position in its stream.

    Reaches for the generator's bit generator state, which is a documented numpy
    API returning a plain dict of integers. The private attribute is the only
    liberty taken, and it is taken because the alternative is replaying every
    batch since step zero on resume: tens of thousands of steps of random reads
    over memmapped streams, to arrive at a position numpy can hand over
    directly. data.py is verified and is left untouched.
    """
    return dict(batcher._rng.bit_generator.state)


def restore_batcher_stream(batcher: MixtureBatcher, state: dict[str, Any]) -> None:
    batcher._rng.bit_generator.state = state


# ----------------------------------------------------------------------------
# checkpoints
# ----------------------------------------------------------------------------

def checkpoint_path(checkpoint_dir: Path, tokens: int) -> Path:
    # Zero padded so a lexical listing is also a chronological one.
    return Path(checkpoint_dir) / f"{CHECKPOINT_PREFIX}{tokens:012d}.pt"


def _atomic_torch_save(obj: Any, path: Path) -> None:
    """Save through a temporary file, so a crash mid-write cannot leave a
    truncated checkpoint that later reads as a real one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(
    checkpoint_dir: Path,
    weights: dict[str, np.ndarray],
    tokens: int,
    step: int,
    target_tokens: int,
    meta: dict[str, Any],
) -> Path:
    """Write one pre-registered checkpoint: bf16 weights, no optimizer state.

    Shaped for the loader in scripts/06_evaluate.py, which was written first:
    the state_dict sits under "model" and the token count under "tokens", and it
    loads with strict=True.
    """
    state = {k: torch.from_numpy(np.asarray(v)).to(torch.bfloat16) for k, v in weights.items()}
    obj = {
        "model": state,
        "tokens": int(tokens),
        "target_tokens": int(target_tokens),
        "step": int(step),
        "dtype": "bfloat16",
        **meta,
    }
    path = checkpoint_path(checkpoint_dir, tokens)
    _atomic_torch_save(obj, path)
    return path


def save_resume_state(
    checkpoint_dir: Path,
    adapter: Adapter,
    batcher: MixtureBatcher,
    step: int,
    tokens: int,
    cfg: TrainConfig,
) -> Path:
    """Overwrite the single resume file. float32 weights plus optimizer state.

    One file, overwritten in place, is what keeps the pre-registration's "only
    the most recent checkpoint carries optimizer state" true by construction.
    """
    obj = {
        "weights_fp32": {
            k: torch.from_numpy(np.asarray(v))
            for k, v in adapter.get_weights_for_checkpoint().items()
        },
        "optimizer": adapter.optimizer_state(),
        "stream_state": batcher_stream_state(batcher),
        "step": int(step),
        "tokens": int(tokens),
        "framework": cfg.framework,
        "run_name": cfg.run_name,
        "seed": int(cfg.seed),
    }
    path = Path(checkpoint_dir) / RESUME_STATE_NAME
    _atomic_torch_save(obj, path)
    return path


# ----------------------------------------------------------------------------
# manifest
# ----------------------------------------------------------------------------

def git_commit() -> dict[str, Any]:
    """Commit hash and whether the tree was dirty. Never fatal."""
    out: dict[str, Any] = {"commit": None, "dirty": None}
    try:
        out["commit"] = subprocess.check_output(
            ["git", "rev-parse", "head"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode()
        out["dirty"] = bool(status.strip())
    except Exception:
        pass
    return out


def file_sha256(path: Path) -> Optional[str]:
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(cfg: TrainConfig, adapter: Adapter, model_cfg: ModelConfig) -> dict[str, Any]:
    """Everything needed to say what produced a run, recorded before it starts."""
    train_cfg = asdict(cfg)
    # Roots are recorded through portable_path so a manifest names the artifact
    # rather than the machine that happened to hold it.
    for key in ("checkpoint_root", "out_root", "tokenized_root"):
        train_cfg[key] = portable_path(train_cfg[key])
    if train_cfg.get("model_config") is not None:
        train_cfg["model_config"] = asdict(cfg.resolved_model_config())

    return {
        "run_name": cfg.run_name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "train_config": train_cfg,
        "model_config": asdict(model_cfg),
        "n_params": n_params(model_cfg),
        "n_params_adapter": adapter.param_count(),
        "tokens_per_step": cfg.tokens_per_step,
        "batch_size": cfg.batch_size,
        "micro_batch_size": cfg.resolved_micro_batch_size,
        "accum_steps": cfg.accum_steps,
        "total_steps": cfg.total_steps,
        "warmup_steps": (
            warmup_steps_for_tokens(
                cfg.warmup_tokens, cfg.tokens_per_step, cfg.total_steps
            )
            if cfg.lr_schedule == "constant"
            else warmup_steps_for(cfg.total_steps, cfg.warmup_fraction)
        ),
        "seed": cfg.seed,
        "data_mix": cfg.mix,
        "framework": cfg.framework,
        "framework_version": adapter.framework_version(),
        "device": cfg.device,
        "torch_version": str(torch.__version__),
        "tokenizer_sha256": file_sha256(TOKENIZER_PATH),
        "tokenizer_path": TOKENIZER_PATH.relative_to(REPO_ROOT).as_posix(),
        "git": git_commit(),
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "optimizer": {
            "name": "adamw",
            "beta1": ADAM_BETA1,
            "beta2": ADAM_BETA2,
            "eps": ADAM_EPS,
            "weight_decay": WEIGHT_DECAY,
            "decay_excludes": ["embedding", "rmsnorm weights"],
            "grad_clip_norm": GRAD_CLIP_NORM,
            "schedule": (
                "constant at peak with linear warmup over warmup_tokens, no decay"
                if cfg.lr_schedule == "constant"
                else "cosine to 10 percent of peak with linear warmup"
            ),
            "lr_schedule": cfg.lr_schedule,
            "peak_lr": cfg.peak_lr,
            "warmup_fraction": cfg.warmup_fraction,
            "warmup_tokens": cfg.warmup_tokens,
        },
        "resumes": [],
    }


# ----------------------------------------------------------------------------
# the loop
# ----------------------------------------------------------------------------

def _build_val_batches(cfg: TrainConfig, sources: Optional[Sequence[DomainSource]]) -> list:
    """A fixed held-out slice, materialized once and reused for every check.

    Fixed means fixed: the same batches at every validation, in every run, at
    every size, drawn with VAL_SEED instead of the run seed. VAL split only.
    """
    if cfg.val_every <= 0 or cfg.val_batches <= 0:
        return []
    if sources is None:
        sources = load_sources(cfg.mix, split=VAL_SPLIT, root=cfg.tokenized_root)
    batcher = MixtureBatcher(
        sources, cfg.batch_size, cfg.resolved_model_config().context_len, VAL_SEED
    )
    return [batcher.next_batch() for _ in range(cfg.val_batches)]


def train(
    cfg: TrainConfig,
    sources: Optional[Sequence[DomainSource]] = None,
    val_sources: Optional[Sequence[DomainSource]] = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run one training run to its token budget. Returns a small summary.

    sources and val_sources exist so a test can hand in synthetic streams
    instead of the tokenized corpus. A real run leaves them None and the streams
    are memmapped from tokenized_root.
    """
    model_cfg = cfg.resolved_model_config()
    run_dir = cfg.run_dir
    checkpoint_dir = cfg.checkpoint_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    adapter = make_adapter(
        cfg.framework,
        model_cfg,
        cfg.device,
        cfg.seed,
        cfg.peak_lr,
        cfg.resolved_micro_batch_size,
        cfg.torch_precision,
    )

    if sources is None:
        sources = load_sources(cfg.mix, split="train", root=cfg.tokenized_root)
    # At batch_size, never at micro_batch_size. The adapter splits the batch it
    # is handed, so the stream draws the same windows in the same order at every
    # accumulation setting and a micro-batch 16 run stays comparable to a
    # micro-batch 32 one.
    batcher = MixtureBatcher(sources, cfg.batch_size, model_cfg.context_len, cfg.seed)
    val_batches = _build_val_batches(cfg, val_sources)

    total_steps = cfg.total_steps
    tokens_per_step = cfg.tokens_per_step
    targets = (
        tuple(cfg.checkpoint_token_targets)
        if cfg.checkpoint_token_targets is not None
        else PREREG_CHECKPOINT_TOKENS
    )
    schedule = checkpoint_steps(total_steps, tokens_per_step, targets)

    end_step = total_steps
    if cfg.stop_after_steps is not None:
        end_step = min(total_steps, cfg.stop_after_steps)

    start_step = 0
    if resume:
        start_step = _restore(cfg, adapter, batcher, checkpoint_dir)

    manifest_path = run_dir / MANIFEST_NAME
    if resume and manifest_path.exists():
        with manifest_path.open("r") as f:
            manifest = json.load(f)
        manifest.setdefault("resumes", []).append(
            {"at": datetime.now(timezone.utc).isoformat(), "from_step": start_step}
        )
    else:
        manifest = build_manifest(cfg, adapter, model_cfg)
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    log_path = run_dir / LOG_NAME
    # Append on resume instead of truncate, so a run's history survives being
    # killed. A crash between two resume-state writes leaves the log ahead of
    # the resume point, and the replayed steps then appear twice. The duplicates
    # are real history and are left in place; read the log as last-record-wins
    # per step rather than assuming one record each.
    log_file = log_path.open("a" if resume else "w")

    meta = {
        "run_name": cfg.run_name,
        "size": cfg.size,
        "framework": cfg.framework,
        "seed": cfg.seed,
        "peak_lr": cfg.peak_lr,
        "model_config": asdict(model_cfg),
    }

    started = time.perf_counter()
    window_started = started
    window_tokens = 0
    last_loss = float("nan")
    last_val = None
    checkpoints_written: list[str] = []
    projected = False

    try:
        for step in range(start_step, end_step):
            is_last = step == end_step - 1
            lr = lr_for_config(cfg, step, total_steps)
            x, y = batcher.next_batch()
            loss, grad_norm = adapter.train_step(x, y, lr)
            tokens = (step + 1) * tokens_per_step
            window_tokens += tokens_per_step
            last_loss = loss

            if not math.isfinite(loss):
                raise RuntimeError(
                    f"loss went non-finite at step {step} (lr {lr:.3e}). Stopping "
                    "rather than writing a checkpoint of a diverged model."
                )

            do_val = bool(
                cfg.val_every > 0
                and val_batches
                and ((step + 1) % cfg.val_every == 0 or is_last)
            )
            if do_val:
                last_val = float(
                    np.mean([adapter.val_loss(vx, vy) for vx, vy in val_batches])
                )

            if (step + 1) % cfg.log_every == 0 or is_last or do_val:
                now = time.perf_counter()
                elapsed_window = max(now - window_started, 1e-9)
                tokens_per_sec = window_tokens / elapsed_window
                record = {
                    "step": step,
                    "tokens": tokens,
                    "loss": loss,
                    "lr": lr,
                    "grad_norm": grad_norm,
                    "tokens_per_sec": tokens_per_sec,
                    "elapsed": now - started,
                    "peak_mem_gb": adapter.peak_memory_bytes() / 1e9,
                }
                if last_val is not None:
                    record["val_loss"] = last_val
                    last_val = None
                log_file.write(json.dumps(record) + "\n")
                # Flushed every record: a killed run keeps its history.
                log_file.flush()
                os.fsync(log_file.fileno())
                window_started = now
                window_tokens = 0

                if cfg.verbose:
                    val_str = (
                        f" val {record['val_loss']:.4f}" if "val_loss" in record else ""
                    )
                    print(
                        f"step {step + 1}/{total_steps}  tokens {tokens:,}  "
                        f"loss {loss:.4f}{val_str}  lr {lr:.3e}  "
                        f"gnorm {grad_norm:.2f}  {tokens_per_sec:,.0f} tok/s",
                        flush=True,
                    )

            if not projected and cfg.verbose and step - start_step + 1 >= cfg.projection_after:
                projected = True
                rate = (step - start_step + 1) * tokens_per_step / (
                    time.perf_counter() - started
                )
                remaining = (total_steps - step - 1) * tokens_per_step
                print(
                    f"measured {rate:,.0f} tok/s over the first "
                    f"{step - start_step + 1} steps -> projected "
                    f"{remaining / rate / 3600:.1f} h for the remaining "
                    f"{remaining:,} tokens",
                    flush=True,
                )

            if step in schedule:
                path = save_checkpoint(
                    checkpoint_dir,
                    adapter.get_weights_for_checkpoint(),
                    tokens,
                    step,
                    schedule[step],
                    meta,
                )
                checkpoints_written.append(str(path))
                save_resume_state(checkpoint_dir, adapter, batcher, step, tokens, cfg)
                if cfg.verbose:
                    print(f"checkpoint {path.name} at {tokens:,} tokens", flush=True)
            elif cfg.resume_state_every > 0 and (step + 1) % cfg.resume_state_every == 0:
                # Cheap insurance between the sparse late checkpoints, and not a
                # checkpoint: it is overwritten, so it adds nothing to the
                # pre-registered set and keeps optimizer state in one place.
                save_resume_state(checkpoint_dir, adapter, batcher, step, tokens, cfg)
    finally:
        log_file.close()

    return {
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "manifest": str(manifest_path),
        "log": str(log_path),
        "final_loss": last_loss,
        "total_steps": total_steps,
        "tokens": total_steps * tokens_per_step,
        "checkpoints": checkpoints_written,
        "elapsed": time.perf_counter() - started,
    }


def _restore(
    cfg: TrainConfig, adapter: Adapter, batcher: MixtureBatcher, checkpoint_dir: Path
) -> int:
    """Load the resume file and return the step to continue from."""
    path = Path(checkpoint_dir) / RESUME_STATE_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"cannot resume: no {RESUME_STATE_NAME} under {checkpoint_dir}"
        )
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["framework"] != cfg.framework:
        raise ValueError(
            f"resume state was written by {state['framework']!r} but this run is "
            f"{cfg.framework!r}. Optimizer state does not carry across frameworks."
        )
    adapter.load_weights(
        {k: v.numpy() for k, v in state["weights_fp32"].items()}
    )
    adapter.load_optimizer_state(state["optimizer"])
    restore_batcher_stream(batcher, state["stream_state"])
    next_step = int(state["step"]) + 1
    if cfg.verbose:
        print(
            f"resumed from step {state['step']} at {state['tokens']:,} tokens; "
            f"continuing at step {next_step}",
            flush=True,
        )
    return next_step
