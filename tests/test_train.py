"""Tests for the shared training loop and the two framework adapters.

Everything here runs on CPU. The shared GPU lock belongs to another project, and
these tests must never contend for it: the torch adapter is built with
device="cpu" and the MLX adapter with device="cpu", which makes it call
mx.set_default_device(mx.cpu) rather than reaching for the GPU.

The load-bearing test is test_loss_decreases_*. A smoke test that only checks a
step runs to completion passes just as happily when the optimizer is silently
not updating anything, which is the failure worth catching. So the anchor is a
trivially learnable synthetic task, run through the real train_step, asserted to
actually bring the loss down, in both frameworks.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from small_lm_lab.config import ModelConfig, n_params
from small_lm_lab.data import DomainSource
from small_lm_lab.train import (
    DEFAULT_WARMUP_TOKENS,
    GRAD_CLIP_NORM,
    LR_FINAL_FRACTION,
    PREREG_CHECKPOINT_TOKENS,
    PREREG_TOKENS_PER_STEP,
    WEIGHT_DECAY,
    MlxAdapter,
    TorchAdapter,
    TrainConfig,
    checkpoint_path,
    checkpoint_steps,
    lr_at_step,
    lr_at_step_constant,
    lr_for_config,
    make_adapter,
    resolve_accumulation,
    disable_tf32,
    resolve_torch_autocast_dtype,
    save_checkpoint,
    should_decay,
    train,
    warmup_steps_for,
    warmup_steps_for_tokens,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = REPO_ROOT / "scripts" / "06_evaluate.py"

FRAMEWORKS = ("torch", "mlx")

# The cycle the synthetic task repeats. Short enough that a two layer model
# learns it from the token identity alone, with no attention needed, so a
# working optimizer cannot fail to drive the loss down.
CYCLE = 8


@pytest.fixture(autouse=True)
def force_cpu():
    """These tests must never touch the shared GPU.

    The MLX adapter already pins the device when built with device="cpu", but
    MLX selects its device from a process-global default that starts at the GPU,
    so anything constructing an mx.array outside an adapter would land there.
    Pinning it per test makes the guarantee independent of test order and of
    whatever ran before.
    """
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    yield


def tiny_config() -> ModelConfig:
    """Small enough to train on CPU inside a test, real vocab so the checkpoint
    plumbing is exercised at the shape the lab actually uses."""
    return ModelConfig(
        vocab_size=16384,
        d_model=64,
        n_layers=2,
        n_heads=1,
        context_len=64,
        head_dim=64,
    )


def cycle_batch(batch_size: int, context_len: int, seed: int = 0):
    """(x, y) for the repeating-cycle task. y is the next-token shift of x."""
    rng = np.random.default_rng(seed)
    offsets = rng.integers(0, CYCLE, size=batch_size)
    pos = np.arange(context_len + 1)[None, :] + offsets[:, None]
    seq = (pos % CYCLE).astype(np.int64)
    return seq[:, :-1], seq[:, 1:]


def cycle_sources(n_tokens: int = 20_000) -> list[DomainSource]:
    """One synthetic stream of the repeating cycle.

    Lets the train() tests run without the tokenized corpus on the bulk root, and
    keeps the loss moving, so a resume that restored the wrong weights, the
    wrong optimizer state, or the wrong stream position shows up as a diverging
    trajectory instead of hiding in the noise of an unlearnable task.
    """
    tokens = np.tile(np.arange(CYCLE, dtype=np.uint16), n_tokens // CYCLE)
    return [DomainSource(name="cycle", weight=1.0, tokens=tokens)]


def tiny_train_config(tmp_path: Path, framework: str, **kwargs) -> TrainConfig:
    """A CPU-sized run. Defaults here are test conveniences, not lab settings:
    validation off, every step logged, and the throughput projection pushed out
    of reach so short runs stay quiet."""
    settings: dict = {
        "run_name": f"test_{framework}",
        "peak_lr": 3e-3,
        "batch_size": 4,
        "val_every": 0,
        "log_every": 1,
        "projection_after": 10**9,
    }
    settings.update(kwargs)
    return TrainConfig(
        size="tiny",
        framework=framework,
        device="cpu",
        seed=1,
        mix={"cycle": 1.0},
        checkpoint_root=tmp_path / "checkpoints",
        out_root=tmp_path / "runs",
        model_config=tiny_config(),
        verbose=False,
        **settings,
    )


def load_eval_script():
    """Import scripts/06_evaluate.py by path.

    The checkpoint format is not something to assert against a copy of the
    loader: it has to satisfy the real one, which was written before this code
    and loads with strict=True. scripts/ is not a package, hence the path
    import.
    """
    spec = importlib.util.spec_from_file_location("eval_script", EVAL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAIN_SCRIPT = REPO_ROOT / "scripts" / "07_train.py"


def load_train_script():
    """Import scripts/07_train.py by path, for load_checkpoint_schedule."""
    spec = importlib.util.spec_from_file_location("train_script", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------
# schedule
# ----------------------------------------------------------------------------

def test_warmup_ends_exactly_at_peak() -> None:
    total, peak, frac = 1000, 1.2e-3, 0.02
    warmup = warmup_steps_for(total, frac)
    assert warmup == 20
    # The last warmup step runs at peak, not at peak minus one increment.
    assert lr_at_step(warmup - 1, total, peak, frac) == pytest.approx(peak, rel=0, abs=0)


def test_schedule_is_continuous_at_the_warmup_boundary() -> None:
    total, peak, frac = 1000, 1.2e-3, 0.02
    warmup = warmup_steps_for(total, frac)
    last_warmup = lr_at_step(warmup - 1, total, peak, frac)
    first_decay = lr_at_step(warmup, total, peak, frac)
    # Both sides sit at peak, so the cosine starts where the ramp finished
    # rather than skipping a step or repeating one at a lower rate.
    assert last_warmup == pytest.approx(peak)
    assert first_decay == pytest.approx(peak)
    assert first_decay == pytest.approx(last_warmup)


def test_schedule_ends_at_exactly_ten_percent_of_peak() -> None:
    total, peak, frac = 1000, 1.2e-3, 0.02
    final = lr_at_step(total - 1, total, peak, frac)
    assert final == pytest.approx(peak * LR_FINAL_FRACTION)
    assert LR_FINAL_FRACTION == 0.10


def test_schedule_warms_up_then_decays_monotonically() -> None:
    total, peak, frac = 500, 1e-3, 0.02
    warmup = warmup_steps_for(total, frac)
    lrs = [lr_at_step(s, total, peak, frac) for s in range(total)]
    assert all(lrs[i] < lrs[i + 1] for i in range(warmup - 1))
    assert all(lrs[i] >= lrs[i + 1] for i in range(warmup, total - 1))
    assert min(lrs) == pytest.approx(peak * LR_FINAL_FRACTION)
    assert max(lrs) == pytest.approx(peak)


def test_schedule_with_no_warmup_starts_at_peak() -> None:
    # The pre-registered warmup ablation runs warmup 0 at the winning lr.
    total, peak = 500, 1e-3
    assert warmup_steps_for(total, 0.0) == 0
    assert lr_at_step(0, total, peak, 0.0) == pytest.approx(peak)
    assert lr_at_step(total - 1, total, peak, 0.0) == pytest.approx(peak * LR_FINAL_FRACTION)


def test_schedule_rejects_steps_outside_the_run() -> None:
    with pytest.raises(ValueError):
        lr_at_step(100, 100, 1e-3, 0.02)
    with pytest.raises(ValueError):
        lr_at_step(-1, 100, 1e-3, 0.02)


# The cosine schedule is the pre-registered Phase A path. Adding the constant
# schedule for the staged run must not perturb it by a single bit, so these are
# characterization values captured from the implementation at the time the
# constant path was added. If any of them move, the cosine schedule changed, and
# every Phase A run and every completed ablation would no longer be reproducible
# from the code. The values are dense around the warmup boundary because that is
# where an off-by-one would hide.
_COSINE_GOLDEN_MAIN = (
    # total_steps=1000, peak=1.2e-3, warmup_fraction=0.02 (warmup=20 steps)
    (0, 5.99999999999999947438e-05),
    (1, 1.19999999999999989488e-04),
    (5, 3.59999999999999968463e-04),
    (10, 6.59999999999999887972e-04),
    (19, 1.19999999999999989488e-03),
    (20, 1.19999999999999967804e-03),
    (21, 1.19999721966098944065e-03),
    (100, 1.18230332686931720321e-03),
    (500, 6.76459524131164057457e-04),
    (900, 1.47021707598784959290e-04),
    (999, 1.19999999999999989488e-04),
)
_COSINE_GOLDEN_NO_WARMUP = (
    # total_steps=500, peak=1e-3, warmup_fraction=0.0
    (0, 1.00000000000000002082e-03),
    (1, 9.99991081748044085414e-04),
    (250, 5.48583452545775757982e-04),
    (499, 1.00000000000000004792e-04),
)


def test_cosine_schedule_is_bit_identical_to_the_golden_grid() -> None:
    """The cosine path must not move. Exact equality, not approx: the whole point
    of a regression anchor is that a change of one ulp trips it."""
    for step, want in _COSINE_GOLDEN_MAIN:
        got = lr_at_step(step, 1000, 1.2e-3, 0.02)
        assert got == want, f"cosine lr moved at step {step}: {got!r} != {want!r}"
    for step, want in _COSINE_GOLDEN_NO_WARMUP:
        got = lr_at_step(step, 500, 1e-3, 0.0)
        assert got == want, f"cosine lr moved at step {step}: {got!r} != {want!r}"


def test_lr_for_config_cosine_matches_lr_at_step_exactly() -> None:
    """The trainer reads its rate through lr_for_config. For a cosine run that
    must be exactly lr_at_step, or the dispatcher perturbed the pre-registered
    path just by existing."""
    cfg = TrainConfig(run_name="x", size="size30m", peak_lr=1.2e-3, total_tokens=10_000_000)
    total = cfg.total_steps
    for step in range(0, total, max(total // 50, 1)):
        assert lr_for_config(cfg, step, total) == lr_at_step(
            step, total, cfg.peak_lr, cfg.warmup_fraction
        )


# ----------------------------------------------------------------------------
# constant schedule (staged deep run)
# ----------------------------------------------------------------------------

def test_constant_warmup_endpoint_hits_peak_exactly() -> None:
    """The last warmup step runs at exactly peak, like the cosine ramp does."""
    tps, peak, warmup_tokens = 16_384, 1.2e-3, 4_000_000
    total = 100_000
    warmup = warmup_steps_for_tokens(warmup_tokens, tps, total)
    assert warmup == round(warmup_tokens / tps) == 244
    # No step runs at zero: the ramp starts at 1/warmup, not 0/warmup.
    assert lr_at_step_constant(0, total, peak, warmup_tokens, tps) == pytest.approx(
        peak / warmup
    )
    assert lr_at_step_constant(0, total, peak, warmup_tokens, tps) > 0.0
    # The endpoint is exactly peak, and it is not reached one step early.
    assert lr_at_step_constant(warmup - 1, total, peak, warmup_tokens, tps) == peak
    assert lr_at_step_constant(warmup - 2, total, peak, warmup_tokens, tps) < peak


def test_constant_is_flat_at_peak_after_warmup() -> None:
    """Every step from the warmup boundary onward is exactly peak, with no decay,
    however far into the run it sits. That flatness is the whole reason the staged
    run uses this schedule."""
    tps, peak, warmup_tokens = 16_384, 1.2e-3, 4_000_000
    total = 500_000
    warmup = warmup_steps_for_tokens(warmup_tokens, tps, total)
    for step in (warmup, warmup + 1, 1000, 250_000, total - 1):
        assert lr_at_step_constant(step, total, peak, warmup_tokens, tps) == peak, (
            f"constant lr decayed at step {step}"
        )


def test_constant_warmup_ramps_up_monotonically() -> None:
    tps, peak, warmup_tokens = 16_384, 1.2e-3, 4_000_000
    total = 100_000
    warmup = warmup_steps_for_tokens(warmup_tokens, tps, total)
    lrs = [lr_at_step_constant(s, total, peak, warmup_tokens, tps) for s in range(warmup)]
    assert all(lrs[i] < lrs[i + 1] for i in range(warmup - 1))
    assert max(lrs) == peak
    assert min(lrs) == pytest.approx(peak / warmup)


def test_constant_warmup_never_eats_the_whole_run() -> None:
    """The warmup is capped so at least one step runs flat at peak, even when the
    token budget is shorter than the warmup, and even for a one-step run."""
    tps, peak = 16_384, 1.2e-3
    # Warmup budget far larger than the run: capped at total - 1, one flat step.
    total = 10
    warmup = warmup_steps_for_tokens(10_000_000, tps, total)
    assert warmup == total - 1
    assert lr_at_step_constant(total - 1, total, peak, 10_000_000, tps) == peak
    # A one-step run has no room for warmup at all.
    assert warmup_steps_for_tokens(10_000_000, tps, 1) == 0
    assert lr_at_step_constant(0, 1, peak, 10_000_000, tps) == peak
    # Zero warmup budget starts flat immediately.
    assert warmup_steps_for_tokens(0, tps, 100) == 0
    assert lr_at_step_constant(0, 100, peak, 0, tps) == peak


def test_constant_schedule_does_not_depend_on_the_horizon() -> None:
    """Two runs with different total_steps but the same warmup and peak see the
    same rate at the same step, which is what keeps checkpoints comparable across
    an outcome-contingent stop. The cosine schedule cannot do this."""
    tps, peak, warmup_tokens = 16_384, 1.2e-3, 4_000_000
    for step in (0, 100, 243, 244, 245, 5000):
        a = lr_at_step_constant(step, 100_000, peak, warmup_tokens, tps)
        b = lr_at_step_constant(step, 5_000_000, peak, warmup_tokens, tps)
        assert a == b, f"constant lr moved with the horizon at step {step}"


def test_constant_schedule_rejects_steps_outside_the_run() -> None:
    with pytest.raises(ValueError):
        lr_at_step_constant(100, 100, 1e-3, 4_000_000, 16_384)
    with pytest.raises(ValueError):
        lr_at_step_constant(-1, 100, 1e-3, 4_000_000, 16_384)


def test_lr_for_config_dispatches_on_schedule() -> None:
    tps = PREREG_TOKENS_PER_STEP
    cfg = TrainConfig(
        run_name="x",
        size="size30m",
        lr_schedule="constant",
        peak_lr=1.2e-3,
        warmup_tokens=4_000_000,
        total_tokens=2_000_000_000,
    )
    total = cfg.total_steps
    warmup = warmup_steps_for_tokens(cfg.warmup_tokens, tps, total)
    assert lr_for_config(cfg, warmup - 1, total) == cfg.peak_lr
    assert lr_for_config(cfg, total - 1, total) == cfg.peak_lr  # flat, no decay
    assert lr_for_config(cfg, 0, total) == pytest.approx(cfg.peak_lr / warmup)


def test_default_lr_schedule_is_cosine_and_default_warmup_tokens() -> None:
    cfg = TrainConfig(run_name="x", size="size30m")
    assert cfg.lr_schedule == "cosine"
    assert cfg.warmup_tokens == DEFAULT_WARMUP_TOKENS == 4_000_000


def test_config_rejects_an_unknown_lr_schedule() -> None:
    with pytest.raises(ValueError, match="unknown lr_schedule"):
        TrainConfig(run_name="x", size="size30m", lr_schedule="linear")
    with pytest.raises(ValueError, match="warmup_tokens must be non-negative"):
        TrainConfig(
            run_name="x", size="size30m", lr_schedule="constant", warmup_tokens=-1
        )


def test_constant_schedule_manifest_records_the_schedule(tmp_path: Path) -> None:
    """A constant-schedule run has to say so in its manifest, or the run record
    cannot be told apart from a cosine one."""
    steps, batch_size, context_len = 4, 4, tiny_config().context_len
    cfg = tiny_train_config(
        tmp_path,
        "torch",
        run_name="constant_manifest",
        total_tokens=steps * batch_size * context_len,
        batch_size=batch_size,
        lr_schedule="constant",
        warmup_tokens=2 * batch_size * context_len,  # 2 steps of warmup
    )
    train(cfg, sources=cycle_sources())
    manifest = json.loads((Path(cfg.run_dir) / "manifest.json").read_text())
    assert manifest["optimizer"]["lr_schedule"] == "constant"
    assert manifest["optimizer"]["warmup_tokens"] == 2 * batch_size * context_len
    assert "constant" in manifest["optimizer"]["schedule"]
    assert manifest["warmup_steps"] == 2


# ----------------------------------------------------------------------------
# checkpoint schedule
# ----------------------------------------------------------------------------

def test_checkpoint_schedule_covers_the_preregistered_counts() -> None:
    tokens_per_step = 16384
    total_steps = 30518  # about 500M tokens
    schedule = checkpoint_steps(total_steps, tokens_per_step)
    hit = set(schedule.values())
    for target in PREREG_CHECKPOINT_TOKENS:
        assert target in hit, f"pre-registered checkpoint {target} never fires"
    # Plus the checkpoint at B itself, at the final step.
    assert schedule[total_steps - 1] == total_steps * tokens_per_step
    assert len(schedule) == len(PREREG_CHECKPOINT_TOKENS) + 1


def test_checkpoint_fires_at_the_first_step_past_each_target() -> None:
    tokens_per_step = 16384
    schedule = checkpoint_steps(30518, tokens_per_step)
    step_for = {target: step for step, target in schedule.items()}
    step = step_for[1_000_000]
    assert step * tokens_per_step < 1_000_000 <= (step + 1) * tokens_per_step


def test_short_run_drops_targets_past_its_budget() -> None:
    # A 3M token run must not try to checkpoint at 450M.
    schedule = checkpoint_steps(200, 16384)
    assert max(schedule.values()) == 200 * 16384
    assert all(t <= 200 * 16384 for t in schedule.values())


# ----------------------------------------------------------------------------
# staged run: JSON checkpoint schedule override and resume with an extended one
# ----------------------------------------------------------------------------

def test_load_checkpoint_schedule_reads_sorts_and_dedups(tmp_path: Path) -> None:
    script = load_train_script()
    path = tmp_path / "sched.json"
    path.write_text(json.dumps([4_000_000, 1_000_000, 2_000_000, 1_000_000]))
    assert script.load_checkpoint_schedule(path) == (1_000_000, 2_000_000, 4_000_000)


def test_load_checkpoint_schedule_rejects_bad_files(tmp_path: Path) -> None:
    script = load_train_script()
    bad_list = tmp_path / "notlist.json"
    bad_list.write_text(json.dumps({"tokens": [1, 2]}))
    with pytest.raises(SystemExit, match="non-empty JSON list"):
        script.load_checkpoint_schedule(bad_list)

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps([]))
    with pytest.raises(SystemExit, match="non-empty JSON list"):
        script.load_checkpoint_schedule(empty)

    non_numeric = tmp_path / "str.json"
    non_numeric.write_text(json.dumps([1_000_000, "oops"]))
    with pytest.raises(SystemExit, match="non-numeric"):
        script.load_checkpoint_schedule(non_numeric)

    non_positive = tmp_path / "neg.json"
    non_positive.write_text(json.dumps([1_000_000, 0]))
    with pytest.raises(SystemExit, match="non-positive"):
        script.load_checkpoint_schedule(non_positive)


def test_07_train_cli_wires_the_staged_flags_into_the_config(
    tmp_path: Path, monkeypatch
) -> None:
    """The pod's Stage A invocation must build the config it reads on the tin:
    constant LR with a token warmup, a JSON checkpoint schedule, and a step cap.
    train() is stubbed so nothing touches the corpus or the GPU."""
    import sys

    script = load_train_script()
    schedule_file = tmp_path / "sched.json"
    schedule_file.write_text(json.dumps([1_000_000, 2_000_000, 500_000_000]))

    captured: dict = {}

    def fake_train(cfg, resume=False):
        captured["cfg"] = cfg
        captured["resume"] = resume
        return {"stub": True}

    monkeypatch.setattr(script, "train", fake_train)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "07_train.py", "--size", "size30m", "--framework", "mlx", "--device", "gpu",
            "--lr", "1.2e-3", "--lr-schedule", "constant", "--warmup-tokens", "4000000",
            "--tokens", "2e9", "--seed", "1", "--stop-after-steps", "500",
            "--checkpoint-schedule", str(schedule_file),
            "--run-name", "size30m_staged_seed1",
            "--out-root", str(tmp_path / "runs"),
            "--checkpoint-root", str(tmp_path / "ckpts"),
        ],
    )
    script.main()

    cfg = captured["cfg"]
    assert cfg.lr_schedule == "constant"
    assert cfg.warmup_tokens == 4_000_000
    assert cfg.total_tokens == 2_000_000_000
    assert cfg.stop_after_steps == 500
    assert cfg.checkpoint_token_targets == (1_000_000, 2_000_000, 500_000_000)
    assert cfg.peak_lr == 1.2e-3
    assert cfg.seed == 1


def _checkpoint_tokens_on_disk(checkpoint_dir: Path) -> dict[int, tuple[str, int]]:
    """{token count -> (sha256, mtime_ns)} for every checkpoint in a directory."""
    from small_lm_lab.train import file_sha256

    out: dict[int, tuple[str, int]] = {}
    for path in sorted(checkpoint_dir.glob("tokens_*.pt")):
        obj = torch.load(path, map_location="cpu", weights_only=False)
        out[int(obj["tokens"])] = (file_sha256(path), path.stat().st_mtime_ns)
    return out


def _checkpoint_weights(checkpoint_dir: Path) -> dict[int, dict[str, torch.Tensor]]:
    """{token count -> model weight tensors} for every checkpoint in a directory.

    Compares the model itself rather than the whole file, so bookkeeping that
    legitimately differs between two runs (the run name in the meta block) does
    not read as a difference in the trained weights.
    """
    out: dict[int, dict[str, torch.Tensor]] = {}
    for path in sorted(checkpoint_dir.glob("tokens_*.pt")):
        obj = torch.load(path, map_location="cpu", weights_only=False)
        out[int(obj["tokens"])] = obj["model"]
    return out


def test_resume_with_an_extended_schedule_keeps_old_and_honors_new(
    tmp_path: Path,
) -> None:
    """The densification workflow: stop, extend the checkpoint schedule with new
    future counts, resume. Already-written checkpoints must not be rewritten, and
    the new future counts must fire.

    This is the mechanism the amendment relies on for mid-run densification. It
    works because the trainer only fires a checkpoint at a step it actually
    reaches, and a resumed run starts past every step it already ran, so old
    counts (whose steps are in the past) are left alone and new future counts
    (whose steps lie ahead) are honored.
    """
    batch_size, context_len = 4, tiny_config().context_len
    tps = batch_size * context_len  # 256

    # Stage A: run to a pause at step 6 of an 8-step budget, checkpointing at
    # the token counts 2*tps and 4*tps (steps 1 and 3). A small constant-LR
    # warmup so the schedule is the staged run's schedule, not cosine.
    stage_a = tiny_train_config(
        tmp_path,
        "torch",
        run_name="staged",
        total_tokens=8 * tps,
        batch_size=batch_size,
        lr_schedule="constant",
        warmup_tokens=2 * tps,
        checkpoint_token_targets=(2 * tps, 4 * tps),
        stop_after_steps=6,
        resume_state_every=2,  # last resume state at step 5, so resume starts at 6
    )
    train(stage_a, sources=cycle_sources())

    ckpt_dir = Path(stage_a.checkpoint_dir)
    after_a = _checkpoint_tokens_on_disk(ckpt_dir)
    assert set(after_a) == {2 * tps, 4 * tps}, "stage A wrote the wrong checkpoints"

    # Stage B: resume with an extended superset schedule. It repeats the two
    # already-written counts and adds two new future ones (steps 7 and 9), and
    # the final step (11) always checkpoints.
    extended = (2 * tps, 4 * tps, 8 * tps, 10 * tps)
    stage_b = tiny_train_config(
        tmp_path,
        "torch",
        run_name="staged",
        total_tokens=12 * tps,
        batch_size=batch_size,
        lr_schedule="constant",
        warmup_tokens=2 * tps,
        checkpoint_token_targets=extended,
    )
    train(stage_b, sources=cycle_sources(), resume=True)

    after_b = _checkpoint_tokens_on_disk(ckpt_dir)

    # The already-written checkpoints are byte-identical and were not rewritten.
    for tokens in (2 * tps, 4 * tps):
        assert after_b[tokens] == after_a[tokens], (
            f"checkpoint at {tokens} tokens was rewritten on resume"
        )

    # The new future counts fired, plus the final checkpoint at the new budget.
    assert set(after_b) == {2 * tps, 4 * tps, 8 * tps, 10 * tps, 12 * tps}

    # The log is continuous: every step once, no gap and no duplicate.
    steps = [s for s, _ in read_losses(Path(stage_b.run_dir) / "train_log.jsonl")]
    assert steps == list(range(12))


def test_resume_extended_schedule_matches_a_straight_run(tmp_path: Path) -> None:
    """The densified checkpoints must be the same model a straight run would have
    written at those counts, or densification changed the trajectory.

    A constant-LR run so the schedule does not depend on the horizon, which is
    the whole point of the staged run using it: the resumed segment sees exactly
    the rate the uninterrupted run sees at that step.
    """
    batch_size, context_len = 4, tiny_config().context_len
    tps = batch_size * context_len
    targets = (2 * tps, 6 * tps, 8 * tps)

    straight = tiny_train_config(
        tmp_path,
        "torch",
        run_name="straight",
        total_tokens=10 * tps,
        batch_size=batch_size,
        lr_schedule="constant",
        warmup_tokens=2 * tps,
        checkpoint_token_targets=targets,
    )
    train(straight, sources=cycle_sources())
    want = _checkpoint_weights(Path(straight.checkpoint_dir))

    part1 = tiny_train_config(
        tmp_path,
        "torch",
        run_name="densified",
        total_tokens=10 * tps,
        batch_size=batch_size,
        lr_schedule="constant",
        warmup_tokens=2 * tps,
        checkpoint_token_targets=(2 * tps,),  # sparse schedule to start
        stop_after_steps=5,
        resume_state_every=5,
    )
    train(part1, sources=cycle_sources())

    part2 = tiny_train_config(
        tmp_path,
        "torch",
        run_name="densified",
        total_tokens=10 * tps,
        batch_size=batch_size,
        lr_schedule="constant",
        warmup_tokens=2 * tps,
        checkpoint_token_targets=targets,  # extended before resuming
    )
    train(part2, sources=cycle_sources(), resume=True)
    got = _checkpoint_weights(Path(part2.checkpoint_dir))

    # The densified run's late checkpoints (written after the resume) match the
    # straight run's weights exactly. The count at 2*tps was written before the
    # pause on a sparse schedule and is shared history.
    assert set(got) == set(want) == {2 * tps, 6 * tps, 8 * tps, 10 * tps}
    for tokens in (6 * tps, 8 * tps, 10 * tps):
        assert set(got[tokens]) == set(want[tokens])
        for name, tensor in want[tokens].items():
            assert torch.equal(got[tokens][name], tensor), (
                f"densified checkpoint at {tokens} tokens differs from the "
                f"straight run at {name}"
            )


# ----------------------------------------------------------------------------
# weight decay
# ----------------------------------------------------------------------------

def test_should_decay_rule() -> None:
    assert should_decay("blocks.0.attn.q_proj.weight")
    assert should_decay("blocks.3.mlp.w2.weight")
    assert not should_decay("embed.weight")
    assert not should_decay("lm_head.weight")
    assert not should_decay("blocks.0.attn_norm.weight")
    assert not should_decay("blocks.0.mlp_norm.weight")
    assert not should_decay("final_norm.weight")


def test_torch_param_groups_exclude_norms_and_embedding() -> None:
    adapter = TorchAdapter(tiny_config(), "cpu", seed=1, peak_lr=1e-3)
    wd_by_id = {
        id(p): group["weight_decay"]
        for group in adapter.opt.param_groups
        for p in group["params"]
    }
    decayed, undecayed = [], []
    for name, param in adapter.model.named_parameters():
        assert id(param) in wd_by_id, f"{name} is in no optimizer group"
        wd = wd_by_id[id(param)]
        if name == "embed.weight" or name.endswith("_norm.weight"):
            assert wd == 0.0, f"{name} must not be decayed, got {wd}"
            undecayed.append(name)
        else:
            assert wd == WEIGHT_DECAY, f"{name} must be decayed at {WEIGHT_DECAY}"
            decayed.append(name)

    assert "embed.weight" in undecayed
    assert "final_norm.weight" in undecayed
    # Every norm weight, two per block plus the final one.
    assert len(undecayed) == 2 * tiny_config().n_layers + 1 + 1
    # Every matrix: four attention projections and three MLP ones per block.
    assert len(decayed) == 7 * tiny_config().n_layers
    # No parameter counted twice across the two groups.
    total = sum(len(g["params"]) for g in adapter.opt.param_groups)
    assert total == len(decayed) + len(undecayed)


def test_mlx_decay_set_excludes_norms_and_embedding() -> None:
    from mlx.utils import tree_flatten

    adapter = MlxAdapter(tiny_config(), "cpu", seed=1, peak_lr=1e-3)
    names = [name for name, _ in tree_flatten(adapter.model.parameters())]
    for name in names:
        if name == "embed.weight" or name.endswith("_norm.weight"):
            assert name not in adapter._decay_names, f"{name} must not be decayed"
        else:
            assert name in adapter._decay_names, f"{name} must be decayed"
    assert "embed.weight" not in adapter._decay_names
    assert len(adapter._decay_names) == 7 * tiny_config().n_layers


def test_torch_weight_decay_shrinks_only_matrices() -> None:
    """With zero gradients the Adam update is zero, so anything that moves moved
    because of weight decay. Catches a decay applied to the wrong set."""
    adapter = TorchAdapter(tiny_config(), "cpu", seed=1, peak_lr=1e-3)
    lr = 1e-3
    for group in adapter.opt.param_groups:
        group["lr"] = lr
    before = {n: p.detach().clone() for n, p in adapter.model.named_parameters()}
    for p in adapter.model.parameters():
        p.grad = torch.zeros_like(p)
    adapter.opt.step()

    for name, param in adapter.model.named_parameters():
        moved = not torch.equal(before[name], param.detach())
        if should_decay(name):
            assert moved, f"{name} should have been decayed but did not move"
            expected = before[name] * (1.0 - lr * WEIGHT_DECAY)
            assert torch.allclose(param.detach(), expected, rtol=1e-6, atol=1e-9)
        else:
            assert not moved, f"{name} moved under zero gradients, so it was decayed"


def test_mlx_weight_decay_shrinks_only_matrices() -> None:
    """The same check in MLX, where it is doing real work: mlx.optimizers.AdamW
    has no parameter groups and would decay every array handed to it, embedding
    and norms included. The adapter uses plain Adam plus a decay applied by
    name, and this is what holds that to the pre-registration."""
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    adapter = MlxAdapter(tiny_config(), "cpu", seed=1, peak_lr=1e-3)
    lr = 1e-3
    before = {
        name: np.array(arr) for name, arr in tree_flatten(adapter.model.parameters())
    }
    zero_grads = tree_unflatten(
        [(name, mx.zeros(arr.shape)) for name, arr in tree_flatten(adapter.model.parameters())]
    )
    adapter.opt.learning_rate = lr
    adapter._apply_weight_decay(lr)
    adapter.opt.update(adapter.model, zero_grads)
    mx.eval(adapter.model.parameters(), adapter.opt.state)

    after = {
        name: np.array(arr) for name, arr in tree_flatten(adapter.model.parameters())
    }
    for name in before:
        if should_decay(name):
            expected = before[name] * (1.0 - lr * WEIGHT_DECAY)
            assert np.allclose(after[name], expected, rtol=1e-5, atol=1e-8), (
                f"{name} was not decayed as expected"
            )
        else:
            assert np.array_equal(after[name], before[name]), (
                f"{name} moved under zero gradients, so it was decayed"
            )


# ----------------------------------------------------------------------------
# the anchor: does the loss actually go down
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_loss_decreases_on_a_learnable_task(framework: str) -> None:
    """Train the real step on a task that cannot fail to be learnable, and
    require the loss to fall a long way.

    A repeated cycle of cycle tokens: every token determines the next one, so a
    working optimizer drives this from ln(vocab) toward zero within a few dozen
    steps. An optimizer that is not updating, a learning rate wired to nothing,
    or a gradient that never reaches the parameters all leave the loss flat here
    while still passing any test that just checks a step returns a number.
    """
    cfg = tiny_config()
    adapter = make_adapter(framework, cfg, "cpu", seed=1, peak_lr=3e-3)
    x, y = cycle_batch(8, cfg.context_len, seed=0)

    losses = [adapter.train_step(x, y, 3e-3)[0] for _ in range(60)]

    first = float(np.mean(losses[:3]))
    last = float(np.mean(losses[-3:]))
    assert np.isfinite(losses).all(), f"{framework} produced a non-finite loss"
    assert first > 8.0, (
        f"{framework} started at {first:.3f}, expected about ln(16384) = 9.7; "
        "the task or the init is not what this test assumes"
    )
    assert last < first - 4.0, (
        f"{framework} loss went {first:.3f} -> {last:.3f}, a fall of only "
        f"{first - last:.3f}. The optimizer is not learning a task that is "
        "trivially learnable."
    )


def test_the_two_adapters_implement_the_same_optimizer() -> None:
    """From identical weights and identical batches, both frameworks must walk
    the same trajectory.

    This is the test that protects the thing the whole lab rests on. The
    framework is chosen by a throughput pilot, and that choice is only allowed
    to be about speed: the pre-registration fixes one optimizer for every run,
    so torch and MLX have to be doing the same arithmetic or the size sweep is
    comparing two different experiments.

    It is not hypothetical. mlx.optimizers.Adam defaults to
    bias_correction=False while torch's AdamW always corrects. Left at the
    default, the same settings give a max loss difference of about 0.47 over
    these 25 steps. With the correction on, the two agree to about 3e-6, which
    is the float32 noise floor the twins already show on a single forward pass
    in tests/test_equivalence.py. The tolerance below sits between those two
    numbers on purpose.
    """
    cfg = tiny_config()
    torch_adapter = make_adapter("torch", cfg, "cpu", seed=1, peak_lr=3e-3)
    mlx_adapter = make_adapter("mlx", cfg, "cpu", seed=99, peak_lr=3e-3)
    # Different seeds, then the same weights, so the test would fail if
    # load_weights quietly did nothing.
    mlx_adapter.load_weights(torch_adapter.get_weights_for_checkpoint())

    torch_losses, mlx_losses = [], []
    for i in range(25):
        x, y = cycle_batch(8, cfg.context_len, seed=i)
        torch_losses.append(torch_adapter.train_step(x, y, 3e-3)[0])
        mlx_losses.append(mlx_adapter.train_step(x, y, 3e-3)[0])

    diff = np.max(np.abs(np.array(torch_losses) - np.array(mlx_losses)))
    assert diff < 1e-4, (
        f"torch and mlx diverged by {diff:.3e} in loss from the same start on "
        "the same data. The two adapters are not running the same optimizer."
    )
    # And both actually trained, so agreement is not two flat lines matching.
    assert torch_losses[-1] < torch_losses[0] - 4.0
    assert mlx_losses[-1] < mlx_losses[0] - 4.0


# ----------------------------------------------------------------------------
# gradient accumulation
# ----------------------------------------------------------------------------

# Measured, not guessed. On CPU in float32 the accumulated path and the
# un-accumulated path are near-exact but not bit-identical, and the cause is
# float associativity in the reductions, established directly rather than
# assumed:
#
#   At step 0 the weights are identical across every micro_batch_size, so any
#   difference is reduction order alone. There the returned gradient norm is
#   bit-identical at every setting in both frameworks, and the returned loss
#   differs by strictly under one float32 ulp (0.5 to 0.75 ulp; micro=4 is
#   bit-exact). The full-batch mean over N tokens and the mean of k micro-batch
#   means over N/k tokens each are the same number in exact arithmetic and
#   differ only in the order the partial sums are rounded.
#
#   The residue is non-monotone in accum_steps: micro=4 is sometimes closer than
#   micro=2 and micro=1 sometimes further than both. A systematic scale error
#   would grow with accum_steps. Rounding does not.
#
# Over 12 steps that sub-ulp noise compounds through Adam to at most 2.2e-6 in
# any weight and 2.1e-6 in any loss, across both frameworks and every setting.
# ACCUM_TOL sits about 5x above the worst measured value, which leaves the real
# bugs far outside it: per-micro-batch clipping moves the weights by 6e-3 to
# 1.7e-2 and a dropped loss scaling moves the reported loss by 9.7 to 29.1,
# which are 3 and 6 orders of magnitude beyond this tolerance.
ACCUM_TOL = 1e-5


def accumulation_trajectory(framework: str, micro: int, steps: int = 12):
    """Run steps optimizer steps at a fixed seed and micro_batch_size.

    Same seed, same batches, same order, so the only variable is how the batch
    is split for the backward.
    """
    cfg = tiny_config()
    adapter = make_adapter(
        framework, cfg, "cpu", seed=1, peak_lr=3e-3, micro_batch_size=micro
    )
    losses = []
    for i in range(steps):
        x, y = cycle_batch(8, cfg.context_len, seed=i)
        losses.append(adapter.train_step(x, y, 3e-3)[0])
    return np.array(losses), adapter.get_weights_for_checkpoint()


def max_weight_diff(a: dict, b: dict) -> float:
    assert set(a) == set(b)
    return max(float(np.max(np.abs(a[k] - b[k]))) for k in a)


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("micro", [4, 2])
def test_accumulation_reproduces_the_full_batch_step(
    framework: str, micro: int
) -> None:
    """The anchor. Accumulation must be the same optimization, not a nearby one.

    The pre-registration fixes 16,384 tokens per step and requires the optimizer
    to see the same batch regardless, so micro_batch_size is only allowed to
    change memory. Batch 8 split into micro-batches of 8, 4, and 2 must walk the
    same trajectory and land on the same weights.

    Both halves are load-bearing and they catch different bugs:

      The loss trajectory is what catches a wrong per-micro-batch loss scale, the
        classic bug here. It catches it by six orders of magnitude.
      The weights are what catch a wrong clipping placement, and per-micro-batch
        clipping moves them 3 orders of magnitude beyond ACCUM_TOL.

    The weights alone would NOT catch a dropped 1/accum_steps scaling, and it is
    a mistake to assume they would. Two things hide it. Global-norm clipping is
    engaged on every step of this task, and it rescales the gradient to a fixed
    norm, which cancels a constant factor exactly. Adam then divides by
    sqrt(v_hat), which is invariant to a global gradient scale except through
    eps. Measured: the unscaled mutant leaves max|dW| at 7.7e-7, indistinguishable
    from the correct implementation, while its reported loss is off by 9.7.
    test_accumulation_matches_the_full_batch_gradient_without_clipping covers the
    late-training regime where clipping is inactive and the scale does reach the
    weights.
    """
    want_losses, want_weights = accumulation_trajectory(framework, micro=8)
    got_losses, got_weights = accumulation_trajectory(framework, micro=micro)

    assert np.isfinite(want_losses).all() and np.isfinite(got_losses).all()
    # Not two flat lines agreeing: the task has to have actually been learned,
    # or this test passes on a broken optimizer that never updates anything.
    assert want_losses[-1] < want_losses[0] - 3.0

    loss_diff = float(np.max(np.abs(got_losses - want_losses)))
    assert loss_diff < ACCUM_TOL, (
        f"{framework} micro-batch {micro}: loss trajectory diverged from the "
        f"un-accumulated path by {loss_diff:.3e}, over ACCUM_TOL {ACCUM_TOL:.0e}. "
        "That is far past float associativity, so the per-micro-batch loss is "
        "being scaled wrong and the effective learning rate has moved with "
        "accum_steps."
    )
    weight_diff = max_weight_diff(got_weights, want_weights)
    assert weight_diff < ACCUM_TOL, (
        f"{framework} micro-batch {micro}: final weights differ from the "
        f"un-accumulated path by {weight_diff:.3e}, over ACCUM_TOL "
        f"{ACCUM_TOL:.0e}. Accumulation changed the optimization, which it is "
        "not allowed to do."
    )


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_accumulation_matches_the_full_batch_gradient_without_clipping(
    framework: str, monkeypatch
) -> None:
    """The same equivalence with the clip lifted out of the way.

    Clipping normalizes the gradient to a fixed norm, so with it engaged a
    constant factor on the accumulated gradient cancels and the weights cannot
    see a dropped loss scaling. Late in a real run the gradient norm falls below
    the threshold, clipping stops firing, and that factor reaches the weights
    directly. Raising GRAD_CLIP_NORM out of reach puts this test in that regime
    on a short run, so the gradient scale is actually being checked here rather
    than being hidden by the clip.

    Measured: correct implementation 8e-7 to 9e-7, the unscaled mutant 1.0e-3.
    """
    monkeypatch.setattr("small_lm_lab.train.GRAD_CLIP_NORM", 1e9)
    want_losses, want_weights = accumulation_trajectory(framework, micro=8)
    got_losses, got_weights = accumulation_trajectory(framework, micro=2)

    weight_diff = max_weight_diff(got_weights, want_weights)
    loss_diff = float(np.max(np.abs(got_losses - want_losses)))
    assert loss_diff < ACCUM_TOL, f"{framework}: loss diverged by {loss_diff:.3e}"
    assert weight_diff < ACCUM_TOL, (
        f"{framework}: with clipping inactive the accumulated gradient moved the "
        f"weights {weight_diff:.3e} away from the full-batch gradient, over "
        f"ACCUM_TOL {ACCUM_TOL:.0e}. The per-micro-batch loss scaling is wrong."
    )


def torch_per_micro_clip_step(adapter, x, y, lr: float, micro: int) -> None:
    """The wrong algorithm, written out: clip every micro-batch, then accumulate.

    Kept here rather than in train.py because its only purpose is to be
    different from what train.py does.
    """
    accum_steps = x.shape[0] // micro
    for group in adapter.opt.param_groups:
        group["lr"] = lr
    adapter.model.train()
    accum = {n: torch.zeros_like(p) for n, p in adapter.model.named_parameters()}
    for i in range(accum_steps):
        adapter.opt.zero_grad(set_to_none=True)
        xt = torch.from_numpy(x[i * micro : (i + 1) * micro])
        yt = torch.from_numpy(y[i * micro : (i + 1) * micro])
        (adapter._loss(xt, yt) / accum_steps).backward()
        torch.nn.utils.clip_grad_norm_(adapter.model.parameters(), GRAD_CLIP_NORM)
        for n, p in adapter.model.named_parameters():
            accum[n] += p.grad
    adapter.opt.zero_grad(set_to_none=True)
    for n, p in adapter.model.named_parameters():
        p.grad = accum[n]
    adapter.opt.step()


def mlx_per_micro_clip_step(adapter, x, y, lr: float, micro: int) -> None:
    """The same wrong algorithm in MLX."""
    mx = adapter.mx
    accum_steps = x.shape[0] // micro
    accum = None
    for i in range(accum_steps):
        xa = mx.array(x[i * micro : (i + 1) * micro])
        ya = mx.array(y[i * micro : (i + 1) * micro])
        _, grads = adapter._loss_and_grad(adapter.model, xa, ya, 1.0 / accum_steps)
        grads, _ = adapter.mlx_optim.clip_grad_norm(grads, GRAD_CLIP_NORM)
        accum = (
            grads
            if accum is None
            else adapter._tree_map(lambda a, g: a + g, accum, grads)
        )
        mx.eval(accum)
    adapter.opt.learning_rate = lr
    adapter._apply_weight_decay(lr)
    adapter.opt.update(adapter.model, accum)
    mx.eval(adapter.model.parameters(), adapter.opt.state)


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_clip_is_applied_once_to_the_accumulated_gradient(framework: str) -> None:
    """Clipping per micro-batch is a different algorithm, and is not what runs here.

    The construction: at init the gradient norm is about 4.4, well over the clip
    threshold of 1.0, so clipping actually engages, and the individual
    micro-batch gradients have different norms from each other. Clipping each
    micro-batch rescales each piece by its own norm before summing, which points
    the accumulated update somewhere the clipped full-batch gradient does not.
    If the norms were equal, or if clipping never fired, the two would coincide
    and this test could not tell them apart.

    Three single steps from identical weights on the identical batch: the
    un-accumulated path (which clips the full gradient once, and is the
    reference), the implementation at micro-batch 4, and the per-micro-batch
    clip. The implementation must sit on the reference and far off the mutant.
    """
    cfg = tiny_config()
    x, y = cycle_batch(8, cfg.context_len, seed=0)
    lr = 3e-3

    reference = make_adapter(framework, cfg, "cpu", seed=1, peak_lr=lr)
    _, full_norm = reference.train_step(x, y, lr)
    want = reference.get_weights_for_checkpoint()

    accumulated = make_adapter(
        framework, cfg, "cpu", seed=1, peak_lr=lr, micro_batch_size=4
    )
    _, accum_norm = accumulated.train_step(x, y, lr)
    got = accumulated.get_weights_for_checkpoint()

    per_micro = make_adapter(
        framework, cfg, "cpu", seed=1, peak_lr=lr, micro_batch_size=4
    )
    step = torch_per_micro_clip_step if framework == "torch" else mlx_per_micro_clip_step
    step(per_micro, x, y, lr, micro=4)
    mutant = per_micro.get_weights_for_checkpoint()

    # The premise: clipping has to be engaging, or the two algorithms agree
    # trivially and this proves nothing.
    assert full_norm > GRAD_CLIP_NORM, (
        f"gradient norm {full_norm:.3f} is under the clip threshold "
        f"{GRAD_CLIP_NORM}, so this test is not exercising clipping at all"
    )
    # The reported norm is the pre-clip norm of the accumulated gradient, so it
    # must be the full-batch norm, not a micro-batch's.
    assert accum_norm == pytest.approx(full_norm, abs=1e-5), (
        f"{framework}: micro-batch 4 reported a pre-clip norm of {accum_norm:.6f} "
        f"but the full batch is {full_norm:.6f}. The norm is not being taken on "
        "the accumulated gradient."
    )

    matches_accumulated = max_weight_diff(got, want)
    matches_per_micro = max_weight_diff(mutant, want)
    assert matches_accumulated < ACCUM_TOL, (
        f"{framework}: the implementation is {matches_accumulated:.3e} from the "
        "clip-once-on-the-accumulation reference"
    )
    assert matches_per_micro > 100 * ACCUM_TOL, (
        f"{framework}: clipping per micro-batch only moved the weights "
        f"{matches_per_micro:.3e}, so this test cannot distinguish the two "
        "algorithms and is not proving anything. Fix the construction."
    )
    assert matches_accumulated < matches_per_micro / 100, (
        f"{framework}: the implementation is {matches_accumulated:.3e} from the "
        f"accumulated clip and {matches_per_micro:.3e} from the per-micro-batch "
        "clip. It does not match the accumulated one."
    )


def test_accumulation_does_not_change_the_token_order() -> None:
    """A run at micro-batch 16 must see the same tokens in the same order as the
    same run at micro-batch 32, or the two are not comparable and a resume across
    a config change silently reads a different stream.

    The batcher is constructed at batch_size and the adapter splits the batch it
    is handed, so this is true by construction. Asserted anyway, because the
    tempting alternative (asking the batcher for micro_batch_size rows
    accum_steps times) would reorder the stream while still training correctly,
    and nothing else in the suite would notice.
    """
    from small_lm_lab.data import MixtureBatcher

    full = MixtureBatcher(cycle_sources(), 8, 64, seed=1)
    batches = [full.next_batch() for _ in range(4)]

    # The rows the adapter sees at micro-batch 2, concatenated back, must be the
    # rows the un-accumulated path saw, in the same order.
    again = MixtureBatcher(cycle_sources(), 8, 64, seed=1)
    for i, (want_x, want_y) in enumerate(batches):
        got_x, got_y = again.next_batch()
        assert np.array_equal(got_x, want_x), f"batch {i} x differs"
        assert np.array_equal(got_y, want_y), f"batch {i} y differs"
        micro_rows_x = np.concatenate([got_x[j : j + 2] for j in range(0, 8, 2)])
        assert np.array_equal(micro_rows_x, want_x), (
            f"batch {i}: splitting into micro-batches of 2 and rejoining did not "
            "reproduce the full batch"
        )


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_adapter_refuses_a_micro_batch_that_does_not_divide(framework: str) -> None:
    cfg = tiny_config()
    adapter = make_adapter(framework, cfg, "cpu", seed=1, peak_lr=1e-3, micro_batch_size=3)
    x, y = cycle_batch(8, cfg.context_len, seed=0)
    with pytest.raises(ValueError, match="does not divide"):
        adapter.train_step(x, y, 1e-3)


def test_resolve_accumulation() -> None:
    assert resolve_accumulation(32, None) == (32, 1)
    assert resolve_accumulation(32, 32) == (32, 1)
    assert resolve_accumulation(32, 16) == (16, 2)
    assert resolve_accumulation(32, 8) == (8, 4)
    assert resolve_accumulation(32, 1) == (1, 32)
    with pytest.raises(ValueError, match="does not divide"):
        resolve_accumulation(32, 5)
    with pytest.raises(ValueError, match="does not divide"):
        resolve_accumulation(32, 64)
    with pytest.raises(ValueError, match="at least 1"):
        resolve_accumulation(32, 0)


def test_config_refuses_a_micro_batch_that_does_not_divide_the_batch() -> None:
    """Refused at construction, before a run starts, rather than at the first
    step after the manifest is already on disk."""
    with pytest.raises(ValueError, match="does not divide batch_size"):
        TrainConfig(run_name="x", size="size30m", batch_size=32, micro_batch_size=5)
    with pytest.raises(ValueError, match="does not divide batch_size"):
        TrainConfig(run_name="x", size="size30m", batch_size=32, micro_batch_size=7)
    # Larger than the batch is also not a split.
    with pytest.raises(ValueError, match="does not divide batch_size"):
        TrainConfig(run_name="x", size="size30m", batch_size=32, micro_batch_size=64)
    with pytest.raises(ValueError, match="at least 1"):
        TrainConfig(run_name="x", size="size30m", batch_size=32, micro_batch_size=0)
    with pytest.raises(ValueError, match="at least 1"):
        TrainConfig(run_name="x", size="size30m", batch_size=32, micro_batch_size=-4)

    # The message has to say what would work, not just that this did not.
    with pytest.raises(ValueError, match=r"Valid micro_batch_size values"):
        TrainConfig(run_name="x", size="size30m", batch_size=32, micro_batch_size=5)


def test_micro_batch_never_changes_tokens_per_step() -> None:
    """The pre-registration fixes 32 x 512 at every size and says the optimizer
    must see the same batch regardless. micro_batch_size is a memory setting, so
    it must not touch the token budget arithmetic anywhere."""
    for micro in (None, 32, 16, 8, 4, 2, 1):
        cfg = TrainConfig(
            run_name="x", size="size120m", batch_size=32, micro_batch_size=micro
        )
        assert cfg.tokens_per_step == PREREG_TOKENS_PER_STEP
        assert cfg.tokens_per_step == 32 * 512
        expected_micro = 32 if micro is None else micro
        assert cfg.resolved_micro_batch_size == expected_micro
        assert cfg.accum_steps == 32 // expected_micro
        # Every micro-batch carries the same number of tokens, and they add up.
        assert cfg.accum_steps * cfg.resolved_micro_batch_size == cfg.batch_size

    # The pre-registered size120m setting, spelled out.
    cfg = TrainConfig(run_name="x", size="size120m", micro_batch_size=16)
    assert cfg.accum_steps == 2
    assert cfg.tokens_per_step == 16384

    # And the default is no accumulation.
    default = TrainConfig(run_name="x", size="size120m")
    assert default.micro_batch_size is None
    assert default.resolved_micro_batch_size == 32
    assert default.accum_steps == 1


def test_train_records_the_accumulation_in_the_manifest(tmp_path: Path) -> None:
    """A run has to say how it was split, or the memory result is not reproducible
    from the artifacts."""
    steps, batch_size, context_len = 4, 8, tiny_config().context_len
    cfg = tiny_train_config(
        tmp_path,
        "torch",
        run_name="accum_manifest",
        total_tokens=steps * batch_size * context_len,
        batch_size=batch_size,
        micro_batch_size=2,
    )
    train(cfg, sources=cycle_sources())
    manifest = json.loads((Path(cfg.run_dir) / "manifest.json").read_text())
    assert manifest["batch_size"] == 8
    assert manifest["micro_batch_size"] == 2
    assert manifest["accum_steps"] == 4
    assert manifest["tokens_per_step"] == batch_size * context_len
    assert manifest["train_config"]["micro_batch_size"] == 2


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_train_end_to_end_matches_across_micro_batch(framework: str, tmp_path: Path) -> None:
    """The equivalence through the real loop, not just the adapter: same logged
    loss trajectory whether or not the step is split."""
    steps, batch_size, context_len = 6, 8, tiny_config().context_len
    total_tokens = steps * batch_size * context_len

    full = tiny_train_config(
        tmp_path,
        framework,
        run_name=f"full_{framework}",
        total_tokens=total_tokens,
        batch_size=batch_size,
    )
    train(full, sources=cycle_sources())
    want = read_losses(Path(full.run_dir) / "train_log.jsonl")

    split = tiny_train_config(
        tmp_path,
        framework,
        run_name=f"split_{framework}",
        total_tokens=total_tokens,
        batch_size=batch_size,
        micro_batch_size=2,
    )
    train(split, sources=cycle_sources())
    got = read_losses(Path(split.run_dir) / "train_log.jsonl")

    assert [s for s, _ in got] == [s for s, _ in want]
    for (step, want_loss), (_, got_loss) in zip(want, got):
        assert got_loss == pytest.approx(want_loss, abs=ACCUM_TOL), (
            f"{framework} step {step}: micro-batch 2 logged {got_loss!r} but the "
            f"un-accumulated run logged {want_loss!r}"
        )


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_train_step_reports_a_real_gradient_norm(framework: str) -> None:
    cfg = tiny_config()
    adapter = make_adapter(framework, cfg, "cpu", seed=1, peak_lr=1e-3)
    x, y = cycle_batch(4, cfg.context_len, seed=0)
    _, grad_norm = adapter.train_step(x, y, 1e-3)
    # Pre-clip norm, as torch's clip_grad_norm_ returns, so at init it is free
    # to exceed the clip threshold. It must be a real finite positive number.
    assert np.isfinite(grad_norm)
    assert grad_norm > 0.0


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_param_count_matches_the_config(framework: str) -> None:
    cfg = tiny_config()
    adapter = make_adapter(framework, cfg, "cpu", seed=1, peak_lr=1e-3)
    assert adapter.param_count() == n_params(cfg)


def test_adapters_honor_a_cpu_device() -> None:
    """device="cpu" has to mean CPU in both frameworks, which is what keeps a
    smoke test off a GPU another run may be holding."""
    import mlx.core as mx

    torch_adapter = TorchAdapter(tiny_config(), "cpu", seed=1, peak_lr=1e-3)
    assert torch_adapter.device.type == "cpu"
    assert all(p.device.type == "cpu" for p in torch_adapter.model.parameters())
    assert torch_adapter.autocast_dtype is None, "no autocast on CPU"

    MlxAdapter(tiny_config(), "cpu", seed=1, peak_lr=1e-3)
    assert mx.default_device() == mx.Device(mx.cpu)


def test_mlx_adapter_rejects_an_unknown_device() -> None:
    with pytest.raises(ValueError, match="cpu or gpu"):
        MlxAdapter(tiny_config(), "cuda", seed=1, peak_lr=1e-3)


def test_torch_precision_policy() -> None:
    """The registered precision is float32, and the CUDA staged run holds it with
    torch_precision="fp32". This pins the policy without touching a GPU, since it
    is a pure function of (device_type, precision):

      float32 (None) on CPU whatever the precision, so the CPU tests stay float32;
      float32 on any device for "fp32", so cuda/fp32 is the registered regime;
      bf16 autocast on an accelerator for "auto" and "bf16", the prior behavior.
    """
    import torch

    # CPU is always plain float32, whatever precision is asked for.
    for precision in ("auto", "fp32", "bf16"):
        assert resolve_torch_autocast_dtype("cpu", precision) is None

    # fp32 forces float32 on an accelerator too: the registered precision.
    assert resolve_torch_autocast_dtype("cuda", "fp32") is None
    assert resolve_torch_autocast_dtype("mps", "fp32") is None

    # auto and bf16 keep bf16 autocast on an accelerator, the prior behavior.
    assert resolve_torch_autocast_dtype("cuda", "auto") is torch.bfloat16
    assert resolve_torch_autocast_dtype("cuda", "bf16") is torch.bfloat16
    assert resolve_torch_autocast_dtype("mps", "auto") is torch.bfloat16

    with pytest.raises(ValueError, match="unknown torch_precision"):
        resolve_torch_autocast_dtype("cuda", "float32")


def test_train_config_rejects_unknown_torch_precision() -> None:
    with pytest.raises(ValueError, match="unknown torch_precision"):
        TrainConfig(run_name="bad", torch_precision="float32")


def test_disable_tf32_actually_turns_tf32_off() -> None:
    """The registered float32 needs TF32 refused, not just bf16 autocast refused.

    Suppressing autocast leaves tensors float32 while an Ampere matmul may still
    run them in TF32, which keeps the dtype and loses 13 mantissa bits. That is
    invisible to every other assertion in this suite, so it gets its own.
    Runs on CPU: these are global framework flags, not device work.
    """
    import torch

    # Flip them the wrong way first so a no-op implementation cannot pass.
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:  # pragma: no cover - version dependent
        pass

    applied = disable_tf32()

    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.get_float32_matmul_precision() == "highest"
    # The report is what lands in the run record, so it must describe what happened.
    assert applied["cuda.matmul.allow_tf32"] is False
    assert applied["cudnn.allow_tf32"] is False
    assert applied["float32_matmul_precision"] == "highest"


def test_disable_tf32_is_idempotent() -> None:
    """Calling it twice must not drift: every adapter construction calls it."""
    first = disable_tf32()
    second = disable_tf32()
    assert first == second


# ----------------------------------------------------------------------------
# checkpoint round trip
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_checkpoint_round_trips_into_a_fresh_model(framework: str, tmp_path: Path) -> None:
    """Save, load with the real eval loader, and compare logits.

    Two assertions, because they catch different things. The loaded weights must
    equal the bf16 rounding of what was saved, exactly: that is the plumbing,
    and it fails on a wrong name, a missing tied head, or a transposed matrix.
    The loaded logits must then match the original model's within a bf16
    tolerance, which is the claim the eval harness relies on: a checkpoint is
    the model.
    """
    from small_lm_lab.model_torch import TransformerLM

    cfg = tiny_config()
    adapter = make_adapter(framework, cfg, "cpu", seed=1, peak_lr=3e-3)
    x, y = cycle_batch(4, cfg.context_len, seed=0)
    # Train a little first: at init the weights are a symmetric random draw and
    # a bug that dropped or swapped one could hide.
    for _ in range(5):
        adapter.train_step(x, y, 3e-3)

    weights = adapter.get_weights_for_checkpoint()
    tokens = 1_234_567
    path = save_checkpoint(
        tmp_path, weights, tokens=tokens, step=4, target_tokens=1_000_000, meta={}
    )
    assert path == checkpoint_path(tmp_path, tokens)

    eval_script = load_eval_script()
    loaded, read_tokens = eval_script.load_model(cfg, path, "cpu")
    assert read_tokens == tokens, "the eval loader could not read the token count"

    # The checkpoint holds exactly bf16(weights), so a fresh model loaded from
    # it must hold exactly that, key for key.
    loaded_state = loaded.state_dict()
    assert set(loaded_state) == set(weights), "checkpoint keys do not match the model"
    for name, array in weights.items():
        expected = torch.from_numpy(array).to(torch.bfloat16).to(torch.float32)
        assert torch.equal(loaded_state[name].cpu(), expected), f"{name} did not round trip"

    # And the logits must agree with the original model within bf16 rounding.
    probe = torch.from_numpy(cycle_batch(2, cfg.context_len, seed=7)[0])
    reference = TransformerLM(cfg)
    reference.load_state_dict(
        {k: torch.from_numpy(v) for k, v in weights.items()}, strict=True
    )
    reference.eval()
    with torch.no_grad():
        want = reference(probe)
        got = loaded(probe)
    # bf16 keeps 8 mantissa bits, so every weight carries up to about 0.4
    # percent relative error into the forward pass. The tolerance is scaled to
    # the logits rather than absolute, since logit magnitude grows with training.
    scale = float(want.abs().max())
    assert float((got - want).abs().max()) < 0.05 * scale, (
        "logits moved more than bf16 rounding can explain"
    )


def test_mlx_checkpoint_carries_the_tied_head(tmp_path: Path) -> None:
    """The MLX model has no lm_head parameter, since the head is embed.as_linear.
    A torch state_dict lists the tie explicitly and the eval loader is
    strict=True, so the adapter has to emit it or every MLX checkpoint would be
    unloadable."""
    cfg = tiny_config()
    assert cfg.tie_embeddings
    adapter = MlxAdapter(cfg, "cpu", seed=1, peak_lr=1e-3)
    weights = adapter.get_weights_for_checkpoint()
    assert "lm_head.weight" in weights
    assert np.array_equal(weights["lm_head.weight"], weights["embed.weight"])


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_weights_survive_a_load_back_into_the_adapter(framework: str) -> None:
    """load_weights is the inverse of get_weights_for_checkpoint, which is what
    resume leans on."""
    cfg = tiny_config()
    a = make_adapter(framework, cfg, "cpu", seed=1, peak_lr=3e-3)
    x, y = cycle_batch(4, cfg.context_len, seed=0)
    for _ in range(3):
        a.train_step(x, y, 3e-3)
    weights = a.get_weights_for_checkpoint()

    b = make_adapter(framework, cfg, "cpu", seed=2, peak_lr=3e-3)
    b.load_weights(weights)
    after = b.get_weights_for_checkpoint()
    assert set(after) == set(weights)
    for name in weights:
        assert np.array_equal(after[name], weights[name]), f"{name} changed on reload"


# ----------------------------------------------------------------------------
# resume
# ----------------------------------------------------------------------------

def read_losses(log_path: Path) -> list[tuple[int, float]]:
    out = []
    with log_path.open() as f:
        for line in f:
            record = json.loads(line)
            out.append((record["step"], record["loss"]))
    return out


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_resume_reproduces_the_uninterrupted_trajectory(
    framework: str, tmp_path: Path
) -> None:
    """N steps, save, resume, continue must match N+M steps straight through.

    This is exact rather than approximate, and deliberately so. Everything the
    trajectory depends on is restored exactly: float32 weights (the resume file
    is float32 for this reason, while the pre-registered checkpoints are bf16),
    the full optimizer state including Adam's step counter that drives bias
    correction, and the data stream position via the batcher's bit generator
    state. The learning rate schedule is keyed to total_tokens, which both runs
    share, so a resumed step sees the rate the uninterrupted run would have used
    at that step. On CPU in float32 both frameworks are deterministic, so any
    difference at all would mean something was not restored. If this ever needs
    a tolerance, that is a finding, not a nuisance.
    """
    total_steps = 12
    interrupt_at = 5
    sources = cycle_sources()
    batch_size, context_len = 4, tiny_config().context_len
    total_tokens = total_steps * batch_size * context_len

    straight = tiny_train_config(
        tmp_path,
        framework,
        run_name=f"straight_{framework}",
        total_tokens=total_tokens,
        batch_size=batch_size,
    )
    train(straight, sources=cycle_sources())
    want = read_losses(Path(straight.run_dir) / "train_log.jsonl")
    assert len(want) == total_steps

    first_half = tiny_train_config(
        tmp_path,
        framework,
        run_name=f"resumed_{framework}",
        total_tokens=total_tokens,
        batch_size=batch_size,
        stop_after_steps=interrupt_at,
        resume_state_every=interrupt_at,
    )
    train(first_half, sources=cycle_sources())

    second_half = tiny_train_config(
        tmp_path,
        framework,
        run_name=f"resumed_{framework}",
        total_tokens=total_tokens,
        batch_size=batch_size,
    )
    train(second_half, sources=cycle_sources(), resume=True)

    got = read_losses(Path(second_half.run_dir) / "train_log.jsonl")
    assert len(got) == total_steps, "the resumed log lost or duplicated steps"
    assert [s for s, _ in got] == [s for s, _ in want]
    for (step, want_loss), (_, got_loss) in zip(want, got):
        assert got_loss == want_loss, (
            f"{framework} step {step}: resumed loss {got_loss!r} != "
            f"uninterrupted {want_loss!r}. Something was not restored."
        )


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_resume_restores_the_data_stream_position(framework: str, tmp_path: Path) -> None:
    """A resume that rewound or advanced the stream would show a different token
    order. Checked directly, since the loss test above would also fail for other
    reasons."""
    from small_lm_lab.data import MixtureBatcher
    from small_lm_lab.train import batcher_stream_state, restore_batcher_stream

    sources = cycle_sources()
    a = MixtureBatcher(sources, 4, 64, seed=1)
    for _ in range(5):
        a.next_batch()
    state = batcher_stream_state(a)
    want = [a.next_batch()[0] for _ in range(3)]

    b = MixtureBatcher(cycle_sources(), 4, 64, seed=1)
    restore_batcher_stream(b, state)
    got = [b.next_batch()[0] for _ in range(3)]
    for i, (w, g) in enumerate(zip(want, got)):
        assert np.array_equal(w, g), f"batch {i} after restore differs"


def test_resume_refuses_to_cross_frameworks(tmp_path: Path) -> None:
    """Optimizer state does not transfer between frameworks, so a resume that
    silently accepted it would restart Adam's moments while claiming not to."""
    sources = cycle_sources()
    cfg = tiny_train_config(
        tmp_path,
        "torch",
        run_name="crossed",
        total_tokens=4 * 4 * 64,
        batch_size=4,
        resume_state_every=1,
    )
    train(cfg, sources=sources)

    other = tiny_train_config(
        tmp_path,
        "mlx",
        run_name="crossed",
        total_tokens=8 * 4 * 64,
        batch_size=4,
    )
    with pytest.raises(ValueError, match="framework"):
        train(other, sources=cycle_sources(), resume=True)


def test_resume_without_a_state_file_is_an_error(tmp_path: Path) -> None:
    cfg = tiny_train_config(
        tmp_path, "torch", run_name="nothing_here", total_tokens=4 * 4 * 64, batch_size=4
    )
    with pytest.raises(FileNotFoundError):
        train(cfg, sources=cycle_sources(), resume=True)


# ----------------------------------------------------------------------------
# the loop end to end
# ----------------------------------------------------------------------------

def test_train_writes_log_manifest_and_checkpoints(tmp_path: Path) -> None:
    steps, batch_size, context_len = 6, 4, tiny_config().context_len
    tokens_per_step = batch_size * context_len
    cfg = tiny_train_config(
        tmp_path,
        "torch",
        run_name="endtoend",
        total_tokens=steps * tokens_per_step,
        batch_size=batch_size,
        val_every=3,
        val_batches=2,
        checkpoint_token_targets=(2 * tokens_per_step, 4 * tokens_per_step),
    )
    summary = train(cfg, sources=cycle_sources(), val_sources=cycle_sources())

    assert summary["total_steps"] == steps
    assert summary["tokens"] == steps * tokens_per_step

    # Three checkpoints: the two targets plus the one at B.
    written = sorted(Path(cfg.checkpoint_dir).glob("tokens_*.pt"))
    assert len(written) == 3, [p.name for p in written]
    assert (Path(cfg.checkpoint_dir) / "resume_state.pt").exists()

    # The pre-registration keeps optimizer state for the most recent checkpoint
    # only, so no checkpoint may carry it.
    for path in written:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        assert "optimizer" not in obj, f"{path.name} carries optimizer state"
        assert obj["dtype"] == "bfloat16"
        assert all(t.dtype == torch.bfloat16 for t in obj["model"].values())

    records = [json.loads(l) for l in (Path(cfg.run_dir) / "train_log.jsonl").read_text().splitlines()]
    assert len(records) == steps
    for key in ("step", "tokens", "loss", "lr", "grad_norm", "tokens_per_sec", "elapsed", "peak_mem_gb"):
        assert key in records[0], f"the log is missing {key}"
    assert any("val_loss" in r for r in records), "validation never ran"

    manifest = json.loads((Path(cfg.run_dir) / "manifest.json").read_text())
    assert manifest["n_params"] == n_params(tiny_config())
    assert manifest["n_params_adapter"] == manifest["n_params"]
    assert manifest["framework"] == "torch"
    assert manifest["seed"] == 1
    assert manifest["data_mix"] == {"cycle": 1.0}
    assert manifest["tokens_per_step"] == tokens_per_step
    assert manifest["optimizer"]["beta1"] == 0.9
    assert manifest["optimizer"]["beta2"] == 0.95
    assert manifest["optimizer"]["eps"] == 1e-8
    assert manifest["optimizer"]["weight_decay"] == WEIGHT_DECAY
    assert manifest["optimizer"]["grad_clip_norm"] == GRAD_CLIP_NORM
    assert manifest["tokenizer_sha256"] is not None
    assert "commit" in manifest["git"]
    assert manifest["machine"]["platform"]


def test_default_config_matches_the_preregistered_tokens_per_step() -> None:
    cfg = TrainConfig(run_name="x", size="size30m")
    assert cfg.batch_size == 32
    assert cfg.resolved_model_config().context_len == 512
    assert cfg.tokens_per_step == 16384
    assert cfg.seed == 1
    assert cfg.mix == {"tinystories": 0.70, "fineweb_edu": 0.30}
    assert cfg.warmup_fraction == 0.02


def test_train_stops_on_a_non_finite_loss(tmp_path: Path, monkeypatch) -> None:
    """A diverged run must stop rather than write a checkpoint of a dead model."""
    cfg = tiny_train_config(
        tmp_path, "torch", run_name="diverged", total_tokens=8 * 4 * 64, batch_size=4
    )
    monkeypatch.setattr(
        TorchAdapter, "train_step", lambda self, x, y, lr: (float("nan"), 1.0)
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        train(cfg, sources=cycle_sources())
    assert not list(Path(cfg.checkpoint_dir).glob("tokens_*.pt"))
