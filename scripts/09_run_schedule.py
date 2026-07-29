"""Run the next pending step of the pre-registered training schedule.

Invoked by the caller each time it acquires the shared training lock. Runs
exactly one step, then exits, so the caller can release the lock between steps
and other jobs waiting on the same GPU interleave instead of starving.

The queue is not stored anywhere: it is recomputed from what exists on disk,
which makes the engine idempotent. Kill it at any point, relaunch it, and it
continues where the artifacts say it should. A training run is complete when a
checkpoint at or beyond its token budget exists; an incomplete run with a
resume_state.pt is resumed, not restarted.

The schedule itself is fixed by docs/PROTOCOL.md:

  1. abl_lr6e-4      size30m, lr 6e-4,   40M tokens, warmup 2 percent
  2. abl_lr2.4e-3    size30m, lr 2.4e-3, 40M tokens, warmup 2 percent
     (abl_lr1.2e-3 already completed 2026-07-17)
  3. score the three LR ablations on TinyStories validation perplexity
     (the in-run val_loss is on the 0.7/0.3 mixture, so it cannot be the
     pre-registered score; the winner is argmin TinyStories val perplexity)
  4. abl_warmup0     size30m, winning lr, warmup 0, 40M tokens
  5. main_size30m    winning lr, 200M tokens
  6. main_size60m    winning lr, 200M tokens
  7. main_size120m   winning lr, 200M tokens, micro-batch 16, with the
     pre-registered stability check on the first 2 percent of steps:
     divergence there triggers one halving of the learning rate, recorded,
     and a second divergence stops the whole queue as a hard failure.

Before any step, once per repo code state, the engine runs scripts/smoke_gate.py
on the device the eval path uses. The gate exercises every downstream stage
(eval suites, interp battery) against a tiny checkpoint in minutes. It exists
because the 2026-07-21 MPS float64 bug sat un-executed in the scoring path
through three full training runs and then killed the schedule in two seconds:
a step that has never run on its real device must fail before GPU-hours are
spent, not after.

A deliberate pause can be staged: if analysis/schedule/hold_before.txt names the
next step, the engine exits 30 without running it and the caller parks with a
SCHEDULE_HOLD marker instead of failing. This is for out-of-band measurements,
for example a capacity benchmark, that must land between steps.

Exit codes: 0 = ran one step, more remain. 10 = schedule complete.
20 = hard failure, stop the queue and leave a marker.
30 = holding before the named step, per hold_before.txt.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from small_lm_lab.paths import BULK_ROOT, CHECKPOINT_ROOT, RUN_ROOT

REPO = Path(__file__).resolve().parents[1]
# The scheduler launches training as a subprocess, so it needs an interpreter path
# rather than the current one only when it is itself run outside the project
# environment. SMALL_LM_LAB_PYTHON overrides for a non-standard environment layout.
PYTHON = Path(os.environ.get("SMALL_LM_LAB_PYTHON", REPO / ".venv" / "bin" / "python"))
CKPT_ROOT = CHECKPOINT_ROOT
ABL_DIR = REPO / "analysis" / "ablations"
SCHED_DIR = REPO / "analysis" / "schedule"

ABL_TOKENS = 40e6
MAIN_TOKENS = 200e6
TOKENS_PER_STEP = 16384
LR_ABLATIONS = {
    "abl_lr6e-4": 6e-4,
    "abl_lr1.2e-3": 1.2e-3,
    "abl_lr2.4e-3": 2.4e-3,
}
WINNER_FILE = ABL_DIR / "lr_winner.json"
SMOKE_MARKER = SCHED_DIR / "smoke_ok.json"
HOLD_FILE = SCHED_DIR / "hold_before.txt"

# Stability gate for size120m, per the pre-registration: "the first 2 percent
# of steps" of the real run. Divergence is defined mechanically here, in
# advance of the run: any non-finite loss inside the window, or the mean of
# the last 8 logged losses in the window exceeding the mean of the first 8.
# The trainer logs every 10 steps, so a 244 step window holds about 24 loss
# records; 8 and 8 keeps the two means disjoint. One halving is allowed; a
# second divergence is a hard failure.
GATE_FRACTION = 0.02
GATE_POLL_SECONDS = 60


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def note(msg: str) -> None:
    print(f"{ts()} {msg}", flush=True)
    SCHED_DIR.mkdir(parents=True, exist_ok=True)
    with open(SCHED_DIR / "engine.log", "a") as f:
        f.write(f"{ts()} {msg}\n")


def ckpt_tokens(run_name: str) -> list[int]:
    d = CKPT_ROOT / run_name
    if not d.is_dir():
        return []
    out = []
    for p in d.iterdir():
        m = re.fullmatch(r"tokens_(\d+)\.pt", p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def run_complete(run_name: str, budget: float) -> bool:
    toks = ckpt_tokens(run_name)
    return bool(toks) and toks[-1] >= budget


def final_ckpt(run_name: str) -> Path:
    toks = ckpt_tokens(run_name)
    return CKPT_ROOT / run_name / f"tokens_{toks[-1]:012d}.pt"


def train_cmd(
    run_name: str,
    size: str,
    lr: float,
    tokens: float,
    warmup: float | None = None,
    micro_batch: int | None = None,
) -> list[str]:
    cmd = [
        str(PYTHON),
        str(REPO / "scripts" / "07_train.py"),
        "--size", size,
        "--framework", "mlx",
        "--lr", repr(lr),
        "--tokens", repr(tokens),
        "--seed", "1",
        "--run-name", run_name,
    ]
    if warmup is not None:
        cmd += ["--warmup-fraction", repr(warmup)]
    if micro_batch is not None:
        cmd += ["--micro-batch", str(micro_batch)]
    if (CKPT_ROOT / run_name / "resume_state.pt").exists():
        cmd += ["--resume"]
    return cmd


def run_training(cmd: list[str], run_name: str) -> int:
    note(f"step start: {' '.join(cmd)}")
    (SCHED_DIR / "current_step.txt").write_text(f"{ts()} {run_name}\n")
    proc = subprocess.run(cmd, cwd=REPO)
    note(f"step end: {run_name} exit {proc.returncode}")
    return proc.returncode


def max_logged_step(log_path: Path) -> int:
    """Highest step number in a train_log.jsonl, tolerant of a torn last line."""
    best = -1
    if not log_path.exists():
        return best
    with open(log_path) as f:
        for line in f:
            try:
                best = max(best, json.loads(line).get("step", -1))
            except (json.JSONDecodeError, AttributeError):
                continue
    return best


def gate_losses(log_path: Path, window: int) -> tuple[list[float], list[float], bool]:
    """(first 8 in-window losses, last 8 in-window losses, saw_nonfinite)."""
    losses: list[tuple[int, float]] = []
    nonfinite = False
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                step, loss = rec.get("step"), rec.get("loss")
                if step is None or loss is None or step >= window:
                    continue
                if not math.isfinite(loss):
                    nonfinite = True
                losses.append((step, loss))
    losses.sort()
    vals = [v for _, v in losses]
    return vals[:8], vals[-8:], nonfinite


def run_size120m_gated(lr: float) -> int:
    """The headline size120m run with the pre-registered stability check.

    The trainer has no stop-after flag on its CLI, so the gate watches the
    live log for the first 2 percent of steps. If the run is already past the
    window (a resume), the gate is skipped: it already passed once.
    """
    run_name = "main_size120m"
    total_steps = math.ceil(MAIN_TOKENS / TOKENS_PER_STEP)
    window = max(1, math.ceil(GATE_FRACTION * total_steps))
    log_path = RUN_ROOT / run_name / "train_log.jsonl"
    halved_marker = SCHED_DIR / "size120m_lr_halved.json"

    if halved_marker.exists():
        lr = json.loads(halved_marker.read_text())["halved_lr"]
        note(f"size120m: using previously halved lr {lr}")

    already = max_logged_step(log_path)
    gate_active = already < window
    if not gate_active:
        note(f"size120m: resume at step {already} >= gate window {window}, gate passed earlier")

    cmd = train_cmd(run_name, "size120m", lr, MAIN_TOKENS, micro_batch=16)
    note(f"step start: {' '.join(cmd)} (gate window {window} steps, active={gate_active})")
    (SCHED_DIR / "current_step.txt").write_text(f"{ts()} {run_name}\n")
    proc = subprocess.Popen(cmd, cwd=REPO)

    diverged = False
    while proc.poll() is None:
        time.sleep(GATE_POLL_SECONDS)
        if not gate_active:
            continue
        head, tail, nonfinite = gate_losses(log_path, window)
        step_now = max_logged_step(log_path)
        if nonfinite:
            diverged = True
            note(f"size120m GATE: non-finite loss inside first {window} steps")
        elif step_now >= window and len(head) >= 8 and len(tail) >= 8:
            h, t = sum(head) / len(head), sum(tail) / len(tail)
            if t > h:
                diverged = True
                note(f"size120m GATE: end-of-window mean loss {t:.4f} > start {h:.4f}")
            else:
                note(f"size120m GATE passed: start {h:.4f} -> end {t:.4f} over {window} steps")
                gate_active = False
        if diverged:
            proc.terminate()
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
            break

    if not diverged:
        rc = proc.returncode or 0
        note(f"step end: {run_name} exit {rc}")
        return rc

    # Divergence: allowed exactly one halving, recorded, then a fresh run.
    if halved_marker.exists():
        note("size120m diverged again after the one allowed halving: hard failure")
        return 20
    stamp = ts().replace(":", "")
    for root in (RUN_ROOT, CKPT_ROOT):
        d = root / run_name
        if d.exists():
            d.rename(root / f"{run_name}_diverged_{stamp}")
    halved_marker.write_text(json.dumps(
        {"time": ts(), "original_lr": lr, "halved_lr": lr / 2,
         "reason": "divergence in first 2 percent of steps, per pre-registration"},
        indent=2))
    note(f"size120m: lr halved to {lr / 2}, diverged run archived, will restart next cycle")
    return 0


def score_lr_ablations() -> int:
    """TinyStories validation perplexity for each LR ablation, then the winner."""
    ABL_DIR.mkdir(parents=True, exist_ok=True)
    (SCHED_DIR / "current_step.txt").write_text(f"{ts()} score_lr_ablations\n")
    scores: dict[str, float] = {}
    for run_name in LR_ABLATIONS:
        out = ABL_DIR / f"valppl_{run_name}.json"
        if not out.exists():
            cmd = [
                str(PYTHON), str(REPO / "scripts" / "06_evaluate.py"),
                "--checkpoint", str(final_ckpt(run_name)),
                "--config", "size30m",
                "--device", "mps",
                "--suite", "perplexity",
                "--split", "val",
                "--out", str(out),
            ]
            note(f"scoring: {' '.join(cmd)}")
            rc = subprocess.run(cmd, cwd=REPO).returncode
            if rc != 0:
                note(f"scoring {run_name} failed with exit {rc}")
                return 20
        blob = json.loads(out.read_text())

        def find_ppl(node) -> float | None:
            if isinstance(node, dict):
                if node.get("domain") == "tinystories" and "perplexity" in node:
                    return node["perplexity"]
                for v in node.values():
                    got = find_ppl(v)
                    if got is not None:
                        return got
            elif isinstance(node, list):
                for v in node:
                    got = find_ppl(v)
                    if got is not None:
                        return got
            return None

        ppl = find_ppl(blob)
        if ppl is None:
            note(f"no tinystories perplexity found in {out}")
            return 20
        scores[run_name] = ppl
        note(f"{run_name}: tinystories val perplexity {ppl:.4f}")

    winner = min(scores, key=scores.get)
    WINNER_FILE.write_text(json.dumps(
        {"time": ts(), "rule": "argmin TinyStories validation perplexity, "
         "final 40M checkpoint, per docs/PROTOCOL.md",
         "scores": scores, "winner": winner,
         "winning_lr": LR_ABLATIONS[winner]}, indent=2))
    note(f"LR winner: {winner} (lr {LR_ABLATIONS[winner]})")
    return 0


def repo_code_state() -> str:
    """head hash, suffixed -dirty when the working tree has changes."""
    head = subprocess.run(
        ["git", "rev-parse", "head"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip() or "unknown"
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()
    return f"{head}-dirty" if dirty else head


def ensure_smoke_gate() -> int:
    """Run the smoke gate once per code state; refuse the queue if it fails.

    A dirty tree never caches a pass: the gate reruns every cycle until the
    code is committed, which is a few minutes against the hours it protects.
    """
    state = repo_code_state()
    if SMOKE_MARKER.exists() and not state.endswith("-dirty"):
        try:
            if json.loads(SMOKE_MARKER.read_text()).get("code_state") == state:
                return 0
        except json.JSONDecodeError:
            pass
    note(f"smoke gate: running for code state {state}")
    proc = subprocess.run(
        [str(PYTHON), str(REPO / "scripts" / "smoke_gate.py"), "--device", "mps"],
        cwd=REPO,
    )
    if proc.returncode != 0:
        note("smoke gate failed: no schedule step runs until the downstream "
             "pipeline passes end to end on its real device")
        return 20
    SMOKE_MARKER.write_text(
        json.dumps({"code_state": state, "time": ts()}, indent=2) + "\n")
    note("smoke gate passed")
    return 0


def held_before(step_name: str) -> bool:
    if HOLD_FILE.exists() and HOLD_FILE.read_text().strip() == step_name:
        note(f"holding before {step_name}: named by hold_before.txt, exiting")
        return True
    return False


def main() -> int:
    if not BULK_ROOT.is_dir():
        note(f"bulk artifact root {BULK_ROOT} does not exist: "
             "hard failure, will not train into a void")
        return 20

    rc = ensure_smoke_gate()
    if rc != 0:
        return rc

    # 1-2. The two remaining LR ablations.
    for run_name, lr in LR_ABLATIONS.items():
        if not run_complete(run_name, ABL_TOKENS):
            if held_before(run_name):
                return 30
            return run_training(train_cmd(run_name, "size30m", lr, ABL_TOKENS), run_name)

    # 3. Score them and fix the winner.
    if not WINNER_FILE.exists():
        if held_before("score_lr_ablations"):
            return 30
        return score_lr_ablations()
    win_lr = json.loads(WINNER_FILE.read_text())["winning_lr"]

    # 4. Warmup ablation at the winning lr.
    if not run_complete("abl_warmup0", ABL_TOKENS):
        if held_before("abl_warmup0"):
            return 30
        return run_training(
            train_cmd("abl_warmup0", "size30m", win_lr, ABL_TOKENS, warmup=0.0),
            "abl_warmup0")

    # 5-6. Headline runs, smallest first so lock holds start short.
    for run_name, size in (("main_size30m", "size30m"), ("main_size60m", "size60m")):
        if not run_complete(run_name, MAIN_TOKENS):
            if held_before(run_name):
                return 30
            return run_training(train_cmd(run_name, size, win_lr, MAIN_TOKENS), run_name)

    # 7. size120m with the stability gate.
    if not run_complete("main_size120m", MAIN_TOKENS):
        if held_before("main_size120m"):
            return 30
        return run_size120m_gated(win_lr)

    note("schedule complete: all ablations and all three headline runs done")
    return 10


if __name__ == "__main__":
    sys.exit(main())
