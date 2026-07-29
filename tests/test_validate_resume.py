"""The resume validator guards the project's most destructive failure mode.

If a supervisor restarts after a power cut, finds no usable resume state, and quietly
starts fresh, it overwrites an in-progress run and restarts from token zero while
still looking like a healthy run. The checkpoint grid is the experiment for an
induction-head emergence study, so that silently destroys the result.

These tests pin the asymmetric contract: fresh only when genuinely empty, resume only
when the state is valid AND belongs to this run, refuse in every other case.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "17_validate_resume.py"

EXIT_RESUME, EXIT_FRESH, EXIT_REFUSE = 0, 10, 20
RUN = "size30m_staged_seed1"
TPS = 16384


def _state(step: int, tokens: int, run: str = RUN, fw: str = "torch", seed: int = 1) -> dict:
    return {
        "weights_fp32": {"w": torch.zeros(2)},
        "optimizer": {},
        "stream_state": {},
        "step": step,
        "tokens": tokens,
        "framework": fw,
        "run_name": run,
        "seed": seed,
    }


def _run(ckpt_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--checkpoint-dir", str(ckpt_dir),
            "--run-name", RUN,
            "--framework", "torch",
            "--seed", "1",
            "--tokens-per-step", str(TPS),
        ],
        capture_output=True, text=True,
    )


def test_empty_directory_is_a_safe_fresh_start(tmp_path: Path) -> None:
    assert _run(tmp_path).returncode == EXIT_FRESH


def test_valid_state_resumes(tmp_path: Path) -> None:
    torch.save(_state(309, 310 * TPS), tmp_path / "resume_state.pt")
    torch.save({"w": 1}, tmp_path / "tokens_000004014080.pt")
    r = _run(tmp_path)
    assert r.returncode == EXIT_RESUME
    assert "5,079,040" in r.stdout


def test_checkpoints_without_resume_state_REFUSE_never_fresh(tmp_path: Path) -> None:
    """The destructive case. A fresh start here would overwrite a live run."""
    torch.save({"w": 1}, tmp_path / "tokens_000004014080.pt")
    r = _run(tmp_path)
    assert r.returncode == EXIT_REFUSE
    assert r.returncode != EXIT_FRESH
    assert "missing" in r.stdout


def test_resume_state_from_a_different_run_is_refused(tmp_path: Path) -> None:
    torch.save(_state(309, 310 * TPS, run="some_other_run"), tmp_path / "resume_state.pt")
    assert _run(tmp_path).returncode == EXIT_REFUSE


@pytest.mark.parametrize("fw,seed", [("mlx", 1), ("torch", 7)])
def test_framework_or_seed_mismatch_is_refused(tmp_path: Path, fw: str, seed: int) -> None:
    torch.save(_state(309, 310 * TPS, fw=fw, seed=seed), tmp_path / "resume_state.pt")
    assert _run(tmp_path).returncode == EXIT_REFUSE


def test_tokens_inconsistent_with_step_is_refused(tmp_path: Path) -> None:
    torch.save(_state(309, 12345), tmp_path / "resume_state.pt")
    r = _run(tmp_path)
    assert r.returncode == EXIT_REFUSE
    assert "inconsistent" in r.stdout


def test_unloadable_resume_state_is_refused(tmp_path: Path) -> None:
    (tmp_path / "resume_state.pt").write_bytes(b"not a torch file")
    r = _run(tmp_path)
    assert r.returncode == EXIT_REFUSE
    assert "will not load" in r.stdout


def test_missing_required_key_is_refused(tmp_path: Path) -> None:
    bad = _state(309, 310 * TPS)
    del bad["optimizer"]
    torch.save(bad, tmp_path / "resume_state.pt")
    r = _run(tmp_path)
    assert r.returncode == EXIT_REFUSE
    assert "missing keys" in r.stdout


def test_grid_ahead_of_resume_point_is_refused(tmp_path: Path) -> None:
    """A grid checkpoint newer than the resume state means unaccounted-for work."""
    torch.save(_state(60, 61 * TPS), tmp_path / "resume_state.pt")
    torch.save({"w": 1}, tmp_path / "tokens_000004014080.pt")
    r = _run(tmp_path)
    assert r.returncode == EXIT_REFUSE
    assert "ahead of the resume point" in r.stdout
