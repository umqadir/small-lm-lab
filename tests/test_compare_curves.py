"""Unit tests for the cross-backend curve gate, scripts/11_compare_curves.py.

Synthetic logs in the repo's own train_log.jsonl format, one pass case and
several fail cases, so the gate's arithmetic and its exit code are both checked
without needing a real run. Nothing here touches the corpus, the GPU, or the
training lock.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "11_compare_curves.py"


def load_compare_script():
    spec = importlib.util.spec_from_file_location("compare_script", COMPARE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the script defines a dataclass, and dataclasses looks
    # its module up in sys.modules by __module__ while building the class. Loaded
    # as __main__ on the command line this is already satisfied; under importlib
    # the module has to be registered first. This is the documented idiom.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_log(path: Path, losses: dict[int, float], extra_keys: bool = True) -> None:
    """Write a train_log.jsonl with the fields train.py writes."""
    with path.open("w") as f:
        for step, loss in losses.items():
            record = {"step": step, "loss": loss}
            if extra_keys:
                record.update({"tokens": (step + 1) * 16384, "lr": 1.2e-3})
            f.write(json.dumps(record) + "\n")


def logged_steps(start: int = 409, end: int = 499, every: int = 10) -> list[int]:
    """Steps a log_every=10 run records inside a window, like the banked run."""
    return list(range(start, end + 1, every))


def test_pass_case_within_both_tolerances(tmp_path: Path) -> None:
    script = load_compare_script()
    steps = logged_steps()
    ref = {s: 2.5 - 0.001 * s for s in steps}
    # Candidate differs by a hair, well under both tolerances everywhere.
    cand = {s: ref[s] + 0.005 for s in steps}

    rp, cp = tmp_path / "r.jsonl", tmp_path / "c.jsonl"
    write_log(rp, ref)
    write_log(cp, cand)
    cmp = script.compare_logs(
        rp, cp, step_start=400, step_end=500, tol_end=0.02, tol_mean=0.015
    )
    assert cmp.passed
    assert cmp.end_step == 499
    assert cmp.end_delta == pytest.approx(0.005)
    assert cmp.mean_delta == pytest.approx(0.005)
    assert cmp.common_steps == steps


def test_fail_on_end_delta(tmp_path: Path) -> None:
    """Matches on average but blows the tolerance at the final step: still a
    failure, because the end of the window is checked on its own."""
    script = load_compare_script()
    steps = logged_steps()
    ref = {s: 2.0 for s in steps}
    cand = {s: 2.0 for s in steps}
    cand[499] = 2.0 + 0.05  # only the last step diverges, past tol_end 0.02

    rp, cp = tmp_path / "r.jsonl", tmp_path / "c.jsonl"
    write_log(rp, ref)
    write_log(cp, cand)
    cmp = script.compare_logs(rp, cp, 400, 500, 0.02, 0.015)
    assert not cmp.passed
    assert cmp.end_delta == pytest.approx(0.05)
    # The mean stays small (one diverged step out of ten), so this is the end
    # tolerance failing on its own.
    assert cmp.mean_delta < 0.015


def test_fail_on_mean_delta(tmp_path: Path) -> None:
    """Ends inside tolerance but drifts across the window: the mean catches it."""
    script = load_compare_script()
    steps = logged_steps()
    ref = {s: 2.0 for s in steps}
    cand = {s: 2.0 + 0.03 for s in steps}
    cand[499] = 2.0 + 0.01  # end is fine, but every earlier step is off by 0.03

    rp, cp = tmp_path / "r.jsonl", tmp_path / "c.jsonl"
    write_log(rp, ref)
    write_log(cp, cand)
    cmp = script.compare_logs(rp, cp, 400, 500, 0.02, 0.015)
    assert not cmp.passed
    assert cmp.end_delta == pytest.approx(0.01)  # end passes
    assert cmp.mean_delta > 0.015  # mean fails


def test_last_record_wins_on_a_duplicated_step(tmp_path: Path) -> None:
    """A resumed run appends replayed steps, so a step can appear twice. The
    later record is the one that ran, and the gate must read it that way."""
    script = load_compare_script()
    ref = tmp_path / "r.jsonl"
    cand = tmp_path / "c.jsonl"
    write_log(ref, {409: 2.0, 419: 2.0, 429: 2.0})
    # Candidate logs step 429 twice: a stale 9.9 then the real 2.0 on resume.
    with cand.open("w") as f:
        for step, loss in [(409, 2.0), (419, 2.0), (429, 9.9), (429, 2.0)]:
            f.write(json.dumps({"step": step, "loss": loss}) + "\n")
    cmp = script.compare_logs(ref, cand, 400, 430, 0.02, 0.015)
    assert cmp.passed, "the stale duplicate record should have been overwritten"
    assert cmp.end_step == 429
    assert cmp.end_delta == pytest.approx(0.0)


def test_no_common_steps_is_an_error(tmp_path: Path) -> None:
    script = load_compare_script()
    rp, cp = tmp_path / "r.jsonl", tmp_path / "c.jsonl"
    write_log(rp, {9: 2.0, 19: 2.0})  # only early steps
    write_log(cp, {409: 2.0, 419: 2.0})  # only late steps
    with pytest.raises(SystemExit, match="nothing to compare"):
        script.compare_logs(rp, cp, 400, 500, 0.02, 0.015)


def test_main_exits_zero_on_pass_and_nonzero_on_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exit code is the whole point: a shell gate reads it to go or stop."""
    script = load_compare_script()
    steps = logged_steps()
    ref = tmp_path / "r.jsonl"
    write_log(ref, {s: 2.0 for s in steps})

    passing = tmp_path / "pass.jsonl"
    write_log(passing, {s: 2.0 + 0.001 for s in steps})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "11_compare_curves.py", "--reference", str(ref),
            "--candidate", str(passing), "--step-start", "400", "--step-end", "500",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        script.main()
    assert exc.value.code == 0

    failing = tmp_path / "fail.jsonl"
    bad = {s: 2.0 for s in steps}
    bad[499] = 2.5
    write_log(failing, bad)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "11_compare_curves.py", "--reference", str(ref),
            "--candidate", str(failing), "--step-start", "400", "--step-end", "500",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        script.main()
    assert exc.value.code == 1


def test_reference_default_points_at_the_banked_ablation_log() -> None:
    """The default reference is the real gate's reference, recorded so the pod
    invocation does not have to hardcode the path."""
    script = load_compare_script()
    assert script.REFERENCE_ABL_LR12E3.name == "train_log.jsonl"
    assert script.REFERENCE_ABL_LR12E3.parent.name == "abl_lr1.2e-3"
