from __future__ import annotations

import importlib.util
from pathlib import Path


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "27_replication_controls.py"
    spec = importlib.util.spec_from_file_location("replication_controls", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_replication_location_threshold() -> None:
    module = _script()
    headline = {"phase_change": {"crossed": True, "end_tokens": 48}}
    close = {"phase_change": {"crossed": True, "end_tokens": 96}}
    far = {"phase_change": {"crossed": True, "end_tokens": 120}}
    assert module.replication_verdict(headline, close)["verdict"] == "replicated_within_2x"
    assert module.replication_verdict(headline, far)["verdict"] == "location_uncertain"


def test_replication_without_crossing_is_censored() -> None:
    module = _script()
    headline = {"phase_change": {"crossed": True, "end_tokens": 48}}
    absent = {"phase_change": {"crossed": False}, "final_tokens": 64}
    assert module.replication_verdict(headline, absent)["verdict"] == "censored"
