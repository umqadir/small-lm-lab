from __future__ import annotations

import importlib.util
from pathlib import Path


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "25_anchor_selection.py"
    spec = importlib.util.spec_from_file_location("anchor_selection", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_select_anchors_and_neighbours() -> None:
    module = _script()
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    scores = [0.01, 0.02, 0.08, 0.25, 0.48, 0.50, 0.49, 0.50]
    result = module.select_anchors(tokens, scores, mu=0.01, sigma=0.01)
    assert result["rise"]["confirmation_tokens"] == 5
    assert result["plateau"]["confirmation_tokens"] == 7
    assert result["anchors"]["pre"]["tokens"] == 2
    assert result["anchors"]["mid"]["tokens"] == 4
    assert result["anchors"]["post"]["tokens"] == 5
    assert result["anchors"]["final"]["tokens"] == 8
    assert result["causal_checkpoint_tokens"] == [1, 2, 3, 4, 5, 6, 7, 8]
