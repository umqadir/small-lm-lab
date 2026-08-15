from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "26_ngram_null.py"
    spec = importlib.util.spec_from_file_location("ngram_null", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class State:
    def __init__(self):
        self.context = ()


class ConstantModel:
    def BeginSentenceWrite(self, state):
        state.context = ("<s>",)

    def BaseScore(self, state, token, next_state):
        next_state.context = (*state.context[-3:], token)
        return -1.0


def test_score_window_converts_log10_and_resets_at_eot() -> None:
    module = _script()
    score = module.score_window(
        ConstantModel(), np.array([4, 5, 0, 7], dtype=np.int64), state_factory=State
    )
    assert np.allclose(score, np.log(10.0))
