"""Test-wide defaults.

MLX defaults to the GPU, so an MLX forward pass in a test reaches for the GPU
unless something says otherwise. The test suite is meant to be runnable at any
time, including while another job holds the shared training lock, so it pins MLX
to the CPU for every test. Tests are small enough that this costs little.

This is a test-only default. It does not touch what train.py selects at run time.
"""

from __future__ import annotations

import mlx.core as mx
import pytest


@pytest.fixture(autouse=True, scope="session")
def _pin_mlx_to_cpu() -> None:
    mx.set_default_device(mx.cpu)
