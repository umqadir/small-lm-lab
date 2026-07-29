"""Percentile bootstrap, one implementation shared by every measurement.

The pre-registration asks for the same uncertainty treatment everywhere: 1000
resamples, percentile interval. That is one procedure, so it lives in one place
instead of being written out once per metric where the four copies could drift
apart.

Resampling is over rows of `values`, always. A row is one independent unit: one
evaluation window, one BLiMP item, one copying sequence. Metrics that need more
than one number per unit (a per-window mean loss and its token count, say) pass
a 2-D array and read the columns inside their own statistic, so the resampling
convention stays identical no matter how many quantities a unit carries. The
statistic is applied to the whole resampled array instead of to a flat vector
of per-unit scores, which is what lets a ratio-of-sums like token-weighted
perplexity be recomputed correctly on each resample instead of being
approximated by an average of per-unit values.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

DEFAULT_N_RESAMPLES = 1000
DEFAULT_ALPHA = 0.05


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = 0,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[float, float, float]:
    """Return (point, lo, hi): the statistic and its percentile bootstrap interval.

    values: array of shape (n,) or (n, k). Row i is unit i.
    statistic: maps an array shaped like `values` to one float. Called once on
        the observed data for the point estimate and once per resample.
    n_resamples: number of bootstrap resamples, each of size n, drawn with
        replacement.
    seed: fixes the resampling, so a rerun reproduces the interval exactly.
    alpha: 0.05 gives the 2.5th and 97.5th percentiles.

    The point estimate is the statistic on the observed data, not the mean of
    the resamples, so it is not shifted by bootstrap noise.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 0:
        raise ValueError("values must have at least one dimension")
    n = values.shape[0]
    if n == 0:
        raise ValueError("values must hold at least one row")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be positive, got {n_resamples}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")

    point = float(statistic(values))

    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples, dtype=np.float64)
    for r in range(n_resamples):
        # Drawn per resample rather than as one (n_resamples, n) block: at ~19k
        # windows and 1000 resamples that block would be 150MB for no gain.
        idx = rng.integers(0, n, size=n)
        stats[r] = float(statistic(values[idx]))

    lo = float(np.percentile(stats, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(stats, 100.0 * (1.0 - alpha / 2.0)))
    return point, lo, hi
