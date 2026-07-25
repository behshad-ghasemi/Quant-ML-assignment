"""Diebold-Mariano test for comparing two forecasts' loss series.

We apply it to the *hourly* mean-pinball-loss differential between two
models, concatenated across all backtest tasks in chronological order.
Hourly load-forecast errors are strongly autocorrelated (daily/weekly
cycles in the error itself), so a naive variance estimate assuming i.i.d.
loss differences would be badly anti-conservative (understated p-values).
We use a Newey-West / Bartlett-kernel HAC variance estimator instead,
which is the standard fix used in the DM-test literature (Diebold &
Mariano, 1995; Harvey, Leybourne & Newbold, 1997 small-sample correction).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class DMTestResult:
    dm_stat: float
    p_value: float
    mean_loss_diff: float
    n_obs: int
    lag: int
    better_model: str

    def __str__(self) -> str:
        return (
            f"DM stat={self.dm_stat:.3f}, p={self.p_value:.4f}, "
            f"mean_diff={self.mean_loss_diff:+.4f} (n={self.n_obs}, lag={self.lag}) "
            f"-> {self.better_model}"
        )


def _newey_west_long_run_variance(x: np.ndarray, lag: int) -> float:
    n = len(x)
    x = x - x.mean()
    gamma0 = np.dot(x, x) / n
    var = gamma0
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)  # Bartlett kernel
        gamma_k = np.dot(x[k:], x[:-k]) / n
        var += 2 * w * gamma_k
    return var


def diebold_mariano_test(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    model_a_name: str = "model_a",
    model_b_name: str = "model_b",
    lag: int | None = None,
    small_sample_correction: bool = True,
) -> DMTestResult:
    """Two-sided DM test on d_t = loss_a[t] - loss_b[t].

    H0: E[d_t] = 0 (equal predictive accuracy).
    Negative dm_stat / mean_loss_diff => model_a has lower average loss.

    `lag` defaults to a standard rule of thumb, floor(4*(n/100)^(2/9)),
    capped so it can't exceed the horizon of a single forecast month
    (744h) by more than a small multiple -- with 15 monthly folds
    concatenated there is enough data for this to be stable, but we still
    cap it defensively.
    """
    loss_a = np.asarray(loss_a, dtype=float)
    loss_b = np.asarray(loss_b, dtype=float)
    if loss_a.shape != loss_b.shape:
        raise ValueError("loss_a and loss_b must have the same shape")
    d = loss_a - loss_b
    n = len(d)
    if lag is None:
        lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
        lag = max(1, min(lag, n // 4))

    d_bar = d.mean()
    long_run_var = _newey_west_long_run_variance(d, lag)
    var_d_bar = long_run_var / n
    if var_d_bar <= 0:
        dm_stat = 0.0
    else:
        dm_stat = d_bar / np.sqrt(var_d_bar)

    if small_sample_correction:
        # Harvey, Leybourne & Newbold (1997) correction, then use the
        # Student-t(n-1) distribution instead of the standard normal.
        correction = np.sqrt((n + 1 - 2 * lag + lag * (lag - 1) / n) / n)
        dm_stat *= correction
        p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat), df=n - 1))
    else:
        p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

    if d_bar < 0:
        better = f"{model_a_name} (lower loss)"
    elif d_bar > 0:
        better = f"{model_b_name} (lower loss)"
    else:
        better = "tie"

    return DMTestResult(
        dm_stat=float(dm_stat),
        p_value=float(p_value),
        mean_loss_diff=float(d_bar),
        n_obs=n,
        lag=lag,
        better_model=better,
    )
