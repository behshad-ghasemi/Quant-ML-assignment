"""Pinball (quantile) loss and related quantile-forecast utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd

QUANTILE_LEVELS = np.round(np.arange(1, 100) / 100, 2)


def pinball_loss(y_true: np.ndarray, y_pred_quantiles: np.ndarray, quantile_levels: np.ndarray) -> np.ndarray:
    """Elementwise pinball loss.

    y_true: shape (n,)
    y_pred_quantiles: shape (n, n_quantiles)
    quantile_levels: shape (n_quantiles,), values in (0, 1)

    Returns an (n, n_quantiles) array of losses (not yet averaged), so
    callers can aggregate over whichever axis they need (per-hour,
    per-quantile, per-task, ...).
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1, 1)
    diff = y_true - y_pred_quantiles
    q = quantile_levels.reshape(1, -1)
    return np.maximum(q * diff, (q - 1) * diff)


def mean_pinball_loss(y_true: np.ndarray, y_pred_quantiles: np.ndarray, quantile_levels: np.ndarray) -> float:
    return float(pinball_loss(y_true, y_pred_quantiles, quantile_levels).mean())


def per_hour_pinball_loss(y_true: np.ndarray, y_pred_quantiles: np.ndarray, quantile_levels: np.ndarray) -> np.ndarray:
    """Average pinball loss across quantiles, for each hour. Shape (n,).
    This is the natural series to feed into the Diebold-Mariano test."""
    return pinball_loss(y_true, y_pred_quantiles, quantile_levels).mean(axis=1)


def enforce_monotonicity(y_pred_quantiles: np.ndarray) -> np.ndarray:
    """Sort quantile predictions along the quantile axis so that
    q_1 <= q_2 <= ... <= q_99 for every row. Gradient-boosted trees fit
    independently per quantile can (rarely) produce small crossings;
    rearranging is the standard fix (Chernozhukov, Fernandez-Val &
    Galichon, 2010)."""
    return np.sort(y_pred_quantiles, axis=1)


def coverage(y_true: np.ndarray, y_pred_quantiles: np.ndarray, quantile_levels: np.ndarray, lower: float, upper: float) -> float:
    """Empirical coverage of the [lower, upper] central interval, e.g.
    lower=0.05, upper=0.95 for a nominal 90% interval."""
    lo_idx = int(np.argmin(np.abs(quantile_levels - lower)))
    hi_idx = int(np.argmin(np.abs(quantile_levels - upper)))
    lo = y_pred_quantiles[:, lo_idx]
    hi = y_pred_quantiles[:, hi_idx]
    y_true = np.asarray(y_true, dtype=float)
    inside = (y_true >= lo) & (y_true <= hi)
    return float(inside.mean())


def interval_width(y_pred_quantiles: np.ndarray, quantile_levels: np.ndarray, lower: float, upper: float) -> float:
    lo_idx = int(np.argmin(np.abs(quantile_levels - lower)))
    hi_idx = int(np.argmin(np.abs(quantile_levels - upper)))
    return float((y_pred_quantiles[:, hi_idx] - y_pred_quantiles[:, lo_idx]).mean())
