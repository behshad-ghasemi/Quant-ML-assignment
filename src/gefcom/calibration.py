"""Calibration diagnostics for probabilistic forecasts."""
from __future__ import annotations

import numpy as np
import pandas as pd


def reliability_curve(y_true: np.ndarray, y_pred_quantiles: np.ndarray, quantile_levels: np.ndarray) -> pd.DataFrame:
    """For each nominal quantile level q, the empirical fraction of
    observations with y_true <= predicted q-quantile. A well-calibrated
    forecaster has empirical ~= nominal along the whole curve."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1, 1)
    empirical = (y_true <= y_pred_quantiles).mean(axis=0)
    return pd.DataFrame({"nominal": quantile_levels, "empirical": empirical})


def interval_coverage_table(
    y_true: np.ndarray,
    y_pred_quantiles: np.ndarray,
    quantile_levels: np.ndarray,
    nominal_intervals=(0.5, 0.8, 0.9, 0.95, 0.98),
) -> pd.DataFrame:
    from .metrics import coverage, interval_width

    rows = []
    for nominal in nominal_intervals:
        alpha = (1 - nominal) / 2
        lower, upper = alpha, 1 - alpha
        cov = coverage(y_true, y_pred_quantiles, quantile_levels, lower, upper)
        width = interval_width(y_pred_quantiles, quantile_levels, lower, upper)
        rows.append({"nominal_coverage": nominal, "empirical_coverage": cov, "mean_interval_width": width})
    return pd.DataFrame(rows)
