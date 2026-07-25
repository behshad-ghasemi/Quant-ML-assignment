"""Baseline probabilistic forecasters.

Two baselines are used in this project:

1. `EmpiricalQuantileClimatology` (built here): for each (month,
   hour-of-day, is_weekend) combination, the empirical 1st..99th
   percentiles of historical LOAD in that group, estimated only from
   data strictly before the target month. This is the "empirical-
   quantile / climatology baseline" the assignment calls out explicitly,
   and it is a materially stronger baseline than a flat interval because
   it already captures time-of-day and seasonal shape.

2. The official GEFCom2014 benchmark shipped in each Ln-benchmark.csv
   (same month last year, expanded flatly to all 99 quantiles). We don't
   need to reimplement this -- `TaskBundle.benchmark_q` already carries
   it -- but we evaluate it alongside our own baseline for reference.

All sophisticated models in this project are required to beat baseline
(1); baseline (2) is reported mainly as a sanity floor, since the
assignment notes it is a deliberately weak, naive baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import QUANTILE_LEVELS

_MIN_GROUP_SIZE = 20


class EmpiricalQuantileClimatology:
    GROUP_COLS = ["month", "hour", "is_weekend"]

    def fit(self, history: pd.DataFrame) -> "EmpiricalQuantileClimatology":
        valid = history.dropna(subset=["LOAD"])
        key = pd.DataFrame(
            {
                "month": valid.index.month,
                "hour": valid.index.hour,
                "is_weekend": (valid.index.dayofweek >= 5).astype(int),
                "LOAD": valid["LOAD"].to_numpy(),
            }
        )
        self._by_full_group: dict[tuple, np.ndarray] = {
            k: np.quantile(g["LOAD"].to_numpy(), QUANTILE_LEVELS) for k, g in key.groupby(self.GROUP_COLS)
        }
        self._group_size: dict[tuple, int] = {k: len(g) for k, g in key.groupby(self.GROUP_COLS)}
        # Fallback tables for sparse groups: drop is_weekend, then drop month too.
        self._by_month_hour: dict[tuple, np.ndarray] = {
            k: np.quantile(g["LOAD"].to_numpy(), QUANTILE_LEVELS) for k, g in key.groupby(["month", "hour"])
        }
        self._by_hour: dict[int, np.ndarray] = {
            k: np.quantile(g["LOAD"].to_numpy(), QUANTILE_LEVELS) for k, g in key.groupby("hour")
        }
        return self

    def predict(self, index: pd.DatetimeIndex) -> np.ndarray:
        out = np.empty((len(index), len(QUANTILE_LEVELS)))
        months, hours = index.month, index.hour
        weekend = (index.dayofweek >= 5).astype(int)
        for i in range(len(index)):
            key = (months[i], hours[i], weekend[i])
            if key in self._by_full_group and self._group_size.get(key, 0) >= _MIN_GROUP_SIZE:
                out[i] = self._by_full_group[key]
            elif (months[i], hours[i]) in self._by_month_hour:
                out[i] = self._by_month_hour[(months[i], hours[i])]
            else:
                out[i] = self._by_hour[hours[i]]
        return np.sort(out, axis=1)
