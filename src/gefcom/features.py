"""Feature engineering for one-month-ahead hourly load forecasting.

Design principle (this is the single most important decision in this
project, see README section "Leakage protection"):

  We forecast an entire month (up to 744 hours) in one shot, not one hour
  recursively-fed at a time. That means any feature must be computable
  for EVERY hour of the target month using *only* information that would
  genuinely be known at the moment the forecast is issued (end of the
  prior month). Concretely this rules out short load lags/rolling windows
  (lag-1h, lag-24h, lag-168h, ...): for an hour deep into the target
  month, "load 24h ago" or "load 168h ago" often falls *inside* the same
  unobserved target month, which would require either leaking the answer
  or an iterative/recursive forecast (feeding predictions back in as
  features). We deliberately avoid recursive forecasting here: it
  compounds error and uncertainty across up to 744 steps and adds a lot
  of complexity for a probabilistic (not just point) forecast, which the
  assignment explicitly says is not required ("unnecessary complexity
  will not receive additional credit").

  Instead we use three feature families, ALL of which are honestly
  available for any hour of the target month without touching that
  month's own data:
    1. Calendar features (hour/day/month/weekday/holiday, cyclic
       encodings, a linear trend term) -- exactly known for any future
       timestamp.
    2. Load climatology -- for each (month, hour-of-day, is_weekend)
       combination, the historical mean/median/std/quantiles of LOAD,
       estimated ONLY from data strictly before the target month. This
       plays the same role a "seasonal-naive" feature would, but let's
       the GBM combine it with everything else instead of using it as a
       final answer directly (that's what the baseline models do).
    3. Weather-ensemble climatology -- aggregate statistics (mean,
       median, std, min, max, IQR) across the 25 weather stations,
       again estimated per (month, hour-of-day) from history strictly
       before the target month.

  A separate, clearly-labelled "oracle" feature set uses the *actual*
  realised weather-station readings for the target month, when available
  from a solution file. This is deliberately kept as an opt-in,
  separately-reported variant used ONLY to measure how much the
  temperature-leakage assumption matters (assignment requirement #4),
  never as the default.

IMPORTANT -- the `reference_date` parameter (bug history, read this):
  `add_calendar_features` and `build_feature_frame` both take a required
  `reference_date`, used only to anchor the "trend" (years-since-start)
  feature. An earlier version computed this as `index.min()` of whatever
  slice was passed in -- which is correct for training rows (anchored at
  the start of history) but WRONG for target-month rows, where it
  silently reset to the start of the *target month itself*. That created
  a hard discontinuity: training saw `trend` climb over ~10 years, while
  every target-month prediction saw `trend` reset to ~0, regardless of
  which real year the target month fell in. Any model that leaned on
  `trend` for secular growth would then apply "start of history" behaviour
  to every forecast. `reference_date` must always be the same fixed point
  (in practice: `history.index.min()` for the task/fold in question) for
  BOTH the training-row call and the target-row call -- never left to
  default from whatever index happens to be passed in.
"""
from __future__ import annotations
from builtins import sorted

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from .data_loading import WEATHER_COLUMNS

_HOLIDAYS = USFederalHolidayCalendar().holidays(start="1999-01-01", end="2015-01-01")

def days_to_nearest_holiday(index: pd.DatetimeIndex) -> np.ndarray:
    holidays = pd.DatetimeIndex(sorted(_HOLIDAYS))
    days = pd.DatetimeIndex(index.normalize().unique()).sort_values()

    pos = holidays.searchsorted(days)
    pos_prev = np.clip(pos - 1, 0, len(holidays) - 1)
    pos_next = np.clip(pos, 0, len(holidays) - 1)

    dist_prev = (days - holidays[pos_prev]).days.to_numpy()
    dist_next = (holidays[pos_next] - days).days.to_numpy()

    signed = np.where(dist_prev <= dist_next, -dist_prev, dist_next)
    lookup = pd.Series(signed, index=days)
    return lookup.reindex(index.normalize()).to_numpy()

def add_calendar_features(index: pd.DatetimeIndex, reference_date: pd.Timestamp) -> pd.DataFrame:
    """`reference_date` anchors the `trend` feature and MUST be the same
    fixed point of time used for every call within a given task/fold (see
    module docstring) -- typically `history.index.min()`, never
    `index.min()` of whatever slice is being featurised right now."""
    df = pd.DataFrame(index=index)
    df["hour"] = index.hour
    df["dow"] = index.dayofweek
    df["month"] = index.month
    df["day"] = index.day
    df["doy"] = index.dayofyear
    df["is_weekend"] = (index.dayofweek >= 5).astype(int)
    df["is_holiday"] = index.normalize().isin(_HOLIDAYS).astype(int)
    df["days_to_holiday"] = days_to_nearest_holiday(index)
    df["trend"] = (index - reference_date) / pd.Timedelta(days=365.25)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)
    return df


def weather_ensemble_stats(weather_df: pd.DataFrame, hdd_cdd_base: float = 65.0) -> pd.DataFrame:
    """Row-wise aggregate stats across the 25 weather stations, plus
    heating/cooling degree-day features. Load-temperature relationships
    are typically U-shaped (both cold and hot weather raise consumption),
    which raw mean/std temperature stats don't capture directly -- HDD/CDD
    make that non-linearity explicit as two separate, always-non-negative
    features instead of asking the model to discover the U-shape itself
    from a single linear-ish temperature feature. `hdd_cdd_base=65`F is the
    standard base used in US utility load forecasting."""
    arr = weather_df[WEATHER_COLUMNS].to_numpy(dtype=float)
    out = pd.DataFrame(index=weather_df.index)
    out["w_mean"] = arr.mean(axis=1)
    out["w_median"] = np.median(arr, axis=1)
    out["w_std"] = arr.std(axis=1)
    out["w_min"] = arr.min(axis=1)
    out["w_max"] = arr.max(axis=1)
    out["w_p25"] = np.percentile(arr, 25, axis=1)
    out["w_p75"] = np.percentile(arr, 75, axis=1)
    out["w_iqr"] = out["w_p25"] - out["w_p75"]
    out["hdd"] = np.maximum(0, hdd_cdd_base - out["w_mean"])
    out["cdd"] = np.maximum(0, out["w_mean"] - hdd_cdd_base)
    return out


def _climatology_table(series: pd.Series, index: pd.DatetimeIndex, group_cols: list[str]) -> pd.DataFrame:
    key = pd.DataFrame({c: getattr(index, c) for c in group_cols})
    key["_val"] = series.to_numpy()
    grouped = key.groupby(group_cols)["_val"]
    table = grouped.agg(["mean", "median", "std", lambda s: s.quantile(0.1), lambda s: s.quantile(0.9)])
    table.columns = ["mean", "median", "std", "q10", "q90"]
    return table.reset_index()


class ClimatologyModel:
    """Fits (month, hour, is_weekend) climatology tables for LOAD and for
    weather-ensemble stats using only the rows passed to `fit` -- callers
    are responsible for only passing historical (pre-target-month) data,
    which is the natural state of `TaskBundle.history` in this project.

    Falls back from the full (month, hour, is_weekend) group to
    (month, hour), then to (hour) alone, when the exact group was never
    observed in a given task's history (e.g. short-history early tasks,
    or a task/fold missing that particular combination after a data gap)
    -- this mirrors the fallback already used by
    EmpiricalQuantileClimatology in baselines.py, so a target-month row
    never silently gets NaN features.
    """

    GROUP_COLS = ["month", "hour", "dayofweek"]

    def fit(self, history: pd.DataFrame) -> "ClimatologyModel":
        valid = history.dropna(subset=["LOAD"])
        self.load_table_ = _climatology_table(valid["LOAD"], valid.index, self.GROUP_COLS)
        self.load_table_month_hour_ = _climatology_table(valid["LOAD"], valid.index, ["month", "hour"])
        self.load_table_hour_ = _climatology_table(valid["LOAD"], valid.index, ["hour"])

        w_stats = weather_ensemble_stats(history)
        self.weather_tables_ = {}
        self.weather_tables_month_hour_ = {}
        self.weather_tables_hour_ = {}
        for col in w_stats.columns:
            self.weather_tables_[col] = _climatology_table(w_stats[col], history.index, self.GROUP_COLS)
            self.weather_tables_month_hour_[col] = _climatology_table(w_stats[col], history.index, ["month", "hour"])
            self.weather_tables_hour_[col] = _climatology_table(w_stats[col], history.index, ["hour"])
        return self

    def _lookup(
        self,
        table: pd.DataFrame,
        index: pd.DatetimeIndex,
        prefix: str,
        table_month_hour: pd.DataFrame | None = None,
        table_hour: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        key = pd.DataFrame({
            "month": index.month,
            "hour": index.hour,
            "dayofweek": index.dayofweek,
        })
        
        merged = key.merge(table, on=self.GROUP_COLS, how="left")
        stat_cols = [c for c in table.columns if c not in self.GROUP_COLS]

        missing = merged[stat_cols[0]].isna()
        if missing.any() and table_month_hour is not None:
            fallback_mh = key.merge(table_month_hour, on=["month", "hour"], how="left")
            for c in stat_cols:
                merged.loc[missing, c] = fallback_mh.loc[missing, c]
            missing = merged[stat_cols[0]].isna()

        if missing.any() and table_hour is not None:
            fallback_h = key.merge(table_hour, on=["hour"], how="left")
            for c in stat_cols:
                merged.loc[missing, c] = fallback_h.loc[missing, c]

        merged.index = index
        merged = merged.drop(columns=self.GROUP_COLS)
        merged.columns = [f"{prefix}_{c}" for c in merged.columns]
        return merged

    def transform_load(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        return self._lookup(
            self.load_table_, index, "load_clim",
            table_month_hour=self.load_table_month_hour_, table_hour=self.load_table_hour_,
        )

    def transform_weather(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        parts = [
            self._lookup(
                tbl, index, f"clim_{name}",
                table_month_hour=self.weather_tables_month_hour_[name],
                table_hour=self.weather_tables_hour_[name],
            )
            for name, tbl in self.weather_tables_.items()
        ]
        return pd.concat(parts, axis=1)


def build_feature_frame(
    index: pd.DatetimeIndex,
    climatology: ClimatologyModel,
    reference_date: pd.Timestamp,
    weather_mode: str = "climatology",
    observed_weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble the full feature matrix for `index`.

    `reference_date` anchors the `trend` feature (see module docstring for
    why this must be threaded through explicitly rather than defaulted).
    It must be the SAME value for the training-row call and the
    target-row call within one task/fold -- typically
    `history.index.min()` for whichever `history`/`train_hist` produced
    the climatology and training rows in question.

    `weather_mode` controls how weather features are derived, and MUST be
    the same for every row a given model is trained or predicted on --
    mixing modes between training and prediction would silently change
    the meaning/columns of the weather features between fit and predict.

      - "climatology" (default, leakage-safe): weather features are the
        (month, hour, is_weekend) climatology lookup, estimated only from
        history strictly before the target month.
      - "observed": row-wise stats computed directly from
        `observed_weather` (real w1..w25). Used solely for the
        explicitly-flagged "oracle" comparison, never the default.
    """
    if weather_mode not in ("climatology", "observed"):
        raise ValueError(f"Unknown weather_mode: {weather_mode!r}")
    if weather_mode == "observed" and observed_weather is None:
        raise ValueError("weather_mode='observed' requires observed_weather to be supplied")

    feats = [add_calendar_features(index, reference_date), climatology.transform_load(index)]
    if weather_mode == "observed":
        feats.append(weather_ensemble_stats(observed_weather))
    else:
        feats.append(climatology.transform_weather(index))
    return pd.concat(feats, axis=1)

