"""Orchestrates a full run (features -> baselines -> models -> predict ->
evaluate) for a single GEFCom2014-L task, following the rolling-origin
structure that is native to the dataset: each task's own train/target
split already IS one expanding-window backtest fold, so no additional
manual CV splitting of the 15 tasks is needed (see README, "Validation
design").
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .baselines import EmpiricalQuantileClimatology
from .calibration import interval_coverage_table, reliability_curve
from .data_loading import TaskBundle, load_task
from .features import ClimatologyModel, build_feature_frame
from .metrics import QUANTILE_LEVELS, enforce_monotonicity, mean_pinball_loss, per_hour_pinball_loss
from .quantile_models import DEFAULT_KNOTS, KnotQuantileRegressor


LINEAR_MODEL_FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend", "is_holiday", "trend",
    "load_clim_mean", "load_clim_q10", "load_clim_q90",
    "clim_w_mean_mean", "clim_w_mean_q10", "clim_w_mean_q90",
    "clim_hdd_mean", "clim_cdd_mean",   
]

LINEAR_MODEL_FEATURES_OBSERVED = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend", "is_holiday", "trend",
    "load_clim_mean", "load_clim_q10", "load_clim_q90",
    "w_mean", "w_std", "w_min", "w_max",
    "hdd", "cdd",   
]

GBM_MODEL_FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "doy_sin", "doy_cos", "is_weekend", "is_holiday", "trend", "days_to_holiday",
    "load_clim_mean", "load_clim_median", "load_clim_std", "load_clim_q10", "load_clim_q90",
    "clim_w_mean_mean", "clim_hdd_mean", "clim_cdd_mean",
]

GBM_MODEL_FEATURES_OBSERVED = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "doy_sin", "doy_cos", "is_weekend", "is_holiday", "trend", "days_to_holiday",
    "load_clim_mean", "load_clim_median", "load_clim_std", "load_clim_q10", "load_clim_q90",
    "w_mean", "hdd", "cdd",
]


@dataclass
class ModelSpec:
    name: str
    family: str  # "linear_qr" | "lightgbm" | "xgboost"
    params: dict = field(default_factory=dict)
    knots: np.ndarray = field(default_factory=lambda: DEFAULT_KNOTS)
    feature_subset: list | None = None
    max_train_rows: int | None = None


@dataclass
class TaskResult:
    task_id: int
    n_target_hours: int
    predictions: dict  # name -> (n, 99) np.ndarray, aligned to target_index
    target_index: pd.DatetimeIndex
    has_ground_truth: bool
    mean_pinball: dict | None = None  # name -> float
    per_hour_pinball: dict | None = None  # name -> np.ndarray (n,)
    coverage_tables: dict | None = None  # name -> DataFrame
    fit_seconds: dict | None = None  # name -> float


def run_task(
    load_dir,
    task_id: int,
    model_specs: list[ModelSpec],
    val_fraction: float = 0.1,
    early_stopping_rounds: int = 50,
    run_oracle_weather: bool = False,) -> TaskResult:
    bundle = load_task(load_dir, task_id)
    valid_hist = bundle.history.dropna(subset=["LOAD"])
    if len(valid_hist) == 0:
        raise ValueError(f"Task {task_id}: no historical rows with observed LOAD -- cannot train.")

    climatology = ClimatologyModel().fit(bundle.history)
    reference_date = bundle.history.index.min()

    X_train = build_feature_frame(valid_hist.index, climatology, reference_date, weather_mode="climatology")
    y_train = valid_hist["LOAD"]
    X_target = build_feature_frame(bundle.target_index, climatology, reference_date, weather_mode="climatology")

    predictions: dict[str, np.ndarray] = {}
    fit_seconds: dict[str, float] = {}

    # -- Baselines --------------------------------------------------------
    predictions["benchmark_official"] = enforce_monotonicity(bundle.benchmark_q.to_numpy())

    t0 = time.time()
    emp_clim = EmpiricalQuantileClimatology().fit(bundle.history)
    predictions["baseline_empirical_climatology"] = enforce_monotonicity(emp_clim.predict(bundle.target_index))
    fit_seconds["baseline_empirical_climatology"] = time.time() - t0

    median_idx = int(np.where(np.isclose(QUANTILE_LEVELS, 0.5))[0][0])
    baseline_median_target = predictions["baseline_empirical_climatology"][:, median_idx]
    baseline_median_train = emp_clim.predict(valid_hist.index)[:, median_idx]
    y_train_residual = pd.Series(y_train.to_numpy() - baseline_median_train, index=y_train.index)
    

    # -- Models -------------------------------------------------------------
    for spec in model_specs:
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = KnotQuantileRegressor(
                family=spec.family, knots=spec.knots, params=spec.params,
                val_fraction=val_fraction, early_stopping_rounds=early_stopping_rounds,
                feature_subset=spec.feature_subset, max_train_rows=spec.max_train_rows,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_target, QUANTILE_LEVELS)
        predictions[spec.name] = enforce_monotonicity(preds)
        fit_seconds[spec.name] = time.time() - t0

        # -- residual variant: same family/params, trained on the residual --
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_res = KnotQuantileRegressor(
                family=spec.family, knots=spec.knots, params=spec.params,
                val_fraction=val_fraction, early_stopping_rounds=early_stopping_rounds,
                feature_subset=spec.feature_subset, max_train_rows=spec.max_train_rows,
            )
            model_res.fit(X_train, y_train_residual)
            preds_res = model_res.predict(X_target, QUANTILE_LEVELS)
        preds_res_shifted = preds_res + baseline_median_target.reshape(-1, 1)
        predictions[f"{spec.name}__residual"] = enforce_monotonicity(preds_res_shifted)
        fit_seconds[f"{spec.name}__residual"] = time.time() - t0
    for spec in model_specs:
        if spec.name in predictions and "baseline_empirical_climatology" in predictions:
            ens = 0.5 * predictions[spec.name] + 0.5 * predictions["baseline_empirical_climatology"]
            predictions[f"{spec.name}__ens_baseline"] = enforce_monotonicity(ens)

    # -- Optional oracle-weather comparison (explicit leakage diagnostic) --
    if run_oracle_weather and bundle.solution_weather is not None:
        X_train_oracle = build_feature_frame(
            valid_hist.index, climatology, reference_date, weather_mode="observed", observed_weather=valid_hist
        )
        X_target_oracle = build_feature_frame(
            bundle.target_index, climatology, reference_date,
            weather_mode="observed", observed_weather=bundle.solution_weather
        )
        for spec in model_specs:
            t0 = time.time()
            oracle_feature_subset = spec.feature_subset
            if spec.family == "linear_qr":
                oracle_feature_subset = LINEAR_MODEL_FEATURES_OBSERVED
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = KnotQuantileRegressor(
                    family=spec.family, knots=spec.knots, params=spec.params,
                    val_fraction=val_fraction, early_stopping_rounds=early_stopping_rounds,
                    feature_subset=oracle_feature_subset, max_train_rows=spec.max_train_rows,
                )
                model.fit(X_train_oracle, y_train)
                preds = model.predict(X_target_oracle, QUANTILE_LEVELS)
            predictions[f"{spec.name}__oracle_weather"] = enforce_monotonicity(preds)
            fit_seconds[f"{spec.name}__oracle_weather"] = time.time() - t0

    result = TaskResult(
        task_id=task_id,
        n_target_hours=len(bundle.target_index),
        predictions=predictions,
        target_index=bundle.target_index,
        has_ground_truth=bundle.solution_load is not None,
        fit_seconds=fit_seconds,
    )

    if bundle.solution_load is not None:
        y_true = bundle.solution_load.to_numpy()
        result.mean_pinball = {}
        result.per_hour_pinball = {}
        result.coverage_tables = {}
        for name, preds in predictions.items():
            result.mean_pinball[name] = mean_pinball_loss(y_true, preds, QUANTILE_LEVELS)
            result.per_hour_pinball[name] = per_hour_pinball_loss(y_true, preds, QUANTILE_LEVELS)
            result.coverage_tables[name] = interval_coverage_table(y_true, preds, QUANTILE_LEVELS)

    return result
