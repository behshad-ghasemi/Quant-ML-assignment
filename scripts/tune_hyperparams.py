#!/usr/bin/env python3
"""Hyperparameter tuning for the LightGBM / XGBoost quantile models.

Leakage-safety design: tuning is done using ONLY the historical window of
a single, early "tuning task" (Task 1 by default), by carving several
internal expanding-window mini-folds out of that history alone (e.g. the
last few months of Task 1's history are held out one at a time). We never
look at the real target months (Task N's actual forecast month) during
tuning, and the tuning task is excluded from the set of tasks used for
final backtest reporting by default -- this keeps hyperparameter
selection from leaking into the numbers reported in the README.

Performance note (why this version is much faster than a naive one):
feature engineering (ClimatologyModel.fit + build_feature_frame) depends
only on `train_hist`/`target_index`, which are FIXED per fold -- they do
NOT depend on the hyperparameters being trialled. A naive implementation
that rebuilds features inside the Optuna objective recomputes the same
climatology tables and feature frames on every single trial (e.g. 30x for
--n-trials 30), which is wasted work that grows with history size. Here
we build each fold's features exactly once, before optimization starts,
and every trial just refits the (cheap) model on the already-built
arrays.

Usage:
    python scripts/tune_hyperparams.py --config configs/config.yaml --family lightgbm --n-trials 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gefcom.data_loading import load_task  # noqa: E402
from gefcom.features import ClimatologyModel, build_feature_frame  # noqa: E402
from gefcom.metrics import QUANTILE_LEVELS, enforce_monotonicity, mean_pinball_loss  # noqa: E402
from gefcom.quantile_models import DEFAULT_KNOTS, KnotQuantileRegressor  # noqa: E402
from gefcom.pipeline import GBM_MODEL_FEATURES  # noqa: E402   <-- add

optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_internal_folds(history: pd.DataFrame, n_folds: int = 3, fold_months: int = 1):
    """Build `n_folds` expanding-window (train, held_out_month) slices from
    the tail of `history`, oldest-fold-first. Each held-out slice is one
    calendar month, mimicking the real task structure."""
    valid = history.dropna(subset=["LOAD"])
    month_counts = valid.index.to_period("M").value_counts()
    months = sorted(m for m, c in month_counts.items() if c >= 100)
    if len(months) < n_folds + 6:
        raise ValueError("Not enough history in the tuning task to build internal folds.")
    holdout_months = months[-n_folds:]
    folds = []
    for m in holdout_months:
        month_start = m.start_time
        train_hist = history.loc[history.index < month_start]
        target_slice = valid.loc[valid.index.to_period("M") == m]
        if len(target_slice) == 0:
            continue
        folds.append((train_hist, target_slice.index, target_slice["LOAD"]))
    return folds


def precompute_fold_features(folds, feature_subset=None):
    """Build climatology + feature frames ONCE per fold, up front. This is
    the main speedup vs. rebuilding them inside the Optuna objective (see
    module docstring) -- every trial then just reuses these arrays."""
    prepared = []
    for train_hist, target_index, y_true in folds:
        climatology = ClimatologyModel().fit(train_hist)
        valid_hist = train_hist.dropna(subset=["LOAD"])
        reference_date = train_hist.index.min()
        X_train = build_feature_frame(valid_hist.index, climatology, reference_date, weather_mode="climatology")
        y_train = valid_hist["LOAD"]
        X_target = build_feature_frame(target_index, climatology, reference_date, weather_mode="climatology")
        if feature_subset is not None:
            X_train = X_train[feature_subset]
            X_target = X_target[feature_subset]
        prepared.append((X_train, y_train, X_target, y_true.to_numpy()))
    return prepared


def evaluate_params(family: str, params: dict, prepared_folds, knots, max_train_rows=None) -> float:
    losses = []
    for X_train, y_train, X_target, y_true in prepared_folds:
        model = KnotQuantileRegressor(
            family=family, knots=knots, params=params, val_fraction=0.1,
            early_stopping_rounds=30, feature_subset=None, max_train_rows=max_train_rows,
        )
        model.fit(X_train, y_train)
        preds = enforce_monotonicity(model.predict(X_target, QUANTILE_LEVELS))
        losses.append(mean_pinball_loss(y_true, preds, QUANTILE_LEVELS))
    return float(np.mean(losses))


def suggest_params(trial: "optuna.Trial", family: str) -> dict:
    if family == "lightgbm":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 63),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
    if family == "xgboost":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
    raise ValueError(family)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--family", choices=["lightgbm", "xgboost"], required=True)
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--tuning-knots", type=str, default="0.1,0.5,0.9",
                     help="Comma-separated quantile knots to evaluate during tuning "
                          "(a small subset is enough to rank hyperparameters and keeps tuning fast; "
                          "the full knot set from config is still used for the final backtest).")
    ap.add_argument("--max-train-rows", type=int, default=20000,
                     help="Subsample training rows per fold during tuning for speed "
                          "(systematic/deterministic subsample, same as the real pipeline's "
                          "max_train_rows option -- ranking hyperparameters doesn't need the "
                          "full multi-year history on every trial). Use 0 to disable.")
    ap.add_argument("--n-jobs", type=int, default=1,
                     help="Optuna trials to run in parallel. Keep model n_jobs=1 in "
                          "configs/config.yaml if you raise this, to avoid oversubscribing CPUs.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    load_dir = cfg["paths"]["load_dir"]
    tuning_task_id = cfg.get("tuning", {}).get("tuning_task_id", 1)
    n_folds = cfg.get("tuning", {}).get("n_internal_folds", 3)
    max_train_rows = args.max_train_rows if args.max_train_rows > 0 else None

    bundle = load_task(load_dir, tuning_task_id)
    folds = make_internal_folds(bundle.history, n_folds=n_folds)
    print(f"Built {len(folds)} internal folds from Task {tuning_task_id}'s history "
          f"(held-out months: {[str(f[1][0].to_period('M')) for f in folds]})")

    t0 = time.time()
    prepared_folds = precompute_fold_features(folds, feature_subset=GBM_MODEL_FEATURES)
    print(f"Precomputed features for all folds once in {time.time()-t0:.1f}s "
          f"(this used to be redone on every trial -- the main speedup here)")
    if max_train_rows:
        print(f"Subsampling each fold's training rows to <= {max_train_rows} for tuning speed "
              f"(final backtest in run_backtest.py is unaffected -- this only limits tuning)")

    knots = np.array([float(x) for x in args.tuning_knots.split(",")])

    def objective(trial: "optuna.Trial") -> float:
        params = suggest_params(trial, args.family)
        return evaluate_params(args.family, params, prepared_folds, knots, max_train_rows=max_train_rows)

    study = optuna.create_study(direction="minimize")
    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False, n_jobs=args.n_jobs)
    print(f"\nRan {args.n_trials} trials in {time.time()-t0:.1f}s")

    print(f"\nBest {args.family} params (mean pinball loss = {study.best_value:.4f}):")
    print(json.dumps(study.best_params, indent=2))

    out_dir = Path(cfg["paths"].get("output_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"best_params_{args.family}.json"
    out_path.write_text(json.dumps(study.best_params, indent=2))
    print(f"\nSaved -> {out_path}")
    print("Copy these into configs/config.yaml under models.<family> to use them in run_backtest.py.")


if __name__ == "__main__":
    main()