#!/usr/bin/env python3
"""Multi-fold internal backtest -- gets you a statistically meaningful
comparison of baseline vs. models WITHOUT needing official Solution files.

Motivation
----------
The real backtest (run_backtest.py) can only be *scored* on tasks that
have a Kaggle Solution file. You cannot conclude "model X is worse than
baseline Y" from a single held-out month; the variance across months
(holidays, weather regimes, etc.) is large enough that one month is not
representative.

This script reuses the exact same "hold out trailing month(s) from a
task's own *training* history" trick that tune_hyperparams.py already
uses for leakage-safe hyperparameter search -- except here we use it for
MODEL COMPARISON, not tuning, and we pull folds from every task's history
(not just one tuning task). Every fold's "true" LOAD is data you already
have (it's inside the train CSV), so no Solution file is required at all.

Usage:
    python scripts/internal_multi_fold_backtest.py --config configs/config.yaml \
        --tasks 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --n-folds 6

Output:
    outputs/internal_backtest_by_fold.csv   (one row per task x fold x model)
    outputs/internal_backtest_summary.csv   (mean/std/count per model, n>>1)
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gefcom.features import ClimatologyModel, build_feature_frame  # noqa: E402
from gefcom.baselines import EmpiricalQuantileClimatology  # noqa: E402
from gefcom.data_loading import load_task  # noqa: E402
from gefcom.pipeline import LINEAR_MODEL_FEATURES, GBM_MODEL_FEATURES  # noqa: E402
from gefcom.features import ClimatologyModel, build_feature_frame  # noqa: E402
from gefcom.metrics import QUANTILE_LEVELS, enforce_monotonicity, mean_pinball_loss, per_hour_pinball_loss  # noqa: E402
from gefcom.quantile_models import DEFAULT_KNOTS, KnotQuantileRegressor  # noqa: E402
from gefcom.stats_tests import diebold_mariano_test  # noqa: E402



def make_internal_folds(history: pd.DataFrame, n_folds: int, fold_months: int = 1):
    """Same idea as tune_hyperparams.make_internal_folds: carve `n_folds`
    expanding-window (train_hist, held_out_index, held_out_y) slices from
    the tail of `history`, oldest-fold-first. Skips folds that don't have
    enough real LOAD data to be meaningful.

    NOTE: `train_hist` is always a *prefix* slice of the original `history`
    (`history.index < month_start`), so `train_hist.index.min()` is the
    same fixed point for every fold of a given task -- this matters
    because that's exactly the value used as `reference_date` below (see
    features.py docstring on the trend-continuity bug)."""
    valid = history.dropna(subset=["LOAD"])
    # A train file's last row is the hour-24 rollover into the next
    # calendar date, which creates a phantom 1-row "next month" period
    # that would otherwise silently eat one of the requested n_folds
    # slots without being replaced. Filter months with too few rows out
    # BEFORE slicing to the last n_folds.
    month_counts = valid.index.to_period("M").value_counts()
    months = sorted(m for m, c in month_counts.items() if c >= 100)
    if len(months) < n_folds + 6:
        n_folds = max(0, len(months) - 6)
    holdout_months = months[-n_folds:] if n_folds > 0 else []
    folds = []
    for m in holdout_months:
        month_start = m.start_time
        train_hist = history.loc[history.index < month_start]
        target_slice = valid.loc[valid.index.to_period("M") == m]
        if len(target_slice) < 100:  # skip partial/tiny months
            continue
        if train_hist.dropna(subset=["LOAD"]).shape[0] < 500:  # need enough to train on
            continue
        folds.append((train_hist, target_slice.index, target_slice["LOAD"]))
    return folds


def build_model_specs(cfg: dict, knots: np.ndarray):
    specs = []
    models_cfg = cfg.get("models", {})
    if models_cfg.get("linear_qr", {}).get("enabled", True):
        p = models_cfg.get("linear_qr", {})
        specs.append(dict(name="linear_qr", family="linear_qr", knots=knots,
                           params={"l1_reg": p.get("l1_reg", 0.001)},
                           feature_subset=LINEAR_MODEL_FEATURES,
                           max_train_rows=p.get("max_train_rows", 4000)))
    if models_cfg.get("lightgbm", {}).get("enabled", True):
        specs.append(dict(name="lightgbm", family="lightgbm", knots=knots,
                           params=models_cfg.get("lightgbm", {}), feature_subset=GBM_MODEL_FEATURES, max_train_rows=None))
    if models_cfg.get("xgboost", {}).get("enabled", True):
        specs.append(dict(name="xgboost", family="xgboost", knots=knots,
                           params=models_cfg.get("xgboost", {}), feature_subset=GBM_MODEL_FEATURES, max_train_rows=None))
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--tasks", type=str, default=",".join(str(i) for i in range(1, 16)))
    ap.add_argument("--n-folds", type=int, default=6, help="held-out months per task")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    load_dir = cfg["paths"]["load_dir"]
    output_dir = Path(cfg["paths"].get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    task_ids = [int(t) for t in args.tasks.split(",")]
    knots = np.array(cfg.get("quantile_knots", None) or DEFAULT_KNOTS)
    specs = build_model_specs(cfg, knots)
    print(f"Models: {[s['name'] for s in specs]}")
    print(f"Tasks: {task_ids}  |  folds/task: {args.n_folds}")

    rows = []
    per_hour_loss: dict[str, list[np.ndarray]] = {}

    for task_id in task_ids:
        try:
            bundle = load_task(load_dir, task_id)
        except FileNotFoundError as e:
            print(f"  SKIPPING task {task_id}: {e}")
            continue

        folds = make_internal_folds(bundle.history, n_folds=args.n_folds)
        if not folds:
            print(f"  Task {task_id}: not enough history for internal folds, skipping")
            continue

        for fi, (train_hist, held_index, y_true) in enumerate(folds):
            t0 = time.time()
            climatology = ClimatologyModel().fit(train_hist)
            valid_hist = train_hist.dropna(subset=["LOAD"])
            reference_date = train_hist.index.min()
            X_train = build_feature_frame(valid_hist.index, climatology, reference_date, weather_mode="climatology")
            y_train = valid_hist["LOAD"]
            X_held = build_feature_frame(held_index, climatology, reference_date, weather_mode="climatology")

            y_true_arr = y_true.to_numpy()

            # -- baseline --
            emp_clim = EmpiricalQuantileClimatology().fit(train_hist)
            preds = enforce_monotonicity(emp_clim.predict(held_index))
            loss = mean_pinball_loss(y_true_arr, preds, QUANTILE_LEVELS)
            rows.append({"task_id": task_id, "fold": fi, "model": "baseline_empirical_climatology",
                         "mean_pinball_loss": loss, "held_month": str(held_index[0].to_period("M"))})
            per_hour_loss.setdefault("baseline_empirical_climatology", []).append(
                per_hour_pinball_loss(y_true_arr, preds, QUANTILE_LEVELS))

            # -- models --
            for spec in specs:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = KnotQuantileRegressor(
                        family=spec["family"], knots=spec["knots"], params=spec["params"],
                        val_fraction=args.val_fraction, early_stopping_rounds=args.early_stopping_rounds,
                        feature_subset=spec["feature_subset"], max_train_rows=spec["max_train_rows"],
                    )
                    model.fit(X_train, y_train)
                    preds = enforce_monotonicity(model.predict(X_held, QUANTILE_LEVELS))
                loss = mean_pinball_loss(y_true_arr, preds, QUANTILE_LEVELS)
                rows.append({"task_id": task_id, "fold": fi, "model": spec["name"],
                             "mean_pinball_loss": loss, "held_month": str(held_index[0].to_period("M"))})
                per_hour_loss.setdefault(spec["name"], []).append(
                    per_hour_pinball_loss(y_true_arr, preds, QUANTILE_LEVELS))

            print(f"  task {task_id} fold {fi} ({held_index[0].to_period('M')}): "
                  f"done in {time.time()-t0:.1f}s")

    if not rows:
        print("No folds could be built -- not enough history anywhere.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "internal_backtest_by_fold.csv", index=False)

    summary = df.groupby("model")["mean_pinball_loss"].agg(["mean", "std", "count"]).sort_values("mean")
    summary.to_csv(output_dir / "internal_backtest_summary.csv")
    n_pairs = df[["task_id", "fold"]].drop_duplicates().shape[0]
    print(f"\n=== Mean pinball loss across {n_pairs} (task, fold) pairs ===")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    # -- DM tests: every model vs. baseline, on the pooled per-hour series --
    baseline_key = "baseline_empirical_climatology"
    if baseline_key in per_hour_loss:
        loss_base = np.concatenate(per_hour_loss[baseline_key])
        print("\n=== Diebold-Mariano tests vs. baseline_empirical_climatology (pooled across all folds) ===")
        for name in per_hour_loss:
            if name == baseline_key:
                continue
            loss_model = np.concatenate(per_hour_loss[name])
            if len(loss_model) != len(loss_base):
                continue
            res = diebold_mariano_test(loss_model, loss_base, name, baseline_key)
            print(f"  {name:12s} vs {baseline_key}: {res}")

    print(f"\nSaved -> {output_dir/'internal_backtest_by_fold.csv'}")
    print(f"Saved -> {output_dir/'internal_backtest_summary.csv'}")


if __name__ == "__main__":
    main()
