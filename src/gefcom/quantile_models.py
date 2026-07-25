"""Quantile-regression model families, all exposed through one interface.

Compute-budget design decision: fitting a separate model for every one of
the 99 required quantiles, for every one of 15 backtest tasks, for every
model family, is unnecessary and slow on a laptop CPU (the assignment
explicitly says "unnecessary complexity will not receive additional
credit" and requires the solution to run on a normal laptop). Instead we
fit models at a smaller set of quantile "knots" (23 by default, including
both tails explicitly so we never extrapolate past a fitted quantile) and
linearly interpolate the remaining quantiles between knots. This cuts the
model count by roughly 4x with negligible loss of accuracy, since the
underlying conditional distribution is smooth.

Three families are implemented behind the same interface:
  - "linear_qr"  : sklearn QuantileRegressor (linear pinball-loss
                   regression) -- the classical / interpretable model.
  - "lightgbm"   : LightGBM with objective="quantile" -- primary model.
  - "xgboost"    : XGBoost with objective="reg:quantileerror" -- secondary
                   comparison model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_KNOTS = np.array(
    [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
     0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99]
)


def _make_estimator(family: str, alpha: float, params: dict):
    if family == "linear_qr":
        from sklearn.linear_model import QuantileRegressor

        return QuantileRegressor(
            quantile=alpha,
            alpha=params.get("l1_reg", 0.001),
            solver="highs",
        )
    if family == "lightgbm":
        import lightgbm as lgb

        p = dict(
            objective="quantile",
            alpha=alpha,
            n_estimators=params.get("n_estimators", 400),
            learning_rate=params.get("learning_rate", 0.05),
            num_leaves=params.get("num_leaves", 31),
            min_child_samples=params.get("min_child_samples", 30),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            reg_lambda=params.get("reg_lambda", 0.1),
            random_state=params.get("random_state", 0),
            n_jobs=params.get("n_jobs", -1),
            verbosity=-1,
        )
        return lgb.LGBMRegressor(**p)
    if family == "xgboost":
        import xgboost as xgb

        p = dict(
            objective="reg:quantileerror",
            quantile_alpha=alpha,
            n_estimators=params.get("n_estimators", 400),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 5),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            reg_lambda=params.get("reg_lambda", 1.0),
            random_state=params.get("random_state", 0),
            n_jobs=params.get("n_jobs", -1),
        )
        return xgb.XGBRegressor(**p)
    raise ValueError(f"Unknown model family: {family!r}")


def _systematic_subsample(X: np.ndarray, y: np.ndarray, max_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic every-k-th-row subsample that preserves the
    chronological spread of the data (unlike random subsampling, this
    keeps seasonal/time-of-day coverage even and is fully reproducible)."""
    n = len(y)
    if n <= max_rows:
        return X, y
    step = n / max_rows
    idx = (np.arange(max_rows) * step).astype(int)
    idx = np.clip(idx, 0, n - 1)
    return X[idx], y[idx]


def _supports_early_stopping(family: str) -> bool:
    return family in ("lightgbm", "xgboost")


@dataclass
class KnotQuantileRegressor:
    """Fits one model per knot quantile and linearly interpolates the rest.

    `family`: "linear_qr" | "lightgbm" | "xgboost"
    `knots`: sorted array of quantile levels to fit directly.
    `params`: hyperparameters passed to the underlying estimator.
    `val_fraction`: fraction of the (chronologically-ordered) training
        rows held out as a trailing validation slice for early stopping
        (lightgbm/xgboost only; ignored for linear_qr). This validation
        slice is carved from the *training* period only -- it never
        touches the target month.
    """

    family: str
    knots: np.ndarray = field(default_factory=lambda: DEFAULT_KNOTS)
    params: dict = field(default_factory=dict)
    val_fraction: float = 0.1
    early_stopping_rounds: int = 50
    feature_subset: list | None = None
    max_train_rows: int | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "KnotQuantileRegressor":
        if self.feature_subset is not None:
            X = X[self.feature_subset]
        self.feature_names_ = list(X.columns)
        X_arr = X.to_numpy(dtype=float)
        y_arr = y.to_numpy(dtype=float)
        n = len(y_arr)

        use_val = _supports_early_stopping(self.family) and self.val_fraction > 0 and n > 200
        if use_val:
            n_val = max(1, int(n * self.val_fraction))
            X_tr, y_tr = X_arr[:-n_val], y_arr[:-n_val]
            X_val, y_val = X_arr[-n_val:], y_arr[-n_val:]
        else:
            X_tr, y_tr = X_arr, y_arr
            X_val = y_val = None

        if self.max_train_rows is not None:
            X_tr, y_tr = _systematic_subsample(X_tr, y_tr, self.max_train_rows)

        self.models_ = {}
        for alpha in self.knots:
            model = _make_estimator(self.family, float(alpha), self.params)
            if use_val:
                if self.family == "lightgbm":
                    import lightgbm as lgb

                    model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
                    )
                elif self.family == "xgboost":
                    model.set_params(early_stopping_rounds=self.early_stopping_rounds)
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            else:
                model.fit(X_tr, y_tr)
            self.models_[float(alpha)] = model
        return self

    def predict_knots(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = X[self.feature_names_].to_numpy(dtype=float)
        preds = np.column_stack([self.models_[float(a)].predict(X_arr) for a in self.knots])
        return np.sort(preds, axis=1)  # guard against knot-level crossing

    def predict(self, X: pd.DataFrame, target_quantiles: np.ndarray) -> np.ndarray:
        """Interpolate from the fitted knots to `target_quantiles` (e.g. the
        full 1..99 grid). Because both tails (0.01, 0.99 by default) are
        fitted knots, this never needs to extrapolate beyond the fitted
        range for the standard GEFCom2014 quantile grid."""
        knot_preds = self.predict_knots(X)  # (n, n_knots)
        out = np.empty((knot_preds.shape[0], len(target_quantiles)))
        for i in range(knot_preds.shape[0]):
            out[i] = np.interp(target_quantiles, self.knots, knot_preds[i])
        return np.sort(out, axis=1)
