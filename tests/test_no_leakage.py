from pathlib import Path

import numpy as np
import pandas as pd

from gefcom.data_loading import load_task
from gefcom.features import ClimatologyModel, build_feature_frame

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "GEFCom2014-L_V2" / "Load"


def test_history_never_overlaps_target_month():
    """The most basic leakage guarantee: every timestamp in `history` must be
    strictly before every timestamp in `target_index`."""
    bundle = load_task(FIXTURE_DIR, 1)
    assert bundle.history.index.max() < bundle.target_index.min()


def test_train_and_target_feature_columns_are_identical():
    """A model can only be honestly applied to the target month if its
    feature columns mean the same thing there as during training -- this
    guards against the weather-mode mismatch bug (see features.py docstring)
    where target-month climatology columns could silently diverge from
    historical-row columns."""
    bundle = load_task(FIXTURE_DIR, 1)
    clim = ClimatologyModel().fit(bundle.history)
    valid_hist = bundle.history.dropna(subset=["LOAD"])
    ref = bundle.history.index.min()

    train_feats = build_feature_frame(valid_hist.index, clim, ref, weather_mode="climatology")
    target_feats = build_feature_frame(bundle.target_index, clim, ref, weather_mode="climatology")

    assert list(train_feats.columns) == list(target_feats.columns)
    assert not target_feats.isna().any().any(), "target-month features must never be NaN"


def test_trend_feature_is_continuous_across_the_train_target_boundary():
    """Regression test for a real bug: `trend` used to be computed as
    `(index - index.min())` using whatever slice was passed to
    add_calendar_features, which reset to ~0 for every target-month call
    (since target_index's own min is the first hour of that month) instead
    of continuing from where training left off. That silently told every
    model 'this target month is the very start of history' regardless of
    the real calendar year, and is a very plausible explanation for models
    systematically underperforming a baseline that doesn't use `trend` at
    all. `reference_date` must be fixed and shared between the training
    and target calls so trend increases smoothly across that boundary."""
    bundle = load_task(FIXTURE_DIR, 1)
    clim = ClimatologyModel().fit(bundle.history)
    valid_hist = bundle.history.dropna(subset=["LOAD"])
    ref = bundle.history.index.min()

    train_feats = build_feature_frame(valid_hist.index, clim, ref, weather_mode="climatology")
    target_feats = build_feature_frame(bundle.target_index, clim, ref, weather_mode="climatology")

    last_train_trend = train_feats["trend"].iloc[-1]
    first_target_trend = target_feats["trend"].iloc[0]
    # One hour apart in real time -> trend should differ by ~1/(365.25*24),
    # not reset to (near) zero.
    expected_step = 1 / (365.25 * 24)
    assert first_target_trend > last_train_trend
    assert np.isclose(first_target_trend - last_train_trend, expected_step, atol=1e-6)


def test_target_features_do_not_require_target_month_data():
    """Structural leakage guard: building target-month features must not
    require passing in anything about the target month itself (no LOAD, no
    weather) -- only the DatetimeIndex, a fixed reference_date, and a
    climatology fitted on history."""
    bundle = load_task(FIXTURE_DIR, 1)
    clim = ClimatologyModel().fit(bundle.history)
    ref = bundle.history.index.min()
    # Note: no `observed_weather` argument is passed here, and the function
    # signature for weather_mode="climatology" has no path to look at
    # `bundle.solution_load` / `bundle.solution_weather` at all.
    feats = build_feature_frame(bundle.target_index, clim, ref, weather_mode="climatology")
    assert len(feats) == len(bundle.target_index)


def test_climatology_is_a_pure_function_of_the_history_it_is_given():
    """Fitting on the same history twice must give identical tables (i.e.
    nothing external/global leaks into the climatology fit), and a
    climatology fitted on a shorter history must not contain groups for
    months it never saw."""
    bundle = load_task(FIXTURE_DIR, 1)
    clim_a = ClimatologyModel().fit(bundle.history)
    clim_b = ClimatologyModel().fit(bundle.history)
    pd.testing.assert_frame_equal(clim_a.load_table_, clim_b.load_table_)

    # Target month for this fixture is February; restrict history to only
    # January and confirm no February group appears in the fitted table.
    jan_only = bundle.history.loc[bundle.history.index.month == 1]
    clim_jan = ClimatologyModel().fit(jan_only)
    assert not (clim_jan.load_table_["month"] == 2).any()


def test_oracle_weather_mode_requires_explicit_observed_weather():
    """The leakage-diagnostic 'observed' weather mode must fail loudly if
    called without real weather data, rather than silently falling back to
    climatology (which would make the oracle/default comparison meaningless)."""
    import pytest

    bundle = load_task(FIXTURE_DIR, 1)
    clim = ClimatologyModel().fit(bundle.history)
    ref = bundle.history.index.min()
    with pytest.raises(ValueError):
        build_feature_frame(bundle.target_index, clim, ref, weather_mode="observed")
