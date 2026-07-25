from pathlib import Path

import numpy as np

from gefcom.pipeline import LINEAR_MODEL_FEATURES, ModelSpec, run_task

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "GEFCom2014-L_V2" / "Load"


def test_run_task_end_to_end_on_fixture():
    specs = [
        ModelSpec(name="linear_qr", family="linear_qr", knots=np.array([0.1, 0.5, 0.9]),
                  feature_subset=LINEAR_MODEL_FEATURES),
        ModelSpec(name="lightgbm", family="lightgbm", knots=np.array([0.1, 0.5, 0.9]),
                  params={"n_estimators": 20}),
    ]
    result = run_task(FIXTURE_DIR, 1, specs, val_fraction=0.1, early_stopping_rounds=5)

    assert result.has_ground_truth
    expected_models = {"benchmark_official", "baseline_empirical_climatology", "linear_qr", "lightgbm"}
    assert expected_models.issubset(result.predictions.keys())

    for name, preds in result.predictions.items():
        assert preds.shape == (result.n_target_hours, 99)
        assert (np.diff(preds, axis=1) >= -1e-9).all(), f"{name} predictions are not monotonic"
        assert not np.isnan(preds).any(), f"{name} predictions contain NaN"

    assert result.mean_pinball is not None
    for name, loss in result.mean_pinball.items():
        assert loss >= 0
        assert np.isfinite(loss)

    # Our fitted models should not be dramatically worse than the flat
    # official benchmark on this easy synthetic (pure sinusoid) series.
    assert result.mean_pinball["lightgbm"] < result.mean_pinball["benchmark_official"] * 2


def test_run_task_oracle_weather_comparison_uses_matching_feature_subset():
    """Regression test: the linear model's curated feature subset is
    defined in terms of climatology-mode column names (e.g.
    'clim_w_mean_mean'). Running the oracle-weather diagnostic (which uses
    weather_mode='observed', with different column names like 'w_mean')
    must not silently reuse that subset -- it previously raised a KeyError."""
    specs = [
        ModelSpec(name="linear_qr", family="linear_qr", knots=np.array([0.1, 0.5, 0.9]),
                  feature_subset=LINEAR_MODEL_FEATURES),
        ModelSpec(name="lightgbm", family="lightgbm", knots=np.array([0.1, 0.5, 0.9]),
                  params={"n_estimators": 20}),
    ]
    result = run_task(FIXTURE_DIR, 1, specs, val_fraction=0.1, early_stopping_rounds=5,
                       run_oracle_weather=True)
    assert "linear_qr__oracle_weather" in result.predictions
    assert "lightgbm__oracle_weather" in result.predictions
    preds = result.predictions["linear_qr__oracle_weather"]
    assert preds.shape == (result.n_target_hours, 99)
    assert not np.isnan(preds).any()
