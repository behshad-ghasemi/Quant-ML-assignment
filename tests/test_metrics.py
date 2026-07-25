import numpy as np

from gefcom.metrics import (
    coverage,
    enforce_monotonicity,
    interval_width,
    mean_pinball_loss,
    per_hour_pinball_loss,
    pinball_loss,
)


def test_pinball_loss_matches_hand_calculation():
    # y_true=10, single quantile q=0.9, pred=8 (under-prediction)
    # loss = q * (y - pred) = 0.9 * 2 = 1.8
    y_true = np.array([10.0])
    preds = np.array([[8.0]])
    q = np.array([0.9])
    loss = pinball_loss(y_true, preds, q)
    assert np.isclose(loss[0, 0], 1.8)

    # over-prediction: y_true=10, pred=12, q=0.9 -> loss = (q-1)*(y-pred) = -0.1*-2 = 0.2
    preds2 = np.array([[12.0]])
    loss2 = pinball_loss(y_true, preds2, q)
    assert np.isclose(loss2[0, 0], 0.2)


def test_pinball_loss_is_zero_for_perfect_point_prediction():
    y_true = np.array([5.0, 5.0, 5.0])
    preds = np.tile(5.0, (3, 5))
    q = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    assert mean_pinball_loss(y_true, preds, q) == 0.0


def test_median_quantile_pinball_loss_is_half_absolute_error():
    y_true = np.array([10.0])
    preds = np.array([[6.0]])
    q = np.array([0.5])
    loss = pinball_loss(y_true, preds, q)
    assert np.isclose(loss[0, 0], 0.5 * abs(10 - 6))


def test_per_hour_pinball_loss_shape():
    y_true = np.random.RandomState(0).normal(100, 10, size=50)
    preds = np.tile(y_true.reshape(-1, 1), (1, 10)) + np.random.RandomState(1).normal(0, 5, size=(50, 10))
    q = np.linspace(0.1, 0.9, 10)
    out = per_hour_pinball_loss(y_true, preds, q)
    assert out.shape == (50,)
    assert (out >= 0).all()


def test_enforce_monotonicity_sorts_crossing_quantiles():
    preds = np.array([[5.0, 3.0, 8.0, 1.0]])  # deliberately not monotonic
    fixed = enforce_monotonicity(preds)
    assert (np.diff(fixed, axis=1) >= 0).all()
    assert sorted(preds[0].tolist()) == fixed[0].tolist()


def test_coverage_and_interval_width():
    q = np.array([0.05, 0.5, 0.95])
    y_true = np.array([10.0, 10.0, 10.0, 100.0])  # last one way outside the interval
    preds = np.tile(np.array([[8.0, 10.0, 12.0]]), (4, 1))
    cov = coverage(y_true, preds, q, 0.05, 0.95)
    assert np.isclose(cov, 0.75)  # 3/4 inside [8, 12]
    width = interval_width(preds, q, 0.05, 0.95)
    assert np.isclose(width, 4.0)
