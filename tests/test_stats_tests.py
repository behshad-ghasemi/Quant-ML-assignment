import numpy as np

from gefcom.stats_tests import diebold_mariano_test


def test_dm_test_detects_clearly_better_model():
    rng = np.random.RandomState(0)
    n = 2000
    # model_a has systematically lower loss than model_b
    loss_a = rng.gamma(shape=2.0, scale=1.0, size=n)
    loss_b = loss_a + rng.gamma(shape=2.0, scale=1.0, size=n) + 0.5
    res = diebold_mariano_test(loss_a, loss_b, "A", "B")
    assert res.mean_loss_diff < 0
    assert res.p_value < 0.01
    assert "A" in res.better_model


def test_dm_test_null_case_gives_high_p_value():
    rng = np.random.RandomState(1)
    n = 2000
    loss_a = rng.gamma(shape=2.0, scale=1.0, size=n)
    loss_b = rng.gamma(shape=2.0, scale=1.0, size=n)  # same distribution, independent
    res = diebold_mariano_test(loss_a, loss_b, "A", "B")
    assert res.p_value > 0.05


def test_dm_test_is_antisymmetric():
    rng = np.random.RandomState(2)
    loss_a = rng.gamma(2.0, 1.0, size=500)
    loss_b = rng.gamma(2.0, 1.0, size=500) + 0.3
    res_ab = diebold_mariano_test(loss_a, loss_b, "A", "B")
    res_ba = diebold_mariano_test(loss_b, loss_a, "B", "A")
    assert np.isclose(res_ab.dm_stat, -res_ba.dm_stat)
    assert np.isclose(res_ab.p_value, res_ba.p_value)


def test_dm_test_accounts_for_autocorrelation():
    """A naive i.i.d. variance estimate would understate the true variance
    of an autocorrelated loss-differential series and over-reject. Compare
    the HAC-based p-value against the (wrong) naive normal p-value on a
    strongly autocorrelated series with no true difference -- HAC should be
    noticeably more conservative (larger p-value)."""
    rng = np.random.RandomState(3)
    n = 3000
    innovations = rng.normal(0, 1, n)
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.8 * ar[t - 1] + innovations[t]
    loss_a = 5 + ar
    loss_b = 5 + rng.permutation(ar)  # same marginal, breaks the AR structure

    res_hac = diebold_mariano_test(loss_a, loss_b, "A", "B", small_sample_correction=False)
    from scipy import stats as _stats
    naive_var = np.var(loss_a - loss_b, ddof=1) / n
    naive_stat = (loss_a - loss_b).mean() / np.sqrt(naive_var)
    naive_p = 2 * (1 - _stats.norm.cdf(np.abs(naive_stat)))

    assert res_hac.p_value >= naive_p
