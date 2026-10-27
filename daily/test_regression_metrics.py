"""Tests for regression_metrics. Run: python test_regression_metrics.py (or pytest)."""
import numpy as np

from regression_metrics import interval_metrics, point_metrics, rolling_bands


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = 2000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    y = rng.normal(0.0003, 0.01, n)
    return close, close * np.exp(y), y


def test_no_change_forecast_has_zero_skill():
    close, nxt, y = _toy()
    m = point_metrics(close, nxt, y, np.zeros_like(y), np.full_like(y, y.mean()))
    assert abs(m["mae_skill_vs_naive"]) < 1e-12 and abs(m["rmse_skill_vs_naive"]) < 1e-12
    assert abs(m["theil_u"] - 1) < 1e-12 and abs(m["r2_os_vs_zero"]) < 1e-12


def test_perfect_forecast():
    close, nxt, y = _toy()
    m = point_metrics(close, nxt, y, y, np.zeros_like(y))
    assert m["mape_pct"] < 1e-9 and m["rmse_usd"] < 1e-6
    assert abs(m["r2_os_vs_zero"] - 1) < 1e-12 and abs(m["ic_spearman"] - 1) < 1e-9


def test_known_values_by_hand():
    close = np.array([100.0, 200.0, 100.0, 200.0])
    nxt = np.array([110.0, 190.0, 90.0, 210.0])
    forecast = np.array([105.0, 200.0, 95.0, 205.0])  # absolute price errors 5, 10, 5, 5; the no-change errors are all 10
    y, pred = np.log(nxt / close), np.log(forecast / close)
    m = point_metrics(close, nxt, y, pred, np.zeros(4))
    assert abs(m["mae_usd"] - 6.25) < 1e-9
    assert abs(m["rmse_usd"] - np.sqrt(43.75)) < 1e-9
    assert abs(m["mape_pct"] - (5 / 110 + 10 / 190 + 5 / 90 + 5 / 210) / 4 * 100) < 1e-9
    assert abs(m["mae_skill_vs_naive"] - (1 - 6.25 / 10)) < 1e-9


def test_bands_use_only_past_errors():
    rng = np.random.default_rng(1)
    err, pred = rng.normal(0, 0.01, 300), np.zeros(300)
    lo, hi = rolling_bands(pred, err, "ewma_gaussian")
    corrupted = err.copy()
    corrupted[200:] = 5.0  # change the future only
    lo2, hi2 = rolling_bands(pred, corrupted, "ewma_gaussian")
    k = 200 - 60  # forecast days 60..199 depend on errors[:200] at most
    assert np.allclose(lo[:k], lo2[:k]) and np.allclose(hi[:k], hi2[:k])
    assert not np.allclose(lo[k:], lo2[k:])  # the check can fail: once the change is in the past, the bands move


def test_ewma_band_covers_about_90_percent_on_iid_errors():
    rng = np.random.default_rng(2)
    err = rng.normal(0, 0.012, 4000)
    lo, hi = rolling_bands(np.zeros(4000), err, "ewma_gaussian")
    assert 0.87 < interval_metrics(np.full(4000, 2000.0), err, lo, hi)["coverage"] < 0.93


def test_band_from_a_calm_past_undercovers_when_volatility_rises():
    rng = np.random.default_rng(3)
    err = np.r_[rng.normal(0, 0.005, 300), rng.normal(0, 0.012, 300)]
    close = np.full(600, 2000.0)
    cov = {}
    for method in ("empirical", "ewma_gaussian"):
        lo, hi = rolling_bands(np.zeros(600), err, method)
        cov[method] = interval_metrics(close[-250:], err[-250:], lo[-250:], hi[-250:])["coverage"]
    assert cov["empirical"] < 0.85 and cov["ewma_gaussian"] > cov["empirical"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
