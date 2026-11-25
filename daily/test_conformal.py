"""Tests for the E7 conformal-interval experiment (run: python test_conformal.py). Synthetic data only: no network."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import conformal_experiment as ce  # noqa: E402
from regression_metrics import ewma_sigma  # noqa: E402


def test_sigma_path_equals_the_shipped_ewma_sigma():
    e = np.random.default_rng(0).normal(0, 0.01, 300)
    s = ce.sigma_path(e)
    for i in (1, 5, 60, 200, 299):
        assert abs(s[i] - ewma_sigma(e[:i])) < 1e-12, i
    assert np.isnan(s[0])


def test_conformal_rank_is_the_finite_sample_quantile():
    assert ce.conformal_rank(np.arange(1, 101, dtype=float), 0.1) == 91          # ceil(101 * 0.9) = 91st smallest
    assert ce.conformal_rank(np.array([3.0, 1.0, 2.0]), 0.01) == 3.0              # rank above n: the largest score
    assert ce.conformal_rank(np.arange(1, 11, dtype=float), 0.5) == 6              # ceil(11 * 0.5) = 6


def test_split_conformal_covers_about_ninety_percent_on_exchangeable_errors():
    rng = np.random.default_rng(1)
    n = 4000
    y = rng.normal(0, 0.01, n); c = np.zeros(n)
    lo, hi = ce.make_bands(y, c, ce.sigma_path(y - c), "scp_raw")
    ok = np.isfinite(lo)
    assert abs(((y >= lo) & (y <= hi))[ok].mean() - 0.9) < 0.02


def test_aci_restores_coverage_after_a_variance_shift_where_a_long_memory_conformal_band_does_not():
    rng = np.random.default_rng(2)
    n = 6000
    y = rng.normal(0, 0.01, n); y[2500:] *= 2.5; c = np.zeros(n)
    sig = np.full(n, 0.01)                                                       # a volatility forecast that never adapts
    inside = lambda kind, g: (lambda b: ((y >= b[0]) & (y <= b[1]))[3000:].mean())(ce.make_bands(y, c, sig, kind, g, win=10_000))
    scp, aci = inside("scp_norm", None), inside("aci_norm", 0.02)
    assert scp < 0.80, scp                                                       # the calibration set is dominated by the calm past
    assert abs(aci - 0.9) < 0.04, aci                                            # ACI steers the long-run coverage back to 90%


def test_bands_never_look_ahead():
    rng = np.random.default_rng(3)
    n = 400
    y = rng.normal(0, 0.01, n); c = rng.normal(0, 0.002, n)
    y2 = y.copy(); y2[250:] = 5.0                                                # change the outcome of origin 250 and everything after it
    for kind, g in (("gauss", None), ("scp_raw", None), ("scp_norm", None), ("aci_norm", 0.02)):
        a = ce.make_bands(y, c, ce.sigma_path(y - c), kind, g)
        b = ce.make_bands(y2, c, ce.sigma_path(y2 - c), kind, g)
        assert np.allclose(a[0][:251], b[0][:251], equal_nan=True) and np.allclose(a[1][:251], b[1][:251], equal_nan=True), kind


def test_christoffersen_flags_clustered_misses_and_not_independent_ones():
    rng = np.random.default_rng(4)
    iid = rng.random(2000) > 0.1                                                  # 10% independent misses
    clustered = np.ones(2000, dtype=bool)
    for s in range(0, 2000, 100): clustered[s:s + 10] = False                      # the same 10%, in runs of ten
    assert ce.christoffersen(iid)["ind_p"] > 0.01
    assert ce.christoffersen(clustered)["ind_p"] < 1e-6
    assert ce.kupiec_lr(clustered) < 1e-9                                         # exactly 90% covered: Kupiec sees nothing wrong


def test_wilson_interval_contains_the_proportion_and_shrinks_with_n():
    lo, hi = ce.wilson(90, 100); lo2, hi2 = ce.wilson(900, 1000)
    assert lo < 0.9 < hi and lo2 < 0.9 < hi2 and (hi2 - lo2) < (hi - lo)


def test_evaluate_matches_regression_metrics():
    from regression_metrics import interval_metrics
    rng = np.random.default_rng(5)
    n = 300
    close = 4000 * np.exp(np.cumsum(rng.normal(0, 0.01, n))); y = rng.normal(0, 0.01, n)
    lo, hi = y * 0 - 0.015, y * 0 + 0.015
    m, _, _ = ce.evaluate(close, y, lo, hi)
    ref = interval_metrics(close, y, lo, hi)
    for k in ("coverage", "mean_width_usd", "mean_width_pct", "interval_score_usd"):
        assert abs(m[k] - ref[k]) < 1e-9, k


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
