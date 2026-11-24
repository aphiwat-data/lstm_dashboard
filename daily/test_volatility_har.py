"""Tests for the E6 volatility experiment (run: python test_volatility_har.py). Synthetic data only: no network, no TensorFlow."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import volatility_har_experiment as v  # noqa: E402


def test_realised_variance_adds_up_and_respects_the_settlement_boundary():
    # two trading days; hourly bars every hour for 48 hours starting Monday 2024-01-01 00:00 UTC (winter: ET = UTC - 5, 13:30 ET = 18:30 UTC)
    open_utc = pd.Series(pd.date_range("2024-01-01 00:00", periods=48, freq="h", tz="UTC"))
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.001, 48))))
    h = pd.DataFrame({"open_utc": open_utc, "close": close})
    dates = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"])
    out = v.realised_variance(h, dates, 13 * 60 + 30, min_bars=1)
    lr = np.log(close).diff().to_numpy()
    close_et = (open_utc + pd.Timedelta(hours=1)).dt.tz_convert("America/New_York")
    mins = (close_et.dt.hour * 60 + close_et.dt.minute).to_numpy()
    day = close_et.dt.normalize().dt.tz_localize(None).to_numpy()
    # a bar whose close is at or before 13:30 ET belongs to that ET date, a later one to the next date
    for k, d in enumerate(dates):
        want = sum(lr[i] ** 2 for i in range(1, 48) if (pd.Timestamp(day[i]) + pd.Timedelta(days=int(mins[i] > 810))) == d)
        assert abs(out["rv"].iloc[k] - want) < 1e-15 if want > 0 else np.isnan(out["rv"].iloc[k]) or out["rv"].iloc[k] == 0
    assert abs(np.nansum(out["rv"]) - np.nansum(lr[1:] ** 2)) < 1e-12 or out["n"].sum() < 47  # nothing lost unless a label falls beyond the daily index


def test_bars_after_the_boundary_move_to_the_next_day():
    # one bar closing at 13:30 ET exactly (opens 12:30 ET = 17:30 UTC) stays on its day; the next one (closes 14:30 ET) goes to the next
    h = pd.DataFrame({"open_utc": pd.Series(pd.to_datetime(["2024-01-02 16:30", "2024-01-02 17:30", "2024-01-02 18:30"], utc=True)), "close": [100.0, 101.0, 102.0]})
    pos = v.day_positions(h["open_utc"], pd.DatetimeIndex(["2024-01-02", "2024-01-03"]), 13 * 60 + 30)
    assert list(pos) == [0, 0, 1], pos


def test_har_inputs_never_use_the_future():
    rng = np.random.default_rng(1)
    rv = rng.uniform(1e-5, 4e-5, 200)
    ev = v.event_flags(pd.bdate_range("2020-01-01", periods=200))
    a = v.har_inputs(rv, ev)
    rv2 = rv.copy(); rv2[120:] = 9.0                         # change the future of origin 100
    b = v.har_inputs(rv2, ev)
    for k in ("d", "w", "m"):
        assert np.allclose(a[k][:120], b[k][:120], equal_nan=True), k   # features of origins before the change are untouched
    assert np.isclose(a["y"][118], rv[119]) and np.isnan(a["y"][199])   # the target of row t is the next day's variance


def test_design_shapes_and_event_dummies_are_for_the_target_day():
    dates = pd.bdate_range("2024-01-01", periods=20)                     # January only; 2024-01-05 is the first Friday
    ev = v.event_flags(dates)
    assert ev["nfp"].sum() == 1 and ev.loc["2024-01-05", "nfp"] == 1
    rv = np.linspace(1e-5, 2e-5, 20)
    inp = v.har_inputs(rv, ev)
    i = list(dates).index(pd.Timestamp("2024-01-04"))                     # origin: the day before the NFP day
    assert inp["nfp1"][i] == 1 and inp["nfp1"][i + 1] == 0               # the dummy of row t describes day t+1
    X = v.design(inp, "x")
    assert X.shape == (20, 8) and v.design(inp, "logs").shape == (20, 4) and v.design(inp, "levels").shape == (20, 4)


def test_ewma_path_matches_the_recursion_and_garch_filter_matches_a_loop():
    x = np.array([0.01, -0.02, 0.005, np.nan, 0.03])
    out = v.ewma_path(x, 0.94, 1e-4, True)
    s = 1e-4; ref = []
    for xi in x:
        if np.isfinite(xi): s = 0.94 * s + 0.06 * xi * xi
        ref.append(s)
    assert np.allclose(out, ref)
    e = np.random.default_rng(2).normal(0, 0.01, 50); om, a, b = 1e-6, 0.08, 0.9
    vv = [e.var()]
    for t in range(len(e)): vv.append(om + a * e[t] ** 2 + b * vv[-1])
    assert np.allclose(v.garch_variance(e, om, a, b), vv)


def test_qlike_is_minimised_at_the_true_variance_and_dm_detects_a_better_forecast():
    rng = np.random.default_rng(3)
    true = rng.uniform(1e-5, 3e-5, 4000)
    rv = true * rng.chisquare(1, 4000)                                    # a noisy proxy with the right mean
    q = lambda f: v.qlike(rv, f).mean()
    assert q(true) < q(true * 0.5) and q(true) < q(true * 2) and q(true) < q(np.full(4000, true.mean()))
    d = v.qlike(rv, true) - v.qlike(rv, np.full(4000, true.mean()))
    assert d.mean() < 0 and v.hac_p(d) < 0.05
    assert v.hac_p(np.random.default_rng(4).normal(0, 1, 500)) > 0.001   # no difference: p is not tiny


def test_har_regression_recovers_a_known_process():
    rng = np.random.default_rng(5)
    n = 3000; lv = np.zeros(n)
    for t in range(1, n): lv[t] = 0.9 * lv[t - 1] + rng.normal(0, 0.3)
    rv = np.exp(lv) * 1e-5
    inp = v.har_inputs(rv, v.event_flags(pd.bdate_range("2000-01-01", periods=n)))
    X, y = v.design(inp, "logs"), np.log(inp["y"])
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    b = v.ols(X[ok], y[ok])
    assert 0.5 < b[1] + b[2] + b[3] < 1.1                                  # the persistence of the process is recovered (sum of the three slopes)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
