"""Tests for the cross-asset model (run: python test_crossasset.py). The checks against the E2 search need the local files data/e2_prices.parquet and data/e2_macro.parquet
(python search_experiment.py prepare); without them those tests are skipped."""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import crossasset as ca  # noqa: E402

HAVE_DATA = (HERE / "data" / "e2_prices.parquet").exists() and (HERE / "data" / "e2_macro.parquet").exists()


def _payload(tz: str, stamps_utc: list[str], closes: list) -> dict:
    ts = [int(pd.Timestamp(s, tz="UTC").timestamp()) for s in stamps_utc]
    return {"chart": {"result": [{"meta": {"exchangeTimezoneName": tz}, "timestamp": ts, "indicators": {"quote": [{"close": closes}]}}], "error": None}}


def test_fit_predict_equals_sklearn_standardscaler_plus_ridge():
    try:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("  (skipped: needs scikit-learn)")
        return
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 6)) * [1, 5, 0.1, 2, 0.3, 1] + [0, 1, 2, 0, 0.5, 0]
    y = 0.001 + 0.05 * X[:, 1] * 0.01 + rng.normal(0, 0.01, 1000)
    m = ca.fit(X, y)
    sd = float(y.std())
    ref = make_pipeline(StandardScaler(), Ridge(alpha=100.0)).fit(X, y / sd)
    Xn = rng.normal(size=(50, 6)) * [1, 5, 0.1, 2, 0.3, 1]
    assert np.allclose(ca.predict(m, Xn), ref.predict(Xn) * sd, atol=1e-12)
    assert m["n_train"] == 1000 and abs(m["ret_sd"] - sd) < 1e-15


def test_parse_chart_labels_bars_with_the_exchange_date_and_drops_gaps_and_duplicates():
    futures = _payload("America/New_York", ["2026-11-24 05:00", "2026-11-25 05:00", "2026-11-25 05:00", "2026-11-26 07:59"], [3561.9, 3535.8, 3536.0, None])
    s = ca.parse_chart(futures)
    assert list(s.index.strftime("%Y-%m-%d")) == ["2026-11-24", "2026-11-25"] and s.iloc[-1] == 3536.0, "NaN dropped, duplicate date keeps the last value"
    tnx = ca.parse_chart(_payload("America/Chicago", ["2026-11-24 13:20"], [4.9]))        # 07:20 in Chicago: the same calendar date
    equity = ca.parse_chart(_payload("America/New_York", ["2026-11-24 14:30"], [7201.0]))  # 09:30 in New York
    assert tnx.index[0] == pd.Timestamp("2026-11-24") and equity.index[0] == pd.Timestamp("2026-11-24")
    late = ca.parse_chart(_payload("America/New_York", ["2026-07-02 04:00"], [1.0]))        # summer time: midnight in New York is 04:00 UTC
    assert late.index[0] == pd.Timestamp("2026-07-02")
    try:
        ca.parse_chart({"chart": {"result": None, "error": {"code": "Not Found"}}})
        raise AssertionError("expected an error")
    except RuntimeError as e:
        assert "no result" in str(e)


def test_parse_chart_prefers_the_dividend_adjusted_close_and_stale_series_are_refused():
    p = _payload("America/New_York", ["2026-11-24 14:30", "2026-11-25 14:30"], [106.0, 105.8])
    p["chart"]["result"][0]["indicators"]["adjclose"] = [{"adjclose": [105.5, 105.8]}]
    assert list(ca.parse_chart(p)) == [105.5, 105.8]
    dates = pd.bdate_range("2026-11-02", "2026-11-25")
    m = pd.DataFrame({c: 1.0 for c in ca.MACRO_COLS}, index=dates)
    ca.check_fresh(m, pd.Timestamp("2026-11-27"))                                   # a two-day-old series is fine (one market's holiday)
    m.loc[m.index > "2026-11-18", "yield10y"] = np.nan
    try:
        ca.check_fresh(m, pd.Timestamp("2026-11-25"))
        raise AssertionError("expected a stale-input error")
    except ValueError as e:
        assert "stale inputs" in str(e) and "yield10y" in str(e)


def test_fetch_recent_macro_retries_and_keeps_only_settled_bars():
    import threading

    calls, lock = {"n": 0}, threading.Lock()

    def fake(url: str) -> dict:
        with lock:
            calls["n"] += 1
            n = calls["n"]
        if n <= 2:
            raise OSError("temporary failure")
        return _payload("America/New_York", ["2026-11-24 05:00", "2026-11-25 05:00", "2026-11-26 05:00"], [1.0, 2.0, 3.0])

    early = datetime(2026, 11, 26, 8, 0, tzinfo=timezone.utc)                         # 03:00 in New York on the 26th: the 26th is not over
    m = ca.fetch_recent_macro("10d", early, fetch=fake, attempts=3, pause=0.0)
    assert list(m.columns) == ca.MACRO_COLS and list(m.index.strftime("%Y-%m-%d")) == ["2026-11-24", "2026-11-25"] and calls["n"] == 7, "two failures retried, then one call per ticker"
    late = datetime(2026, 11, 26, 22, 30, tzinfo=timezone.utc)                        # 17:30 in New York: the 26th has settled
    m2 = ca.fetch_recent_macro("10d", late, fetch=lambda u: _payload("America/New_York", ["2026-11-26 05:00"], [3.0]), attempts=1, pause=0.0)
    assert list(m2.index.strftime("%Y-%m-%d")) == ["2026-11-26"]
    try:
        ca.fetch_recent_macro("10d", early, fetch=lambda u: (_ for _ in ()).throw(OSError("down")), attempts=2, pause=0.0)
        raise AssertionError("a ticker that keeps failing must raise")
    except OSError:
        pass


def test_macro_features_are_causal_and_use_the_latest_close_at_or_before_the_gold_date():
    dates = pd.bdate_range("2024-01-01", periods=60)
    close = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, 60))), index=dates)
    macro = pd.DataFrame({c: 50 + np.arange(25) for c in ca.MACRO_COLS}, index=dates[::2][:25])      # a macro market that trades every other day
    f = ca.macro_features(close, macro)
    assert list(f.columns) == ca.FEATURES and f.index.equals(close.index)
    i = 11                                                                                           # an odd position: the macro close of the day before is used, the day-over-day change is 0 there
    assert abs(f["f_yield10y_chg_1"].iloc[i]) < 1e-12 or abs(f["f_yield10y_chg_1"].iloc[i] - 1.0) < 1e-12
    f2 = ca.macro_features(close.iloc[:40], macro.loc[:close.index[39]])
    assert np.allclose(f2.to_numpy(float), f.iloc[:40].to_numpy(float), equal_nan=True), "features at t must not depend on later data"


def _e2():
    import search_experiment as se
    prices, macro = pd.read_parquet(se.PRICES), pd.read_parquet(se.MACRO)
    return se, se.build_panel(prices, macro), prices, macro


def test_features_match_the_search_panel_on_real_data():
    if not HAVE_DATA:
        print("  (skipped: needs the local E2 inputs)")
        return
    se, panel, prices, macro = _e2()
    close = prices.assign(date=pd.to_datetime(prices["date"])).drop_duplicates("date").sort_values("date").set_index("date")["close"]
    mine = ca.macro_features(close, macro)
    both = mine.loc[panel.dates, ca.FEATURES].to_numpy(float)
    assert np.allclose(both, panel.X[ca.FEATURES].to_numpy(float), atol=1e-12), "the six inputs must equal the ones the search used"


def test_the_live_code_reproduces_the_e2_winner_forecast_at_a_block_start():
    if not HAVE_DATA:
        print("  (skipped: needs the local E2 inputs)")
        return
    se, panel, prices, macro = _e2()
    spec = se.Spec("ridge[a100]|X|L1000", "ridge", (100,), "X", "L1000")
    X, y = panel.X[se.SETS["X"]].to_numpy(float), panel.y[1]
    close = prices.assign(date=pd.to_datetime(prices["date"])).drop_duplicates("date").sort_values("date").set_index("date")["close"]
    for k in (60, 120, 190):
        s = panel.first + 21 * k
        ref = se.forecast_block(spec, X, y, s, s + 1)[0]
        d = panel.dates[s]
        out = ca.forecast_latest(close.loc[:d], macro.loc[:d], errors=np.full(100, 0.01))
        assert out["as_of_date"] == str(d.date()) and out["n_train"] == 1000
        assert abs(out["predicted_return"] - ref) < 1e-12, (str(d.date()), out["predicted_return"], ref)
        assert out["interval_90_close"][0] < out["predicted_next_close"] < out["interval_90_close"][1]
    frozen = pd.read_csv(HERE / "results" / "paper_e2" / "e2_frozen_forecasts_h1.csv", parse_dates=["date"]).set_index("date")["dev_best"]
    s = panel.first + 21 * 120
    assert abs(frozen.loc[panel.dates[s]] - se.forecast_block(spec, X, y, s, s + 1)[0]) < 1e-6, "the committed E2 forecast is the same series"


def test_forecast_latest_refuses_to_guess_without_inputs_or_training_rows():
    dates = pd.bdate_range("2022-01-03", periods=400)
    close = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(2).normal(0, 0.01, 400))), index=dates)
    macro = pd.DataFrame({c: 50 + np.random.default_rng(3).normal(size=400).cumsum() * 0.1 + 100 for c in ca.MACRO_COLS}, index=dates)
    ok = ca.forecast_latest(close, macro, errors=np.full(50, 0.01))
    assert ok["n_train"] == 399 - 1 or ok["n_train"] >= 250
    late_only = macro.loc[macro.index > dates[-1]]
    try:
        ca.forecast_latest(close, late_only, errors=np.full(50, 0.01))
        raise AssertionError("expected an error")
    except (ValueError, KeyError, IndexError):
        pass
    try:
        ca.forecast_latest(close.iloc[:100], macro.iloc[:100], errors=np.full(50, 0.01))
        raise AssertionError("expected an error")
    except ValueError as e:
        assert "labelled rows" in str(e)


if __name__ == "__main__":
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        fn()
        print("ok  ", name)
    print("all tests passed")
