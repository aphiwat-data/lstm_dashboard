"""End-to-end tests of the Lambda handler against an in-memory S3 (no AWS). Run: python test_lambda_function.py (or pytest)."""
import json
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "daily"))
sys.path.insert(0, str(HERE))
import lambda_function as lf  # noqa: E402


class FakeS3:
    class exceptions:
        class NoSuchKey(Exception):
            pass

    def __init__(self):
        self.objs = {}

    def get_object(self, Bucket, Key):
        if Key not in self.objs:
            raise self.exceptions.NoSuchKey(Key)
        return {"Body": BytesIO(self.objs[Key]), "ETag": str(hash(self.objs[Key]))}

    def put_object(self, Bucket, Key, Body):
        self.objs[Key] = Body


def _bars(start, periods, seed=0, base=4000.0):
    rng = np.random.default_rng(seed)
    close = base * np.exp(np.cumsum(rng.normal(0.0003, 0.01, periods)))
    return pd.DataFrame({"date": pd.bdate_range(start, periods=periods), "open": close, "high": close * 1.003, "low": close * 0.997, "close": close, "volume": 1e5})


MACRO_FUTURE = pd.DataFrame()


def _macro(start, periods, seed=1):
    rng = np.random.default_rng(seed)
    cols = {c: base * np.exp(np.cumsum(rng.normal(0, 0.006, periods))) for c, base in (("dxy", 100.0), ("silver", 30.0), ("spx", 5000.0), ("tip", 108.0))}
    cols["yield10y"] = 4.0 + np.cumsum(rng.normal(0, 0.03, periods))
    return pd.DataFrame(cols, index=pd.bdate_range(start, periods=periods))[["dxy", "silver", "spx", "yield10y", "tip"]]


def _setup(n_silver=600):
    global MACRO_FUTURE
    allb = _bars("2024-06-03", n_silver + 6)
    silver, future = allb.iloc[:n_silver].copy(), allb.iloc[n_silver:].reset_index(drop=True)
    fake = FakeS3()
    buf = BytesIO()
    silver.to_parquet(buf)
    fake.objs[lf.SILVER_KEY] = buf.getvalue()
    mb = BytesIO()
    np.savez(mb, **dict(np.load(HERE.parent / "daily" / "serving_artifacts" / "serving_model.npz")))
    fake.objs[lf.MODEL_KEY] = mb.getvalue()
    macro_all = _macro("2024-06-03", n_silver + 6)
    fake.objs[lf.MACRO_HISTORY_KEY] = macro_all.iloc[:n_silver].rename_axis("date").reset_index().to_csv(index=False).encode()
    MACRO_FUTURE = macro_all.iloc[n_silver:]
    xb = BytesIO()
    np.savez(xb, errors=np.random.default_rng(5).normal(0, 0.01, 449), alpha=100.0, train_len=1000)
    fake.objs[lf.XMODEL_KEY] = xb.getvalue()
    lf.s3 = fake
    lf._model_cache.clear()
    lf._xmodel_cache.clear()
    return fake, silver, future


def _run(future_rows, now, macro=None):
    lf.fetch_recent_bars = lambda days_behind=0, now_utc=None: future_rows
    rows = MACRO_FUTURE.loc[:future_rows["date"].iloc[-1]] if macro is None else macro
    lf.crossasset.fetch_recent_macro = lambda range_, now_utc=None, **kw: rows
    return lf.lambda_handler({"now": now})


def test_first_run_second_run_is_idempotent_and_third_scores_the_first_forecast():
    fake, silver, future = _setup()
    now = "2026-01-01T01:00:00+00:00"
    r1 = _run(future.iloc[:3], now)
    assert r1["outcome"] == "forecast_updated" and r1["as_of"] == str(future["date"].iloc[2].date())
    stream = pd.read_csv(BytesIO(fake.objs[lf.STREAM_LOG_KEY]))
    assert len(stream) == 3
    lo, hi = r1["interval_90"]
    assert lo < r1["forecast"] < hi
    r2 = _run(future.iloc[:3], now)  # same bars again
    assert r2["outcome"] == "no_new_data" and len(pd.read_csv(BytesIO(fake.objs[lf.STREAM_LOG_KEY]))) == 3
    assert len(pd.read_csv(BytesIO(fake.objs[lf.FORECAST_LOG_KEY]))) == 1
    r3 = _run(future.iloc[:4], now)  # one more settled bar arrives
    log = pd.read_csv(BytesIO(fake.objs[lf.FORECAST_LOG_KEY]))
    assert len(log) == 2 and r3["as_of"] == str(future["date"].iloc[3].date())
    first = log.iloc[0]
    assert np.isclose(first["actual_next_close"], future["close"].iloc[3]) and first["inside_90"] in (True, False)
    assert np.isclose(first["log_error"], np.log(future["close"].iloc[3] / first["latest_close"]) - first["predicted_return"])
    assert pd.isna(log.iloc[1]["actual_next_close"])


def test_cross_asset_forecast_is_published_logged_scored_and_the_macro_log_is_idempotent():
    fake, silver, future = _setup()
    now = "2026-01-01T01:00:00+00:00"
    r1 = _run(future.iloc[:3], now)
    assert r1["crossasset"] is not None and r1["crossasset_failure"] == ""
    pub = json.loads(fake.objs[lf.FORECAST_KEY])
    x = pub["crossasset"]
    assert x["as_of_date"] == pub["as_of_date"] and x["model"] == "Cross-asset model" and x["n_train"] >= 250
    assert x["interval_90_close"][0] < x["predicted_next_close"] < x["interval_90_close"][1] and set(x["inputs"]) == set(lf.crossasset.FEATURES)
    assert len(pd.read_csv(BytesIO(fake.objs[lf.MACRO_STREAM_KEY]))) == 3
    inputs = pd.read_csv(BytesIO(fake.objs[lf.INPUTS_KEY]))
    assert list(inputs.columns) == ["date"] + lf.crossasset.FEATURES and len(inputs) == 130 and inputs["date"].iloc[-1] == pub["as_of_date"]
    assert np.allclose(inputs.iloc[-1][lf.crossasset.FEATURES].to_numpy(float), [x["inputs"][k] for k in lf.crossasset.FEATURES])
    log = pd.read_csv(BytesIO(fake.objs[lf.FORECAST_LOG_KEY]))
    assert all(c in log.columns for c in lf.X_COLS) and np.isclose(log.iloc[0]["x_predicted_next_close"], x["predicted_next_close"]) and log.iloc[0]["x_lo_90"] == x["interval_90_close"][0]
    _run(future.iloc[:3], now)  # the same bars again
    assert len(pd.read_csv(BytesIO(fake.objs[lf.MACRO_STREAM_KEY]))) == 3 and len(pd.read_csv(BytesIO(fake.objs[lf.FORECAST_LOG_KEY]))) == 1
    _run(future.iloc[:4], now)  # one more settled day: the first cross-asset forecast is scored
    log = pd.read_csv(BytesIO(fake.objs[lf.FORECAST_LOG_KEY]))
    first = log.iloc[0]
    assert np.isclose(first["x_log_error"], np.log(first["actual_next_close"] / first["latest_close"]) - first["x_predicted_return"]) and first["x_inside_90"] in (True, False)
    assert np.isfinite(log.iloc[1]["x_predicted_return"]) and pd.isna(log.iloc[1]["x_log_error"])
    again = json.loads(fake.objs[lf.FORECAST_KEY])["crossasset"]
    assert again["as_of_date"] == str(future["date"].iloc[3].date()) and again["interval_90_close"] != x["interval_90_close"], "the band is resized with the newest error"


def test_a_macro_failure_never_blocks_the_lstm_forecast_and_does_not_change_it():
    fake, silver, future = _setup()
    now = "2026-01-01T01:00:00+00:00"
    ok = _run(future.iloc[:3], now)
    fake2, _, future2 = _setup()
    lf.fetch_recent_bars = lambda days_behind=0, now_utc=None: future2.iloc[:3]

    def down(range_, now_utc=None, **kw):
        raise OSError("Yahoo is down")

    lf.crossasset.fetch_recent_macro = down
    bad = lf.lambda_handler({"now": now})
    assert bad["outcome"] == "forecast_updated" and bad["crossasset"] is None and "OSError" in bad["crossasset_failure"]
    assert bad["forecast"] == ok["forecast"] and bad["interval_90"] == ok["interval_90"], "the LSTM forecast is identical whether or not the cross-asset step works"
    pub = json.loads(fake2.objs[lf.FORECAST_KEY])
    assert pub["crossasset"] is None and "predicted_next_close" in pub
    assert "x_failed=OSError" in pd.read_csv(BytesIO(fake2.objs[lf.RUN_LOG_KEY]))["detail"].iloc[-1]
    log = pd.read_csv(BytesIO(fake2.objs[lf.FORECAST_LOG_KEY]))
    assert len(log) == 1 and pd.isna(log.iloc[0]["x_predicted_return"])


def test_a_log_written_before_the_cross_asset_model_is_upgraded_and_old_rows_stay_as_they_were():
    fake, silver, future = _setup()
    legacy = pd.DataFrame([{"as_of_date": "2025-12-30", "latest_close": 4100.0, "predicted_return": 0.001, "predicted_next_close": 4104.1, "p_up": 0.55, "lo_90": 4000.0, "hi_90": 4200.0,
                            "model_version": "old", "next_date": "2025-12-31", "actual_next_close": 4110.0, "log_error": 0.0014, "inside_90": True}])
    fake.objs[lf.FORECAST_LOG_KEY] = legacy.to_csv(index=False).encode()
    _run(future.iloc[:3], "2026-01-01T01:00:00+00:00")
    log = pd.read_csv(BytesIO(fake.objs[lf.FORECAST_LOG_KEY]))
    assert len(log) == 2 and all(c in log.columns for c in lf.X_COLS)
    old = log.iloc[0]
    assert old["model_version"] == "old" and old["actual_next_close"] == 4110.0 and old["inside_90"] in (True, "True") and pd.isna(old["x_predicted_return"]) and pd.isna(old["x_log_error"])
    assert np.isfinite(log.iloc[1]["x_predicted_return"])


def test_critical_data_quality_failure_publishes_no_forecast_and_raises():
    fake, silver, future = _setup()
    bad = future.iloc[:2].copy()
    bad.loc[bad.index[1], "close"] = -5.0
    try:
        _run(bad, "2026-01-01T01:00:00+00:00")
        raise AssertionError("expected failure")
    except RuntimeError as e:
        assert "data quality" in str(e)
    assert lf.FORECAST_KEY not in fake.objs
    dq = json.loads(fake.objs[lf.DQ_KEY])
    assert not dq["passed"] and "nonpositive_close" in dq["critical"]
    assert "failed" in pd.read_csv(BytesIO(fake.objs[lf.RUN_LOG_KEY]))["outcome"].tolist()


def test_model_is_reloaded_when_the_object_changes():
    fake, silver, future = _setup()
    m1 = lf.load_model()
    assert lf.load_model() is m1  # unchanged object: served from the cache
    arrays = dict(np.load(BytesIO(fake.objs[lf.MODEL_KEY])))
    arrays["model_version"] = np.array("newer")
    buf = BytesIO()
    np.savez(buf, **arrays)
    fake.objs[lf.MODEL_KEY] = buf.getvalue()
    assert str(lf.load_model()["model_version"]) == "newer"


def test_unsettled_bar_is_dropped():
    bars = pd.DataFrame({"date": pd.to_datetime(["2026-10-26", "2026-10-27"])})
    # 01:00 UTC on Oct 27 is 21:00 in New York on Oct 26: the Oct 27 session has only just started
    kept = lf.settled(bars, datetime(2026, 10, 27, 1, 0, tzinfo=timezone.utc))
    assert kept["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-10-26"]
    kept2 = lf.settled(bars, datetime(2026, 10, 27, 22, 30, tzinfo=timezone.utc))  # 18:30 New York on the 27th: the 27th is settled
    assert len(kept2) == 2


def test_chart_range_covers_the_gap():
    assert lf.chart_range(2) == "10d" and lf.chart_range(30) == "3mo" and lf.chart_range(400) == "2y" and lf.chart_range(5000) == "5y"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
