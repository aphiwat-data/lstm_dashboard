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


def _setup(n_silver=600):
    allb = _bars("2024-06-03", n_silver + 6)
    silver, future = allb.iloc[:n_silver].copy(), allb.iloc[n_silver:].reset_index(drop=True)
    fake = FakeS3()
    buf = BytesIO()
    silver.to_parquet(buf)
    fake.objs[lf.SILVER_KEY] = buf.getvalue()
    mb = BytesIO()
    np.savez(mb, **dict(np.load(HERE.parent / "daily" / "serving_artifacts" / "serving_model.npz")))
    fake.objs[lf.MODEL_KEY] = mb.getvalue()
    lf.s3 = fake
    lf._model_cache.clear()
    return fake, silver, future


def _run(future_rows, now):
    lf.fetch_recent_bars = lambda days_behind=0, now_utc=None: future_rows
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
