"""Daily gold pipeline Lambda (v2): ingest -> validate -> forecast -> log. Return-target LSTM served with numpy (no TensorFlow).

EventBridge Scheduler (daily) -> this function:
  1. fetch the latest settled daily bars from Yahoo's chart endpoint (stdlib urllib); a bar is accepted only once its COMEX session has ended
  2. append new bars to its own append-only log (idempotent, keyed by date) -- never rewrites Silver
  3. data-quality checks on the price series; a critical failure stops the run (no forecast is published) and raises
  4. features from the trailing prices -> numpy LSTM (weights from s3://<bucket>/model/serving/serving_model.npz, retrained weekly elsewhere)
     -> next-day forecast with a 90% band (EWMA of past errors)
  5. upsert the forecast log (live track record: each past forecast is scored once its next close is known) and write latest_forecast.json
  6. emit CloudWatch metrics (embedded metric format) and a run-log row
The previous price-level/TFLite version is kept as lambda_function_v1_price_level.py.old.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from datetime import datetime, timezone
from io import BytesIO, StringIO
from zoneinfo import ZoneInfo

import boto3
import numpy as np
import pandas as pd

import serving

logger = logging.getLogger()
logger.setLevel(logging.INFO)

S3_BUCKET = os.environ.get("S3_BUCKET", "gold-lstm-forecast")
REGION = os.environ.get("AWS_REGION", "ap-southeast-2")
SILVER_KEY = "silver/xauusd_daily_clean.parquet"                          # read-only
MODEL_KEY = "model/serving/serving_model.npz"                              # read-only, written by the weekly export
STREAM_LOG_KEY = "bronze/streaming/xauusd_daily_incremental.csv"           # this function's append-only log
RUN_LOG_KEY = "bronze/streaming/automation_log.csv"
FORECAST_KEY = "gold/xauusd_daily/predictions/latest_forecast.json"
FORECAST_LOG_KEY = "gold/xauusd_daily/predictions/forecast_log.csv"
DQ_KEY = "gold/xauusd_daily/predictions/dq_report.json"
OHLCV = ["date", "open", "high", "low", "close", "volume"]
MAX_GAP_DAYS, EXTREME_LOGRET, CRITICAL_LOGRET = 5, 0.07, 0.15
TICKER = "GC%3DF"

s3 = boto3.client("s3", region_name=REGION)
_model_cache: dict = {}


# --------------------------------------------------------------------------------------------------------------------- ingest
def chart_range(days_behind: int) -> str:
    """Smallest Yahoo range that covers the gap since the last known bar (daily runs need 10d; a long outage or the first run needs more)."""
    for limit, rng in ((8, "10d"), (25, "1mo"), (80, "3mo"), (170, "6mo"), (350, "1y"), (700, "2y")):
        if days_behind <= limit:
            return rng
    return "5y"


def settled(bars: pd.DataFrame, now_utc: datetime) -> pd.DataFrame:
    """Keep bars whose COMEX session has ended (17:00 New York). At 08:00 Thailand time a new session has already started and Yahoo shows a partial bar for it."""
    now_et = now_utc.astimezone(ZoneInfo("America/New_York"))
    today = pd.Timestamp(now_et.date())
    ok = (bars["date"] < today) | ((bars["date"] == today) & (now_et.hour >= 17))
    return bars[ok]


def fetch_recent_bars(days_behind: int = 0, now_utc: datetime | None = None) -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{TICKER}?range={chart_range(days_behind)}&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; gold-lstm-forecast-bot/2.0)"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    result = payload.get("chart", {}).get("result")
    if not result:
        raise RuntimeError(f"Yahoo chart API returned no result: {payload.get('chart', {}).get('error')}")
    q = result[0]["indicators"]["quote"][0]
    bars = pd.DataFrame({"date": pd.to_datetime(result[0]["timestamp"], unit="s", utc=True).tz_localize(None).normalize(),
                         "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"], "volume": q["volume"]})
    bars = bars.dropna(subset=["close"]).drop_duplicates("date", keep="last").sort_values("date")
    return settled(bars, now_utc or datetime.now(timezone.utc)).reset_index(drop=True)


def _read_csv(key: str, **kw) -> pd.DataFrame | None:
    try:
        return pd.read_csv(BytesIO(s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()), **kw)
    except s3.exceptions.NoSuchKey:
        return None


def _put_csv(key: str, df: pd.DataFrame) -> None:
    buf = StringIO()
    df.to_csv(buf, index=False)
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue().encode("utf-8"))


def load_prices() -> pd.DataFrame:
    """Silver (read-only) plus this function's own log of newer bars."""
    silver = pd.read_parquet(BytesIO(s3.get_object(Bucket=S3_BUCKET, Key=SILVER_KEY)["Body"].read()), columns=OHLCV)
    silver["date"] = pd.to_datetime(silver["date"])
    stream = _read_csv(STREAM_LOG_KEY, parse_dates=["date"])
    if stream is not None and len(stream):
        silver = pd.concat([silver, stream[OHLCV][stream["date"] > silver["date"].max()]], ignore_index=True)
    return silver.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def append_stream_log(new_rows: pd.DataFrame) -> None:
    """Idempotent append keyed by date: replaying overlapping days never duplicates or reorders rows."""
    existing = _read_csv(STREAM_LOG_KEY, parse_dates=["date"])
    add = new_rows[OHLCV].assign(fetched_at=datetime.now(timezone.utc).isoformat())
    merged = add if existing is None else pd.concat([existing, add], ignore_index=True)
    _put_csv(STREAM_LOG_KEY, merged.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True))


# -------------------------------------------------------------------------------------------------------------- data quality
def data_quality(prices: pd.DataFrame, new_rows: pd.DataFrame, now_utc: datetime) -> dict:
    """Checks on the price series the forecast will use. `critical` failures stop the run; the rest are reported."""
    p = prices.tail(400)
    lr = np.log(p["close"]).diff()
    gap = int((pd.Timestamp(now_utc.date()) - p["date"].max()).days)
    n = new_rows
    bad_ohlc = int(((n["high"] < n[["open", "close"]].max(axis=1)) | (n["low"] > n[["open", "close"]].min(axis=1))).sum()) if len(n) else 0
    checks = {
        "schema_ok": all(c in prices.columns for c in OHLCV),
        "nonpositive_close": int((p["close"] <= 0).sum()),
        "duplicate_dates": int(p["date"].duplicated().sum()),
        "extreme_moves_gt_7pct": int((lr.abs() > EXTREME_LOGRET).sum()),
        "max_abs_logret_last_5": float(lr.abs().tail(5).max()),
        "days_since_last_bar": gap,
        "invalid_ohlc_new_rows": bad_ohlc,
        "new_rows": int(len(n)),
        "last_bar": str(p["date"].max().date()),
    }
    critical = [k for k, bad in (("schema_ok", not checks["schema_ok"]), ("nonpositive_close", checks["nonpositive_close"] > 0),
                                 ("duplicate_dates", checks["duplicate_dates"] > 0), ("max_abs_logret_last_5", checks["max_abs_logret_last_5"] > CRITICAL_LOGRET)) if bad]
    warnings = [k for k, bad in (("stale_data", gap > MAX_GAP_DAYS), ("invalid_ohlc_new_rows", bad_ohlc > 0), ("extreme_move_recent", checks["max_abs_logret_last_5"] > EXTREME_LOGRET)) if bad]
    return {"checked_at": now_utc.isoformat(timespec="seconds"), "checks": checks, "critical": critical, "warnings": warnings, "passed": not critical}


# --------------------------------------------------------------------------------------------------------------- model + log
def load_model() -> dict:
    """Weights are cached across warm invocations but re-read as soon as the weekly retrain replaces the object (ETag check)."""
    obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_KEY)
    etag = obj.get("ETag", "")
    if _model_cache.get("etag") != etag or "model" not in _model_cache:
        _model_cache["model"] = {k: v for k, v in np.load(BytesIO(obj["Body"].read()), allow_pickle=False).items()}
        _model_cache["etag"] = etag
    return _model_cache["model"]


def update_forecast_log(forecast: dict, prices: pd.DataFrame, model_version: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Upsert today's forecast and score every earlier one whose next close is now known. Returns (log, realised errors in date order)."""
    log = _read_csv(FORECAST_LOG_KEY, parse_dates=["as_of_date"])
    cols = ["as_of_date", "latest_close", "predicted_return", "predicted_next_close", "p_up", "lo_90", "hi_90", "model_version", "next_date", "actual_next_close", "log_error", "inside_90"]
    log = pd.DataFrame(columns=cols) if log is None else log
    row = {"as_of_date": pd.Timestamp(forecast["as_of_date"]), "latest_close": forecast["latest_close"], "predicted_return": forecast["predicted_return"],
           "predicted_next_close": forecast["predicted_next_close"], "p_up": forecast["p_up"], "lo_90": forecast["interval_90_close"][0],
           "hi_90": forecast["interval_90_close"][1], "model_version": model_version}
    log = pd.concat([log[log["as_of_date"] != row["as_of_date"]], pd.DataFrame([row])], ignore_index=True).sort_values("as_of_date").reset_index(drop=True)
    log = log.astype({"next_date": "object", "inside_90": "object", "model_version": "object"})  # an all-empty CSV column is read back as float
    log["as_of_date"] = pd.to_datetime(log["as_of_date"])
    close = prices.set_index("date")["close"]
    for i, r in log.iterrows():
        if pd.isna(r.get("actual_next_close")):
            later = close[close.index > r["as_of_date"]]
            if len(later):
                act = float(later.iloc[0])
                log.at[i, "next_date"], log.at[i, "actual_next_close"] = str(later.index[0].date()), act
                log.at[i, "log_error"] = float(np.log(act / r["latest_close"]) - r["predicted_return"])
                log.at[i, "inside_90"] = bool(r["lo_90"] <= act <= r["hi_90"])
    errors = log["log_error"].dropna().to_numpy(float)
    return log, errors


def emit_metrics(**values: float) -> None:
    """CloudWatch embedded metric format: a structured log line becomes metrics, no extra IAM permission needed."""
    names = list(values)
    print(json.dumps({"_aws": {"Timestamp": int(time.time() * 1000), "CloudWatchMetrics": [{"Namespace": "GoldLSTM", "Dimensions": [["FunctionName"]],
                      "Metrics": [{"Name": n, "Unit": "Seconds" if n == "RunSeconds" else "Count"} for n in names]}]},
                      "FunctionName": os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "gold-lstm-forecast-daily-refresh"), **values}))


def append_run_log(outcome: str, detail: str = "") -> None:
    row = pd.DataFrame([{"invoked_at": datetime.now(timezone.utc).isoformat(), "outcome": outcome, "detail": detail}])
    old = _read_csv(RUN_LOG_KEY)
    _put_csv(RUN_LOG_KEY, (row if old is None else pd.concat([old, row], ignore_index=True)).tail(500))


# ------------------------------------------------------------------------------------------------------------------ handler
def lambda_handler(event=None, context=None):
    t0 = time.time()
    now_utc = datetime.fromisoformat(event["now"]) if event and "now" in event else datetime.now(timezone.utc)  # "now" is for tests
    try:
        prices = load_prices()
        bars = fetch_recent_bars((pd.Timestamp(now_utc.date()) - prices["date"].max()).days, now_utc)
        new_rows = bars[bars["date"] > prices["date"].max()]
        if len(new_rows):
            append_stream_log(new_rows)
            prices = pd.concat([prices, new_rows[OHLCV]], ignore_index=True).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
        dq = data_quality(prices, new_rows, now_utc)
        s3.put_object(Bucket=S3_BUCKET, Key=DQ_KEY, Body=json.dumps(dq, indent=1).encode("utf-8"))
        if not dq["passed"]:
            raise RuntimeError(f"data quality failed: {dq['critical']}")
        model = load_model()
        rows = serving.latest_feature_rows(prices.tail(400))
        base_errors = model["errors"]
        forecast = serving.forecast(model, rows, base_errors)
        log, live_errors = update_forecast_log(forecast, prices, str(model["model_version"]))
        forecast["interval_90_close"] = serving.band(np.r_[base_errors, live_errors], forecast["latest_close"], forecast["predicted_return"])
        log.loc[log["as_of_date"] == pd.Timestamp(forecast["as_of_date"]), ["lo_90", "hi_90"]] = forecast["interval_90_close"]
        forecast.update(generated_at=now_utc.isoformat(timespec="seconds"), model_version=str(model["model_version"]), model_trained_through=str(model["trained_through"]),
                        interval_method=f"Gaussian, sigma = EWMA(lambda {serving.EWMA_LAMBDA}) of past errors, z = {serving.Z90:.3f}")
        s3.put_object(Bucket=S3_BUCKET, Key=FORECAST_KEY, Body=json.dumps(forecast, indent=2).encode("utf-8"))
        _put_csv(FORECAST_LOG_KEY, log)
        scored = log.dropna(subset=["inside_90"])
        outcome = "forecast_updated" if len(new_rows) else "no_new_data"
        append_run_log(outcome, f"as_of={forecast['as_of_date']} new_rows={len(new_rows)} warnings={dq['warnings']}")
        emit_metrics(NewRows=len(new_rows), DQWarnings=len(dq["warnings"]), Failure=0, ForecastsScored=len(scored), RunSeconds=round(time.time() - t0, 2))
        return {"outcome": outcome, "as_of": forecast["as_of_date"], "forecast": forecast["predicted_next_close"], "interval_90": forecast["interval_90_close"], "dq_warnings": dq["warnings"]}
    except Exception as exc:  # noqa: BLE001 - record the failure, then let Lambda mark the invocation as failed
        logger.exception("run failed")
        try:
            append_run_log("failed", str(exc)[:300])
        except Exception:  # noqa: BLE001
            pass
        emit_metrics(NewRows=0, DQWarnings=0, Failure=1, ForecastsScored=0, RunSeconds=round(time.time() - t0, 2))
        raise
