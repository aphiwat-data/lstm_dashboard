"""
Next-Day Gold Price Forecasting with LSTM — Daily Inference-Refresh Lambda
=============================================================================

WHAT THIS FUNCTION DOES (and, just as important, what it deliberately does
NOT do — see the main README's automation section for the full reasoning).
On an EventBridge Scheduler tick (once daily, 01:00 UTC = 08:00 Thailand
time — see "Why 08:00 Thailand time" below for why that specific time):

  1. Check whether a new daily close for GC=F (COMEX Gold Futures) has
     posted since the last time this function ran.
  2. If yes (the expected outcome on every weekday run): record it in an
     *incremental, append-only* Bronze-layer log (never touches the
     canonical bronze/silver/gold files the manual notebooks own — see
     "Why an isolated write path" below) and recompute the 7 engineered
     features for the new day.
  3. Run INFERENCE ONLY (no training) using the already-trained model to
     produce a genuine next-day forecast, and write it to
     gold/xauusd_daily/predictions/latest_forecast.json.
  4. If no new close has posted (expected on weekends and market holidays
     — COMEX Gold doesn't trade Friday 5pm ET through Sunday 6pm ET), do
     nothing and exit. This is intended, expected behavior on those days,
     not a failure.

WHY DAILY, AND WHY 08:00 THAILAND TIME SPECIFICALLY
-----------------------------------------------------------------------------
This dataset is DAILY granularity — one new (close, feature-row)
observation becomes available per trading day, not continuously — so the
check is scheduled to match that cadence exactly, once/day, rather than
polling more often than the data can possibly change. COMEX Gold's
electronic session runs nearly 24 hours but halts daily 5:00-6:00pm ET for
maintenance, and its official daily settlement price is fixed earlier, at
1:30pm ET. Converting both to Thailand time (ICT, UTC+7): the 5pm ET halt
is ~04:00 ICT the next calendar day, and the 1:30pm ET settlement is
~00:30 ICT — so 08:00 ICT lands comfortably after both (a 3-7 hour buffer
depending on which convention the data source's "daily bar" actually
follows, and on US daylight-saving), meaning the prior trading day's close
should reliably already be available by the time this function runs. This
is a deliberately chosen, checked buffer, not an arbitrary round number.

WHY INFERENCE-ONLY, NOT RETRAINING
-----------------------------------------------------------------------------
Retraining the walk-forward LSTM on this same daily schedule would re-run
the single most computationally expensive stage in the whole pipeline (see
main README §6, notebook 05) — the exact workload that caused CPU-credit
exhaustion on ml.t3.medium during development — every single day,
regardless of whether that much new data justifies it. Full walk-forward
retraining belongs on a cadence tied to how much new data has actually
accumulated — the pipeline already encodes this idea via
`RETRAIN_EVERY = 7` in notebook 05's walk-forward loop. The natural
extension is a *separate*, much-less-frequent scheduled job (weekly is a
reasonable starting point) that re-runs the full SageMaker retraining
pipeline — NOT this Lambda. This function's only job is to keep the
*forecast* fresh between those retraining cycles, using whatever model is
currently trained.

WHY AN ISOLATED WRITE PATH (bronze/streaming/, not bronze/xauusd_daily/raw/)
-----------------------------------------------------------------------------
The canonical bronze/silver/gold objects are owned by the manual notebook
pipeline (01-04) and everything downstream (05 training, 06 evaluation, the
dashboard) trusts their exact schema. An unattended daily Lambda and a
human occasionally re-running notebooks are two independent writers; if
both touched the same canonical files there's a real risk of a race (the
Lambda appending mid-write of a manual re-run, or a malformed row silently
corrupting a column the Lambda's writer doesn't know about — Silver in
particular carries 1W/1M-derived columns this function never re-derives).
So this function never writes to bronze/xauusd_daily/raw/,
silver/xauusd_daily_clean.parquet, or gold/xauusd_daily/features/
xauusd_features.parquet. It only ever reads Silver (read-only) and writes
to its own append-only log + a predictions/ output. A full notebook re-run
is what "absorbs" the streaming log into the canonical tables — this
function does not attempt to replace that step, only to bridge the gap
between runs of it.

WHY THIS FIXES A REAL BUG FOUND WHILE BUILDING IT (dropna() drops the
freshest row's FEATURES along with its undefined TARGET)
-----------------------------------------------------------------------------
notebook 03's `df.dropna()` has no `subset=`, so it drops ANY row with a
null in ANY column — including the most recent row, whose `target` (next
day's close) is always undefined at collection time. That row's FEATURES
(close, ma7, ma30, ...) are perfectly well-defined; only `target` is
missing. Dropping it means the persisted Gold table's last row is always
ONE trading day behind the true latest known close. Feeding
`Gold.tail(SEQ_LEN)` to the model (as the dashboard's `forecast_next_day()`
currently does) therefore predicts a value that is already fully knowable
from data already collected — not a genuine unknown-future forecast. This
function avoids the bug structurally: it builds its 60-day input window
from Silver's `close` series (which is never target-shifted or
dropna-filtered) plus any newer streaming rows, so the window's last row is
always the TRUE latest known trading day, and the resulting prediction is a
genuine next-day-ahead forecast. See the main README's automation section
for the full write-up — this is flagged there as a discovered issue worth
fixing in `dashboard/app.py` too, not only here.

RUNTIME
-----------------------------------------------------------------------------
Deployed as a container-image Lambda (see ../Dockerfile) so `tflite-runtime`
and `pandas` — both compiled C-extension packages — are guaranteed to match
Lambda's actual Amazon Linux runtime, rather than risking a "works on my
Mac, breaks in Lambda" mismatch from a hand-built zip layer. No `sklearn`
or `tensorflow` in this function at all: the trained model is converted to
TFLite and the two StandardScalers are reduced to their `mean_`/`scale_`
arrays ahead of time (see ../convert_and_export.py) — inference only needs
numpy arithmetic and the tiny TFLite interpreter, not the full training
stack.
"""

import json
import logging
import os
import urllib.request
from datetime import datetime, timezone
from io import BytesIO, StringIO

import boto3
import numpy as np
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Config — must match the S3_BUCKET / paths used by notebooks 01-06 and the
# dashboard (see main README §5, Data Architecture table).
# ---------------------------------------------------------------------------
S3_BUCKET = os.environ.get("S3_BUCKET", "gold-lstm-forecast")
REGION = os.environ.get("AWS_REGION", "ap-southeast-2")

SILVER_PATH = "silver/xauusd_daily_clean.parquet"                       # read-only
MODEL_DIR = "gold/xauusd_daily/features"                                # read-only artifacts live here
TFLITE_KEY = f"{MODEL_DIR}/lstm_model.tflite"
SCALER_PARAMS_KEY = f"{MODEL_DIR}/scaler_params.json"

STREAM_LOG_KEY = "bronze/streaming/xauusd_daily_incremental.csv"        # this function's own append-only log
RUN_LOG_KEY = "bronze/streaming/automation_log.csv"                     # one row per invocation, for observability
FORECAST_KEY = "gold/xauusd_daily/predictions/latest_forecast.json"     # this function's only "real" output

FEATURES = ["close", "return", "ma7", "ma14", "ma30", "ma60", "volatility_7", "momentum_7"]
SEQ_LEN = 60
# Trailing window fed into the rolling-feature computation. Needs to be >=
# SEQ_LEN (60) + the longest rolling window used (ma60, 60) so that even the
# FIRST row of the final 60-row forecast window has a fully-populated ma60 —
# i.e. >= 119. 130 leaves a comfortable margin without reading much extra data.
FEATURE_WARMUP_ROWS = 130

TICKER = "GC=F"
# range=10d, not just a few days: with a DAILY check (one shot/day, unlike an
# hourly check's 24 chances), a wider lookback gives more room to catch up
# cleanly if a run errors out or the function is disabled for a few days —
# append_to_streaming_log()/dedup-by-date makes replaying overlapping days safe.
YAHOO_CHART_URL = (
    f"https://query1.finance.yahoo.com/v8/finance/chart/{TICKER.replace('=', '%3D')}"
    "?range=10d&interval=1d"
)

s3 = boto3.client("s3", region_name=REGION)


# ---------------------------------------------------------------------------
# Step 1 — fetch the most recent daily bars directly from the same Yahoo
# Finance endpoint `yfinance` itself wraps, using only the standard library.
# Deliberately NOT the `yfinance` package: it pulls in extra dependencies
# and scraping logic this function doesn't need, and a single stdlib
# `urllib.request` call is far less to go wrong in an unattended container.
#
# KNOWN RISK (documented, not hidden): this is an unofficial, undocumented
# endpoint. Yahoo can change its shape or start blocking cloud/datacenter
# IPs without notice — this is exactly the kind of "real-world pipeline"
# fragility a static Kaggle CSV never has. A production system would use a
# licensed market-data API with an SLA instead; that swap only touches this
# one function, everything downstream (features, scaling, model) is
# unaffected by where the raw price came from.
# ---------------------------------------------------------------------------
def fetch_recent_bars():
    req = urllib.request.Request(
        YAHOO_CHART_URL,
        headers={"User-Agent": "Mozilla/5.0 (compatible; gold-lstm-forecast-bot/1.0)"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    result = payload.get("chart", {}).get("result")
    if not result:
        raise RuntimeError(f"Yahoo chart API returned no result: {payload.get('chart', {}).get('error')}")

    r = result[0]
    timestamps = r["timestamp"]
    quote = r["indicators"]["quote"][0]

    bars = pd.DataFrame({
        "date": pd.to_datetime(timestamps, unit="s", utc=True).tz_localize(None).normalize(),
        "open": quote["open"],
        "high": quote["high"],
        "low": quote["low"],
        "close": quote["close"],
        "volume": quote["volume"],
    })
    # Yahoo occasionally includes a trailing in-progress/unsettled bar with a
    # null close (today's session, still open) — drop anything incomplete
    # rather than risk feeding a half-formed price into the feature window.
    bars = bars.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    return bars


# ---------------------------------------------------------------------------
# Step 2 — load what this function already knows: Silver (read-only, the
# canonical cleaned close series) plus any rows already sitting in this
# function's own streaming log ahead of Silver's last date.
# ---------------------------------------------------------------------------
def load_known_close_series():
    silver_obj = s3.get_object(Bucket=S3_BUCKET, Key=SILVER_PATH)
    silver = pd.read_parquet(BytesIO(silver_obj["Body"].read()))
    silver["date"] = pd.to_datetime(silver["date"])
    silver = silver[["date", "close"]].sort_values("date").reset_index(drop=True)

    stream_tail = pd.DataFrame(columns=["date", "close"])
    try:
        log_obj = s3.get_object(Bucket=S3_BUCKET, Key=STREAM_LOG_KEY)
        stream = pd.read_csv(BytesIO(log_obj["Body"].read()), parse_dates=["date"])
        stream_tail = stream[stream["date"] > silver["date"].max()][["date", "close"]]
    except s3.exceptions.NoSuchKey:
        logger.info("No streaming log yet at s3://%s/%s — first run.", S3_BUCKET, STREAM_LOG_KEY)

    combined = pd.concat([silver.tail(FEATURE_WARMUP_ROWS + 10), stream_tail], ignore_index=True)
    combined = combined.drop_duplicates(subset="date", keep="last").sort_values("date").reset_index(drop=True)
    return combined


def append_to_streaming_log(new_rows: pd.DataFrame):
    """Idempotent append: only ever adds rows, keyed by date, never rewrites
    or reorders existing ones. Safe to invoke every day even when nothing
    new is found (this is only called when new_rows is non-empty)."""
    try:
        existing_obj = s3.get_object(Bucket=S3_BUCKET, Key=STREAM_LOG_KEY)
        existing = pd.read_csv(BytesIO(existing_obj["Body"].read()), parse_dates=["date"])
    except s3.exceptions.NoSuchKey:
        existing = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "fetched_at"])

    new_rows = new_rows.copy()
    new_rows["fetched_at"] = datetime.now(timezone.utc).isoformat()
    merged = pd.concat([existing, new_rows], ignore_index=True)
    merged = merged.drop_duplicates(subset="date", keep="last").sort_values("date").reset_index(drop=True)

    buf = StringIO()
    merged.to_csv(buf, index=False)
    s3.put_object(Bucket=S3_BUCKET, Key=STREAM_LOG_KEY, Body=buf.getvalue().encode("utf-8"))
    logger.info("Streaming log now has %d rows (added %d).", len(merged), len(new_rows))


def append_run_log(outcome: str, detail: str = ""):
    row = pd.DataFrame([{
        "invoked_at": datetime.now(timezone.utc).isoformat(),
        "outcome": outcome,
        "detail": detail,
    }])
    try:
        existing_obj = s3.get_object(Bucket=S3_BUCKET, Key=RUN_LOG_KEY)
        existing = pd.read_csv(BytesIO(existing_obj["Body"].read()))
        merged = pd.concat([existing, row], ignore_index=True).tail(500)  # cap growth
    except s3.exceptions.NoSuchKey:
        merged = row
    buf = StringIO()
    merged.to_csv(buf, index=False)
    s3.put_object(Bucket=S3_BUCKET, Key=RUN_LOG_KEY, Body=buf.getvalue().encode("utf-8"))


# ---------------------------------------------------------------------------
# Step 3 — recompute the 7 engineered features on the trailing window.
# Formulas copied verbatim from notebooks/03_Feature_Engineering.ipynb —
# any change there must be mirrored here, or the live forecast will silently
# diverge from what the model was actually trained on.
# ---------------------------------------------------------------------------
def compute_features(close_df: pd.DataFrame) -> pd.DataFrame:
    df = close_df.sort_values("date").reset_index(drop=True).copy()
    df["return"] = df["close"].pct_change()
    df["ma7"] = df["close"].rolling(7).mean()
    df["ma14"] = df["close"].rolling(14).mean()
    df["ma30"] = df["close"].rolling(30).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["volatility_7"] = df["return"].rolling(7).std()
    df["momentum_7"] = df["close"] - df["close"].shift(7)
    return df


# ---------------------------------------------------------------------------
# Step 4 — load the TFLite model + scaler params (small, cached across warm
# invocations at module scope would be ideal; kept simple/per-call here
# since this runs at most once/day, so the extra ~200ms S3 read is
# negligible against the 1-day period — a warm container is unlikely to
# even survive 24 hours between invocations anyway).
# ---------------------------------------------------------------------------
def load_model_and_scalers():
    # tflite_runtime is the primary target (see ../Dockerfile), but if the
    # container image was built with the tensorflow-cpu fallback instead
    # (documented there — used when no tflite-runtime wheel is available for
    # the chosen Python version/architecture), tensorflow.lite.Interpreter
    # is a drop-in replacement with an identical API. Trying both here means
    # this file never needs editing to match whichever path the Dockerfile
    # took.
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite

    model_obj = s3.get_object(Bucket=S3_BUCKET, Key=TFLITE_KEY)
    with open("/tmp/lstm_model.tflite", "wb") as f:
        f.write(model_obj["Body"].read())

    interpreter = tflite.Interpreter(model_path="/tmp/lstm_model.tflite")
    interpreter.allocate_tensors()

    params_obj = s3.get_object(Bucket=S3_BUCKET, Key=SCALER_PARAMS_KEY)
    params = json.loads(params_obj["Body"].read())
    return interpreter, params


def standardize(x: np.ndarray, mean_: list, scale_: list) -> np.ndarray:
    """Manual re-implementation of sklearn StandardScaler.transform(): the
    Lambda has no dependency on scikit-learn at all — the only thing
    inference needs from a fitted StandardScaler is its mean_ and scale_
    arrays, exported once by convert_and_export.py."""
    return (x - np.array(mean_)) / np.array(scale_)


def inverse_standardize(x_scaled: float, mean_: list, scale_: list) -> float:
    return float(x_scaled * scale_[0] + mean_[0])


def run_inference(interpreter, params, feature_window: pd.DataFrame) -> float:
    X = feature_window[FEATURES].to_numpy(dtype=np.float64)
    fs = params["feature_scaler"]
    X_scaled = standardize(X, fs["mean"], fs["scale"]).astype(np.float32).reshape(1, SEQ_LEN, len(FEATURES))

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], X_scaled)
    interpreter.invoke()
    pred_scaled = float(interpreter.get_tensor(output_details[0]["index"])[0][0])

    ts = params["target_scaler"]
    return inverse_standardize(pred_scaled, ts["mean"], ts["scale"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def lambda_handler(event, context):
    try:
        known = load_known_close_series()
        known_max_date = known["date"].max()

        fresh = fetch_recent_bars()
        new_rows = fresh[fresh["date"] > known_max_date]

        if new_rows.empty:
            logger.info("No new trading-day close since %s — nothing to do.", known_max_date.date())
            append_run_log("no_new_data", f"latest known date still {known_max_date.date()}")
            return {"statusCode": 200, "body": json.dumps({"status": "no_new_data", "as_of": str(known_max_date.date())})}

        append_to_streaming_log(new_rows)
        updated_close = pd.concat([known, new_rows[["date", "close"]]], ignore_index=True) \
                           .drop_duplicates(subset="date", keep="last") \
                           .sort_values("date").reset_index(drop=True)

        feature_table = compute_features(updated_close)
        forecast_window = feature_table.dropna(subset=FEATURES).tail(SEQ_LEN)
        if len(forecast_window) < SEQ_LEN:
            raise RuntimeError(
                f"Only {len(forecast_window)} fully-featured rows available, need {SEQ_LEN} — "
                "warm-up window too short (increase FEATURE_WARMUP_ROWS)."
            )

        interpreter, params = load_model_and_scalers()
        predicted_next_close = run_inference(interpreter, params, forecast_window)

        latest_known_close = float(forecast_window["close"].iloc[-1])
        result = {
            "as_of_date": str(forecast_window["date"].iloc[-1].date()),
            "latest_known_close": round(latest_known_close, 2),
            "predicted_next_close": round(predicted_next_close, 2),
            "delta": round(predicted_next_close - latest_known_close, 2),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "lambda-daily-refresh",
        }
        s3.put_object(
            Bucket=S3_BUCKET, Key=FORECAST_KEY,
            Body=json.dumps(result, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        logger.info("New forecast written: %s", result)
        append_run_log("new_data_processed", json.dumps(result))
        return {"statusCode": 200, "body": json.dumps(result)}

    except Exception as exc:  # noqa: BLE001 — top-level handler: log and fail loudly, never crash-loop silently
        logger.exception("Automation run failed")
        try:
            append_run_log("error", repr(exc))
        except Exception:
            pass  # don't let a logging failure mask the original error
        raise
