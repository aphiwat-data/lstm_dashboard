"""Cached data access. Results come from the repository (`daily/results`), the live pipeline's output and price history from S3.
Nothing here needs TensorFlow: every loader degrades to None / an empty frame when a source is unavailable."""
from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import streamlit as st

S3_BUCKET, REGION = "gold-lstm-forecast", "ap-southeast-2"
FORECAST_KEY = "gold/xauusd_daily/predictions/latest_forecast.json"
FORECAST_LOG_KEY = "gold/xauusd_daily/predictions/forecast_log.csv"
STREAM_LOG_KEY = "bronze/streaming/xauusd_daily_incremental.csv"
INPUTS_KEY = "gold/xauusd_daily/predictions/crossasset_inputs.csv"
HEADLINE = "xasset"  # the model the page is about when its results exist (README section 15.5); otherwise the page falls back to the LSTM
MODELS = {"xasset": {"name": "Cross-asset model", "short": "Cross-asset", "retrain_days": 21, "train_days": "1,000"},
          "lstm": {"name": "LSTM", "short": "LSTM", "retrain_days": 7, "train_days": "365"}}
SILVER_KEY = "silver/xauusd_daily_clean.parquet"
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "daily" / "results"


def _s3_bytes(key: str) -> bytes:
    return boto3.client("s3", region_name=REGION).get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()


def _stamp(*names: str) -> float:
    """Latest modification time of result files: part of the cache key, so a refreshed result shows up without a restart."""
    return max((((RESULTS / n).stat().st_mtime) for n in names if (RESULTS / n).exists()), default=0.0)


def headline_key(res: dict | None, log: pd.DataFrame | None, live: dict | None) -> str:
    """The model whose forecast, numbers and charts the page shows. It is the cross-asset model when its results are in the repository, the live log has its forecasts AND the
    pipeline's latest run published one ("crossasset" in the live file); otherwise the page is the LSTM page (also on a day when only the cross-asset step failed), so it can
    never show a stale cross-asset forecast."""
    started = log is not None and LIVE_COLUMNS[HEADLINE][0] in log.columns and bool(log[LIVE_COLUMNS[HEADLINE][0]].notna().any())
    return HEADLINE if res and HEADLINE in res.get("models", {}) and started and (live or {}).get("crossasset") else "lstm"


# ------------------------------------------------------------------------------------------------------------------ live pipeline (S3)
@st.cache_data(ttl=300, show_spinner=False)
def live_forecast() -> dict | None:
    try:
        return json.loads(_s3_bytes(FORECAST_KEY))
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def live_log() -> pd.DataFrame | None:
    try:
        return pd.read_csv(BytesIO(_s3_bytes(FORECAST_LOG_KEY)), parse_dates=["as_of_date"])
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def crossasset_inputs() -> pd.DataFrame | None:
    """The last 130 days of the cross-asset model's six inputs, written by the daily pipeline (None before its first run or when S3 is unreachable)."""
    try:
        return pd.read_csv(BytesIO(_s3_bytes(INPUTS_KEY)), parse_dates=["date"]).set_index("date")
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def stream_closes() -> pd.DataFrame:
    try:
        return pd.read_csv(BytesIO(_s3_bytes(STREAM_LOG_KEY)), parse_dates=["date"])[["date", "close"]]
    except Exception:
        return pd.DataFrame(columns=["date", "close"])


@st.cache_data(ttl=3600, show_spinner=False)
def prices() -> pd.DataFrame | None:
    """Daily close and volume: the Silver table plus the live pipeline's newer bars. None when S3 is unreachable."""
    try:
        silver = pd.read_parquet(BytesIO(_s3_bytes(SILVER_KEY)), columns=["date", "close", "volume"])
        silver["date"] = pd.to_datetime(silver["date"])
        try:
            extra = pd.read_csv(BytesIO(_s3_bytes(STREAM_LOG_KEY)), parse_dates=["date"])[["date", "close", "volume"]]
            silver = pd.concat([silver, extra[extra["date"] > silver["date"].max()]], ignore_index=True)
        except Exception:
            pass
        return silver.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    except Exception:
        return None


# ------------------------------------------------------------------------------------------------------------ results in the repository
@st.cache_data(show_spinner=False)
def _load(name: str, stamp: float):
    p = RESULTS / name
    try:
        return pd.read_csv(p, parse_dates=["date"]) if name.endswith(".csv") else json.loads(p.read_text())
    except Exception:
        return None


def results() -> tuple[dict | None, pd.DataFrame | None]:
    """Walk-forward results and predictions of the main run."""
    s = _stamp("walk_forward_results.json", "wf_predictions.csv")
    return _load("walk_forward_results.json", s), _load("wf_predictions.csv", s)


def regression() -> dict | None:
    return _load("regression_metrics.json", _stamp("regression_metrics.json"))


def long_window() -> dict | None:
    """The cross-asset model's result on the longer test of the frozen search E2 (about 1,700 days since 2020), for the R² tooltip. None when the file is not there."""
    s = _load("paper_e2/e2_summary.json", _stamp("paper_e2/e2_summary.json"))
    try:
        t = s["tracks"]["h1"]
        c = t["primary"]["P-A dev-selected best"]["test"]
        return {"n": int(t["n_test"]), "year": int(str(t["test_start"])[:4]), "r2": float(c["r2_vs_b"]), "p": float(c["cw_p"])}
    except Exception:
        return None


def volatility() -> tuple[pd.DataFrame, dict] | None:
    s = _stamp("volatility_forecast.csv", "volatility_comparison.json")
    frame, comp = _load("volatility_forecast.csv", s), _load("volatility_comparison.json", s)
    return (frame, comp) if frame is not None and comp is not None else None


def interval_tests(key: str = "lstm") -> tuple[list | None, list | None]:
    """Range-method comparison for the page's model: the cross-asset model has its own file (no quantile network or VIX variant); the LSTM keeps the original two."""
    if key == "xasset":
        return _load("interval_comparison_xasset.json", _stamp("interval_comparison_xasset.json")), None
    s = _stamp("interval_comparison.json", "interval_vix.json")
    return _load("interval_comparison.json", s), _load("interval_vix.json", s)


# --------------------------------------------------------------------------------------------------------------- what the page shows now
def next_trading_day(d) -> pd.Timestamp:
    return pd.Timestamp(d) + pd.offsets.BDay(1)


def current_forecast(res: dict | None, key: str = "lstm") -> dict | None:
    """The live pipeline's forecast of the page's model when it is at least as new as the one stored with the last walk-forward run, otherwise the stored one.
    The live file holds the LSTM's forecast at the top level and the cross-asset model's under "crossasset"."""
    stored = None if not res else (res.get("latest_forecast_xasset") if key == HEADLINE else res["latest_forecast"])
    live = live_forecast()
    mine = None if live is None else (live.get("crossasset") if key == HEADLINE else live)
    if mine and (stored is None or pd.Timestamp(mine["as_of_date"]) >= pd.Timestamp(stored["as_of_date"])):
        return {**mine, "source": "live"}
    return {**stored, "source": "backtest"} if stored else None


def price_history(preds: pd.DataFrame | None, cur: dict) -> pd.DataFrame:
    """Closes for the forecast chart: the walk-forward run's closes plus everything newer."""
    base = preds[["date", "meta_close"]].rename(columns={"meta_close": "close"}) if preds is not None else pd.DataFrame(columns=["date", "close"])
    newer = stream_closes()
    hist = pd.concat([base, newer[newer["date"] > base["date"].max()]] if len(base) else [newer], ignore_index=True)
    as_of = pd.Timestamp(cur["as_of_date"])
    hist = hist[hist["date"] <= as_of]  # nothing newer than the forecast's own day: a stored (backtest) forecast must not sit beside closes that came after it
    if len(hist) == 0 or as_of > hist["date"].max():
        hist = pd.concat([hist, pd.DataFrame({"date": [as_of], "close": [cur["latest_close"]]})], ignore_index=True)
    return hist.sort_values("date").reset_index(drop=True)


LIVE_COLUMNS = {"lstm": ("predicted_next_close", "lo_90", "hi_90", "inside_90"), "xasset": ("x_predicted_next_close", "x_lo_90", "x_hi_90", "x_inside_90")}


def live_scores(log: pd.DataFrame | None, key: str) -> pd.DataFrame | None:
    """The live log seen from one model: the columns predicted_next_close, lo_90, hi_90 and inside_90 hold that model's forecast (rows without one are dropped)."""
    if log is None or len(log) == 0 or LIVE_COLUMNS[key][0] not in log.columns:
        return None
    fc, lo, hi, ins = LIVE_COLUMNS[key]
    d = log.dropna(subset=[fc]).copy()
    return d.assign(predicted_next_close=d[fc], lo_90=d[lo], hi_90=d[hi], inside_90=d[ins]) if key != "lstm" else d


def replayed(preds: pd.DataFrame | None, key: str) -> pd.DataFrame | None:
    """The cross-asset forecasts between the end of the backtest and the start of its live record, replayed like the backtest (the committed E2 walk-forward forecasts, then the
    days up to the first live one from xasset_replay_bridge.csv), with the 90% range sized from the errors known before each day. Source "replay": the header chart shows
    them, the 449-day backtest does not."""
    if key != HEADLINE or preds is None:
        return None
    files = [_load(name, _stamp(name)) for name in ("paper_e2/e2_frozen_forecasts_h1.csv", "xasset_replay_bridge.csv")]
    if files[0] is None:
        return None
    new = pd.concat([f for f in files if f is not None], ignore_index=True).drop_duplicates("date").sort_values("date")
    new = new[new["date"] > preds["date"].max()].reset_index(drop=True)
    if new.empty:
        return None
    errors = list((preds["y_ret_next"] - preds[f"pred_ret_{key}"]).to_numpy(float))
    forecast = new["close"] * np.exp(new["dev_best"])
    lo, hi = [], []
    for fc, y, f in zip(forecast, new["y"], new["dev_best"]):
        e = np.asarray(errors)
        w = 0.94 ** np.arange(len(e))[::-1]
        z = 1.6449 * np.sqrt(np.sum(w * e**2) / w.sum())
        lo.append(fc * np.exp(-z)), hi.append(fc * np.exp(z))
        errors.append(y - f)
    for_date = new["date"].shift(-1)
    for_date.iloc[-1] = new["date"].iloc[-1] + pd.offsets.BDay(1)
    return pd.DataFrame({"for_date": for_date, "forecast": forecast, "lo": lo, "hi": hi, "actual": new["next_close"], "source": "replay"})


def forecast_history(preds: pd.DataFrame | None, log: pd.DataFrame | None, vdf: pd.DataFrame | None, key: str = "lstm") -> pd.DataFrame:
    """Every past forecast of the page's model placed on the day it was FOR, beside the actual close of that day.

    Columns: for_date, forecast, lo, hi (the 90% range, NaN when unknown), actual, miss (actual - forecast) and source ("backtest", "replay" or "live").
    The walk-forward backtest comes first, then (for the cross-asset model) the later replayed days; the live log (forecasts the pipeline saved before the day) wins when several cover a day."""
    parts = []
    if preds is not None and len(preds):
        fc = preds["meta_close"] * np.exp(preds[f"pred_ret_{key}"])
        b = pd.DataFrame({"for_date": preds["date"].shift(-1), "forecast": fc, "lo": np.nan, "hi": np.nan, "actual": preds["meta_next_close"], "source": "backtest"})
        b.loc[b.index[-1], "for_date"] = preds["date"].iloc[-1] + pd.offsets.BDay(1)  # the last test row's next day is not in the table
        if vdf is not None:  # 90% range = forecast x e^(+-1.645 sigma), sigma = the EWMA of past errors stored with the volatility results
            z = 1.6449 * vdf.set_index("date")["sigma_ewma_pct"].reindex(preds["date"]).to_numpy() / 100
            b["lo"], b["hi"] = fc * np.exp(-z), fc * np.exp(z)
        parts.append(b)
    extra = replayed(preds, key)
    if extra is not None:
        parts.append(extra)
    mine = live_scores(log, key)
    if mine is not None and len(mine):
        s = mine.dropna(subset=["actual_next_close"])
        parts.append(pd.DataFrame({"for_date": pd.to_datetime(s["next_date"]), "forecast": s["predicted_next_close"], "lo": s["lo_90"], "hi": s["hi_90"], "actual": s["actual_next_close"], "source": "live"}))
    if not parts:
        return pd.DataFrame(columns=["for_date", "forecast", "lo", "hi", "actual", "miss", "source"])
    out = pd.concat(parts, ignore_index=True)
    out["for_date"] = pd.to_datetime(out["for_date"])
    out = out.drop_duplicates("for_date", keep="last").sort_values("for_date").reset_index(drop=True)
    out["miss"] = out["actual"] - out["forecast"]
    return out


def run_age_hours(cur: dict) -> float | None:
    try:
        ts = pd.Timestamp(cur["generated_at"]).tz_localize(None)
        return (pd.Timestamp.now(tz="UTC").tz_localize(None) - ts).total_seconds() / 3600
    except Exception:
        return None


def indicators(close: pd.Series) -> pd.DataFrame | None:
    """The model's own input features (from daily/build_daily_dataset.py), computed from the close series."""
    try:
        sys.path.insert(0, str(ROOT / "daily"))
        from build_daily_dataset import make_features

        d = pd.DataFrame({"close": close.to_numpy(float)}, index=pd.DatetimeIndex(close.index))
        return make_features(d)
    except Exception:
        return None
