"""Cached data access. Results come from the repository (`daily/results`), the live pipeline's output and price history from S3.
Nothing here needs TensorFlow: every loader degrades to None / an empty frame when a source is unavailable."""
from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path

import boto3
import pandas as pd
import streamlit as st

S3_BUCKET, REGION = "gold-lstm-forecast", "ap-southeast-2"
FORECAST_KEY = "gold/xauusd_daily/predictions/latest_forecast.json"
FORECAST_LOG_KEY = "gold/xauusd_daily/predictions/forecast_log.csv"
STREAM_LOG_KEY = "bronze/streaming/xauusd_daily_incremental.csv"
SILVER_KEY = "silver/xauusd_daily_clean.parquet"
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "daily" / "results"
REPO_URL = "https://github.com/aphiwat-data/lstm_dashboard"


def _s3_bytes(key: str) -> bytes:
    return boto3.client("s3", region_name=REGION).get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()


def _stamp(*names: str) -> float:
    """Latest modification time of result files: part of the cache key, so a refreshed result shows up without a restart."""
    return max((((RESULTS / n).stat().st_mtime) for n in names if (RESULTS / n).exists()), default=0.0)


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


def volatility() -> tuple[pd.DataFrame, dict] | None:
    s = _stamp("volatility_forecast.csv", "volatility_comparison.json")
    frame, comp = _load("volatility_forecast.csv", s), _load("volatility_comparison.json", s)
    return (frame, comp) if frame is not None and comp is not None else None


def interval_tests() -> tuple[list | None, list | None]:
    s = _stamp("interval_comparison.json", "interval_vix.json")
    return _load("interval_comparison.json", s), _load("interval_vix.json", s)


def horizons() -> dict:
    out = {}
    for h in (5, 20):
        r = _load(f"horizon_h{h}/walk_forward_results.json", _stamp(f"horizon_h{h}/walk_forward_results.json"))
        if r:
            out[h] = r
    return out


FEATURE_SETS = [("Price history only", ""), ("+ macro and cross-asset prices", "with_macro"), ("+ NFP release days", "with_macro_calendar"),
                ("+ FOMC and CPI release days", "with_macro_calendar_full"), ("+ CFTC positioning", "with_cot")]


def feature_rows() -> list[dict]:
    """One row per feature-search experiment, read from the committed results (nothing typed in by hand)."""
    rows = []
    for label, folder in FEATURE_SETS:
        pre = f"{folder}/" if folder else ""
        s = _stamp(f"{pre}walk_forward_results.json", f"{pre}regression_metrics.json")
        res, reg = _load(f"{pre}walk_forward_results.json", s), _load(f"{pre}regression_metrics.json", s)
        if not res or not reg:
            continue
        m, hd_all = reg["models"]["lstm"], res["models"]["lstm"]
        hd = hd_all["direction_from_head"]
        rows.append({"label": label, "n": res["protocol"]["n_test"], "mae": m["mae_usd"], "naive_mae": reg["no_change"]["mae_usd"], "dm_p": hd_all["dm_p_vs_naive"],
                     "ic": m["ic_spearman"], "r2_os": m["r2_os_vs_rolling_mean"], "acc": hd["acc"], "auc": hd.get("auc"), "p_dir": hd["p_vs_always_up"],
                     "says_up": m["says_up_share"]})
    return rows


# --------------------------------------------------------------------------------------------------------------- what the page shows now
def next_trading_day(d) -> pd.Timestamp:
    return pd.Timestamp(d) + pd.offsets.BDay(1)


def current_forecast(res: dict | None) -> dict | None:
    """The live pipeline's forecast when it is newer than the one stored with the last walk-forward run, otherwise the stored one."""
    stored = res["latest_forecast"] if res else None
    live = live_forecast()
    if live is not None and (stored is None or pd.Timestamp(live["as_of_date"]) > pd.Timestamp(stored["as_of_date"])):
        return {**live, "source": "live"}
    return {**stored, "source": "backtest"} if stored else None


def price_history(preds: pd.DataFrame | None, cur: dict) -> pd.DataFrame:
    """Closes for the forecast chart: the walk-forward run's closes plus everything newer."""
    base = preds[["date", "meta_close"]].rename(columns={"meta_close": "close"}) if preds is not None else pd.DataFrame(columns=["date", "close"])
    newer = stream_closes()
    hist = pd.concat([base, newer[newer["date"] > base["date"].max()]] if len(base) else [newer], ignore_index=True)
    as_of = pd.Timestamp(cur["as_of_date"])
    if len(hist) == 0 or as_of > hist["date"].max():
        hist = pd.concat([hist, pd.DataFrame({"date": [as_of], "close": [cur["latest_close"]]})], ignore_index=True)
    return hist.sort_values("date").reset_index(drop=True)


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
