"""The cross-asset model: a ridge regression on the day's moves in gold, the dollar, silver, stocks, the 10-year yield and inflation-linked bonds (TIP).

It is the forecaster that won the frozen search E2 (README section 15.8: ridge, alpha 100, the last 1,000 labelled days, six inputs "X"), now also served live. The
daily Lambda, the results script and the tests share this module; it needs only numpy and pandas.

Inputs (all causal, computed from closes up to the forecast day; each macro series is aligned to gold's calendar with its latest close at or before the gold date):
  f_ret_1 (gold's log return), f_dxy_ret_1, f_silver_ret_1, f_spx_ret_1, f_tip_ret_1 (log returns), f_yield10y_chg_1 (change of the 10-year yield in percentage points).
Model: standardise the inputs with the training mean and standard deviation, scale the next-day log return by its training standard deviation, ridge with an unpenalised
intercept (the same arithmetic as sklearn's StandardScaler + Ridge(alpha=100); a test checks it), forecast price = close * exp(predicted log return).
Timing caveat (README section 15.5, E2d): gold's daily price is the 13:30 New York settlement, the S&P 500 and TIP close at 16:00 and the dollar index and yield about 17:00, so
part of what the inputs carry is news that arrived after gold's price was set. The forecast is made after the closes, so it is a legitimate forecast of the next settlement, but
it is not a forecast of news the market has not seen.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from build_daily_dataset import align_macro
from serving import band

FEATURES = ["f_ret_1", "f_dxy_ret_1", "f_silver_ret_1", "f_spx_ret_1", "f_yield10y_chg_1", "f_tip_ret_1"]
LABELS = {"f_ret_1": "Gold", "f_dxy_ret_1": "Dollar index (DXY)", "f_silver_ret_1": "Silver", "f_spx_ret_1": "S&P 500", "f_yield10y_chg_1": "10-year yield", "f_tip_ret_1": "Inflation-linked bonds (TIP)"}
TICKERS = {"dxy": "DX-Y.NYB", "silver": "SI=F", "spx": "^GSPC", "yield10y": "^TNX", "tip": "TIP"}
MACRO_COLS = list(TICKERS)
ALPHA, TRAIN_LEN, MIN_TRAIN = 100.0, 1000, 250
MODEL_NAME = "Cross-asset model"


# --------------------------------------------------------------------------------------------------------------------------------- features
def macro_features(close: pd.Series, macro: pd.DataFrame) -> pd.DataFrame:
    """The six inputs on gold's calendar. `close` is indexed by gold's dates (sorted); `macro` has one column per MACRO_COLS indexed by its own dates."""
    aligned = align_macro(pd.DatetimeIndex(close.index), macro[MACRO_COLS])
    f = pd.DataFrame(index=close.index)
    f["f_ret_1"] = np.log(close).diff(1)
    for name in ("dxy", "silver", "spx"):
        f[f"f_{name}_ret_1"] = np.log(aligned[name]).diff(1)
    f["f_yield10y_chg_1"] = aligned["yield10y"].diff(1)
    f["f_tip_ret_1"] = np.log(aligned["tip"]).diff(1)
    return f[FEATURES]


# ----------------------------------------------------------------------------------------------------------------------------------- model
def fit(X: np.ndarray, y: np.ndarray, alpha: float = ALPHA) -> dict:
    """Ridge on standardised inputs and a scaled target, intercept unpenalised (equals StandardScaler + Ridge(alpha) on y / std(y), rescaled)."""
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    Z = (X - mu) / sd
    ret_sd = float(y.std())
    ys = y / ret_sd
    ym = float(ys.mean())
    coef = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (ys - ym))
    return {"mu": mu, "sd": sd, "coef": coef, "intercept": ym, "ret_sd": ret_sd, "alpha": alpha, "n_train": int(len(y))}


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    return (((np.atleast_2d(X) - model["mu"]) / model["sd"]) @ model["coef"] + model["intercept"]) * model["ret_sd"]


def next_day_returns(close: pd.Series) -> pd.Series:
    """Row t holds log(close[t+1] / close[t]); the last row is NaN (its label does not exist yet)."""
    return np.log(close.shift(-1) / close)


def forecast_latest(close: pd.Series, macro: pd.DataFrame, errors: np.ndarray, alpha: float = ALPHA, train_len: int = TRAIN_LEN) -> dict:
    """Tomorrow's forecast from the newest gold bar. The model is refit on the last `train_len` rows whose next-day return is known; `errors` are the realised
    log-return errors of this model so far (oldest first) and size the 90% band (Gaussian, EWMA 0.94, as for every forecast on the page)."""
    feats = macro_features(close, macro)
    last = feats.index[-1]
    x_last = feats.iloc[-1]
    if not np.isfinite(x_last.to_numpy(float)).all():
        raise ValueError(f"inputs are not available for {last.date()}: missing {[k for k, v in x_last.items() if not np.isfinite(v)]}")
    y = next_day_returns(close)
    ok = np.isfinite(feats.to_numpy(float)).all(axis=1) & np.isfinite(y.to_numpy(float))
    rows = np.flatnonzero(ok)[-train_len:]
    if len(rows) < MIN_TRAIN:
        raise ValueError(f"only {len(rows)} labelled rows with complete inputs (need {MIN_TRAIN})")
    model = fit(feats.to_numpy(float)[rows], y.to_numpy(float)[rows], alpha)
    r_hat = float(predict(model, x_last.to_numpy(float))[0])
    last_close = float(close.iloc[-1])
    return {"as_of_date": str(last.date()), "latest_close": round(last_close, 2), "predicted_return": r_hat, "predicted_next_close": round(last_close * float(np.exp(r_hat)), 2),
            "interval_90_close": band(np.asarray(errors, float), last_close, r_hat), "n_train": int(len(rows)), "trained_through": str(close.index[rows[-1] + 1].date()),
            "inputs": {k: float(v) for k, v in x_last.items()}, "alpha": alpha, "train_len": train_len}


# ------------------------------------------------------------------------------------------------------------------------------- Yahoo chart
def parse_chart(payload: dict) -> pd.Series:
    """Daily closes of one ticker from Yahoo's chart JSON, indexed by the date in the EXCHANGE's time zone (what yfinance labels the bar with)."""
    result = payload.get("chart", {}).get("result")
    if not result:
        raise RuntimeError(f"Yahoo chart API returned no result: {payload.get('chart', {}).get('error')}")
    r = result[0]
    tz = ZoneInfo(r.get("meta", {}).get("exchangeTimezoneName") or "UTC")
    stamps = pd.to_datetime(r["timestamp"], unit="s", utc=True).tz_convert(tz)
    s = pd.Series(r["indicators"]["quote"][0]["close"], index=pd.DatetimeIndex(stamps.tz_localize(None).normalize()), dtype=float)
    return s.dropna()[~s.dropna().index.duplicated(keep="last")].sort_index()


def _http_json(url: str, timeout: float = 10.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; gold-lstm-forecast-bot/2.0)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_one(ticker: str, range_: str, fetch: Callable[[str], dict], attempts: int, pause: float) -> pd.Series:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ticker, safe='')}?range={range_}&interval=1d"
    for k in range(attempts):
        try:
            return parse_chart(fetch(url))
        except Exception:  # noqa: BLE001 - retried, then raised
            if k == attempts - 1:
                raise
            time.sleep(pause * (k + 1))
    raise AssertionError("unreachable")


def fetch_recent_macro(range_: str, now_utc: datetime | None = None, fetch: Callable[[str], dict] = _http_json, attempts: int = 3, pause: float = 1.5) -> pd.DataFrame:
    """The latest daily closes of the five macro tickers (columns MACRO_COLS, indexed by date); a bar is kept only once its session is over (New York time: an earlier
    date, or today after 17:00), like the gold bars. The tickers are fetched in parallel, each tried `attempts` times (at most about 35 s in all with the defaults);
    if one still fails the error is raised and the caller decides what to do."""
    now_et = (now_utc or datetime.now(timezone.utc)).astimezone(ZoneInfo("America/New_York"))
    today = pd.Timestamp(now_et.date())
    with ThreadPoolExecutor(max_workers=len(TICKERS)) as pool:
        futures = {name: pool.submit(_fetch_one, ticker, range_, fetch, attempts, pause) for name, ticker in TICKERS.items()}
        cols = {name: f.result() for name, f in futures.items()}
    cols = {name: s[(s.index < today) | ((s.index == today) & (now_et.hour >= 17))] for name, s in cols.items()}
    return pd.DataFrame(cols).sort_index()
