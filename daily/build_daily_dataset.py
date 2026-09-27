"""
Daily XAU/USD (GC=F) dataset builder: scale-free features, return targets, causal by construction.

Why this replaces the notebook 03-05 setup: those notebooks predict the next-day close PRICE from price-level
inputs (close, MA7/14/30/60, USD momentum) scaled with train-only statistics, so once gold traded far above the
training range (2025-26, $4,000+) both inputs and target fell outside what the scalers/model had seen.
Here every f_* feature is a return, ratio or oscillator and targets are log-returns.

Convention: row t is trading day t. f_* use days <= t only; y_* describe day t+h; meta_* are bookkeeping.
The silver table's *_1w / *_1m columns are deliberately NOT used: they hold the period's FINAL values, so on a
Wednesday they contain Friday's close (future information).

Usage: python build_daily_dataset.py [--silver s3://gold-lstm-forecast/silver/xauusd_daily_clean.parquet] --out-dir data/processed
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SILVER = "s3://gold-lstm-forecast/silver/xauusd_daily_clean.parquet"
HORIZONS = (5,)  # in trading days, in addition to the next-day target
SPLITS = {"train_end": "2023-12-31", "val_end": "2024-12-31"}  # test = 2025-01-01 onward
EMBARGO_BARS = 5  # must be >= max(HORIZONS)
MAX_END_LAG_DAYS = 5
EXTREME_LOGRET = 0.07

# Cross-asset/macro tickers (Phase 8, README §12 item 13): none of this is derivable from gold's own price
# history, which is the ceiling the price-only feature set already ran into (AUC 0.50-0.52, every model).
MACRO_TICKERS = {
    "dxy": "DX-Y.NYB",     # US Dollar Index - gold is USD-priced, inverse relationship expected
    "silver": "SI=F",      # precious-metals complex co-movement + gold/silver ratio
    "spx": "^GSPC",        # risk-on/risk-off sentiment
    "vix": "^VIX",         # fear gauge - gold's safe-haven demand
    "yield10y": "^TNX",    # nominal rate proxy (opportunity cost of holding non-yielding gold)
    "tip": "TIP",          # TIPS ETF price, inverse proxy for real yield (gold's most cited driver)
}


COT_API = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"  # CFTC Legacy COT (Socrata), public, no auth
COT_MARKET = "GOLD - COMMODITY EXCHANGE INC."
COT_PUBLICATION_LAG_DAYS = 3  # report is "as of" Tuesday but not released until the following Friday


def _parse_cot_rows(rows: list[dict]) -> pd.DataFrame:
    """Pure transform, unit-testable without a network call. Index is the AVAILABILITY date (report date
    + COT_PUBLICATION_LAG_DAYS), not the report's own "as of" date -- the report covers positions as of
    Tuesday but the CFTC does not publish it until the following Friday, so using the Tuesday date
    directly would leak 3 days of future information into the alignment."""
    df = pd.DataFrame(rows)
    report_date = pd.to_datetime(df["report_date_as_yyyy_mm_dd"]).dt.tz_localize(None)
    oi = df["open_interest_all"].astype(float).to_numpy()
    long_ = df["noncomm_positions_long_all"].astype(float).to_numpy()
    short_ = df["noncomm_positions_short_all"].astype(float).to_numpy()
    avail_date = (report_date + pd.Timedelta(days=COT_PUBLICATION_LAG_DAYS)).to_numpy()
    out = pd.DataFrame({"cot_net_pct": (long_ - short_) / oi}, index=pd.DatetimeIndex(avail_date))
    return out[~out.index.duplicated(keep="last")].sort_index()


def fetch_cot(cache: Path | None = None) -> pd.DataFrame:
    """Weekly CFTC Commitments of Traders positioning for COMEX gold: net non-commercial (speculator)
    position as a share of open interest, the standard sentiment/positioning proxy."""
    if cache is not None and cache.exists():
        return pd.read_parquet(cache)
    import requests

    params = {
        "$where": f"market_and_exchange_names='{COT_MARKET}'",
        "$select": "report_date_as_yyyy_mm_dd,open_interest_all,noncomm_positions_long_all,noncomm_positions_short_all",
        "$order": "report_date_as_yyyy_mm_dd ASC",
        "$limit": 5000,
    }
    rows = requests.get(COT_API, params=params, timeout=30).json()
    out = _parse_cot_rows(rows)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache)
    return out


def make_cot_features(aligned: pd.DataFrame) -> pd.DataFrame:
    """The net-positioning ratio is already bounded/stationary (never a raw unbounded level), so unlike the
    price-based macro features this one IS used at its own level, plus a rolling z-score (extreme
    positioning is the classic contrarian-signal framing) and short/medium-term changes."""
    f = pd.DataFrame(index=aligned.index)
    s = aligned["cot_net_pct"]
    f["f_cot_net_pct"] = s
    roll = s.rolling(260, min_periods=52)  # ~52 trading weeks
    f["f_cot_net_z_52w"] = (s - roll.mean()) / roll.std()
    f["f_cot_net_chg_4w"] = s.diff(20)   # ~4 trading weeks
    f["f_cot_net_chg_12w"] = s.diff(60)  # ~12 trading weeks
    return f


def load_silver(src: str) -> pd.DataFrame:
    if src.startswith("s3://"):
        import boto3

        bucket, key = src[5:].split("/", 1)
        body = boto3.client("s3", region_name="ap-southeast-2").get_object(Bucket=bucket, Key=key)["Body"].read()
        return pd.read_parquet(io.BytesIO(body))
    return pd.read_parquet(src)


def clean_and_report(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = raw[["date", "open", "high", "low", "close", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    rep: dict = {"rows_raw": int(len(df)), "duplicate_dates": int(df["date"].duplicated().sum())}
    df = df.drop_duplicates("date", keep="last").sort_values("date").set_index("date")
    bad_ohlc = (df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1))
    rep["invalid_ohlc_rows"] = int(bad_ohlc.sum())  # informational: concentrated in 2004-2011; O/H/L are not used, so rows are kept
    rep["invalid_ohlc_share_by_year"] = {int(y): round(float(v), 3) for y, v in bad_ohlc.groupby(df.index.year).mean().items() if v > 0}
    nonpositive = df["close"] <= 0
    rep["nonpositive_close_rows"] = int(nonpositive.sum())
    df = df[~nonpositive]  # only an unusable close is dropped: dropping rows would create multi-day "1-day" returns
    rep["zero_volume_share"] = round(float((df["volume"] <= 0).mean()), 4)  # volume is not used as a feature
    gaps = df.index.to_series().diff().dt.days.dropna()
    rep["gaps_gt_5_days"] = [{"before": str(d.date()), "days": int(g)} for d, g in gaps[gaps > 5].items()][:10]
    lr = np.log(df["close"]).diff().dropna()
    rep["extreme_1day_moves"] = [{"date": str(d.date()), "logret": round(float(x), 4)} for d, x in lr[lr.abs() > EXTREME_LOGRET].items()][:10]
    lag = (pd.Timestamp.now() - df.index.max()).days
    rep.update(rows_clean=int(len(df)), start=str(df.index.min().date()), end=str(df.index.max().date()))
    rep["coverage_warnings"] = [f"data ends {df.index.max():%Y-%m-%d}, {lag} days ago (tail missing?)"] if lag > MAX_END_LAG_DAYS else []
    return df, rep


def fetch_macro(start: str, end: str, cache: Path | None = None) -> pd.DataFrame:
    """One Close column per MACRO_TICKERS series, each on its OWN native trading calendar (indices, bonds and
    commodities don't all close on the same days as COMEX gold). Cached to parquet since yfinance is a live
    network call and re-fetching the full 2004-present history on every build is unnecessary."""
    if cache is not None and cache.exists():
        return pd.read_parquet(cache)
    import yfinance as yf

    cols = {}
    for name, ticker in MACRO_TICKERS.items():
        h = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        cols[name] = h["Close"][ticker] if isinstance(h.columns, pd.MultiIndex) else h["Close"]
    macro = pd.DataFrame(cols)
    macro.index = pd.to_datetime(macro.index).tz_localize(None)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        macro.to_parquet(cache)
    return macro


def align_macro(gold_index: pd.DatetimeIndex, macro: pd.DataFrame) -> pd.DataFrame:
    """As-of (backward) align each macro series onto gold's trading calendar: row t gets the most recent macro
    close AT OR BEFORE gold day t, never a later one, so a holiday/weekend mismatch between markets (e.g. a
    bond-market holiday when COMEX trades) can only look backward, never leak a future macro value."""
    out = pd.DataFrame(index=gold_index)
    for col in macro.columns:
        s = macro[col].dropna().sort_index()
        pos = s.index.searchsorted(gold_index, side="right") - 1
        val = np.where(pos >= 0, s.to_numpy()[np.clip(pos, 0, len(s) - 1)], np.nan)
        out[col] = val
    return out


FOMC_DATES = [  # decision day (last day of each meeting/conference call); source: federalreserve.gov
    # historical-materials-by-year pages (fomchistorical{year}.htm, 2004-2020) and fomccalendars.htm
    # (2021-2027), fetched 2026-09-27. Notation votes (procedural, no new statement) are excluded.
    "2004-01-28", "2004-03-16", "2004-05-04", "2004-06-30", "2004-08-10", "2004-09-21", "2004-11-10", "2004-12-14",
    "2005-02-02", "2005-03-22", "2005-05-03", "2005-06-30", "2005-08-09", "2005-09-20", "2005-11-01", "2005-12-13",
    "2006-01-31", "2006-03-28", "2006-05-10", "2006-06-29", "2006-08-08", "2006-09-20", "2006-10-25", "2006-12-12",
    "2007-01-31", "2007-03-21", "2007-05-09", "2007-06-28", "2007-08-07", "2007-08-10", "2007-08-16", "2007-09-18",
    "2007-10-31", "2007-12-06", "2007-12-11",
    "2008-01-09", "2008-01-21", "2008-01-30", "2008-03-10", "2008-03-18", "2008-04-30", "2008-06-25", "2008-07-24",
    "2008-08-05", "2008-09-16", "2008-09-29", "2008-10-07", "2008-10-29", "2008-12-16",
    "2009-01-16", "2009-01-28", "2009-02-07", "2009-03-18", "2009-04-29", "2009-06-03", "2009-06-24", "2009-08-12",
    "2009-09-23", "2009-11-04", "2009-12-16",
    "2010-01-27", "2010-03-16", "2010-04-28", "2010-05-09", "2010-06-23", "2010-08-10", "2010-09-21", "2010-10-15",
    "2010-11-03", "2010-12-14",
    "2011-01-26", "2011-03-15", "2011-04-27", "2011-06-22", "2011-08-01", "2011-08-09", "2011-09-21", "2011-11-02",
    "2011-11-28", "2011-12-13",
    "2012-01-25", "2012-03-13", "2012-04-25", "2012-06-20", "2012-08-01", "2012-09-13", "2012-10-24", "2012-12-12",
    "2013-01-30", "2013-03-20", "2013-05-01", "2013-06-19", "2013-07-31", "2013-09-18", "2013-10-16", "2013-10-30",
    "2013-12-18",
    "2014-01-29", "2014-03-04", "2014-03-19", "2014-04-30", "2014-06-18", "2014-07-30", "2014-09-17", "2014-10-29",
    "2014-12-17",
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17", "2015-07-29", "2015-09-17", "2015-10-28", "2015-12-16",
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15", "2016-07-27", "2016-09-21", "2016-11-02", "2016-12-14",
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14", "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13", "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31", "2019-09-18", "2019-10-04", "2019-10-30",
    "2019-12-11",
    "2020-01-29", "2020-03-02", "2020-03-15", "2020-04-29", "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05",
    "2020-12-16",
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
]

CPI_DATES = [  # actual release date (not the data month); source: bls.gov/bls/news-release/cpi.htm, fetched 2026-09-27
    "2004-02-20", "2004-03-17", "2004-04-14", "2004-05-14", "2004-06-15", "2004-07-16", "2004-08-17", "2004-09-16",
    "2004-10-19", "2004-11-17", "2004-12-17",
    "2005-01-19", "2005-02-23", "2005-03-23", "2005-04-20", "2005-05-18", "2005-06-15", "2005-07-14", "2005-08-16",
    "2005-09-15", "2005-10-14", "2005-11-16", "2005-12-15",
    "2006-01-18", "2006-02-22", "2006-03-16", "2006-04-19", "2006-05-17", "2006-06-14", "2006-07-19", "2006-08-16",
    "2006-09-15", "2006-10-18", "2006-11-16", "2006-12-15",
    "2007-01-18", "2007-02-21", "2007-03-16", "2007-04-17", "2007-05-15", "2007-06-15", "2007-07-18", "2007-08-15",
    "2007-09-19", "2007-10-17", "2007-11-15", "2007-12-14",
    "2008-01-16", "2008-02-20", "2008-03-14", "2008-04-16", "2008-05-14", "2008-06-13", "2008-07-16", "2008-08-14",
    "2008-09-16", "2008-10-16", "2008-11-19", "2008-12-16",
    "2009-01-16", "2009-02-20", "2009-03-18", "2009-04-15", "2009-05-15", "2009-06-17", "2009-07-15", "2009-08-14",
    "2009-09-16", "2009-10-15", "2009-11-18", "2009-12-16",
    "2010-01-15", "2010-02-19", "2010-03-18", "2010-04-14", "2010-05-19", "2010-06-17", "2010-07-16", "2010-08-13",
    "2010-09-17", "2010-10-15", "2010-11-17", "2010-12-15",
    "2011-01-14", "2011-02-17", "2011-03-17", "2011-04-15", "2011-05-13", "2011-06-15", "2011-07-15", "2011-08-18",
    "2011-09-15", "2011-10-19", "2011-11-16", "2011-12-16",
    "2012-01-19", "2012-02-17", "2012-03-16", "2012-04-13", "2012-05-15", "2012-06-14", "2012-07-17", "2012-08-15",
    "2012-09-14", "2012-10-16", "2012-11-15", "2012-12-14",
    "2013-01-16", "2013-02-21", "2013-03-15", "2013-04-16", "2013-05-16", "2013-06-18", "2013-07-16", "2013-08-15",
    "2013-09-17", "2013-10-30", "2013-11-20", "2013-12-17",
    "2014-01-16", "2014-02-20", "2014-03-18", "2014-04-15", "2014-05-15", "2014-06-17", "2014-07-22", "2014-08-19",
    "2014-09-17", "2014-10-22", "2014-11-20", "2014-12-17",
    "2015-01-16", "2015-02-26", "2015-03-24", "2015-04-17", "2015-05-22", "2015-06-18", "2015-07-17", "2015-08-19",
    "2015-09-16", "2015-10-15", "2015-11-17", "2015-12-15",
    "2016-01-20", "2016-02-19", "2016-03-16", "2016-04-14", "2016-05-17", "2016-06-16", "2016-07-15", "2016-08-16",
    "2016-09-16", "2016-10-18", "2016-11-17", "2016-12-15",
    "2017-01-18", "2017-02-15", "2017-03-15", "2017-04-14", "2017-05-12", "2017-06-14", "2017-07-14", "2017-08-11",
    "2017-09-14", "2017-10-13", "2017-11-15", "2017-12-13",
    "2018-01-12", "2018-02-14", "2018-03-13", "2018-04-11", "2018-05-10", "2018-06-12", "2018-07-12", "2018-08-10",
    "2018-09-13", "2018-10-11", "2018-11-14", "2018-12-12",
    "2019-01-11", "2019-02-13", "2019-03-12", "2019-04-10", "2019-05-10", "2019-06-12", "2019-07-11", "2019-08-13",
    "2019-09-12", "2019-10-10", "2019-11-13", "2019-12-11",
    "2020-01-14", "2020-02-13", "2020-03-11", "2020-04-10", "2020-05-12", "2020-06-10", "2020-07-14", "2020-08-12",
    "2020-09-11", "2020-10-13", "2020-11-12", "2020-12-10",
    "2021-01-13", "2021-02-10", "2021-03-10", "2021-04-13", "2021-05-12", "2021-06-10", "2021-07-13", "2021-08-11",
    "2021-09-14", "2021-10-13", "2021-11-10", "2021-12-10",
    "2022-01-12", "2022-02-10", "2022-03-10", "2022-04-12", "2022-05-11", "2022-06-10", "2022-07-13", "2022-08-10",
    "2022-09-13", "2022-10-13", "2022-11-10", "2022-12-13",
    "2023-01-12", "2023-02-14", "2023-03-14", "2023-04-12", "2023-05-10", "2023-06-13", "2023-07-12", "2023-08-10",
    "2023-09-13", "2023-10-12", "2023-11-14", "2023-12-12",
    "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10", "2024-05-15", "2024-06-12", "2024-07-11", "2024-08-14",
    "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11",
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10", "2025-05-13", "2025-06-11", "2025-07-15", "2025-08-12",
    "2025-09-11", "2025-10-24", "2025-12-18",  # October 2025 data was never published (appropriations lapse)
    "2026-01-13", "2026-02-13", "2026-03-11", "2026-04-10", "2026-05-12", "2026-06-10", "2026-07-14", "2026-08-12",
    "2026-09-11",
]


def _days_since(index: pd.DatetimeIndex, event_dates: list[str]) -> tuple[pd.Series, pd.Series]:
    """(is_event_day, trading_days_since_last_event) for a fixed, externally-known calendar (event dates are
    known in advance or are historical fact, so this carries no look-ahead risk the way a market-data join
    would). is_event_day matches on the calendar date; days-since only ever looks backward."""
    events = pd.to_datetime(event_dates)
    is_event = index.isin(events)
    pos = np.arange(len(index), dtype=float)
    last_event_pos = pd.Series(np.where(is_event, pos, np.nan), index=index).ffill().to_numpy()
    return pd.Series(is_event.astype(float), index=index), pd.Series(pos - last_event_pos, index=index)


def make_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Three US macro-calendar event proxies, all causal by construction: NFP release day (first Friday of
    the month — a deterministic BLS convention), FOMC decision days, and CPI release days (both from
    fixed historical/announced-in-advance lists above, sourced from federalreserve.gov and bls.gov)."""
    f = pd.DataFrame(index=index)
    is_first_friday = (index.dayofweek == 4) & (index.day <= 7)
    f["f_nfp_day"] = is_first_friday.astype(float)
    pos = np.arange(len(index), dtype=float)
    last_nfp_pos = pd.Series(np.where(is_first_friday, pos, np.nan), index=index).ffill().to_numpy()
    f["f_days_since_nfp"] = pos - last_nfp_pos  # NaN until the first NFP day in the sample; causal (backward-only)
    f["f_fomc_day"], f["f_days_since_fomc"] = _days_since(index, FOMC_DATES)
    f["f_cpi_day"], f["f_days_since_cpi"] = _days_since(index, CPI_DATES)
    return f


def make_macro_features(aligned: pd.DataFrame, gold_close: pd.Series) -> pd.DataFrame:
    """Same convention as make_features: every column is a return/ratio/z-score (never a raw level), computed
    only from `aligned` rows <= t, so it's causal and scale-free the same way the price-only features are."""
    f = pd.DataFrame(index=aligned.index)
    for name in ("dxy", "silver", "spx"):
        lp = np.log(aligned[name])
        for n in (1, 5, 20):
            f[f"f_{name}_ret_{n}"] = lp.diff(n)
    for n in (1, 5, 20):
        f[f"f_yield10y_chg_{n}"] = aligned["yield10y"].diff(n)  # already in percentage points, additive not log
        f[f"f_tip_ret_{n}"] = np.log(aligned["tip"]).diff(n)
    f["f_vix_chg_5"] = aligned["vix"].diff(5)
    roll = aligned["vix"].rolling(60)
    f["f_vix_z_60"] = (aligned["vix"] - roll.mean()) / roll.std()
    f["f_gold_silver_ratio_ret_20"] = np.log(gold_close / aligned["silver"]).diff(20)
    return f


def make_features(d: pd.DataFrame) -> pd.DataFrame:
    c = d["close"]  # every feature is derived from close alone: O/H/L are unreliable before 2012
    lc = np.log(c)
    lr = lc.diff()
    f = pd.DataFrame(index=d.index)
    for n in (1, 3, 5, 10, 20, 60):
        f[f"f_ret_{n}"] = lc.diff(n)
    for n in (5, 20, 60):
        f[f"f_vol_{n}"] = lr.rolling(n).std()
    for n in (7, 14, 30, 60):
        f[f"f_ma_gap_{n}"] = c / c.rolling(n).mean() - 1
    delta = c.diff()
    gain, loss = delta.clip(lower=0).rolling(14).mean(), (-delta.clip(upper=0)).rolling(14).mean()
    f["f_rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    dow = d.index.dayofweek
    f["f_dow_sin"], f["f_dow_cos"] = np.sin(2 * np.pi * dow / 5), np.cos(2 * np.pi * dow / 5)
    return f


def make_targets(d: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    c = d["close"]
    r1 = np.log(c.shift(-1) / c)
    cols = {
        "y_ret_next": r1,
        "y_dir_next": (r1 > 0).astype(float).where(r1.notna() & (r1 != 0)),
        "meta_close": c,
        "meta_next_close": c.shift(-1),
    }
    for h in horizons:
        r = np.log(c.shift(-h) / c)
        cols[f"y_ret_h{h}"] = r
        cols[f"y_dir_h{h}"] = (r > 0).astype(float).where(r.notna() & (r != 0))
        cols[f"meta_close_h{h}"] = c.shift(-h)
    return pd.DataFrame(cols)


def assign_splits(index: pd.DatetimeIndex) -> pd.Series:
    s = pd.Series("train", index=index)
    s[index > pd.Timestamp(SPLITS["train_end"])] = "val"
    s[index > pd.Timestamp(SPLITS["val_end"])] = "test"
    for boundary in ("val", "test"):  # embargo: first bars after each boundary, so multi-day labels never straddle a split
        pos = np.flatnonzero((s == boundary).to_numpy())
        if len(pos):
            s.iloc[pos[:EMBARGO_BARS]] = "embargo"
    return s


def build(src: str, use_macro: bool = False, macro_cache: Path | None = None, use_calendar: bool = False,
          use_cot: bool = False, cot_cache: Path | None = None) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Returns (labelled dataset, report, latest feature rows). The latest rows include the newest day, whose
    next-day label does not exist yet, so a forecast for tomorrow can be made from them."""
    d, rep = clean_and_report(load_silver(src))
    feats_all = make_features(d)
    rep["macro"] = {"enabled": use_macro}
    if use_macro:
        macro_raw = fetch_macro(str(d.index.min().date()), str((d.index.max() + pd.Timedelta(days=1)).date()), macro_cache)
        aligned = align_macro(d.index, macro_raw)
        feats_all = feats_all.join(make_macro_features(aligned, d["close"]))
        rep["macro"]["tickers"] = MACRO_TICKERS
    rep["calendar"] = {"enabled": use_calendar}
    if use_calendar:
        feats_all = feats_all.join(make_calendar_features(d.index))
        rep["calendar"]["note"] = "NFP (first-Friday-of-month rule) + FOMC/CPI from fixed historical date lists (federalreserve.gov, bls.gov)"
    rep["cot"] = {"enabled": use_cot}
    if use_cot:
        cot_raw = fetch_cot(cot_cache)
        aligned_cot = align_macro(d.index, cot_raw)  # same as-of backward join as macro/cross-asset data
        feats_all = feats_all.join(make_cot_features(aligned_cot))
        rep["cot"]["source"] = "CFTC Legacy COT report (Socrata 6dca-aqww), gold non-commercial net % of open interest"
    feat = [c for c in feats_all.columns if c.startswith("f_")]
    latest = feats_all.dropna(subset=feat).join(d["close"].rename("meta_close")).tail(120)
    df = feats_all.join(make_targets(d))
    df = df.dropna(subset=feat)
    df["meta_split"] = assign_splits(df.index)
    df = df[(df["meta_split"] != "embargo") & df["y_dir_next"].notna()]
    rep["dataset"] = {
        "rows": int(len(df)), "start": str(df.index.min().date()), "end": str(df.index.max().date()), "n_features": len(feat),
        "per_split": {
            k: {"rows": int(len(g)), "share_up": round(float(g["y_dir_next"].mean()), 4),
                **{f"share_up_h{h}": round(float(g[f"y_dir_h{h}"].mean()), 4) for h in HORIZONS}}
            for k, g in df.groupby("meta_split")
        },
    }
    return df, rep, latest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver", default=DEFAULT_SILVER)
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--macro", action="store_true", help="add cross-asset/macro features (DXY, silver, SPX, VIX, 10Y yield, TIP)")
    ap.add_argument("--macro-cache", type=Path, default=Path("data/macro_cache.parquet"))
    ap.add_argument("--calendar", action="store_true", help="add NFP/FOMC/CPI release-day calendar features")
    ap.add_argument("--cot", action="store_true", help="add CFTC Commitment of Traders (gold positioning) features")
    ap.add_argument("--cot-cache", type=Path, default=Path("data/cot_cache.parquet"))
    args = ap.parse_args()
    df, rep, latest = build(args.silver, use_macro=args.macro, macro_cache=args.macro_cache if args.macro else None,
                             use_calendar=args.calendar, use_cot=args.cot, cot_cache=args.cot_cache if args.cot else None)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_dir / "xauusd_d1_features.parquet")
    latest.to_parquet(args.out_dir / "xauusd_d1_latest_features.parquet")
    (args.out_dir / "build_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep["dataset"], indent=2))
    for w in rep["coverage_warnings"]:
        print("COVERAGE WARNING:", w)


if __name__ == "__main__":
    main()
