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


def build(src: str, use_macro: bool = False, macro_cache: Path | None = None) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
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
    args = ap.parse_args()
    df, rep, latest = build(args.silver, use_macro=args.macro, macro_cache=args.macro_cache if args.macro else None)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_dir / "xauusd_d1_features.parquet")
    latest.to_parquet(args.out_dir / "xauusd_d1_latest_features.parquet")
    (args.out_dir / "build_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep["dataset"], indent=2))
    for w in rep["coverage_warnings"]:
        print("COVERAGE WARNING:", w)


if __name__ == "__main__":
    main()
