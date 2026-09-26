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


def build(src: str) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Returns (labelled dataset, report, latest feature rows). The latest rows include the newest day, whose
    next-day label does not exist yet, so a forecast for tomorrow can be made from them."""
    d, rep = clean_and_report(load_silver(src))
    feats_all = make_features(d)
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
    args = ap.parse_args()
    df, rep, latest = build(args.silver)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_dir / "xauusd_d1_features.parquet")
    latest.to_parquet(args.out_dir / "xauusd_d1_latest_features.parquet")
    (args.out_dir / "build_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep["dataset"], indent=2))
    for w in rep["coverage_warnings"]:
        print("COVERAGE WARNING:", w)


if __name__ == "__main__":
    main()
