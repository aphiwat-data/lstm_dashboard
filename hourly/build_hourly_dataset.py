"""
Hourly XAU/USD dataset builder (Dukascopy h1 bars, UTC).

Causality convention (the point of this file):
  * Row t is the bar that OPENS at t and covers [t, t+1h). It is fully known at t+1h.
  * Every f_* feature on row t uses bars <= t only.
  * Targets (y_*) describe the NEXT bar t+1 relative to close_t. They are never features.
  * meta_* columns are bookkeeping only and must not be fed to a model.
Scaling is deliberately NOT done here: fit scalers on the train split only, at model time.

Usage:
    python build_hourly_dataset.py --raw-dir data/raw --out-dir data/processed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

AUX_TAGS = {"xagusd": "xag", "eurusd": "eur", "usa500idxusd": "spx"}
SPLITS = {"train_end": "2023-12-31", "val_end": "2024-12-31"}  # test = after val_end (matches the daily project's 2024-12-31 cut)
HORIZONS = (4, 24)
EMBARGO_BARS = 24  # must be >= max(HORIZONS) so multi-bar labels never straddle a split boundary
EXTREME_LOGRET = 0.05
MAX_END_LAG_DAYS = 5
THIN_MONTH_SHARE = 0.5


def load_raw(instrument: str, raw_dir: Path) -> pd.DataFrame:
    files = sorted(raw_dir.glob(f"{instrument}-h1-*.csv"))
    frames = [pd.read_csv(f) for f in files if f.stat().st_size > 0]
    if not frames:
        raise FileNotFoundError(f"no non-empty {instrument}-h1-*.csv files in {raw_dir}")
    df = pd.concat(frames, ignore_index=True)
    df.index = pd.to_datetime(df.pop("timestamp"), unit="ms", utc=True)
    df.index.name = "bar_open"
    return df.sort_index()


def clean_and_report(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, dict]:
    rep: dict = {"instrument": name, "rows_raw": int(len(df))}
    rep["duplicate_timestamps"] = int(df.index.duplicated().sum())
    df = df[~df.index.duplicated(keep="last")]

    bad_ohlc = (
        (df["high"] < df[["open", "close"]].max(axis=1))
        | (df["low"] > df[["open", "close"]].min(axis=1))
        | (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    )
    closed = df["volume"] <= 0
    rep["invalid_ohlc_rows"] = int(bad_ohlc.sum())
    rep["zero_volume_rows"] = int(closed.sum())
    df = df[~bad_ohlc & ~closed]

    gaps = df.index.to_series().diff().dropna()
    long_gaps = gaps[gaps > pd.Timedelta(hours=1)]
    rep["gaps_gt_1h"] = int(len(long_gaps))
    rep["gap_length_hours_counts"] = {
        str(k): int(v) for k, v in (long_gaps.dt.total_seconds() / 3600).round().value_counts().head(8).items()
    }
    rep["largest_gaps"] = [
        {"before": str(ts), "hours": round(g.total_seconds() / 3600, 1)}
        for ts, g in long_gaps.sort_values(ascending=False).head(5).items()
    ]

    lr = np.log(df["close"]).diff().dropna()
    extreme = lr[lr.abs() > EXTREME_LOGRET]
    rep["extreme_1bar_moves_gt_5pct"] = [
        {"bar_open": str(ts), "logret": round(float(x), 4)} for ts, x in extreme.head(10).items()
    ]
    rep.update(rows_clean=int(len(df)), start=str(df.index.min()), end=str(df.index.max()))
    rep["coverage_warnings"] = coverage_warnings(df, name)
    return df, rep


def coverage_warnings(df: pd.DataFrame, name: str) -> list[str]:
    """Catches silently truncated downloads: missing tail, missing months, or thin months."""
    warnings = []
    lag_days = (pd.Timestamp.now(tz="UTC") - df.index.max()).days
    if lag_days > MAX_END_LAG_DAYS:
        warnings.append(f"{name}: data ends {df.index.max():%Y-%m-%d}, {lag_days} days ago (tail missing?)")
    monthly = df.groupby(df.index.tz_localize(None).to_period("M")).size()
    full_range = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    absent = [str(p) for p in full_range.difference(monthly.index)]
    if absent:
        warnings.append(f"{name}: months with no bars: {absent[:12]}{' ...' if len(absent) > 12 else ''}")
    inner = monthly.iloc[1:-1]
    thin = inner[inner < THIN_MONTH_SHARE * inner.median()]
    if len(thin):
        warnings.append(f"{name}: thin months (<{THIN_MONTH_SHARE:.0%} of median): {[str(p) for p in thin.index[:12]]}")
    return warnings


def _ret_features(close: pd.Series, tag: str, windows=(1, 6)) -> pd.DataFrame:
    lc = np.log(close)
    return pd.DataFrame({f"f_{tag}_ret_{n}": lc.diff(n) for n in windows})


def make_features(gold: pd.DataFrame, aux: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    o, h, l, c, v = (gold[k] for k in ("open", "high", "low", "close", "volume"))
    lc = np.log(c)
    lr = lc.diff()
    f = pd.DataFrame(index=gold.index)

    for n in (1, 3, 6, 24, 120):
        f[f"f_ret_{n}"] = lc.diff(n)
    for n in (6, 24, 120):
        f[f"f_vol_{n}"] = lr.rolling(n).std()
    f["f_range"] = (h - l) / c
    f["f_body"] = (c - o) / o
    f["f_close_pos"] = (c - l) / (h - l).replace(0, np.nan)
    for n in (24, 120):
        f[f"f_ma_gap_{n}"] = c / c.rolling(n).mean() - 1

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    f["f_rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    lv = np.log(v)
    f["f_logvol_z_120"] = (lv - lv.rolling(120).mean()) / lv.rolling(120).std()
    f["f_vol_ratio_24"] = v / v.rolling(24).mean()

    hour, dow = gold.index.hour, gold.index.dayofweek
    f["f_hour_sin"], f["f_hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    f["f_dow_sin"], f["f_dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    f["f_after_gap"] = (gold.index.to_series().diff() > pd.Timedelta(hours=1)).astype(int)

    for name, tag in AUX_TAGS.items():
        if aux and name in aux:
            feats = _ret_features(aux[name]["close"], tag)
            f = f.join(feats.reindex(gold.index).ffill(limit=3))
    return f


def make_targets(gold: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    c = gold["close"]
    y_ret = np.log(c.shift(-1) / c)
    y_dir = (y_ret > 0).astype(float).where(y_ret.notna() & (y_ret != 0))
    idx = gold.index.to_series()
    cols = {
        "y_ret_next": y_ret,
        "y_dir_next": y_dir,
        "meta_close": c,
        "meta_next_close": c.shift(-1),
        "meta_gap_to_next_min": (idx.shift(-1) - idx).dt.total_seconds() / 60,
    }
    for h in horizons:  # h counts bars (not clock hours); meta_span_h shows the real elapsed time
        r = np.log(c.shift(-h) / c)
        cols[f"y_ret_h{h}"] = r
        cols[f"y_dir_h{h}"] = (r > 0).astype(float).where(r.notna() & (r != 0))
        cols[f"meta_close_h{h}"] = c.shift(-h)
        cols[f"meta_span_h{h}"] = (idx.shift(-h) - idx).dt.total_seconds() / 3600
    return pd.DataFrame(cols)


def assign_splits(df: pd.DataFrame) -> pd.Series:
    s = pd.Series("train", index=df.index)
    s[df.index > pd.Timestamp(SPLITS["train_end"], tz="UTC")] = "val"
    s[df.index > pd.Timestamp(SPLITS["val_end"], tz="UTC")] = "test"
    # embargo: drop the first bars after each boundary so windows/targets cannot straddle a split
    for boundary in ("val", "test"):
        pos = np.flatnonzero((s == boundary).to_numpy())
        if len(pos):
            s.iloc[pos[:EMBARGO_BARS]] = "embargo"
    return s


def build(raw_dir: Path, use_aux: bool = True) -> tuple[pd.DataFrame, dict]:
    report: dict = {}
    gold_raw = load_raw("xauusd", raw_dir)
    gold, report["xauusd"] = clean_and_report(gold_raw, "xauusd")

    aux: dict[str, pd.DataFrame] = {}
    if use_aux:
        for name in AUX_TAGS:
            try:
                clean, report[name] = clean_and_report(load_raw(name, raw_dir), name)
                aux[name] = clean
            except FileNotFoundError as e:
                report[name] = {"skipped": str(e)}

    df = make_features(gold, aux).join(make_targets(gold))
    feat_cols = [c for c in df.columns if c.startswith("f_") and not any(c.startswith(f"f_{t}_") for t in AUX_TAGS.values())]
    df = df.dropna(subset=feat_cols)
    df["meta_split"] = assign_splits(df)
    df = df[df["meta_split"] != "embargo"]
    df = df[df["y_dir_next"].notna()]

    report["dataset"] = {
        "rows": int(len(df)),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "n_features": int(len([c for c in df.columns if c.startswith("f_")])),
        "per_split": {
            k: {
                "rows": int(len(g)),
                "share_up": round(float(g["y_dir_next"].mean()), 4),
                **{f"share_up_h{h}": round(float(g[f"y_dir_h{h}"].mean()), 4) for h in HORIZONS},
                "targets_spanning_market_gap": round(float((g["meta_gap_to_next_min"] > 60).mean()), 4),
            }
            for k, g in df.groupby("meta_split")
        },
        "feature_nan_share": {c: round(float(df[c].isna().mean()), 4) for c in df.columns if c.startswith("f_") and df[c].isna().any()},
    }
    return df, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--no-aux", action="store_true")
    args = ap.parse_args()

    df, report = build(args.raw_dir, use_aux=not args.no_aux)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_dir / "xauusd_h1_features.parquet")
    (args.out_dir / "build_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["dataset"], indent=2))

    warnings = [w for r in report.values() for w in r.get("coverage_warnings", [])]
    if warnings:
        print("\nCOVERAGE WARNINGS - do not trust results until resolved:")
        for w in warnings:
            print("  -", w)


if __name__ == "__main__":
    main()
