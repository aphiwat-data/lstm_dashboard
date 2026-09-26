"""
Is the rollover-hour "edge" in the bid-price baseline a bid/spread artifact?

Hypothesis: around the daily break (hours 19-23 UTC) the spread widens, so BID-only bars drift
predictably (bid falls when the spread widens, then recovers) - a pattern that mid/ask prices do not have.

This script (1) aligns bid and ask bars, (2) reports spread by hour, (3) compares the mean next-bar
return by hour for bid vs mid, and (4) writes mid-price raw files so build_hourly_dataset.py and
baseline.py can be re-run on mid prices (optionally also bid restricted to the same bars).

Usage: python spread_artifact_check.py --bid-dir data/raw --ask-dir data/raw_ask --mid-dir data/raw_mid --bid-common-dir data/raw_bidcommon
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from build_hourly_dataset import clean_and_report, load_raw

TEST_START = pd.Timestamp("2025-01-01", tz="UTC")


def next_bar_return_by_hour(close: pd.Series) -> pd.DataFrame:
    y = np.log(close.shift(-1) / close).dropna()
    g = y.groupby(y.index.hour)
    return pd.DataFrame({"mean_bp": g.mean() * 1e4, "t": g.mean() / (g.std() / np.sqrt(g.count())), "n": g.count()})


def write_yearly(mid: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ms = (mid.index - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)
    for year, g in mid.groupby(mid.index.year):
        frame = g.copy()
        frame.insert(0, "timestamp", ms[mid.index.year == year])
        frame.to_csv(out_dir / f"xauusd-h1-{year}.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bid-dir", type=Path, required=True)
    ap.add_argument("--ask-dir", type=Path, required=True)
    ap.add_argument("--mid-dir", type=Path, required=True)
    ap.add_argument("--summary-json", type=Path, default=None, help="save the spread/drift tables (used by make_summary_figure.py)")
    ap.add_argument("--bid-common-dir", type=Path, default=None, help="also write the bid bars restricted to timestamps present in both feeds (like-for-like comparison)")
    args = ap.parse_args()

    bid, rb = clean_and_report(load_raw("xauusd", args.bid_dir), "bid")
    ask, ra = clean_and_report(load_raw("xauusd", args.ask_dir), "ask")
    for r in (rb, ra):
        print(f"{r['instrument']}: {r['rows_clean']:,} bars {r['start'][:10]} -> {r['end'][:10]} warnings={r['coverage_warnings']}")

    common = bid.index.intersection(ask.index)
    print(f"bars only in bid: {len(bid) - len(common):,} | only in ask: {len(ask) - len(common):,} | common: {len(common):,}")
    b, a = bid.loc[common], ask.loc[common]
    print(f"ask < bid at close: {(a['close'] < b['close']).mean():.4%} of bars")

    mid = pd.DataFrame({k: (b[k] + a[k]) / 2 for k in ("open", "high", "low", "close")})
    mid["volume"] = b["volume"]
    write_yearly(mid, args.mid_dir)
    if args.bid_common_dir:
        write_yearly(b, args.bid_common_dir)

    spread_bp = (a["close"] - b["close"]) / mid["close"] * 1e4
    recent = spread_bp[spread_bp.index >= TEST_START]
    by_hour = pd.DataFrame({"all_years": spread_bp.groupby(spread_bp.index.hour).median(), "2025+": recent.groupby(recent.index.hour).median()}).round(2)
    print("\nMedian close spread (bp of price) by UTC hour of bar:")
    print(by_hour.T.to_string())
    summary = {"common_bars": int(len(common)), "spread_bp_by_hour": by_hour.rename_axis("bar_open").reset_index().to_dict(orient="list"), "next_bar_return_by_hour": {}}

    for label, mask in (("ALL YEARS", slice(None)), ("TEST PERIOD 2025+", mid.index >= TEST_START)):
        rb_, rm_ = next_bar_return_by_hour(b["close"][mask]), next_bar_return_by_hour(mid["close"][mask])
        tab = pd.concat({"bid_mean_bp": rb_["mean_bp"], "bid_t": rb_["t"], "mid_mean_bp": rm_["mean_bp"], "mid_t": rm_["t"], "n": rb_["n"]}, axis=1).round(2)
        summary["next_bar_return_by_hour"][label] = tab.reset_index().to_dict(orient="list")
        flag = tab[(tab["bid_t"].abs() > 2) | (tab["mid_t"].abs() > 2)]
        print(f"\nMean NEXT-bar return by UTC hour of the current bar, {label} (hours with |t|>2 in bid or mid):")
        print(flag.to_string() if len(flag) else "  none")
        print(f"  hours with |t|>2: bid={(tab['bid_t'].abs() > 2).sum()}  mid={(tab['mid_t'].abs() > 2).sum()}  (of 24; ~1 expected by chance)")
    if args.summary_json:
        import json

        args.summary_json.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
