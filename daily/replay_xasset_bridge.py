"""Replay the cross-asset model for the days between the last committed E2 forecast and its first live one, so that the header chart has no hole at the switch (README
section 10, Phase 14).

Each row is the live code (crossasset.forecast_latest: ridge refit on the last 1,000 labelled days known that day) on the E2 data cut at that day. The close of the day it
was FOR is read from Yahoo (GC=F) and must match --next-closes. The rows are a replay, computed after the fact like the E2 rows they continue, never a live forecast.
Output: results/xasset_replay_bridge.csv, columns date, y, close, next_close, dev_best (as in results/paper_e2/e2_frozen_forecasts_h1.csv).

Usage: python replay_xasset_bridge.py --first-live 2026-11-25 --next-closes 3535.80   (needs data/e2_prices.parquet and data/e2_macro.parquet: python search_experiment.py prepare)"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import crossasset as ca  # noqa: E402


def bridge(prices: pd.DataFrame, macro: pd.DataFrame, e2: pd.DataFrame, later: pd.Series, first_live: pd.Timestamp, errors: np.ndarray) -> pd.DataFrame:
    """`later` holds closes after the last day of `prices` (they only label the rows; no forecast sees them)."""
    close = prices.set_index("date")["close"].sort_index()
    labels = pd.concat([close, later[later.index > close.index[-1]]])
    days = close.index[(close.index > e2["date"].max()) & (close.index < first_live)]
    rows = []
    for day in days:
        fc = ca.forecast_latest(close.loc[:day], macro.loc[:day], errors)
        assert fc["as_of_date"] == str(day.date()), fc["as_of_date"]
        after = labels[labels.index > day]
        assert len(after), f"no close after {day.date()}"
        rows.append({"date": day.date(), "y": float(np.log(after.iloc[0] / close.loc[day])), "close": float(close.loc[day]), "next_close": float(after.iloc[0]), "dev_best": fc["predicted_return"]})
    return pd.DataFrame(rows, columns=["date", "y", "close", "next_close", "dev_best"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--first-live", required=True, help="as-of date of the first live cross-asset forecast")
    ap.add_argument("--next-closes", required=True, nargs="+", type=float, help="the closes the replayed days were for, in date order, as the live log has them")
    ap.add_argument("--results", default=str(HERE / "results"))
    args = ap.parse_args()
    results = Path(args.results)
    prices = pd.read_parquet(HERE / "data" / "e2_prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    macro = pd.read_parquet(HERE / "data" / "e2_macro.parquet")
    e2 = pd.read_csv(results / "paper_e2" / "e2_frozen_forecasts_h1.csv", parse_dates=["date"])
    preds = pd.read_csv(results / "wf_predictions.csv")
    errors = (preds["y_ret_next"] - preds["pred_ret_xasset"]).to_numpy(float)
    later = ca.parse_chart(ca._http_json("https://query1.finance.yahoo.com/v8/finance/chart/GC%3DF?range=1mo&interval=1d"))
    out = bridge(prices, macro, e2, later, pd.Timestamp(args.first_live), errors)
    assert len(out) == len(args.next_closes) and np.allclose(out["next_close"], args.next_closes, atol=0.05), f"closes differ from the live log:\n{out}"
    out.to_csv(results / "xasset_replay_bridge.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
