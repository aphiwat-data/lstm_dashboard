"""Always-up accuracy by forecast horizon (README Phase 8 table, thesis Table 3.5).

Test column: for each horizon h, the share of test days whose h-day-ahead return is positive, using only price pairs that lie inside the
test window (n = N - h, n_eff = n // h because neighbouring labels overlap). All-years column: the same rule on the whole silver history.

Usage: python horizon_always_up.py [--predictions results/wf_predictions.csv] [--silver s3://gold-lstm-forecast/silver/xauusd_daily_clean.parquet | local.parquet]
"""
from __future__ import annotations

import argparse

import pandas as pd

from build_daily_dataset import DEFAULT_SILVER, load_silver

HORIZONS = (1, 5, 20, 60, 120)


def up_share(close: pd.Series, h: int) -> tuple[float, int, int]:
    """(share of positive h-day returns, number of price pairs, n_eff = pairs // h)."""
    r = (close.shift(-h) / close - 1).dropna()
    return float((r > 0).mean()), len(r), len(r) // h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="results/wf_predictions.csv")
    ap.add_argument("--silver", default=DEFAULT_SILVER)
    args = ap.parse_args()

    test_close = pd.read_csv(args.predictions, parse_dates=["date"]).set_index("date")["meta_close"]
    silver = load_silver(args.silver)
    silver["date"] = pd.to_datetime(silver["date"])
    history = silver.drop_duplicates("date", keep="last").sort_values("date").set_index("date")["close"]

    print(f"test window {test_close.index.min().date()} .. {test_close.index.max().date()} ({len(test_close)} days); history {history.index.min().date()} .. {history.index.max().date()}")
    print("| Horizon (trading days) | Always-up, test window | Always-up, all years | Test n_eff |")
    print("|---|---|---|---|")
    for h in HORIZONS:
        test_share, _, n_eff = up_share(test_close, h)
        all_share, _, _ = up_share(history, h)
        print(f"| {h} | {test_share:.1%} | {all_share:.1%} | {n_eff} |")


if __name__ == "__main__":
    main()
