"""
Verified, resumable Dukascopy h1 downloader: one request per month, so a month is either complete or empty.

Setup (once):  cd hourly && npm install          # installs dukascopy-node from package.json
Usage:         python download_dukascopy.py xauusd --start 2004-01
               python download_dukascopy.py xagusd eurusd usa500idxusd --start 2010-01
               python download_dukascopy.py xauusd --start 2004-01 --price ask --raw-dir data/raw_ask

Why monthly + verification: the CLI can save a PARTIAL file without an error when the feed answers HTTP 429
(rate limit), so a month is only accepted if it has >= MIN_ROWS bars (the current month: >= 1).
Retries back off; a 429 backs off for 60s. Run one instance at a time - parallel runs trigger the limit.
Volumes are requested (-v), which also makes the CLI drop zero-volume "market closed" bars.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
BIN = Path(os.environ.get("DUKASCOPY_BIN", HERE / "node_modules/.bin/dukascopy-node"))
MIN_ROWS = 200
TODAY = dt.date.today()


def months(start: tuple[int, int], end: tuple[int, int]):
    y, m = start
    while (y, m) <= end:
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def rows_ok(path: Path, current: bool) -> bool:
    try:
        return len(pd.read_csv(path)) >= (1 if current else MIN_ROWS)
    except Exception:
        return False


def fetch(ins: str, y: int, m: int, seen_data: bool, price: str, raw: Path) -> str:
    current = (y, m) == (TODAY.year, TODAY.month)
    out = raw / f"{ins}-h1-{y}-{m:02d}.csv"
    if out.exists() and out.stat().st_size > 0 and rows_ok(out, current):
        return "cached"
    nxt = dt.date(y + (m == 12), 1 if m == 12 else m + 1, 1)
    to = TODAY + dt.timedelta(days=1) if current else nxt  # the CLI's 'to' is exclusive
    tmp = raw / f"_tmp_{ins}_{y}{m:02d}"
    attempts = 5 if seen_data else 2  # before an instrument's first data month, empties are expected: fail fast
    for a in range(attempts):
        shutil.rmtree(tmp, ignore_errors=True)
        r = subprocess.run(
            [str(BIN), "-i", ins, "-from", f"{y}-{m:02d}-01", "-to", to.isoformat(), "-t", "h1", "-p", price, "-f", "csv", "-v", "-s", "-dir", str(tmp)],
            capture_output=True, text=True, timeout=180,
        )
        files = list(tmp.glob("*.csv"))
        if files and rows_ok(files[0], current):
            shutil.move(str(files[0]), out)
            shutil.rmtree(tmp, ignore_errors=True)
            return "ok"
        time.sleep(60 if "429" in (r.stdout + r.stderr) else (10 * 2**a if seen_data else 3))
    shutil.rmtree(tmp, ignore_errors=True)
    return "FAILED" if seen_data else "no-data"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("instruments", nargs="+", help="e.g. xauusd xagusd eurusd usa500idxusd")
    ap.add_argument("--start", default="2004-01", help="first month, YYYY-MM")
    ap.add_argument("--price", choices=["bid", "ask"], default="bid")
    ap.add_argument("--raw-dir", type=Path, default=HERE / "data/raw")
    args = ap.parse_args()

    if not BIN.exists():
        raise SystemExit(f"{BIN} not found - run `npm install` in {HERE} (or set DUKASCOPY_BIN)")
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    y0, m0 = map(int, args.start.split("-"))
    for ins in args.instruments:
        seen = False
        for y, m in months((y0, m0), (TODAY.year, TODAY.month)):
            status = fetch(ins, y, m, seen, args.price, args.raw_dir)
            seen = seen or status in ("ok", "cached")
            if status != "cached":
                print(f"{time.strftime('%T')} {ins} {y}-{m:02d} {status}", flush=True)
            if status == "ok":
                time.sleep(1.5)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
