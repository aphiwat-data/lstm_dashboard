#!/bin/bash
set -e
cd "$(dirname "$0")"
export DUKASCOPY_BIN="$(pwd)/node_modules/.bin/dukascopy-node"

echo "=== [1/6] XAUUSD bid ==="
python3 download_dukascopy.py xauusd --start 2004-01 --price bid --raw-dir data/raw

echo "=== [2/6] XAUUSD ask ==="
python3 download_dukascopy.py xauusd --start 2004-01 --price ask --raw-dir data/raw_ask

echo "=== [3/6] Aux instruments (silver, EURUSD, S&P500, DXY) bid ==="
python3 download_dukascopy.py xagusd eurusd usa500idxusd dollaridxusd --start 2004-01 --price bid --raw-dir data/raw

echo "=== [4/6] Building mid-price series (bid+ask -> mid) ==="
python3 spread_artifact_check.py --bid-dir data/raw --ask-dir data/raw_ask --mid-dir data/raw_mid --bid-common-dir data/raw_bidcommon

echo "=== [5/6] Copying aux instrument files into mid dir (aux has no separate mid) ==="
for ins in xagusd eurusd usa500idxusd dollaridxusd; do
  cp data/raw/${ins}-h1-*.csv data/raw_mid/ 2>/dev/null || true
  cp data/raw/${ins}-h1-*.csv data/raw_bidcommon/ 2>/dev/null || true
done

echo "=== [6/6] Building datasets (mid, with vs without aux) ==="
python3 build_hourly_dataset.py --raw-dir data/raw_mid --out-dir data/processed_mid_aux
python3 build_hourly_dataset.py --raw-dir data/raw_mid --out-dir data/processed_mid_noaux --no-aux

echo "=== PIPELINE DONE ==="
