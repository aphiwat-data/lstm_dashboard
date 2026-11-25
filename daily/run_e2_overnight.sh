#!/usr/bin/env bash
# E2 overnight run (README section 15.8): every forecaster group in parallel, then the evaluation.
# Resumable: a finished group is skipped and an unfinished one continues from its last checkpoint (every 10 blocks);
# a group that failed is tried once more in a second pass before the evaluation starts.
# Usage (from the repository root or from daily/):  caffeinate -i daily/run_e2_overnight.sh [PYTHON]
# PYTHON must have numpy, pandas, scipy, scikit-learn, tensorflow and arch (default: python3). Five workers, each under about 1 GB, fit in 8 GB of memory (see run_e2_group.sh).
set -u
cd "$(dirname "$0")"
PY="${1:-python3}"
OUT=results/paper_e2
mkdir -p "$OUT/forecasts" "$OUT/logs"
{ [ -f data/e2_prices.parquet ] && [ -f data/e2_macro.parquet ]; } || "$PY" -W ignore search_experiment.py prepare
echo "$(date '+%F %T') started" >> "$OUT/logs/_status.txt"
# the long recurrent fits first so that the short groups fill the gaps
ORDER="nn_lstm_w20_u32_P_L1000 nn_lstm_w20_u32_A_L1000 nn_lstm_w60_u32_P_L1000 nn_gru_w20_u32_A_L1000 nn_lstm_w20_u32_X_L1000 \
nn_lstm_w20_u32_P_L365 nn_lstm_w20_u32_A_L365 nn_gru_w20_u32_P_L365 nn_lstm_w5_u32_P_L365 nn_lstm_w10_u32_P_L365 \
nn_lstm_w20_u16_P_L365 nn_lstm_w20_u64_P_L365 hgb_h1 et_h1 linear_h1 linear_h5 linear_h20"
for pass in 1 2; do
  printf '%s\n' $ORDER | xargs -n 1 -P 5 ./run_e2_group.sh "$PY"
done
missing=0
for g in $ORDER; do [ -f "$OUT/forecasts/$g.npz" ] || { echo "$(date '+%F %T') MISSING group $g" >> "$OUT/logs/_status.txt"; missing=$((missing + 1)); }; done
"$PY" -W ignore search_experiment.py evaluate > "$OUT/logs/evaluate.log" 2>&1
echo "$(date '+%F %T') evaluate exit $? (groups missing: $missing)" >> "$OUT/logs/_status.txt"
