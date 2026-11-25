#!/usr/bin/env bash
# E2 overnight run (README section 15.8): every forecaster group in parallel, then the evaluation.
# Resumable: a finished group is skipped and an unfinished one continues from its last checkpoint (every 10 blocks).
# Usage (from the repository root or from daily/):  caffeinate -i daily/run_e2_overnight.sh [PYTHON]
# PYTHON must have numpy, pandas, scipy, scikit-learn, tensorflow and arch (default: python3). Four workers fit in 8 GB of memory.
set -u
cd "$(dirname "$0")"
PY="${1:-python3}"
OUT=results/paper_e2
mkdir -p "$OUT/forecasts" "$OUT/logs"
{ [ -f data/e2_prices.parquet ] && [ -f data/e2_macro.parquet ]; } || "$PY" -W ignore search_experiment.py prepare
echo "started $(date '+%F %T')" >> "$OUT/logs/_status.txt"
# the long recurrent fits first so that the short groups fill the gaps
ORDER="nn_lstm_w20_u32_P_L1000 nn_lstm_w20_u32_A_L1000 nn_lstm_w60_u32_P_L1000 nn_gru_w20_u32_A_L1000 nn_lstm_w20_u32_X_L1000 \
nn_lstm_w20_u32_P_L365 nn_lstm_w20_u32_A_L365 nn_gru_w20_u32_P_L365 nn_lstm_w5_u32_P_L365 nn_lstm_w10_u32_P_L365 \
nn_lstm_w20_u16_P_L365 nn_lstm_w20_u64_P_L365 hgb_h1 et_h1 linear_h1 linear_h5 linear_h20"
printf '%s\n' $ORDER | xargs -P 4 -I{} sh -c "'$PY' -W ignore search_experiment.py forecast --group {} > '$OUT/logs/{}.log' 2>&1; echo \"\$(date '+%F %T') {} exit \$?\" >> '$OUT/logs/_status.txt'"
"$PY" -W ignore search_experiment.py evaluate > "$OUT/logs/evaluate.log" 2>&1
echo "$(date '+%F %T') evaluate exit $?" >> "$OUT/logs/_status.txt"
