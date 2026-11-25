#!/usr/bin/env bash
# Runs one E2 forecaster group to completion in short-lived processes (called by run_e2_overnight.sh).
# TensorFlow leaks about 85 MB per block of recurrent fits (a worker reached 2.7 GB after 30 blocks and the 8 GB machine thrashed), so a recurrent group is
# computed in chunks of 6 blocks, each in a fresh process that resumes from the checkpoint; the other groups run in one process.
# Usage: run_e2_group.sh PYTHON GROUP
set -u
cd "$(dirname "$0")"
PY="$1"; g="$2"; OUT=results/paper_e2
chunk=100000
case "$g" in nn_*) chunk=6;; esac
for i in $(seq 1 120); do
  [ -f "$OUT/forecasts/$g.npz" ] && break
  "$PY" -W ignore search_experiment.py forecast --group "$g" --chunk "$chunk" >> "$OUT/logs/$g.log" 2>&1 || { rc=$?; echo "$(date '+%F %T') $g chunk failed rc=$rc" >> "$OUT/logs/_status.txt"; exit $rc; }
done
[ -f "$OUT/forecasts/$g.npz" ]
rc=$?
echo "$(date '+%F %T') $g exit $rc" >> "$OUT/logs/_status.txt"
exit $rc
