#!/bin/bash
# Runs every experiment in order (01 designs the voices the others use). Logs go to results/<experiment>.log
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results
for experiment_dir in experiments/[0-9][0-9]-*/; do
  experiment=$(basename "$experiment_dir")
  echo "== $experiment"
  PYTHONUNBUFFERED=1 uv run python "$experiment_dir/run.py" > "results/$experiment.log" 2>&1 \
    || { echo "failed: see results/$experiment.log"; exit 1; }
done
