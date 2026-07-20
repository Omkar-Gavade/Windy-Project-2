#!/usr/bin/env bash
# Container health probe.
#
# This application exposes no HTTP endpoint — it is a batch loop. So health is
# defined by its actual work product: the pipeline rewrites
# energy_predictions/<PLANT>_energy_generation.csv on every cycle, so a recent
# mtime on that file proves capture -> feature extraction -> prediction -> write
# all completed. A stale file means the loop is wedged even if the process is up.
set -euo pipefail

PRED_DIR="${PRED_DIR:-/app/energy_predictions}"

# Allow ~2 cycles plus margin before declaring the loop unhealthy.
# RUN_INTERVAL_SECONDS in config.py is 1200 (20 min), so the default is 45 min.
MAX_AGE="${HEALTH_MAX_AGE_SECONDS:-2700}"

# 1. The main loop process must be alive.
pgrep -f "test_multi_image.py" >/dev/null 2>&1 \
    || { echo "UNHEALTHY: test_multi_image.py is not running"; exit 1; }

# 2. A predictions CSV must exist and be recent.
newest="$(find "$PRED_DIR" -name '*_energy_generation.csv' -type f -printf '%T@ %p\n' 2>/dev/null \
          | sort -rn | head -1 | cut -d' ' -f2- || true)"

if [ -z "$newest" ]; then
    echo "UNHEALTHY: no *_energy_generation.csv found in $PRED_DIR"
    exit 1
fi

age=$(( $(date +%s) - $(stat -c %Y "$newest") ))
if [ "$age" -gt "$MAX_AGE" ]; then
    echo "UNHEALTHY: $(basename "$newest") is ${age}s old (limit ${MAX_AGE}s) — loop appears stalled"
    exit 1
fi

echo "healthy: $(basename "$newest") updated ${age}s ago"
exit 0
