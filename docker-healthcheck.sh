#!/usr/bin/env bash
# Container health probe.
#
# The app now captures only at fixed times (see CAPTURE_TIMES in config.py),
# with gaps up to ~15h overnight. So "a predictions CSV was written recently"
# is NO LONGER a valid health signal -- it would report UNHEALTHY for most of
# the day. Instead the main loop refreshes a heartbeat file every
# HEARTBEAT_INTERVAL_SECONDS (60s), including while waiting for the next
# scheduled capture. Health = the loop process is alive AND the heartbeat is
# fresh. This reflects "the scheduler is ticking", independent of the schedule.
set -euo pipefail

HEARTBEAT_FILE="${HEARTBEAT_FILE:-/app/.heartbeat}"

# Max heartbeat age before unhealthy. The loop ticks every 60s, so 300s (5 min)
# tolerates a slow capture cycle without false alarms.
MAX_AGE="${HEALTH_MAX_AGE_SECONDS:-300}"

# 1. The main loop process must be alive.
pgrep -f "test_multi_image.py" >/dev/null 2>&1 \
    || { echo "UNHEALTHY: test_multi_image.py is not running"; exit 1; }

# 2. The heartbeat file must exist and be recent.
if [ ! -f "$HEARTBEAT_FILE" ]; then
    echo "UNHEALTHY: heartbeat file $HEARTBEAT_FILE not found (loop not started?)"
    exit 1
fi

age=$(( $(date +%s) - $(stat -c %Y "$HEARTBEAT_FILE") ))
if [ "$age" -gt "$MAX_AGE" ]; then
    echo "UNHEALTHY: heartbeat is ${age}s old (limit ${MAX_AGE}s) -- loop appears stalled"
    exit 1
fi

echo "healthy: heartbeat updated ${age}s ago"
exit 0
