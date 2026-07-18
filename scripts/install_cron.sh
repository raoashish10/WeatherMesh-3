#!/bin/bash
# Installs the every-6h WM-3 forecast cycle into crontab.
# GFS 0.25deg cycles (00/06/12/18z) typically finish publishing on NOMADS ~3.5-4.5h
# after init time, hence the offset; run_live_rollout.py's find_latest_cycle() also
# falls back to the previous cycle on its own if a given cycle isn't ready yet.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_DIR="$(pwd)"

CRON_LINE="30 4,10,16,22 * * * ${REPO_DIR}/scripts/cron_cycle.sh"

( (crontab -l 2>/dev/null || true) | grep -vF "cron_cycle.sh" || true ; echo "$CRON_LINE" ) | crontab -
echo "Installed crontab entry:"
crontab -l
