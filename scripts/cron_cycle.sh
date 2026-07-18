#!/bin/bash
# Runs one WM-3 forecast cycle, logs it, and raises a simple alert on failure.
# Invoked by cron every 6h (see scripts/install_cron.sh).
set -uo pipefail
cd "$(dirname "$0")/.."

# cron runs with a minimal environment, so export upload config explicitly here rather
# than relying on env vars set in an interactive shell. AWS credentials themselves live in
# ~/.aws/credentials (outside the repo, not version-controlled), not here.
export S3_BUCKET="${S3_BUCKET:-windbornesystem-mlops-assignment}"

# cron's PATH doesn't include the conda env, so a bare `python3` resolves to the system
# interpreter (no torch/numpy/etc installed) instead of this one -- confirmed the hard way
# when the first real scheduled run failed with ModuleNotFoundError: No module named
# 'torch'. Use the exact interpreter this pipeline was built/tested against.
PYTHON3="/opt/conda/bin/python3"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/cron_$(date -u +%Y%m%d_%H%M%S).log"
ALERT_FILE="$LOG_DIR/alerts.log"

echo "=== WM-3 cycle start: $(date -u -Iseconds) ===" | tee -a "$LOG_FILE"
PYTHONPATH=. "$PYTHON3" scripts/run_live_rollout.py >> "$LOG_FILE" 2>&1
STATUS=$?
echo "=== WM-3 cycle end: $(date -u -Iseconds), exit=$STATUS ===" | tee -a "$LOG_FILE"

if [ $STATUS -ne 0 ]; then
    MSG="$(date -u -Iseconds) WM-3 cycle FAILED (exit=$STATUS), see $LOG_FILE"
    echo "$MSG" | tee -a "$ALERT_FILE"
    if [ -n "${ALERT_WEBHOOK_URL:-}" ]; then
        curl -s -X POST -H 'Content-Type: application/json' \
            -d "{\"text\": \"$MSG\"}" "$ALERT_WEBHOOK_URL" >/dev/null || true
    fi
fi

# Keep only the 20 most recent cron logs so logs/ doesn't grow unbounded over 24h+.
ls -t "$LOG_DIR"/cron_*.log 2>/dev/null | tail -n +21 | xargs -r rm --

exit $STATUS
