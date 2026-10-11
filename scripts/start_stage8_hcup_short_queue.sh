#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="$ROOT/runtime/stage8_hcup_short_queue"
mkdir -p "$RUNTIME" "$ROOT/logs/stage8_hcup_short_queue"
PID_FILE="$RUNTIME/scheduler.pid"
LOG_FILE="$ROOT/logs/stage8_hcup_short_queue/scheduler.log"
if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(<"$PID_FILE")"
  if ps -p "$OLD_PID" -o pid= >/dev/null 2>&1; then
    echo "stage8 short HCUP queue already running pid=$OLD_PID"
    exit 0
  fi
fi
rm -f "$RUNTIME/stop.requested"
cd "$ROOT"
nohup setsid /mnt/data/zza/anaconda3/envs/s4m_missing_forecasting/bin/python \
  scripts/run_stage8_hcup_short_queue.py \
  --gpu-candidates "${GPU_CANDIDATES:-GPU-c88b779d-77c7-92b9-e031-943bae990121,GPU-a3af3617-5bcd-4c65-a68e-17e716b31edb}" \
  --max-jobs "${MAX_JOBS:-5}" \
  --poll-seconds "${POLL_SECONDS:-30}" \
  >"$LOG_FILE" 2>&1 < /dev/null &
echo $! >"$PID_FILE"
echo "started stage8 short HCUP queue pid=$(<"$PID_FILE") max_jobs=${MAX_JOBS:-5} log=$LOG_FILE"
