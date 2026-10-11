#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="$ROOT/runtime/stage14_equal_ensemble_queue"
LOG_ROOT="$ROOT/logs/stage14_equal_ensemble_queue"
PID_FILE="$RUNTIME/scheduler.pid"
LOG_FILE="$LOG_ROOT/scheduler.log"
mkdir -p "$RUNTIME" "$LOG_ROOT"
if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(<"$PID_FILE")"
  OLD_COMMAND="$(ps -p "$OLD_PID" -o args= 2>/dev/null || true)"
  if [[ "$OLD_COMMAND" == *"run_stage14_equal_ensemble_queue.py"* ]]; then
    echo "stage14 equal ensemble queue already running pid=$OLD_PID"
    exit 0
  fi
fi
rm -f "$RUNTIME/stop.requested"
cd "$ROOT"
nohup setsid /mnt/data/zza/anaconda3/envs/s4m_missing_forecasting/bin/python \
  scripts/run_stage14_equal_ensemble_queue.py \
  --gpu-candidates "${GPU_CANDIDATES:-GPU-c88b779d-77c7-92b9-e031-943bae990121,GPU-a3af3617-5bcd-4c65-a68e-17e716b31edb}" \
  --max-jobs "${MAX_JOBS:-5}" \
  --poll-seconds "${POLL_SECONDS:-15}" \
  --resume \
  >"$LOG_FILE" 2>&1 < /dev/null &
echo $! >"$PID_FILE"
echo "started stage14 equal ensemble queue pid=$(<"$PID_FILE") max_jobs=${MAX_JOBS:-5} log=$LOG_FILE"
