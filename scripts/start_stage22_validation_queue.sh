#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="$ROOT/runtime/stage22_validation_queue"
LOG_ROOT="$ROOT/logs/stage22_validation_queue"
PID_FILE="$RUNTIME/scheduler.pid"
LOG_FILE="$LOG_ROOT/scheduler.log"
mkdir -p "$RUNTIME" "$LOG_ROOT"
if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(<"$PID_FILE")"
  OLD_COMMAND="$(ps -p "$OLD_PID" -o args= 2>/dev/null || true)"
  if [[ "$OLD_COMMAND" == *"run_stage22_validation_queue.py"* ]]; then
    echo "stage22 validation queue already running pid=$OLD_PID"
    exit 0
  fi
fi
rm -f "$RUNTIME/stop.requested"
cd "$ROOT"
REQUESTED_RESET="${1:-${RESET_JOBS:-}}"
REQUESTED_REASON="${2:-${RESET_REASON:-verified_stage22_code_fix}}"
RESET_ARGS=()
if [[ -n "$REQUESTED_RESET" ]]; then
  RESET_ARGS+=(--reset-jobs "$REQUESTED_RESET" --reset-reason "$REQUESTED_REASON")
fi
nohup setsid /mnt/data/zza/anaconda3/envs/s4m_missing_forecasting/bin/python \
  scripts/run_stage22_validation_queue.py \
  --gpu-candidates "${GPU_CANDIDATES:-GPU-c88b779d-77c7-92b9-e031-943bae990121,GPU-a3af3617-5bcd-4c65-a68e-17e716b31edb}" \
  --max-jobs "${MAX_JOBS:-5}" \
  --poll-seconds "${POLL_SECONDS:-30}" \
  --resume \
  "${RESET_ARGS[@]}" \
  >"$LOG_FILE" 2>&1 < /dev/null &
echo $! >"$PID_FILE"
echo "started stage22 validation queue pid=$(<"$PID_FILE") max_jobs=${MAX_JOBS:-5} log=$LOG_FILE"
