#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="$ROOT/runtime/stage22_mcpr_attribution"
LOG_ROOT="$ROOT/logs/stage22_mcpr_attribution"
PID_FILE="$RUNTIME/scheduler.pid"
LOG_FILE="$LOG_ROOT/scheduler.log"
GPU_CANDIDATES="${GPU_CANDIDATES:-GPU-c88b779d-77c7-92b9-e031-943bae990121,GPU-a3af3617-5bcd-4c65-a68e-17e716b31edb}"
RESET_JOBS="${RESET_JOBS:-}"
RESET_REASON="${RESET_REASON:-verified_code_fix}"
RESET_ARGS=()
if [[ -n "$RESET_JOBS" ]]; then
  RESET_ARGS=(--reset-jobs "$RESET_JOBS" --reset-reason "$RESET_REASON")
fi
mkdir -p "$RUNTIME" "$LOG_ROOT"
if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(<"$PID_FILE")"
  OLD_COMMAND="$(ps -p "$OLD_PID" -o args= 2>/dev/null || true)"
  if [[ "$OLD_COMMAND" == *"run_stage22_mcpr_attribution_queue.py"* ]]; then
    echo "stage22.17 MCPR attribution queue already running pid=$OLD_PID"
    exit 0
  fi
fi
rm -f "$RUNTIME/stop.requested"
cd "$ROOT"
nohup setsid /mnt/data/zza/anaconda3/envs/s4m_missing_forecasting/bin/python \
  scripts/run_stage22_mcpr_attribution_queue.py \
  --gpu-candidates "$GPU_CANDIDATES" \
  --max-jobs "${MAX_JOBS:-5}" \
  --poll-seconds "${POLL_SECONDS:-15}" \
  --resume \
  "${RESET_ARGS[@]}" \
  >"$LOG_FILE" 2>&1 < /dev/null &
echo $! >"$PID_FILE"
echo "started stage22.17 MCPR attribution queue pid=$(<"$PID_FILE") gpu_candidates=$GPU_CANDIDATES max_jobs=${MAX_JOBS:-5} reset_jobs=${RESET_JOBS:-none} reset_reason=$RESET_REASON log=$LOG_FILE"
