#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="$ROOT/runtime/stage22_18_inference_timing"
LOG_ROOT="$ROOT/logs/stage22_18_inference_timing"
PID_FILE="$RUNTIME/scheduler.pid"
LOG_FILE="$LOG_ROOT/scheduler.log"
# Exactly one stable UUID is admitted so all six jobs remain on the same GPU.
GPU_CANDIDATES="${GPU_CANDIDATES:-GPU-c88b779d-77c7-92b9-e031-943bae990121}"
if [[ "$GPU_CANDIDATES" == *,* ]]; then
  echo "Step-22.18 timing requires exactly one GPU UUID" >&2
  exit 2
fi
mkdir -p "$RUNTIME" "$LOG_ROOT"
if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(<"$PID_FILE")"
  OLD_COMMAND="$(ps -p "$OLD_PID" -o args= 2>/dev/null || true)"
  if [[ "$OLD_COMMAND" == *"run_stage22_18_inference_timing_queue.py"* ]]; then
    echo "stage22.18 timing queue already running pid=$OLD_PID"
    exit 0
  fi
fi
rm -f "$RUNTIME/stop.requested"
cd "$ROOT"
nohup setsid /mnt/data/zza/anaconda3/envs/s4m_missing_forecasting/bin/python \
  scripts/run_stage22_18_inference_timing_queue.py \
  --gpu-candidates "$GPU_CANDIDATES" \
  --max-jobs 1 \
  --poll-seconds "${POLL_SECONDS:-15}" \
  --resume >"$LOG_FILE" 2>&1 < /dev/null &
echo $! >"$PID_FILE"
echo "started stage22.18 same-GPU timing queue pid=$(<"$PID_FILE") gpu=$GPU_CANDIDATES log=$LOG_FILE"
