#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="$ROOT/runtime/stage8_hcup_queue"
mkdir -p "$RUNTIME" "$ROOT/logs/stage8_hcup_queue"
PID_FILE="$RUNTIME/scheduler.pid"
LOG_FILE="$ROOT/logs/stage8_hcup_queue/scheduler.log"
if [[ -f "$PID_FILE" ]] && kill -0 "$(<"$PID_FILE")" 2>/dev/null; then
  echo "stage8 HCUP queue already running pid=$(<"$PID_FILE")"
  exit 0
fi
rm -f "$RUNTIME/stop.requested"
cd "$ROOT"
nohup setsid /mnt/data/zza/anaconda3/envs/s4m_missing_forecasting/bin/python \
  scripts/run_stage8_hcup_queue.py \
  --gpu-candidates "${GPU_CANDIDATES:-GPU-c88b779d-77c7-92b9-e031-943bae990121,GPU-a3af3617-5bcd-4c65-a68e-17e716b31edb}" \
  --max-jobs "${MAX_JOBS:-5}" \
  --poll-seconds "${POLL_SECONDS:-30}" \
  >"$LOG_FILE" 2>&1 < /dev/null &
echo $! >"$PID_FILE"
echo "started stage8 HCUP queue pid=$(<"$PID_FILE") max_jobs=${MAX_JOBS:-5} gpu_candidates=${GPU_CANDIDATES:-GPU-c88b779d-77c7-92b9-e031-943bae990121,GPU-a3af3617-5bcd-4c65-a68e-17e716b31edb} log=$LOG_FILE"
