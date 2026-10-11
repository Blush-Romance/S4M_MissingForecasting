#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/zza/code/S4M_MissingForecasting"
PY="/mnt/data/zza/anaconda3/envs/s4m_missing_forecasting/bin/python"
PID_FILE="$ROOT/runtime/stage5_baseline_queue.pid"
LOG_FILE="$ROOT/logs/stage5_baseline_queue/nohup.log"
STOP_FILE="$ROOT/runtime/stage5_baseline_queue/stop.requested"

mkdir -p "$ROOT/runtime/stage5_baseline_queue" "$ROOT/logs/stage5_baseline_queue"
if [[ -f "$PID_FILE" ]] && kill -0 "$(<"$PID_FILE")" 2>/dev/null; then
  echo "stage5 queue already running pid=$(<"$PID_FILE")"
  exit 0
fi

"$PY" -c "from pathlib import Path; Path('$STOP_FILE').unlink(missing_ok=True)"
cd "$ROOT"
RESET_ARGS=()
if [[ -n "${RESET_JOBS:-}" ]]; then
  RESET_ARGS=(--reset-jobs "$RESET_JOBS")
fi
nohup setsid "$PY" scripts/run_stage5_baseline_queue.py \
  --gpu-candidates GPU-c88b779d-77c7-92b9-e031-943bae990121,GPU-a3af3617-5bcd-4c65-a68e-17e716b31edb \
  --max-jobs 2 \
  --poll-seconds 60 \
  --resume \
  "${RESET_ARGS[@]}" \
  >"$LOG_FILE" 2>&1 < /dev/null &
echo $! > "$PID_FILE"
echo "started stage5 queue pid=$(<"$PID_FILE") log=$LOG_FILE"
