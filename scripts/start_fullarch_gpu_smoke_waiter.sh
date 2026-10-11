#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/zza/code/S4M_MissingForecasting"
PY="/mnt/data/zza/anaconda3/envs/s4m_missing_forecasting/bin/python"
PID_FILE="$ROOT/runtime/fullarch_gpu_smoke_waiter.pid"
LOG_FILE="$ROOT/logs/fullarch_gpu_smoke_waiter.log"
STOP_FILE="$ROOT/runtime/fullarch_gpu_smoke_waiter/stop.requested"

mkdir -p "$ROOT/runtime/fullarch_gpu_smoke_waiter" "$ROOT/logs"
if [[ -f "$PID_FILE" ]] && kill -0 "$(<"$PID_FILE")" 2>/dev/null; then
  echo "waiter already running pid=$(<"$PID_FILE")"
  exit 0
fi

"$PY" -c "from pathlib import Path; Path('$STOP_FILE').unlink(missing_ok=True)"
cd "$ROOT"
nohup setsid "$PY" scripts/wait_and_run_s4m_gpu_smoke.py \
  --config configs/baseline/s4m_weather_point_r006_fullarch_gpu_smoke.yaml \
  --gpu-candidates GPU-c88b779d-77c7-92b9-e031-943bae990121,GPU-a3af3617-5bcd-4c65-a68e-17e716b31edb \
  --poll-seconds 60 \
  --initial-required-memory-mib 14336 \
  --initial-oom-retry-count 4 \
  --initial-last-launch-free-mib 12969 \
  --initial-last-oom-time 2026-07-15T14:22:36.560961+00:00 \
  --resource-estimate-note official_extension_enabled_and_launcher_cuda_context_removed_after_attempt4 \
  >"$LOG_FILE" 2>&1 < /dev/null &
echo $! > "$PID_FILE"
echo "started waiter pid=$(<"$PID_FILE") log=$LOG_FILE"
