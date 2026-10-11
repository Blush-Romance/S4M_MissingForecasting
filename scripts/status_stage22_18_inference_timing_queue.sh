#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="$ROOT/runtime/stage22_18_inference_timing/state.json"
PID_FILE="$ROOT/runtime/stage22_18_inference_timing/scheduler.pid"
if [[ -f "$PID_FILE" ]]; then
  PID="$(<"$PID_FILE")"
  ps -p "$PID" -o pid,ppid,sid,stat,etime,args || true
fi
if [[ -f "$STATE" ]]; then
  /mnt/data/zza/anaconda3/envs/s4m_missing_forecasting/bin/python -m json.tool "$STATE"
else
  echo "state not found: $STATE"
fi
