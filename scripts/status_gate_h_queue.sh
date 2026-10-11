#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/zza/code/S4M_MissingForecasting"
PID_FILE="$ROOT/runtime/gate_h_queue.pid"
STATE_FILE="$ROOT/runtime/gate_h_queue/state.json"
LOG_FILE="$ROOT/logs/gate_h_queue/nohup.log"

if [[ -f "$PID_FILE" ]]; then
  PID="$(<"$PID_FILE")"
  ps -p "$PID" -o pid,ppid,sid,stat,etime,cmd || true
else
  echo "pid_file=missing"
fi
if [[ -f "$STATE_FILE" ]]; then
  sed -n '1,420p' "$STATE_FILE"
else
  echo "state_file=missing"
fi
if [[ -f "$LOG_FILE" ]]; then
  tail -n 80 "$LOG_FILE"
fi
