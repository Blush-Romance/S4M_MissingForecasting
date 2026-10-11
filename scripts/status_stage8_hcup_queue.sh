#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="$ROOT/runtime/stage8_hcup_queue/state.json"
PID_FILE="$ROOT/runtime/stage8_hcup_queue/scheduler.pid"
if [[ -f "$PID_FILE" ]]; then
  PID="$(<"$PID_FILE")"
  if [[ -r "/proc/$PID/stat" ]]; then
    echo "scheduler=RUNNING pid=$PID"
  else
    echo "scheduler=NOT_RUNNING stale_pid=$PID"
  fi
else
  echo "scheduler=NOT_STARTED"
fi
if [[ -f "$STATE" ]]; then
  /mnt/data/zza/anaconda3/envs/s4m_missing_forecasting/bin/python -c \
    'import json,sys; d=json.load(open(sys.argv[1])); print("queue_state="+str(d.get("queue_state"))); [(print(k, v.get("state"), v.get("run_dir"))) for k,v in d.get("jobs",{}).items()]' \
    "$STATE"
else
  echo "state=ABSENT"
fi
