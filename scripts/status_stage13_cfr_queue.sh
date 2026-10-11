#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT/runtime/stage13_cfr_queue/scheduler.pid"
STATE="$ROOT/runtime/stage13_cfr_queue/state.json"
if [[ -f "$PID_FILE" ]]; then
  echo "scheduler_pid_file=$(<"$PID_FILE")"
else
  echo "scheduler_pid_file=ABSENT"
fi
if [[ -f "$STATE" ]]; then
  /mnt/data/zza/anaconda3/envs/s4m_missing_forecasting/bin/python -c \
    'import json,sys; d=json.load(open(sys.argv[1])); print("queue_state="+str(d.get("queue_state"))); [(print(k,v.get("state"),v.get("run_dir"))) for k,v in d.get("jobs",{}).items()]' \
    "$STATE"
else
  echo "state=ABSENT"
fi
