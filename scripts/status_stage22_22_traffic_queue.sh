#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT/runtime/stage22_22_traffic_pipeline/scheduler.pid"
STATE="$ROOT/runtime/stage22_22_traffic_pipeline/state.json"
[[ -f "$PID_FILE" ]] && echo "scheduler_pid_file=$(<"$PID_FILE")" || echo "scheduler_pid_file=ABSENT"
if [[ -f "$STATE" ]]; then
  /mnt/data/zza/anaconda3/envs/s4m_missing_forecasting/bin/python -c \
    'import collections,json,sys; d=json.load(open(sys.argv[1])); c=collections.Counter(v.get("state") for v in d.get("jobs",{}).values()); print("queue_state="+str(d.get("queue_state"))); print("counts="+json.dumps(dict(sorted(c.items())),sort_keys=True)); [(print(k,v.get("state"),(v.get("gpu") or {}).get("uuid"),v.get("run_dir"))) for k,v in d.get("jobs",{}).items()]' \
    "$STATE"
else
  echo "state=ABSENT"
fi
