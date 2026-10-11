#!/bin/sh
set -eu
ROOT=/mnt/data/zza/code/S4M_MissingForecasting
PY=/mnt/data/zza/anaconda3/envs/s4m_missing_forecasting/bin/python
RUNTIME="$ROOT/runtime/stage22_34_crn"
LOG="$ROOT/logs/stage22_34_crn"
GPU=GPU-c88b779d-77c7-92b9-e031-943bae990121
mkdir -p "$RUNTIME" "$LOG"
cd "$ROOT"
case "${1:-status}" in
  status) "$PY" -m json.tool "$RUNTIME/state.json" ;;
  stop) touch "$RUNTIME/stop.requested" ;;
  start|resume)
    nohup setsid "$PY" -u scripts/run_stage22_34_crn_queue.py --gpu-candidates "$GPU" \
      --max-jobs 3 --poll-seconds 15 --resume >> "$LOG/scheduler.log" 2>&1 < /dev/null &
    echo $! > "$RUNTIME/scheduler.pid"
    echo "Launch requested PID=$!; verify live process and GPU before claiming RUNNING."
    ;;
  *) echo 'usage: start_stage22_34_crn.sh [start|resume|status|stop]' >&2; exit 2 ;;
esac
