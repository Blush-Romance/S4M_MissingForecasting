#!/bin/sh
set -eu
ROOT=/mnt/data/zza/code/S4M_MissingForecasting
PY=/mnt/data/zza/anaconda3/envs/s4m_missing_forecasting/bin/python
RUNTIME="$ROOT/runtime/stage22_34_factorial"
LOG="$ROOT/logs/stage22_34_factorial"
GPUS=GPU-c88b779d-77c7-92b9-e031-943bae990121,GPU-a3af3617-5bcd-4c65-a68e-17e716b31edb
mkdir -p "$RUNTIME" "$LOG"
cd "$ROOT"
case "${1:-status}" in
  status) "$PY" -m json.tool "$RUNTIME/state.json" ;;
  stop) touch "$RUNTIME/stop.requested" ;;
  run)
    exec "$PY" -u scripts/run_stage22_34_factorial_queue.py --gpu-candidates "$GPUS" --max-jobs 3 --poll-seconds 15 --resume
    ;;
  start|resume)
    # setsid prevents terminal/process-group cleanup; scheduler flock rejects duplicates.
    nohup setsid "$PY" -u scripts/run_stage22_34_factorial_queue.py --gpu-candidates "$GPUS" \
      --max-jobs 3 --poll-seconds 15 --resume >> "$LOG/scheduler.log" 2>&1 < /dev/null &
    echo $! > "$RUNTIME/scheduler.pid"
    echo "Launch requested PID=$!; verify $RUNTIME/state.json before claiming RUNNING."
    ;;
  *) echo 'usage: start_stage22_34_factorial.sh [start|resume|status|stop|run]' >&2; exit 2 ;;
esac
