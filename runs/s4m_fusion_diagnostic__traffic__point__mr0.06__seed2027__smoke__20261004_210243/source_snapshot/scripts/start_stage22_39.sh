#!/bin/sh
set -eu
ROOT=/mnt/data/zza/code/S4M_MissingForecasting
PY=/mnt/data/zza/anaconda3/envs/s4m_missing_forecasting/bin/python
RT="$ROOT/runtime/stage22_39_fusion"
LOG="$ROOT/logs/stage22_39_fusion"
mkdir -p "$RT" "$LOG"
cd "$ROOT"
case "${1:-status}" in
 status) "$PY" -m json.tool "$RT/state.json" ;;
 stop) touch "$RT/stop.requested" ;;
 start|resume)
  if [ -f "$RT/scheduler.pid" ] && kill -0 "$(cat "$RT/scheduler.pid")" 2>/dev/null; then
    echo 'Recorded scheduler PID live; inspect before restarting.';exit 3
  fi
  nohup setsid "$PY" -u scripts/run_stage22_39_queue.py \
    --gpu-candidates GPU-c88b779d-77c7-92b9-e031-943bae990121,GPU-a3af3617-5bcd-4c65-a68e-17e716b31edb \
    --max-jobs 3 --poll-seconds 15 --resume >> "$LOG/scheduler.log" 2>&1 < /dev/null &
  echo $! > "$RT/scheduler.pid"
  echo "Launch requested PID=$!; verify actual worker/GPU/log."
  ;;
 *) echo 'usage: start_stage22_39.sh [start|resume|status|stop]' >&2;exit 2 ;;
esac
