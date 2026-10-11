#!/bin/sh
set -eu
ROOT=/mnt/data/zza/code/S4M_MissingForecasting
PY=/mnt/data/zza/anaconda3/envs/s4m_missing_forecasting/bin/python
RT="$ROOT/runtime/stage22_41_distance_fix"
LOG="$ROOT/logs/stage22_41_distance_fix"
mkdir -p "$RT" "$LOG"
cd "$ROOT"
case "${1:-status}" in
 status) "$PY" -m json.tool "$RT/state.json" ;;
 stop) touch "$RT/stop.requested" ;;
 start|resume)
  if [ -f "$RT/scheduler.pid" ] && kill -0 "$(cat "$RT/scheduler.pid")" 2>/dev/null; then
    echo 'Recorded scheduler PID is live; inspect state before restarting.'; exit 3
  fi
  nohup setsid "$PY" -u scripts/run_stage22_41_queue.py \
    --gpu-candidates GPU-c88b779d-77c7-92b9-e031-943bae990121,GPU-a3af3617-5bcd-4c65-a68e-17e716b31edb \
    --max-jobs 3 --poll-seconds 15 --resume \
    --reset-jobs "${2:-}" --reset-reason "${3:-user_authorized_bug_fix_replication_retry}" \
    >> "$LOG/scheduler.log" 2>&1 < /dev/null &
  echo $! > "$RT/scheduler.pid"
  echo "Launch requested PID=$!; verify live PID, GPU process, state and growing logs."
  ;;
 *) echo 'usage: start_stage22_41.sh [start|resume|status|stop] [reset_jobs] [reason]' >&2; exit 2 ;;
esac
