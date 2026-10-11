#!/bin/sh
# One short CPU composite pipeline: tests -> synthetic smoke -> frozen 3-seed audit.
# No GPU job is submitted, no OOM retry is needed, and no GPU memory is reserved.
set -eu
ROOT=/mnt/data/zza/code/S4M_MissingForecasting
PYTHON=/mnt/data/zza/anaconda3/envs/s4m_missing_forecasting/bin/python
RUNTIME="$ROOT/runtime/stage22_33_psavs"
mkdir -p "$RUNTIME"
case "${1:-start}" in
  run)
    # Tool-managed execution session keeps the child alive when detached shells
    # are cleaned up by the execution service. The same pipeline lock still applies.
    cd "$ROOT"
    exec env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      "$PYTHON" -u scripts/run_stage22_33_psavs.py --mode pipeline
    ;;
  status)
    if [ -f "$RUNTIME/state.json" ]; then
      "$PYTHON" -m json.tool "$RUNTIME/state.json"
    else
      echo "NOT_STARTED"
    fi
    ;;
  start|resume)
    (
      flock -n 9 || exit 0
      if [ -f "$RUNTIME/launcher.pid" ]; then
        PID=$(head -n 1 "$RUNTIME/launcher.pid")
        if kill -0 "$PID" 2>/dev/null && ps -p "$PID" -o args= | rg -q '[p]ython.*scripts/run_stage22_33_psavs.py --mode pipeline'; then
          echo "PSAVS pipeline PID $PID exists; inspect status before starting a duplicate."
          exit 0
        fi
      fi
      cd "$ROOT"
      nohup env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        "$PYTHON" -u scripts/run_stage22_33_psavs.py --mode pipeline \
        >> "$RUNTIME/pipeline.log" 2>&1 < /dev/null 9>&- &
      PID=$!
      echo "$PID" > "$RUNTIME/launcher.pid"
      echo "Started CPU pipeline PID=$PID; log=$RUNTIME/pipeline.log"
    ) 9> "$RUNTIME/launch.lock"
    ;;
  *) echo "usage: sh $0 [start|status|resume|run]" >&2; exit 2 ;;
esac
