#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/zza/code/S4M_MissingForecasting"
PY="/mnt/data/zza/anaconda3/envs/s4m_missing_forecasting/bin/python"
PID_FILE="$ROOT/runtime/gate_h_queue.pid"
LOG_FILE="$ROOT/logs/gate_h_queue/nohup.log"
STOP_FILE="$ROOT/runtime/gate_h_queue/stop.requested"
SESSION="${SESSION:-s4m_gate_h_step6}"
MAX_JOBS="${MAX_JOBS:-3}"
RESET_JOBS="${RESET_JOBS:-}"
RESET_REASON="${RESET_REASON:-verified_code_fix}"

mkdir -p "$ROOT/runtime/gate_h_queue" "$ROOT/logs/gate_h_queue"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Gate H tmux queue already running session=$SESSION"
  exit 0
fi
if [[ -f "$PID_FILE" ]] && kill -0 "$(<"$PID_FILE")" 2>/dev/null; then
  echo "Gate H queue already running pid=$(<"$PID_FILE")"
  exit 0
fi

"$PY" -c "from pathlib import Path; Path('$STOP_FILE').unlink(missing_ok=True)"
tmux new-session -d -s "$SESSION" \
  "cd '$ROOT' && exec '$PY' scripts/run_gate_h_queue.py --gpu-candidates GPU-c88b779d-77c7-92b9-e031-943bae990121,GPU-a3af3617-5bcd-4c65-a68e-17e716b31edb --max-jobs '$MAX_JOBS' --poll-seconds 30 --resume --reset-jobs '$RESET_JOBS' --reset-reason '$RESET_REASON' >>'$LOG_FILE' 2>&1"
PANE_PID="$(tmux list-panes -t "$SESSION" -F '#{pane_pid}' | head -n 1)"
printf '%s\n' "$PANE_PID" > "$PID_FILE"
echo "started Gate H tmux queue session=$SESSION pid=$PANE_PID max_jobs=$MAX_JOBS reset_jobs=$RESET_JOBS log=$LOG_FILE"
