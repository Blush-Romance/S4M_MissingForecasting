#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="$ROOT/runtime/stage22_mrra_internal/state.json"
LOG="$ROOT/logs/stage22_mrra_internal/scheduler.log"
if [[ -f "$STATE" ]]; then
  /mnt/data/zza/anaconda3/envs/s4m_missing_forecasting/bin/python -m json.tool "$STATE"
else
  echo "state file does not exist: $STATE"
fi
if [[ -f "$LOG" ]]; then
  tail -n 30 "$LOG"
fi
