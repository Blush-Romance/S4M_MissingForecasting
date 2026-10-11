#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="$ROOT/runtime/stage22_24_traffic_pipeline/state.json"
if [[ ! -f "$STATE" ]]; then
  echo "stage22.24 state not created"
  exit 0
fi
jq '{queue_state,summary,jobs: (.jobs | with_entries(.value |= {state,attempt_count,oom_retry_count,gpu_uuid,run_dir,last_error_type}))}' "$STATE"
