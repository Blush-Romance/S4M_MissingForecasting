#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="$ROOT/runtime/stage22_31_pba_diagnostic/state.json"
if [[ ! -f "$STATE" ]]; then echo "stage22.31 state not created"; exit 0; fi
jq '{queue_state,jobs: (.jobs | with_entries(.value |= {state,attempt_count,oom_retry_count,run_dir,failure_summary,scientific_dependency}))}' "$STATE"

