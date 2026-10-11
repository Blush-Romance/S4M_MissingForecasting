#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/runtime/stage8_hcup_queue"
touch "$ROOT/runtime/stage8_hcup_queue/stop.requested"
echo "stage8 HCUP queue stop requested; running diagnostics will finish safely"
