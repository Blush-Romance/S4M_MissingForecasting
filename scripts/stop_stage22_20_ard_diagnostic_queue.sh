#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/runtime/stage22_20_ard_diagnostic"
touch "$ROOT/runtime/stage22_20_ard_diagnostic/stop.requested"
echo "stop requested; running jobs will not be interrupted"
