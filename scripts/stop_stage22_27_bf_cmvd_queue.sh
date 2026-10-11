#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/runtime/stage22_27_bf_cmvd_diagnostic"
touch "$ROOT/runtime/stage22_27_bf_cmvd_diagnostic/stop.requested"
echo "stage22.27 scheduler stop requested; running experiment is not killed"
