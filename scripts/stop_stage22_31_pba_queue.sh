#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/runtime/stage22_31_pba_diagnostic"
touch "$ROOT/runtime/stage22_31_pba_diagnostic/stop.requested"
echo "stage22.31 scheduler stop requested; running experiment is not killed"
