#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/runtime/stage22_29_dmrb_diagnostic"
touch "$ROOT/runtime/stage22_29_dmrb_diagnostic/stop.requested"
echo "stage22.29 scheduler stop requested; running experiment is not killed"
