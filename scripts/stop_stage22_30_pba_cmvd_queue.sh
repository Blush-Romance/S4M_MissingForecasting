#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/runtime/stage22_30_pba_cmvd"
touch "$ROOT/runtime/stage22_30_pba_cmvd/stop.requested"
echo "stage22.30 scheduler stop requested; running experiment is not killed"
