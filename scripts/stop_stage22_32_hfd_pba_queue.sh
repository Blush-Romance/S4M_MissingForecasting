#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/runtime/stage22_32_hfd_pba"
touch "$ROOT/runtime/stage22_32_hfd_pba/stop.requested"
echo "stage22.32 scheduler stop requested; running experiment is not killed"
