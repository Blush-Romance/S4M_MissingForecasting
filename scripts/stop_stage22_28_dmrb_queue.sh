#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/runtime/stage22_28_dmrb"
touch "$ROOT/runtime/stage22_28_dmrb/stop.requested"
echo "stage22.28 scheduler stop requested; running experiment is not killed"
