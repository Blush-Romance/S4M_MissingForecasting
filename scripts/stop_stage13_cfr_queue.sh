#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/runtime/stage13_cfr_queue"
touch "$ROOT/runtime/stage13_cfr_queue/stop.requested"
echo "stage13 CFR queue stop requested; running experiments will finish safely"
