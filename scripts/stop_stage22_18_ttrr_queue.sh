#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/runtime/stage22_18_ttrr"
touch "$ROOT/runtime/stage22_18_ttrr/stop.requested"
echo "stop requested; running jobs will not be interrupted"
