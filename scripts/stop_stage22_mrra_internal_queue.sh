#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="$ROOT/runtime/stage22_mrra_internal"
mkdir -p "$RUNTIME"
touch "$RUNTIME/stop.requested"
echo "stop requested; running jobs will not be interrupted"
