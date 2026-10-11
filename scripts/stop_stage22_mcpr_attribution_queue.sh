#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="$ROOT/runtime/stage22_mcpr_attribution"
mkdir -p "$RUNTIME"
touch "$RUNTIME/stop.requested"
echo "stage22.17 MCPR attribution queue will stop after active jobs finish"
