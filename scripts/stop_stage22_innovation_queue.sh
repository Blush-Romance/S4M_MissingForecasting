#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="$ROOT/runtime/stage22_innovation"
mkdir -p "$RUNTIME"
touch "$RUNTIME/stop.requested"
echo "stage22 innovation queue will stop after its active jobs finish"
