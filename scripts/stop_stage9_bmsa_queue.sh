#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/runtime/stage9_bmsa_queue"
touch "$ROOT/runtime/stage9_bmsa_queue/stop.requested"
echo "stage9 BMSA queue stop requested; running experiment will finish safely"
