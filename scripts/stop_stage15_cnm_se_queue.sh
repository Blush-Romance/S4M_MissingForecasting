#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/runtime/stage15_cnm_se_queue"
touch "$ROOT/runtime/stage15_cnm_se_queue/stop.requested"
echo "stage15 CNM-SE queue stop requested; running jobs will finish safely"
