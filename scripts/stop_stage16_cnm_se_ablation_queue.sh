#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/runtime/stage16_cnm_se_ablation_queue"
touch "$ROOT/runtime/stage16_cnm_se_ablation_queue/stop.requested"
echo "stage16 CNM-SE ablation queue stop requested; running jobs will finish safely"
