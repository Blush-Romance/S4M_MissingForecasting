#!/usr/bin/env python3
"""Smoke-gated dynamic GPU queue for Step 22.20 ARD diagnosis."""

from __future__ import annotations

import run_stage5_baseline_queue as scheduler


ROOT = scheduler.PROJECT_ROOT
scheduler.RUNTIME_ROOT = ROOT / "runtime/stage22_20_ard_diagnostic"
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / "state.json"
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / "events.jsonl"
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / "stop.requested"
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / "scheduler.lock"
scheduler.LOG_ROOT = ROOT / "logs/stage22_20_ard_diagnostic"
scheduler.RUNNER = ROOT / "scripts/run_stage22_20_ard_diagnostic.py"
scheduler.AGGREGATOR = ROOT / "scripts/analyze_stage22_20_ard_information.py"
scheduler.QUEUE_NAME = "stage22_20_ard_information_bottleneck_diagnostic"
scheduler.JOB_DEFINITIONS = [
    {
        "id": "ard_information_smoke",
        "config": "configs/stage22_ard_diagnostic/ard_seed2025_smoke.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 6144,
    },
    {
        "id": "ard_information_formal",
        "config": "configs/stage22_ard_diagnostic/ard_seed2025.yaml",
        "depends_on": ["ard_information_smoke"],
        "required_memory_estimate_mib": 6144,
    },
]


if __name__ == "__main__":
    raise SystemExit(scheduler.main())

