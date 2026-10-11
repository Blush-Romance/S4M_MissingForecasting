#!/usr/bin/env python3
"""Durable smoke-to-formal queue for the frozen MRRA-v3 diagnosis."""

from __future__ import annotations

import run_stage5_baseline_queue as scheduler


ROOT = scheduler.PROJECT_ROOT
scheduler.RUNTIME_ROOT = ROOT / "runtime" / "stage22_mrra_v3_diagnostic"
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / "state.json"
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / "events.jsonl"
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / "stop.requested"
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / "scheduler.lock"
scheduler.LOG_ROOT = ROOT / "logs" / "stage22_mrra_v3_diagnostic"
scheduler.RUNNER = ROOT / "scripts" / "run_stage12_validation_cache.py"
scheduler.AGGREGATOR = ROOT / "scripts" / "aggregate_stage22_mrra_v3_diagnostic.py"
scheduler.QUEUE_NAME = "stage22_mrra_v3_frozen_validation_error_diagnosis"
scheduler.JOB_DEFINITIONS = [
    {
        "id": "mrra_v3_electricity_seed2023_validation_smoke",
        "config": "configs/stage22_validation/mrra_v3_electricity_seed2023_smoke.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 6144,
    },
    {
        "id": "mrra_v3_electricity_seed2023_validation_formal",
        "config": "configs/stage22_validation/mrra_v3_electricity_seed2023.yaml",
        "depends_on": ["mrra_v3_electricity_seed2023_validation_smoke"],
        "required_memory_estimate_mib": 6144,
    },
]


if __name__ == "__main__":
    raise SystemExit(scheduler.main())
