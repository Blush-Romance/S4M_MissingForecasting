#!/usr/bin/env python3
"""Durable smoke-to-formal queue for the Stage-22 Electricity diagnosis."""

from __future__ import annotations

import run_stage5_baseline_queue as scheduler


ROOT = scheduler.PROJECT_ROOT
scheduler.RUNTIME_ROOT = ROOT / "runtime" / "stage22_validation_queue"
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / "state.json"
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / "events.jsonl"
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / "stop.requested"
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / "scheduler.lock"
scheduler.LOG_ROOT = ROOT / "logs" / "stage22_validation_queue"
scheduler.RUNNER = ROOT / "scripts" / "run_stage12_validation_cache.py"
scheduler.AGGREGATOR = ROOT / "scripts" / "aggregate_stage22_validation_results.py"
scheduler.QUEUE_NAME = "stage22_electricity_validation_error_and_mask_response_diagnosis"

SMOKE_IDS = [
    "electricity_s4m_seed2023_validation_smoke",
    "electricity_pg95_seed2023_validation_smoke",
]
scheduler.JOB_DEFINITIONS = [
    {
        "id": "electricity_s4m_seed2023_validation_smoke",
        "config": "configs/stage22_validation/s4m_electricity_seed2023_smoke.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 6144,
    },
    {
        "id": "electricity_pg95_seed2023_validation_smoke",
        "config": "configs/stage22_validation/pg95_electricity_seed2023_smoke.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 6144,
    },
    {
        "id": "electricity_s4m_seed2023_validation_formal",
        "config": "configs/stage22_validation/s4m_electricity_seed2023.yaml",
        "depends_on": SMOKE_IDS,
        "required_memory_estimate_mib": 6144,
    },
    {
        "id": "electricity_pg95_seed2023_validation_formal",
        "config": "configs/stage22_validation/pg95_electricity_seed2023.yaml",
        "depends_on": SMOKE_IDS,
        "required_memory_estimate_mib": 6144,
    },
]


if __name__ == "__main__":
    raise SystemExit(scheduler.main())
