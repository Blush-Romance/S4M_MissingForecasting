#!/usr/bin/env python3
"""Durable smoke-to-formal queue for Stage22.15 internal features."""

from __future__ import annotations

import run_stage5_baseline_queue as scheduler


ROOT = scheduler.PROJECT_ROOT
scheduler.RUNTIME_ROOT = ROOT / "runtime" / "stage22_mrra_internal"
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / "state.json"
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / "events.jsonl"
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / "stop.requested"
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / "scheduler.lock"
scheduler.LOG_ROOT = ROOT / "logs" / "stage22_mrra_internal"
scheduler.RUNNER = ROOT / "scripts" / "run_stage12_validation_cache.py"
scheduler.AGGREGATOR = ROOT / "scripts" / "analyze_stage22_mrra_internal_features.py"
scheduler.QUEUE_NAME = "stage22_15_mrra_internal_feature_screen"
scheduler.JOB_DEFINITIONS = [
    {
        "id": "mrra_v3_internal_electricity_seed2023_smoke",
        "config": "configs/stage22_internal/mrra_v3_electricity_seed2023_smoke.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 6144,
    },
    {
        "id": "mrra_v3_internal_electricity_seed2023_formal",
        "config": "configs/stage22_internal/mrra_v3_electricity_seed2023.yaml",
        "depends_on": ["mrra_v3_internal_electricity_seed2023_smoke"],
        "required_memory_estimate_mib": 6144,
    },
]


if __name__ == "__main__":
    raise SystemExit(scheduler.main())
