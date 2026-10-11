#!/usr/bin/env python3
"""Dynamic UUID queue for Step-22.24 seed2027 smoke and formal timing."""

from __future__ import annotations

import run_stage5_baseline_queue as scheduler


ROOT = scheduler.PROJECT_ROOT
scheduler.RUNTIME_ROOT = ROOT / "runtime/stage22_24_traffic_pipeline"
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / "state.json"
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / "events.jsonl"
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / "stop.requested"
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / "scheduler.lock"
scheduler.LOG_ROOT = ROOT / "logs/stage22_24_traffic_pipeline"
scheduler.RUNNER = ROOT / "scripts/run_stage22_24_traffic_pipeline.py"
scheduler.AGGREGATOR = ROOT / "scripts/aggregate_stage22_24_batch_parallel.py"
scheduler.QUEUE_NAME = "stage22_24_traffic_batch_parallel_seed2027"
scheduler.JOB_DEFINITIONS = [
    {
        "id": "traffic_batch_parallel_validation_seed2027_smoke",
        "config": (
            "configs/stage22_24_traffic/"
            "batch_parallel_traffic_validation_seed2027_smoke.yaml"
        ),
        "depends_on": [],
        "required_memory_estimate_mib": 12288,
    },
    {
        "id": "traffic_batch_parallel_validation_seed2027_formal",
        "config": (
            "configs/stage22_24_traffic/"
            "batch_parallel_traffic_validation_seed2027.yaml"
        ),
        "depends_on": ["traffic_batch_parallel_validation_seed2027_smoke"],
        "required_memory_estimate_mib": 12288,
    },
]


if __name__ == "__main__":
    raise SystemExit(scheduler.main())
