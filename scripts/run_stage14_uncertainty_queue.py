#!/usr/bin/env python3
"""Durable smoke-to-formal queue for Stage-14 uncertainty diagnosis."""

from __future__ import annotations

import run_stage5_baseline_queue as scheduler


ROOT = scheduler.PROJECT_ROOT
scheduler.RUNTIME_ROOT = ROOT / "runtime" / "stage14_uncertainty_queue"
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / "state.json"
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / "events.jsonl"
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / "stop.requested"
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / "scheduler.lock"
scheduler.LOG_ROOT = ROOT / "logs" / "stage14_uncertainty_queue"
scheduler.RUNNER = ROOT / "scripts" / "run_stage14_uncertainty_cache.py"
scheduler.AGGREGATOR = ROOT / "scripts" / "aggregate_stage14_uncertainty_results.py"
scheduler.QUEUE_NAME = "stage14_weather_pg95_validation_nested_mask_uncertainty"
scheduler.JOB_DEFINITIONS = [
    {
        "id": "weather_pg95_seed2024_uncertainty_smoke",
        "config": "configs/stage14_uncertainty/s4m_weather_pg95_seed2024_smoke.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 4096,
    },
] + [
    {
        "id": f"weather_pg95_seed{seed}_uncertainty_formal",
        "config": f"configs/stage14_uncertainty/s4m_weather_pg95_seed{seed}.yaml",
        "depends_on": ["weather_pg95_seed2024_uncertainty_smoke"],
        "required_memory_estimate_mib": 4096,
    }
    for seed in (2024, 2025)
]


if __name__ == "__main__":
    raise SystemExit(scheduler.main())
