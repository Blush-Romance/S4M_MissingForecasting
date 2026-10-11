#!/usr/bin/env python3
"""Durable smoke-to-formal queue for the Stage-14 equal ensemble audit."""

from __future__ import annotations

import run_stage5_baseline_queue as scheduler


ROOT = scheduler.PROJECT_ROOT
scheduler.RUNTIME_ROOT = ROOT / "runtime" / "stage14_equal_ensemble_queue"
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / "state.json"
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / "events.jsonl"
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / "stop.requested"
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / "scheduler.lock"
scheduler.LOG_ROOT = ROOT / "logs" / "stage14_equal_ensemble_queue"
scheduler.RUNNER = ROOT / "scripts" / "run_stage14_uncertainty_cache.py"
scheduler.AGGREGATOR = ROOT / "scripts" / "aggregate_stage14_equal_ensemble_results.py"
scheduler.QUEUE_NAME = "stage14_weather_pg95_validation_equal_nested_ensemble"
scheduler.JOB_DEFINITIONS = [
    {
        "id": "weather_pg95_seed2024_equal_ensemble_smoke",
        "config": (
            "configs/stage14_equal_ensemble/"
            "s4m_weather_pg95_equal_ensemble_seed2024_smoke.yaml"
        ),
        "depends_on": [],
        "required_memory_estimate_mib": 4096,
    },
] + [
    {
        "id": f"weather_pg95_seed{seed}_equal_ensemble_formal",
        "config": (
            "configs/stage14_equal_ensemble/"
            f"s4m_weather_pg95_equal_ensemble_seed{seed}.yaml"
        ),
        "depends_on": ["weather_pg95_seed2024_equal_ensemble_smoke"],
        "required_memory_estimate_mib": 4096,
    }
    for seed in (2024, 2025)
]


if __name__ == "__main__":
    raise SystemExit(scheduler.main())
