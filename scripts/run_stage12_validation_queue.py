#!/usr/bin/env python3
"""Durable smoke-to-formal queue for the Stage-12 validation-only diagnosis."""

from __future__ import annotations

import run_stage5_baseline_queue as scheduler


ROOT = scheduler.PROJECT_ROOT
scheduler.RUNTIME_ROOT = ROOT / "runtime" / "stage12_validation_queue"
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / "state.json"
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / "events.jsonl"
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / "stop.requested"
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / "scheduler.lock"
scheduler.LOG_ROOT = ROOT / "logs" / "stage12_validation_queue"
scheduler.RUNNER = ROOT / "scripts" / "run_stage12_validation_cache.py"
scheduler.AGGREGATOR = ROOT / "scripts" / "aggregate_stage12_validation_results.py"
scheduler.QUEUE_NAME = "stage12_weather_validation_only_pg95_mrba_mismatch_diagnosis"

SMOKE_IDS = [
    "weather_pg95_seed2024_validation_smoke",
    "weather_mrba50_seed2024_validation_smoke",
]
scheduler.JOB_DEFINITIONS = [
    {
        "id": "weather_pg95_seed2024_validation_smoke",
        "config": "configs/stage12_validation/s4m_weather_pg95_seed2024_smoke.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 6144,
    },
    {
        "id": "weather_mrba50_seed2024_validation_smoke",
        "config": "configs/stage12_validation/s4m_weather_mrba50_seed2024_smoke.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 6144,
    },
] + [
    {
        "id": f"weather_{method}_seed{seed}_validation_formal",
        "config": f"configs/stage12_validation/s4m_weather_{method}_seed{seed}.yaml",
        "depends_on": SMOKE_IDS,
        "required_memory_estimate_mib": 6144,
    }
    for seed in (2024, 2025)
    for method in ("pg95", "mrba50")
]


if __name__ == "__main__":
    raise SystemExit(scheduler.main())
