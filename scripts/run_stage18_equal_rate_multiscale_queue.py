#!/usr/bin/env python3
"""Same-GPU serial smoke-to-formal queue for Stage-18 validation."""

from __future__ import annotations

import run_stage5_baseline_queue as scheduler


ROOT = scheduler.PROJECT_ROOT
scheduler.RUNTIME_ROOT = ROOT / "runtime" / "stage18_equal_rate_multiscale_queue"
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / "state.json"
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / "events.jsonl"
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / "stop.requested"
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / "scheduler.lock"
scheduler.LOG_ROOT = ROOT / "logs" / "stage18_equal_rate_multiscale_queue"
scheduler.RUNNER = ROOT / "scripts" / "run_stage14_uncertainty_cache.py"
scheduler.AGGREGATOR = ROOT / "scripts" / "analyze_stage18_equal_rate_multiscale.py"
scheduler.QUEUE_NAME = "stage18_weather_validation_equal_rate_multiscale_serial_timing"
scheduler.JOB_DEFINITIONS = [
    {
        "id": "weather_equal_rate_multiscale_seed2024_smoke",
        "config": (
            "configs/stage18_equal_rate_multiscale/"
            "s4m_weather_pg95_seed2024_smoke.yaml"
        ),
        "depends_on": [],
        "required_memory_estimate_mib": 4096,
    },
    {
        "id": "weather_equal_rate_multiscale_seed2024_formal",
        "config": (
            "configs/stage18_equal_rate_multiscale/"
            "s4m_weather_pg95_seed2024.yaml"
        ),
        "depends_on": ["weather_equal_rate_multiscale_seed2024_smoke"],
        "required_memory_estimate_mib": 4096,
    },
    {
        "id": "weather_equal_rate_multiscale_seed2025_formal",
        "config": (
            "configs/stage18_equal_rate_multiscale/"
            "s4m_weather_pg95_seed2025.yaml"
        ),
        "depends_on": ["weather_equal_rate_multiscale_seed2024_formal"],
        "required_memory_estimate_mib": 4096,
    },
]


if __name__ == "__main__":
    raise SystemExit(scheduler.main())
