#!/usr/bin/env python3
"""Durable queue for the Step-10 Weather PG95/BMSA paired-seed study."""

from __future__ import annotations

import run_stage5_baseline_queue as engine


ROOT = engine.PROJECT_ROOT
engine.RUNTIME_ROOT = ROOT / "runtime" / "stage10_weather_multiseed_queue"
engine.STATE_PATH = engine.RUNTIME_ROOT / "state.json"
engine.EVENTS_PATH = engine.RUNTIME_ROOT / "events.jsonl"
engine.STOP_PATH = engine.RUNTIME_ROOT / "stop.requested"
engine.SCHEDULER_LOCK_PATH = engine.RUNTIME_ROOT / "scheduler.lock"
engine.LOG_ROOT = ROOT / "logs" / "stage10_weather_multiseed_queue"
engine.QUEUE_NAME = "stage10_weather_pg95_bmsa_pre_registered_three_seed_replication"
engine.AGGREGATOR = ROOT / "scripts" / "aggregate_stage10_weather_multiseed_results.py"
engine.JOB_DEFINITIONS = [
    {
        "id": f"weather_{method}_seed{seed}_smoke",
        "config": f"configs/mipp/s4m_weather_mipp_{method}_seed{seed}_smoke.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 6656,
    }
    for seed in (2024, 2025)
    for method in ("pg95", "bmsa50")
] + [
    {
        "id": f"weather_{method}_seed{seed}_formal",
        "config": f"configs/mipp/s4m_weather_mipp_{method}_seed{seed}.yaml",
        "depends_on": [f"weather_{method}_seed{seed}_smoke"],
        "required_memory_estimate_mib": 6656,
    }
    for seed in (2024, 2025)
    for method in ("pg95", "bmsa50")
]


if __name__ == "__main__":
    raise SystemExit(engine.main())
