#!/usr/bin/env python3
"""Durable smoke-to-formal queue for the Step-11 MRBA anchor screen."""

from __future__ import annotations

import run_stage5_baseline_queue as engine


ROOT = engine.PROJECT_ROOT
engine.RUNTIME_ROOT = ROOT / "runtime" / "stage11_mrba_anchor_queue"
engine.STATE_PATH = engine.RUNTIME_ROOT / "state.json"
engine.EVENTS_PATH = engine.RUNTIME_ROOT / "events.jsonl"
engine.STOP_PATH = engine.RUNTIME_ROOT / "stop.requested"
engine.SCHEDULER_LOCK_PATH = engine.RUNTIME_ROOT / "scheduler.lock"
engine.LOG_ROOT = ROOT / "logs" / "stage11_mrba_anchor_queue"
engine.QUEUE_NAME = "stage11_weather_mrba_seed2024_2025_pre_registered_anchor_screen"
engine.AGGREGATOR = ROOT / "scripts" / "aggregate_stage11_mrba_anchor_results.py"
engine.JOB_DEFINITIONS = [
    {
        "id": f"weather_mrba50_seed{seed}_smoke",
        "config": f"configs/mipp/s4m_weather_mipp_mrba50_seed{seed}_smoke.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 6656,
    }
    for seed in (2024, 2025)
] + [
    {
        "id": f"weather_mrba50_seed{seed}_formal",
        "config": f"configs/mipp/s4m_weather_mipp_mrba50_seed{seed}.yaml",
        "depends_on": [f"weather_mrba50_seed{seed}_smoke"],
        "required_memory_estimate_mib": 6656,
    }
    for seed in (2024, 2025)
]


if __name__ == "__main__":
    raise SystemExit(engine.main())
