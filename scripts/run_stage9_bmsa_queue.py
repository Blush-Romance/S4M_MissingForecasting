#!/usr/bin/env python3
"""Durable smoke-to-formal queue for the Step-9 BMSA screen."""

from __future__ import annotations

import run_stage5_baseline_queue as engine


ROOT = engine.PROJECT_ROOT
engine.RUNTIME_ROOT = ROOT / "runtime" / "stage9_bmsa_queue"
engine.STATE_PATH = engine.RUNTIME_ROOT / "state.json"
engine.EVENTS_PATH = engine.RUNTIME_ROOT / "events.jsonl"
engine.STOP_PATH = engine.RUNTIME_ROOT / "stop.requested"
engine.SCHEDULER_LOCK_PATH = engine.RUNTIME_ROOT / "scheduler.lock"
engine.LOG_ROOT = ROOT / "logs" / "stage9_bmsa_queue"
engine.QUEUE_NAME = "stage9_weather_bmsa_budget_matched_stability_screen"
engine.AGGREGATOR = ROOT / "scripts" / "aggregate_stage9_bmsa_results.py"
engine.JOB_DEFINITIONS = [
    {
        "id": "weather_bmsa50_smoke",
        "config": "configs/mipp/s4m_weather_mipp_bmsa50_seed2023_smoke.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 6656,
    },
    {
        "id": "weather_bmsa50_formal",
        "config": "configs/mipp/s4m_weather_mipp_bmsa50_seed2023.yaml",
        "depends_on": ["weather_bmsa50_smoke"],
        "required_memory_estimate_mib": 6656,
    },
]


if __name__ == "__main__":
    raise SystemExit(engine.main())
