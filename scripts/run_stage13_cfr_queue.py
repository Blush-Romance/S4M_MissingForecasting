#!/usr/bin/env python3
"""Durable smoke-to-formal queue for the Step-13 CFR screen."""

from __future__ import annotations

import run_stage5_baseline_queue as engine


ROOT = engine.PROJECT_ROOT
engine.RUNTIME_ROOT = ROOT / "runtime" / "stage13_cfr_queue"
engine.STATE_PATH = engine.RUNTIME_ROOT / "state.json"
engine.EVENTS_PATH = engine.RUNTIME_ROOT / "events.jsonl"
engine.STOP_PATH = engine.RUNTIME_ROOT / "stop.requested"
engine.SCHEDULER_LOCK_PATH = engine.RUNTIME_ROOT / "scheduler.lock"
engine.LOG_ROOT = ROOT / "logs" / "stage13_cfr_queue"
engine.QUEUE_NAME = "stage13_weather_pg95_counterfactual_regret_screen"
engine.AGGREGATOR = ROOT / "scripts" / "aggregate_stage13_cfr_results.py"
engine.JOB_DEFINITIONS = [
    {
        "id": "weather_pg95_default_off_regression_smoke",
        "config": "configs/stage13_cfr/s4m_weather_pg95_seed2023_regression_smoke.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 6656,
    },
    {
        "id": "weather_pg95_cfr_seed2023_smoke",
        "config": "configs/stage13_cfr/s4m_weather_pg95_cfr_seed2023_smoke.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 9216,
    },
    {
        "id": "weather_pg95_cfr_seed2023_formal",
        "config": "configs/stage13_cfr/s4m_weather_pg95_cfr_seed2023.yaml",
        "depends_on": [
            "weather_pg95_default_off_regression_smoke",
            "weather_pg95_cfr_seed2023_smoke",
        ],
        "required_memory_estimate_mib": 9216,
    },
]


if __name__ == "__main__":
    raise SystemExit(engine.main())
