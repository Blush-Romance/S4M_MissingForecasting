#!/usr/bin/env python3
"""Step-7 MIPP queue, reusing the verified Stage-5 scheduler engine."""

from __future__ import annotations

from pathlib import Path

import run_stage5_baseline_queue as engine


ROOT = Path(__file__).resolve().parents[1]
engine.RUNTIME_ROOT = ROOT / "runtime" / "stage7_mipp_queue"
engine.STATE_PATH = engine.RUNTIME_ROOT / "state.json"
engine.EVENTS_PATH = engine.RUNTIME_ROOT / "events.jsonl"
engine.STOP_PATH = engine.RUNTIME_ROOT / "stop.requested"
engine.SCHEDULER_LOCK_PATH = engine.RUNTIME_ROOT / "scheduler.lock"
engine.LOG_ROOT = ROOT / "logs" / "stage7_mipp_queue"
engine.QUEUE_NAME = "stage7_weather_mipp_equal_budget_screen"
engine.JOB_DEFINITIONS = [
    {
        "id": "weather_mipp_smoke",
        "config": "configs/mipp/s4m_weather_mipp_seed2023_smoke.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 7168,
    },
    {
        "id": "weather_mipp_formal",
        "config": "configs/mipp/s4m_weather_mipp_seed2023.yaml",
        "depends_on": ["weather_mipp_smoke"],
        "required_memory_estimate_mib": 8192,
    },
]


if __name__ == "__main__":
    raise SystemExit(engine.main())
