#!/usr/bin/env python3
"""Gate H queue using the already-tested durable Step-5 scheduler engine."""

from __future__ import annotations

from pathlib import Path

import run_stage5_baseline_queue as scheduler


scheduler.RUNTIME_ROOT = scheduler.PROJECT_ROOT / "runtime" / "gate_h_queue"
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / "state.json"
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / "events.jsonl"
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / "stop.requested"
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / "scheduler.lock"
scheduler.LOG_ROOT = scheduler.PROJECT_ROOT / "logs" / "gate_h_queue"
scheduler.RUNNER = scheduler.PROJECT_ROOT / "scripts" / "run_gate_h_diagnostic.py"
scheduler.AGGREGATOR = scheduler.PROJECT_ROOT / "scripts" / "aggregate_gate_h_results.py"
scheduler.QUEUE_NAME = "step6_gate_h_id_and_seed2023_shift_diagnostics"
scheduler.JOB_DEFINITIONS = [
    {
        "id": "weather_id_formal",
        "config": "configs/gate_h/s4m_weather_id_seed2023.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 4096,
    },
    {
        "id": "etth1_id_formal",
        "config": "configs/gate_h/s4m_etth1_id_seed2023.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 4096,
    },
    {
        "id": "weather_rate_shift_seed2023_formal",
        "config": "configs/gate_h/s4m_weather_rate_shift_seed2023.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 4096,
    },
    {
        "id": "weather_geometry_shift_seed2023_formal",
        "config": "configs/gate_h/s4m_weather_geometry_shift_seed2023.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 4096,
    },
    {
        "id": "etth1_rate_shift_seed2023_formal",
        "config": "configs/gate_h/s4m_etth1_rate_shift_seed2023.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 4096,
    },
    {
        "id": "etth1_geometry_shift_seed2023_formal",
        "config": "configs/gate_h/s4m_etth1_geometry_shift_seed2023.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 4096,
    },
]


if __name__ == "__main__":
    raise SystemExit(scheduler.main())
