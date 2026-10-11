#!/usr/bin/env python3
"""Durable queue for predeclared short-horizon HCUP validation screens."""

from __future__ import annotations

import run_stage5_baseline_queue as scheduler


scheduler.RUNTIME_ROOT = scheduler.PROJECT_ROOT / "runtime" / "stage8_hcup_short_queue"
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / "state.json"
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / "events.jsonl"
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / "stop.requested"
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / "scheduler.lock"
scheduler.LOG_ROOT = scheduler.PROJECT_ROOT / "logs" / "stage8_hcup_short_queue"
scheduler.RUNNER = scheduler.PROJECT_ROOT / "scripts" / "run_hcup_diagnostic.py"
scheduler.AGGREGATOR = scheduler.PROJECT_ROOT / "scripts" / "aggregate_hcup_short_results.py"
scheduler.QUEUE_NAME = "step8_fusr_v3_short_hcup_validation_screen"
scheduler.JOB_DEFINITIONS = []
for lag in (1, 4, 8, 16):
    smoke_id = f"weather_hcup_h{lag}_validation_smoke"
    scheduler.JOB_DEFINITIONS.extend(
        [
            {
                "id": smoke_id,
                "config": f"configs/fusr_v3_hcup/s4m_weather_val_h{lag}_seed2023_smoke.yaml",
                "depends_on": [],
                "required_memory_estimate_mib": 4096,
            },
            {
                "id": f"weather_hcup_h{lag}_validation_formal",
                "config": f"configs/fusr_v3_hcup/s4m_weather_val_h{lag}_seed2023.yaml",
                "depends_on": [smoke_id],
                "required_memory_estimate_mib": 4096,
            },
        ]
    )


if __name__ == "__main__":
    raise SystemExit(scheduler.main())
