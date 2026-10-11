#!/usr/bin/env python3
"""Four existing-checkpoint evaluations; shared atomic GPU/OOM queue."""
import math
from pathlib import Path

import run_stage5_baseline_queue as scheduler

ROOT = scheduler.PROJECT_ROOT
scheduler.RUNTIME_ROOT = ROOT / "runtime/stage22_34_factorial"
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / "state.json"
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / "events.jsonl"
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / "stop.requested"
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / "scheduler.lock"
scheduler.LOG_ROOT = ROOT / "logs/stage22_34_factorial"
scheduler.RUNNER = ROOT / "scripts/stage22_34_factorial.py"
scheduler.AGGREGATOR = scheduler.RUNNER
scheduler.QUEUE_NAME = "stage22_34_validation_factorial_no_training_no_test"


def refresh_memory(job, jobs):
    # Do not erase estimates increased by a genuine OOM retry.
    peak = 3932.
    for source in jobs.values():
        if source.get("state") == "SUCCESS" and source.get("run_dir"):
            data = scheduler.read_json(Path(source["run_dir"]) / "resource_usage.json")
            peak = max(peak, float(data.get("peak_gpu_reserved_mb") or 0))
    margin = max(512., .15 * peak)
    estimate = int(math.ceil((peak + margin) / 256) * 256)
    job["required_memory_estimate_mib"] = max(job["required_memory_estimate_mib"], estimate)
    job["memory_estimate_source"] = dict(historical_and_successful_peak_reserved_mib=peak,
        safety_margin_mib=margin, configuration_unchanged=True)


scheduler.refresh_formal_memory_estimate = refresh_memory
scheduler.JOB_DEFINITIONS = []
for seed, smoke in ((2027, True), (2027, False), (2026, False), (2028, False)):
    parent = [] if smoke else [f"traffic_s4m_validation_seed2027_{'smoke' if seed == 2027 else 'formal'}"]
    scheduler.JOB_DEFINITIONS.append(dict(id=f"traffic_s4m_validation_seed{seed}_{'smoke' if smoke else 'formal'}",
        config=f"configs/stage22_34_factorial/s4m_traffic_validation_seed{seed}{'_smoke' if smoke else ''}.yaml",
        depends_on=parent, required_memory_estimate_mib=4608))


if __name__ == "__main__":
    raise SystemExit(scheduler.main())
