#!/usr/bin/env python3
"""Dependency-aware 22-job Step-22.22 Traffic replication queue."""

from __future__ import annotations

import run_stage5_baseline_queue as scheduler


ROOT = scheduler.PROJECT_ROOT
scheduler.RUNTIME_ROOT = ROOT / "runtime/stage22_22_traffic_pipeline"
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / "state.json"
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / "events.jsonl"
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / "stop.requested"
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / "scheduler.lock"
scheduler.LOG_ROOT = ROOT / "logs/stage22_22_traffic_pipeline"
scheduler.RUNNER = ROOT / "scripts/run_stage22_22_traffic_pipeline.py"
scheduler.AGGREGATOR = ROOT / "scripts/aggregate_stage22_22_traffic.py"
scheduler.QUEUE_NAME = "stage22_22_traffic_frozen_three_seed_replication"


def training_job(family: str, seed: int, smoke: bool) -> dict:
    suffix = "_smoke" if smoke else "_formal"
    return {
        "id": f"traffic_{family}_seed{seed}{suffix}",
        "config": (
            f"configs/stage22_22_traffic/{family}_traffic_seed{seed}"
            f"{'_smoke' if smoke else ''}.yaml"
        ),
        "depends_on": (
            [] if smoke else [f"traffic_{family}_seed{seed}_smoke"]
        ),
        "required_memory_estimate_mib": (
            14336 if family == "s4m" else 20480
        ),
    }


def eval_job(method: str, split: str, seed: int, smoke: bool) -> dict:
    suffix = "_smoke" if smoke else "_formal"
    middle = "multiscale_validation" if split == "validation" else "multiscale_test"
    identifier = f"traffic_{method}_{middle}_seed{seed}{suffix}"
    if smoke:
        dependencies = [] if seed == 2026 else [f"traffic_{method}_seed{seed}_formal"]
    else:
        dependencies = [f"traffic_{method}_{middle}_seed{seed}_smoke"]
    return {
        "id": identifier,
        "config": (
            f"configs/stage22_22_traffic/{method}_traffic_{middle}_seed{seed}"
            f"{'_smoke' if smoke else ''}.yaml"
        ),
        "depends_on": dependencies,
        # Step-22.21 identical evaluator reserved 3932 MiB; retain >2 GiB
        # launch headroom without changing model, batch size, or protocol.
        "required_memory_estimate_mib": 6144,
    }


jobs = [
    eval_job("s4m", "test", 2026, True),
    eval_job("s4m", "test", 2026, False),
]
for replication_seed in (2027, 2028):
    jobs.extend([
        training_job("s4m", replication_seed, True),
        training_job("s4m", replication_seed, False),
        training_job("pg95", replication_seed, True),
        training_job("pg95", replication_seed, False),
        eval_job("pg95", "validation", replication_seed, True),
        eval_job("pg95", "validation", replication_seed, False),
        eval_job("s4m", "test", replication_seed, True),
        eval_job("s4m", "test", replication_seed, False),
        eval_job("pg95", "test", replication_seed, True),
        eval_job("pg95", "test", replication_seed, False),
    ])

if len(jobs) != 22 or len({job["id"] for job in jobs}) != 22:
    raise RuntimeError("Step-22.22 queue must contain 22 unique jobs")
scheduler.JOB_DEFINITIONS = jobs


if __name__ == "__main__":
    raise SystemExit(scheduler.main())
