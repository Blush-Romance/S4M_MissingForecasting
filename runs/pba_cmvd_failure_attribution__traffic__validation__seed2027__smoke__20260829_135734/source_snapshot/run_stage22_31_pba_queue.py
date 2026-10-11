#!/usr/bin/env python3
"""Step22.31 PBA failure-attribution smoke-to-formal queue."""

from __future__ import annotations

import json
from pathlib import Path

import run_stage5_baseline_queue as scheduler


ROOT = scheduler.PROJECT_ROOT
SMOKE_JOB = "pba_cmvd_failure_attribution_seed2027_smoke"
FORMAL_JOB = "pba_cmvd_failure_attribution_seed2027_formal"
scheduler.RUNTIME_ROOT = ROOT / "runtime/stage22_31_pba_diagnostic"
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / "state.json"
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / "events.jsonl"
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / "stop.requested"
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / "scheduler.lock"
scheduler.LOG_ROOT = ROOT / "logs/stage22_31_pba_diagnostic"
scheduler.RUNNER = ROOT / "scripts/run_stage22_31_pba_diagnostic.py"
scheduler.AGGREGATOR = ROOT / "scripts/analyze_stage22_31_pba_failure.py"
scheduler.QUEUE_NAME = "stage22_31_pba_failure_attribution_validation_only"
scheduler.JOB_DEFINITIONS = [
    {
        "id": SMOKE_JOB,
        "config": "configs/stage22_31_pba_diagnostic/pba_seed2027_smoke.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 8192,
    },
    {
        "id": FORMAL_JOB,
        "config": "configs/stage22_31_pba_diagnostic/pba_seed2027.yaml",
        "depends_on": [SMOKE_JOB],
        "required_memory_estimate_mib": 8192,
    },
]


_base_update_dependency_states = scheduler.update_dependency_states


def enforce_scientific_gate(state: dict) -> None:
    _base_update_dependency_states(state)
    smoke = state["jobs"][SMOKE_JOB]
    formal = state["jobs"][FORMAL_JOB]
    if smoke.get("state") != "SUCCESS" or formal.get("state") not in scheduler.WAITING_STATES:
        return
    metrics_path = Path(smoke["run_dir"]) / "metrics.json" if smoke.get("run_dir") else None
    passed = False
    reason = "smoke metrics missing"
    checks = {}
    if metrics_path and metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        artifact_checks = metrics.get("artifact_checks", {})
        checks = {
            "artifact_status": (
                metrics.get("artifact_status") == "PASS"
                and bool(artifact_checks)
                and all(artifact_checks.values())
            ),
            "validation_only": (
                metrics.get("evaluation_split") == "validation"
                and metrics.get("test_split_accessed") is False
                and metrics.get("uses_test_split") is False
                and int(metrics.get("validation_batches_processed", -1)) == 3
            ),
            "frozen_models": (
                metrics.get("pba_model_hash_before")
                == metrics.get("pba_model_hash_after")
                and metrics.get("source_model_hash_before")
                == metrics.get("source_model_hash_after")
            ),
            "frozen_memories": (
                metrics.get("pba_memory_hash_before")
                == metrics.get("pba_memory_hash_after")
                and metrics.get("source_memory_hash_before")
                == metrics.get("source_memory_hash_after")
            ),
            "paired_forwards": (
                int(metrics.get("pba_student_forward_count", -1)) == 3
                and int(metrics.get("source_forward_count", -1)) == 9
            ),
            "pba_delta_verified": (
                metrics.get("checkpoint_format") == "pba_delta_v1"
                and int(metrics.get("checkpoint_pba_tail_layers", -1)) == 2
                and len(metrics.get("checkpoint_pba_state_keys", [])) == 60
            ),
        }
        passed = all(checks.values())
        reason = (
            "Step22.31 smoke artifact gate PASS" if passed
            else "Step22.31 paired diagnostic evidence failed"
        )
    formal["scientific_dependency"] = {
        "source_job": SMOKE_JOB,
        "passed": passed,
        "reason": reason,
        "checks": checks,
        "metrics_path": str(metrics_path) if metrics_path else None,
    }
    if not passed:
        formal["state"] = "BLOCKED_DEPENDENCY"
        formal["failure_summary"] = f"scientific gate blocked formal: {reason}"
        formal["finished_at"] = scheduler.iso_now()
        scheduler.append_event(
            "blocked_scientific_dependency",
            job_id=FORMAL_JOB,
            source_job=SMOKE_JOB,
            reason=reason,
            checks=checks,
        )


scheduler.update_dependency_states = enforce_scientific_gate


if __name__ == "__main__":
    raise SystemExit(scheduler.main())

