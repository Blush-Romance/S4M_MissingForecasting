#!/usr/bin/env python3
"""Step-22.30 PBA-CMVD smoke-to-formal validation queue."""

from __future__ import annotations

import json
from pathlib import Path

import run_stage5_baseline_queue as scheduler


ROOT = scheduler.PROJECT_ROOT
SMOKE_JOB = "pba_cmvd_traffic_validation_seed2027_smoke"
FORMAL_JOB = "pba_cmvd_traffic_validation_seed2027_formal"
scheduler.RUNTIME_ROOT = ROOT / "runtime/stage22_30_pba_cmvd"
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / "state.json"
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / "events.jsonl"
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / "stop.requested"
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / "scheduler.lock"
scheduler.LOG_ROOT = ROOT / "logs/stage22_30_pba_cmvd"
scheduler.RUNNER = ROOT / "scripts/run_s4m_experiment.py"
scheduler.AGGREGATOR = ROOT / "scripts/aggregate_stage22_30_pba_cmvd.py"
scheduler.QUEUE_NAME = "stage22_30_pba_cmvd_traffic_seed2027"
scheduler.JOB_DEFINITIONS = [
    {
        "id": SMOKE_JOB,
        "config": "configs/stage22_30_pba_cmvd/pba_cmvd_traffic_seed2027_smoke.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 14336,
    },
    {
        "id": FORMAL_JOB,
        "config": "configs/stage22_30_pba_cmvd/pba_cmvd_traffic_seed2027.yaml",
        "depends_on": [SMOKE_JOB],
        "required_memory_estimate_mib": 14336,
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
        groups = metrics.get("pba_trainable_group_gradient_norm_max", {})
        checks = {
            "pba_enabled": metrics.get("pba_cmvd_enabled") is True,
            "exact_tail_scope": (
                int(metrics.get("pba_tail_layers", -1)) == 2
                and int(metrics.get("pba_tail_start_index", -1)) == 6
                and int(metrics.get("pba_trainable_parameter_count", -1)) == 4127582
            ),
            "all_tail_groups_active": (
                set(groups) == {"s4_layers_6", "s4_layers_7", "norm", "out_proj"}
                and all(float(value) > 0.0 for value in groups.values())
            ),
            "token_filter_active": (
                metrics.get("pba_filter_granularity") == "token"
                and 0 < int(metrics.get("pba_beneficial_units", 0))
                < int(metrics.get("pba_eligible_units", 0))
                and float(metrics.get("pba_distill_l1_mean", 0.0)) > 0.0
            ),
            "frozen_student_and_memory": (
                metrics.get("pba_frozen_hash_before")
                == metrics.get("pba_frozen_hash_after")
                and metrics.get("pba_memory_hash_before")
                == metrics.get("pba_memory_hash_after")
            ),
            "tail_changed": (
                metrics.get("pba_trainable_hash_before")
                != metrics.get("pba_trainable_hash_after")
            ),
            "teacher_isolated": (
                metrics.get("pba_teacher_hash_before")
                == metrics.get("pba_teacher_hash_after")
                and metrics.get("pba_teacher_memory_hash_before")
                == metrics.get("pba_teacher_memory_hash_after")
                and metrics.get("pba_teacher_shared_parameter_storage") is False
                and metrics.get("pba_teacher_output_requires_grad") is False
                and metrics.get("pba_teacher_gradient_found") is False
            ),
            "single_view_inference": (
                int(metrics.get("pba_inference_student_forward_count", -1)) == 1
                and int(metrics.get("pba_validation_teacher_forward_count", -1)) == 0
            ),
            "delta_checkpoint_verified": (
                metrics.get("pba_checkpoint_format") == "pba_delta_v1"
                and metrics.get("checkpoint_format") == "pba_delta_v1"
                and int(metrics.get("checkpoint_pba_tail_layers", -1)) == 2
                and len(metrics.get("checkpoint_pba_state_keys", [])) > 0
                and int(metrics.get("checkpoint_size_bytes", 10**9))
                < 32 * 1024 * 1024
            ),
            "validation_only": (
                metrics.get("evaluation_split") == "validation"
                and metrics.get("test_split_accessed") is False
                and int(metrics.get("validation_batches_processed", -1)) == 3
            ),
        }
        passed = all(checks.values())
        reason = "PBA-CMVD smoke PASS" if passed else "PBA-CMVD mechanism/artifact evidence failed"
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
