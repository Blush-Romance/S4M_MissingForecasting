#!/usr/bin/env python3
"""Aggregate Step-22.25 CUDA-stream smoke/formal evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime/stage22_25_traffic_pipeline/state.json"
REPORT_JSON = ROOT / "reports/stage22_25_cuda_stream_seed2027.json"
REPORT_MD = ROOT / "reports/stage22_25_cuda_stream_seed2027.md"
SMOKE_JOB = "traffic_cuda_stream_validation_seed2027_smoke"
FORMAL_JOB = "traffic_cuda_stream_validation_seed2027_formal"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def run_metrics(job: dict) -> tuple[dict, str | None]:
    if job.get("state") != "SUCCESS" or not job.get("run_dir"):
        return {}, None
    path = Path(job["run_dir"]).resolve() / "metrics.json"
    return (load_json(path), str(path.parent)) if path.is_file() else ({}, None)


def decision(smoke: dict, formal: dict, formal_state: str) -> str:
    if not smoke:
        return "CUDA_STREAM_SMOKE_INCOMPLETE"
    if smoke.get("artifact_status") != "PASS":
        return "CUDA_STREAM_PROTOCOL_FAIL"
    if not smoke.get("prediction_equivalence_pass"):
        return "CUDA_STREAM_EQUIVALENCE_FAIL_FORMAL_BLOCKED"
    if not formal:
        return (
            "CUDA_STREAM_FORMAL_INCOMPLETE"
            if formal_state != "BLOCKED_DEPENDENCY"
            else "CUDA_STREAM_FORMAL_BLOCKED"
        )
    if formal.get("artifact_status") != "PASS":
        return "CUDA_STREAM_PROTOCOL_FAIL"
    if not formal.get("prediction_equivalence_pass"):
        return "CUDA_STREAM_EQUIVALENCE_FAIL"
    if not formal.get("speed_gate_pass"):
        return "CUDA_STREAM_SPEED_GATE_FAIL"
    return "CUDA_STREAM_REPLICATION_SEED2028_AUTHORIZED"


def compact(metrics: dict) -> dict:
    if not metrics:
        return {}
    return {
        "artifact_status": metrics.get("artifact_status"),
        "prediction_equivalence_pass": metrics.get("prediction_equivalence_pass"),
        "maximum_absolute_prediction_difference": metrics.get(
            "maximum_absolute_prediction_difference"
        ),
        "serial_mse": metrics.get("complete_target_mse_serial_multiscale"),
        "parallel_mse": metrics.get("complete_target_mse_parallel_multiscale"),
        "serial_mae": metrics.get("complete_target_mae_serial_multiscale"),
        "parallel_mae": metrics.get("complete_target_mae_parallel_multiscale"),
        "speed_gate_pass": metrics.get("speed_gate_pass"),
        "serial_pipeline_timing": metrics.get("serial_pipeline_timing"),
        "parallel_pipeline_timing": metrics.get("parallel_pipeline_timing"),
        "paired_latency_reduction_percent_mean": metrics.get(
            "paired_latency_reduction_percent_mean"
        ),
        "paired_latency_reduction_percent_bootstrap_ci95": metrics.get(
            "paired_latency_reduction_percent_bootstrap_ci95"
        ),
        "peak_gpu_memory_mb": metrics.get("peak_gpu_memory_mb"),
    }


def main() -> int:
    if not STATE.is_file():
        return 0
    state = load_json(STATE)
    smoke, smoke_run = run_metrics(state["jobs"][SMOKE_JOB])
    formal, formal_run = run_metrics(state["jobs"][FORMAL_JOB])
    scientific_decision = decision(
        smoke, formal, state["jobs"][FORMAL_JOB].get("state", "")
    )
    report = {
        "step": "22.25",
        "queue_state": state.get("queue_state"),
        "smoke_run": smoke_run,
        "formal_run": formal_run,
        "scientific_dependency": state["jobs"][FORMAL_JOB].get(
            "scientific_dependency"
        ),
        "smoke": compact(smoke),
        "formal": compact(formal),
        "decision": scientific_decision,
    }
    atomic_text(REPORT_JSON, json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# 步骤22.25 CUDA多流结果", "",
        f"- 决策：`{scientific_decision}`",
        f"- 队列：`{state.get('queue_state')}`",
        f"- smoke等价：`{smoke.get('prediction_equivalence_pass', 'NA')}`",
        f"- formal状态：`{state['jobs'][FORMAL_JOB].get('state')}`",
    ]
    if formal:
        lines += [
            f"- 正式平均耗时降低：{formal['paired_latency_reduction_percent_mean']:.6f}%",
            f"- 正式速度门：`{formal['speed_gate_pass']}`",
        ]
    atomic_text(REPORT_MD, "\n".join(lines) + "\n")
    print(json.dumps({
        "decision": scientific_decision,
        "queue_state": state.get("queue_state"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
