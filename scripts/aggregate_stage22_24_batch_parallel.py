#!/usr/bin/env python3
"""Aggregate the Step-22.24 seed2027 batch-parallel comparison."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime/stage22_24_traffic_pipeline/state.json"
REPORT_JSON = ROOT / "reports/stage22_24_batch_parallel_seed2027.json"
REPORT_MD = ROOT / "reports/stage22_24_batch_parallel_seed2027.md"
FORMAL_JOB = "traffic_batch_parallel_validation_seed2027_formal"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def scientific_decision(
    *, complete: bool, protocol_pass: bool,
    equivalence_pass: bool, speed_gate_pass: bool,
) -> str:
    if not complete:
        return "BATCH_PARALLEL_SEED2027_INCOMPLETE"
    if not protocol_pass:
        return "BATCH_PARALLEL_PROTOCOL_FAIL"
    if not equivalence_pass:
        return "BATCH_PARALLEL_EQUIVALENCE_FAIL"
    if not speed_gate_pass:
        return "BATCH_PARALLEL_SPEED_GATE_FAIL"
    return "BATCH_PARALLEL_REPLICATION_SEED2028_AUTHORIZED"


def main() -> int:
    if not STATE.is_file():
        return 0
    state = load_json(STATE)
    job = state.get("jobs", {}).get(FORMAL_JOB, {})
    run_dir = Path(job["run_dir"]).resolve() if job.get("run_dir") else None
    metrics_path = run_dir / "metrics.json" if run_dir else None
    status_path = run_dir / "status.json" if run_dir else None
    complete = bool(
        job.get("state") == "SUCCESS" and metrics_path
        and metrics_path.is_file() and status_path and status_path.is_file()
    )
    metrics = load_json(metrics_path) if complete else {}
    status = load_json(status_path) if complete else {}
    protocol_pass = bool(
        complete and status.get("state") == "SUCCESS"
        and metrics.get("artifact_status") == "PASS"
        and all(metrics.get("artifact_checks", {}).values())
        and metrics.get("uses_test_split") is False
        and metrics.get("execution_optimization_only") is True
        and metrics.get("timing_measured_batches") == 93
    )
    equivalence_pass = bool(metrics.get("prediction_equivalence_pass", False))
    speed_gate_pass = bool(metrics.get("speed_gate_pass", False))
    decision = scientific_decision(
        complete=complete,
        protocol_pass=protocol_pass,
        equivalence_pass=equivalence_pass,
        speed_gate_pass=speed_gate_pass,
    )
    report = {
        "step": "22.24.3",
        "queue_state": state.get("queue_state"),
        "complete": complete,
        "protocol_pass": protocol_pass,
        "run_dir": str(run_dir) if run_dir else None,
        "equivalence": {
            "pass": equivalence_pass,
            "maximum_absolute_prediction_difference": metrics.get(
                "maximum_absolute_prediction_difference"
            ),
            "serial_mse": metrics.get("complete_target_mse_serial_multiscale"),
            "parallel_mse": metrics.get("complete_target_mse_parallel_multiscale"),
            "serial_mae": metrics.get("complete_target_mae_serial_multiscale"),
            "parallel_mae": metrics.get("complete_target_mae_parallel_multiscale"),
        },
        "efficiency": {
            "speed_gate_pass": speed_gate_pass,
            "serial_pipeline": metrics.get("serial_pipeline_timing"),
            "parallel_pipeline": metrics.get("parallel_pipeline_timing"),
            "parallel_over_serial_mean_ratio": metrics.get(
                "parallel_over_serial_mean_ratio"
            ),
            "paired_latency_reduction_percent_mean": metrics.get(
                "paired_latency_reduction_percent_mean"
            ),
            "paired_latency_reduction_percent_bootstrap_ci95": metrics.get(
                "paired_latency_reduction_percent_bootstrap_ci95"
            ),
            "peak_gpu_memory_mb": metrics.get("peak_gpu_memory_mb"),
            "peak_gpu_reserved_mb": metrics.get("peak_gpu_reserved_mb"),
        },
        "decision": decision,
    }
    atomic_text(REPORT_JSON, json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# 步骤22.24 seed2027批处理并行结果", "",
        f"- 决策：`{decision}`",
        f"- 工件协议：{'PASS' if protocol_pass else 'FAIL/INCOMPLETE'}",
        f"- 数值等价：{equivalence_pass}",
        f"- 速度门：{speed_gate_pass}",
    ]
    if complete:
        lines += [
            f"- 串行平均：{metrics['serial_pipeline_timing']['mean_ms']:.6f} ms/batch",
            f"- 并行平均：{metrics['parallel_pipeline_timing']['mean_ms']:.6f} ms/batch",
            f"- 平均耗时降低：{metrics['paired_latency_reduction_percent_mean']:.6f}%",
            "- 95%自助法置信区间："
            f"[{metrics['paired_latency_reduction_percent_bootstrap_ci95'][0]:.6f}%, "
            f"{metrics['paired_latency_reduction_percent_bootstrap_ci95'][1]:.6f}%]",
            f"- 峰值显存：{metrics['peak_gpu_memory_mb']:.3f} MiB",
        ]
    atomic_text(REPORT_MD, "\n".join(lines) + "\n")
    print(json.dumps({
        "decision": decision,
        "complete": complete,
        "protocol_pass": protocol_pass,
    }, indent=2, sort_keys=True))
    return 0 if not complete or protocol_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
