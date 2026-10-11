#!/usr/bin/env python3
"""Aggregate the frozen Step-22.32 HFD-PBA development screen."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime/stage22_32_hfd_pba/state.json"
REPORT_JSON = ROOT / "reports/stage22_32_hfd_pba_seed2027.json"
REPORT_MD = ROOT / "reports/stage22_32_hfd_pba_seed2027.md"
SMOKE_JOB = "hfd_pba_traffic_validation_seed2027_smoke"
FORMAL_JOB = "hfd_pba_traffic_validation_seed2027_formal"
PG95_RUN = ROOT / "runs/mipp_pg95_traffic__traffic__point__mr0.06__seed2027__20260820_174327"
CNM_SE_RUN = ROOT / "runs/cnm_se_equal_rate_multiscale_pg95__traffic__validation__seed2027__20260820_184433"
BF_CMVD_V2_RUN = ROOT / "runs/bf_cmvd_v2_pg95_traffic__traffic__point__mr0.06__seed2027__20260820_212927"
PBA_RUN = ROOT / "runs/pba_cmvd_pg95_traffic__traffic__point__mr0.06__seed2027__20260826_222702"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def job_metrics(job: dict) -> tuple[dict, Path | None]:
    if job.get("state") != "SUCCESS" or not job.get("run_dir"):
        return {}, None
    run_dir = Path(job["run_dir"]).resolve()
    metrics_path = run_dir / "metrics.json"
    return (load_json(metrics_path), run_dir) if metrics_path.is_file() else ({}, run_dir)


def relative_gain(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def main() -> int:
    if not STATE.is_file():
        return 0
    state = load_json(STATE)
    smoke, smoke_run = job_metrics(state["jobs"][SMOKE_JOB])
    formal, formal_run = job_metrics(state["jobs"][FORMAL_JOB])
    decision = "HFD_PBA_SMOKE_INCOMPLETE"
    comparison: dict[str, Any] = {}
    costs: dict[str, Any] = {}
    if smoke:
        decision = "HFD_PBA_FORMAL_INCOMPLETE"
    if state["jobs"][FORMAL_JOB].get("state") == "BLOCKED_DEPENDENCY":
        decision = "HFD_PBA_SMOKE_MECHANISM_FAIL"
    if formal:
        pg95 = load_json(PG95_RUN / "metrics.json")
        teacher = load_json(CNM_SE_RUN / "metrics.json")
        bf = load_json(BF_CMVD_V2_RUN / "metrics.json")
        pba = load_json(PBA_RUN / "metrics.json")
        pg95_mse, pg95_mae = float(pg95["mse"]), float(pg95["mae"])
        teacher_mse = float(teacher["complete_target_mse_multiscale"])
        teacher_mae = float(teacher["complete_target_mae_multiscale"])
        bf_mse, bf_mae = float(bf["mse"]), float(bf["mae"])
        pba_mse, pba_mae = float(pba["mse"]), float(pba["mae"])
        hfd_mse, hfd_mae = float(formal["mse"]), float(formal["mae"])
        teacher_gain_mse = relative_gain(pg95_mse, teacher_mse)
        teacher_gain_mae = relative_gain(pg95_mae, teacher_mae)
        hfd_gain_mse = relative_gain(pg95_mse, hfd_mse)
        hfd_gain_mae = relative_gain(pg95_mae, hfd_mae)
        retention_mse = hfd_gain_mse / teacher_gain_mse
        retention_mae = hfd_gain_mae / teacher_gain_mae
        pba_groups = formal.get("pba_trainable_group_gradient_norm_max", {})
        hfd_gradients = formal.get("hfd_layer_feature_gradient_norm_max", {})
        protocol_pass = bool(
            formal.get("test_split_accessed") is False
            and int(formal.get("validation_batches_processed", -1)) == 103
            and int(formal.get("validation_samples_processed", -1)) == 1648
            and formal.get("pba_frozen_hash_before") == formal.get("pba_frozen_hash_after")
            and formal.get("pba_memory_hash_before") == formal.get("pba_memory_hash_after")
            and formal.get("pba_teacher_hash_before") == formal.get("pba_teacher_hash_after")
            and formal.get("pba_teacher_memory_hash_before") == formal.get("pba_teacher_memory_hash_after")
            and formal.get("pba_trainable_hash_before") != formal.get("pba_trainable_hash_after")
            and int(formal.get("pba_trainable_parameter_count", -1)) == 4127582
            and int(formal.get("pba_inference_student_forward_count", -1)) == 1
            and int(formal.get("pba_validation_teacher_forward_count", -1)) == 0
            and formal.get("checkpoint_format") == "pba_delta_v1"
            and formal.get("hfd_pba_enabled") is True
            and formal.get("hfd_layers") == [6, 7]
            and formal.get("hfd_validation_feature_capture_enabled") is False
            and formal.get("hfd_teacher_feature_requires_grad") is False
            and int(formal.get("hfd_active_steps", 0)) > 0
            and set(pba_groups) == {"s4_layers_6", "s4_layers_7", "norm", "out_proj"}
            and all(float(value) > 0.0 for value in pba_groups.values())
            and set(hfd_gradients) == {"6", "7"}
            and all(float(value) > 0.0 for value in hfd_gradients.values())
        )
        accuracy_gate = bool(
            hfd_mse < bf_mse
            and hfd_mae < bf_mae
            and hfd_mse < pba_mse
            and hfd_mae < pba_mae
            and retention_mse >= 0.60
            and retention_mae >= 0.60
        )
        gate_pass = protocol_pass and accuracy_gate
        comparison = {
            "pg95_validation": {"mse": pg95_mse, "mae": pg95_mae},
            "bf_cmvd_v2_validation": {"mse": bf_mse, "mae": bf_mae},
            "pba_cmvd_validation": {"mse": pba_mse, "mae": pba_mae},
            "cnm_se_teacher_validation": {"mse": teacher_mse, "mae": teacher_mae},
            "hfd_pba_validation": {"mse": hfd_mse, "mae": hfd_mae},
            "hfd_gain_vs_pg95_percent": {"mse": hfd_gain_mse, "mae": hfd_gain_mae},
            "hfd_gain_vs_bf_percent": {
                "mse": relative_gain(bf_mse, hfd_mse),
                "mae": relative_gain(bf_mae, hfd_mae),
            },
            "hfd_gain_vs_pba_percent": {
                "mse": relative_gain(pba_mse, hfd_mse),
                "mae": relative_gain(pba_mae, hfd_mae),
            },
            "teacher_gain_retention_fraction": {
                "mse": retention_mse, "mae": retention_mae,
            },
            "protocol_gate_pass": protocol_pass,
            "accuracy_gate_pass": accuracy_gate,
            "validation_development_gate_pass": gate_pass,
        }
        resources = load_json(formal_run / "resource_usage.json")
        costs = {
            "trainable_parameters": int(formal.get("pba_trainable_parameter_count", -1)),
            "total_deployment_parameters": int(formal.get("parameter_count", -1)),
            "deployment_student_forward_count": int(formal.get("pba_inference_student_forward_count", -1)),
            "training_teacher_forward_count": int(formal.get("pba_teacher_training_forward_count", -1)),
            "elapsed_seconds": resources.get("elapsed_seconds"),
            "peak_gpu_memory_mb": resources.get("peak_gpu_memory_mb"),
            "checkpoint_size_bytes": formal.get("checkpoint_size_bytes"),
            "validation_elapsed_seconds": formal.get("validation_elapsed_seconds"),
            "validation_ms_per_batch": formal.get("validation_ms_per_batch"),
        }
        decision = (
            "HFD_PBA_SEED2028_VALIDATION_CONFIRMATION_AUTHORIZED"
            if gate_pass else "HFD_PBA_SEED2027_STOP"
        )
    report = {
        "step": "22.32",
        "queue_state": state.get("queue_state"),
        "smoke_run": str(smoke_run) if smoke_run else None,
        "formal_run": str(formal_run) if formal_run else None,
        "scientific_dependency": state["jobs"][FORMAL_JOB].get("scientific_dependency"),
        "comparison": comparison,
        "costs": costs,
        "test_sources_accessed": False,
        "seed2027_is_development_not_confirmation": True,
        "decision": decision,
    }
    atomic_text(REPORT_JSON, json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Step 22.32 HFD-PBA 结果", "",
        f"- 决策：`{decision}`",
        f"- 队列：`{state.get('queue_state')}`",
        f"- test 来源被访问：`{report['test_sources_accessed']}`",
        "- seed2027 定位：开发筛选，非独立确认",
    ]
    if comparison:
        lines += [
            f"- HFD-PBA MSE/MAE：{comparison['hfd_pba_validation']['mse']:.9f} / {comparison['hfd_pba_validation']['mae']:.9f}",
            f"- 相对 BF-CMVD-v2 改善：{comparison['hfd_gain_vs_bf_percent']['mse']:.6f}% / {comparison['hfd_gain_vs_bf_percent']['mae']:.6f}%",
            f"- 相对原 PBA 改善：{comparison['hfd_gain_vs_pba_percent']['mse']:.6f}% / {comparison['hfd_gain_vs_pba_percent']['mae']:.6f}%",
            f"- 教师收益保留率：{comparison['teacher_gain_retention_fraction']['mse']:.6f} / {comparison['teacher_gain_retention_fraction']['mae']:.6f}",
            f"- 开发门：`{comparison['validation_development_gate_pass']}`",
        ]
    atomic_text(REPORT_MD, "\n".join(lines) + "\n")
    if state.get("queue_state") in {"SUCCESS", "FAILED", "STOPPED"}:
        import aggregate_results
        aggregate_results.main()
    print(json.dumps({"decision": decision, "queue_state": state.get("queue_state")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
