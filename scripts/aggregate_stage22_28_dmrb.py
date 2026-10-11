#!/usr/bin/env python3
"""Aggregate the frozen Step-22.28 DMRB development screen."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime/stage22_28_dmrb/state.json"
REPORT_JSON = ROOT / "reports/stage22_28_dmrb_seed2027.json"
REPORT_MD = ROOT / "reports/stage22_28_dmrb_seed2027.md"
SMOKE_JOB = "dmrb_traffic_validation_seed2027_smoke"
FORMAL_JOB = "dmrb_traffic_validation_seed2027_formal"
PG95_RUN = ROOT / "runs/mipp_pg95_traffic__traffic__point__mr0.06__seed2027__20260820_174327"
CNM_SE_RUN = ROOT / "runs/cnm_se_equal_rate_multiscale_pg95__traffic__validation__seed2027__20260820_184433"
BF_CMVD_V2_RUN = ROOT / "runs/bf_cmvd_v2_pg95_traffic__traffic__point__mr0.06__seed2027__20260820_212927"


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
    path = run_dir / "metrics.json"
    return (load_json(path), run_dir) if path.is_file() else ({}, run_dir)


def relative_gain(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def main() -> int:
    if not STATE.is_file():
        return 0
    state = load_json(STATE)
    smoke, smoke_run = job_metrics(state["jobs"][SMOKE_JOB])
    formal, formal_run = job_metrics(state["jobs"][FORMAL_JOB])
    decision = "DMRB_SMOKE_INCOMPLETE"
    comparison: dict[str, Any] = {}
    costs: dict[str, Any] = {}
    if smoke:
        decision = "DMRB_FORMAL_INCOMPLETE"
    if state["jobs"][FORMAL_JOB].get("state") == "BLOCKED_DEPENDENCY":
        decision = "DMRB_SMOKE_MECHANISM_FAIL"
    if formal:
        pg95 = load_json(PG95_RUN / "metrics.json")
        teacher = load_json(CNM_SE_RUN / "metrics.json")
        v2 = load_json(BF_CMVD_V2_RUN / "metrics.json")
        pg95_mse, pg95_mae = float(pg95["mse"]), float(pg95["mae"])
        teacher_mse = float(teacher["complete_target_mse_multiscale"])
        teacher_mae = float(teacher["complete_target_mae_multiscale"])
        v2_mse, v2_mae = float(v2["mse"]), float(v2["mae"])
        dmrb_mse, dmrb_mae = float(formal["mse"]), float(formal["mae"])
        teacher_gain_mse = relative_gain(pg95_mse, teacher_mse)
        teacher_gain_mae = relative_gain(pg95_mae, teacher_mae)
        dmrb_gain_mse = relative_gain(pg95_mse, dmrb_mse)
        dmrb_gain_mae = relative_gain(pg95_mae, dmrb_mae)
        retention_mse = dmrb_gain_mse / teacher_gain_mse
        retention_mae = dmrb_gain_mae / teacher_gain_mae
        protocol_pass = bool(
            formal.get("test_split_accessed") is False
            and formal.get("mcpr_backbone_hash_before")
            == formal.get("mcpr_backbone_hash_after")
            and formal.get("mcpr_memory_hash_before")
            == formal.get("mcpr_memory_hash_after")
            and formal.get("checkpoint_format") == "mcpr_delta_v1"
            and formal.get("checkpoint_dmrb_branches_verified") is True
            and int(formal.get("mcpr_trainable_parameter_count", -1)) == 90364
            and int(formal.get("dmrb_inference_backbone_forward_count", -1)) == 1
        )
        accuracy_gate = bool(
            dmrb_mse < v2_mse
            and dmrb_mae < v2_mae
            and retention_mse >= 0.60
            and retention_mae >= 0.60
        )
        gate_pass = protocol_pass and accuracy_gate
        comparison = {
            "pg95_validation": {"mse": pg95_mse, "mae": pg95_mae},
            "cnm_se_teacher_validation": {"mse": teacher_mse, "mae": teacher_mae},
            "bf_cmvd_v2_validation": {"mse": v2_mse, "mae": v2_mae},
            "dmrb_validation": {"mse": dmrb_mse, "mae": dmrb_mae},
            "dmrb_gain_vs_pg95_percent": {"mse": dmrb_gain_mse, "mae": dmrb_gain_mae},
            "dmrb_gain_vs_bf_cmvd_v2_percent": {
                "mse": relative_gain(v2_mse, dmrb_mse),
                "mae": relative_gain(v2_mae, dmrb_mae),
            },
            "teacher_gain_vs_pg95_percent": {
                "mse": teacher_gain_mse, "mae": teacher_gain_mae,
            },
            "teacher_gain_retention_fraction": {
                "mse": retention_mse, "mae": retention_mae,
            },
            "protocol_gate_pass": protocol_pass,
            "accuracy_gate_pass": accuracy_gate,
            "validation_development_gate_pass": gate_pass,
        }
        dmrb_resources = load_json(formal_run / "resource_usage.json")
        v2_resources = load_json(BF_CMVD_V2_RUN / "resource_usage.json")
        costs = {
            "bf_cmvd_v2_trainable_parameters": int(
                v2.get("mcpr_trainable_parameter_count", -1)
            ),
            "dmrb_trainable_parameters": int(
                formal.get("mcpr_trainable_parameter_count", -1)
            ),
            "parameter_multiplier_vs_v2": (
                int(formal.get("mcpr_trainable_parameter_count", -1))
                / int(v2.get("mcpr_trainable_parameter_count", 1))
            ),
            "dmrb_elapsed_seconds": dmrb_resources.get("elapsed_seconds"),
            "bf_cmvd_v2_elapsed_seconds": v2_resources.get("elapsed_seconds"),
            "dmrb_peak_gpu_memory_mb": dmrb_resources.get("peak_gpu_memory_mb"),
            "bf_cmvd_v2_peak_gpu_memory_mb": v2_resources.get("peak_gpu_memory_mb"),
            "dmrb_checkpoint_size_bytes": formal.get("checkpoint_size_bytes"),
            "inference_backbone_forward_count": formal.get(
                "dmrb_inference_backbone_forward_count"
            ),
        }
        decision = (
            "DMRB_SEED2028_VALIDATION_CONFIRMATION_AUTHORIZED"
            if gate_pass else "DMRB_SEED2027_STOP_DIAGNOSE"
        )
    report = {
        "step": "22.28",
        "queue_state": state.get("queue_state"),
        "smoke_run": str(smoke_run) if smoke_run else None,
        "formal_run": str(formal_run) if formal_run else None,
        "scientific_dependency": state["jobs"][FORMAL_JOB].get(
            "scientific_dependency"
        ),
        "smoke_mechanism": {
            key: smoke.get(key) for key in (
                "dmrb_l5_distill_l1_mean", "dmrb_l20_distill_l1_mean",
                "dmrb_l5_residual_energy_mean", "dmrb_l20_residual_energy_mean",
                "dmrb_l5_gradient_norm_max", "dmrb_l20_gradient_norm_max",
                "checkpoint_format", "checkpoint_size_bytes",
            )
        } if smoke else {},
        "formal_mechanism": {
            key: formal.get(key) for key in (
                "bf_cmvd_beneficial_fraction", "dmrb_l5_distill_l1_mean",
                "dmrb_l20_distill_l1_mean", "dmrb_l5_residual_energy_mean",
                "dmrb_l20_residual_energy_mean", "dmrb_l5_gradient_norm_max",
                "dmrb_l20_gradient_norm_max",
            )
        } if formal else {},
        "comparison": comparison,
        "costs": costs,
        "test_sources_accessed": False,
        "seed2027_is_development_not_confirmation": True,
        "decision": decision,
    }
    atomic_text(REPORT_JSON, json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Step 22.28 DMRB 结果", "",
        f"- 决策：`{decision}`",
        f"- 队列：`{state.get('queue_state')}`",
        f"- test 来源被访问：`{report['test_sources_accessed']}`",
        f"- seed2027 定位：开发筛选，非独立确认",
    ]
    if comparison:
        lines += [
            f"- DMRB MSE/MAE：{comparison['dmrb_validation']['mse']:.9f} / {comparison['dmrb_validation']['mae']:.9f}",
            f"- 相对 BF-CMVD-v2 MSE/MAE 改善：{comparison['dmrb_gain_vs_bf_cmvd_v2_percent']['mse']:.6f}% / {comparison['dmrb_gain_vs_bf_cmvd_v2_percent']['mae']:.6f}%",
            f"- 教师收益保留率 MSE/MAE：{comparison['teacher_gain_retention_fraction']['mse']:.6f} / {comparison['teacher_gain_retention_fraction']['mae']:.6f}",
            f"- 60% 开发门：`{comparison['validation_development_gate_pass']}`",
        ]
    atomic_text(REPORT_MD, "\n".join(lines) + "\n")
    if state.get("queue_state") in {"SUCCESS", "FAILED", "STOPPED"}:
        import aggregate_results
        aggregate_results.main()
    print(json.dumps({"decision": decision, "queue_state": state.get("queue_state")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
