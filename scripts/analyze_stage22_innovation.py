#!/usr/bin/env python3
"""Audit Step-22 MRRA/RADR validation results against the frozen PG95 anchor."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
ANCHOR_STATE = ROOT / "runtime" / "stage21_electricity_pipeline" / "state.json"
STATE = ROOT / "runtime" / "stage22_innovation" / "state.json"
MRRA_V3_STATE = ROOT / "runtime" / "stage22_mrra_v3" / "state.json"
REPORT_JSON = ROOT / "reports" / "stage22_innovation_gate.json"
REPORT_MD = ROOT / "reports" / "stage22_innovation_gate.md"
ANCHOR_JOB = "electricity_pg95_seed2023_formal"
CANDIDATE_SOURCES = {
    "mrra_v3": (MRRA_V3_STATE, "mrra_v3_pg95_electricity_seed2023_formal"),
    "radr": (STATE, "radr_pg95_electricity_seed2023_formal"),
}
CANDIDATE_JOBS = {
    name: job_id for name, (_, job_id) in CANDIDATE_SOURCES.items()
}
RUN_IDENTITY_FIELDS = {
    "model_id", "des", "checkpoints", "metrics_path",
    "results_root", "test_results_root",
}
INNOVATION_FIELDS = {
    "use_mrra", "mrra_version", "mrra_rank", "mrra_kernel_size",
    "mrra_reconstruction_weight", "mrra_lr_multiplier", "use_radr",
}


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def gain_percent(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference * 100.0


def normalized_training_args(arguments: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(arguments)
    defaults = {
        "loss": "MSE",
        "use_mrra": False,
        "mrra_version": "v1",
        "mrra_rank": 32,
        "mrra_kernel_size": 5,
        "mrra_reconstruction_weight": 0.1,
        "mrra_lr_multiplier": 1.0,
        "use_radr": False,
    }
    for key, value in defaults.items():
        normalized.setdefault(key, value)
    for key in RUN_IDENTITY_FIELDS | INNOVATION_FIELDS:
        normalized.pop(key, None)
    return normalized


def candidate_gate(anchor: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    comparisons = {
        metric: float(candidate[metric]) <= float(anchor[metric])
        for metric in ("mse", "mae")
    }
    strict = any(
        float(candidate[metric]) < float(anchor[metric])
        for metric in ("mse", "mae")
    )
    return {
        "mse_non_degrade": comparisons["mse"],
        "mae_non_degrade": comparisons["mae"],
        "has_strict_improvement": strict,
        "pass": all(comparisons.values()) and strict,
        "mse_gain_percent": gain_percent(
            float(anchor["mse"]), float(candidate["mse"])
        ),
        "mae_gain_percent": gain_percent(
            float(anchor["mae"]), float(candidate["mae"])
        ),
    }


def load_run(
    state: dict[str, Any], job_id: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    job = state.get("jobs", {}).get(job_id)
    if not job or job.get("state") != "SUCCESS" or not job.get("run_dir"):
        raise RuntimeError(f"{job_id} is not successful")
    run_dir = Path(job["run_dir"]).resolve()
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    config = yaml.safe_load(
        (run_dir / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    if status.get("state") != "SUCCESS":
        raise RuntimeError(f"{job_id} run status is not SUCCESS")
    return metrics, status, config, run_dir


def main() -> int:
    anchor_state = json.loads(ANCHOR_STATE.read_text(encoding="utf-8"))
    anchor, anchor_status, anchor_config, anchor_run = load_run(
        anchor_state, ANCHOR_JOB
    )
    candidates = {}
    for name, (state_path, job_id) in CANDIDATE_SOURCES.items():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        candidates[name] = load_run(state, job_id)
    protocol_checks: dict[str, bool] = {
        "anchor_success": anchor_status.get("state") == "SUCCESS",
        "anchor_validation_only": (
            anchor.get("evaluation_split") == "validation"
            and anchor.get("test_split_accessed") is False
        ),
        "anchor_complete_validation": (
            int(anchor.get("validation_batches_processed", 0)) == 158
            and int(anchor.get("validation_samples_processed", 0)) == 2528
        ),
    }
    result_candidates: dict[str, Any] = {}
    anchor_args = anchor_config["official_args"]
    for name, (metrics, status, config, run_dir) in candidates.items():
        args = config["official_args"]
        mechanism_pass = False
        if name.startswith("mrra"):
            mechanism_pass = (
                metrics.get("mrra_enabled") is True
                and metrics.get("mrra_version")
                == "v3_gradient_isolated_two_timescale"
                and int(metrics.get("mrra_steps", 0)) > 0
                and float(metrics.get("mrra_reconstruction_loss_mean", 0.0)) > 0.0
                and float(metrics.get("mrra_context_residual_energy_mean", 0.0)) > 0.0
                and len(metrics.get("checkpoint_mrra_state_keys", [])) > 0
                and metrics.get("mrra_gradient_isolated") is True
                and metrics.get("mrra_hidden_normalization_affine") is False
                and int(metrics.get("mrra_optimizer_group_count", 0)) == 2
                and float(metrics.get("mrra_learning_rate_multiplier", -1.0))
                == 0.1
                and int(metrics.get("mrra_diagnostic_steps", 0)) > 0
                and float(metrics.get("mrra_gradient_l2_max", 0.0)) > 0.0
            )
        else:
            mechanism_pass = (
                metrics.get("radr_enabled") is True
                and int(metrics.get("radr_steps", 0)) > 0
                and float(metrics.get("radr_difficulty_rank_std_mean", 0.0)) > 0.0
                and float(metrics.get("radr_hard_body_mse_gap_mean", -1.0)) >= 0.0
                and int(metrics.get("parameter_count", -1))
                == int(anchor.get("parameter_count", -2))
            )
        checks = {
            "run_success": status.get("state") == "SUCCESS",
            "same_fixed_data": config["data"] == anchor_config["data"],
            "same_training_budget": (
                normalized_training_args(args)
                == normalized_training_args(anchor_args)
            ),
            "full_validation_budget": (
                int(metrics.get("validation_batches_processed", 0)) == 158
                and int(metrics.get("validation_samples_processed", 0)) == 2528
            ),
            "validation_only": (
                metrics.get("evaluation_split") == "validation"
                and metrics.get("post_train_evaluation") == "validation"
                and metrics.get("test_split_accessed") is False
                and metrics.get("chronological_loader") is True
            ),
            "checkpoint_reload_verified": (
                metrics.get("checkpoint_load_verified") is True
                and metrics.get("checkpoint_all_model_tensors_finite") is True
            ),
            "mechanism_executed": mechanism_pass,
        }
        protocol_checks.update({f"{name}_{key}": value for key, value in checks.items()})
        gate = candidate_gate(anchor, metrics)
        result_candidates[name] = {
            "run_dir": str(run_dir),
            "checkpoint": metrics.get("checkpoint"),
            "mse": float(metrics["mse"]),
            "mae": float(metrics["mae"]),
            "parameter_count": int(metrics["parameter_count"]),
            "peak_gpu_memory_mb": metrics.get("peak_gpu_memory_mb"),
            "protocol_checks": checks,
            "gate": gate,
        }

    protocol_pass = all(protocol_checks.values())
    passed = [
        name for name, result in result_candidates.items()
        if result["gate"]["pass"]
    ]
    if not protocol_pass:
        decision = "PROTOCOL_FAIL_NO_SCIENTIFIC_DECISION"
    elif len(passed) == 2:
        decision = "BOTH_PASS_VALIDATE_COMBINATION_NEXT"
    elif passed:
        decision = f"{passed[0].upper()}_PASS_EXPAND_SINGLE_MODULE"
    else:
        decision = "BOTH_FAIL_ITERATE_NEW_HYPOTHESIS"

    payload = {
        "step": 22,
        "status": "PASS" if protocol_pass else "FAIL",
        "scope": "Electricity seed2023 validation-only rapid innovation screen",
        "decision": decision,
        "test_accessed": False,
        "gate_definition": (
            "Relative to the frozen PG95 anchor, validation MSE and MAE must "
            "both be non-degrading and at least one must be strictly lower; "
            "no numerical gain threshold."
        ),
        "anchor": {
            "run_dir": str(anchor_run),
            "mse": float(anchor["mse"]),
            "mae": float(anchor["mae"]),
            "parameter_count": int(anchor["parameter_count"]),
        },
        "candidates": result_candidates,
        "passed_candidates": passed,
        "protocol_pass": protocol_pass,
        "protocol_checks": protocol_checks,
        "limitations": [
            "single seed is directional evidence, not statistical stability",
            "Electricity official missing-mask seed and selected layer count are unpublished",
            "no Electricity test split is accessed or authorized by this report",
        ],
    }
    atomic_text(REPORT_JSON, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# 步骤22创新验证门", "", f"- 协议状态：`{payload['status']}`",
        f"- 决策：`{decision}`", "- 测试集访问：`false`", "",
        "| 候选 | MSE | MAE | MSE变化 | MAE变化 | 单模块门 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, result in result_candidates.items():
        gate = result["gate"]
        lines.append(
            f"| {name.upper()} | {result['mse']:.9f} | {result['mae']:.9f} | "
            f"{gate['mse_gain_percent']:+.4f}% | {gate['mae_gain_percent']:+.4f}% | "
            f"{'PASS' if gate['pass'] else 'FAIL'} |"
        )
    lines.extend(["", "基准为步骤21冻结的PG95 Electricity seed2023完整验证结果。", ""])
    atomic_text(REPORT_MD, "\n".join(lines))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if protocol_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
