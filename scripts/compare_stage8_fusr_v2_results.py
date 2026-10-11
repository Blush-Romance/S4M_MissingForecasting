#!/usr/bin/env python3
"""Compare the Step-8 Weather baseline, FUSR V1, and frozen-router FUSR V2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from compare_stage7_mipp_results import (
    BASELINE_RUN,
    atomic_text,
    load_json,
    load_yaml,
    nested_differences,
    protocol_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
V1_QUEUE_STATE = ROOT / "runtime" / "stage8_fusr_queue" / "state.json"
V2_QUEUE_STATE = ROOT / "runtime" / "stage8_fusr_v2_queue" / "state.json"
REPORT_JSON = ROOT / "reports" / "stage8_fusr_v2_comparison.json"
REPORT_MD = ROOT / "reports" / "stage8_fusr_v2_comparison.md"
V1_JOB_ID = "weather_fusr_formal"
V2_JOB_ID = "weather_fusr_v2_formal"


def successful_run(queue_path: Path, job_id: str) -> Path:
    queue = load_json(queue_path)
    job = queue.get("jobs", {}).get(job_id)
    if not isinstance(job, dict):
        raise KeyError(f"missing queue job {job_id!r} in {queue_path}")
    if job.get("state") != "SUCCESS" or not job.get("run_dir"):
        raise RuntimeError(
            f"experiment artifact is incomplete: job={job_id}, "
            f"state={job.get('state')}, run_dir={job.get('run_dir')}"
        )
    run_dir = Path(job["run_dir"])
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    return run_dir


def metric_change(value: float, reference: float) -> dict[str, float]:
    return {
        "value": value,
        "absolute_change": value - reference,
        "relative_gain": (reference - value) / reference,
    }


def gate_mean(metrics: dict[str, Any]) -> float | None:
    for key in ("fusr_eval_gate_mean", "fusr_gate_mean", "fusr_gate_mean_mean"):
        value = metrics.get(key)
        if value is not None:
            return float(value)
    return None


def routing_state(metrics: dict[str, Any]) -> str:
    value = gate_mean(metrics)
    if value is None:
        return "UNKNOWN"
    if value >= 0.999:
        return "COLLAPSED_ALL_ON"
    if value <= 0.001:
        return "COLLAPSED_ALL_OFF"
    return "SELECTIVE"


def artifact_record(run_dir: Path, expected_version: str | None) -> dict[str, Any]:
    status = load_json(run_dir / "status.json")
    metrics = load_json(run_dir / "metrics.json")
    config = load_yaml(run_dir / "resolved_config.yaml")
    resource = load_json(run_dir / "resource_usage.json")
    args = config["official_args"]
    checks = {
        "run_status_success": status.get("state") == "SUCCESS",
        "checkpoint_load_verified": metrics.get("checkpoint_load_verified") is True,
        "fusr_enabled": args.get("use_fusr") is True,
        "mipp_disabled": args.get("use_mipp") is False,
    }
    if expected_version is not None:
        checks.update(
            {
                "objective_version_matches": args.get("fusr_objective_version")
                == expected_version
                and metrics.get("fusr_objective_version") == expected_version,
                "router_only_enabled": args.get("fusr_router_only") is True,
                "source_backbone_checkpoint_exists": Path(
                    str(args.get("fusr_backbone_checkpoint", ""))
                ).is_file(),
                "calibration_uses_validation_only": metrics.get(
                    "fusr_calibration_split"
                )
                == "validation",
                "test_targets_not_used_for_calibration": metrics.get(
                    "fusr_used_test_targets"
                )
                is False,
                "calibration_locked": metrics.get("fusr_calibration_state")
                == "CALIBRATION_LOCKED",
                "calibrated_threshold_recorded": metrics.get(
                    "fusr_calibrated_threshold"
                )
                is not None,
                "train_cache_nonempty": int(
                    metrics.get("fusr_cache_train_samples", 0)
                )
                > 0,
                "validation_cache_nonempty": int(
                    metrics.get("fusr_cache_validation_samples", 0)
                )
                > 0,
                "both_utility_signs_present": int(
                    metrics.get("fusr_cache_positive_samples", 0)
                )
                > 0
                and int(metrics.get("fusr_cache_negative_samples", 0)) > 0,
                "rank_pairs_generated": int(
                    metrics.get("fusr_rank_pairs_generated", 0)
                )
                > 0,
                "router_optimized": int(metrics.get("fusr_router_gradient_steps", 0))
                > 0,
                "backbone_frozen": metrics.get("fusr_backbone_frozen") is True,
                "memory_unchanged": metrics.get("fusr_memory_unchanged") is True,
            }
        )
    state = "PASS" if all(checks.values()) else "FAIL"
    return {
        "artifact_status": state,
        "artifact_checks": checks,
        "run_dir": str(run_dir),
        "status": status,
        "metrics": metrics,
        "config": config,
        "resource": resource,
    }


def compact_model_record(
    artifact: dict[str, Any], baseline_mse: float, baseline_mae: float
) -> dict[str, Any]:
    metrics = artifact["metrics"]
    args = artifact["config"]["official_args"]
    mse = float(metrics["mse"])
    mae = float(metrics["mae"])
    return {
        "artifact_status": artifact["artifact_status"],
        "artifact_checks": artifact["artifact_checks"],
        "run_dir": artifact["run_dir"],
        "mse": metric_change(mse, baseline_mse),
        "mae": metric_change(mae, baseline_mae),
        "routing_state": routing_state(metrics),
        "gate_mean": gate_mean(metrics),
        "parameter_count": metrics.get("parameter_count"),
        "parameter_delta_vs_baseline": (
            int(metrics["parameter_count"])
            if metrics.get("parameter_count") is not None
            else None
        ),
        "peak_gpu_memory_mb": metrics.get("peak_gpu_memory_mb"),
        "elapsed_seconds": artifact["resource"].get("elapsed_seconds"),
        "checkpoint_load_verified": metrics.get("checkpoint_load_verified"),
        "mechanism_config": {
            key: value
            for key, value in args.items()
            if key.startswith("fusr_")
            or key
            in {
                "use_fusr",
                "use_utility_read_gate",
                "use_fusr_spectral_uncertainty",
                "use_mipp",
            }
        },
        "mechanism_metrics": {
            key: value for key, value in metrics.items() if key.startswith("fusr_")
        },
    }


def research_decision(v2: dict[str, Any]) -> dict[str, Any]:
    mse_gain = float(v2["mse"]["relative_gain"])
    mae_gain = float(v2["mae"]["relative_gain"])
    route = str(v2["routing_state"])
    if v2["artifact_status"] != "PASS":
        code = "ARTIFACT_FAILURE_FIX_BEFORE_INTERPRETATION"
    elif route != "SELECTIVE":
        code = "NEGATIVE_ROUTING_COLLAPSE_ITERATE_WITHIN_STEP8"
    elif mse_gain > 0.0 and mae_gain > 0.0:
        code = "PROMISING_CONTINUE_TO_MULTI_SEED_AFTER_MECHANISM_AUDIT"
    elif mse_gain > 0.0:
        code = "MSE_PROMISING_MAE_MIXED_REQUIRE_ONE_MORE_SEED"
    else:
        code = "NEGATIVE_NO_MSE_LIFT_ITERATE_WITHIN_STEP8"
    return {
        "code": code,
        "primary_mse_improved": mse_gain > 0.0,
        "mae_improved": mae_gain > 0.0,
        "routing_state": route,
        "single_seed_only": True,
        "claim_boundary": (
            "A successful run is only artifact validity. This single-seed Weather screen "
            "does not establish statistical stability, cross-dataset generality, or an SCI tier."
        ),
    }


def main() -> int:
    v1_run = successful_run(V1_QUEUE_STATE, V1_JOB_ID)
    v2_run = successful_run(V2_QUEUE_STATE, V2_JOB_ID)
    baseline_metrics = load_json(BASELINE_RUN / "metrics.json")
    baseline_config = load_yaml(BASELINE_RUN / "resolved_config.yaml")
    baseline_resource = load_json(BASELINE_RUN / "resource_usage.json")
    v1_artifact = artifact_record(v1_run, expected_version=None)
    v2_artifact = artifact_record(v2_run, expected_version="v2_frozen_balanced_rank")

    baseline_mse = float(baseline_metrics["mse"])
    baseline_mae = float(baseline_metrics["mae"])
    v1 = compact_model_record(v1_artifact, baseline_mse, baseline_mae)
    v2 = compact_model_record(v2_artifact, baseline_mse, baseline_mae)
    baseline_params = int(baseline_metrics["parameter_count"])
    for record in (v1, v2):
        if record["parameter_count"] is not None:
            record["parameter_delta_vs_baseline"] = (
                int(record["parameter_count"]) - baseline_params
            )

    baseline_snapshot = protocol_snapshot(baseline_config)
    v1_differences = nested_differences(
        baseline_snapshot, protocol_snapshot(v1_artifact["config"])
    )
    v2_differences = nested_differences(
        baseline_snapshot, protocol_snapshot(v2_artifact["config"])
    )
    same_protocol = not v1_differences and not v2_differences
    all_artifacts_pass = (
        v1["artifact_status"] == "PASS" and v2["artifact_status"] == "PASS"
    )

    payload: dict[str, Any] = {
        "artifact_status": "PASS" if all_artifacts_pass else "FAIL",
        "protocol_status": "PASS" if same_protocol else "PROTOCOL_MISMATCH",
        "same_backbone_data_test_protocol": same_protocol,
        "protocol_differences": {
            "v1_vs_baseline": v1_differences,
            "v2_vs_baseline": v2_differences,
        },
        "protocol_exclusions_by_design": [
            "experiment.model/model_id/des and output paths",
            "all fusr_* router-only fields and FUSR mechanism switches",
            "runtime GPU identity and scheduler metadata",
        ],
        "dataset": "weather",
        "seed": 2023,
        "fixed_mask_asset": baseline_config["data"]["mask_asset"],
        "official_paper_mask_seed": baseline_config["data"]["official_mask_seed"],
        "exact_mask_match_to_paper_claimed": False,
        "baseline": {
            "run_dir": str(BASELINE_RUN),
            "mse": baseline_mse,
            "mae": baseline_mae,
            "parameter_count": baseline_params,
            "peak_gpu_memory_mb": baseline_metrics.get("peak_gpu_memory_mb"),
            "elapsed_seconds": baseline_resource.get("elapsed_seconds"),
        },
        "fusr_v1": v1,
        "fusr_v2": v2,
        "v2_vs_v1": {
            "mse": metric_change(float(v2["mse"]["value"]), float(v1["mse"]["value"])),
            "mae": metric_change(float(v2["mae"]["value"]), float(v1["mae"]["value"])),
        },
        "routing_state": {
            "fusr_v1": v1["routing_state"],
            "fusr_v2": v2["routing_state"],
        },
        "research_decision": research_decision(v2),
    }
    atomic_text(REPORT_JSON, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    v1_mse = float(v1["mse"]["value"])
    v1_mae = float(v1["mae"]["value"])
    v2_mse = float(v2["mse"]["value"])
    v2_mae = float(v2["mae"]["value"])
    lines = [
        "# 步骤8：Weather FUSR V2 冻结路由器单 seed 比较",
        "",
        f"- 产物状态：`{payload['artifact_status']}`；主干、数据与测试协议：`{payload['protocol_status']}`。",
        f"- 固定测试掩码：`{payload['fixed_mask_asset']}`。",
        f"- 基线 MSE（均方误差）/ MAE（平均绝对误差）：`{baseline_mse:.9f}` / `{baseline_mae:.9f}`。",
        f"- FUSR V1 MSE / MAE：`{v1_mse:.9f}` / `{v1_mae:.9f}`；路由状态：`{v1['routing_state']}`。",
        f"- FUSR V2 MSE / MAE：`{v2_mse:.9f}` / `{v2_mae:.9f}`；路由状态：`{v2['routing_state']}`。",
        f"- V2 相对基线 MSE 收益：`{float(v2['mse']['relative_gain']) * 100:+.4f}%`。",
        f"- V2 相对基线 MAE 收益：`{float(v2['mae']['relative_gain']) * 100:+.4f}%`。",
        f"- 研究判断：`{payload['research_decision']['code']}`。",
        "",
        "`artifact_status=PASS` 仅表示代码、检查点与指标产物完整，不等于科学结论为正。",
        "本报告只用于 Weather、seed=2023 的快速方向筛选；原论文 mask seed 未公开，因此不声称与原论文随机 mask 完全一致。",
    ]
    atomic_text(REPORT_MD, "\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if same_protocol and all_artifacts_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
