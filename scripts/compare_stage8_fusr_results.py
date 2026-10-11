#!/usr/bin/env python3
"""Compare the Step-8 Weather FUSR formal run with the frozen baseline."""

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
    training_trace,
)


ROOT = Path(__file__).resolve().parents[1]
QUEUE_STATE = ROOT / "runtime" / "stage8_fusr_queue" / "state.json"
REPORT_JSON = ROOT / "reports" / "stage8_fusr_comparison.json"
REPORT_MD = ROOT / "reports" / "stage8_fusr_comparison.md"
FORMAL_JOB_ID = "weather_fusr_formal"


def require_successful_run() -> Path:
    queue = load_json(QUEUE_STATE)
    job = queue.get("jobs", {}).get(FORMAL_JOB_ID)
    if not isinstance(job, dict):
        raise KeyError(f"missing queue job: {FORMAL_JOB_ID}")
    if job.get("state") != "SUCCESS" or not job.get("run_dir"):
        raise RuntimeError(
            f"FUSR formal result is not successful: state={job.get('state')}, "
            f"run_dir={job.get('run_dir')}"
        )
    return Path(job["run_dir"])


def metric_change(candidate: float, baseline: float) -> dict[str, float]:
    return {
        "value": candidate,
        "absolute_change_vs_baseline": candidate - baseline,
        "relative_gain_vs_baseline": (baseline - candidate) / baseline,
    }


def main() -> int:
    fusr_run = require_successful_run()
    baseline_metrics = load_json(BASELINE_RUN / "metrics.json")
    baseline_config = load_yaml(BASELINE_RUN / "resolved_config.yaml")
    fusr_metrics = load_json(fusr_run / "metrics.json")
    fusr_config = load_yaml(fusr_run / "resolved_config.yaml")
    fusr_status = load_json(fusr_run / "status.json")
    fusr_resource = load_json(fusr_run / "resource_usage.json")
    baseline_resource = load_json(BASELINE_RUN / "resource_usage.json")

    if fusr_status.get("state") != "SUCCESS":
        raise RuntimeError(f"FUSR run status is not SUCCESS: {fusr_status.get('state')}")
    if fusr_metrics.get("checkpoint_load_verified") is not True:
        raise RuntimeError("FUSR checkpoint reload was not verified")
    if fusr_config["official_args"].get("use_fusr") is not True:
        raise RuntimeError("formal FUSR config does not enable use_fusr")
    if fusr_config["official_args"].get("use_mipp") is not False:
        raise RuntimeError("independent FUSR screen must keep use_mipp=false")

    differences = nested_differences(
        protocol_snapshot(baseline_config), protocol_snapshot(fusr_config)
    )
    same_protocol = not differences
    baseline_mse = float(baseline_metrics["mse"])
    baseline_mae = float(baseline_metrics["mae"])
    fusr_mse = float(fusr_metrics["mse"])
    fusr_mae = float(fusr_metrics["mae"])
    trace = training_trace(fusr_run, fusr_metrics)
    trace["configured_max_epochs"] = int(fusr_config["official_args"]["train_epochs"])
    trace["configured_patience"] = int(fusr_config["official_args"]["patience"])

    fusr_args = fusr_config["official_args"]
    mechanism_config = {
        key: value
        for key, value in fusr_args.items()
        if key.startswith("fusr_") or key in {"use_fusr", "use_utility_read_gate", "use_fusr_spectral_uncertainty", "use_mipp"}
    }
    payload: dict[str, Any] = {
        "status": "PASS" if same_protocol else "PROTOCOL_MISMATCH",
        "same_protocol": same_protocol,
        "protocol_differences": differences,
        "baseline_run": str(BASELINE_RUN),
        "fusr_run": str(fusr_run),
        "dataset": "weather",
        "seed": int(fusr_config["experiment"]["seed"]),
        "fixed_mask_asset": fusr_config["data"]["mask_asset"],
        "project_internal_fixed_mask_comparison": (
            baseline_config["data"]["mask_asset"] == fusr_config["data"]["mask_asset"]
        ),
        "official_paper_mask_seed": fusr_config["data"]["official_mask_seed"],
        "exact_mask_match_to_paper_claimed": False,
        "baseline": {
            "mse": baseline_mse,
            "mae": baseline_mae,
            "parameter_count": int(baseline_metrics["parameter_count"]),
            "peak_gpu_memory_mb": baseline_metrics.get("peak_gpu_memory_mb"),
            "elapsed_seconds": baseline_resource.get("elapsed_seconds"),
            "actual_optimizer_steps": int(baseline_metrics["optimizer_steps"]),
        },
        "fusr": {
            "mse": metric_change(fusr_mse, baseline_mse),
            "mae": metric_change(fusr_mae, baseline_mae),
            "parameter_count": int(fusr_metrics["parameter_count"]),
            "parameter_delta": int(fusr_metrics["parameter_count"])
            - int(baseline_metrics["parameter_count"]),
            "peak_gpu_memory_mb": fusr_metrics.get("peak_gpu_memory_mb"),
            "peak_gpu_reserved_mb": fusr_metrics.get("peak_gpu_reserved_mb"),
            "elapsed_seconds": fusr_resource.get("elapsed_seconds"),
            "training_trace": trace,
            "mechanism_config": mechanism_config,
            "mechanism_metrics": {
                key: value for key, value in fusr_metrics.items() if key.startswith("fusr_")
            },
            "checkpoint_load_verified": True,
        },
        "directional_decision": {
            "improves_both_mse_and_mae": fusr_mse < baseline_mse and fusr_mae < baseline_mae,
            "single_seed_only": True,
            "note": (
                "This is an independent FUSR read-gate screen with use_mipp=false; "
                "multi-seed and cross-dataset evidence remain required."
            ),
        },
    }
    atomic_text(REPORT_JSON, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# 步骤8：Weather FUSR 独立单 seed 比较",
        "",
        f"- 协议状态：`{payload['status']}`；协议差异数：`{len(differences)}`。",
        f"- 固定 mask：`{payload['fixed_mask_asset']}`。",
        f"- 基线 MSE / MAE：`{baseline_mse:.9f}` / `{baseline_mae:.9f}`。",
        f"- FUSR MSE / MAE：`{fusr_mse:.9f}` / `{fusr_mae:.9f}`。",
        f"- MSE 相对收益：`{payload['fusr']['mse']['relative_gain_vs_baseline'] * 100:+.4f}%`。",
        f"- MAE 相对收益：`{payload['fusr']['mae']['relative_gain_vs_baseline'] * 100:+.4f}%`。",
        f"- 参数增量：`{payload['fusr']['parameter_delta']:+d}`；峰值显存："
        f"`{payload['fusr']['peak_gpu_memory_mb']}` MB。",
        f"- 实际优化器 steps / early stopping（早停）epoch："
        f"`{trace['actual_optimizer_steps']}` / `{trace['actual_epochs_completed']}`。",
        f"- 同时改善 MSE 与 MAE：`{payload['directional_decision']['improves_both_mse_and_mae']}`。",
        "",
        "项目内基线与 FUSR 沿用同一固定 mask，但原论文 mask seed 未公开，因此不声称与原论文具有完全相同的随机 mask。",
        "本报告仅用于 Weather、seed=2023 的方向筛选，不构成统计稳定性或 SCI 分区结论。",
    ]
    atomic_text(REPORT_MD, "\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if same_protocol else 2


if __name__ == "__main__":
    raise SystemExit(main())
