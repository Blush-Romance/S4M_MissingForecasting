#!/usr/bin/env python3
"""Analyze the pre-registered CNM-SE block and fusion structure ablations."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MAIN_STATE = ROOT / "runtime" / "stage15_cnm_se_queue" / "state.json"
ABLATION_STATE = ROOT / "runtime" / "stage16_cnm_se_ablation_queue" / "state.json"
REPORT_JSON = ROOT / "reports" / "stage16_cnm_se_ablation_analysis.json"
REPORT_MD = ROOT / "reports" / "stage16_cnm_se_ablation_analysis.md"
SEEDS = (2023, 2024, 2025)
BLOCK_LENGTHS = (1, 5, 20)
ROUTES = ("on", "nested", "ensemble")


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def successful_run(state: dict[str, Any], job_id: str) -> Path:
    job = state["jobs"][job_id]
    if job.get("state") != "SUCCESS" or not job.get("run_dir"):
        raise RuntimeError(f"{job_id} is not successful")
    return Path(job["run_dir"])


def metric_record(run_dir: Path, block_length: int, seed: int) -> dict[str, Any]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    checks = {
        "artifact_status_pass": metrics.get("artifact_status") == "PASS",
        "test_split": (
            metrics.get("uses_test_split") is True
            and metrics.get("source_split") == "test"
        ),
        "frozen_checkpoint": metrics.get("updates_model_or_memory") is False,
        "seed_matches": int(metrics["evaluation_seed"]) == seed,
        "block_length_matches": int(metrics["nested_block_length"]) == block_length,
        "fixed_target_rate": float(metrics["nested_target_rate"]) == 0.15,
        "fixed_equal_weights": (
            float(metrics["ensemble_weight_on"]) == 0.5
            and float(metrics["ensemble_weight_nested"]) == 0.5
        ),
        "two_forward_routes": float(metrics["inference_forward_multiplier"]) == 2.0,
    }
    routes = {
        route: {
            "mse": float(metrics[f"complete_target_mse_{route}"]),
            "mae": float(metrics[f"complete_target_mae_{route}"]),
        }
        for route in ROUTES
    }
    return {
        "seed": seed,
        "block_length": block_length,
        "routes": routes,
        "realized_additional_missing_rate": float(
            metrics["realized_additional_missing_rate"]
        ),
        "evaluation_elapsed_seconds": float(metrics["evaluation_elapsed_seconds"]),
        "peak_gpu_memory_mb": metrics.get("peak_gpu_memory_mb"),
        "parameter_count": int(metrics["parameter_count"]),
        "checks": checks,
        "protocol_pass": bool(all(checks.values())),
        "run_dir": str(run_dir),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> int:
    main_state = json.loads(MAIN_STATE.read_text(encoding="utf-8"))
    ablation_state = json.loads(ABLATION_STATE.read_text(encoding="utf-8"))
    results: dict[str, dict[str, Any]] = {}
    for block_length in BLOCK_LENGTHS:
        block_results: dict[str, Any] = {}
        for seed in SEEDS:
            if block_length == 5:
                job_id = f"weather_cnm_se_seed{seed}_formal"
                run_dir = successful_run(main_state, job_id)
            else:
                job_id = f"weather_cnm_se_l{block_length}_seed{seed}_formal"
                run_dir = successful_run(ablation_state, job_id)
            block_results[str(seed)] = metric_record(run_dir, block_length, seed)
        results[str(block_length)] = block_results

    means: dict[str, Any] = {}
    for block_length in BLOCK_LENGTHS:
        block = results[str(block_length)]
        route_means = {
            route: {
                metric: mean(
                    [block[str(seed)]["routes"][route][metric] for seed in SEEDS]
                )
                for metric in ("mse", "mae")
            }
            for route in ROUTES
        }
        on = route_means["on"]
        ensemble = route_means["ensemble"]
        means[str(block_length)] = {
            "routes": route_means,
            "ensemble_mse_gain_vs_on_pct": (
                (on["mse"] - ensemble["mse"]) / on["mse"] * 100.0
            ),
            "ensemble_mae_gain_vs_on_pct": (
                (on["mae"] - ensemble["mae"]) / on["mae"] * 100.0
            ),
            "realized_additional_missing_rate": mean(
                [
                    block[str(seed)]["realized_additional_missing_rate"]
                    for seed in SEEDS
                ]
            ),
            "evaluation_elapsed_seconds": mean(
                [block[str(seed)]["evaluation_elapsed_seconds"] for seed in SEEDS]
            ),
        }

    l5_fusion_beats_both = all(
        results["5"][str(seed)]["routes"]["ensemble"][metric]
        < min(
            results["5"][str(seed)]["routes"]["on"][metric],
            results["5"][str(seed)]["routes"]["nested"][metric],
        )
        for seed in SEEDS
        for metric in ("mse", "mae")
    )
    l5_best_mean_mse = means["5"]["routes"]["ensemble"]["mse"] == min(
        means[str(block)]["routes"]["ensemble"]["mse"] for block in BLOCK_LENGTHS
    )
    l5_best_mean_mae = means["5"]["routes"]["ensemble"]["mae"] == min(
        means[str(block)]["routes"]["ensemble"]["mae"] for block in BLOCK_LENGTHS
    )
    protocol_pass = all(
        results[str(block)][str(seed)]["protocol_pass"]
        for block in BLOCK_LENGTHS
        for seed in SEEDS
    )
    conclusions = {
        "all_artifacts_and_protocols_pass": protocol_pass,
        "l5_equal_ensemble_beats_both_single_routes_for_every_seed_and_metric": (
            l5_fusion_beats_both
        ),
        "l5_has_best_three_seed_mean_ensemble_mse": l5_best_mean_mse,
        "l5_has_best_three_seed_mean_ensemble_mae": l5_best_mean_mae,
        "non_nested_add_back_ablation_omitted_as_input_leakage": True,
        "no_test_based_block_length_reselection": True,
    }
    if l5_fusion_beats_both and l5_best_mean_mse and l5_best_mean_mae:
        decision = "CNM_SE_FUSION_AND_CAUSAL_L5_STRUCTURE_SUPPORTED"
    elif l5_fusion_beats_both:
        decision = "CNM_SE_FUSION_SUPPORTED_BLOCK_LENGTH_SENSITIVITY_MIXED"
    else:
        decision = "CNM_SE_STRUCTURE_ABLATION_MIXED_REVIEW_MECHANISM"
    payload = {
        "status": "PASS" if protocol_pass else "FAIL",
        "scope": (
            "Weather fixed test assets; frozen PG95 checkpoints; "
            "pre-registered L1/L5/L20 nested-mask block and route fusion ablations"
        ),
        "pre_registered_design": {
            "block_lengths": list(BLOCK_LENGTHS),
            "seeds": list(SEEDS),
            "nested_target_rate": 0.15,
            "ensemble_weights": [0.5, 0.5],
            "routes": list(ROUTES),
            "main_configuration_remains_block_length": 5,
        },
        "results": results,
        "means": means,
        "conclusions": conclusions,
        "decision": decision,
    }
    atomic_text(REPORT_JSON, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# 步骤16：CNM-SE结构消融",
        "",
        "L=5是validation阶段冻结的主模型；L=1与L=20仅用于结构解释，不依据test结果重选主配置。",
        "",
        "| 块长 | 路由 | 三seed均值 MSE | 三seed均值 MAE |",
        "|---:|---|---:|---:|",
    ]
    route_names = {
        "on": "原始路（original-only）",
        "nested": "嵌套路（nested-only）",
        "ensemble": "等权集成（equal-ensemble）",
    }
    for block_length in BLOCK_LENGTHS:
        for route in ROUTES:
            route_mean = means[str(block_length)]["routes"][route]
            lines.append(
                f"| {block_length} | {route_names[route]} | "
                f"{route_mean['mse']:.9f} | {route_mean['mae']:.9f} |"
            )
    lines.extend([
        "",
        "| 块长 | 集成MSE相对原始路 | 集成MAE相对原始路 | 实际新增缺失率 |",
        "|---:|---:|---:|---:|",
    ])
    for block_length in BLOCK_LENGTHS:
        summary = means[str(block_length)]
        lines.append(
            f"| {block_length} | {summary['ensemble_mse_gain_vs_on_pct']:.4f}% | "
            f"{summary['ensemble_mae_gain_vs_on_pct']:.4f}% | "
            f"{summary['realized_additional_missing_rate']:.6f} |"
        )
    lines.extend([
        "",
        f"- L=5等权集成是否在每个seed、每个指标上都优于两条单路：`{l5_fusion_beats_both}`",
        f"- L=5是否取得三seed最优集成均值MSE/MAE：`{l5_best_mean_mse}/{l5_best_mean_mae}`",
        "- 非嵌套“加回”原始缺失真值会造成输入泄漏，因此不作为合法消融。",
        f"- 决策：`{decision}`",
        "",
    ])
    atomic_text(REPORT_MD, "\n".join(lines))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if protocol_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
