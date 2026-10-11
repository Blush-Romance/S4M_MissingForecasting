#!/usr/bin/env python3
"""Analyze equal-rate block scales, fixed multiscale fusion, and serial latency."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime" / "stage18_equal_rate_multiscale_queue" / "state.json"
REPORT_JSON = ROOT / "reports" / "stage18_equal_rate_multiscale_analysis.json"
REPORT_MD = ROOT / "reports" / "stage18_equal_rate_multiscale_analysis.md"
SEEDS = (2024, 2025)
ROUTES = (
    "on", "l1", "l5", "l20",
    "ensemble_l1", "ensemble_l5", "ensemble_l20", "multiscale",
)


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def load_formal(state: dict[str, Any], seed: int) -> dict[str, Any]:
    job_id = f"weather_equal_rate_multiscale_seed{seed}_formal"
    job = state["jobs"][job_id]
    if job.get("state") != "SUCCESS" or not job.get("run_dir"):
        raise RuntimeError(f"{job_id} is not successful")
    run_dir = Path(job["run_dir"])
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    checks = {
        "artifact_status_pass": metrics.get("artifact_status") == "PASS",
        "validation_only": (
            metrics.get("uses_test_split") is False
            and metrics.get("source_split") == "val"
        ),
        "frozen_checkpoint": metrics.get("updates_model_or_memory") is False,
        "seed_matches": int(metrics["evaluation_seed"]) == seed,
        "equal_rate_protocol": (
            float(metrics["target_additional_missing_rate"]) == 0.15
            and metrics["block_lengths"] == [1, 5, 20]
            and metrics["target_count_rounding"] == "numpy_rint_ties_to_even"
            and metrics["artifact_checks"][
                "strict_equal_removed_count_per_sample"
            ]
        ),
        "fixed_multiscale_weights": metrics["multiscale_weights"] == {
            "original": 0.5, "l5": 0.25, "l20": 0.25
        },
        "serial_timing": (
            metrics.get("timing_device_serial") is True
            and metrics.get("timing_fixed_forward_order")
            == ["original", "l1", "l5", "l20"]
            and int(metrics["timing_measured_batches"]) > 0
        ),
    }
    routes = {
        route: {
            metric: float(metrics[f"complete_target_{metric}_{route}"])
            for metric in ("mse", "mae")
        }
        for route in ROUTES
    }
    latency = {
        route: float(metrics["estimated_serial_latency"][route]["mean_ms"])
        for route in ("original", "ensemble_l1", "ensemble_l5", "ensemble_l20", "multiscale")
    }
    return {
        "seed": seed,
        "routes": routes,
        "realized_rates": {
            str(block): float(
                metrics[f"realized_additional_missing_rate_l{block}"]
            )
            for block in (1, 5, 20)
        },
        "latency_mean_ms": latency,
        "multiscale_measured_multiplier_vs_original": float(
            metrics["estimated_serial_latency"]["multiscale"][
                "measured_multiplier_vs_original"
            ]
        ),
        "gpu_uuid": metrics.get("gpu_uuid"),
        "timing_measured_batches": int(metrics["timing_measured_batches"]),
        "sample_count": int(metrics["sample_count"]),
        "run_dir": str(run_dir),
        "checks": checks,
        "protocol_pass": all(checks.values()),
    }


def relative_gain(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference * 100.0


def main() -> int:
    state = json.loads(STATE.read_text(encoding="utf-8"))
    results = {str(seed): load_formal(state, seed) for seed in SEEDS}
    protocol_pass = all(result["protocol_pass"] for result in results.values())
    gpu_uuids = {result["gpu_uuid"] for result in results.values()}
    same_gpu = len(gpu_uuids) == 1 and None not in gpu_uuids
    equal_rates = all(
        max(result["realized_rates"].values())
        - min(result["realized_rates"].values()) <= 1e-12
        for result in results.values()
    )
    multiscale_beats_l5 = {
        str(seed): {
            metric: (
                results[str(seed)]["routes"]["multiscale"][metric]
                < results[str(seed)]["routes"]["ensemble_l5"][metric]
            )
            for metric in ("mse", "mae")
        }
        for seed in SEEDS
    }
    validation_gate_pass = all(
        multiscale_beats_l5[str(seed)][metric]
        for seed in SEEDS
        for metric in ("mse", "mae")
    )
    means = {
        route: {
            metric: mean(
                [results[str(seed)]["routes"][route][metric] for seed in SEEDS]
            )
            for metric in ("mse", "mae")
        }
        for route in ROUTES
    }
    gains = {
        str(seed): {
            metric: relative_gain(
                results[str(seed)]["routes"]["ensemble_l5"][metric],
                results[str(seed)]["routes"]["multiscale"][metric],
            )
            for metric in ("mse", "mae")
        }
        for seed in SEEDS
    }
    latency_means = {
        route: mean(
            [results[str(seed)]["latency_mean_ms"][route] for seed in SEEDS]
        )
        for route in (
            "original", "ensemble_l1", "ensemble_l5", "ensemble_l20", "multiscale"
        )
    }
    decision = (
        "MULTISCALE_VALIDATION_GATE_PASS_EXPAND_ONLY_TO_NEW_PROTOCOLS"
        if protocol_pass and same_gpu and equal_rates and validation_gate_pass
        else "MULTISCALE_VALIDATION_GATE_FAILED_KEEP_FROZEN_L5"
    )
    payload = {
        "status": (
            "PASS" if protocol_pass and same_gpu and equal_rates else "FAIL"
        ),
        "scope": (
            "Weather validation only; frozen PG95 checkpoints; exact per-sample "
            "additional removal counts; same-GPU serial timing"
        ),
        "pre_registered_design": {
            "seeds": list(SEEDS),
            "block_lengths": [1, 5, 20],
            "target_rate": 0.15,
            "pair_weights": [0.5, 0.5],
            "multiscale_weights": {
                "original": 0.5, "l5": 0.25, "l20": 0.25
            },
            "gate": (
                "multiscale MSE and MAE must both beat L5 equal ensemble "
                "independently on seed2024 and seed2025"
            ),
        },
        "results": results,
        "two_seed_means": means,
        "multiscale_gain_vs_l5_pair_pct": gains,
        "serial_latency_two_seed_mean_ms": latency_means,
        "checks": {
            "all_artifacts_and_protocols_pass": protocol_pass,
            "both_formals_measured_on_same_gpu_uuid": same_gpu,
            "strict_equal_realized_rate_across_scales": equal_rates,
            "multiscale_beats_l5_by_seed_and_metric": multiscale_beats_l5,
            "validation_gate_pass": validation_gate_pass,
            "weather_test_not_revisited": True,
        },
        "decision": decision,
    }
    atomic_text(REPORT_JSON, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# 步骤18：等实际缺失率复核与多尺度CNM-SE验证",
        "",
        "本报告仅使用validation（验证集）；没有再次读取Weather test来选择结构或权重。",
        "",
        "| seed | 路由 | MSE | MAE |",
        "|---:|---|---:|---:|",
    ]
    route_names = {
        "on": "original",
        "l1": "L1单路",
        "l5": "L5单路",
        "l20": "L20单路",
        "ensemble_l1": "original+L1等权",
        "ensemble_l5": "original+L5等权",
        "ensemble_l20": "original+L20等权",
        "multiscale": "original/L5/L20多尺度",
    }
    for seed in SEEDS:
        for route in ROUTES:
            record = results[str(seed)]["routes"][route]
            lines.append(
                f"| {seed} | {route_names[route]} | "
                f"{record['mse']:.9f} | {record['mae']:.9f} |"
            )
    lines.extend([
        "",
        "| 路由 | 同GPU串行批延迟均值（ms） |",
        "|---|---:|",
    ])
    for route, latency in latency_means.items():
        lines.append(f"| {route_names.get(route, route)} | {latency:.4f} |")
    lines.extend([
        "",
        f"- 两个formal是否在同一GPU UUID计时：`{same_gpu}`（`{sorted(gpu_uuids)}`）",
        f"- 三种块长逐样本实际移除数/缺失率是否严格相同：`{equal_rates}`",
        f"- 多尺度是否在两个seed的MSE与MAE都优于L5等权：`{validation_gate_pass}`",
        f"- 决策：`{decision}`",
        "",
    ])
    atomic_text(REPORT_MD, "\n".join(lines))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
