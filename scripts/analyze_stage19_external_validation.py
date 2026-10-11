#!/usr/bin/env python3
"""Analyze the preregistered Stage-19 ETTh1 external validation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime" / "stage19_external_validation_queue" / "state.json"
REPORT_JSON = ROOT / "reports" / "stage19_external_validation.json"
REPORT_MD = ROOT / "reports" / "stage19_external_validation.md"
FORMAL_JOB = "etth1_equal_rate_multiscale_seed2023_formal"
ROUTES = (
    "on", "l1", "l5", "l20",
    "ensemble_l1", "ensemble_l5", "ensemble_l20", "multiscale",
)


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def relative_gain(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference * 100.0


def decision(external_gate: bool, incremental_gate: bool) -> str:
    if external_gate and incremental_gate:
        return "EXTERNAL_TRANSFER_FULL_PASS"
    if external_gate:
        return "EXTERNAL_TRANSFER_PARTIAL_PASS"
    return "EXTERNAL_TRANSFER_FAIL"


def load_formal(state: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    job = state["jobs"][FORMAL_JOB]
    if job.get("state") != "SUCCESS" or not job.get("run_dir"):
        raise RuntimeError(f"{FORMAL_JOB} is not successful")
    run_dir = Path(job["run_dir"])
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return metrics, run_dir


def main() -> int:
    state = json.loads(STATE.read_text(encoding="utf-8"))
    metrics, run_dir = load_formal(state)
    routes = {
        route: {
            metric: float(metrics[f"complete_target_{metric}_{route}"])
            for metric in ("mse", "mae")
        }
        for route in ROUTES
    }
    protocol_checks = {
        "artifact_status_pass": metrics.get("artifact_status") == "PASS",
        "external_etth1_test_only": (
            metrics.get("dataset") == "etth1"
            and metrics.get("uses_test_split") is True
            and metrics.get("source_split") == "test"
        ),
        "source_is_frozen_s4m": (
            metrics.get("updates_model_or_memory") is False
            and metrics.get("method") == "cnm_se_equal_rate_multiscale_s4m"
        ),
        "seed_is_2023": int(metrics["evaluation_seed"]) == 2023,
        "complete_test_budget": (
            int(metrics["test_batches_processed"]) == 211
            and int(metrics["sample_count"]) == 3376
        ),
        "matched_historical_source": bool(
            metrics["artifact_checks"].get("matched_source_baseline")
        ),
        "equal_count_protocol": (
            float(metrics["target_additional_missing_rate"]) == 0.15
            and metrics["block_lengths"] == [1, 5, 20]
            and metrics["target_count_rounding"] == "numpy_rint_ties_to_even"
            and bool(
                metrics["artifact_checks"][
                    "strict_equal_removed_count_per_sample"
                ]
            )
        ),
        "fixed_multiscale_weights": metrics["multiscale_weights"] == {
            "original": 0.5,
            "l5": 0.25,
            "l20": 0.25,
        },
        "serial_timing_recorded": (
            metrics.get("timing_device_serial") is True
            and int(metrics["timing_measured_batches"]) == 201
        ),
    }
    rate_values = [
        float(metrics[f"realized_additional_missing_rate_l{block}"])
        for block in (1, 5, 20)
    ]
    equal_realized_rates = max(rate_values) - min(rate_values) <= 1e-12
    protocol_checks["strict_equal_realized_rate_across_scales"] = (
        equal_realized_rates
    )
    protocol_pass = all(protocol_checks.values())

    external_by_metric = {
        metric: routes["multiscale"][metric] < routes["on"][metric]
        for metric in ("mse", "mae")
    }
    incremental_by_metric = {
        metric: (
            routes["multiscale"][metric] < routes["ensemble_l5"][metric]
        )
        for metric in ("mse", "mae")
    }
    external_gate = all(external_by_metric.values())
    incremental_gate = all(incremental_by_metric.values())
    scientific_decision = decision(external_gate, incremental_gate)
    gains = {
        "multiscale_vs_original_pct": {
            metric: relative_gain(
                routes["on"][metric],
                routes["multiscale"][metric],
            )
            for metric in ("mse", "mae")
        },
        "multiscale_vs_l5_pair_pct": {
            metric: relative_gain(
                routes["ensemble_l5"][metric],
                routes["multiscale"][metric],
            )
            for metric in ("mse", "mae")
        },
    }
    latency = {
        route: float(metrics["estimated_serial_latency"][route]["mean_ms"])
        for route in (
            "original", "ensemble_l1", "ensemble_l5",
            "ensemble_l20", "multiscale",
        )
    }
    payload = {
        "status": "PASS" if protocol_pass else "FAIL",
        "scope": (
            "single-seed ETTh1 external test of a Weather-validation-frozen "
            "CNM-SE inference protocol on the original S4M checkpoint"
        ),
        "protocol_limitations": [
            "source checkpoint is original S4M, not PG95",
            "official ETTh1 mask seed is unavailable",
            "single seed is directional evidence, not statistical stability",
            "historical ETTh1 reproduction differs materially from the paper value",
        ],
        "pre_registered_design": {
            "dataset": "etth1",
            "seed": 2023,
            "split": "test",
            "target_rate": 0.15,
            "block_lengths": [1, 5, 20],
            "multiscale_weights": {
                "original": 0.5,
                "l5": 0.25,
                "l20": 0.25,
            },
            "external_gate": (
                "multiscale MSE and MAE both strictly beat original"
            ),
            "incremental_gate": (
                "multiscale MSE and MAE both strictly beat the L5 equal pair"
            ),
        },
        "routes": routes,
        "gains": gains,
        "realized_additional_missing_rates": {
            f"l{block}": rate
            for block, rate in zip((1, 5, 20), rate_values)
        },
        "serial_latency_mean_ms": latency,
        "multiscale_multiplier_vs_original": float(
            metrics["estimated_serial_latency"]["multiscale"][
                "measured_multiplier_vs_original"
            ]
        ),
        "peak_gpu_memory_mb": metrics.get("peak_gpu_memory_mb"),
        "parameter_count": metrics.get("parameter_count"),
        "run_dir": str(run_dir),
        "protocol_checks": protocol_checks,
        "external_gate_by_metric": external_by_metric,
        "incremental_gate_by_metric": incremental_by_metric,
        "external_gate_pass": external_gate,
        "incremental_gate_pass": incremental_gate,
        "decision": scientific_decision,
    }
    atomic_text(REPORT_JSON, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    route_names = {
        "on": "原始S4M",
        "l1": "L1单路",
        "l5": "L5单路",
        "l20": "L20单路",
        "ensemble_l1": "原始+L1等权",
        "ensemble_l5": "原始+L5等权",
        "ensemble_l20": "原始+L20等权",
        "multiscale": "原始/L5/L20多尺度",
    }
    lines = [
        "# 步骤19：ETTh1冻结多尺度CNM-SE外部验证",
        "",
        "| 路由 | MSE | MAE |",
        "|---|---:|---:|",
    ]
    for route in ROUTES:
        record = routes[route]
        lines.append(
            f"| {route_names[route]} | {record['mse']:.9f} | "
            f"{record['mae']:.9f} |"
        )
    lines.extend([
        "",
        f"- 工件与协议检查：`{'PASS' if protocol_pass else 'FAIL'}`",
        f"- 相对原始S4M双指标门：`{external_gate}`",
        f"- 相对L5双视图增量门：`{incremental_gate}`",
        f"- 决策：`{scientific_decision}`",
        f"- 正式run：`{run_dir}`",
        "",
        "限制：这是ETTh1单seed、近似固定mask上的方向性外部证据；"
        "来源模型为原始S4M而非PG95，不能据此声称PG95跨数据集有效。",
        "",
    ])
    atomic_text(REPORT_MD, "\n".join(lines))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if protocol_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
