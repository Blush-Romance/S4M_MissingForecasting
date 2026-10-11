#!/usr/bin/env python3
"""Frozen cross-seed accuracy gate for Stage-15 CNM-SE."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime" / "stage15_cnm_se_queue" / "state.json"
REPORT_JSON = ROOT / "reports" / "stage15_cnm_se_analysis.json"
REPORT_MD = ROOT / "reports" / "stage15_cnm_se_analysis.md"
MEAN_MSE_GAIN_MIN_PCT = 1.0
MEAN_MAE_GAIN_MIN_PCT = 0.5


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def evaluate_seed(metrics: dict[str, Any]) -> dict[str, Any]:
    mse_on = float(metrics["complete_target_mse_on"])
    mae_on = float(metrics["complete_target_mae_on"])
    mse_ensemble = float(metrics["complete_target_mse_ensemble"])
    mae_ensemble = float(metrics["complete_target_mae_ensemble"])
    mse_gain = (mse_on - mse_ensemble) / mse_on * 100.0
    mae_gain = (mae_on - mae_ensemble) / mae_on * 100.0
    checks = {
        "artifact_status_pass": metrics.get("artifact_status") == "PASS",
        "test_split": (
            metrics.get("uses_test_split") is True
            and metrics.get("source_split") == "test"
        ),
        "frozen_checkpoint": metrics.get("updates_model_or_memory") is False,
        "fixed_equal_weights": (
            float(metrics.get("ensemble_weight_on", -1.0)) == 0.5
            and float(metrics.get("ensemble_weight_nested", -1.0)) == 0.5
        ),
        "matched_on_route_mse_improves": mse_gain > 0.0,
        "matched_on_route_mae_improves": mae_gain > 0.0,
    }
    return {
        "mse_on": mse_on,
        "mae_on": mae_on,
        "mse_nested": float(metrics["complete_target_mse_nested"]),
        "mae_nested": float(metrics["complete_target_mae_nested"]),
        "mse_cnm_se": mse_ensemble,
        "mae_cnm_se": mae_ensemble,
        "mse_gain_pct": mse_gain,
        "mae_gain_pct": mae_gain,
        "evaluation_elapsed_seconds": float(metrics["evaluation_elapsed_seconds"]),
        "inference_forward_multiplier": float(metrics["inference_forward_multiplier"]),
        "peak_gpu_memory_mb": metrics.get("peak_gpu_memory_mb"),
        "checks": checks,
        "seed_pass": bool(all(checks.values())),
    }


def decide(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    mean_mse_gain = sum(value["mse_gain_pct"] for value in results.values()) / len(results)
    mean_mae_gain = sum(value["mae_gain_pct"] for value in results.values()) / len(results)
    checks = {
        "both_seeds_improve_both_metrics": all(
            value["seed_pass"] for value in results.values()
        ),
        "mean_mse_gain_at_least_1_pct": mean_mse_gain >= MEAN_MSE_GAIN_MIN_PCT,
        "mean_mae_gain_at_least_0_5_pct": mean_mae_gain >= MEAN_MAE_GAIN_MIN_PCT,
    }
    return {
        "mean_mse_gain_pct": mean_mse_gain,
        "mean_mae_gain_pct": mean_mae_gain,
        "checks": checks,
        "gate_pass": bool(all(checks.values())),
    }


def main() -> int:
    state = json.loads(STATE.read_text(encoding="utf-8"))
    results = {}
    runs = {}
    for seed in (2024, 2025):
        job_name = f"weather_cnm_se_seed{seed}_formal"
        job = state["jobs"][job_name]
        if job.get("state") != "SUCCESS" or not job.get("run_dir"):
            raise RuntimeError(f"{job_name} is not successful")
        run_dir = Path(job["run_dir"])
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        results[str(seed)] = evaluate_seed(metrics)
        runs[str(seed)] = str(run_dir)
    summary = decide(results)
    decision = (
        "CNM_SE_TWO_SEED_TEST_GATE_PASS_EXPAND"
        if summary["gate_pass"]
        else "CNM_SE_TWO_SEED_TEST_GATE_FAILED_RETURN_TO_VALIDATION"
    )
    payload = {
        "status": "PASS" if summary["gate_pass"] else "FAIL",
        "scope": "Weather fixed point-missing test assets; frozen PG95 checkpoints",
        "pre_registered_gate": {
            "each_seed_mse_and_mae_must_improve": True,
            "mean_mse_gain_min_pct": MEAN_MSE_GAIN_MIN_PCT,
            "mean_mae_gain_min_pct": MEAN_MAE_GAIN_MIN_PCT,
            "equal_weights": [0.5, 0.5],
        },
        "results": results,
        "summary": summary,
        "runs": runs,
        "decision": decision,
    }
    atomic_text(REPORT_JSON, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# 步骤15：CNM-SE双seed正式测试",
        "",
        "| seed | PG95 MSE | CNM-SE MSE | MSE gain | PG95 MAE | CNM-SE MAE | MAE gain | PASS |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for seed in (2024, 2025):
        result = results[str(seed)]
        lines.append(
            f"| {seed} | {result['mse_on']:.9f} | {result['mse_cnm_se']:.9f} | "
            f"{result['mse_gain_pct']:.4f}% | {result['mae_on']:.9f} | "
            f"{result['mae_cnm_se']:.9f} | {result['mae_gain_pct']:.4f}% | "
            f"{result['seed_pass']} |"
        )
    lines.extend([
        "",
        f"- 两seed平均MSE改善：`{summary['mean_mse_gain_pct']:.4f}%`",
        f"- 两seed平均MAE改善：`{summary['mean_mae_gain_pct']:.4f}%`",
        f"- 决策：`{decision}`",
        "",
    ])
    atomic_text(REPORT_MD, "\n".join(lines))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if summary["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
