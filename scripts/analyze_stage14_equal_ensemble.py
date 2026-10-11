#!/usr/bin/env python3
"""Frozen dual-seed gate for the fixed Stage-14 equal ensemble."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime" / "stage14_equal_ensemble_queue" / "state.json"
IDENTITY = ROOT / "reports" / "stage14_equal_ensemble_identity.json"
REPORT_JSON = ROOT / "reports" / "stage14_equal_ensemble_analysis.json"
REPORT_MD = ROOT / "reports" / "stage14_equal_ensemble_analysis.md"
MSE_GAIN_MIN_PCT = 1.5
MAE_GAIN_MIN_PCT = 0.5
IDENTITY_ATOL = 1e-5


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def evaluate_metrics(
    metrics: dict[str, Any], identity_mse: float
) -> dict[str, Any]:
    mse_on = float(metrics["complete_target_mse_on"])
    mae_on = float(metrics["complete_target_mae_on"])
    mse_nested = float(metrics["complete_target_mse_nested"])
    mae_nested = float(metrics["complete_target_mae_nested"])
    mse_ensemble = float(metrics["complete_target_mse_ensemble"])
    mae_ensemble = float(metrics["complete_target_mae_ensemble"])
    mse_gain = (mse_on - mse_ensemble) / mse_on * 100.0
    mae_gain = (mae_on - mae_ensemble) / mae_on * 100.0
    checks = {
        "artifact_status_pass": metrics.get("artifact_status") == "PASS",
        "validation_only": (
            metrics.get("uses_test_split") is False
            and metrics.get("source_split") == "val"
        ),
        "frozen_checkpoint": metrics.get("updates_model_or_memory") is False,
        "fixed_equal_weights": (
            float(metrics.get("ensemble_weight_on", -1.0)) == 0.5
            and float(metrics.get("ensemble_weight_nested", -1.0)) == 0.5
        ),
        "identity_mse_matches_prior_cache": (
            abs(mse_ensemble - float(identity_mse)) <= IDENTITY_ATOL
        ),
        "mse_gain_at_least_1_5_pct": mse_gain >= MSE_GAIN_MIN_PCT,
        "mae_gain_at_least_0_5_pct": mae_gain >= MAE_GAIN_MIN_PCT,
    }
    return {
        "mse_on": mse_on,
        "mae_on": mae_on,
        "mse_nested": mse_nested,
        "mae_nested": mae_nested,
        "mse_equal_ensemble": mse_ensemble,
        "mae_equal_ensemble": mae_ensemble,
        "mse_gain_vs_on_pct": mse_gain,
        "mae_gain_vs_on_pct": mae_gain,
        "mse_gain_vs_nested_pct": (mse_nested - mse_ensemble) / mse_nested * 100.0,
        "mae_gain_vs_nested_pct": (mae_nested - mae_ensemble) / mae_nested * 100.0,
        "checks": checks,
        "gate_pass": bool(all(checks.values())),
    }


def main() -> int:
    state = json.loads(STATE.read_text(encoding="utf-8"))
    identity = json.loads(IDENTITY.read_text(encoding="utf-8"))
    results = {}
    runs = {}
    for seed in (2024, 2025):
        job_name = f"weather_pg95_seed{seed}_equal_ensemble_formal"
        job = state["jobs"][job_name]
        if job.get("state") != "SUCCESS" or not job.get("run_dir"):
            raise RuntimeError(f"{job_name} is not successful")
        run_dir = Path(job["run_dir"])
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        results[str(seed)] = evaluate_metrics(
            metrics,
            identity["results"][str(seed)]["mse_equal_ensemble"],
        )
        runs[str(seed)] = str(run_dir)
    gate_pass = all(result["gate_pass"] for result in results.values())
    decision = (
        "STAGE14_EQUAL_ENSEMBLE_GATE_PASS_IMPLEMENT_TEST_MODULE"
        if gate_pass
        else "STAGE14_EQUAL_ENSEMBLE_GATE_FAILED_DO_NOT_TEST"
    )
    payload = {
        "status": "PASS" if gate_pass else "FAIL",
        "scope": "Weather PG95 frozen checkpoints; validation only; no test samples",
        "pre_analysis_gate": {
            "all_seeds_required": [2024, 2025],
            "mse_gain_vs_pg95_each_seed_min_pct": MSE_GAIN_MIN_PCT,
            "mae_gain_vs_pg95_each_seed_min_pct": MAE_GAIN_MIN_PCT,
            "equal_weights": [0.5, 0.5],
            "identity_mse_atol": IDENTITY_ATOL,
        },
        "results": results,
        "runs": runs,
        "decision": decision,
    }
    atomic_text(REPORT_JSON, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# 步骤14.5：固定等权双视图validation分析",
        "",
        "- 范围：Weather PG95冻结checkpoint；仅validation；未读取test。",
        "- 固定权重：原始视图0.5，嵌套缺失视图0.5。",
        f"- 门槛：每个seed MSE改善≥{MSE_GAIN_MIN_PCT}%且MAE改善≥{MAE_GAIN_MIN_PCT}%。",
        "",
        "| seed | on MSE | ensemble MSE | MSE gain | on MAE | ensemble MAE | MAE gain | PASS |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for seed in (2024, 2025):
        result = results[str(seed)]
        lines.append(
            f"| {seed} | {result['mse_on']:.9f} | "
            f"{result['mse_equal_ensemble']:.9f} | "
            f"{result['mse_gain_vs_on_pct']:.4f}% | "
            f"{result['mae_on']:.9f} | {result['mae_equal_ensemble']:.9f} | "
            f"{result['mae_gain_vs_on_pct']:.4f}% | {result['gate_pass']} |"
        )
    lines.extend(["", f"- 决策：`{decision}`", ""])
    atomic_text(REPORT_MD, "\n".join(lines))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
