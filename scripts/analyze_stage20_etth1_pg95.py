#!/usr/bin/env python3
"""Analyze the frozen Stage-20 ETTh1 PG95 plus multiscale experiment."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime" / "stage20_etth1_pg95_pipeline" / "state.json"
BASELINE_RUN = (
    ROOT / "runs"
    / "s4m_paper_adam_reprofix__etth1__point__mr0.06__seed2023__20260717_195944"
)
REPORT_JSON = ROOT / "reports" / "stage20_etth1_pg95_full_method.json"
REPORT_MD = ROOT / "reports" / "stage20_etth1_pg95_full_method.md"
TRAIN_JOB = "etth1_pg95_seed2023_formal"
EVAL_JOB = "etth1_pg95_multiscale_seed2023_formal"
ROUTES = (
    "on", "l1", "l5", "l20",
    "ensemble_l1", "ensemble_l5", "ensemble_l20", "multiscale",
)


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def load_successful_run(
    state: dict[str, Any], job_id: str
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    job = state["jobs"][job_id]
    if job.get("state") != "SUCCESS" or not job.get("run_dir"):
        raise RuntimeError(f"{job_id} is not successful")
    run_dir = Path(job["run_dir"])
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if status.get("state") != "SUCCESS":
        raise RuntimeError(f"{job_id} run status is not SUCCESS")
    return metrics, status, run_dir


def relative_gain(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference * 100.0


def decision(
    pg95_gate: bool,
    cnm_se_gate: bool,
    incremental_gate: bool,
) -> str:
    if pg95_gate and cnm_se_gate and incremental_gate:
        return "FULL_METHOD_DIRECTIONAL_PASS"
    if pg95_gate and cnm_se_gate:
        return "MULTISCALE_INCREMENT_FAIL"
    if pg95_gate:
        return "PG95_ONLY_PASS"
    if cnm_se_gate:
        return "CNM_SE_ON_PG95_ONLY_PASS"
    return "FULL_METHOD_FAIL"


def main() -> int:
    state = json.loads(STATE.read_text(encoding="utf-8"))
    baseline = json.loads(
        (BASELINE_RUN / "metrics.json").read_text(encoding="utf-8")
    )
    training, training_status, training_run = load_successful_run(
        state, TRAIN_JOB
    )
    evaluation, evaluation_status, evaluation_run = load_successful_run(
        state, EVAL_JOB
    )
    training_config = yaml.safe_load(
        (training_run / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    official = training_config["official_args"]
    routes = {
        route: {
            metric: float(evaluation[f"complete_target_{metric}_{route}"])
            for metric in ("mse", "mae")
        }
        for route in ROUTES
    }
    training_checkpoint = Path(training["checkpoint"]).resolve()
    evaluation_checkpoint = Path(evaluation["source_checkpoint"]).resolve()

    protocol_checks = {
        "queue_success": state.get("queue_state") == "SUCCESS",
        "training_artifact_success": (
            training_status.get("state") == "SUCCESS"
            and training.get("checkpoint_load_verified") is True
        ),
        "training_is_frozen_etth1_pg95": (
            training_config["experiment"]["dataset"] == "etth1"
            and int(training_config["experiment"]["seed"]) == 2023
            and official.get("use_mipp") is True
            and official.get("use_content_mask_factorization") is False
            and official.get("use_mask_intervention_pair") is True
            and official.get("use_stability_write_gate") is True
            and official.get("mipp_write_gate_mode") == "absolute"
            and float(official.get("mipp_inv_weight")) == 0.05
            and official.get("use_fusr") is False
            and official.get("use_cfr") is False
        ),
        "full_training_budget": (
            int(official["batch_size"]) == 16
            and int(official["train_epochs"]) == 15
            and int(official["max_train_batches"]) == -1
            and int(official["max_eval_batches"]) == -1
            and official["optimizer"] == "adam"
            and float(official["learning_rate"]) == 0.005
            and int(official["patience"]) == 3
        ),
        "mipp_checkpoint_verified": (
            bool(training.get("checkpoint_mipp_state_keys"))
            and training.get("mipp_enabled") is True
        ),
        "evaluation_artifact_pass": (
            evaluation_status.get("state") == "SUCCESS"
            and evaluation.get("artifact_status") == "PASS"
        ),
        "evaluation_is_etth1_pg95_test": (
            evaluation.get("dataset") == "etth1"
            and evaluation.get("method")
            == "cnm_se_equal_rate_multiscale_pg95"
            and evaluation.get("uses_test_split") is True
            and evaluation.get("source_split") == "test"
        ),
        "exact_training_checkpoint_handoff": (
            training_checkpoint == evaluation_checkpoint
        ),
        "matched_training_source_metric": bool(
            evaluation["artifact_checks"].get("matched_source_baseline")
        ),
        "complete_test_budget": (
            int(evaluation["test_batches_processed"]) == 211
            and int(evaluation["sample_count"]) == 3376
        ),
        "equal_count_protocol": (
            float(evaluation["target_additional_missing_rate"]) == 0.15
            and evaluation["block_lengths"] == [1, 5, 20]
            and evaluation["target_count_rounding"] == "numpy_rint_ties_to_even"
            and bool(
                evaluation["artifact_checks"][
                    "strict_equal_removed_count_per_sample"
                ]
            )
        ),
        "fixed_multiscale_weights": evaluation["multiscale_weights"] == {
            "original": 0.5,
            "l5": 0.25,
            "l20": 0.25,
        },
        "frozen_evaluation": (
            evaluation.get("updates_model_or_memory") is False
        ),
        "serial_timing_recorded": (
            evaluation.get("timing_device_serial") is True
            and int(evaluation["timing_measured_batches"]) == 201
        ),
    }
    realized_rates = [
        float(evaluation[f"realized_additional_missing_rate_l{block}"])
        for block in (1, 5, 20)
    ]
    protocol_checks["equal_realized_rate_across_scales"] = (
        max(realized_rates) - min(realized_rates) <= 1e-12
    )
    protocol_pass = all(protocol_checks.values())

    pg95_by_metric = {
        metric: float(training[metric]) < float(baseline[metric])
        for metric in ("mse", "mae")
    }
    cnm_se_by_metric = {
        metric: routes["multiscale"][metric] < routes["on"][metric]
        for metric in ("mse", "mae")
    }
    incremental_by_metric = {
        metric: (
            routes["multiscale"][metric]
            < routes["ensemble_l5"][metric]
        )
        for metric in ("mse", "mae")
    }
    pg95_gate = all(pg95_by_metric.values())
    cnm_se_gate = all(cnm_se_by_metric.values())
    incremental_gate = all(incremental_by_metric.values())
    scientific_decision = decision(
        pg95_gate, cnm_se_gate, incremental_gate
    )

    gains = {
        "pg95_vs_s4m_pct": {
            metric: relative_gain(
                float(baseline[metric]), float(training[metric])
            )
            for metric in ("mse", "mae")
        },
        "multiscale_vs_pg95_original_pct": {
            metric: relative_gain(
                routes["on"][metric], routes["multiscale"][metric]
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
        "full_method_vs_s4m_pct": {
            metric: relative_gain(
                float(baseline[metric]), routes["multiscale"][metric]
            )
            for metric in ("mse", "mae")
        },
    }
    latency = {
        route: float(
            evaluation["estimated_serial_latency"][route]["mean_ms"]
        )
        for route in (
            "original", "ensemble_l1", "ensemble_l5",
            "ensemble_l20", "multiscale",
        )
    }
    payload = {
        "status": "PASS" if protocol_pass else "FAIL",
        "scope": (
            "post-hoc single-seed ETTh1 full-method external validation: "
            "new PG95 training followed by frozen equal-rate multiscale CNM-SE"
        ),
        "protocol_limitations": [
            "the hypothesis was proposed after the Stage-19 negative result",
            "the official ETTh1 missing-mask seed is unavailable",
            "the historical ETTh1 baseline differs materially from the paper",
            "single-seed results are directional, not statistical stability",
        ],
        "baseline": {
            "run_dir": str(BASELINE_RUN),
            "mse": float(baseline["mse"]),
            "mae": float(baseline["mae"]),
        },
        "pg95": {
            "run_dir": str(training_run),
            "checkpoint": str(training_checkpoint),
            "mse": float(training["mse"]),
            "mae": float(training["mae"]),
            "parameter_count": training.get("parameter_count"),
            "peak_gpu_memory_mb": training.get("peak_gpu_memory_mb"),
            "elapsed_seconds": training.get("elapsed_seconds"),
        },
        "evaluation_run": str(evaluation_run),
        "routes": routes,
        "gains": gains,
        "realized_additional_missing_rates": {
            f"l{block}": rate
            for block, rate in zip((1, 5, 20), realized_rates)
        },
        "serial_latency_mean_ms": latency,
        "multiscale_multiplier_vs_original": float(
            evaluation["estimated_serial_latency"]["multiscale"][
                "measured_multiplier_vs_original"
            ]
        ),
        "evaluation_peak_gpu_memory_mb": evaluation.get(
            "peak_gpu_memory_mb"
        ),
        "protocol_checks": protocol_checks,
        "pg95_gate_by_metric": pg95_by_metric,
        "cnm_se_gate_by_metric": cnm_se_by_metric,
        "incremental_gate_by_metric": incremental_by_metric,
        "pg95_gate_pass": pg95_gate,
        "cnm_se_gate_pass": cnm_se_gate,
        "incremental_gate_pass": incremental_gate,
        "decision": scientific_decision,
    }
    atomic_text(
        REPORT_JSON,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )

    route_names = {
        "on": "PG95原始路",
        "l1": "L1单路",
        "l5": "L5单路",
        "l20": "L20单路",
        "ensemble_l1": "原始+L1等权",
        "ensemble_l5": "原始+L5等权",
        "ensemble_l20": "原始+L20等权",
        "multiscale": "原始/L5/L20多尺度",
    }
    lines = [
        "# 步骤20：ETTh1完整PG95+多尺度CNM-SE",
        "",
        "| 方法/路由 | MSE | MAE |",
        "|---|---:|---:|",
        f"| 步骤5 S4M | {float(baseline['mse']):.9f} | "
        f"{float(baseline['mae']):.9f} |",
        f"| 新训练PG95 | {float(training['mse']):.9f} | "
        f"{float(training['mae']):.9f} |",
    ]
    for route in ROUTES:
        record = routes[route]
        lines.append(
            f"| {route_names[route]} | {record['mse']:.9f} | "
            f"{record['mae']:.9f} |"
        )
    lines.extend(
        [
            "",
            f"- 工件与协议检查：`{'PASS' if protocol_pass else 'FAIL'}`",
            f"- PG95相对步骤5 S4M双指标门：`{pg95_gate}`",
            f"- 多尺度相对同run PG95原始路双指标门：`{cnm_se_gate}`",
            f"- 多尺度相对L5双路增量门：`{incremental_gate}`",
            f"- 决策：`{scientific_decision}`",
            f"- PG95正式run：`{training_run}`",
            f"- 多尺度正式run：`{evaluation_run}`",
            "",
            "限制：这是步骤19之后提出的事后、ETTh1单seed方向性验证；"
            "不能改写步骤19结论，也不能单独支持统计稳定或二区投稿判断。",
            "",
        ]
    )
    atomic_text(REPORT_MD, "\n".join(lines))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if protocol_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
