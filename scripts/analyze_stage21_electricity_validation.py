#!/usr/bin/env python3
"""Audit Stage-21 Electricity validation results and freeze the test gate."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime" / "stage21_electricity_pipeline" / "state.json"
REPORT_JSON = ROOT / "reports" / "stage21_electricity_validation_gate.json"
REPORT_MD = ROOT / "reports" / "stage21_electricity_validation_gate.md"
BASELINE_JOB = "electricity_s4m_seed2023_formal"
PG95_JOB = "electricity_pg95_seed2023_formal"
MULTISCALE_JOB = "electricity_pg95_multiscale_validation_seed2023_formal"
ROUTES = (
    "on", "l1", "l5", "l20",
    "ensemble_l1", "ensemble_l5", "ensemble_l20", "multiscale",
)
RUN_IDENTITY_DIFFERENCES = {
    "model_id", "des", "checkpoints", "metrics_path",
    "results_root", "test_results_root",
}
PG95_MODULE_DIFFERENCES = {
    "use_mipp", "use_content_mask_factorization",
    "use_mask_intervention_pair", "use_stability_write_gate",
    "mipp_write_gate_mode", "mipp_inv_weight", "mipp_var_weight",
    "mipp_cov_weight", "mipp_dec_weight", "mipp_mask_weight",
    "mipp_variance_target", "mipp_stability_threshold",
    "mipp_pair_rate_min", "mipp_pair_rate_max",
    "mipp_pair_block_lengths",
}
REQUIRED_METHOD_DIFFERENCES = {"model_id", "des", "use_mipp"}
ALLOWED_METHOD_DIFFERENCES = (
    RUN_IDENTITY_DIFFERENCES | PG95_MODULE_DIFFERENCES
)


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def load_successful_run(
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


def gain_percent(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference * 100.0


def audit_method_differences(
    baseline_args: dict[str, Any], pg95_args: dict[str, Any]
) -> tuple[set[str], set[str], bool]:
    """Separate expected run/module identity from real protocol drift."""
    actual = {
        key
        for key in set(baseline_args) | set(pg95_args)
        if baseline_args.get(key) != pg95_args.get(key)
    }
    unexpected = actual - ALLOWED_METHOD_DIFFERENCES
    required_present = REQUIRED_METHOD_DIFFERENCES <= actual
    return actual, unexpected, required_present


def main() -> int:
    state = json.loads(STATE.read_text(encoding="utf-8"))
    baseline, baseline_status, baseline_config, baseline_run = load_successful_run(
        state, BASELINE_JOB
    )
    pg95, pg95_status, pg95_config, pg95_run = load_successful_run(
        state, PG95_JOB
    )
    multiscale, multiscale_status, multiscale_config, multiscale_run = (
        load_successful_run(state, MULTISCALE_JOB)
    )
    baseline_args = baseline_config["official_args"]
    pg95_args = pg95_config["official_args"]
    routes = {
        route: {
            metric: float(multiscale[f"complete_target_{metric}_{route}"])
            for metric in ("mse", "mae")
        }
        for route in ROUTES
    }
    baseline_checkpoint = Path(baseline["checkpoint"]).resolve()
    pg95_checkpoint = Path(pg95["checkpoint"]).resolve()
    evaluation_checkpoint = Path(multiscale["source_checkpoint"]).resolve()

    (
        actual_method_differences,
        unexpected_method_differences,
        required_method_differences_present,
    ) = audit_method_differences(baseline_args, pg95_args)
    validation_identity = {
        "evaluation_split": "validation",
        "post_train_evaluation": "validation",
        "test_split_accessed": False,
        "chronological_loader": True,
    }

    protocol_checks = {
        "required_jobs_success": all(
            state["jobs"][job_id].get("state") == "SUCCESS"
            for job_id in (BASELINE_JOB, PG95_JOB, MULTISCALE_JOB)
        ),
        "training_artifacts_success": (
            baseline_status.get("state") == "SUCCESS"
            and pg95_status.get("state") == "SUCCESS"
            and baseline.get("checkpoint_load_verified") is True
            and pg95.get("checkpoint_load_verified") is True
        ),
        "both_training_metrics_are_validation_only": all(
            all(metrics.get(key) == value for key, value in validation_identity.items())
            for metrics in (baseline, pg95)
        ),
        "same_fixed_data_assets": baseline_config["data"] == pg95_config["data"],
        "same_budget_except_pg95_module": (
            not unexpected_method_differences
            and required_method_differences_present
        ),
        "frozen_full_training_budget": all(
            int(args["batch_size"]) == 16
            and float(args["learning_rate"]) == 0.01
            and args["optimizer"] == "adam"
            and int(args["train_epochs"]) == 15
            and int(args["patience"]) == 3
            and int(args["max_train_batches"]) == -1
            and int(args["max_eval_batches"]) == -1
            and int(args["seq_len"]) == 192
            and int(args["pred_len"]) == 96
            for args in (baseline_args, pg95_args)
        ),
        "frozen_electricity_architecture": all(
            int(args["enc_in"]) == int(args["dec_in"]) == int(args["c_out"]) == 321
            and int(args["d_model"]) == int(args["d_ff"]) == 512
            and int(args["en_conv_hidden_size"]) == 512
            and int(args["e_layers"]) == 8
            for args in (baseline_args, pg95_args)
        ),
        "baseline_modules_disabled": all(
            baseline_args.get(key, False) is False
            for key in ("use_mipp", "use_fusr", "use_cfr")
        ),
        "pg95_module_frozen": (
            pg95_args.get("use_mipp") is True
            and pg95_args.get("use_content_mask_factorization") is False
            and pg95_args.get("use_mask_intervention_pair") is True
            and pg95_args.get("use_stability_write_gate") is True
            and pg95_args.get("mipp_write_gate_mode") == "absolute"
            and float(pg95_args.get("mipp_inv_weight")) == 0.05
            and float(pg95_args.get("mipp_stability_threshold")) == 0.95
            and pg95_args.get("use_fusr") is False
            and pg95_args.get("use_cfr") is False
        ),
        "complete_validation_budget": all(
            int(metrics["validation_batches_processed"]) == 158
            and int(metrics["validation_samples_processed"]) == 2528
            for metrics in (baseline, pg95)
        ),
        "evaluation_artifact_pass": (
            multiscale_status.get("state") == "SUCCESS"
            and multiscale.get("artifact_status") == "PASS"
        ),
        "evaluation_is_validation_only": (
            multiscale.get("dataset") == "electricity"
            and multiscale.get("method")
            == "cnm_se_equal_rate_multiscale_pg95"
            and multiscale.get("uses_test_split") is False
            and multiscale.get("source_split") == "val"
            and int(multiscale["validation_batches_processed"]) == 158
            and int(multiscale["sample_count"]) == 2528
        ),
        "exact_pg95_checkpoint_handoff": pg95_checkpoint == evaluation_checkpoint,
        "source_validation_metric_roundtrip": bool(
            multiscale["artifact_checks"].get(
                "matched_source_validation_baseline"
            )
        ),
        "source_is_validation_only": bool(
            multiscale["artifact_checks"].get("source_is_validation_only")
        ),
        "equal_count_protocol": (
            float(multiscale["target_additional_missing_rate"]) == 0.15
            and multiscale["block_lengths"] == [1, 5, 20]
            and multiscale["target_count_rounding"] == "numpy_rint_ties_to_even"
            and bool(
                multiscale["artifact_checks"][
                    "strict_equal_removed_count_per_sample"
                ]
            )
        ),
        "fixed_multiscale_weights": multiscale["multiscale_weights"] == {
            "original": 0.5, "l5": 0.25, "l20": 0.25
        },
        "frozen_evaluation": multiscale.get("updates_model_or_memory") is False,
        "serial_timing_recorded": (
            multiscale.get("timing_device_serial") is True
            and int(multiscale["timing_measured_batches"]) == 148
        ),
        "test_not_accessed_before_gate": (
            baseline.get("test_split_accessed") is False
            and pg95.get("test_split_accessed") is False
            and multiscale.get("uses_test_split") is False
        ),
    }
    realized_rates = [
        float(multiscale[f"realized_additional_missing_rate_l{block}"])
        for block in (1, 5, 20)
    ]
    protocol_checks["equal_realized_rate_across_scales"] = (
        max(realized_rates) - min(realized_rates) <= 1e-12
    )
    protocol_pass = all(protocol_checks.values())

    pg95_by_metric = {
        metric: float(pg95[metric]) <= float(baseline[metric])
        for metric in ("mse", "mae")
    }
    pg95_has_strict_improvement = any(
        float(pg95[metric]) < float(baseline[metric])
        for metric in ("mse", "mae")
    )
    pg95_gate = all(pg95_by_metric.values()) and pg95_has_strict_improvement
    multiscale_vs_original = {
        metric: routes["multiscale"][metric] < routes["on"][metric]
        for metric in ("mse", "mae")
    }
    multiscale_vs_l5 = {
        metric: (
            routes["multiscale"][metric] < routes["ensemble_l5"][metric]
        )
        for metric in ("mse", "mae")
    }
    multiscale_gate = (
        all(multiscale_vs_original.values())
        and all(multiscale_vs_l5.values())
    )
    test_authorized = protocol_pass and pg95_gate and multiscale_gate
    if not protocol_pass:
        decision = "PROTOCOL_FAIL_TEST_FORBIDDEN"
    elif not pg95_gate and not multiscale_gate:
        decision = "FULL_VALIDATION_FAIL_TEST_FORBIDDEN"
    elif not pg95_gate:
        decision = "PG95_VALIDATION_FAIL_TEST_FORBIDDEN"
    elif not multiscale_gate:
        decision = "MULTISCALE_VALIDATION_FAIL_TEST_FORBIDDEN"
    else:
        decision = "VALIDATION_PASS_ONE_TEST_AUTHORIZED"

    payload = {
        "status": "PASS" if protocol_pass else "FAIL",
        "scope": "single-seed Electricity validation-only scientific gate",
        "decision": decision,
        "test_authorized": test_authorized,
        "gate_definition": {
            "pg95": (
                "MSE and MAE must both be <= S4M, with at least one strict "
                "improvement; no numerical gain threshold"
            ),
            "multiscale": (
                "MSE and MAE must both be strictly lower than PG95 original "
                "and original+L5; no numerical gain threshold"
            ),
            "test_policy": "one test evaluation only when every protocol and gate passes",
        },
        "protocol_limitations": [
            "the official Electricity missing-mask seed is unavailable",
            "the paper discloses a layer search set but not the selected Electricity layer count",
            "the pre-frozen eight-layer architecture is therefore an approximate protocol match",
            "single-seed validation is directional and not statistical stability",
            "paper test values 0.274/0.367 are references only and are not compared to validation",
        ],
        "paper_test_reference_only": {"mse": 0.274, "mae": 0.367},
        "baseline": {
            "run_dir": str(baseline_run),
            "checkpoint": str(baseline_checkpoint),
            "mse": float(baseline["mse"]),
            "mae": float(baseline["mae"]),
        },
        "pg95": {
            "run_dir": str(pg95_run),
            "checkpoint": str(pg95_checkpoint),
            "mse": float(pg95["mse"]),
            "mae": float(pg95["mae"]),
        },
        "multiscale_run": str(multiscale_run),
        "routes": routes,
        "gains_percent": {
            "pg95_vs_s4m": {
                metric: gain_percent(
                    float(baseline[metric]), float(pg95[metric])
                )
                for metric in ("mse", "mae")
            },
            "multiscale_vs_pg95_original": {
                metric: gain_percent(
                    routes["on"][metric], routes["multiscale"][metric]
                )
                for metric in ("mse", "mae")
            },
            "multiscale_vs_original_plus_l5": {
                metric: gain_percent(
                    routes["ensemble_l5"][metric],
                    routes["multiscale"][metric],
                )
                for metric in ("mse", "mae")
            },
        },
        "protocol_checks": protocol_checks,
        "protocol_pass": protocol_pass,
        "pg95_gate_by_metric": pg95_by_metric,
        "pg95_has_strict_improvement": pg95_has_strict_improvement,
        "pg95_gate_pass": pg95_gate,
        "multiscale_vs_original_by_metric": multiscale_vs_original,
        "multiscale_vs_l5_by_metric": multiscale_vs_l5,
        "multiscale_gate_pass": multiscale_gate,
        "actual_method_differences": sorted(actual_method_differences),
        "unexpected_method_differences": sorted(
            unexpected_method_differences
        ),
    }
    atomic_text(
        REPORT_JSON, json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# 步骤21：Electricity validation科学门控",
        "",
        f"- 协议审计：`{'PASS' if protocol_pass else 'FAIL'}`",
        f"- 决策：`{decision}`",
        f"- 是否授权唯一一次test：`{str(test_authorized).lower()}`",
        "",
        "| 路径 | MSE | MAE |",
        "|---|---:|---:|",
        f"| S4M validation | {float(baseline['mse']):.8f} | {float(baseline['mae']):.8f} |",
        f"| PG95 validation | {float(pg95['mse']):.8f} | {float(pg95['mae']):.8f} |",
        f"| PG95原始路复核 | {routes['on']['mse']:.8f} | {routes['on']['mae']:.8f} |",
        f"| 原始+L5 | {routes['ensemble_l5']['mse']:.8f} | {routes['ensemble_l5']['mae']:.8f} |",
        f"| 多尺度CNM-SE | {routes['multiscale']['mse']:.8f} | {routes['multiscale']['mae']:.8f} |",
        "",
        "门控仅使用validation；论文的0.274/0.367是test参考值，不参与门控。",
    ]
    atomic_text(REPORT_MD, "\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if protocol_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
