#!/usr/bin/env python3
"""Aggregate the preregistered Step 22.18 cross-mask confirmation."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = ROOT / "reports/stage22_18_ttrr_comparison.json"
REPORT_MD = ROOT / "reports/stage22_18_ttrr_comparison.md"


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def latest(prefix: str) -> Path | None:
    candidates: list[Path] = []
    for path in (ROOT / "runs").glob(prefix + "*"):
        status_path = path / "status.json"
        config_path = path / "resolved_config.yaml"
        metrics_path = path / "metrics.json"
        if not all(item.is_file() for item in (status_path, config_path, metrics_path)):
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if status.get("state") != "SUCCESS" or config["experiment"].get("smoke_test"):
            continue
        candidates.append(path)
    return sorted(candidates)[-1] if candidates else None


def load_metrics(path: Path) -> dict[str, Any]:
    return json.loads((path / "metrics.json").read_text(encoding="utf-8"))


def load_resource(path: Path) -> dict[str, Any]:
    resource_path = path / "resource_usage.json"
    return (
        json.loads(resource_path.read_text(encoding="utf-8"))
        if resource_path.is_file() else {}
    )


def gain(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def dual_noninferior(reference: dict[str, float], candidate: dict[str, float]) -> bool:
    return bool(
        candidate["mse"] <= reference["mse"]
        and candidate["mae"] <= reference["mae"]
        and (
            candidate["mse"] < reference["mse"]
            or candidate["mae"] < reference["mae"]
        )
    )


def main() -> int:
    paths = {
        "pg95": latest(
            "stage22_crossmask_valcache_pg95__electricity__validation__seed2024__"
        ),
        "mcpr_mse": latest(
            "mcpr_mse_crossmask_electricity__electricity__point__mr0.06__seed2024__"
        ),
        "ttrr_mcpr": latest(
            "ttrr_mcpr_pg95_electricity__electricity__point__mr0.06__seed2024__"
        ),
    }
    if not all(paths.values()):
        payload = {
            "step": "22.18",
            "status": "IN_PROGRESS",
            "decision": "TTRR_CROSSMASK_RUNNING",
            "runs": {key: str(value) if value else None for key, value in paths.items()},
            "test_accessed": False,
        }
        atomic_text(REPORT_JSON, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        atomic_text(
            REPORT_MD,
            "# 步骤22.18：TTRR交叉掩码确认\n\n- 状态：`IN_PROGRESS`\n",
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    resolved_paths = {key: value for key, value in paths.items() if value is not None}
    metrics = {key: load_metrics(path) for key, path in resolved_paths.items()}
    resources = {key: load_resource(path) for key, path in resolved_paths.items()}
    scores = {
        "pg95": {
            "mse": float(metrics["pg95"]["complete_target_mse_original"]),
            "mae": float(metrics["pg95"]["complete_target_mae_original"]),
        },
        "mcpr_mse": {
            "mse": float(metrics["mcpr_mse"]["mse"]),
            "mae": float(metrics["mcpr_mse"]["mae"]),
        },
        "ttrr_mcpr": {
            "mse": float(metrics["ttrr_mcpr"]["mse"]),
            "mae": float(metrics["ttrr_mcpr"]["mae"]),
        },
    }

    control_config = yaml.safe_load(
        (resolved_paths["mcpr_mse"] / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    candidate_config = yaml.safe_load(
        (resolved_paths["ttrr_mcpr"] / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    allowed_differences = {
        "model_id", "des", "use_ttrr", "mcpr_objective",
        "ttrr_tail_fraction", "ttrr_tail_weight",
    }
    control_args = control_config["official_args"]
    candidate_args = candidate_config["official_args"]
    unexpected_differences = sorted(
        key for key in set(control_args) | set(candidate_args)
        if control_args.get(key) != candidate_args.get(key)
        and key not in allowed_differences
        and key not in {
            "checkpoints", "results_root", "test_results_root", "metrics_path"
        }
    )
    control_mask = Path(control_config["data"]["mask_asset"])
    candidate_mask = Path(candidate_config["data"]["mask_asset"])
    if not control_mask.is_absolute():
        control_mask = (ROOT / control_mask).resolve()
    if not candidate_mask.is_absolute():
        candidate_mask = (ROOT / candidate_mask).resolve()
    expected_mask = (
        ROOT / "artifacts/masks/electricity/point/rate_0.06/seed_2024.npz"
    ).resolve()
    common_asset = control_mask == candidate_mask == expected_mask

    protocol_checks = {
        "same_fixed_seed2024_mask": common_asset,
        "same_training_budget": not unexpected_differences,
        "same_frozen_pg95_source": (
            control_args["mcpr_backbone_checkpoint"]
            == candidate_args["mcpr_backbone_checkpoint"]
        ),
        "all_validation_only": all(
            item.get("test_split_accessed") is False
            for item in (metrics["mcpr_mse"], metrics["ttrr_mcpr"])
        ) and metrics["pg95"].get("uses_test_split") is False,
        "all_full_validation": (
            int(metrics["pg95"].get("sample_count", 0)) == 2528
            and int(metrics["pg95"].get("validation_batches_processed", 0)) == 158
            and all(
                int(metrics[name].get("validation_samples_processed", 0)) == 2528
                and int(metrics[name].get("validation_batches_processed", 0)) == 158
                for name in ("mcpr_mse", "ttrr_mcpr")
            )
        ),
        "checkpoints_finite": all(
            metrics[name].get("checkpoint_all_model_tensors_finite") is True
            for name in ("pg95", "mcpr_mse", "ttrr_mcpr")
        ),
        "trained_backbones_frozen": all(
            metrics[name].get("mcpr_backbone_frozen") is True
            and metrics[name].get("mcpr_backbone_hash_before")
            == metrics[name].get("mcpr_backbone_hash_after")
            and metrics[name].get("mcpr_memory_frozen") is True
            and metrics[name].get("mcpr_memory_hash_before")
            == metrics[name].get("mcpr_memory_hash_after")
            for name in ("mcpr_mse", "ttrr_mcpr")
        ),
        "control_executed": (
            metrics["mcpr_mse"].get("mcpr_objective") == "mse_only"
            and metrics["mcpr_mse"].get("ttrr_enabled") is False
            and int(metrics["mcpr_mse"].get("mcpr_steps", 0)) > 0
        ),
        "ttrr_executed": (
            metrics["ttrr_mcpr"].get("mcpr_objective") == "tail_trust_mse"
            and metrics["ttrr_mcpr"].get("ttrr_enabled") is True
            and int(metrics["ttrr_mcpr"].get("ttrr_diagnostic_steps", 0)) > 0
            and math.isfinite(float(metrics["ttrr_mcpr"].get("ttrr_gate_mean", float("nan"))))
        ),
    }
    protocol_pass = all(protocol_checks.values())
    versus_control = dual_noninferior(scores["mcpr_mse"], scores["ttrr_mcpr"])
    versus_pg95 = dual_noninferior(scores["pg95"], scores["ttrr_mcpr"])
    if not protocol_pass:
        decision = "PROTOCOL_FAIL_NO_SCIENTIFIC_DECISION"
    elif versus_control and versus_pg95:
        decision = "TTRR_DUAL_ANCHOR_GATE_PASS_EXTERNAL_VALIDATION_NEXT"
    elif versus_control:
        decision = "TTRR_OVERSHOOT_CONTROL_WORKS_FULL_METHOD_STILL_FAILS"
    else:
        decision = "TTRR_CONFIGURATION_STOP"

    comparisons = {
        "ttrr_vs_mcpr_mse": {
            "mse_gain_percent": gain(scores["mcpr_mse"]["mse"], scores["ttrr_mcpr"]["mse"]),
            "mae_gain_percent": gain(scores["mcpr_mse"]["mae"], scores["ttrr_mcpr"]["mae"]),
            "dual_noninferior_one_strict": versus_control,
        },
        "ttrr_vs_pg95": {
            "mse_gain_percent": gain(scores["pg95"]["mse"], scores["ttrr_mcpr"]["mse"]),
            "mae_gain_percent": gain(scores["pg95"]["mae"], scores["ttrr_mcpr"]["mae"]),
            "dual_noninferior_one_strict": versus_pg95,
        },
    }
    timing_paths = {
        name: latest(
            f"stage22_18_timing_{name}__electricity__validation__seed2024__"
        )
        for name in ("pg95", "mcpr_mse", "ttrr_mcpr")
    }
    timing_metrics = {
        name: load_metrics(path)
        for name, path in timing_paths.items() if path is not None
    }
    timing_complete = len(timing_metrics) == 3
    timing_checks: dict[str, bool] = {}
    timing_summary: dict[str, Any] = {
        "status": "IN_PROGRESS",
        "runs": {
            name: str(path) if path is not None else None
            for name, path in timing_paths.items()
        },
        "test_accessed": False,
    }
    if timing_complete:
        gpu_uuids = {
            str(timing_metrics[name].get("gpu_uuid"))
            for name in timing_metrics
        }
        timing_checks = {
            "same_gpu_uuid": len(gpu_uuids) == 1 and "None" not in gpu_uuids,
            "all_serial_forward_only": all(
                item.get("timing_device_serial") is True
                and str(item.get("timing_scope", "")).startswith("model forward only")
                for item in timing_metrics.values()
            ),
            "all_cuda_event_timed": all(
                item.get("timing_clock") == "torch.cuda.Event"
                for item in timing_metrics.values()
            ),
            "all_full_validation": all(
                int(item.get("validation_batches_processed", 0)) == 158
                and int(item.get("validation_samples_processed", 0)) == 2528
                and int(item.get("timing_repeats", 0)) == 2
                and int(item.get("timing_warmup_batches_per_repeat", -1)) == 10
                for item in timing_metrics.values()
            ),
            "all_validation_only": all(
                item.get("test_split_accessed") is False
                and item.get("uses_test_split") is False
                for item in timing_metrics.values()
            ),
            "all_artifacts_pass": all(
                item.get("artifact_status") == "PASS"
                for item in timing_metrics.values()
            ),
            "all_metric_roundtrips": all(
                abs(float(item["mse"]) - float(item["source_validation_mse_reference"]))
                <= 1e-5
                and abs(float(item["mae"]) - float(item["source_validation_mae_reference"]))
                <= 1e-5
                for item in timing_metrics.values()
            ),
        }
        timing_pass = all(timing_checks.values())
        timing_summary = {
            "status": "PASS" if timing_pass else "FAIL",
            "runs": {name: str(path) for name, path in timing_paths.items()},
            "gpu_uuid": next(iter(gpu_uuids)) if len(gpu_uuids) == 1 else None,
            "scope": "same_gpu_serial_model_forward_only_full_validation",
            "test_accessed": False,
            "checks": timing_checks,
            "methods": {
                name: {
                    key: timing_metrics[name].get(key)
                    for key in (
                        "mse", "mae", "parameter_count", "parameter_bytes",
                        "buffer_bytes", "forward_mean_ms_per_batch",
                        "forward_std_ms_per_batch", "forward_p50_ms_per_batch",
                        "forward_p95_ms_per_batch", "forward_mean_ms_per_sample",
                        "timing_measured_batches_total", "peak_gpu_memory_mb",
                        "peak_gpu_reserved_mb",
                    )
                }
                for name in timing_metrics
            },
            "overhead_percent": {
                "mcpr_mse_vs_pg95": 100.0 * (
                    float(timing_metrics["mcpr_mse"]["forward_mean_ms_per_batch"])
                    / float(timing_metrics["pg95"]["forward_mean_ms_per_batch"])
                    - 1.0
                ),
                "ttrr_mcpr_vs_pg95": 100.0 * (
                    float(timing_metrics["ttrr_mcpr"]["forward_mean_ms_per_batch"])
                    / float(timing_metrics["pg95"]["forward_mean_ms_per_batch"])
                    - 1.0
                ),
                "ttrr_mcpr_vs_mcpr_mse": 100.0 * (
                    float(timing_metrics["ttrr_mcpr"]["forward_mean_ms_per_batch"])
                    / float(timing_metrics["mcpr_mse"]["forward_mean_ms_per_batch"])
                    - 1.0
                ),
            },
        }
    payload = {
        "step": "22.18",
        "status": "PASS" if protocol_pass else "FAIL",
        "decision": decision,
        "scope": "cross_mask_seed2024_validation_only_not_independent_time_or_dataset",
        "test_accessed": False,
        "runs": {key: str(value) for key, value in resolved_paths.items()},
        "scores": scores,
        "comparisons": comparisons,
        "protocol_checks": protocol_checks,
        "unexpected_budget_differences": unexpected_differences,
        "mechanism": {
            key: metrics["ttrr_mcpr"].get(key)
            for key in (
                "ttrr_tail_fraction", "ttrr_tail_weight", "ttrr_tail_cvar_mean",
                "ttrr_positive_harm_fraction_mean", "ttrr_gate_mean",
                "ttrr_gate_std_mean", "ttrr_gate_min", "ttrr_gate_max",
                "ttrr_raw_residual_energy_mean", "ttrr_applied_residual_energy_mean",
            )
        },
        "inference_timing": timing_summary,
        "cost": {
            name: {
                "trainable_parameter_count": metrics[name].get("mcpr_trainable_parameter_count"),
                "total_parameters": metrics[name].get("parameter_count"),
                "formal_run_elapsed_seconds": resources[name].get("elapsed_seconds"),
                "peak_gpu_allocated_mb": metrics[name].get("peak_gpu_memory_mb"),
                "peak_gpu_reserved_mb": metrics[name].get("peak_gpu_reserved_mb"),
                "epochs_completed": metrics[name].get("epochs_completed"),
                "optimizer_steps": metrics[name].get("optimizer_steps"),
                "gpu_uuid": resources[name].get("gpu_uuid"),
            }
            for name in ("mcpr_mse", "ttrr_mcpr")
        },
    }
    atomic_text(REPORT_JSON, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# 步骤22.18：TTRR交叉掩码确认", "",
        f"- 协议状态：`{payload['status']}`",
        f"- 科学决策：`{decision}`",
        "- 范围：Electricity seed2024新缺失掩码validation-only；不是独立时间或数据集验证",
        "- test访问：`false`", "",
        f"- PG95 MSE/MAE：`{scores['pg95']['mse']:.9f}/{scores['pg95']['mae']:.9f}`",
        f"- MCPR-MSE MSE/MAE：`{scores['mcpr_mse']['mse']:.9f}/{scores['mcpr_mse']['mae']:.9f}`",
        f"- TTRR-MCPR MSE/MAE：`{scores['ttrr_mcpr']['mse']:.9f}/{scores['ttrr_mcpr']['mae']:.9f}`",
        f"- TTRR相对MCPR-MSE收益：`{comparisons['ttrr_vs_mcpr_mse']['mse_gain_percent']:+.6f}% / {comparisons['ttrr_vs_mcpr_mse']['mae_gain_percent']:+.6f}%`",
        f"- TTRR相对PG95收益：`{comparisons['ttrr_vs_pg95']['mse_gain_percent']:+.6f}% / {comparisons['ttrr_vs_pg95']['mae_gain_percent']:+.6f}%`",
        f"- 同GPU串行推理计时：`{timing_summary['status']}`",
        "",
    ]
    if timing_summary["status"] == "PASS":
        lines.extend([
            "| 方法 | 参数量 | 前向均值（ms/batch） | P95（ms/batch） |", 
            "|---|---:|---:|---:|",
            *[
                "| {} | {} | {:.6f} | {:.6f} |".format(
                    name,
                    timing_summary["methods"][name]["parameter_count"],
                    timing_summary["methods"][name]["forward_mean_ms_per_batch"],
                    timing_summary["methods"][name]["forward_p95_ms_per_batch"],
                )
                for name in ("pg95", "mcpr_mse", "ttrr_mcpr")
            ],
            "",
        ])
    atomic_text(REPORT_MD, "\n".join(lines))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if protocol_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
