#!/usr/bin/env python3
"""Aggregate the preregistered Step 22.19 ARD cross-mask confirmation."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = ROOT / "reports/stage22_19_ard_comparison.json"
REPORT_MD = ROOT / "reports/stage22_19_ard_comparison.md"


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


def load_json(path: Path, filename: str) -> dict[str, Any]:
    target = path / filename
    return json.loads(target.read_text(encoding="utf-8")) if target.is_file() else {}


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
            "stage22_crossmask_valcache_pg95__electricity__validation__seed2025__"
        ),
        "mcpr_mse": latest(
            "mcpr_mse_seed2025_control_electricity__electricity__point__mr0.06__seed2025__"
        ),
        "ard_mcpr": latest(
            "ard_mcpr_electricity__electricity__point__mr0.06__seed2025__"
        ),
    }
    if not all(paths.values()):
        payload = {
            "step": "22.19",
            "status": "IN_PROGRESS",
            "decision": "ARD_CROSSMASK_RUNNING",
            "runs": {key: str(value) if value else None for key, value in paths.items()},
            "test_accessed": False,
        }
        atomic_text(REPORT_JSON, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        atomic_text(
            REPORT_MD,
            "# 步骤22.19：ARD交叉掩码确认\n\n- 状态：`IN_PROGRESS`\n",
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    resolved = {key: value for key, value in paths.items() if value is not None}
    metrics = {key: load_json(path, "metrics.json") for key, path in resolved.items()}
    resources = {
        key: load_json(path, "resource_usage.json") for key, path in resolved.items()
    }
    configs = {
        key: yaml.safe_load(
            (path / "resolved_config.yaml").read_text(encoding="utf-8")
        )
        for key, path in resolved.items()
    }
    scores = {
        "pg95": {
            "mse": float(metrics["pg95"]["complete_target_mse_original"]),
            "mae": float(metrics["pg95"]["complete_target_mae_original"]),
        },
        "mcpr_mse": {
            "mse": float(metrics["mcpr_mse"]["mse"]),
            "mae": float(metrics["mcpr_mse"]["mae"]),
        },
        "ard_mcpr": {
            "mse": float(metrics["ard_mcpr"]["mse"]),
            "mae": float(metrics["ard_mcpr"]["mae"]),
        },
    }

    control_args = configs["mcpr_mse"]["official_args"]
    candidate_args = configs["ard_mcpr"]["official_args"]
    allowed_differences = {"model_id", "des", "use_ard", "mcpr_objective"}
    runtime_identity_keys = {
        "checkpoints", "results_root", "test_results_root", "metrics_path"
    }
    unexpected_differences = sorted(
        key for key in set(control_args) | set(candidate_args)
        if control_args.get(key) != candidate_args.get(key)
        and key not in allowed_differences
        and key not in runtime_identity_keys
    )
    expected_mask = (
        ROOT / "artifacts/masks/electricity/point/rate_0.06/seed_2025.npz"
    ).resolve()
    resolved_masks = []
    for name in ("mcpr_mse", "ard_mcpr"):
        mask_path = Path(configs[name]["data"]["mask_asset"])
        resolved_masks.append(
            mask_path.resolve()
            if mask_path.is_absolute()
            else (ROOT / mask_path).resolve()
        )
    action_counts = [
        int(metrics["ard_mcpr"].get(key, 0))
        for key in (
            "ard_teacher_off_count", "ard_teacher_partial_count",
            "ard_teacher_full_count",
        )
    ]
    protocol_checks = {
        "same_preregistered_seed2025_mask": all(
            path == expected_mask for path in resolved_masks
        ),
        "same_training_budget": not unexpected_differences,
        "same_frozen_pg95_source": (
            control_args["mcpr_backbone_checkpoint"]
            == candidate_args["mcpr_backbone_checkpoint"]
        ),
        "all_validation_only": (
            metrics["pg95"].get("uses_test_split") is False
            and all(
                metrics[name].get("test_split_accessed") is False
                for name in ("mcpr_mse", "ard_mcpr")
            )
        ),
        "all_full_validation": (
            int(metrics["pg95"].get("sample_count", 0)) == 2528
            and int(metrics["pg95"].get("validation_batches_processed", 0)) == 158
            and all(
                int(metrics[name].get("validation_samples_processed", 0)) == 2528
                and int(metrics[name].get("validation_batches_processed", 0)) == 158
                for name in ("mcpr_mse", "ard_mcpr")
            )
        ),
        "checkpoints_finite": all(
            metrics[name].get("checkpoint_all_model_tensors_finite") is True
            for name in ("pg95", "mcpr_mse", "ard_mcpr")
        ),
        "trained_backbones_and_memory_frozen": all(
            metrics[name].get("mcpr_backbone_frozen") is True
            and metrics[name].get("mcpr_backbone_hash_before")
            == metrics[name].get("mcpr_backbone_hash_after")
            and metrics[name].get("mcpr_memory_frozen") is True
            and metrics[name].get("mcpr_memory_hash_before")
            == metrics[name].get("mcpr_memory_hash_after")
            for name in ("mcpr_mse", "ard_mcpr")
        ),
        "control_executed": (
            metrics["mcpr_mse"].get("mcpr_objective") == "mse_only"
            and metrics["mcpr_mse"].get("ard_enabled") is False
            and int(metrics["mcpr_mse"].get("mcpr_steps", 0)) > 0
        ),
        "ard_executed": (
            metrics["ard_mcpr"].get("mcpr_objective")
            == "analytic_decision_distill"
            and metrics["ard_mcpr"].get("ard_enabled") is True
            and int(metrics["ard_mcpr"].get("ard_distillation_steps", 0)) > 0
            and int(metrics["ard_mcpr"].get("ard_teacher_valid_samples", 0)) > 0
            and sum(count > 0 for count in action_counts) >= 2
            and math.isfinite(float(metrics["ard_mcpr"].get(
                "ard_gate_std_mean", float("nan")
            )))
            and float(metrics["ard_mcpr"].get("ard_gate_std_mean", 0.0)) > 0.0
        ),
        "ard_checkpoint_contains_gate": len(
            metrics["ard_mcpr"].get("checkpoint_ard_state_keys", [])
        ) > 0,
        "control_checkpoint_has_no_ard_gate": not metrics["mcpr_mse"].get(
            "checkpoint_ard_state_keys", []
        ),
    }
    protocol_pass = all(protocol_checks.values())
    versus_control = dual_noninferior(scores["mcpr_mse"], scores["ard_mcpr"])
    versus_pg95 = dual_noninferior(scores["pg95"], scores["ard_mcpr"])
    if not protocol_pass:
        decision = "PROTOCOL_FAIL_NO_SCIENTIFIC_DECISION"
    elif versus_control and versus_pg95:
        decision = "ARD_DUAL_ANCHOR_GATE_PASS_MULTI_SEED_NEXT"
    elif versus_control:
        decision = "ARD_BEATS_MCPR_CONTROL_BUT_NOT_PG95_STOP"
    else:
        decision = "ARD_CONFIGURATION_STOP"

    comparisons = {
        "ard_vs_mcpr_mse": {
            "mse_gain_percent": gain(scores["mcpr_mse"]["mse"], scores["ard_mcpr"]["mse"]),
            "mae_gain_percent": gain(scores["mcpr_mse"]["mae"], scores["ard_mcpr"]["mae"]),
            "dual_noninferior_one_strict": versus_control,
        },
        "ard_vs_pg95": {
            "mse_gain_percent": gain(scores["pg95"]["mse"], scores["ard_mcpr"]["mse"]),
            "mae_gain_percent": gain(scores["pg95"]["mae"], scores["ard_mcpr"]["mae"]),
            "dual_noninferior_one_strict": versus_pg95,
        },
    }
    payload = {
        "step": "22.19",
        "status": "PASS" if protocol_pass else "FAIL",
        "decision": decision,
        "scope": "preregistered_cross_mask_seed2025_validation_only",
        "test_accessed": False,
        "runs": {key: str(value) for key, value in resolved.items()},
        "scores": scores,
        "comparisons": comparisons,
        "protocol_checks": protocol_checks,
        "unexpected_budget_differences": unexpected_differences,
        "mechanism": {
            key: metrics["ard_mcpr"].get(key)
            for key in (
                "ard_teacher_valid_samples", "ard_teacher_off_count",
                "ard_teacher_partial_count", "ard_teacher_full_count",
                "ard_teacher_alpha_mean", "ard_teacher_alpha_std",
                "ard_teacher_alpha_min", "ard_teacher_alpha_max",
                "ard_action_loss_mean", "ard_scale_loss_mean",
                "ard_gate_mean", "ard_gate_std_mean", "ard_gate_min",
                "ard_gate_max", "ard_action_entropy_mean",
                "ard_raw_residual_energy_mean", "ard_applied_residual_energy_mean",
            )
        },
        "cost": {
            name: {
                "trainable_parameter_count": metrics[name].get(
                    "mcpr_trainable_parameter_count"
                ),
                "total_parameters": metrics[name].get("parameter_count"),
                "formal_elapsed_seconds": resources[name].get("elapsed_seconds"),
                "peak_gpu_allocated_mb": metrics[name].get("peak_gpu_memory_mb"),
                "peak_gpu_reserved_mb": metrics[name].get("peak_gpu_reserved_mb"),
                "epochs_completed": metrics[name].get("epochs_completed"),
                "optimizer_steps": metrics[name].get("optimizer_steps"),
                "gpu_uuid": resources[name].get("gpu_uuid"),
            }
            for name in ("mcpr_mse", "ard_mcpr")
        },
    }
    atomic_text(REPORT_JSON, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# 步骤22.19：ARD交叉掩码确认", "",
        f"- 协议状态：`{payload['status']}`",
        f"- 科学决策：`{decision}`",
        "- 范围：预注册Electricity seed2025缺失掩码，validation-only",
        "- test访问：`false`", "",
        f"- PG95 MSE/MAE：`{scores['pg95']['mse']:.9f}/{scores['pg95']['mae']:.9f}`",
        f"- MCPR-MSE MSE/MAE：`{scores['mcpr_mse']['mse']:.9f}/{scores['mcpr_mse']['mae']:.9f}`",
        f"- ARD-MCPR MSE/MAE：`{scores['ard_mcpr']['mse']:.9f}/{scores['ard_mcpr']['mae']:.9f}`",
        f"- ARD相对MCPR-MSE收益：`{comparisons['ard_vs_mcpr_mse']['mse_gain_percent']:+.6f}% / {comparisons['ard_vs_mcpr_mse']['mae_gain_percent']:+.6f}%`",
        f"- ARD相对PG95收益：`{comparisons['ard_vs_pg95']['mse_gain_percent']:+.6f}% / {comparisons['ard_vs_pg95']['mae_gain_percent']:+.6f}%`",
        "",
    ]
    atomic_text(REPORT_MD, "\n".join(lines))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if protocol_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
