#!/usr/bin/env python3
"""Aggregate Stage22.16 MCPR validation-only screening results."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = ROOT / "reports" / "stage22_mcpr_comparison.json"
REPORT_MD = ROOT / "reports" / "stage22_mcpr_comparison.md"
PG95_RUN = ROOT / "runs" / "mipp_pg95_electricity__electricity__point__mr0.06__seed2023__20260804_222346"


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def latest(model: str) -> Path | None:
    prefix = f"{model}__electricity__point__mr0.06__seed2023__"
    candidates = []
    for path in (ROOT / "runs").glob(prefix + "*"):
        status_path = path / "status.json"
        config_path = path / "resolved_config.yaml"
        if not status_path.is_file() or not config_path.is_file():
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if status.get("state") != "SUCCESS" or config["experiment"].get("smoke_test"):
            continue
        candidates.append(path)
    return sorted(candidates)[-1] if candidates else None


def load_metrics(path: Path) -> dict[str, Any]:
    return json.loads((path / "metrics.json").read_text(encoding="utf-8"))


def percent(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def main() -> int:
    pg95 = load_metrics(PG95_RUN)
    runs = {
        "mcpr_mse": latest("mcpr_mse_pg95_electricity"),
        "mcpr_pcd": latest("mcpr_pcd_pg95_electricity"),
    }
    methods: dict[str, Any] = {}
    for name, path in runs.items():
        if path is None:
            methods[name] = {"state": "PENDING", "run": None}
            continue
        metrics = load_metrics(path)
        mse_gain = percent(float(pg95["mse"]), float(metrics["mse"]))
        mae_gain = percent(float(pg95["mae"]), float(metrics["mae"]))
        checks = {
            "validation_only": metrics.get("test_split_accessed") is False,
            "full_validation": (
                int(metrics.get("validation_batches_processed", 0)) == 158
                and int(metrics.get("validation_samples_processed", 0)) == 2528
            ),
            "checkpoint_finite": metrics.get("checkpoint_all_model_tensors_finite") is True,
            "backbone_frozen": (
                metrics.get("mcpr_backbone_frozen") is True
                and metrics.get("mcpr_backbone_hash_before")
                == metrics.get("mcpr_backbone_hash_after")
            ),
            "memory_frozen": (
                metrics.get("mcpr_memory_frozen") is True
                and metrics.get("mcpr_memory_hash_before")
                == metrics.get("mcpr_memory_hash_after")
            ),
            "mechanism_executed": (
                int(metrics.get("mcpr_steps", 0)) > 0
                and float(metrics.get("mcpr_residual_energy_mean", 0.0)) > 0.0
            ),
        }
        dual_gate = bool(
            float(metrics["mse"]) <= float(pg95["mse"])
            and float(metrics["mae"]) <= float(pg95["mae"])
            and (float(metrics["mse"]) < float(pg95["mse"])
                 or float(metrics["mae"]) < float(pg95["mae"]))
        )
        methods[name] = {
            "state": "SUCCESS",
            "run": str(path),
            "mse": float(metrics["mse"]),
            "mae": float(metrics["mae"]),
            "mse_gain_percent": mse_gain,
            "mae_gain_percent": mae_gain,
            "protocol_checks": checks,
            "protocol_pass": all(checks.values()),
            "dual_metric_gate_pass": dual_gate,
            "objective": metrics.get("mcpr_objective"),
            "trainable_parameter_count": metrics.get("mcpr_trainable_parameter_count"),
            "gradient_cosine_mean": metrics.get("mcpr_gradient_cosine_mean"),
        }
    complete = all(value.get("state") == "SUCCESS" for value in methods.values())
    winners = [
        name for name, value in methods.items()
        if value.get("protocol_pass") and value.get("dual_metric_gate_pass")
    ]
    decision = (
        "MCPR_SCREEN_RUNNING" if not complete
        else "MCPR_GATE_PASS_ADD_EQUAL_BUDGET_CONTROL" if winners
        else "MCPR_BRANCH_STOP"
    )
    payload = {
        "step": "22.16",
        "status": "PASS" if complete else "IN_PROGRESS",
        "decision": decision,
        "test_accessed": False,
        "pg95": {"run": str(PG95_RUN), "mse": pg95["mse"], "mae": pg95["mae"]},
        "methods": methods,
        "winners": winners,
        "fairness_note": "post-hoc screen only; any winner requires equal-budget PG95 continuation control",
    }
    atomic_text(REPORT_JSON, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# 步骤22.16：MCPR验证集快速筛选", "",
        f"- 状态：`{payload['status']}`",
        f"- 决策：`{decision}`",
        f"- PG95 MSE/MAE：`{pg95['mse']:.9f} / {pg95['mae']:.9f}`",
        "- test访问：`false`", "",
    ]
    for name, value in methods.items():
        if value.get("state") == "SUCCESS":
            lines.append(
                f"- {name}: MSE/MAE `{value['mse']:.9f}/{value['mae']:.9f}`，"
                f"收益 `{value['mse_gain_percent']:+.6f}%/{value['mae_gain_percent']:+.6f}%`，"
                f"双指标门 `{value['dual_metric_gate_pass']}`"
            )
        else:
            lines.append(f"- {name}: `PENDING`")
    lines.extend(["", "若有候选通过，仍需PG95等额继续训练控制后才能归因。", ""])
    atomic_text(REPORT_MD, "\n".join(lines))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
