#!/usr/bin/env python3
"""Aggregate validation-only Step-22.26 BF-CMVD screening evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime/stage22_26_traffic/state.json"
REPORT_JSON = ROOT / "reports/stage22_26_bf_cmvd_seed2027.json"
REPORT_MD = ROOT / "reports/stage22_26_bf_cmvd_seed2027.md"
SMOKE_JOB = "bf_cmvd_traffic_validation_seed2027_smoke"
FORMAL_JOB = "bf_cmvd_traffic_validation_seed2027_formal"
PG95_RUN = ROOT / "runs/mipp_pg95_traffic__traffic__point__mr0.06__seed2027__20260820_174327"
CNM_SE_RUN = ROOT / "runs/cnm_se_equal_rate_multiscale_pg95__traffic__validation__seed2027__20260820_184433"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def run_metrics(job: dict) -> tuple[dict, str | None]:
    if job.get("state") != "SUCCESS" or not job.get("run_dir"):
        return {}, None
    path = Path(job["run_dir"]).resolve() / "metrics.json"
    return (load_json(path), str(path.parent)) if path.is_file() else ({}, None)


def relative_gain(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def main() -> int:
    if not STATE.is_file():
        return 0
    state = load_json(STATE)
    smoke, smoke_run = run_metrics(state["jobs"][SMOKE_JOB])
    formal, formal_run = run_metrics(state["jobs"][FORMAL_JOB])
    comparison: dict[str, Any] = {}
    decision = "BF_CMVD_SMOKE_INCOMPLETE"
    if smoke:
        decision = "BF_CMVD_FORMAL_INCOMPLETE"
    if state["jobs"][FORMAL_JOB].get("state") == "BLOCKED_DEPENDENCY":
        decision = "BF_CMVD_SMOKE_MECHANISM_FAIL"
    if formal:
        pg95 = load_json(PG95_RUN / "metrics.json")
        cnm_se = load_json(CNM_SE_RUN / "metrics.json")
        pg95_mse, pg95_mae = float(pg95["mse"]), float(pg95["mae"])
        teacher_mse = float(cnm_se["complete_target_mse_multiscale"])
        teacher_mae = float(cnm_se["complete_target_mae_multiscale"])
        student_mse, student_mae = float(formal["mse"]), float(formal["mae"])
        mse_gain = relative_gain(pg95_mse, student_mse)
        mae_gain = relative_gain(pg95_mae, student_mae)
        teacher_mse_gain = relative_gain(pg95_mse, teacher_mse)
        teacher_mae_gain = relative_gain(pg95_mae, teacher_mae)
        mse_retention = mse_gain / teacher_mse_gain if teacher_mse_gain > 0.0 else None
        mae_retention = mae_gain / teacher_mae_gain if teacher_mae_gain > 0.0 else None
        gate_pass = bool(
            mse_gain > 0.0 and mae_gain > 0.0
            and mse_retention is not None and mse_retention >= 0.50
            and mae_retention is not None and mae_retention >= 0.50
            and formal.get("test_split_accessed") is False
        )
        comparison = {
            "pg95_validation": {"mse": pg95_mse, "mae": pg95_mae},
            "cnm_se_teacher_validation": {"mse": teacher_mse, "mae": teacher_mae},
            "bf_cmvd_student_validation": {"mse": student_mse, "mae": student_mae},
            "student_gain_vs_pg95_percent": {"mse": mse_gain, "mae": mae_gain},
            "teacher_gain_vs_pg95_percent": {
                "mse": teacher_mse_gain, "mae": teacher_mae_gain,
            },
            "teacher_gain_retention_fraction": {
                "mse": mse_retention, "mae": mae_retention,
            },
            "validation_gate_pass": gate_pass,
        }
        decision = (
            "BF_CMVD_SEED2028_VALIDATION_AUTHORIZED"
            if gate_pass else "BF_CMVD_SEED2027_STOP_OR_REDESIGN"
        )
    report = {
        "step": "22.26",
        "queue_state": state.get("queue_state"),
        "smoke_run": smoke_run,
        "formal_run": formal_run,
        "scientific_dependency": state["jobs"][FORMAL_JOB].get("scientific_dependency"),
        "smoke_mechanism": {
            key: smoke.get(key) for key in (
                "bf_cmvd_diagnostic_steps", "bf_cmvd_beneficial_samples",
                "bf_cmvd_total_samples", "bf_cmvd_beneficial_fraction",
                "bf_cmvd_distill_l1_mean", "bf_cmvd_teacher_delta_energy_mean",
                "mcpr_gradient_norm_max", "mcpr_residual_energy_mean",
            )
        } if smoke else {},
        "formal_mechanism": {
            key: formal.get(key) for key in (
                "bf_cmvd_diagnostic_steps", "bf_cmvd_beneficial_fraction",
                "bf_cmvd_distill_l1_mean", "bf_cmvd_teacher_delta_energy_mean",
                "mcpr_gradient_norm_max", "mcpr_residual_energy_mean",
            )
        } if formal else {},
        "comparison": comparison,
        "test_sources_accessed": False,
        "decision": decision,
    }
    atomic_text(REPORT_JSON, json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# 步骤22.26 BF-CMVD结果", "",
        f"- 决策：`{decision}`",
        f"- 队列：`{state.get('queue_state')}`",
        f"- test来源被访问：`{report['test_sources_accessed']}`",
        f"- smoke获益样本比例：`{report['smoke_mechanism'].get('bf_cmvd_beneficial_fraction', 'NA')}`",
        f"- formal状态：`{state['jobs'][FORMAL_JOB].get('state')}`",
    ]
    if comparison:
        lines += [
            f"- 学生相对PG95 MSE提升：{comparison['student_gain_vs_pg95_percent']['mse']:.6f}%",
            f"- 学生相对PG95 MAE提升：{comparison['student_gain_vs_pg95_percent']['mae']:.6f}%",
            f"- MSE教师收益保留率：{comparison['teacher_gain_retention_fraction']['mse']:.6f}",
            f"- MAE教师收益保留率：{comparison['teacher_gain_retention_fraction']['mae']:.6f}",
        ]
    atomic_text(REPORT_MD, "\n".join(lines) + "\n")
    print(json.dumps({"decision": decision, "queue_state": state.get("queue_state")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
