#!/usr/bin/env python3
"""Aggregate the frozen Step 22.29 DMRB failure attribution."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from analyze_stage22_29_dmrb_failure import analyze_arrays


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime/stage22_29_dmrb_diagnostic/state.json"
REPORT_JSON = ROOT / "reports/stage22_29_dmrb_failure_attribution.json"
REPORT_MD = ROOT / "reports/stage22_29_dmrb_failure_attribution.md"
FORMAL_JOB = "dmrb_failure_attribution_seed2027_formal"


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    if not STATE.is_file():
        return 0
    state = json.loads(STATE.read_text(encoding="utf-8"))
    job = state["jobs"][FORMAL_JOB]
    report = {
        "step": "22.29",
        "queue_state": state.get("queue_state"),
        "decision": "DMRB_DIAGNOSTIC_INCOMPLETE",
        "formal_run": job.get("run_dir"),
        "scientific_dependency": job.get("scientific_dependency"),
        "test_sources_accessed": False,
    }
    if job.get("state") == "BLOCKED_DEPENDENCY":
        report["decision"] = "DMRB_DIAGNOSTIC_SMOKE_GATE_FAIL"
    elif job.get("state") == "SUCCESS" and job.get("run_dir"):
        run_dir = Path(job["run_dir"]).resolve()
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        with np.load(metrics["samples_path"], allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        checks = metrics.get("artifact_checks", {})
        protocol_pass = bool(
            metrics.get("artifact_status") == "PASS"
            and checks and all(checks.values())
            and metrics.get("test_split_accessed") is False
            and int(metrics.get("validation_samples_processed", 0)) == 1648
        )
        report = analyze_arrays(metrics, arrays, protocol_pass)
        report.update({
            "queue_state": state.get("queue_state"),
            "formal_run": str(run_dir),
            "scientific_dependency": job.get("scientific_dependency"),
        })
    atomic_text(REPORT_JSON, json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Step 22.29 DMRB 失效归因", "",
        f"- 决策：`{report['decision']}`",
        f"- 队列：`{report.get('queue_state')}`",
        f"- test来源访问：`{report.get('test_sources_accessed', False)}`",
    ]
    if "metric_table" in report:
        table = report["metric_table"]
        supervision = report["supervision_alignment"]
        expression = report["branch_expression"]
        lines += [
            f"- p0 MSE/MAE：{table['backbone']['mse']:.9f} / {table['backbone']['mae']:.9f}",
            f"- DMRB MSE/MAE：{table['student_joint']['mse']:.9f} / {table['student_joint']['mae']:.9f}",
            f"- P(B5|Bjoint) / P(B20|Bjoint)：{supervision['p_b5_given_bjoint']:.6f} / {supervision['p_b20_given_bjoint']:.6f}",
            f"- 教师分量余弦 / 抵消率：{supervision['teacher_component_cosine']:.6f} / {supervision['teacher_cancellation_ratio']:.6f}",
            f"- L5自目标余弦/投影/失配：{expression['l5']['cosine']:.6f} / {expression['l5']['projection']:.6f} / {expression['l5']['normalized_mismatch']:.6f}",
            f"- L20自目标余弦/投影/失配：{expression['l20']['cosine']:.6f} / {expression['l20']['projection']:.6f} / {expression['l20']['normalized_mismatch']:.6f}",
        ]
    atomic_text(REPORT_MD, "\n".join(lines) + "\n")
    if state.get("queue_state") in {"SUCCESS", "FAILED", "STOPPED"}:
        import aggregate_results
        aggregate_results.main()
    print(json.dumps({
        "decision": report["decision"],
        "queue_state": state.get("queue_state"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
