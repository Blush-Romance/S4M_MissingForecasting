#!/usr/bin/env python3
"""Aggregate Stage-21 Electricity queue artifacts and validation decision."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime" / "stage21_electricity_pipeline" / "state.json"
REPORT_JSON = ROOT / "reports" / "stage21_electricity_raw_results.json"
REPORT_MD = ROOT / "reports" / "stage21_electricity_raw_results.md"
REQUIRED_VALIDATION_JOBS = (
    "electricity_s4m_seed2023_formal",
    "electricity_pg95_seed2023_formal",
    "electricity_pg95_multiscale_validation_seed2023_formal",
)


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    if not STATE.is_file():
        return 0
    state = json.loads(STATE.read_text(encoding="utf-8"))
    metric_keys = (
        "scope", "mse", "mae", "evaluation_split", "test_split_accessed",
        "validation_batches_processed", "validation_samples_processed",
        "complete_target_mse_on", "complete_target_mae_on",
        "complete_target_mse_ensemble_l5",
        "complete_target_mae_ensemble_l5",
        "complete_target_mse_multiscale",
        "complete_target_mae_multiscale",
        "parameter_count", "peak_gpu_memory_mb", "artifact_status",
        "test_accessed", "test_authorized", "scientific_decision",
    )
    records = []
    for job_id, job in sorted(state.get("jobs", {}).items()):
        record = {
            "job_id": job_id,
            "state": job.get("state"),
            "run_dir": job.get("run_dir"),
        }
        if job.get("run_dir"):
            metrics_path = Path(job["run_dir"]) / "metrics.json"
            if metrics_path.is_file():
                metrics = json.loads(
                    metrics_path.read_text(encoding="utf-8")
                )
                record.update(
                    {key: metrics.get(key) for key in metric_keys}
                )
        records.append(record)
    atomic_text(
        REPORT_JSON,
        json.dumps(
            {"queue_state": state.get("queue_state"), "records": records},
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    lines = [
        "# 步骤21：Electricity原始结果",
        "",
        "| job | 状态 | split | MSE | MAE | 多尺度MSE | 多尺度MAE | test访问 |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for record in records:
        lines.append(
            f"| {record['job_id']} | {record.get('state')} | "
            f"{record.get('evaluation_split', 'NA')} | "
            f"{record.get('mse', record.get('complete_target_mse_on', 'NA'))} | "
            f"{record.get('mae', record.get('complete_target_mae_on', 'NA'))} | "
            f"{record.get('complete_target_mse_multiscale', 'NA')} | "
            f"{record.get('complete_target_mae_multiscale', 'NA')} | "
            f"{record.get('test_accessed', record.get('test_split_accessed', 'NA'))} |"
        )
    atomic_text(REPORT_MD, "\n".join(lines) + "\n")

    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "aggregate_results.py")],
        cwd=ROOT,
        check=False,
    )
    if all(
        state.get("jobs", {}).get(job_id, {}).get("state") == "SUCCESS"
        for job_id in REQUIRED_VALIDATION_JOBS
    ):
        subprocess.run(
            [
                sys.executable,
                str(
                    ROOT
                    / "scripts"
                    / "analyze_stage21_electricity_validation.py"
                ),
            ],
            cwd=ROOT,
            check=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
