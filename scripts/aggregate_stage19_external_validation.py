#!/usr/bin/env python3
"""Aggregate Stage-19 raw state, analysis, and the project-wide result table."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime" / "stage19_external_validation_queue" / "state.json"
REPORT_JSON = ROOT / "reports" / "stage19_external_raw_results.json"
REPORT_MD = ROOT / "reports" / "stage19_external_raw_results.md"


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    if not STATE.is_file():
        return 0
    state = json.loads(STATE.read_text(encoding="utf-8"))
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
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                for key in (
                    "scope", "sample_count", "test_batches_processed",
                    "complete_target_mse_on", "complete_target_mae_on",
                    "complete_target_mse_ensemble_l5",
                    "complete_target_mae_ensemble_l5",
                    "complete_target_mse_multiscale",
                    "complete_target_mae_multiscale",
                    "evaluation_elapsed_seconds", "peak_gpu_memory_mb",
                    "parameter_count", "artifact_status",
                ):
                    record[key] = metrics.get(key)
        records.append(record)
    atomic_text(
        REPORT_JSON,
        json.dumps(
            {"queue_state": state.get("queue_state"), "records": records},
            indent=2,
            sort_keys=True,
        ) + "\n",
    )
    lines = [
        "# 步骤19：ETTh1外部验证原始结果",
        "",
        "| job | 状态 | 样本 | 原始MSE | 多尺度MSE | 原始MAE | 多尺度MAE | run |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        lines.append(
            f"| {record['job_id']} | {record.get('state')} | "
            f"{record.get('sample_count', 'NA')} | "
            f"{record.get('complete_target_mse_on', 'NA')} | "
            f"{record.get('complete_target_mse_multiscale', 'NA')} | "
            f"{record.get('complete_target_mae_on', 'NA')} | "
            f"{record.get('complete_target_mae_multiscale', 'NA')} | "
            f"`{record.get('run_dir') or ''}` |"
        )
    atomic_text(REPORT_MD, "\n".join(lines) + "\n")

    formal = state.get("jobs", {}).get(
        "etth1_equal_rate_multiscale_seed2023_formal", {}
    )
    if formal.get("state") == "SUCCESS":
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_stage19_external_validation.py"),
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "aggregate_results.py")],
            cwd=ROOT,
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
