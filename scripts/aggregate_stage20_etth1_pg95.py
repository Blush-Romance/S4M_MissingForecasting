#!/usr/bin/env python3
"""Aggregate Stage-20 queue state, metrics, final analysis, and global results."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime" / "stage20_etth1_pg95_pipeline" / "state.json"
REPORT_JSON = ROOT / "reports" / "stage20_etth1_pg95_raw_results.json"
REPORT_MD = ROOT / "reports" / "stage20_etth1_pg95_raw_results.md"
FINAL_JOB = "etth1_pg95_multiscale_seed2023_formal"


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
    metric_keys = (
        "scope", "mse", "mae", "sample_count", "test_batches_processed",
        "complete_target_mse_on", "complete_target_mae_on",
        "complete_target_mse_ensemble_l5",
        "complete_target_mae_ensemble_l5",
        "complete_target_mse_multiscale",
        "complete_target_mae_multiscale",
        "parameter_count", "peak_gpu_memory_mb", "artifact_status",
    )
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
                for key in metric_keys:
                    record[key] = metrics.get(key)
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
        "# 步骤20：ETTh1 PG95与冻结多尺度原始结果",
        "",
        "| job | 状态 | MSE | MAE | 多尺度MSE | 多尺度MAE | run |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for record in records:
        lines.append(
            f"| {record['job_id']} | {record.get('state')} | "
            f"{record.get('mse', record.get('complete_target_mse_on', 'NA'))} | "
            f"{record.get('mae', record.get('complete_target_mae_on', 'NA'))} | "
            f"{record.get('complete_target_mse_multiscale', 'NA')} | "
            f"{record.get('complete_target_mae_multiscale', 'NA')} | "
            f"`{record.get('run_dir') or ''}` |"
        )
    atomic_text(REPORT_MD, "\n".join(lines) + "\n")

    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "aggregate_results.py")],
        cwd=ROOT,
        check=True,
    )
    final = state.get("jobs", {}).get(FINAL_JOB, {})
    if final.get("state") == "SUCCESS":
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_stage20_etth1_pg95.py"),
            ],
            cwd=ROOT,
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
