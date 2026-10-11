#!/usr/bin/env python3
"""Aggregate Stage-15 CNM-SE jobs and run the frozen gate."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime" / "stage15_cnm_se_queue" / "state.json"
REPORT_JSON = ROOT / "reports" / "stage15_cnm_se_raw_results.json"
REPORT_MD = ROOT / "reports" / "stage15_cnm_se_raw_results.md"


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
        "evaluation_seed", "sample_count", "complete_target_mse_on",
        "complete_target_mae_on", "complete_target_mse_nested",
        "complete_target_mae_nested", "complete_target_mse_ensemble",
        "complete_target_mae_ensemble", "evaluation_elapsed_seconds",
        "inference_forward_multiplier", "artifact_status", "peak_gpu_memory_mb",
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
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                record.update({key: metrics.get(key) for key in metric_keys})
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
        "# 步骤15：CNM-SE原始结果",
        "",
        "| job | 状态 | PG95 MSE | CNM-SE MSE | PG95 MAE | CNM-SE MAE | run |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for record in records:
        lines.append(
            f"| {record['job_id']} | {record.get('state')} | "
            f"{record.get('complete_target_mse_on', 'NA')} | "
            f"{record.get('complete_target_mse_ensemble', 'NA')} | "
            f"{record.get('complete_target_mae_on', 'NA')} | "
            f"{record.get('complete_target_mae_ensemble', 'NA')} | "
            f"`{record.get('run_dir') or ''}` |"
        )
    atomic_text(REPORT_MD, "\n".join(lines) + "\n")
    jobs = state.get("jobs", {})
    ready = all(
        jobs.get(f"weather_cnm_se_seed{seed}_formal", {}).get("state") == "SUCCESS"
        for seed in (2024, 2025)
    )
    if ready:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "analyze_stage15_cnm_se.py")],
            cwd=ROOT,
            check=True,
        )
    three_seed_ready = all(
        jobs.get(f"weather_cnm_se_seed{seed}_formal", {}).get("state") == "SUCCESS"
        for seed in (2023, 2024, 2025)
    )
    if three_seed_ready:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_stage16_cnm_se_three_seed.py"),
            ],
            cwd=ROOT,
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
