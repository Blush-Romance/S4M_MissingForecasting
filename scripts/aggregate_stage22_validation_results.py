#!/usr/bin/env python3
"""Aggregate Stage-22 jobs and run the paired diagnosis when both formals finish."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime" / "stage22_validation_queue" / "state.json"
REPORT_JSON = ROOT / "reports" / "stage22_validation_raw_results.json"
REPORT_MD = ROOT / "reports" / "stage22_validation_raw_results.md"


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    if not STATE.is_file():
        return 0
    state = json.loads(STATE.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    metric_keys = (
        "method", "scope", "sample_count", "complete_target_mse_original",
        "complete_target_mae_original", "prediction_disagreement_mse_mean",
        "prediction_exact_equal_sample_fraction", "peak_gpu_memory_mb",
        "artifact_status",
    )
    for job_id, job in sorted(state.get("jobs", {}).items()):
        record: dict[str, Any] = {
            "job_id": job_id,
            "state": job.get("state"),
            "attempt_count": job.get("attempt_count", 0),
            "run_dir": job.get("run_dir"),
        }
        if job.get("run_dir"):
            metrics_path = Path(job["run_dir"]) / "metrics.json"
            if metrics_path.is_file():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                for key in metric_keys:
                    record[key] = metrics.get(key)
        records.append(record)
    payload = {
        "queue_state": state.get("queue_state"),
        "queue_finished_at": state.get("finished_at"),
        "records": records,
    }
    atomic_text(REPORT_JSON, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# 步骤22：Electricity验证缓存原始结果",
        "",
        "| job | 状态 | samples | original MSE | original MAE | disagreement | run |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for record in records:
        lines.append(
            f"| {record['job_id']} | {record.get('state')} | "
            f"{record.get('sample_count', 'NA')} | "
            f"{record.get('complete_target_mse_original', 'NA')} | "
            f"{record.get('complete_target_mae_original', 'NA')} | "
            f"{record.get('prediction_disagreement_mse_mean', 'NA')} | "
            f"`{record.get('run_dir') or ''}` |"
        )
    atomic_text(REPORT_MD, "\n".join(lines) + "\n")

    jobs = state.get("jobs", {})
    ready = all(
        jobs.get(f"electricity_{method}_seed2023_validation_formal", {}).get("state") == "SUCCESS"
        for method in ("s4m", "pg95")
    )
    if ready:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "analyze_stage22_validation_tradeoff.py")],
            cwd=ROOT,
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
