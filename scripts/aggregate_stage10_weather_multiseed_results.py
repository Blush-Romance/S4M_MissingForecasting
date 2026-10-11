#!/usr/bin/env python3
"""Aggregate Step-10 queue state and run paired statistics after success."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime" / "stage10_weather_multiseed_queue" / "state.json"
REPORT_JSON = ROOT / "reports" / "stage10_weather_multiseed_raw_results.json"
REPORT_MD = ROOT / "reports" / "stage10_weather_multiseed_raw_results.md"


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
    for job_id, job in sorted(state.get("jobs", {}).items()):
        record: dict[str, Any] = {
            "job_id": job_id,
            "state": job.get("state"),
            "attempt_count": job.get("attempt_count"),
            "run_dir": job.get("run_dir"),
        }
        if job.get("run_dir"):
            metrics_path = Path(job["run_dir"]) / "metrics.json"
            if metrics_path.is_file():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                for key in (
                    "mse",
                    "mae",
                    "optimizer_steps",
                    "parameter_count",
                    "peak_gpu_memory_mb",
                    "peak_gpu_reserved_mb",
                    "mipp_write_gate_mode",
                    "mipp_write_budget",
                    "mipp_stable_write_acceptance_rate_mean",
                    "checkpoint_load_verified",
                ):
                    record[key] = metrics.get(key)
        records.append(record)

    payload = {
        "queue_state": state.get("queue_state"),
        "queue_finished_at": state.get("finished_at"),
        "records": records,
    }
    atomic_text(REPORT_JSON, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# 步骤10：Weather多seed原始队列结果",
        "",
        "| job | 状态 | attempt | MSE | MAE | 模式 | run |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for record in records:
        lines.append(
            f"| {record['job_id']} | {record.get('state')} | "
            f"{record.get('attempt_count', 'NA')} | {record.get('mse', 'NA')} | "
            f"{record.get('mae', 'NA')} | {record.get('mipp_write_gate_mode', 'NA')} | "
            f"`{record.get('run_dir') or ''}` |"
        )
    atomic_text(REPORT_MD, "\n".join(lines) + "\n")

    formal_ids = [
        f"weather_{method}_seed{seed}_formal"
        for seed in (2024, 2025)
        for method in ("pg95", "bmsa50")
    ]
    jobs = state.get("jobs", {})
    if all(jobs.get(job_id, {}).get("state") == "SUCCESS" for job_id in formal_ids):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "compare_stage10_weather_multiseed.py")],
            cwd=ROOT,
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
