#!/usr/bin/env python3
"""Aggregate Step-9 BMSA queue state and run the final comparison when ready."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime" / "stage9_bmsa_queue" / "state.json"
REPORT_JSON = ROOT / "reports" / "stage9_bmsa_raw_results.json"
REPORT_MD = ROOT / "reports" / "stage9_bmsa_raw_results.md"


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
            "run_dir": job.get("run_dir"),
        }
        if job.get("run_dir"):
            metrics_path = Path(job["run_dir"]) / "metrics.json"
            if metrics_path.is_file():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                for key in (
                    "mse",
                    "mae",
                    "mipp_write_gate_mode",
                    "mipp_write_budget",
                    "mipp_stable_write_acceptance_rate_mean",
                    "mipp_write_eligible_count_mean",
                    "mipp_write_sampled_count_mean",
                    "mipp_write_updated_count_mean",
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
        "# 步骤9：BMSA原始队列结果",
        "",
        "| job | 状态 | MSE | MAE | 模式 | 预算 | 接受率 | run |",
        "|---|---|---:|---:|---|---:|---:|---|",
    ]
    for record in records:
        lines.append(
            f"| {record['job_id']} | {record.get('state')} | "
            f"{record.get('mse', 'NA')} | {record.get('mae', 'NA')} | "
            f"{record.get('mipp_write_gate_mode', 'NA')} | "
            f"{record.get('mipp_write_budget', 'NA')} | "
            f"{record.get('mipp_stable_write_acceptance_rate_mean', 'NA')} | "
            f"`{record.get('run_dir') or ''}` |"
        )
    atomic_text(REPORT_MD, "\n".join(lines) + "\n")

    formal = state.get("jobs", {}).get("weather_bmsa50_formal", {})
    if formal.get("state") == "SUCCESS" and formal.get("run_dir"):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "compare_stage9_bmsa_results.py")],
            cwd=ROOT,
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
