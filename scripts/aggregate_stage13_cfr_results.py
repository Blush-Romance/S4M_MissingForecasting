#!/usr/bin/env python3
"""Aggregate Step-13 CFR queue state and compare the formal result."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime" / "stage13_cfr_queue" / "state.json"
REPORT_JSON = ROOT / "reports" / "stage13_cfr_raw_results.json"
REPORT_MD = ROOT / "reports" / "stage13_cfr_raw_results.md"


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
        record = {"job_id": job_id, "state": job.get("state"), "run_dir": job.get("run_dir")}
        if job.get("run_dir"):
            metrics_path = Path(job["run_dir"]) / "metrics.json"
            if metrics_path.is_file():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                for key in (
                    "mse", "mae", "cfr_enabled", "cfr_steps", "cfr_samples",
                    "cfr_positive_regret_mean", "cfr_harmful_retrieval_rate_mean",
                    "optimizer_steps", "checkpoint_load_verified", "peak_gpu_memory_mb",
                ):
                    record[key] = metrics.get(key)
        records.append(record)
    payload = {"queue_state": state.get("queue_state"), "records": records}
    atomic_text(REPORT_JSON, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# 步骤13：CFR原始队列结果", "",
        "| job | 状态 | MSE | MAE | CFR | 正后悔 | 有害率 | run |",
        "|---|---|---:|---:|---|---:|---:|---|",
    ]
    for record in records:
        lines.append(
            f"| {record['job_id']} | {record.get('state')} | {record.get('mse', 'NA')} | "
            f"{record.get('mae', 'NA')} | {record.get('cfr_enabled', 'NA')} | "
            f"{record.get('cfr_positive_regret_mean', 'NA')} | "
            f"{record.get('cfr_harmful_retrieval_rate_mean', 'NA')} | "
            f"`{record.get('run_dir') or ''}` |"
        )
    atomic_text(REPORT_MD, "\n".join(lines) + "\n")
    formal = state.get("jobs", {}).get("weather_pg95_cfr_seed2023_formal", {})
    if formal.get("state") == "SUCCESS" and formal.get("run_dir"):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "compare_stage13_cfr_results.py")],
            cwd=ROOT,
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
