#!/usr/bin/env python3
"""Persist current Step-8 HCUP queue results and analyze formal validation when ready."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime" / "stage8_hcup_queue" / "state.json"
REPORT_JSON = ROOT / "reports" / "stage8_hcup_raw_results.json"
REPORT_MD = ROOT / "reports" / "stage8_hcup_raw_results.md"


def main() -> int:
    if not STATE.is_file():
        return 0
    state = json.loads(STATE.read_text(encoding="utf-8"))
    records = []
    for job_id, job in state.get("jobs", {}).items():
        record = {"job_id": job_id, "state": job.get("state"), "run_dir": job.get("run_dir")}
        if job.get("run_dir"):
            metrics_path = Path(job["run_dir"]) / "metrics.json"
            if metrics_path.is_file():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                record.update(
                    {
                        "artifact_status": metrics.get("artifact_status"),
                        "scope": metrics.get("scope"),
                        "sample_count": metrics.get("sample_count"),
                        "probe_valid_count": metrics.get("probe_valid_count"),
                        "prototype_on_mse": metrics.get("prototype_on_mse"),
                        "routed_mse_threshold0": metrics.get("routed_mse"),
                    }
                )
        records.append(record)
    payload = {"queue_state": state.get("queue_state"), "jobs": records}
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT_JSON.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(REPORT_JSON)
    lines = [
        "# 步骤8：HCUP原始队列结果",
        "",
        "| job | 状态 | 工件 | 样本 | 有效探针 | run |",
        "|---|---|---|---:|---:|---|",
    ]
    for record in records:
        lines.append(
            f"| {record['job_id']} | {record.get('state')} | {record.get('artifact_status', 'NA')} | "
            f"{record.get('sample_count', 'NA')} | {record.get('probe_valid_count', 'NA')} | "
            f"`{record.get('run_dir') or ''}` |"
        )
    md_temp = REPORT_MD.with_suffix(".md.tmp")
    md_temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    md_temp.replace(REPORT_MD)
    formal = state.get("jobs", {}).get("weather_hcup_validation_formal", {})
    if formal.get("state") == "SUCCESS" and formal.get("run_dir"):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "analyze_hcup_validation.py")],
            cwd=ROOT,
            check=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
