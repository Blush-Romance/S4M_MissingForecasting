#!/usr/bin/env python3
"""Aggregate MCPR attribution caches and run the frozen paired analysis."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import aggregate_results


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime" / "stage22_mcpr_attribution" / "state.json"
REPORT = ROOT / "reports" / "stage22_mcpr_attribution_raw.json"
FORMAL_JOBS = {
    "mcpr_mse": "mcpr_mse_attribution_formal",
    "mcpr_pcd": "mcpr_pcd_attribution_formal",
}
PG95_RUN = ROOT / "runs" / "stage22_valcache_pg95__electricity__validation__seed2023__20260805_164843"


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    aggregate_results.main()
    if not STATE.is_file():
        print("stage22_mcpr_attribution=WAITING_STATE")
        return 0
    state = json.loads(STATE.read_text(encoding="utf-8"))
    jobs = state.get("jobs", {})
    payload = {
        "queue_state": state.get("queue_state"),
        "formal": {
            method: {
                "state": jobs.get(job_id, {}).get("state"),
                "run_dir": jobs.get(job_id, {}).get("run_dir"),
                "attempt_count": jobs.get(job_id, {}).get("attempt_count", 0),
            }
            for method, job_id in FORMAL_JOBS.items()
        },
    }
    atomic_text(REPORT, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not all(
        jobs.get(job_id, {}).get("state") == "SUCCESS"
        and jobs.get(job_id, {}).get("run_dir")
        for job_id in FORMAL_JOBS.values()
    ):
        print("stage22_mcpr_attribution=WAITING_FORMALS")
        return 0
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analyze_stage22_mcpr_failure.py"),
            "--pg95-run", str(PG95_RUN),
            "--mcpr-mse-run", str(jobs[FORMAL_JOBS["mcpr_mse"]]["run_dir"]),
            "--mcpr-pcd-run", str(jobs[FORMAL_JOBS["mcpr_pcd"]]["run_dir"]),
        ],
        cwd=ROOT,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
