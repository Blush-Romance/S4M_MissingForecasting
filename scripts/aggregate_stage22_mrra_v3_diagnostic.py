#!/usr/bin/env python3
"""Aggregate MRRA-v3 validation caches and run the paired diagnosis."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import aggregate_results


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime" / "stage22_mrra_v3_diagnostic" / "state.json"
FORMAL = "mrra_v3_electricity_seed2023_validation_formal"
PG95_RUN = ROOT / "runs" / "stage22_valcache_pg95__electricity__validation__seed2023__20260805_164843"
REPORT = ROOT / "reports" / "stage22_mrra_v3_diagnostic_raw.json"


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    aggregate_results.main()
    if not STATE.is_file():
        print("stage22_mrra_v3_diagnostic=WAITING_STATE")
        return 0
    state = json.loads(STATE.read_text(encoding="utf-8"))
    job = state.get("jobs", {}).get(FORMAL, {})
    payload = {
        "queue_state": state.get("queue_state"),
        "formal_state": job.get("state"),
        "formal_run_dir": job.get("run_dir"),
        "attempt_count": job.get("attempt_count", 0),
    }
    atomic_text(REPORT, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if job.get("state") != "SUCCESS" or not job.get("run_dir"):
        print("stage22_mrra_v3_diagnostic=WAITING_FORMAL")
        return 0
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analyze_stage22_mrra_v3_error.py"),
            "--pg95-run", str(PG95_RUN),
            "--mrra-v3-run", str(job["run_dir"]),
        ],
        cwd=ROOT,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
