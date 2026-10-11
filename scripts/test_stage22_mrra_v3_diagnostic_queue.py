#!/usr/bin/env python3
"""Static queue checks for the Stage22.14 MRRA-v3 diagnosis."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_stage22_mrra_v3_diagnostic_queue as queue  # noqa: E402


def main() -> int:
    jobs = {job["id"]: job for job in queue.scheduler.JOB_DEFINITIONS}
    smoke = "mrra_v3_electricity_seed2023_validation_smoke"
    formal = "mrra_v3_electricity_seed2023_validation_formal"
    checks = {
        "two_job_queue": set(jobs) == {smoke, formal},
        "smoke_has_no_dependency": jobs[smoke]["depends_on"] == [],
        "formal_depends_on_smoke": jobs[formal]["depends_on"] == [smoke],
        "configs_exist": all((ROOT / job["config"]).is_file() for job in jobs.values()),
        "runtime_isolated": (
            queue.scheduler.RUNTIME_ROOT
            == ROOT / "runtime" / "stage22_mrra_v3_diagnostic"
        ),
        "uses_frozen_cache_runner": (
            queue.scheduler.RUNNER.name == "run_stage12_validation_cache.py"
        ),
        "uses_diagnostic_aggregator": (
            queue.scheduler.AGGREGATOR.name
            == "aggregate_stage22_mrra_v3_diagnostic.py"
        ),
        "memory_estimates_are_safe": all(
            int(job["required_memory_estimate_mib"]) == 6144
            for job in jobs.values()
        ),
    }
    report = {
        "step": "22.14",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    path = ROOT / "reports" / "stage22_mrra_v3_diagnostic_queue_test.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
