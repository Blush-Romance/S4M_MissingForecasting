#!/usr/bin/env python3
"""Static queue checks for the Stage22 MRRA-v2 recovery branch."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_stage22_mrra_v2_queue as queue  # noqa: E402


def main() -> int:
    jobs = {job["id"]: job for job in queue.scheduler.JOB_DEFINITIONS}
    smoke_id = "mrra_v2_pg95_electricity_seed2023_stability_smoke"
    formal_id = "mrra_v2_pg95_electricity_seed2023_formal"
    baseline_id = "pg95_default_off_post_v2_regression_smoke"
    checks = {
        "three_job_queue": set(jobs) == {baseline_id, smoke_id, formal_id},
        "formal_depends_on_stability_smoke": jobs[formal_id]["depends_on"] == [smoke_id],
        "stability_smoke_depends_on_default_off_regression": (
            jobs[smoke_id]["depends_on"] == [baseline_id]
        ),
        "baseline_has_no_dependency": jobs[baseline_id]["depends_on"] == [],
        "configs_exist": all((ROOT / job["config"]).is_file() for job in jobs.values()),
        "queue_runtime_isolated": queue.scheduler.RUNTIME_ROOT == ROOT / "runtime" / "stage22_mrra_v2",
        "queue_uses_v2_aggregator": queue.scheduler.AGGREGATOR.name == "aggregate_stage22_mrra_v2_results.py",
        "memory_estimates_preserve_formal_shape": all(
            int(job["required_memory_estimate_mib"]) == 13312
            for job in jobs.values()
        ),
    }
    report = {
        "step": 22,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    path = ROOT / "reports" / "stage22_mrra_v2_queue_test.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
