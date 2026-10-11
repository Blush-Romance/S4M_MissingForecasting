#!/usr/bin/env python3
"""Static contracts for the Stage22.15 smoke-to-formal queue."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_stage22_mrra_internal_queue as queue  # noqa: E402


def main() -> int:
    jobs = {job["id"]: job for job in queue.scheduler.JOB_DEFINITIONS}
    smoke = "mrra_v3_internal_electricity_seed2023_smoke"
    formal = "mrra_v3_internal_electricity_seed2023_formal"
    configs = [yaml.safe_load((ROOT / job["config"]).read_text(encoding="utf-8")) for job in jobs.values()]
    checks = {
        "two_job_queue": set(jobs) == {smoke, formal},
        "formal_depends_on_smoke": jobs[formal]["depends_on"] == [smoke],
        "internal_identity_frozen": all(
            item["diagnostic"]["stage_identity"] == "stage22_internal"
            and item["diagnostic"]["internal_features"] is True
            for item in configs
        ),
        "validation_only": all(item["diagnostic"]["split"] == "val" for item in configs),
        "seed_frozen": all(item["experiment"]["seed"] == 2023 for item in configs),
        "runtime_isolated": queue.scheduler.RUNTIME_ROOT == ROOT / "runtime" / "stage22_mrra_internal",
        "analyzer_registered": queue.scheduler.AGGREGATOR.name == "analyze_stage22_mrra_internal_features.py",
        "memory_estimates_safe": all(job["required_memory_estimate_mib"] == 6144 for job in jobs.values()),
    }
    report = {
        "step": "22.15",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    path = ROOT / "reports" / "stage22_mrra_internal_queue_test.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
