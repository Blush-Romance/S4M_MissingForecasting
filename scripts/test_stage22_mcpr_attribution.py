#!/usr/bin/env python3
"""CPU/static gates for Stage22.17 MCPR attribution."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_stage22_mcpr_failure as analysis  # noqa: E402
import run_stage22_mcpr_attribution_queue as queue  # noqa: E402
import run_stage12_validation_cache as cache  # noqa: E402


def main() -> int:
    config_paths = sorted((ROOT / "configs" / "stage22_mcpr_attribution").glob("*.yaml"))
    config_checks = []
    for path in config_paths:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        source = yaml.safe_load(
            (ROOT / config["source"]["resolved_config"]).read_text(encoding="utf-8")
        )
        cache.validate_config(config, source)
        config_checks.append(True)

    negative = yaml.safe_load(config_paths[0].read_text(encoding="utf-8"))
    negative_source = yaml.safe_load(
        (ROOT / negative["source"]["resolved_config"]).read_text(encoding="utf-8")
    )
    negative_source = copy.deepcopy(negative_source)
    negative_source["official_args"]["mcpr_objective"] = "wrong"
    negative_rejected = False
    try:
        cache.validate_config(negative, negative_source)
    except ValueError:
        negative_rejected = True

    rng = np.random.default_rng(2023)
    backbone_error = rng.normal(size=(8, 96, 7))
    residual = rng.normal(scale=0.1, size=backbone_error.shape)
    delta = (backbone_error + residual) ** 2 - backbone_error ** 2
    cross = 2.0 * backbone_error * residual
    energy = residual ** 2
    decomposition_closes = bool(np.allclose(delta, cross + energy, atol=1e-12))
    reduction_tolerance_accepts_measured_float32_gap = bool(np.isclose(
        1.0, 1.0 + 0.00635,
        rtol=cache.MCPR_REDUCTION_RTOL,
        atol=cache.MCPR_REDUCTION_ATOL,
    ))
    reduction_tolerance_rejects_material_gap = not bool(np.isclose(
        1.0, 1.0 + 0.02,
        rtol=cache.MCPR_REDUCTION_RTOL,
        atol=cache.MCPR_REDUCTION_ATOL,
    ))

    jobs = {job["id"]: job for job in queue.scheduler.JOB_DEFINITIONS}
    queue_checks = {
        "four_jobs": len(jobs) == 4,
        "smokes_independent": all(
            jobs[name]["depends_on"] == []
            for name in ("mcpr_mse_attribution_smoke", "mcpr_pcd_attribution_smoke")
        ),
        "formals_depend_on_matching_smoke": (
            jobs["mcpr_mse_attribution_formal"]["depends_on"]
            == ["mcpr_mse_attribution_smoke"]
            and jobs["mcpr_pcd_attribution_formal"]["depends_on"]
            == ["mcpr_pcd_attribution_smoke"]
        ),
        "configs_exist": all((ROOT / job["config"]).is_file() for job in jobs.values()),
        "isolated_runtime": queue.scheduler.RUNTIME_ROOT == ROOT / "runtime" / "stage22_mcpr_attribution",
        "uses_cache_runner": queue.scheduler.RUNNER.name == "run_stage12_validation_cache.py",
        "uses_attribution_aggregator": queue.scheduler.AGGREGATOR.name == "aggregate_stage22_mcpr_attribution.py",
        "memory_estimates": all(int(job["required_memory_estimate_mib"]) == 6144 for job in jobs.values()),
    }
    checks = {
        "four_configs": len(config_paths) == 4,
        "all_configs_validate": all(config_checks),
        "wrong_objective_rejected": negative_rejected,
        "exact_decomposition": decomposition_closes,
        "reduction_tolerance_accepts_measured_float32_gap": reduction_tolerance_accepts_measured_float32_gap,
        "reduction_tolerance_rejects_material_gap": reduction_tolerance_rejects_material_gap,
        "direction_failure_rule": analysis.interpretation(0.1, 0.2, 0.3) == "DIRECTION_ALIGNMENT_FAILURE",
        "overshoot_rule": analysis.interpretation(-0.1, 0.2, 0.1) == "AMPLITUDE_OVERSHOOT",
        **queue_checks,
    }
    report = {
        "step": "22.17",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    path = ROOT / "reports" / "stage22_mcpr_attribution_test.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
