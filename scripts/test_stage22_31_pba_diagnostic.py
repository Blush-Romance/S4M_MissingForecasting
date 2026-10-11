#!/usr/bin/env python3
"""Zero-GPU contracts for Step22.31 PBA failure attribution."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_stage22_31_pba_failure as analyzer  # noqa: E402
import run_stage22_31_pba_diagnostic as runner  # noqa: E402
import run_stage22_31_pba_queue as queue  # noqa: E402


def expect_failure(function, *args) -> bool:
    try:
        function(*args)
    except (ValueError, RuntimeError):
        return True
    return False


def gap(inside: float, outside: float, harm: float = -0.01) -> dict:
    return {
        "beneficial_positive_gap_share": inside,
        "nonbeneficial_positive_gap_share": outside,
        "beneficial_region": {"student_minus_backbone_mean": -0.01},
        "nonbeneficial_region": {"student_minus_backbone_mean": harm},
    }


def geometry(cosine: float, projection: float, mismatch: float) -> dict:
    return {
        "cosine": cosine,
        "projection_coefficient": projection,
        "normalized_mismatch": mismatch,
    }


def tail(trigger: bool) -> dict:
    return {
        "aggregate_pba_minus_bf_mse": 0.01 if trigger else -0.01,
        "aggregate_pba_minus_bf_mae": -0.01,
        "tail_positive_mse_excess_share": 0.70,
        "tail_pba_minus_bf_mse_mean": 0.02,
        "body_pba_minus_bf_mae_mean": -0.01,
    }


def main() -> int:
    configs = sorted((ROOT / "configs/stage22_31_pba_diagnostic").glob("*.yaml"))
    pba_resolved = yaml.safe_load((
        ROOT / "runs/pba_cmvd_pg95_traffic__traffic__point__mr0.06__seed2027__20260826_222702/resolved_config.yaml"
    ).read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    for path in configs:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        runner.validate_config(config, pba_resolved)
    checks["two_configs_validate"] = len(configs) == 2

    formal = yaml.safe_load(configs[0].read_text(encoding="utf-8"))
    if formal["experiment"]["smoke_test"]:
        formal = yaml.safe_load(configs[1].read_text(encoding="utf-8"))
    wrong_tail = copy.deepcopy(formal)
    wrong_tail["diagnostic"]["bf_tail_quantile"] = 0.95
    checks["wrong_tail_rejected"] = expect_failure(
        runner.validate_config, wrong_tail, pba_resolved
    )
    wrong_split = copy.deepcopy(formal)
    wrong_split["diagnostic"]["split"] = "test"
    checks["test_split_rejected"] = expect_failure(
        runner.validate_config, wrong_split, pba_resolved
    )
    wrong_weights = copy.deepcopy(formal)
    wrong_weights["diagnostic"]["teacher_weights"]["l20"] = 0.30
    checks["teacher_weight_change_rejected"] = expect_failure(
        runner.validate_config, wrong_weights, pba_resolved
    )
    no_pba = copy.deepcopy(pba_resolved)
    no_pba["official_args"]["use_pba_cmvd"] = False
    checks["pba_identity_required"] = expect_failure(
        runner.validate_config, formal, no_pba
    )
    with_mcpr = copy.deepcopy(pba_resolved)
    with_mcpr["official_args"]["use_mcpr"] = True
    checks["mcpr_conflict_rejected"] = expect_failure(
        runner.validate_config, formal, with_mcpr
    )

    reference_values = np.linspace(0.01, 5.0, 20000, dtype=np.float32)
    fp32_reordered = reference_values + (
        0.25 * runner.FP32_CROSS_PATH_TOL * (1.0 + reference_values)
    ).astype(np.float32)
    checks["fp32_scale_aware_positive"] = runner.fp32_cross_path_close(
        fp32_reordered, reference_values
    )
    corrupted_values = reference_values.copy()
    corrupted_values[100] += 0.01
    checks["fp32_asset_mismatch_rejected"] = not runner.fp32_cross_path_close(
        corrupted_values, reference_values
    )
    boundary_reference = {
        "backbone_mse": np.ones(20000, dtype=np.float32),
        "teacher_mse": np.full(20000, 0.9, dtype=np.float32),
        "backbone_mae": np.ones(20000, dtype=np.float32),
        "teacher_mae": np.full(20000, 0.9, dtype=np.float32),
    }
    boundary_current = {
        key: value.copy() for key, value in boundary_reference.items()
    }
    boundary_reference["teacher_mse"][0] = 1.0 - 1e-6
    boundary_current["teacher_mse"][0] = 1.0 + 1e-6
    stable, stable_diag = runner.benefit_boundary_stable(
        boundary_current, boundary_reference
    )
    checks["near_zero_benefit_flip_allowed"] = bool(
        stable and stable_diag["disagreement_count"] == 1
    )
    nonboundary_current = {
        key: value.copy() for key, value in boundary_reference.items()
    }
    nonboundary_reference = {
        key: value.copy() for key, value in boundary_reference.items()
    }
    nonboundary_reference["teacher_mse"][0] = 0.99
    nonboundary_current["teacher_mse"][0] = 1.01
    stable, _ = runner.benefit_boundary_stable(
        nonboundary_current, nonboundary_reference
    )
    checks["nonboundary_benefit_flip_rejected"] = not stable
    excessive_current = {
        key: value.copy() for key, value in boundary_reference.items()
    }
    excessive_current["teacher_mse"][:3] = 1.0 + 1e-6
    excessive_reference = {
        key: value.copy() for key, value in boundary_reference.items()
    }
    excessive_reference["teacher_mse"][:3] = 1.0 - 1e-6
    stable, _ = runner.benefit_boundary_stable(
        excessive_current, excessive_reference
    )
    checks["excessive_benefit_flip_rate_rejected"] = not stable

    quiet_gap = gap(0.50, 0.50)
    weak_geometry = geometry(0.2, 0.2, 1.0)
    good_geometry = geometry(0.8, 0.8, 0.2)
    tail_decision = analyzer.decide_diagnosis(
        tail(True), gap(0.80, 0.20), gap(0.80, 0.20),
        weak_geometry, 0.10, True,
    )
    checks["tail_decision"] = (
        tail_decision["decision"]
        == "TAIL_RISK_AMPLIFICATION_FREEZE_CVAR_TR_PBA"
    )
    spillover_decision = analyzer.decide_diagnosis(
        tail(False), gap(0.20, 0.80, 0.01), gap(0.20, 0.80, 0.01),
        good_geometry, 0.40, True,
    )
    checks["spillover_decision"] = (
        spillover_decision["decision"]
        == "NONBENEFICIAL_DRIFT_FREEZE_BTR_PBA"
    )
    direction_decision = analyzer.decide_diagnosis(
        tail(False), gap(0.80, 0.20), gap(0.80, 0.20),
        weak_geometry, 0.10, True,
    )
    checks["direction_decision"] = (
        direction_decision["decision"]
        == "TEACHER_DIRECTION_UNDER_AMORTIZED_FREEZE_HFD_PBA"
    )
    stop_decision = analyzer.decide_diagnosis(
        tail(False), quiet_gap, quiet_gap, good_geometry, 0.10, True,
    )
    checks["indeterminate_stops"] = (
        stop_decision["decision"] == "MIXED_OR_INDETERMINATE_STOP_PBA_ROUTE"
    )
    protocol_decision = analyzer.decide_diagnosis(
        tail(True), gap(0.80, 0.20), gap(0.80, 0.20),
        weak_geometry, 0.10, False,
    )
    checks["protocol_failure_blocks_decision"] = (
        protocol_decision["decision"] == "PROTOCOL_FAILURE_NO_DECISION"
    )
    checks["tail_precedence"] = (
        tail_decision["preregistered_triggers"]["tail_risk_amplification"]
        and tail_decision["preregistered_triggers"][
            "teacher_direction_under_amortized"
        ]
        and tail_decision["main_cause"] == "tail_risk_amplification"
    )

    checks["queue_two_jobs"] = len(queue.scheduler.JOB_DEFINITIONS) == 2
    checks["queue_dependency"] = (
        queue.scheduler.JOB_DEFINITIONS[1]["depends_on"] == [queue.SMOKE_JOB]
    )
    checks["queue_memory_estimate_recorded"] = all(
        int(job["required_memory_estimate_mib"]) == 8192
        for job in queue.scheduler.JOB_DEFINITIONS
    )
    checks["runner_exact_token_contract"] = (
        "source_error_correction_dot" in runner.TOKEN_KEYS
        and len(runner.TOKEN_KEYS) == 21
    )
    protocol_text = (
        ROOT / "docs/STAGE22_31_PBA_FAILURE_ATTRIBUTION_PROTOCOL_ZH.md"
    ).read_text(encoding="utf-8")
    checks["protocol_has_q90_and_60pct"] = (
        "90%" in protocol_text and "至少 60%" in protocol_text
    )
    checks["protocol_forbids_test"] = "不访问 test" in protocol_text
    checks["protocol_records_fp32_amendment"] = (
        "512×float32 epsilon" in protocol_text and "旧 formal 保持" in protocol_text
    )
    checks["all_source_assets_exist"] = all(
        runner.project_path(value).is_file()
        for value in formal["source"].values()
    )
    checks["all_data_assets_exist"] = all(
        runner.project_path(value).is_file()
        for value in formal["data"].values()
    )

    output = {
        "step": "22.31",
        "check_count": len(checks),
        "checks": checks,
        "all_pass": all(checks.values()),
    }
    report_path = ROOT / "reports/stage22_31_component_test.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if output["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
