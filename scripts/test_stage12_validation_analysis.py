#!/usr/bin/env python3
"""Deterministic component tests for the Stage-12 validation analysis."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from analyze_stage12_validation_mismatch import (
    ROOT,
    equal_quartiles,
    group_comparison,
    rarity_gate_checks,
    summarize_feature,
)


REPORT = ROOT / "reports" / "stage12_validation_analysis_test.json"


def samples(mse: np.ndarray, mae: np.ndarray, count: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "complete_sse": mse * count,
        "complete_sae": mae * count,
        "complete_count": count.copy(),
        "observed_sse": mse * count,
        "observed_sae": mae * count,
        "observed_count": count.copy(),
    }


def main() -> int:
    feature = np.arange(8, dtype=np.float64)
    count = np.asarray([1, 2, 1, 2, 1, 2, 1, 2], dtype=np.float64)
    pg = samples(np.full(8, 4.0), np.full(8, 2.0), count)
    mr_pass = samples(
        np.asarray([4.1, 4.1, 4.0, 4.0, 3.8, 3.8, 3.0, 3.0]),
        np.asarray([2.1, 2.1, 2.0, 2.0, 1.9, 1.9, 1.5, 1.5]),
        count,
    )
    mr_fail = samples(
        np.asarray([4.1, 4.1, 4.0, 4.0, 3.8, 3.8, 3.0, 3.0]),
        np.asarray([2.1, 2.1, 2.0, 2.0, 1.9, 1.9, 2.5, 2.5]),
        count,
    )
    quartiles = equal_quartiles(feature)
    primary_pass = summarize_feature(feature, pg, mr_pass)
    primary_fail = summarize_feature(feature, pg, mr_fail)
    pass_checks = rarity_gate_checks(2024, primary_pass)
    fail_checks = rarity_gate_checks(2025, primary_fail)
    weighted = group_comparison(pg, mr_pass, np.asarray([0, 1]), "complete")
    expected_weighted_mr_mse = (4.1 * 1 + 4.1 * 2) / 3

    checks = {
        "quartiles_equal_size": [len(group) for group in quartiles] == [2, 2, 2, 2],
        "quartiles_stable_extremes": bool(np.array_equal(quartiles[0], [0, 1]) and np.array_equal(quartiles[3], [6, 7])),
        "weighted_metric_uses_element_counts": abs(weighted["mrba_mse"] - expected_weighted_mr_mse) <= 1e-12,
        "positive_case_all_gate_checks_pass": all(pass_checks.values()),
        "negative_mae_case_stops_gate": not all(fail_checks.values()),
        "negative_case_mse_can_still_pass": fail_checks["seed2025_rare_q4_mse_gain_positive"],
        "negative_case_mae_is_rejected": not fail_checks["seed2025_rare_q4_mae_gain_positive"],
    }
    report = {
        "schema_version": 1,
        "step": 12,
        "artifact_status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "positive_gate_checks": pass_checks,
        "negative_gate_checks": fail_checks,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_name(f".{REPORT.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, REPORT)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"REPORT={REPORT}")
    return 0 if report["artifact_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
