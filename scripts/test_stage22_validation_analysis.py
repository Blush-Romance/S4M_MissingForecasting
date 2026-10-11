#!/usr/bin/env python3
"""Deterministic unit tests for Stage-22 paired analysis helpers."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_stage22_validation_tradeoff import (
    aggregate_axis_summary,
    moving_block_mean_ci,
    paired_summary,
)


REPORT = ROOT / "reports" / "stage22_validation_analysis_test.json"


def main() -> int:
    count = np.full(20, 10.0)
    s4m_sse = np.linspace(10.0, 200.0, 20)
    pg95_sse = s4m_sse.copy()
    pg95_sse[-2:] -= 30.0
    pg95_sse[:-2] += 0.5
    s4m_sae = np.linspace(10.0, 30.0, 20)
    pg95_sae = s4m_sae + 0.2
    s4m = {
        "complete_count": count,
        "complete_sse_original": s4m_sse,
        "complete_sae_original": s4m_sae,
    }
    pg95 = {
        "complete_count": count.copy(),
        "complete_sse_original": pg95_sse,
        "complete_sae_original": pg95_sae,
    }
    paired = paired_summary(s4m, pg95)
    delta = np.arange(32, dtype=np.float64) / 100.0
    ci1 = moving_block_mean_ci(delta, seed=7, block_length=8, repeats=100)
    ci2 = moving_block_mean_ci(delta, seed=7, block_length=8, repeats=100)

    axis_s4m = {
        "horizon_count": np.full(4, 10.0),
        "horizon_sse": np.array([10.0, 20.0, 30.0, 40.0]),
        "horizon_sae": np.array([8.0, 9.0, 10.0, 11.0]),
        "horizon_signed_error": np.array([1.0, -2.0, 3.0, -4.0]),
    }
    axis_pg95 = {
        "horizon_count": np.full(4, 10.0),
        "horizon_sse": np.array([9.0, 21.0, 29.0, 41.0]),
        "horizon_sae": np.array([7.0, 10.0, 9.0, 12.0]),
        "horizon_signed_error": np.array([0.5, -2.5, 2.5, -4.5]),
    }
    axis = aggregate_axis_summary(axis_s4m, axis_pg95, "horizon")
    checks = {
        "tail_sse_gain_dominates": paired["top10_vs_body"]["top_sse_gain_share_of_total"] > 1.0,
        "body_mae_degrades": paired["top10_vs_body"]["body_mae_gain"] < 0.0,
        "quadrants_sum_to_one": abs(sum(paired["fractions"].values()) - 1.0) <= 1e-12,
        "bootstrap_deterministic": ci1 == ci2,
        "bootstrap_contains_empirical_mean": ci1["ci95_low"] <= ci1["mean"] <= ci1["ci95_high"],
        "axis_size": axis["size"] == 4,
        "axis_mse_improved_fraction": axis["mse_improved_fraction"] == 0.5,
        "axis_mae_improved_fraction": axis["mae_improved_fraction"] == 0.5,
    }
    payload = {
        "step": 22,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(REPORT)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"REPORT={REPORT}")
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
