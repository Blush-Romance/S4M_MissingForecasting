#!/usr/bin/env python3
"""Positive and negative decision tests for the Stage-14 analysis gate."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from analyze_stage14_uncertainty import analyze_arrays


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "stage14_uncertainty_analysis_test.json"


def make_arrays(error: np.ndarray, uncertainty: np.ndarray) -> dict[str, np.ndarray]:
    count = np.full(error.size, 10.0)
    sse = error * count
    sae = np.sqrt(error) * count
    return {
        "complete_sse_on": sse,
        "complete_sae_on": sae,
        "complete_sse_off": sse * 1.05,
        "complete_sae_off": sae * 1.02,
        "complete_sse_nested": sse * 1.10,
        "complete_sae_nested": sae * 1.05,
        "complete_count": count,
        "mask_disagreement_mse": uncertainty,
        "prototype_disagreement_mse": uncertainty * 0.5 + 0.01,
    }


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    uncertainty = np.linspace(0.01, 1.0, 40)
    positive = analyze_arrays(make_arrays(np.linspace(0.1, 1.0, 40), uncertainty))
    mae_fail_error = np.linspace(0.25, 0.29, 40)
    negative = analyze_arrays(make_arrays(mae_fail_error, uncertainty))
    checks = {
        "positive_case_passes": positive["gate_pass"] is True,
        "positive_case_all_checks_pass": all(positive["checks"].values()),
        "weak_ratio_case_fails": negative["gate_pass"] is False,
        "quartiles_are_equal_size": (
            positive["q1"]["sample_count"] == 10
            and positive["q4"]["sample_count"] == 10
        ),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "positive": positive,
        "negative": negative,
    }
    atomic_json(REPORT, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
