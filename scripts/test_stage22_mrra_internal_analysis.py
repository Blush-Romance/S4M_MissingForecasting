#!/usr/bin/env python3
"""Zero-GPU unit test for the frozen internal-feature selector."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_stage22_mrra_internal_features import screen_features  # noqa: E402


def main() -> int:
    count = 2528
    observable = (np.arange(count) % 100).astype(np.float64) / 99.0
    ones = np.ones(count, dtype=np.float64)
    candidate_error = np.where(observable <= 0.4, 0.8, 1.2)
    reference = {"sse": ones, "sae": ones, "mse": ones, "mae": ones, "count": ones}
    candidate = {
        "sse": candidate_error, "sae": candidate_error,
        "mse": candidate_error, "mae": candidate_error, "count": ones,
    }
    result = screen_features(
        {"useful": observable, "constant": np.ones(count)}, reference, candidate
    )
    selected = result.get("selected_rule") or {}
    checks = {
        "one_frozen_rule_reaches_audit": result["audit_gate_pass"] is True,
        "useful_feature_selected": selected.get("feature") == "useful",
        "constant_feature_excluded": (
            result["feature_summaries"]["constant"]["nonconstant_fit"] is False
        ),
        "candidate_fraction_bounded": 0.05 <= float(
            selected["development"]["candidate_fraction"]
        ) <= 0.95,
    }
    report = {
        "step": "22.15",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    path = ROOT / "reports" / "stage22_mrra_internal_analysis_test.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
