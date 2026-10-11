#!/usr/bin/env python3
"""Zero-GPU tests for the frozen MRRA-v3 error-transfer analysis."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_stage22_mrra_v3_error import (  # noqa: E402
    aggregate_choice,
    moving_block_mean_ci,
    rankdata,
    response_routing_screen,
    spearman,
)


def main() -> int:
    count = 2528
    response = (np.arange(count) % 100).astype(np.float64) / 99.0
    ones = np.ones(count, dtype=np.float64)
    candidate_error = np.where(response <= 0.4, 0.8, 1.2)
    reference = {"sse": ones, "sae": ones, "count": ones}
    candidate = {
        "sse": candidate_error,
        "sae": candidate_error,
        "count": ones,
    }
    all_candidate = np.ones(count, dtype=bool)
    aggregate = aggregate_choice(
        reference, candidate, all_candidate, np.arange(count)
    )
    routing = response_routing_screen(response, reference, candidate)
    bootstrap = moving_block_mean_ci(np.linspace(-1.0, 1.0, 512), repeats=100)
    checks = {
        "average_tie_ranks": np.array_equal(
            rankdata(np.array([3.0, 1.0, 1.0, 2.0])),
            np.array([4.0, 1.5, 1.5, 3.0]),
        ),
        "spearman_monotonic_positive": abs(
            float(spearman(np.arange(10), np.arange(10))) - 1.0
        ) < 1e-12,
        "all_candidate_tradeoff_detected": aggregate["gate_pass"] is False,
        "response_rule_survives_audit": routing["audit_gate_pass"] is True,
        "routing_uses_frozen_partitions": routing["split"]["audit"] == [2080, 2528],
        "bootstrap_is_finite": all(
            np.isfinite(float(bootstrap[key]))
            for key in ("mean", "ci95_low", "ci95_high")
        ),
    }
    report = {
        "step": "22.14",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": {key: bool(value) for key, value in checks.items()},
    }
    path = ROOT / "reports" / "stage22_mrra_v3_error_analysis_test.json"
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
