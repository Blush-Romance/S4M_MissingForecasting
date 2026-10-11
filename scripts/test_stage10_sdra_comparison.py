#!/usr/bin/env python3
"""Synthetic unit tests for the frozen SDRA anchor decision rule."""

from __future__ import annotations

import json
from pathlib import Path

from compare_stage10_sdra_anchor_results import (
    anchor_decision,
    checks_status,
    metric_change,
)


ROOT = Path(__file__).resolve().parents[1]


def pair(mse: float, mae: float, candidate_mse: float, candidate_mae: float) -> dict:
    return metric_change(
        {"mse": mse, "mae": mae},
        {"mse": candidate_mse, "mae": candidate_mae},
    )


def main() -> int:
    positive = {
        "2024": {"sdra_vs_pg95": pair(1.0, 1.0, 0.9, 0.9)},
        "2025": {"sdra_vs_pg95": pair(1.0, 1.0, 0.8, 0.95)},
    }
    mixed = {
        "2024": {"sdra_vs_pg95": pair(1.0, 1.0, 0.9, 0.9)},
        "2025": {"sdra_vs_pg95": pair(1.0, 1.0, 0.9, 1.01)},
    }
    negative = {
        "2024": {"sdra_vs_pg95": pair(1.0, 1.0, 1.1, 1.1)},
        "2025": {"sdra_vs_pg95": pair(1.0, 1.0, 0.9, 0.9)},
    }
    positive_decision = anchor_decision(positive)
    mixed_decision = anchor_decision(mixed)
    negative_decision = anchor_decision(negative)
    checks = {
        "metric_change_uses_lower_is_better": (
            positive["2024"]["sdra_vs_pg95"]["mse_relative_gain"] > 0.0
            and positive["2024"]["sdra_vs_pg95"]["dual_metric_win"] is True
        ),
        "both_anchor_dual_wins_unlock_seed2023": positive_decision[0] is True
        and positive_decision[1] == "SDRA_TWO_ANCHOR_DUAL_WIN_RUN_SEED2023",
        "single_metric_failure_stops_candidate": mixed_decision[0] is False
        and mixed_decision[1] == "SDRA_TWO_ANCHOR_GATE_FAILED_STOP_CANDIDATE",
        "single_seed_failure_stops_candidate": negative_decision[0] is False
        and negative_decision[1] == "SDRA_TWO_ANCHOR_GATE_FAILED_STOP_CANDIDATE",
        "failed_mechanism_checks_are_reportable": checks_status(
            {"seed2024_positive_diversity_gain": False, "seed2025_positive_diversity_gain": True}
        )
        == "FAIL",
        "valid_protocol_checks_remain_pass": checks_status(
            {"same_data": True, "same_protocol": True, "same_parameter_count": True}
        )
        == "PASS",
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "positive_decision": positive_decision,
        "mixed_decision": mixed_decision,
        "negative_decision": negative_decision,
    }
    output = ROOT / "reports" / "stage10_sdra_comparison_test.json"
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
