#!/usr/bin/env python3
"""Synthetic regression tests for Step-22.23 validation-only selection."""

from __future__ import annotations

import json

import analyze_stage22_23_efficient_cnm_se as analyzer


def record(seed: int, candidate_gains: dict[str, tuple[float, float]]) -> dict:
    original = {"mse": 1.0, "mae": 1.0}
    multiscale = {"mse": 0.90, "mae": 0.90}
    candidates = {
        name: {"mse": 1.0 - gains[0], "mae": 1.0 - gains[1]}
        for name, gains in candidate_gains.items()
    }
    timing = {
        "original_forward_seconds": {"mean_ms": 10.0},
        "multiscale_fusion_seconds": {"mean_ms": 0.1},
    }
    for block in (1, 5, 20):
        for part, value in (("mask", 0.2), ("preprocess", 0.1),
                            ("forward", 10.0), ("fusion", 0.1)):
            timing[f"l{block}_{part}_seconds"] = {"mean_ms": value}
    return {
        "seed": seed,
        "run_dir": f"validation_seed{seed}",
        "protocol_checks": {"synthetic": True},
        "original": original,
        "multiscale": multiscale,
        "candidates": candidates,
        "timing_components": timing,
        "timing_measured_batches": 3,
    }


def make_records(gains: dict[str, tuple[float, float]]) -> list[dict]:
    return [record(seed, gains) for seed in analyzer.SEEDS]


def main() -> int:
    test_calls: list[str] = []

    def forbidden_test_loader(selected: str) -> list[dict]:
        test_calls.append(selected)
        raise AssertionError("test must not be loaded when validation rejects all")

    rejected = analyzer.analyze(make_records({
        "E1": (0.06, 0.06), "E5": (0.05, 0.05), "E20": (0.04, 0.04),
    }), forbidden_test_loader)

    selected_calls: list[str] = []

    def passing_test_loader(selected: str) -> list[dict]:
        selected_calls.append(selected)
        return [{
            "seed": seed,
            "protocol_checks": {"synthetic": True},
            "both_metrics_improve": True,
        } for seed in analyzer.SEEDS]

    selected = analyzer.analyze(make_records({
        "E1": (0.081, 0.081), "E5": (0.09, 0.09), "E20": (0.085, 0.085),
    }), passing_test_loader)

    tie = analyzer.select_candidate(make_records({
        "E1": (0.09, 0.09), "E5": (0.09, 0.09), "E20": (0.09, 0.09),
    }))

    mae_failure = analyzer.select_candidate([
        record(seed, {
            "E1": (0.09, 0.09), "E5": (0.09, -0.001 if seed == 2028 else 0.09),
            "E20": (0.04, 0.04),
        }) for seed in analyzer.SEEDS
    ])

    checks = {
        "rejection_decision": rejected["decision"] == "EFFICIENT_TWO_VIEW_STOP",
        "test_not_loaded_on_rejection": not test_calls and not rejected["test_sources_accessed"],
        "best_validation_candidate_selected": selected["selection"]["selected_candidate"] == "E5",
        "test_loaded_only_after_selection": selected_calls == ["E5"],
        "fresh_seed_authorized_after_test_gate": selected["decision"] == "EFFICIENT_TWO_VIEW_FRESH_SEED_AUTHORIZED",
        "frozen_tie_priority": tie["selected_candidate"] == "E5",
        "single_seed_mae_failure_rejects_candidate": not mae_failure["candidate_summaries"]["E5"]["passes_validation_gate"],
    }
    failures = [name for name, passed in checks.items() if not passed]
    print(json.dumps({
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
    }, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
