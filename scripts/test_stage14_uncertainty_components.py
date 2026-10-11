#!/usr/bin/env python3
"""CPU component checks for the Stage-14 deterministic nested mask."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
S4M = ROOT / "baselines" / "S4M" / "s4m"
for path in (
    S4M,
    S4M / "model",
    S4M / "model" / "S4" / "s4",
    S4M / "model" / "S4" / "s4" / "extensions" / "kernels",
):
    sys.path.insert(0, str(path))

from experiments.exp_pretrain1 import Exp_Long_Term_Forecast  # noqa: E402


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    original_rows = torch.ones(512, 192, 1)
    original_rows[:, ::7] = 0.0
    original = original_rows.expand(-1, -1, 3).clone()
    indices = list(range(512))
    first = Exp_Long_Term_Forecast._stage14_nested_observation_mask(
        original, indices, 0.15, 5, 2024
    )
    repeated = Exp_Long_Term_Forecast._stage14_nested_observation_mask(
        original, indices, 0.15, 5, 2024
    )
    other_seed = Exp_Long_Term_Forecast._stage14_nested_observation_mask(
        original, indices, 0.15, 5, 2025
    )
    original_observed = original[:, :, 0] > 0.5
    nested_observed = first[:, :, 0] > 0.5
    removed = original_observed & (~nested_observed)
    realized = float(
        (removed.sum(dim=1).float() / original_observed.sum(dim=1).float()).mean()
    )

    non_row_shared_rejected = False
    invalid = original.clone()
    invalid[0, 1, 1] = 0.0
    try:
        Exp_Long_Term_Forecast._stage14_nested_observation_mask(
            invalid, indices, 0.15, 5, 2024
        )
    except ValueError:
        non_row_shared_rejected = True

    checks = {
        "same_seed_is_bitwise_deterministic": bool(torch.equal(first, repeated)),
        "different_seed_changes_mask": not bool(torch.equal(first, other_seed)),
        "nested_mask_is_subset": bool((first <= original).all()),
        "original_missing_rows_remain_missing": bool(
            (first[original == 0.0] == 0.0).all()
        ),
        "variables_remain_row_shared": bool(
            ((first > 0.5) == (first[:, :, :1] > 0.5)).all()
        ),
        "realized_rate_matches_target_with_sampling_tolerance": 0.12 <= realized <= 0.18,
        "non_row_shared_mask_rejected": non_row_shared_rejected,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "target_rate": 0.15,
        "block_length": 5,
        "realized_additional_missing_rate": realized,
        "start_probability": 1.0 - (1.0 - 0.15) ** (1.0 / 5.0),
    }
    atomic_json(ROOT / "reports" / "stage14_uncertainty_component_test.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
