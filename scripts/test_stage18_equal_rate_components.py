#!/usr/bin/env python3
"""CPU checks for Stage-18 exact-count causal block masks."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
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


REPORT = ROOT / "reports" / "stage18_equal_rate_component_test.json"


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def transition_count(mask: torch.Tensor) -> float:
    removed = ~(mask[:, :, 0] > 0.5)
    return float((removed[:, 1:] != removed[:, :-1]).sum(dim=1).float().mean())


def main() -> int:
    original = torch.ones(128, 192, 3)
    indices = np.arange(128, dtype=np.int64)
    generated = {
        block: Exp_Long_Term_Forecast._stage18_equal_count_observation_mask(
            original, indices, 0.15, block, 2024
        )
        for block in (1, 5, 20)
    }
    repeated = Exp_Long_Term_Forecast._stage18_equal_count_observation_mask(
        original, indices, 0.15, 5, 2024
    )
    other_seed = Exp_Long_Term_Forecast._stage18_equal_count_observation_mask(
        original, indices, 0.15, 5, 2025
    )
    original_observed = original[:, :, 0] > 0.5
    target = torch.round(0.15 * original_observed.sum(dim=1).float()).to(torch.int64)
    removed_counts = {
        block: (
            original_observed & (~(mask[:, :, 0] > 0.5))
        ).sum(dim=1)
        for block, mask in generated.items()
    }
    rates = {
        block: removed_counts[block].float() / original_observed.sum(dim=1).float()
        for block in (1, 5, 20)
    }
    invalid = original.clone()
    invalid[0, 0, 1] = 0.0
    non_row_shared_rejected = False
    try:
        Exp_Long_Term_Forecast._stage18_equal_count_observation_mask(
            invalid, indices, 0.15, 5, 2024
        )
    except ValueError:
        non_row_shared_rejected = True

    checks = {
        "same_seed_is_bitwise_deterministic": bool(
            torch.equal(generated[5], repeated)
        ),
        "different_seed_changes_mask": not bool(torch.equal(generated[5], other_seed)),
        "all_scales_are_nested_subsets": all(
            bool((generated[block] <= original).all()) for block in (1, 5, 20)
        ),
        "all_scales_remove_exact_rounded_count": all(
            bool(torch.equal(removed_counts[block], target))
            for block in (1, 5, 20)
        ),
        "actual_rate_is_identical_per_sample": (
            torch.equal(rates[1], rates[5]) and torch.equal(rates[5], rates[20])
        ),
        "scales_change_mask_organization": (
            not torch.equal(generated[1], generated[5])
            and not torch.equal(generated[5], generated[20])
        ),
        "long_blocks_reduce_mask_transitions": (
            transition_count(generated[20]) < transition_count(generated[1])
        ),
        "variables_remain_row_shared": all(
            bool(
                (
                    (generated[block] > 0.5)
                    == (generated[block][:, :, :1] > 0.5)
                ).all()
            )
            for block in (1, 5, 20)
        ),
        "non_row_shared_mask_rejected": non_row_shared_rejected,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "target_removed_count": int(target[0]),
        "realized_rate": float(rates[1].mean()),
        "mean_transitions": {
            str(block): transition_count(generated[block])
            for block in (1, 5, 20)
        },
    }
    atomic_json(REPORT, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
