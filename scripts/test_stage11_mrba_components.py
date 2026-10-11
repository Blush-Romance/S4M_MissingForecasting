#!/usr/bin/env python3
"""CPU component tests for mask-regime balanced prototype admission."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baselines" / "S4M" / "s4m"))

from model.MIPP import MIPPPurifier  # noqa: E402


def main() -> int:
    stability = torch.tensor([[0.99, 0.98, 0.97, 0.50]], dtype=torch.float32)
    # observed-value masks [B,Q,S,D]; candidates 0/1 intentionally duplicate.
    masks = torch.ones(1, 4, 8, 1, dtype=torch.float32)
    masks[0, 0, :4, 0] = 0.0
    masks[0, 1, :4, 0] = 0.0
    masks[0, 2, 4:, 0] = 0.0
    masks[0, 3, ::2, 0] = 0.0

    topk, _ = MIPPPurifier.select_write_candidates(
        stability,
        mode="budget_topk",
        threshold=0.95,
        budget=2,
    )
    selected_a, diagnostics_a = MIPPPurifier.select_write_candidates(
        stability,
        mode="mask_regime_balanced_rank",
        threshold=0.95,
        budget=2,
        mask_features=masks,
    )
    selected_b, diagnostics_b = MIPPPurifier.select_write_candidates(
        stability,
        mode="mask_regime_balanced_rank",
        threshold=0.95,
        budget=2,
        mask_features=masks,
    )
    descriptor = MIPPPurifier.mask_regime_descriptor(masks)

    missing_mask_rejected = False
    try:
        MIPPPurifier.select_write_candidates(
            stability,
            mode="mask_regime_balanced_rank",
            threshold=0.95,
            budget=2,
        )
    except ValueError as exc:
        missing_mask_rejected = "requires historical candidate masks" in str(exc)

    diagnostic_keys = (
        "write_mask_candidate_pairwise_distance",
        "write_mask_selected_pairwise_distance",
        "write_mask_coverage_distance",
        "write_mask_topk_coverage_distance",
        "write_mask_coverage_gain_vs_topk",
        "write_mask_candidate_missing_rate",
        "write_mask_selected_missing_rate",
        "write_selected_stability",
    )
    checks = {
        "legacy_topk_unchanged": topk.tolist() == [[True, True, False, False]],
        "fixed_budget": int(selected_a.sum().item()) == 2,
        "highest_stability_anchor_kept": bool(selected_a[0, 0].item()),
        "duplicate_second_candidate_replaced": not bool(selected_a[0, 1].item()),
        "different_mask_regime_selected": bool(
            selected_a[0, 2].item() or selected_a[0, 3].item()
        ),
        "positive_coverage_gain_vs_topk": float(
            diagnostics_a["write_mask_coverage_gain_vs_topk"].item()
        )
        > 0.0,
        "deterministic_selection": torch.equal(selected_a, selected_b),
        "deterministic_diagnostics": all(
            torch.equal(diagnostics_a[key], diagnostics_b[key])
            for key in diagnostic_keys
        ),
        "descriptor_fixed_width": descriptor.shape == (1, 4, 13),
        "descriptor_temporal_part_exact": torch.equal(
            descriptor[0, 0, :8], 1.0 - masks[0, 0, :, 0]
        ),
        "missing_mask_rejected": missing_mask_rejected,
        "all_diagnostics_finite": all(
            torch.isfinite(diagnostics_a[key]).item() for key in diagnostic_keys
        ),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "stability": stability.tolist(),
        "legacy_topk": topk.tolist(),
        "mrba_selection": selected_a.tolist(),
        "descriptor_shape": list(descriptor.shape),
        "diagnostics": {
            key: float(diagnostics_a[key].item()) for key in diagnostic_keys
        },
    }
    output = ROOT / "reports" / "stage11_mrba_component_test.json"
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
