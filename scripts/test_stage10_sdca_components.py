#!/usr/bin/env python3
"""CPU regression tests for stability-diversity rank admission."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
S4M = ROOT / "baselines" / "S4M" / "s4m"
sys.path.insert(0, str(S4M))

from model.MIPP import MIPPPurifier  # noqa: E402


def main() -> int:
    stability = torch.tensor([[0.90, 0.80, 0.70, 0.60]], dtype=torch.float32)
    features = torch.tensor(
        [[[1.0, 0.0], [0.99, 0.10], [-1.0, 0.0], [0.0, 1.0]]],
        dtype=torch.float32,
    )
    topk, topk_diagnostics = MIPPPurifier.select_write_candidates(
        stability,
        mode="budget_topk",
        threshold=0.85,
        budget=2,
    )
    diverse, diagnostics = MIPPPurifier.select_write_candidates(
        stability,
        mode="stability_diverse_rank",
        threshold=0.85,
        budget=2,
        features=features,
    )
    repeated, repeated_diagnostics = MIPPPurifier.select_write_candidates(
        stability,
        mode="stability_diverse_rank",
        threshold=0.85,
        budget=2,
        features=features,
    )

    missing_feature_rejected = False
    try:
        MIPPPurifier.select_write_candidates(
            stability,
            mode="stability_diverse_rank",
            threshold=0.85,
            budget=2,
        )
    except ValueError as exc:
        missing_feature_rejected = "requires normalized candidate features" in str(exc)

    tied_rank = MIPPPurifier._fractional_percentile_rank(
        torch.tensor([0.2, 0.2, 0.8])
    )
    checks = {
        "legacy_budget_topk_unchanged": topk.tolist()
        == [[True, True, False, False]],
        "diverse_rank_selects_fixed_budget": int(diverse.sum().item()) == 2,
        "diverse_rank_keeps_most_stable_seed": bool(diverse.reshape(-1)[0].item()),
        "diverse_rank_replaces_redundant_second_candidate": diverse.tolist()
        == [[True, False, True, False]],
        "diverse_rank_is_deterministic": torch.equal(diverse, repeated),
        "diverse_rank_reports_lower_selected_pairwise_cosine": float(
            diagnostics["write_selected_pairwise_cosine"].item()
        )
        < float(topk_diagnostics.get("write_selected_pairwise_cosine", torch.tensor(1.0)).item()),
        "diversity_gain_is_positive": float(
            diagnostics["write_diversity_gain"].item()
        )
        > 0.0,
        "selected_stability_is_finite": math.isfinite(
            float(diagnostics["write_selected_stability"].item())
        ),
        "repeat_diagnostics_are_identical": all(
            math.isclose(
                float(diagnostics[key].item()),
                float(repeated_diagnostics[key].item()),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            for key in (
                "write_candidate_pairwise_cosine",
                "write_selected_pairwise_cosine",
                "write_selected_stability",
                "write_diversity_gain",
            )
        ),
        "missing_features_are_rejected": missing_feature_rejected,
        "fractional_rank_averages_ties": torch.allclose(
            tied_rank,
            torch.tensor([0.25, 0.25, 1.0]),
            atol=0.0,
            rtol=0.0,
        ),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "selection": {
            "stability": stability.tolist(),
            "features": features.tolist(),
            "legacy_budget_topk": topk.tolist(),
            "stability_diverse_rank": diverse.tolist(),
        },
        "diagnostics": {
            key: float(value.item()) if torch.is_tensor(value) else value
            for key, value in diagnostics.items()
        },
    }
    output = ROOT / "reports" / "stage10_sdca_component_test.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
