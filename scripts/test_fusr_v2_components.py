#!/usr/bin/env python3
"""CPU-only component and MTNet closure checks for Step-8 FUSR V2."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
S4M = ROOT / "baselines" / "S4M" / "s4m"
S4_MODEL = S4M / "model"
S4_SOURCE = S4_MODEL / "S4" / "s4"
STRUCTURED_KERNELS = S4_SOURCE / "extensions" / "kernels"
for import_root in (S4M, S4_MODEL, S4_SOURCE, STRUCTURED_KERNELS):
    sys.path.insert(0, str(import_root))

from model.FUSR import (  # noqa: E402
    ForecastUtilityRouter,
    MissingnessSpectralFeaturizer,
    RetrievalInterventionFeaturizer,
)


def finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all())


def close(left: torch.Tensor, right: torch.Tensor, atol: float = 1e-6) -> bool:
    return bool(torch.allclose(left, right, rtol=0.0, atol=atol))


def forbidden_target_argument(callable_object) -> bool:
    forbidden = ("target", "future", "label", "seq_y", "true_y")
    names = tuple(inspect.signature(callable_object).parameters)
    return any(any(token in name.lower() for token in forbidden) for name in names)


def make_v2_router() -> ForecastUtilityRouter:
    return ForecastUtilityRouter(
        hidden_dim=16,
        dropout=0.0,
        soft_gate_temperature=0.25,
        objective_version="v2_frozen_balanced_rank",
        feature_set="intervention14",
        gate_score="sign_logit",
        gate_prior_mode="none",
    )


def router_component_checks() -> tuple[dict[str, bool], dict[str, object]]:
    torch.manual_seed(2023)
    batch, length, variables = 9, 24, 3
    history = torch.randn(batch, length, variables, requires_grad=True)
    mask = (torch.rand(batch, length, variables) > 0.25).float()
    top1 = torch.linspace(0.55, 0.95, batch)
    selected = top1 - 0.08
    entropy = torch.linspace(0.05, 1.5, batch)
    base = ForecastUtilityRouter.build_features(
        history,
        mask,
        top1,
        selected,
        entropy,
        candidate_count=12,
    )
    extra_values = tuple(
        torch.linspace(0.02 * index, 0.1 * index + 0.2, batch, requires_grad=True)
        for index in range(1, 7)
    )
    features = ForecastUtilityRouter.build_intervention_features(
        base,
        *extra_values,
    )

    router = make_v2_router()
    fit_evidence = router.fit_feature_normalizer(features)
    normalized = router.normalize_features(features)
    variable_scale = router.feature_scale > router.epsilon * 1.01

    router.eval()
    with torch.no_grad():
        router.utility_head.weight.zero_()
        router.utility_head.bias.zero_()
        router.utility_head.bias[2] = 0.20
    router.set_calibrated_threshold(0.25)
    below = router(features)
    with torch.no_grad():
        router.utility_head.bias[2] = 0.30
    above = router(features)

    state = router.state_dict()
    restored = make_v2_router()
    restored.load_state_dict(state, strict=True)
    restored.eval()
    restored_outputs = restored(features)

    # Fitting the new buffers on a V1 router must not alter its original path.
    v1 = ForecastUtilityRouter(dropout=0.0)
    v1.eval()
    v1_features = base.detach()
    v1_before = v1(v1_features)
    v1.fit_feature_normalizer(v1_features)
    v1_after = v1(v1_features)
    expected_v1_logit = (
        v1_before["conservative_utility"] / v1.soft_gate_temperature
        + v1.gate_prior_logit.to(v1_before["conservative_utility"].dtype)
    )

    checks = {
        "intervention_feature_shape_is_b14": tuple(features.shape)
        == (batch, RetrievalInterventionFeaturizer.FEATURE_DIM),
        "intervention_feature_names_are_exactly_14": len(
            RetrievalInterventionFeaturizer.FEATURE_NAMES
        )
        == 14,
        "intervention_features_are_finite": finite(features),
        "intervention_features_are_detached": not features.requires_grad,
        "normalization_buffers_are_ready": bool(router.normalization_ready.item()),
        "normalized_features_are_finite": finite(normalized),
        "normalized_variable_features_have_zero_mean": bool(
            normalized[:, variable_scale].mean(dim=0).abs().max() < 1e-5
        ),
        "v2_has_no_gate_prior_parameter": router.gate_prior_logit is None
        and "gate_prior_logit" not in state,
        "v2_gate_score_is_sign_logit": torch.equal(
            below["gate_score"], below["sign_logit"]
        ),
        "calibrated_threshold_routes_below_off": torch.equal(
            below["gate"], torch.zeros_like(below["gate"])
        ),
        "calibrated_threshold_routes_above_on": torch.equal(
            above["gate"], torch.ones_like(above["gate"])
        ),
        "calibrated_threshold_formula_has_no_prior": close(
            above["gate_logit"],
            (above["sign_logit"] - router.calibrated_threshold) / 0.25,
            atol=0.0,
        ),
        "calibration_state_is_locked_in_checkpoint": bool(
            restored.calibration_ready.item()
        )
        and float(restored.calibrated_threshold.item()) == 0.25
        and torch.equal(restored_outputs["gate"], above["gate"]),
        "router_forward_has_no_target_input": not forbidden_target_argument(
            ForecastUtilityRouter.forward
        ),
        "intervention_builder_has_no_target_input": not forbidden_target_argument(
            ForecastUtilityRouter.build_intervention_features
        ),
        "intervention_module_has_no_target_input": not forbidden_target_argument(
            RetrievalInterventionFeaturizer.forward
        ),
        "v1_gate_formula_is_unchanged": torch.equal(
            v1_before["gate_logit"], expected_v1_logit
        ),
        "v1_ignores_new_normalization_buffers": all(
            torch.equal(v1_before[key], v1_after[key])
            for key in ("utility_mean", "sign_logit", "gate_logit", "gate")
        ),
        "v1_gate_score_remains_conservative_utility": torch.equal(
            v1_before["gate_score"], v1_before["conservative_utility"]
        ),
    }
    evidence = {
        "feature_names": list(RetrievalInterventionFeaturizer.FEATURE_NAMES),
        "feature_shape": list(features.shape),
        "extra_feature_min": float(features[:, 8:].min().item()),
        "extra_feature_max": float(features[:, 8:].max().item()),
        "fit_feature_normalizer": fit_evidence,
        "normalization_ready": bool(router.normalization_ready.item()),
        "calibration_ready": bool(router.calibration_ready.item()),
        "calibrated_threshold": float(router.calibrated_threshold.item()),
        "below_threshold_gate_mean": float(below["gate"].mean().item()),
        "above_threshold_gate_mean": float(above["gate"].mean().item()),
        "checkpoint_keys": sorted(state),
        "v1_feature_dim": MissingnessSpectralFeaturizer.FEATURE_DIM,
    }
    return checks, evidence


def minimal_mtnet_config(use_fusr: bool, v2: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        n=2,
        short_len=4,
        W=2,
        d_var=2,
        en_conv_hidden_size=4,
        en_rnn_hidden_sizes=[4],
        input_keep_prob=1.0,
        output_keep_prob=1.0,
        factor=1,
        dropout=0.0,
        output_attention=False,
        n_heads=1,
        K=2,
        momentum=0.99,
        is_training=0,
        memory_size=8,
        M=2,
        per_mem_size=3,
        thres1=0.9,
        thres2=0.1,
        topM=8,
        topK=2,
        seed=2023,
        deterministic_cluster=True,
        use_mipp=False,
        use_content_mask_factorization=False,
        use_stability_write_gate=False,
        use_fusr=use_fusr,
        use_utility_read_gate=use_fusr,
        use_fusr_spectral_uncertainty=use_fusr,
        fusr_hidden_dim=16,
        fusr_gate_temperature=0.25,
        fusr_delta=0.0,
        fusr_uncertainty_penalty=1.0,
        fusr_objective_version=(
            "v2_frozen_balanced_rank" if v2 else "v1_joint"
        ),
        fusr_feature_set="intervention14" if v2 else "base8",
        fusr_gate_score="sign_logit" if v2 else "utility_lcb",
        fusr_gate_prior_mode="none" if v2 else "learned",
    )


def install_fixed_memory(model: torch.nn.Module, prototypes: torch.Tensor) -> None:
    model.memory.repre = prototypes.detach().clone()
    model.memory.memory = [prototype.view(1, -1).clone() for prototype in prototypes]
    model.memory.num_dic = [0 for _ in range(model.memory.memory_size)]
    for index in range(len(model.memory.memory)):
        model.memory.num_dic[index] = 1


def mtnet_v2_checks() -> tuple[dict[str, bool], dict[str, object]]:
    from model.Bank import MTNet  # noqa: WPS433

    torch.manual_seed(811)
    baseline = MTNet(minimal_mtnet_config(use_fusr=False))
    torch.manual_seed(811)
    routed = MTNet(minimal_mtnet_config(use_fusr=True, v2=True))
    load_result = routed.load_state_dict(baseline.state_dict(), strict=False)
    baseline.eval()
    routed.eval()

    prototypes = torch.nn.functional.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        ),
        dim=1,
    )
    install_fixed_memory(baseline, prototypes)
    install_fixed_memory(routed, prototypes)
    history = torch.randn(3, 8, 2)
    mask = torch.ones_like(history)
    mask[0, 2:5, 0] = 0.0
    mask[1, 4:7, :] = 0.0
    mask[2, ::2, 1] = 0.0

    baseline.set_prototype_enabled(True)
    baseline_on = baseline(history, seq_mask=mask)
    routed.set_prototype_enabled(True)
    routed.set_fusr_mode("on")
    fusr_on = routed(history, seq_mask=mask)

    baseline.set_prototype_enabled(False)
    baseline_off = baseline(history, seq_mask=mask)
    routed.set_prototype_enabled(True)
    routed.set_fusr_mode("off")
    fusr_off = routed(history, seq_mask=mask)

    routed.set_fusr_mode("route")
    routed.fusr.fit_feature_normalizer(
        routed.get_last_fusr_outputs()["features"]
    )
    routed.fusr.set_calibrated_threshold(0.10)
    with torch.no_grad():
        routed.fusr.utility_head.weight.zero_()
        routed.fusr.utility_head.bias.zero_()
        routed.fusr.utility_head.bias[2] = 0.20
    routed_output = routed(history, seq_mask=mask)
    outputs = routed.get_last_fusr_outputs()
    features = outputs["features"]

    checks = {
        "bank_constructs_intervention14_only_for_v2": (
            routed.use_fusr_intervention_features
            and tuple(features.shape) == (history.shape[0], 14)
        ),
        "bank_intervention14_features_are_finite": finite(features),
        "bank_extra_six_features_are_finite": finite(features[:, 8:]),
        "bank_outputs_gate_score": "gate_score" in outputs
        and torch.equal(outputs["gate_score"], outputs["sign_logit"]),
        "bank_v2_has_no_prior": routed.fusr.gate_prior_logit is None,
        "bank_calibrated_route_is_effective": torch.equal(
            outputs["effective_gate"], torch.ones_like(outputs["effective_gate"])
        ),
        "forced_on_matches_unchanged_baseline": close(baseline_on, fusr_on),
        "forced_off_matches_unchanged_baseline": close(baseline_off, fusr_off),
        "all_bank_outputs_are_finite": all(
            finite(value)
            for value in (baseline_on, baseline_off, fusr_on, fusr_off, routed_output)
        ),
        "loading_baseline_has_no_unexpected_keys": not load_result.unexpected_keys,
        "bank_forward_has_no_future_target_input": not forbidden_target_argument(
            MTNet.forward
        ),
    }
    evidence = {
        "feature_shape": list(features.shape),
        "extra_features": features[:, 8:].detach().tolist(),
        "gate_score": outputs["gate_score"].detach().tolist(),
        "effective_gate": outputs["effective_gate"].detach().tolist(),
        "forced_on_max_abs_error": float((baseline_on - fusr_on).abs().max().item()),
        "forced_off_max_abs_error": float((baseline_off - fusr_off).abs().max().item()),
        "missing_keys_loading_baseline": sorted(load_result.missing_keys),
    }
    return checks, evidence


def main() -> int:
    component_checks, component_evidence = router_component_checks()
    integration_checks, integration_evidence = mtnet_v2_checks()
    checks = {**component_checks, **integration_checks}
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "status": status,
        "passed": sum(bool(value) for value in checks.values()),
        "total": len(checks),
        "checks": checks,
        "evidence": {
            "router": component_evidence,
            "mtnet": integration_evidence,
        },
        "notes": {
            "cpu_component_test_only": True,
            "future_target_is_not_a_feature_or_gate_input": True,
            "smoke_test_is_not_a_formal_result": True,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
