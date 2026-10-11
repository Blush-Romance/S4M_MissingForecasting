#!/usr/bin/env python3
"""CPU component and integration checks for Step-8 FUSR.

The core checks do not depend on S4M's CUDA kernels.  A final best-effort MTNet
check verifies opt-in construction and exact forced on/off numerical closure. If
MTNet cannot be imported or instantiated, the report preserves that blocker and
returns PARTIAL instead of silently dropping the integration checks.
"""

from __future__ import annotations

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
)


def finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all())


def close(left: torch.Tensor, right: torch.Tensor, atol: float = 1e-6) -> bool:
    return bool(torch.allclose(left, right, rtol=0.0, atol=atol))


def core_component_checks() -> tuple[dict[str, bool], dict[str, object]]:
    """Run target-free feature, routing, gradient, and state checks."""
    torch.manual_seed(2023)
    batch, length, variables = 5, 32, 4
    history = torch.randn(batch, length, variables, requires_grad=True)
    mask = torch.ones(batch, length, variables)
    mask[0, 3:11, 0] = 0.0
    mask[1, 8:24, :] = 0.0
    mask[2, ::3, 1:3] = 0.0
    mask[3, :, 2] = 0.0
    top1 = torch.tensor([0.98, 0.91, 0.87, 0.75, 0.62])
    selected = torch.tensor([0.93, 0.85, 0.80, 0.70, 0.55])
    entropy = torch.tensor([0.10, 0.40, 0.80, 1.20, 1.50])

    features = ForecastUtilityRouter.build_features(
        history,
        mask,
        top1,
        selected,
        entropy,
        candidate_count=10,
        use_spectral=True,
    )
    all_observed = ForecastUtilityRouter.build_features(
        history,
        torch.ones_like(mask),
        top1,
        selected,
        entropy,
        candidate_count=10,
        use_spectral=True,
    )
    all_missing = ForecastUtilityRouter.build_features(
        history,
        torch.zeros_like(mask),
        top1,
        selected,
        entropy,
        candidate_count=10,
        use_spectral=True,
    )
    no_spectral = ForecastUtilityRouter.build_features(
        history,
        mask,
        top1,
        selected,
        entropy,
        candidate_count=10,
        use_spectral=False,
    )

    # The router API has no target argument. Changing the future target without
    # updating parameters must therefore leave its deployment gate unchanged.
    leakage_router = ForecastUtilityRouter(dropout=0.0)
    leakage_router.eval()
    leakage_gate_before = leakage_router(features)["gate"].detach().clone()
    target_a = torch.linspace(-0.4, 0.4, batch)
    target_b = target_a.flip(0) + 10.0
    _ = leakage_router.utility_loss(leakage_router(features), target_a)
    leakage_gate_after = leakage_router(features)["gate"].detach().clone()
    _ = leakage_router.utility_loss(leakage_router(features), target_b)

    # With zero output weights, changing output biases isolates the mathematical
    # monotonicity of utility mean and predicted uncertainty.
    monotonic_router = ForecastUtilityRouter(dropout=0.0)
    monotonic_router.train()
    with torch.no_grad():
        monotonic_router.utility_head.weight.zero_()
        monotonic_router.utility_head.bias.zero_()
        monotonic_router.utility_head.bias[1] = -2.0
        monotonic_router.utility_head.bias[0] = -0.5
    low_mean_gate = monotonic_router(features)["gate_soft"].detach()
    with torch.no_grad():
        monotonic_router.utility_head.bias[0] = 0.5
    high_mean_gate = monotonic_router(features)["gate_soft"].detach()
    with torch.no_grad():
        monotonic_router.utility_head.bias[0] = 0.1
        monotonic_router.utility_head.bias[1] = -4.0
    low_uncertainty_gate = monotonic_router(features)["gate_soft"].detach()
    with torch.no_grad():
        monotonic_router.utility_head.bias[1] = 2.0
    high_uncertainty_gate = monotonic_router(features)["gate_soft"].detach()

    monotonic_router.eval()
    with torch.no_grad():
        monotonic_router.utility_head.bias[0] = -8.0
        monotonic_router.utility_head.bias[1] = -4.0
    forced_hard_off = monotonic_router(features)["gate"].detach()
    with torch.no_grad():
        monotonic_router.utility_head.bias[0] = 8.0
    forced_hard_on = monotonic_router(features)["gate"].detach()

    gradient_router = ForecastUtilityRouter(dropout=0.0)
    gradient_router.train()
    gradient_outputs = gradient_router(features)
    utility_target = torch.tensor(
        [0.25, -0.18, 0.08, -0.05, 0.30], requires_grad=True
    )
    valid_mask = torch.tensor([True, True, True, False, True])
    utility_losses = gradient_router.utility_loss(
        gradient_outputs,
        utility_target,
        valid_mask=valid_mask,
        sign_weight=0.5,
        positive_margin=0.0,
    )
    utility_losses["total_loss"].backward()
    router_gradients = {
        name: parameter.grad
        for name, parameter in gradient_router.named_parameters()
        if parameter.grad is not None
    }

    checkpoint_state = gradient_router.state_dict()
    checkpoint_keys = sorted(checkpoint_state)
    restored_router = ForecastUtilityRouter(dropout=0.0)
    restored_router.load_state_dict(checkpoint_state, strict=True)
    gradient_router.eval()
    restored_router.eval()
    restored_gate_error = float(
        (
            gradient_router(features)["gate"]
            - restored_router(features)["gate"]
        ).abs().max().item()
    )

    spectral_values = torch.cat(
        (features[:, 7], all_observed[:, 7], all_missing[:, 7]), dim=0
    )
    checks = {
        "feature_shape_is_b8": tuple(features.shape)
        == (batch, MissingnessSpectralFeaturizer.FEATURE_DIM),
        "all_feature_cases_finite": all(
            finite_tensor(item)
            for item in (features, all_observed, all_missing, no_spectral)
        ),
        "spectral_jsd_is_bounded": bool(
            ((spectral_values >= 0.0) & (spectral_values <= 1.0)).all()
        ),
        "all_observed_spectral_jsd_is_zero": float(
            all_observed[:, 7].abs().max().item()
        )
        < 1e-5,
        "spectral_ablation_keeps_shape_and_zeros_feature": (
            tuple(no_spectral.shape) == tuple(features.shape)
            and bool(torch.equal(no_spectral[:, 7], torch.zeros(batch)))
        ),
        "features_are_detached_from_history": not features.requires_grad,
        "future_target_change_does_not_change_gate": bool(
            torch.equal(leakage_gate_before, leakage_gate_after)
        ),
        "soft_gate_increases_with_predicted_utility": bool(
            torch.all(high_mean_gate > low_mean_gate)
        ),
        "soft_gate_decreases_with_uncertainty": bool(
            torch.all(high_uncertainty_gate < low_uncertainty_gate)
        ),
        "evaluation_gate_is_hard_and_reaches_off": bool(
            torch.equal(forced_hard_off, torch.zeros_like(forced_hard_off))
        ),
        "evaluation_gate_is_hard_and_reaches_on": bool(
            torch.equal(forced_hard_on, torch.ones_like(forced_hard_on))
        ),
        "utility_target_is_detached": utility_target.grad is None,
        "utility_loss_is_finite": bool(
            torch.isfinite(utility_losses["total_loss"])
        ),
        "router_receives_finite_nonzero_gradient": bool(router_gradients)
        and all(finite_tensor(gradient) for gradient in router_gradients.values())
        and any(float(gradient.abs().sum().item()) > 0.0 for gradient in router_gradients.values()),
        "checkpoint_has_router_parameters": (
            "utility_head.weight" in checkpoint_state
            and "utility_head.bias" in checkpoint_state
            and "gate_prior_logit" in checkpoint_state
        ),
        "checkpoint_roundtrip_preserves_gate": restored_gate_error == 0.0,
    }
    evidence = {
        "feature_names": list(MissingnessSpectralFeaturizer.FEATURE_NAMES),
        "feature_shape": list(features.shape),
        "feature_min": float(features.min().item()),
        "feature_max": float(features.max().item()),
        "spectral_jsd_min": float(spectral_values.min().item()),
        "spectral_jsd_max": float(spectral_values.max().item()),
        "initial_gate_mean": float(leakage_gate_before.float().mean().item()),
        "utility_monotonic_gate_delta_min": float(
            (high_mean_gate - low_mean_gate).min().item()
        ),
        "uncertainty_penalty_gate_delta_min": float(
            (low_uncertainty_gate - high_uncertainty_gate).min().item()
        ),
        "utility_loss": float(utility_losses["total_loss"].detach().item()),
        "utility_nll": float(utility_losses["utility_nll"].detach().item()),
        "sign_loss": float(utility_losses["sign_loss"].detach().item()),
        "utility_valid_count": int(utility_losses["valid_count"]),
        "router_parameter_count": sum(
            parameter.numel() for parameter in gradient_router.parameters()
        ),
        "router_gradient_parameter_names": sorted(router_gradients),
        "checkpoint_keys": checkpoint_keys,
        "checkpoint_roundtrip_gate_max_abs_error": restored_gate_error,
    }
    return checks, evidence


def minimal_mtnet_config(use_fusr: bool) -> SimpleNamespace:
    """Configuration containing every field needed by the current MTNet."""
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
        fusr_hidden_dim=32,
        fusr_gate_temperature=0.25,
        fusr_delta=0.0,
        fusr_uncertainty_penalty=1.0,
    )


def install_fixed_memory(model: torch.nn.Module, prototypes: torch.Tensor) -> None:
    """Install a tiny deterministic prototype bank without running clustering."""
    model.memory.repre = prototypes.detach().clone()
    model.memory.memory = [prototype.view(1, -1).clone() for prototype in prototypes]
    model.memory.num_dic = [0 for _ in range(model.memory.memory_size)]
    for index in range(len(model.memory.memory)):
        model.memory.num_dic[index] = 1


def mtnet_integration_checks() -> tuple[str, dict[str, bool], dict[str, object]]:
    """Best-effort exact closure checks against the current MTNet implementation."""
    try:
        from model.Bank import MTNet  # noqa: WPS433

        torch.manual_seed(811)
        baseline = MTNet(minimal_mtnet_config(use_fusr=False))
        torch.manual_seed(811)
        routed = MTNet(minimal_mtnet_config(use_fusr=True))
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
        history = torch.randn(2, 8, 2)
        mask = torch.ones_like(history)
        mask[0, 2:5, 0] = 0.0
        mask[1, 4:7, :] = 0.0

        baseline.set_prototype_enabled(True)
        baseline_on = baseline(history, seq_mask=mask)
        routed.set_prototype_enabled(True)
        routed.set_fusr_mode("on")
        fusr_on = routed(history, seq_mask=mask)
        on_gate = routed.get_last_fusr_outputs()["effective_gate"].detach().clone()

        baseline.set_prototype_enabled(False)
        baseline_off = baseline(history, seq_mask=mask)
        routed.set_prototype_enabled(True)
        routed.set_fusr_mode("off")
        fusr_off = routed(history, seq_mask=mask)
        off_gate = routed.get_last_fusr_outputs()["effective_gate"].detach().clone()

        baseline_keys = sorted(baseline.state_dict())
        routed_keys = sorted(routed.state_dict())
        fusr_keys = sorted(key for key in routed_keys if "fusr." in key)
        on_error = float((baseline_on - fusr_on).abs().max().item())
        off_error = float((baseline_off - fusr_off).abs().max().item())
        checks = {
            "use_fusr_false_does_not_instantiate_router": baseline.fusr is None,
            "use_fusr_false_has_no_fusr_state_keys": not any(
                "fusr." in key for key in baseline_keys
            ),
            "fusr_checkpoint_has_expected_prefix": bool(fusr_keys)
            and all(key.startswith("fusr.") for key in fusr_keys),
            "shared_baseline_state_load_has_no_unexpected_keys": not load_result.unexpected_keys,
            "forced_on_gate_is_one": bool(
                torch.equal(
                    on_gate,
                    torch.ones_like(on_gate),
                )
            ),
            "forced_off_gate_is_zero": bool(
                torch.equal(
                    off_gate,
                    torch.zeros_like(off_gate),
                )
            ),
            "forced_on_matches_baseline_prototype_on": close(
                baseline_on, fusr_on, atol=1e-6
            ),
            "forced_off_matches_baseline_prototype_off": close(
                baseline_off, fusr_off, atol=1e-6
            ),
            "all_mtnet_outputs_finite": all(
                finite_tensor(output)
                for output in (baseline_on, baseline_off, fusr_on, fusr_off)
            ),
        }
        evidence = {
            "baseline_output_shape": list(baseline_on.shape),
            "baseline_parameter_count": sum(
                parameter.numel() for parameter in baseline.parameters()
            ),
            "fusr_parameter_count": sum(
                parameter.numel() for parameter in routed.parameters()
            ),
            "added_parameter_count": (
                sum(parameter.numel() for parameter in routed.parameters())
                - sum(parameter.numel() for parameter in baseline.parameters())
            ),
            "forced_on_max_abs_error": on_error,
            "forced_off_max_abs_error": off_error,
            "fusr_checkpoint_keys": fusr_keys,
            "missing_keys_when_loading_baseline_into_fusr": sorted(
                load_result.missing_keys
            ),
        }
        status = "PASS" if all(checks.values()) else "FAIL"
        return status, checks, evidence
    except Exception as error:  # Preserve integration blockers in the report.
        return (
            "BLOCKED",
            {},
            {
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    core_checks, core_evidence = core_component_checks()
    core_status = "PASS" if all(core_checks.values()) else "FAIL"
    mtnet_status, mtnet_checks, mtnet_evidence = mtnet_integration_checks()
    if core_status == "PASS" and mtnet_status == "PASS":
        status = "PASS"
        exit_code = 0
    elif core_status == "PASS" and mtnet_status == "BLOCKED":
        status = "PARTIAL"
        exit_code = 3
    else:
        status = "FAIL"
        exit_code = 2

    report = {
        "status": status,
        "core_status": core_status,
        "mtnet_status": mtnet_status,
        "checks": {
            "core": core_checks,
            "mtnet": mtnet_checks,
        },
        "evidence": {
            "core": core_evidence,
            "mtnet": mtnet_evidence,
        },
        "notes": {
            "future_target_is_not_a_router_input": True,
            "mtnet_blocker_is_preserved_when_present": True,
            "smoke_test_is_not_a_formal_result": True,
        },
    }
    output = ROOT / "reports" / "stage8_fusr_component_test.json"
    atomic_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report={output}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
