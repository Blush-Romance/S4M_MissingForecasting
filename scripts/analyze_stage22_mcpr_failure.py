#!/usr/bin/env python3
"""Diagnose exact MCPR residual error transfer on frozen validation caches."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = ROOT / "reports" / "stage22_mcpr_failure_attribution.json"
REPORT_MD = ROOT / "reports" / "stage22_mcpr_failure_attribution.md"
PG95_RUN = ROOT / "runs" / "stage22_valcache_pg95__electricity__validation__seed2023__20260805_164843"


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def latest_formal_run(method: str) -> Path:
    prefix = f"stage22_mcpr_valcache_{method}__electricity__validation__seed2023__"
    candidates: list[Path] = []
    for path in (ROOT / "runs").glob(prefix + "*"):
        status_path = path / "status.json"
        config_path = path / "resolved_config.yaml"
        if not status_path.is_file() or not config_path.is_file():
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if status.get("state") != "SUCCESS":
            continue
        if bool(config["experiment"].get("smoke_test")):
            continue
        candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"no successful formal cache found for {method}")
    return sorted(candidates)[-1]


def load_run(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((path / "resolved_config.yaml").read_text(encoding="utf-8"))
    with np.load(path / "stage22_validation_samples.npz", allow_pickle=False) as source:
        samples = {key: source[key].copy() for key in source.files}
    return metrics, samples, config


def percent_gain(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left_rank = rankdata(left)
    right_rank = rankdata(right)
    if float(left_rank.std()) == 0.0 or float(right_rank.std()) == 0.0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def moving_block_mean_ci(
    values: np.ndarray,
    *,
    seed: int = 2023,
    block_length: int = 288,
    repeats: int = 2000,
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("moving-block bootstrap requires a non-empty vector")
    block_length = min(int(block_length), int(values.size))
    starts = np.arange(values.size - block_length + 1)
    blocks_needed = int(np.ceil(values.size / block_length))
    rng = np.random.default_rng(seed)
    estimates = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate(
            [values[start:start + block_length] for start in chosen]
        )[:values.size]
        estimates[index] = sample.mean()
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "block_length": block_length,
        "repeats": repeats,
    }


def mask_structure(packed: np.ndarray, width: int = 192) -> dict[str, np.ndarray]:
    observed = np.unpackbits(packed, axis=1)[:, :width].astype(bool)
    missing = ~observed
    missing_rate = missing.mean(axis=1).astype(np.float64)
    run_count = np.zeros(missing.shape[0], dtype=np.float64)
    longest = np.zeros(missing.shape[0], dtype=np.float64)
    for row_index, row in enumerate(missing):
        current = 0
        for value in row:
            if value:
                if current == 0:
                    run_count[row_index] += 1.0
                current += 1
                longest[row_index] = max(longest[row_index], current)
            else:
                current = 0
    return {
        "missing_rate": missing_rate,
        "missing_run_count": run_count,
        "longest_missing_run": longest,
    }


def interpretation(alignment_cross: float, residual_energy: float, delta_sse: float) -> str:
    if alignment_cross >= 0.0:
        return "DIRECTION_ALIGNMENT_FAILURE"
    if delta_sse > 0.0 and residual_energy > -alignment_cross:
        return "AMPLITUDE_OVERSHOOT"
    if delta_sse <= 0.0:
        return "MSE_IMPROVING_RESIDUAL"
    return "MIXED_RESIDUAL_FAILURE"


def axis_analysis(samples: dict[str, np.ndarray], prefix: str) -> dict[str, Any]:
    count = samples[f"{prefix}_count"].astype(np.float64)
    candidate_mse = samples[f"{prefix}_sse"] / count
    candidate_mae = samples[f"{prefix}_sae"] / count
    backbone_mse = samples[f"{prefix}_sse_backbone"] / count
    backbone_mae = samples[f"{prefix}_sae_backbone"] / count
    delta_mse = samples[f"{prefix}_delta_sse"] / count
    delta_mae = samples[f"{prefix}_delta_sae"] / count
    cross = samples[f"{prefix}_alignment_cross"] / count
    energy = samples[f"{prefix}_residual_energy"] / count
    worst = np.argsort(delta_mse, kind="mergesort")[-10:][::-1]
    best = np.argsort(delta_mse, kind="mergesort")[:10]
    return {
        "size": int(count.size),
        "candidate_mse": [float(value) for value in candidate_mse],
        "candidate_mae": [float(value) for value in candidate_mae],
        "backbone_mse": [float(value) for value in backbone_mse],
        "backbone_mae": [float(value) for value in backbone_mae],
        "delta_mse": [float(value) for value in delta_mse],
        "delta_mae": [float(value) for value in delta_mae],
        "alignment_cross_mean": [float(value) for value in cross],
        "residual_energy_mean": [float(value) for value in energy],
        "mse_improved_fraction": float((delta_mse < 0.0).mean()),
        "mae_improved_fraction": float((delta_mae < 0.0).mean()),
        "both_improved_fraction": float(((delta_mse < 0.0) & (delta_mae < 0.0)).mean()),
        "worst_indices": [int(value) for value in worst],
        "best_indices": [int(value) for value in best],
        "decomposition_max_abs_error": float(np.max(np.abs(
            samples[f"{prefix}_delta_sse"]
            - samples[f"{prefix}_alignment_cross"]
            - samples[f"{prefix}_residual_energy"]
        ))),
    }


def method_analysis(samples: dict[str, np.ndarray]) -> dict[str, Any]:
    count = samples["complete_count"].astype(np.float64)
    total_count = float(count.sum())
    backbone_sse = samples["complete_sse_backbone"].astype(np.float64)
    backbone_sae = samples["complete_sae_backbone"].astype(np.float64)
    candidate_sse = samples["complete_sse_original"].astype(np.float64)
    candidate_sae = samples["complete_sae_original"].astype(np.float64)
    delta_sse = samples["mcpr_delta_sse"].astype(np.float64)
    delta_sae = samples["mcpr_delta_sae"].astype(np.float64)
    cross = samples["mcpr_alignment_cross"].astype(np.float64)
    energy = samples["mcpr_residual_energy"].astype(np.float64)
    backbone_mse_sample = backbone_sse / count
    delta_mse_sample = delta_sse / count
    delta_mae_sample = delta_sae / count

    backbone_mse = float(backbone_sse.sum() / total_count)
    backbone_mae = float(backbone_sae.sum() / total_count)
    candidate_mse = float(candidate_sse.sum() / total_count)
    candidate_mae = float(candidate_sae.sum() / total_count)
    cross_mean = float(cross.sum() / total_count)
    energy_mean = float(energy.sum() / total_count)
    delta_mse = float(delta_sse.sum() / total_count)
    delta_mae = float(delta_sae.sum() / total_count)

    mse_improve = delta_sse < 0.0
    mae_improve = delta_sae < 0.0
    order = np.argsort(backbone_mse_sample, kind="mergesort")
    difficulty_deciles = []
    for number, indices in enumerate(np.array_split(order, 10), start=1):
        difficulty_deciles.append({
            "decile": number,
            "sample_count": int(indices.size),
            "backbone_mse": float(backbone_mse_sample[indices].mean()),
            "delta_mse": float(delta_mse_sample[indices].mean()),
            "delta_mae": float(delta_mae_sample[indices].mean()),
            "alignment_cross_mean": float((cross[indices] / count[indices]).mean()),
            "residual_energy_mean": float((energy[indices] / count[indices]).mean()),
            "both_improve_fraction": float((mse_improve[indices] & mae_improve[indices]).mean()),
        })

    positive_harm = np.maximum(delta_sse, 0.0)
    harm_order = np.argsort(delta_sse, kind="mergesort")[::-1]
    tail = {}
    for percentage in (1, 5, 10):
        size = max(1, int(math.ceil(delta_sse.size * percentage / 100.0)))
        indices = harm_order[:size]
        tail[f"top_{percentage}_percent"] = {
            "sample_count": int(size),
            "positive_harm_fraction": (
                float(positive_harm[indices].sum() / positive_harm.sum())
                if float(positive_harm.sum()) > 0.0 else 0.0
            ),
            "net_delta_sse_fraction": (
                float(delta_sse[indices].sum() / delta_sse.sum())
                if float(delta_sse.sum()) != 0.0 else None
            ),
        }

    quantile_shift = {}
    for suffix in ("q50", "q90", "q95", "q99", "max"):
        backbone = samples[f"backbone_abs_error_{suffix}"].astype(np.float64)
        candidate = samples[f"candidate_abs_error_{suffix}"].astype(np.float64)
        quantile_shift[suffix] = {
            "backbone_mean": float(backbone.mean()),
            "candidate_mean": float(candidate.mean()),
            "delta_mean": float((candidate - backbone).mean()),
        }

    structures = mask_structure(samples["history_mask_packed"])
    outcomes = {
        "delta_mse": delta_mse_sample,
        "delta_mae": delta_mae_sample,
        "alignment_cross": cross / count,
        "residual_energy": energy / count,
    }
    structure_correlations = {
        feature: {
            outcome: spearman(values, outcome_values)
            for outcome, outcome_values in outcomes.items()
        }
        for feature, values in structures.items()
    }

    alpha_unconstrained = (
        float(-cross.sum() / (2.0 * energy.sum()))
        if float(energy.sum()) > 0.0 else 0.0
    )
    alpha_clipped = float(np.clip(alpha_unconstrained, 0.0, 1.0))
    mse_alpha = float(
        backbone_mse
        + (alpha_clipped * cross.sum() + alpha_clipped ** 2 * energy.sum())
        / total_count
    )
    mse_oracle = float(np.minimum(backbone_sse, candidate_sse).sum() / total_count)
    mae_oracle = float(np.minimum(backbone_sae, candidate_sae).sum() / total_count)

    return {
        "aggregate": {
            "backbone_mse": backbone_mse,
            "backbone_mae": backbone_mae,
            "candidate_mse": candidate_mse,
            "candidate_mae": candidate_mae,
            "mse_gain_percent": percent_gain(backbone_mse, candidate_mse),
            "mae_gain_percent": percent_gain(backbone_mae, candidate_mae),
            "delta_mse": delta_mse,
            "delta_mae": delta_mae,
            "alignment_cross_mean": cross_mean,
            "residual_energy_mean": energy_mean,
            "interpretation": interpretation(cross_mean, energy_mean, delta_mse),
            "decomposition_max_abs_error": float(np.max(np.abs(
                delta_sse - cross - energy
            ))),
        },
        "sample_quadrants": {
            "both_improve_fraction": float((mse_improve & mae_improve).mean()),
            "mse_only_improves_fraction": float((mse_improve & ~mae_improve).mean()),
            "mae_only_improves_fraction": float((~mse_improve & mae_improve).mean()),
            "neither_improves_fraction": float((~mse_improve & ~mae_improve).mean()),
        },
        "moving_block_bootstrap": {
            "delta_mse": moving_block_mean_ci(delta_mse_sample),
            "delta_mae": moving_block_mean_ci(delta_mae_sample),
        },
        "pg95_difficulty_deciles": difficulty_deciles,
        "tail_harm_concentration": tail,
        "absolute_error_quantile_shift": quantile_shift,
        "missing_structure_correlations": structure_correlations,
        "oracle_not_deployable_uses_validation_labels": {
            "sample_mse_oracle": mse_oracle,
            "sample_mae_oracle": mae_oracle,
            "global_mse_optimal_alpha_unconstrained": alpha_unconstrained,
            "global_mse_optimal_alpha_clipped_0_1": alpha_clipped,
            "global_mse_at_clipped_alpha": mse_alpha,
            "not_deployable_uses_validation_labels": True,
        },
        "horizon": axis_analysis(samples, "horizon"),
        "variable": axis_analysis(samples, "variable"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pg95-run", default=str(PG95_RUN))
    parser.add_argument("--mcpr-mse-run", default=None)
    parser.add_argument("--mcpr-pcd-run", default=None)
    args = parser.parse_args()
    run_paths = {
        "pg95": Path(args.pg95_run).resolve(),
        "mcpr_mse": (
            Path(args.mcpr_mse_run).resolve()
            if args.mcpr_mse_run else latest_formal_run("mcpr_mse")
        ),
        "mcpr_pcd": (
            Path(args.mcpr_pcd_run).resolve()
            if args.mcpr_pcd_run else latest_formal_run("mcpr_pcd")
        ),
    }
    loaded = {name: load_run(path) for name, path in run_paths.items()}
    pg95_metrics, pg95_samples, pg95_config = loaded["pg95"]
    protocol_checks: dict[str, bool] = {}
    for method in ("mcpr_mse", "mcpr_pcd"):
        metrics, samples, config = loaded[method]
        protocol_checks.update({
            f"{method}_artifact_pass": metrics.get("artifact_status") == "PASS",
            f"{method}_validation_only": metrics.get("uses_test_split") is False,
            f"{method}_frozen": metrics.get("updates_model_or_memory") is False,
            f"{method}_identity": metrics.get("method") == method,
            f"{method}_mcpr_checkpoint": len(metrics.get("checkpoint_mcpr_state_keys", [])) > 0,
            f"{method}_full_validation": (
                int(metrics.get("sample_count", 0)) == 2528
                and int(metrics.get("validation_batches_processed", 0)) == 158
            ),
            f"{method}_same_data": config["data"] == pg95_config["data"],
            f"{method}_sample_index_identical": bool(np.array_equal(
                samples["sample_index"], pg95_samples["sample_index"]
            )),
            f"{method}_history_mask_identical": bool(np.array_equal(
                samples["history_mask_packed"], pg95_samples["history_mask_packed"]
            )),
            f"{method}_nested_mask_identical": bool(np.array_equal(
                samples["nested_mask_packed"], pg95_samples["nested_mask_packed"]
            )),
            f"{method}_complete_count_identical": bool(np.array_equal(
                samples["complete_count"], pg95_samples["complete_count"]
            )),
            f"{method}_embedded_backbone_matches_pg95": bool(np.allclose(
                samples["complete_sse_backbone"],
                pg95_samples["complete_sse_original"],
                rtol=2e-5, atol=2e-2,
            ) and np.allclose(
                samples["complete_sae_backbone"],
                pg95_samples["complete_sae_original"],
                rtol=2e-5, atol=2e-2,
            )),
        })
    protocol_checks.update({
        "pg95_artifact_pass": pg95_metrics.get("artifact_status") == "PASS",
        "pg95_validation_only": pg95_metrics.get("uses_test_split") is False,
        "candidate_sample_indices_identical": bool(np.array_equal(
            loaded["mcpr_mse"][1]["sample_index"],
            loaded["mcpr_pcd"][1]["sample_index"],
        )),
    })
    if not all(protocol_checks.values()):
        raise RuntimeError(f"MCPR attribution protocol failed: {protocol_checks}")

    analyses = {
        method: method_analysis(loaded[method][1])
        for method in ("mcpr_mse", "mcpr_pcd")
    }
    interpretations = {
        method: analyses[method]["aggregate"]["interpretation"]
        for method in analyses
    }
    if len(set(interpretations.values())) == 1:
        decision = "SHARED_" + next(iter(interpretations.values()))
    else:
        decision = "MIXED_MCPR_FAILURE_MECHANISMS"
    payload = {
        "step": "22.17",
        "status": "PASS",
        "scope": "Electricity seed2023 frozen-checkpoint validation-only MCPR exact residual attribution",
        "decision": decision,
        "test_accessed": False,
        "runs": {key: str(value) for key, value in run_paths.items()},
        "protocol_checks": protocol_checks,
        "methods": analyses,
        "comparison": {
            "pcd_to_mse_residual_energy_ratio": (
                analyses["mcpr_pcd"]["aggregate"]["residual_energy_mean"]
                / analyses["mcpr_mse"]["aggregate"]["residual_energy_mean"]
            ),
            "pcd_to_mse_alignment_cross_ratio": (
                analyses["mcpr_pcd"]["aggregate"]["alignment_cross_mean"]
                / analyses["mcpr_mse"]["aggregate"]["alignment_cross_mean"]
                if analyses["mcpr_mse"]["aggregate"]["alignment_cross_mean"] != 0.0
                else None
            ),
        },
        "limitations": [
            "single-dataset single-seed attribution is mechanism evidence, not statistical confirmation",
            "oracle quantities use validation labels and are not deployable",
            "Electricity seed2023 validation is now a discovery asset and cannot alone confirm the next module",
            "this diagnosis does not reopen the stopped MCPR branch or authorize test access",
        ],
    }
    atomic_text(REPORT_JSON, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# 步骤22.17：MCPR失败归因", "",
        "- 协议：`PASS`",
        f"- 决策：`{decision}`",
        "- test访问：`false`", "",
    ]
    for method in ("mcpr_mse", "mcpr_pcd"):
        aggregate = analyses[method]["aggregate"]
        lines.extend([
            f"## {method}", "",
            f"- 主干MSE/MAE：`{aggregate['backbone_mse']:.9f}/{aggregate['backbone_mae']:.9f}`",
            f"- 候选MSE/MAE：`{aggregate['candidate_mse']:.9f}/{aggregate['candidate_mae']:.9f}`",
            f"- alignment cross / residual energy / delta MSE：`{aggregate['alignment_cross_mean']:.9f} / {aggregate['residual_energy_mean']:.9f} / {aggregate['delta_mse']:.9f}`",
            f"- 机制归因：`{aggregate['interpretation']}`", "",
        ])
    lines.append("完整样本四象限、难度分位、尾部集中、缺失结构相关、预测步/变量及Oracle上界见JSON。")
    lines.append("")
    atomic_text(REPORT_MD, "\n".join(lines))
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"REPORT_JSON={REPORT_JSON}")
    print(f"REPORT_MD={REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
