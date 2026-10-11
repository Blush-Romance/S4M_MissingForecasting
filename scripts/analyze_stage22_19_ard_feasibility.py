#!/usr/bin/env python3
"""Assess analytic residual-action distillation using only existing Step-22.17 data."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN = ROOT / (
    "runs/stage22_mcpr_valcache_mcpr_mse__electricity__validation__seed2023__"
    "20260813_191651"
)
REPORT_JSON = ROOT / "reports/stage22_19_ard_feasibility.json"
REPORT_MD = ROOT / "reports/stage22_19_ard_feasibility.md"
FIT = slice(0, 1264)
DEV = slice(1456, 1888)
AUDIT = slice(2080, 2528)
EPSILON = 1e-12


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def analytic_alpha(alignment_cross: np.ndarray, residual_energy: np.ndarray) -> np.ndarray:
    """Closed-form sample-wise minimizer of alpha*a + alpha^2*e on [0, 1]."""
    alignment = np.asarray(alignment_cross, dtype=np.float64)
    energy = np.asarray(residual_energy, dtype=np.float64)
    return np.clip(-alignment / (2.0 * np.maximum(energy, EPSILON)), 0.0, 1.0)


def action_labels(alpha: np.ndarray) -> np.ndarray:
    labels = np.ones(alpha.shape, dtype=np.int64)
    labels[alpha <= EPSILON] = 0
    labels[alpha >= 1.0 - EPSILON] = 2
    return labels


def scaled_metrics(
    gate: np.ndarray,
    backbone_sse: np.ndarray,
    alignment_cross: np.ndarray,
    residual_energy: np.ndarray,
    complete_count: np.ndarray,
    split: slice,
) -> dict[str, float]:
    selected_gate = np.asarray(gate[split], dtype=np.float64)
    scaled_sse = (
        backbone_sse[split]
        + selected_gate * alignment_cross[split]
        + np.square(selected_gate) * residual_energy[split]
    )
    backbone_mse = float(backbone_sse[split].sum() / complete_count[split].sum())
    candidate_mse = float(scaled_sse.sum() / complete_count[split].sum())
    return {
        "backbone_mse": backbone_mse,
        "candidate_mse": candidate_mse,
        "mse_gain_percent": 100.0 * (backbone_mse - candidate_mse) / backbone_mse,
        "gate_mean": float(selected_gate.mean()),
        "gate_std": float(selected_gate.std(ddof=0)),
        "gate_min": float(selected_gate.min()),
        "gate_max": float(selected_gate.max()),
    }


def fit_predictors(
    features: np.ndarray,
    alpha: np.ndarray,
    labels: np.ndarray,
) -> dict[str, np.ndarray]:
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    ridge.fit(features[FIT], alpha[FIT])
    ridge_gate = np.clip(ridge.predict(features), 0.0, 1.0)

    mixture = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=2023,
            solver="lbfgs",
        ),
    )
    mixture.fit(features[FIT], labels[FIT])
    probabilities = mixture.predict_proba(features)
    classes = mixture.named_steps["logisticregression"].classes_
    class_probability = {
        int(label): probabilities[:, index]
        for index, label in enumerate(classes)
    }
    interior_mean = float(alpha[FIT][labels[FIT] == 1].mean())
    mixture_gate = (
        class_probability.get(2, np.zeros(alpha.shape[0]))
        + interior_mean * class_probability.get(1, np.zeros(alpha.shape[0]))
    )
    predicted_action = np.asarray(classes)[probabilities.argmax(axis=1)]
    return {
        "ridge_gate": ridge_gate,
        "mixture_gate": np.clip(mixture_gate, 0.0, 1.0),
        "mixture_action": predicted_action,
        "interior_teacher_mean": np.asarray([interior_mean]),
    }


def main() -> int:
    metrics = json.loads((SOURCE_RUN / "metrics.json").read_text(encoding="utf-8"))
    status = json.loads((SOURCE_RUN / "status.json").read_text(encoding="utf-8"))
    with np.load(SOURCE_RUN / "stage22_validation_samples.npz", allow_pickle=False) as source:
        samples = {key: source[key] for key in source.files}

    alignment = samples["mcpr_alignment_cross"].astype(np.float64)
    energy = samples["mcpr_residual_energy"].astype(np.float64)
    backbone_sse = samples["complete_sse_backbone"].astype(np.float64)
    complete_count = samples["complete_count"].astype(np.float64)
    alpha = analytic_alpha(alignment, energy)
    labels = action_labels(alpha)
    history_mask = np.unpackbits(
        samples["history_mask_packed"], axis=1
    )[:, :192]
    single_feature_names = (
        "log_residual_energy",
        "residual_abs_mean",
        "residual_max_abs",
        "history_missing_rate",
    )
    single_features = np.column_stack((
        np.log1p(energy),
        samples["mcpr_residual_abs_sum"].astype(np.float64) / complete_count,
        samples["mcpr_residual_max_abs"].astype(np.float64),
        1.0 - history_mask.mean(axis=1),
    ))
    counterfactual_feature_names = single_feature_names + ("nested_prediction_disagreement",)
    counterfactual_features = np.column_stack((
        single_features,
        samples["prediction_disagreement_mse"].astype(np.float64),
    ))

    constant_alpha = float(np.clip(
        -alignment[FIT].sum() / (2.0 * max(energy[FIT].sum(), EPSILON)),
        0.0,
        1.0,
    ))
    gates: dict[str, np.ndarray] = {
        "backbone_zero": np.zeros_like(alpha),
        "raw_full": np.ones_like(alpha),
        "fit_global_constant": np.full_like(alpha, constant_alpha),
        "sample_oracle": alpha,
    }
    predictors: dict[str, dict[str, np.ndarray]] = {}
    for name, features in (
        ("single_view", single_features),
        ("counterfactual_view", counterfactual_features),
    ):
        predictors[name] = fit_predictors(features, alpha, labels)
        gates[f"{name}_ridge"] = predictors[name]["ridge_gate"]
        gates[f"{name}_mixture"] = predictors[name]["mixture_gate"]

    results: dict[str, Any] = {}
    for method, gate in gates.items():
        results[method] = {
            "fit": scaled_metrics(
                gate, backbone_sse, alignment, energy, complete_count, FIT
            ),
            "development": scaled_metrics(
                gate, backbone_sse, alignment, energy, complete_count, DEV
            ),
            "audit": scaled_metrics(
                gate, backbone_sse, alignment, energy, complete_count, AUDIT
            ),
        }

    classification: dict[str, Any] = {}
    for name, predictor in predictors.items():
        classification[name] = {}
        for split_name, split in (("development", DEV), ("audit", AUDIT)):
            predicted = predictor["mixture_action"][split]
            truth = labels[split]
            classification[name][split_name] = {
                "accuracy": float(accuracy_score(truth, predicted)),
                "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
                "macro_f1": float(f1_score(truth, predicted, average="macro")),
                "alpha_mae": float(np.mean(np.abs(
                    predictors[name]["mixture_gate"][split] - alpha[split]
                ))),
            }

    correlations = {}
    for index, feature_name in enumerate(counterfactual_feature_names):
        correlation = spearmanr(counterfactual_features[:, index], alpha)
        correlations[feature_name] = {
            "spearman": float(correlation.statistic),
            "pvalue": float(correlation.pvalue),
        }

    primary = results["counterfactual_view_mixture"]
    feasibility_pass = bool(
        primary["development"]["mse_gain_percent"] > 0.0
        and primary["audit"]["mse_gain_percent"] > 0.0
    )
    protocol_checks = {
        "source_success": status.get("state") == "SUCCESS",
        "source_artifact_pass": metrics.get("artifact_status") == "PASS",
        "validation_only": metrics.get("uses_test_split") is False,
        "frozen_checkpoint": metrics.get("updates_model_or_memory") is False,
        "full_chronological_validation": (
            int(metrics.get("sample_count", 0)) == 2528
            and int(metrics.get("validation_batches_processed", 0)) == 158
            and bool(np.array_equal(samples["sample_index"], np.arange(2528)))
        ),
        "analytic_identity_closes": bool(np.all(
            backbone_sse + alpha * alignment + np.square(alpha) * energy
            <= backbone_sse + 1e-8
        )),
        "features_label_free": True,
        "test_not_accessed": True,
    }
    if not all(protocol_checks.values()):
        raise RuntimeError(f"Step-22.19 feasibility protocol failed: {protocol_checks}")

    quantile_points = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    payload = {
        "step": "22.19.1",
        "status": "PASS",
        "decision": (
            "COUNTERFACTUAL_ANALYTIC_DISTILLATION_SUPPORTED"
            if feasibility_pass else "COUNTERFACTUAL_DISTILLATION_NOT_SUPPORTED"
        ),
        "scope": "existing_seed2023_validation_only_exploratory_feasibility",
        "test_accessed": False,
        "source_run": str(SOURCE_RUN),
        "protocol_checks": protocol_checks,
        "split": {
            "fit": [0, 1264],
            "embargo_1": [1264, 1456],
            "development": [1456, 1888],
            "embargo_2": [1888, 2080],
            "audit": [2080, 2528],
        },
        "teacher": {
            "formula": "clip(-alignment_cross/(2*residual_energy),0,1)",
            "quantile_points": quantile_points,
            "quantiles": [float(value) for value in np.quantile(alpha, quantile_points)],
            "mean": float(alpha.mean()),
            "std": float(alpha.std(ddof=0)),
            "zero_fraction": float(np.mean(labels == 0)),
            "interior_fraction": float(np.mean(labels == 1)),
            "full_fraction": float(np.mean(labels == 2)),
            "fit_global_constant": constant_alpha,
        },
        "feature_sets": {
            "single_view": list(single_feature_names),
            "counterfactual_view": list(counterfactual_feature_names),
        },
        "feature_correlations": correlations,
        "classification": classification,
        "mse_results": results,
        "feasibility_gate": {
            "primary": "counterfactual_view_mixture",
            "requires_positive_mse_gain_on_development_and_audit": True,
            "pass": feasibility_pass,
        },
    }
    atomic_text(REPORT_JSON, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# 步骤22.19.1：解析残差动作蒸馏可行性", "",
        f"- 状态：`{payload['status']}`",
        f"- 决策：`{payload['decision']}`",
        "- 范围：仅使用既有seed2023 validation缓存的发现性诊断；不训练模型、不访问test", "",
        f"- 教师动作比例（关/中间/全开）：`{payload['teacher']['zero_fraction']:.6f} / {payload['teacher']['interior_fraction']:.6f} / {payload['teacher']['full_fraction']:.6f}`",
        f"- counterfactual mixture development MSE收益：`{primary['development']['mse_gain_percent']:+.6f}%`",
        f"- counterfactual mixture audit MSE收益：`{primary['audit']['mse_gain_percent']:+.6f}%`",
        "",
    ]
    atomic_text(REPORT_MD, "\n".join(lines))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
