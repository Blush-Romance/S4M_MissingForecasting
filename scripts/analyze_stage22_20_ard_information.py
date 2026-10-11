#!/usr/bin/env python3
"""Frozen time-isolated analysis for the Step 22.20 ARD diagnostic cache."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_s4m_experiment import atomic_json, atomic_text


ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = ROOT / "reports/stage22_20_ard_information.json"
REPORT_MD = ROOT / "reports/stage22_20_ard_information.md"
SPLITS = {
    "fit": (0, 1264),
    "embargo_1": (1264, 1456),
    "development": (1456, 1888),
    "embargo_2": (1888, 2080),
    "audit": (2080, 2528),
}


def latest_formal_run() -> Path | None:
    candidates = []
    for path in (ROOT / "runs").glob(
        "stage22_20_ard_information__electricity__validation__seed2025__*"
    ):
        if "__smoke__" in path.name:
            continue
        status_path = path / "status.json"
        if not status_path.is_file():
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") == "SUCCESS":
            candidates.append(path)
    return max(candidates, key=lambda path: path.name) if candidates else None


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return 0.0
    return float(spearmanr(x, y).statistic)


def linear_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(pearsonr(x, y).statistic)


def evaluate_action_probe(
    model: Any,
    features: np.ndarray,
    actions: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    x = features[valid]
    y = actions[valid]
    probabilities_raw = model.predict_proba(x)
    probabilities = np.zeros((y.size, 3), dtype=np.float64)
    probabilities[:, np.asarray(model[-1].classes_, dtype=np.int64)] = probabilities_raw
    predicted = probabilities.argmax(axis=1)
    return {
        "valid_samples": int(y.size),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "cross_entropy": float(log_loss(y, probabilities, labels=[0, 1, 2])),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1, 2]).tolist(),
    }


def evaluate_alpha_probe(
    model: Any,
    features: np.ndarray,
    teacher_alpha: np.ndarray,
    valid: np.ndarray,
    backbone_sse: np.ndarray,
    alignment_cross: np.ndarray,
    residual_energy: np.ndarray,
    complete_count: np.ndarray,
) -> dict[str, Any]:
    x = features[valid]
    teacher = teacher_alpha[valid]
    predicted = np.clip(model.predict(x), 0.0, 1.0)
    predicted_sse = (
        backbone_sse[valid]
        + predicted * alignment_cross[valid]
        + np.square(predicted) * residual_energy[valid]
    )
    return {
        "valid_samples": int(teacher.size),
        "spearman": rank_correlation(predicted, teacher),
        "mae": float(mean_absolute_error(teacher, predicted)),
        "counterfactual_mse": float(
            predicted_sse.sum() / complete_count[valid].sum()
        ),
        "predicted_alpha_mean": float(predicted.mean()),
        "predicted_alpha_std": float(predicted.std(ddof=0)),
    }


def analyse(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    samples_path = run_dir / "ard_information_samples.npz"
    status_path = run_dir / "status.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    with np.load(samples_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}

    sample_count = int(arrays["sample_index"].size)
    protocol_checks = {
        "formal_run_success": status.get("state") == "SUCCESS",
        "artifact_status_pass": metrics.get("artifact_status") == "PASS",
        "validation_only": (
            metrics.get("test_split_accessed") is False
            and metrics.get("uses_test_split") is False
            and metrics.get("source_split") == "val"
        ),
        "model_and_memory_frozen": metrics.get("updates_model_or_memory") is False,
        "chronological_loader": metrics.get("chronological_loader") is True,
        "complete_2528_samples": sample_count == 2528,
        "complete_158_batches": int(
            metrics.get("validation_batches_processed", 0)
        ) == 158,
        "chronological_indices": np.array_equal(
            arrays["sample_index"], np.arange(2528)
        ),
        "feature_dimensions": (
            arrays["global8"].shape == (2528, 8)
            and arrays["structured108"].shape == (2528, 108)
        ),
        "structured_prefix_exact": np.array_equal(
            arrays["global8"], arrays["structured108"][:, :8]
        ),
        "fixed_split_coverage": (
            SPLITS["fit"] == (0, 1264)
            and SPLITS["development"] == (1456, 1888)
            and SPLITS["audit"] == (2080, 2528)
        ),
    }
    if not all(protocol_checks.values()):
        raise RuntimeError(f"Step 22.20 protocol checks failed: {protocol_checks}")

    complete_valid = arrays["complete_teacher_valid"].astype(bool)
    observed_valid = arrays["observed_teacher_valid"].astype(bool)
    complete_actions = arrays["complete_teacher_actions"].astype(np.int64)
    complete_alpha = arrays["complete_teacher_alpha"].astype(np.float64)
    fit_start, fit_stop = SPLITS["fit"]
    fit_slice = slice(fit_start, fit_stop)
    fit_valid = complete_valid[fit_slice]
    y_action_fit = complete_actions[fit_slice][fit_valid]
    y_alpha_fit = complete_alpha[fit_slice][fit_valid]
    if set(np.unique(y_action_fit).tolist()) != {0, 1, 2}:
        raise RuntimeError("fit split does not contain all three frozen ARD actions")

    probe_results: dict[str, Any] = {}
    fitted_models: dict[str, tuple[Any, Any]] = {}
    for feature_name in ("global8", "structured108"):
        features = arrays[feature_name].astype(np.float64)
        action_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight=None,
                max_iter=2000,
                random_state=2220,
            ),
        )
        alpha_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        action_model.fit(features[fit_slice][fit_valid], y_action_fit)
        alpha_model.fit(features[fit_slice][fit_valid], y_alpha_fit)
        fitted_models[feature_name] = (action_model, alpha_model)
        probe_results[feature_name] = {}
        for split_name in ("development", "audit"):
            start, stop = SPLITS[split_name]
            split_slice = slice(start, stop)
            split_valid = complete_valid[split_slice]
            probe_results[feature_name][split_name] = {
                "action": evaluate_action_probe(
                    action_model,
                    features[split_slice],
                    complete_actions[split_slice],
                    split_valid,
                ),
                "alpha": evaluate_alpha_probe(
                    alpha_model,
                    features[split_slice],
                    complete_alpha[split_slice],
                    split_valid,
                    arrays["backbone_sse"][split_slice].astype(np.float64),
                    arrays["alignment_cross"][split_slice].astype(np.float64),
                    arrays["residual_energy"][split_slice].astype(np.float64),
                    arrays["complete_count"][split_slice].astype(np.float64),
                ),
            }

    improvements: dict[str, dict[str, float]] = {}
    feature_conditions = []
    for split_name in ("development", "audit"):
        global_result = probe_results["global8"][split_name]
        structured_result = probe_results["structured108"][split_name]
        improvements[split_name] = {
            "balanced_accuracy": float(
                structured_result["action"]["balanced_accuracy"]
                - global_result["action"]["balanced_accuracy"]
            ),
            "alpha_spearman": float(
                structured_result["alpha"]["spearman"]
                - global_result["alpha"]["spearman"]
            ),
            "counterfactual_mse": float(
                structured_result["alpha"]["counterfactual_mse"]
                - global_result["alpha"]["counterfactual_mse"]
            ),
        }
        feature_conditions.extend([
            improvements[split_name]["balanced_accuracy"] >= 0.05,
            improvements[split_name]["alpha_spearman"] >= 0.10,
            improvements[split_name]["counterfactual_mse"] <= 1e-12,
        ])
    feature_bottleneck = all(feature_conditions)

    probabilities = arrays["action_probabilities"].astype(np.float64)
    partial = arrays["partial_alpha"].astype(np.float64)
    full_contribution = probabilities[:, 2]
    partial_contribution = probabilities[:, 1] * partial
    gate = arrays["student_gate"].astype(np.float64)
    mixture = {
        "gate_mean": float(gate.mean()),
        "gate_std": float(gate.std(ddof=0)),
        "full_contribution_mean": float(full_contribution.mean()),
        "full_contribution_std": float(full_contribution.std(ddof=0)),
        "partial_contribution_mean": float(partial_contribution.mean()),
        "partial_contribution_std": float(partial_contribution.std(ddof=0)),
        "contribution_pearson": linear_correlation(
            full_contribution, partial_contribution
        ),
        "gate_reconstruction_max_abs": float(np.max(np.abs(
            gate - full_contribution - partial_contribution
        ))),
    }
    mixture_conditions = {
        "gate_std_le_0.01": mixture["gate_std"] <= 0.01,
        "one_contribution_std_ge_0.01": max(
            mixture["full_contribution_std"],
            mixture["partial_contribution_std"],
        ) >= 0.01,
        "contribution_pearson_le_minus_0.5": (
            mixture["contribution_pearson"] <= -0.5
        ),
    }
    mixture_cancellation = all(mixture_conditions.values())

    shared_valid = complete_valid & observed_valid
    teacher_comparison = {
        "complete_valid_samples": int(complete_valid.sum()),
        "observed_valid_samples": int(observed_valid.sum()),
        "shared_valid_samples": int(shared_valid.sum()),
        "action_agreement": float(np.mean(
            complete_actions[shared_valid]
            == arrays["observed_teacher_actions"][shared_valid]
        )),
        "alpha_mae": float(np.mean(np.abs(
            complete_alpha[shared_valid]
            - arrays["observed_teacher_alpha"][shared_valid]
        ))),
        "alpha_spearman": rank_correlation(
            complete_alpha[shared_valid],
            arrays["observed_teacher_alpha"][shared_valid],
        ),
    }
    student_actions = probabilities.argmax(axis=1)
    student_vs_teacher = {
        "action_accuracy": float(accuracy_score(
            complete_actions[complete_valid], student_actions[complete_valid]
        )),
        "action_balanced_accuracy": float(balanced_accuracy_score(
            complete_actions[complete_valid], student_actions[complete_valid]
        )),
        "gate_alpha_spearman": rank_correlation(
            gate[complete_valid], complete_alpha[complete_valid]
        ),
        "gate_alpha_mae": float(mean_absolute_error(
            complete_alpha[complete_valid], gate[complete_valid]
        )),
    }

    if feature_bottleneck and mixture_cancellation:
        decision = "STRUCTURED_TOKEN_ENCODER_PLUS_DISCRETE_ACTION_EXECUTION"
    elif feature_bottleneck:
        decision = "STRUCTURED_TOKEN_ENCODER_KEEP_EXPECTATION_MIXTURE"
    elif mixture_cancellation:
        decision = "KEEP_GLOBAL8_USE_DISCRETE_ACTION_EXECUTION"
    else:
        decision = "ARD_ROUTE_STOP_NO_CAPACITY_EXPANSION"
    payload = {
        "status": "PASS",
        "decision": decision,
        "feature_information_bottleneck": (
            "FEATURE_INFORMATION_BOTTLENECK_CONFIRMED"
            if feature_bottleneck else "FEATURE_INFORMATION_BOTTLENECK_NOT_CONFIRMED"
        ),
        "mixture_cancellation": (
            "MIXTURE_CANCELLATION_CONFIRMED"
            if mixture_cancellation else "MIXTURE_CANCELLATION_NOT_CONFIRMED"
        ),
        "test_accessed": False,
        "run_dir": str(run_dir),
        "samples_path": str(samples_path),
        "protocol_checks": protocol_checks,
        "fixed_splits": {key: list(value) for key, value in SPLITS.items()},
        "probe_hyperparameters": {
            "classification": {
                "model": "multinomial_logistic_regression",
                "standard_scaler": True,
                "C": 1.0,
                "class_weight": None,
                "max_iter": 2000,
                "random_state": 2220,
            },
            "regression": {
                "model": "ridge",
                "standard_scaler": True,
                "alpha": 1.0,
                "prediction_clip": [0.0, 1.0],
            },
        },
        "probe_results": probe_results,
        "structured_minus_global": improvements,
        "feature_bottleneck_conditions": {
            "development_and_audit_balanced_accuracy_gain_ge_0.05": all(
                improvements[name]["balanced_accuracy"] >= 0.05
                for name in ("development", "audit")
            ),
            "development_and_audit_spearman_gain_ge_0.10": all(
                improvements[name]["alpha_spearman"] >= 0.10
                for name in ("development", "audit")
            ),
            "development_and_audit_counterfactual_mse_nonworse": all(
                improvements[name]["counterfactual_mse"] <= 1e-12
                for name in ("development", "audit")
            ),
        },
        "mixture_diagnostic": mixture,
        "mixture_conditions": mixture_conditions,
        "teacher_comparison": teacher_comparison,
        "student_vs_complete_teacher": student_vs_teacher,
        "validation_metrics": {
            "mse": float(metrics["mse"]),
            "mae": float(metrics["mae"]),
        },
        "interpretation_boundary": (
            "validation-only failure attribution; any ARD-v2 must be frozen before "
            "and confirmed on the new seed2026 asset"
        ),
    }
    return payload


def markdown(payload: dict[str, Any]) -> str:
    if payload.get("status") != "PASS":
        return "# Step 22.20 ARD信息诊断\n\n当前状态：IN_PROGRESS。\n"
    rows = []
    for feature_name, label in (
        ("global8", "全局8维"),
        ("structured108", "结构化108维"),
    ):
        for split_name, split_label in (
            ("development", "development（开发段）"),
            ("audit", "audit（审计段）"),
        ):
            result = payload["probe_results"][feature_name][split_name]
            rows.append(
                f"| {label} | {split_label} | "
                f"{result['action']['balanced_accuracy']:.6f} | "
                f"{result['action']['cross_entropy']:.6f} | "
                f"{result['alpha']['spearman']:.6f} | "
                f"{result['alpha']['mae']:.6f} | "
                f"{result['alpha']['counterfactual_mse']:.9f} |"
            )
    mixture = payload["mixture_diagnostic"]
    teacher = payload["teacher_comparison"]
    return "\n".join([
        "# Step 22.20 ARD信息瓶颈诊断",
        "",
        f"- 总决策：`{payload['decision']}`",
        f"- 特征判定：`{payload['feature_information_bottleneck']}`",
        f"- 混合判定：`{payload['mixture_cancellation']}`",
        "- 数据边界：只使用chronological validation（按时间排序的验证集），未访问test。",
        "",
        "## 时间隔离探针",
        "",
        "| 特征 | 区间 | 动作平衡准确率 | 动作交叉熵 | alpha Spearman秩相关 | alpha平均绝对误差 | 预测alpha反事实MSE |",
        "|---|---|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## 混合抵消诊断",
        "",
        f"- 最终门标准差：{mixture['gate_std']:.9f}",
        f"- FULL（全开）贡献标准差：{mixture['full_contribution_std']:.9f}",
        f"- PARTIAL（部分开启）贡献标准差：{mixture['partial_contribution_std']:.9f}",
        f"- 两贡献Pearson线性相关：{mixture['contribution_pearson']:.9f}",
        "",
        "## 教师可靠性",
        "",
        f"- 完整目标与仅观测目标动作一致率：{teacher['action_agreement']:.6f}",
        f"- 两种解析alpha平均绝对误差：{teacher['alpha_mae']:.6f}",
        f"- 两种解析alpha Spearman秩相关：{teacher['alpha_spearman']:.6f}",
        "",
        "该结果只用于冻结下一版结构；若支持ARD-v2，必须先冻结结构与训练协议，再使用新的seed2026固定缺失资产确认精度。",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else latest_formal_run()
    if run_dir is None:
        payload = {
            "status": "IN_PROGRESS",
            "decision": "WAITING_FOR_FORMAL_DIAGNOSTIC",
            "test_accessed": False,
        }
    else:
        payload = analyse(run_dir)
    atomic_json(REPORT_JSON, payload)
    atomic_text(REPORT_MD, markdown(payload))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
