#!/usr/bin/env python3
"""Validation-only temporal generalization audit for Step-8 FUSR V3.

The script never reads test artifacts.  It reuses the frozen V2 validation
counterfactual cache, fits all normalizers/models on the chronological fit
block, locks each routing threshold on the later dev block, and evaluates the
unchanged rule on the final audit block.  The final block is exploratory
validation evidence, not a test set or a publication-grade estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    ROOT
    / "runs"
    / "s4m_fusr_v2__weather__point__mr0.06__seed2023__20260718_193121"
)
DEFAULT_CACHE = DEFAULT_RUN / "fusr_v2_validation_cache.pt"
DEFAULT_CHECKPOINT = next((DEFAULT_RUN / "checkpoints").glob("*/checkpoint.pth"))
DEFAULT_JSON = ROOT / "reports" / "stage8_fusr_v3_temporal_diagnosis.json"
DEFAULT_MD = ROOT / "reports" / "stage8_fusr_v3_temporal_diagnosis.md"

FEATURE_NAMES = (
    "top1_similarity",
    "selected_similarity",
    "normalized_retrieval_entropy",
    "missing_rate",
    "longest_gap_ratio",
    "variable_missing_rate_std",
    "mask_binary_entropy",
    "spectral_js_divergence",
    "top1_similarity_temporal_std",
    "normalized_retrieval_entropy_temporal_std",
    "retrieved_query_norm_ratio",
    "retrieved_query_cosine_mean",
    "context_relative_perturbation_norm",
    "context_on_off_cosine",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def finite_float(value: Any) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return None
    return finite_float(np.corrcoef(x, y)[0, 1])


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.bool_)
    scores = np.asarray(scores, dtype=np.float64)
    positive = int(labels.sum())
    negative = int(labels.size - positive)
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    rank_sum = float(ranks[labels].sum())
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.bool_)
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    ranked = labels[order].astype(np.float64)
    precision = np.cumsum(ranked) / np.arange(1, ranked.size + 1)
    return float(np.sum(precision * ranked) / positives)


def chronological_split(sample_count: int) -> dict[str, np.ndarray]:
    """Reproduce the exact V2 formal split: 55% fit, 20% dev, 288 embargo."""
    fit_count = max(4, int(sample_count * 0.55))
    dev_count = max(4, int(sample_count * 0.20))
    embargo = min(288, max(0, (sample_count - fit_count - dev_count - 4) // 2))
    dev_start = fit_count + embargo
    dev_end = dev_start + dev_count
    audit_start = dev_end + embargo
    if audit_start >= sample_count - 3:
        raise RuntimeError("chronological split leaves no final validation audit block")
    return {
        "fit": np.arange(0, fit_count, dtype=np.int64),
        "embargo_1": np.arange(fit_count, dev_start, dtype=np.int64),
        "dev": np.arange(dev_start, dev_end, dtype=np.int64),
        "embargo_2": np.arange(dev_end, audit_start, dtype=np.int64),
        "audit": np.arange(audit_start, sample_count, dtype=np.int64),
    }


def contiguous_groups(indices: np.ndarray, groups: int) -> list[np.ndarray]:
    return [part.astype(np.int64) for part in np.array_split(indices, groups) if part.size]


def calibrate_threshold(
    scores: np.ndarray, mse_on: np.ndarray, mse_off: np.ndarray
) -> dict[str, Any]:
    """Choose the exact empirical-risk-minimizing score threshold."""
    scores = np.asarray(scores, dtype=np.float64)
    mse_on = np.asarray(mse_on, dtype=np.float64)
    mse_off = np.asarray(mse_off, dtype=np.float64)
    if scores.size == 0:
        raise ValueError("cannot calibrate on an empty block")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_on = mse_on[order]
    sorted_off = mse_off[order]
    prefix_on = np.concatenate(([0.0], np.cumsum(sorted_on)))
    prefix_off = np.concatenate(([0.0], np.cumsum(sorted_off)))
    total_off = float(prefix_off[-1])
    boundaries = [0]
    boundaries.extend(
        index
        for index in range(1, scores.size)
        if sorted_scores[index] != sorted_scores[index - 1]
    )
    boundaries.append(scores.size)
    risks = {
        int(k): float((prefix_on[k] + total_off - prefix_off[k]) / scores.size)
        for k in boundaries
    }
    best_risk = min(risks.values())
    tolerance = max(1e-12, abs(best_risk) * 1e-12)
    # Prefer prototype-on on exact ties so a selector cannot manufacture a
    # nontrivial gate without an empirical validation benefit.
    selected = max(k for k, risk in risks.items() if abs(risk - best_risk) <= tolerance)
    if selected == 0:
        threshold = float(np.nextafter(sorted_scores[0], np.inf))
    elif selected == scores.size:
        threshold = float(np.nextafter(sorted_scores[-1], -np.inf))
    else:
        threshold = float(0.5 * (sorted_scores[selected - 1] + sorted_scores[selected]))
    gate = scores >= threshold
    routed = np.where(gate, mse_on, mse_off)
    return {
        "threshold": threshold,
        "gate_rate": float(gate.mean()),
        "risk": float(routed.mean()),
        "always_on_risk": float(mse_on.mean()),
        "always_off_risk": float(mse_off.mean()),
        "relative_gain_vs_always_on": float((mse_on.mean() - routed.mean()) / mse_on.mean()),
    }


def evaluate_rule(
    scores: np.ndarray,
    threshold: float,
    mse_on: np.ndarray,
    mse_off: np.ndarray,
    labels: np.ndarray,
    utility: np.ndarray,
) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float64)
    gate = scores >= float(threshold)
    routed = np.where(gate, mse_on, mse_off)
    always_on = float(np.mean(mse_on))
    harmful = ~np.asarray(labels, dtype=np.bool_)
    used_harmful = gate & harmful
    return {
        "sample_count": int(scores.size),
        "positive_rate": float(np.mean(labels)),
        "utility_mean": float(np.mean(utility)),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "auroc": binary_auc(labels, scores),
        "auprc": average_precision(labels, scores),
        "gate_rate": float(np.mean(gate)),
        "abstention_rate": float(1.0 - np.mean(gate)),
        "always_on_mse": always_on,
        "always_off_mse": float(np.mean(mse_off)),
        "routed_mse": float(np.mean(routed)),
        "absolute_gain_vs_always_on": float(always_on - np.mean(routed)),
        "relative_gain_vs_always_on": float((always_on - np.mean(routed)) / always_on),
        "harmful_use_rate": float(np.mean(used_harmful)),
    }


def bootstrap_gain_interval(
    scores: np.ndarray,
    threshold: float,
    mse_on: np.ndarray,
    mse_off: np.ndarray,
    seed: int,
    draws: int = 2000,
) -> dict[str, float]:
    gate = np.asarray(scores) >= float(threshold)
    improvement = np.asarray(mse_on) - np.where(gate, mse_on, mse_off)
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sample = rng.integers(0, improvement.size, size=improvement.size)
        means[draw] = float(np.mean(improvement[sample]))
    return {
        "absolute_gain_mean": float(np.mean(improvement)),
        "absolute_gain_ci95_low": float(np.quantile(means, 0.025)),
        "absolute_gain_ci95_high": float(np.quantile(means, 0.975)),
        "bootstrap_draws": int(draws),
    }


class LinearScorer(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.output = nn.Linear(feature_dim, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.output(values).squeeze(-1)


class MLPScorer(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


@dataclass
class FittedScorer:
    name: str
    scores: np.ndarray
    metadata: dict[str, Any]


def weighted_bce(
    logits: torch.Tensor, labels: torch.Tensor, utility: torch.Tensor
) -> torch.Tensor:
    positives = labels.sum().clamp_min(1.0)
    negatives = (1.0 - labels).sum().clamp_min(1.0)
    count = float(labels.numel())
    weights = torch.where(
        labels > 0.5,
        labels.new_tensor(count / (2.0 * float(positives.item()))),
        labels.new_tensor(count / (2.0 * float(negatives.item()))),
    )
    weights = weights * utility.abs().clamp_min(1e-4)
    each = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (each * weights).sum() / weights.sum().clamp_min(1e-8)


def fit_neural_scorer(
    name: str,
    features: np.ndarray,
    utility: np.ndarray,
    labels: np.ndarray,
    fit_index: np.ndarray,
    dev_index: np.ndarray,
    feature_index: np.ndarray,
    architecture: str,
    robust_groups: list[np.ndarray] | None,
    seed: int,
    epochs: int = 300,
    patience: int = 40,
) -> FittedScorer:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    center = features[fit_index][:, feature_index].mean(axis=0)
    scale = features[fit_index][:, feature_index].std(axis=0)
    scale = np.maximum(scale, 1e-6)
    normalized = ((features[:, feature_index] - center) / scale).astype(np.float32)
    x = torch.from_numpy(normalized)
    y = torch.from_numpy(labels.astype(np.float32))
    u = torch.from_numpy(utility.astype(np.float32))
    model: nn.Module
    if architecture == "linear":
        model = LinearScorer(feature_index.size)
        learning_rate = 3e-3
    elif architecture == "mlp":
        model = MLPScorer(feature_index.size)
        learning_rate = 1e-3
    else:
        raise ValueError(f"unknown architecture: {architecture}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_state: dict[str, torch.Tensor] | None = None
    best_dev_loss = float("inf")
    best_epoch = 0
    stale = 0
    fit_tensor = torch.from_numpy(fit_index)
    dev_tensor = torch.from_numpy(dev_index)
    group_tensors = (
        [torch.from_numpy(group) for group in robust_groups]
        if robust_groups is not None
        else None
    )
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if group_tensors is None:
            train_loss = weighted_bce(model(x[fit_tensor]), y[fit_tensor], u[fit_tensor])
        else:
            group_losses = torch.stack(
                [weighted_bce(model(x[group]), y[group], u[group]) for group in group_tensors]
            )
            # Smooth worst-environment risk. beta=8 retains gradients from all
            # chronological environments while emphasizing the least stable one.
            train_loss = torch.logsumexp(8.0 * group_losses, dim=0) / 8.0
        if not torch.isfinite(train_loss):
            raise RuntimeError(f"{name} produced a non-finite training loss")
        train_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            dev_loss = weighted_bce(model(x[dev_tensor]), y[dev_tensor], u[dev_tensor])
        value = float(dev_loss.item())
        if value < best_dev_loss - 1e-10:
            best_dev_loss = value
            best_epoch = epoch
            best_state = {
                key: tensor.detach().clone() for key, tensor in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError(f"{name} did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores = model(x).numpy().astype(np.float64)
    return FittedScorer(
        name=name,
        scores=scores,
        metadata={
            "architecture": architecture,
            "feature_indices": feature_index.tolist(),
            "feature_names": [FEATURE_NAMES[index] for index in feature_index],
            "best_epoch": best_epoch,
            "best_dev_weighted_bce": best_dev_loss,
            "robust_objective": "smooth_worst_chronological_group" if group_tensors else "pooled",
            "fit_only_normalization": True,
            "seed": seed,
        },
    )


def fit_temporal_ridge_ensembles(
    features: np.ndarray,
    utility: np.ndarray,
    fit_groups: list[np.ndarray],
) -> list[FittedScorer]:
    fit_index = np.concatenate(fit_groups)
    center = features[fit_index].mean(axis=0)
    scale = np.maximum(features[fit_index].std(axis=0), 1e-6)
    values = (features - center) / scale
    design = np.concatenate((values, np.ones((values.shape[0], 1))), axis=1)
    member_scores = []
    member_coefficients = []
    for group in fit_groups:
        local = design[group]
        penalty = np.eye(local.shape[1], dtype=np.float64)
        penalty[-1, -1] = 0.0
        coefficient = np.linalg.solve(
            local.T @ local + 1.0 * penalty,
            local.T @ utility[group],
        )
        raw = design @ coefficient
        local_mean = float(np.mean(raw[group]))
        local_std = max(float(np.std(raw[group])), 1e-6)
        member_scores.append((raw - local_mean) / local_std)
        member_coefficients.append(coefficient.tolist())
    stack = np.stack(member_scores, axis=0)
    common = {
        "architecture": "chronological_block_ridge_ensemble",
        "ridge_penalty": 1.0,
        "fit_groups": len(fit_groups),
        "fit_only_normalization": True,
        "member_coefficients": member_coefficients,
    }
    return [
        FittedScorer(
            "temporal_ridge_mean",
            np.mean(stack, axis=0),
            {**common, "aggregation": "mean"},
        ),
        FittedScorer(
            "temporal_ridge_median",
            np.median(stack, axis=0),
            {**common, "aggregation": "median"},
        ),
        FittedScorer(
            "temporal_ridge_lower_quartile",
            np.quantile(stack, 0.25, axis=0),
            {**common, "aggregation": "lower_quartile"},
        ),
    ]


def load_v2_scores(
    checkpoint_path: Path, features: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    prefix = "mem_net.fusr."
    router_state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    required = (
        "backbone.0.weight",
        "backbone.0.bias",
        "backbone.2.weight",
        "backbone.2.bias",
        "utility_head.weight",
        "utility_head.bias",
        "feature_center",
        "feature_scale",
        "calibrated_threshold",
    )
    missing = [key for key in required if key not in router_state]
    if missing:
        raise KeyError(f"V2 checkpoint is missing router keys: {missing}")
    values = torch.from_numpy(features.astype(np.float32))
    center = router_state["feature_center"].float()
    scale = router_state["feature_scale"].float().clamp_min(1e-6)
    hidden = nn.functional.linear(
        (values - center) / scale,
        router_state["backbone.0.weight"].float(),
        router_state["backbone.0.bias"].float(),
    )
    hidden = nn.functional.gelu(hidden)
    hidden = nn.functional.layer_norm(
        hidden,
        (hidden.shape[-1],),
        router_state["backbone.2.weight"].float(),
        router_state["backbone.2.bias"].float(),
    )
    raw = nn.functional.linear(
        hidden,
        router_state["utility_head.weight"].float(),
        router_state["utility_head.bias"].float(),
    )
    return raw[:, 2].numpy().astype(np.float64), {
        "source": "formal_v2_checkpoint",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "saved_threshold": float(router_state["calibrated_threshold"].item()),
    }


def block_statistics(
    name: str,
    index: np.ndarray,
    features: np.ndarray,
    utility: np.ndarray,
    mse_on: np.ndarray,
    mse_off: np.ndarray,
    fit_center: np.ndarray,
    fit_scale: np.ndarray,
) -> dict[str, Any]:
    feature_mean = features[index].mean(axis=0)
    drift = (feature_mean - fit_center) / fit_scale
    correlations = [safe_corr(features[index, column], utility[index]) for column in range(features.shape[1])]
    return {
        "name": name,
        "start": int(index[0]),
        "end_exclusive": int(index[-1] + 1),
        "sample_count": int(index.size),
        "positive_rate": float(np.mean(utility[index] > 0.0)),
        "utility_mean": float(np.mean(utility[index])),
        "utility_std": float(np.std(utility[index])),
        "mse_on": float(np.mean(mse_on[index])),
        "mse_off": float(np.mean(mse_off[index])),
        "oracle_mse": float(np.mean(np.minimum(mse_on[index], mse_off[index]))),
        "oracle_relative_gain_vs_always_on": float(
            (np.mean(mse_on[index]) - np.mean(np.minimum(mse_on[index], mse_off[index])))
            / np.mean(mse_on[index])
        ),
        "feature_mean_z_shift_vs_fit": {
            FEATURE_NAMES[column]: float(drift[column]) for column in range(features.shape[1])
        },
        "feature_utility_pearson": {
            FEATURE_NAMES[column]: correlations[column] for column in range(features.shape[1])
        },
    }


def candidate_record(
    scorer: FittedScorer,
    split: dict[str, np.ndarray],
    mse_on: np.ndarray,
    mse_off: np.ndarray,
    labels: np.ndarray,
    utility: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    dev = split["dev"]
    audit = split["audit"]
    calibration = calibrate_threshold(scorer.scores[dev], mse_on[dev], mse_off[dev])
    threshold = float(calibration["threshold"])
    blocks = {
        name: evaluate_rule(
            scorer.scores[index],
            threshold,
            mse_on[index],
            mse_off[index],
            labels[index],
            utility[index],
        )
        for name, index in (("fit", split["fit"]), ("dev", dev), ("audit", audit))
    }
    audit_ci = bootstrap_gain_interval(
        scorer.scores[audit], threshold, mse_on[audit], mse_off[audit], seed
    )
    audit_metrics = blocks["audit"]
    return {
        "name": scorer.name,
        "metadata": scorer.metadata,
        "threshold_source": "chronological_validation_dev_only",
        "calibration": calibration,
        "blocks": blocks,
        "audit_bootstrap": audit_ci,
        "passes_v3_offline_gate": bool(
            audit_metrics["relative_gain_vs_always_on"] > 0.0
            and 0.0 < audit_metrics["gate_rate"] < 1.0
            and audit_ci["absolute_gain_ci95_low"] > 0.0
        ),
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# 步骤8：FUSR V3时序泛化诊断",
        "",
        f"- 生成时间：`{report['created_at']}`",
        f"- 缓存：`{report['cache']['path']}`",
        f"- 样本数：{report['cache']['sample_count']}；特征数：{report['cache']['feature_dim']}",
        "- 数据边界：只使用validation（验证集）缓存；test（测试集）未用于训练、阈值或候选选择。",
        "- 规则：fit拟合模型，dev锁定阈值，audit只做最后验证块审计。",
        "",
        "## 时间块诊断",
        "",
        "| 时间块 | 样本 | 正效用率 | 平均效用 | always-on MSE | Oracle增益 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for block in report["temporal_blocks"]:
        lines.append(
            f"| {block['name']} | {block['sample_count']} | {block['positive_rate']:.4f} | "
            f"{block['utility_mean']:.5f} | {block['mse_on']:.6f} | "
            f"{100.0 * block['oracle_relative_gain_vs_always_on']:.3f}% |"
        )
    lines.extend(
        [
            "",
            "## 候选方法",
            "",
            "| 候选 | fit AUROC | dev AUROC | audit AUROC | dev门控率 | audit门控率 | audit MSE增益 | 95% CI下界 | 离线门禁 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for candidate in report["candidates"]:
        blocks = candidate["blocks"]
        ci = candidate["audit_bootstrap"]
        def fmt(value: float | None) -> str:
            return "NA" if value is None else f"{value:.4f}"
        lines.append(
            f"| {candidate['name']} | {fmt(blocks['fit']['auroc'])} | "
            f"{fmt(blocks['dev']['auroc'])} | {fmt(blocks['audit']['auroc'])} | "
            f"{blocks['dev']['gate_rate']:.4f} | {blocks['audit']['gate_rate']:.4f} | "
            f"{100.0 * blocks['audit']['relative_gain_vs_always_on']:.4f}% | "
            f"{ci['absolute_gain_ci95_low']:.6g} | "
            f"{'PASS' if candidate['passes_v3_offline_gate'] else 'FAIL'} |"
        )
    decision = report["decision"]
    lines.extend(
        [
            "",
            "## 当前结论",
            "",
            f"- 决策代码：`{decision['code']}`",
            f"- 推荐候选：`{decision.get('selected_candidate') or '无'}`",
            f"- 结论：{decision['summary_zh']}",
            "- 本报告只决定是否值得进入V3 GPU smoke，不能证明跨数据集、跨seed或SCI分区水平。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()
    if not args.cache.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError(f"missing cache/checkpoint: {args.cache}, {args.checkpoint}")

    cache = torch.load(args.cache, map_location="cpu")
    required = ("features", "mse_on", "mse_off", "sample_index", "source_split", "used_test_targets")
    missing = [key for key in required if key not in cache]
    if missing:
        raise KeyError(f"cache is missing keys: {missing}")
    if cache["source_split"] != "validation" or cache["used_test_targets"] is not False:
        raise RuntimeError("V3 audit accepts validation-only caches with used_test_targets=False")
    features = cache["features"].float().numpy().astype(np.float64)
    mse_on = cache["mse_on"].float().numpy().astype(np.float64)
    mse_off = cache["mse_off"].float().numpy().astype(np.float64)
    sample_index = cache["sample_index"].numpy().astype(np.int64)
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError(f"expected [N,{len(FEATURE_NAMES)}] features, got {features.shape}")
    if not np.array_equal(sample_index, np.sort(sample_index)):
        raise RuntimeError("cache sample_index is not chronological")
    if not all(np.isfinite(value).all() for value in (features, mse_on, mse_off)):
        raise RuntimeError("cache contains non-finite values")
    symmetric = np.maximum(0.5 * (mse_on + mse_off), 1e-8)
    utility = np.clip((mse_off - mse_on) / symmetric, -1.0, 1.0)
    labels = utility > 0.0
    split = chronological_split(features.shape[0])
    fit_index = split["fit"]
    fit_groups = contiguous_groups(fit_index, 5)
    fit_center = features[fit_index].mean(axis=0)
    fit_scale = np.maximum(features[fit_index].std(axis=0), 1e-6)

    temporal_blocks = [
        block_statistics(
            name,
            index,
            features,
            utility,
            mse_on,
            mse_off,
            fit_center,
            fit_scale,
        )
        for name, index in (
            ("fit", split["fit"]),
            ("embargo_1", split["embargo_1"]),
            ("dev", split["dev"]),
            ("embargo_2", split["embargo_2"]),
            ("audit", split["audit"]),
        )
    ]

    # Feature stability is determined on fit environments only.  It is not
    # revised after seeing dev/audit outcomes.
    environment_correlations = np.array(
        [
            [safe_corr(features[group, column], utility[group]) for column in range(features.shape[1])]
            for group in fit_groups
        ],
        dtype=np.float64,
    )
    signs = np.sign(np.nan_to_num(environment_correlations, nan=0.0))
    dominant_sign_fraction = np.maximum(
        np.mean(signs > 0.0, axis=0), np.mean(signs < 0.0, axis=0)
    )
    mean_abs_correlation = np.array(
        [
            float(np.mean(np.abs(column[np.isfinite(column)])))
            if np.isfinite(column).any()
            else 0.0
            for column in environment_correlations.T
        ],
        dtype=np.float64,
    )
    stable_feature_index = np.flatnonzero(
        (dominant_sign_fraction >= 0.8) & (mean_abs_correlation >= 0.02)
    )
    if stable_feature_index.size == 0:
        stable_feature_index = np.arange(features.shape[1], dtype=np.int64)

    all_features = np.arange(features.shape[1], dtype=np.int64)
    scorers: list[FittedScorer] = []
    v2_scores, v2_metadata = load_v2_scores(args.checkpoint, features)
    scorers.append(FittedScorer("v2_checkpoint_reference", v2_scores, v2_metadata))
    scorers.append(
        fit_neural_scorer(
            "pooled_linear",
            features,
            utility,
            labels,
            fit_index,
            split["dev"],
            all_features,
            "linear",
            None,
            args.seed + 301,
        )
    )
    scorers.append(
        fit_neural_scorer(
            "stable_feature_linear",
            features,
            utility,
            labels,
            fit_index,
            split["dev"],
            stable_feature_index,
            "linear",
            None,
            args.seed + 302,
        )
    )
    scorers.append(
        fit_neural_scorer(
            "pooled_mlp_retrain",
            features,
            utility,
            labels,
            fit_index,
            split["dev"],
            all_features,
            "mlp",
            None,
            args.seed + 303,
        )
    )
    scorers.append(
        fit_neural_scorer(
            "worst_group_mlp",
            features,
            utility,
            labels,
            fit_index,
            split["dev"],
            all_features,
            "mlp",
            fit_groups,
            args.seed + 304,
        )
    )
    scorers.extend(fit_temporal_ridge_ensembles(features, utility, fit_groups))

    candidates = [
        candidate_record(
            scorer, split, mse_on, mse_off, labels, utility, args.seed + index
        )
        for index, scorer in enumerate(scorers)
    ]
    passing = [candidate for candidate in candidates if candidate["passes_v3_offline_gate"]]
    passing.sort(
        key=lambda item: item["blocks"]["audit"]["relative_gain_vs_always_on"],
        reverse=True,
    )
    if passing:
        selected = passing[0]
        decision = {
            "code": "PROMISING_IMPLEMENT_V3_GPU_SMOKE",
            "selected_candidate": selected["name"],
            "summary_zh": (
                "至少一个预声明候选在后段验证审计块实现非全开/非全关路由，"
                "且bootstrap 95%区间下界仍为正；可以把该候选接入可关闭V3并先跑GPU smoke。"
            ),
        }
    else:
        best = max(
            candidates,
            key=lambda item: item["blocks"]["audit"]["relative_gain_vs_always_on"],
        )
        decision = {
            "code": "NO_ROBUST_CANDIDATE_DO_NOT_SPEND_GPU",
            "selected_candidate": None,
            "best_exploratory_candidate": best["name"],
            "summary_zh": (
                "没有候选同时满足后段验证MSE改善、选择性门控和95%区间下界为正；"
                "当前14维特征不足以支撑稳健V3，暂不盲目启动GPU实验。"
            ),
        }

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "step": 8,
        "purpose": "validation_only_fusr_v3_temporal_generalization_screen",
        "cache": {
            "path": str(args.cache.resolve()),
            "file_sha256": file_sha256(args.cache),
            "embedded_sha256": cache.get("sha256"),
            "source_split": cache["source_split"],
            "used_test_targets": cache["used_test_targets"],
            "sample_count": int(features.shape[0]),
            "feature_dim": int(features.shape[1]),
            "sample_index_sorted": True,
        },
        "protocol": {
            "fit_model_block": [int(split["fit"][0]), int(split["fit"][-1] + 1)],
            "dev_threshold_block": [int(split["dev"][0]), int(split["dev"][-1] + 1)],
            "final_validation_audit_block": [int(split["audit"][0]), int(split["audit"][-1] + 1)],
            "embargo_windows_each": int(split["embargo_1"].size),
            "test_used_for_training_or_selection": False,
            "claim_boundary": "exploratory validation-only screen; not a test or multi-seed result",
        },
        "stable_feature_screen": {
            "rule": "fit environment dominant sign fraction >=0.8 and mean abs Pearson >=0.02",
            "selected_indices": stable_feature_index.tolist(),
            "selected_names": [FEATURE_NAMES[index] for index in stable_feature_index],
            "dominant_sign_fraction": {
                FEATURE_NAMES[index]: float(dominant_sign_fraction[index])
                for index in range(features.shape[1])
            },
            "mean_abs_environment_correlation": {
                FEATURE_NAMES[index]: finite_float(mean_abs_correlation[index])
                for index in range(features.shape[1])
            },
        },
        "temporal_blocks": temporal_blocks,
        "candidates": candidates,
        "decision": decision,
    }
    atomic_text(args.report_json, json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    atomic_text(args.report_md, markdown_report(report))
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    print(f"REPORT_JSON={args.report_json}")
    print(f"REPORT_MD={args.report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
