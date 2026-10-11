#!/usr/bin/env python3
"""Diagnose PG95/BMSA prototype-bank diversity without reading dataset targets."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
COMPARISON_PATH = ROOT / "reports" / "stage10_weather_multiseed_comparison.json"
JSON_PATH = ROOT / "reports" / "stage10_memory_diversity_diagnosis.json"
MARKDOWN_PATH = ROOT / "reports" / "stage10_memory_diversity_diagnosis.md"

PRIMARY_METRICS = {
    "centroid_nearest_neighbor_cosine_mean": "lower",
    "centroid_redundant_pair_fraction_ge_0_95": "lower",
    "centroid_centered_effective_rank_ratio": "higher",
    "stored_vector_centered_effective_rank_ratio": "higher",
    "slot_occupancy_normalized_entropy": "higher",
}


def finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"non-finite diagnostic value: {value}")
    return float(value)


def scalar(value: torch.Tensor) -> float:
    return finite(float(value.detach().cpu().item()))


def quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return scalar(torch.quantile(values.float(), q))


def effective_rank(values: torch.Tensor, centered: bool) -> tuple[float, float]:
    matrix = values.float()
    if centered:
        matrix = matrix - matrix.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(matrix)
    energy = singular.square()
    total = energy.sum()
    maximum_rank = max(1, min(matrix.shape))
    if scalar(total) <= 0.0:
        return 0.0, 0.0
    probability = energy / total
    probability = probability[probability > 0]
    rank = torch.exp(-(probability * probability.log()).sum())
    return scalar(rank), scalar(rank / float(maximum_rank))


def normalized_entropy(counts: torch.Tensor) -> float:
    counts = counts.float()
    counts = counts[counts > 0]
    if counts.numel() <= 1:
        return 0.0
    probability = counts / counts.sum()
    entropy = -(probability * probability.log()).sum()
    return scalar(entropy / math.log(int(counts.numel())))


def checkpoint_path(run_dir: Path) -> Path:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    path = Path(metrics["checkpoint"])
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint missing: {path}")
    return path


def diagnose_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    required = {"memory", "repre", "num_dic"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"checkpoint missing prototype keys: {missing}")

    centroids = payload["repre"].detach().cpu().float()
    memory = payload["memory"]
    num_dic = torch.as_tensor(payload["num_dic"], dtype=torch.long)
    if centroids.ndim != 2 or centroids.shape[0] < 2:
        raise ValueError(f"invalid centroid tensor shape: {tuple(centroids.shape)}")
    if not isinstance(memory, list) or len(memory) != centroids.shape[0]:
        raise ValueError(
            "memory/centroid mismatch: "
            f"memory={type(memory).__name__}:{len(memory) if isinstance(memory, list) else 'NA'} "
            f"centroids={centroids.shape[0]}"
        )

    stored_parts = []
    assignments = []
    actual_sizes = []
    for index, slot in enumerate(memory):
        tensor = slot.detach().cpu().float().reshape(-1, centroids.shape[1])
        if tensor.shape[0] < 1:
            raise ValueError(f"empty prototype slot: {index}")
        stored_parts.append(tensor)
        assignments.append(torch.full((tensor.shape[0],), index, dtype=torch.long))
        actual_sizes.append(int(tensor.shape[0]))
    stored = torch.cat(stored_parts, dim=0)
    assignment = torch.cat(assignments, dim=0)
    actual_count_tensor = torch.tensor(actual_sizes, dtype=torch.long)

    normalized_centroids = torch.nn.functional.normalize(centroids, dim=1)
    cosine = normalized_centroids @ normalized_centroids.T
    diagonal = torch.eye(cosine.shape[0], dtype=torch.bool)
    off_diagonal = cosine[~diagonal]
    upper = cosine[torch.triu_indices(cosine.shape[0], cosine.shape[1], offset=1).unbind()]
    nearest = cosine.masked_fill(diagonal, float("-inf")).max(dim=1).values

    centroid_rank, centroid_rank_ratio = effective_rank(centroids, centered=True)
    stored_rank, stored_rank_ratio = effective_rank(stored, centered=True)
    normalized_stored = torch.nn.functional.normalize(stored, dim=1)
    within_cosine = (
        normalized_stored * normalized_centroids[assignment]
    ).sum(dim=1)

    reported_counts = num_dic[: len(memory)]
    report = {
        "checkpoint": str(path),
        "checkpoint_size_bytes": path.stat().st_size,
        "centroid_count": int(centroids.shape[0]),
        "feature_dimension": int(centroids.shape[1]),
        "stored_vector_count": int(stored.shape[0]),
        "slot_size_min": int(actual_count_tensor.min().item()),
        "slot_size_max": int(actual_count_tensor.max().item()),
        "slot_size_mean": scalar(actual_count_tensor.float().mean()),
        "slot_occupancy_normalized_entropy": normalized_entropy(actual_count_tensor),
        "reported_num_dic_matches_actual_slots": bool(
            reported_counts.shape == actual_count_tensor.shape
            and torch.equal(reported_counts, actual_count_tensor)
        ),
        "reported_positive_slot_count": int((num_dic > 0).sum().item()),
        "centroid_norm_mean": scalar(centroids.norm(dim=1).mean()),
        "centroid_pair_cosine_mean": scalar(upper.mean()),
        "centroid_pair_cosine_std": scalar(upper.std(unbiased=False)),
        "centroid_pair_cosine_p50": quantile(upper, 0.50),
        "centroid_pair_cosine_p90": quantile(upper, 0.90),
        "centroid_pair_cosine_p95": quantile(upper, 0.95),
        "centroid_nearest_neighbor_cosine_mean": scalar(nearest.mean()),
        "centroid_nearest_neighbor_cosine_p95": quantile(nearest, 0.95),
        "centroid_redundant_pair_fraction_ge_0_90": scalar((upper >= 0.90).float().mean()),
        "centroid_redundant_pair_fraction_ge_0_95": scalar((upper >= 0.95).float().mean()),
        "centroid_redundant_pair_fraction_ge_0_99": scalar((upper >= 0.99).float().mean()),
        "centroid_centered_effective_rank": centroid_rank,
        "centroid_centered_effective_rank_ratio": centroid_rank_ratio,
        "stored_vector_centered_effective_rank": stored_rank,
        "stored_vector_centered_effective_rank_ratio": stored_rank_ratio,
        "within_slot_cosine_mean": scalar(within_cosine.mean()),
        "within_slot_cosine_p05": quantile(within_cosine, 0.05),
        "all_off_diagonal_values_finite": bool(torch.isfinite(off_diagonal).all().item()),
    }
    return report


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 步骤10：原型库多样性诊断",
        "",
        "本报告只读取正式checkpoint中的原型库，不加载数据集或测试目标。",
        "",
        "| seed | method | centroids | NN cosine↓ | pair≥0.95↓ | centroid rank ratio↑ | vector rank ratio↑ | slot entropy↑ | within cosine |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in ("2023", "2024", "2025"):
        for method in ("pg95", "bmsa50"):
            item = report["per_seed"][seed][method]
            lines.append(
                "| {seed} | {method} | {centroid_count} | {nn:.6f} | {redundant:.6f} | "
                "{centroid_rank:.6f} | {vector_rank:.6f} | {entropy:.6f} | {within:.6f} |".format(
                    seed=seed,
                    method=method,
                    centroid_count=item["centroid_count"],
                    nn=item["centroid_nearest_neighbor_cosine_mean"],
                    redundant=item["centroid_redundant_pair_fraction_ge_0_95"],
                    centroid_rank=item["centroid_centered_effective_rank_ratio"],
                    vector_rank=item["stored_vector_centered_effective_rank_ratio"],
                    entropy=item["slot_occupancy_normalized_entropy"],
                    within=item["within_slot_cosine_mean"],
                )
            )
    lines.extend(["", "## BMSA - PG95 配对差值", ""])
    for seed in ("2023", "2024", "2025"):
        lines.append(f"- seed {seed}:")
        for metric, direction in PRIMARY_METRICS.items():
            value = report["paired_deltas_bmsa_minus_pg95"][seed][metric]
            lines.append(f"  - `{metric}` ({direction} is better): {value:+.8f}")
    lines.extend(
        [
            "",
            "这些指标只用于检验原型结构假设；不能单独证明预测因果，也不用于从测试误差反向挑选联合准入权重。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    comparison = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
    per_seed: dict[str, dict[str, Any]] = {}
    deltas: dict[str, dict[str, float]] = {}
    for seed in ("2023", "2024", "2025"):
        per_seed[seed] = {}
        for method in ("pg95", "bmsa50"):
            run_dir = Path(comparison["per_seed"][seed][method]["run_dir"])
            per_seed[seed][method] = diagnose_checkpoint(checkpoint_path(run_dir))
        deltas[seed] = {
            metric: finite(per_seed[seed]["bmsa50"][metric] - per_seed[seed]["pg95"][metric])
            for metric in PRIMARY_METRICS
        }

    report = {
        "step": 10,
        "artifact_status": "PASS",
        "target_free": True,
        "primary_metrics_frozen_before_execution": PRIMARY_METRICS,
        "interpretation_boundary": (
            "Checkpoint-bank structure is descriptive; seed2023/2024 methods stopped at "
            "different epochs, while seed2025 has matched optimizer steps."
        ),
        "per_seed": per_seed,
        "paired_deltas_bmsa_minus_pg95": deltas,
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = JSON_PATH.with_name(f".{JSON_PATH.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, JSON_PATH)
    markdown_temporary = MARKDOWN_PATH.with_name(f".{MARKDOWN_PATH.name}.tmp")
    markdown_temporary.write_text(markdown(report), encoding="utf-8")
    os.replace(markdown_temporary, MARKDOWN_PATH)
    print(f"artifact_status={report['artifact_status']}")
    print(f"json={JSON_PATH}")
    print(f"markdown={MARKDOWN_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
