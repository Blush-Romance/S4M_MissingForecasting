#!/usr/bin/env python3
"""Diagnose train/validation mask-regime geometry without loading targets."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = ROOT / "reports" / "stage11_mask_regime_geometry.json"
REPORT_MD = ROOT / "reports" / "stage11_mask_regime_geometry.md"
SEEDS = (2024, 2025)
ROWS = 52696
SEQ_LEN = 192
PRED_LEN = 96
SHORT_LEN = 32
BLOCKS_PER_HISTORY = SEQ_LEN // SHORT_LEN


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_patterns(mask: np.ndarray, border1: int, border2: int) -> np.ndarray:
    sample_count = (border2 - border1) - SEQ_LEN - PRED_LEN + 1
    if sample_count < 1:
        raise ValueError("split is too short for the frozen S4M protocol")
    starts = (
        border1
        + np.arange(sample_count, dtype=np.int64)[:, None]
        + np.arange(BLOCKS_PER_HISTORY, dtype=np.int64)[None, :] * SHORT_LEN
    ).reshape(-1)
    offsets = np.arange(SHORT_LEN, dtype=np.int64)[None, :]
    return mask[starts[:, None] + offsets].astype(np.uint8, copy=False)


def effective_rank_ratio(patterns: np.ndarray) -> float:
    # A deterministic evenly spaced subset bounds CPU cost while preserving
    # the complete chronological span of the split.
    count = min(len(patterns), 50_000)
    ids = np.linspace(0, len(patterns) - 1, count, dtype=np.int64)
    centered = patterns[ids].astype(np.float64)
    centered -= centered.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    probabilities = singular_values / max(float(singular_values.sum()), 1e-12)
    positive = probabilities > 0.0
    effective_rank = math.exp(
        -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
    )
    return effective_rank / float(patterns.shape[1])


def split_metrics(patterns: np.ndarray, random_seed: int) -> dict[str, Any]:
    packed = np.packbits(patterns, axis=1)
    _, counts = np.unique(packed, axis=0, return_counts=True)
    frequencies = counts.astype(np.float64) / float(counts.sum())
    normalized_entropy = (
        -float(np.sum(frequencies * np.log(frequencies))) / math.log(len(counts))
        if len(counts) > 1
        else 0.0
    )
    rng = np.random.default_rng(random_seed)
    sample_count = min(4_000, len(patterns))
    sample_ids = rng.choice(len(patterns), size=sample_count, replace=False)
    sample = patterns[sample_ids]
    pair_ids = rng.integers(0, sample_count, size=(20_000, 2))
    pairwise_hamming = np.abs(
        sample[pair_ids[:, 0]].astype(np.int16)
        - sample[pair_ids[:, 1]].astype(np.int16)
    ).mean(axis=1)
    missing_counts = patterns.sum(axis=1)
    return {
        "candidate_count": int(len(patterns)),
        "unique_pattern_count": int(len(counts)),
        "unique_pattern_fraction": float(len(counts) / len(patterns)),
        "pattern_entropy_normalized": normalized_entropy,
        "missing_count_mean": float(missing_counts.mean()),
        "missing_count_std": float(missing_counts.std()),
        "zero_missing_fraction": float((missing_counts == 0).mean()),
        "pairwise_hamming_mean": float(pairwise_hamming.mean()),
        "pairwise_hamming_p90": float(np.quantile(pairwise_hamming, 0.9)),
        "effective_rank_ratio": effective_rank_ratio(patterns),
    }


def validation_to_train_distance(
    train_patterns: np.ndarray, validation_patterns: np.ndarray
) -> dict[str, float]:
    train_ids = np.linspace(0, len(train_patterns) - 1, 3_000, dtype=np.int64)
    validation_ids = np.linspace(
        0, len(validation_patterns) - 1, 1_000, dtype=np.int64
    )
    train = train_patterns[train_ids]
    validation = validation_patterns[validation_ids]
    nearest: list[float] = []
    for chunk in np.array_split(validation, 10):
        distances = np.abs(
            chunk[:, None, :].astype(np.int16)
            - train[None, :, :].astype(np.int16)
        ).mean(axis=2)
        nearest.extend(distances.min(axis=1).tolist())
    values = np.asarray(nearest, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(values.max()),
    }


def main() -> int:
    num_train = int(ROWS * 0.7)
    num_test = int(ROWS * 0.2)
    num_validation = ROWS - num_train - num_test
    borders = {
        "train": (0, num_train),
        "validation": (num_train - SEQ_LEN, num_train + num_validation),
    }
    records: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for seed in SEEDS:
        mask_path = (
            ROOT
            / "artifacts"
            / "masks"
            / "weather"
            / "point"
            / "rate_0.06"
            / f"seed_{seed}.npz"
        )
        with np.load(mask_path) as asset:
            mask = asset["row_mask"].astype(np.uint8)
            checks[f"seed{seed}_row_count"] = len(mask) == ROWS
            checks[f"seed{seed}_block_length"] = int(asset["block_length"]) == 5
            checks[f"seed{seed}_identity"] = int(asset["seed"]) == seed
        patterns = {
            split: candidate_patterns(mask, border1, border2)
            for split, (border1, border2) in borders.items()
        }
        checks[f"seed{seed}_binary_mask"] = bool(
            np.all((mask == 0) | (mask == 1))
        )
        checks[f"seed{seed}_candidate_width"] = all(
            value.shape[1] == SHORT_LEN for value in patterns.values()
        )
        records[str(seed)] = {
            "mask_asset": str(mask_path),
            "mask_asset_sha256": sha256(mask_path),
            "global_missing_rate": float(mask.mean()),
            "train": split_metrics(patterns["train"], seed + 11),
            "validation": split_metrics(patterns["validation"], seed + 17),
            "validation_to_train_nearest_hamming": validation_to_train_distance(
                patterns["train"], patterns["validation"]
            ),
        }

    report = {
        "schema_version": 1,
        "step": 11,
        "artifact_status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "Weather input masks; train and validation histories only; no values or targets loaded",
        "protocol": {
            "rows": ROWS,
            "split": [0.7, 0.1, 0.2],
            "seq_len": SEQ_LEN,
            "pred_len": PRED_LEN,
            "short_len": SHORT_LEN,
            "blocks_per_history": BLOCKS_PER_HISTORY,
        },
        "checks": checks,
        "records": records,
        "interpretation_boundary": (
            "Non-collapsed mask geometry supports testing a mask-space admission rule; "
            "it does not by itself establish forecasting improvement."
        ),
    }
    atomic_text(REPORT_JSON, json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        "# 步骤11：缺失形态几何诊断",
        "",
        "- 范围：仅Weather输入mask的train/validation历史；未加载数值、未来目标或test目标。",
        f"- 工件状态：`{report['artifact_status']}`。",
        "",
        "| seed | split | candidates | unique patterns | entropy | mean Hamming | effective-rank ratio | zero-missing |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        for split in ("train", "validation"):
            item = records[str(seed)][split]
            lines.append(
                f"| {seed} | {split} | {item['candidate_count']} | "
                f"{item['unique_pattern_count']} | {item['pattern_entropy_normalized']:.6f} | "
                f"{item['pairwise_hamming_mean']:.6f} | "
                f"{item['effective_rank_ratio']:.6f} | "
                f"{item['zero_missing_fraction']:.6f} |"
            )
    lines.extend(
        [
            "",
            "- 解释边界：mask空间未塌缩只支持进入MRBA快速验证，不能单独证明预测会提升。",
            "",
        ]
    )
    atomic_text(REPORT_MD, "\n".join(lines))
    print(json.dumps({"artifact_status": report["artifact_status"], "records": records}, indent=2, sort_keys=True))
    print(f"REPORT_JSON={REPORT_JSON}")
    print(f"REPORT_MD={REPORT_MD}")
    return 0 if report["artifact_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
