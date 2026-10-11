#!/usr/bin/env python3
"""Analyze the preregistered Stage-12 validation-only rarity gate."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime" / "stage12_validation_queue" / "state.json"
REPORT_JSON = ROOT / "reports" / "stage12_validation_mismatch.json"
REPORT_MD = ROOT / "reports" / "stage12_validation_mismatch.md"
SEEDS = (2024, 2025)
METHODS = ("pg95", "mrba50")
ROWS = 52696
SEQ_LEN = 192
PRED_LEN = 96
SHORT_LEN = 32
BLOCKS = SEQ_LEN // SHORT_LEN


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def formal_job_id(method: str, seed: int) -> str:
    return f"weather_{method}_seed{seed}_validation_formal"


def load_run(state: dict[str, Any], method: str, seed: int) -> dict[str, Any]:
    job_id = formal_job_id(method, seed)
    job = state.get("jobs", {}).get(job_id, {})
    if job.get("state") != "SUCCESS" or not job.get("run_dir"):
        raise RuntimeError(f"formal cache is not successful: {job_id}")
    run_dir = Path(job["run_dir"])
    metrics_path = run_dir / "metrics.json"
    samples_path = run_dir / "validation_samples.npz"
    if not metrics_path.is_file() or not samples_path.is_file():
        raise FileNotFoundError(f"stage12 cache artifacts missing: {run_dir}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    with np.load(samples_path) as handle:
        samples = {key: handle[key].copy() for key in handle.files}
    return {
        "run_dir": str(run_dir),
        "metrics_path": str(metrics_path),
        "samples_path": str(samples_path),
        "metrics": metrics,
        "samples": samples,
    }


def split_borders() -> tuple[int, int, int]:
    num_train = int(ROWS * 0.7)
    num_test = int(ROWS * 0.2)
    num_validation = ROWS - num_train - num_test
    validation_border1 = num_train - SEQ_LEN
    return num_train, num_validation, validation_border1


def candidate_patterns(mask: np.ndarray, border1: int, border2: int) -> np.ndarray:
    sample_count = (border2 - border1) - SEQ_LEN - PRED_LEN + 1
    starts = (
        border1
        + np.arange(sample_count, dtype=np.int64)[:, None]
        + np.arange(BLOCKS, dtype=np.int64)[None, :] * SHORT_LEN
    ).reshape(-1)
    return mask[starts[:, None] + np.arange(SHORT_LEN, dtype=np.int64)[None, :]]


def mask_features_and_rarity(
    row_missing_mask: np.ndarray, sample_index: np.ndarray
) -> dict[str, np.ndarray]:
    num_train, _, validation_border1 = split_borders()
    train_patterns = candidate_patterns(row_missing_mask, 0, num_train).astype(np.uint8)
    train_packed = np.packbits(train_patterns, axis=1)
    unique, counts = np.unique(train_packed, axis=0, return_counts=True)
    frequency = {row.tobytes(): int(count) for row, count in zip(unique, counts)}
    denominator = float(len(train_packed) + len(unique))

    starts = validation_border1 + sample_index
    offsets = np.arange(SEQ_LEN, dtype=np.int64)[None, :]
    histories = row_missing_mask[starts[:, None] + offsets].astype(np.uint8)
    blocks = histories.reshape(len(histories), BLOCKS, SHORT_LEN)
    packed_blocks = np.packbits(blocks, axis=2)
    rarity = np.empty(len(histories), dtype=np.float64)
    for index, sample_blocks in enumerate(packed_blocks):
        probabilities = np.asarray(
            [
                (frequency.get(block.tobytes(), 0) + 1.0) / denominator
                for block in sample_blocks
            ],
            dtype=np.float64,
        )
        rarity[index] = float(-np.log(probabilities).mean())

    transitions = np.abs(np.diff(histories.astype(np.int16), axis=1)).mean(axis=1)
    max_gap = np.zeros(len(histories), dtype=np.float64)
    for index, history in enumerate(histories):
        padded = np.pad(history, (1, 1), constant_values=0)
        changes = np.flatnonzero(np.diff(padded.astype(np.int16)))
        max_gap[index] = float(np.max(changes[1::2] - changes[::2], initial=0))
    expected_observed_packed = np.packbits(1 - histories, axis=1)
    return {
        "rarity": rarity,
        "missing_rate": histories.mean(axis=1).astype(np.float64),
        "transition_rate": transitions.astype(np.float64),
        "max_missing_run": max_gap,
        "expected_observed_packed": expected_observed_packed,
        "train_candidate_count": np.asarray([len(train_packed)], dtype=np.int64),
        "train_unique_pattern_count": np.asarray([len(unique)], dtype=np.int64),
    }


def equal_quartiles(values: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(values, kind="stable")
    return [group.astype(np.int64, copy=False) for group in np.array_split(order, 4)]


def metric_arrays(samples: dict[str, np.ndarray], scope: str) -> tuple[np.ndarray, np.ndarray]:
    count = samples[f"{scope}_count"].astype(np.float64)
    return (
        samples[f"{scope}_sse"].astype(np.float64) / count,
        samples[f"{scope}_sae"].astype(np.float64) / count,
    )


def group_comparison(
    pg: dict[str, np.ndarray], mr: dict[str, np.ndarray], indices: np.ndarray, scope: str
) -> dict[str, float]:
    pg_count = float(pg[f"{scope}_count"][indices].sum())
    mr_count = float(mr[f"{scope}_count"][indices].sum())
    if pg_count <= 0 or mr_count <= 0:
        raise RuntimeError(f"non-positive {scope} target count in a quartile")
    pg_mse = float(pg[f"{scope}_sse"][indices].sum() / pg_count)
    mr_mse = float(mr[f"{scope}_sse"][indices].sum() / mr_count)
    pg_mae = float(pg[f"{scope}_sae"][indices].sum() / pg_count)
    mr_mae = float(mr[f"{scope}_sae"][indices].sum() / mr_count)
    return {
        "sample_count": int(len(indices)),
        "pg95_mse": pg_mse,
        "mrba_mse": mr_mse,
        "mse_gain": pg_mse - mr_mse,
        "mse_relative_gain": (pg_mse - mr_mse) / pg_mse,
        "pg95_mae": pg_mae,
        "mrba_mae": mr_mae,
        "mae_gain": pg_mae - mr_mae,
        "mae_relative_gain": (pg_mae - mr_mae) / pg_mae,
    }


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else None


def summarize_feature(
    feature: np.ndarray, pg: dict[str, np.ndarray], mr: dict[str, np.ndarray]
) -> dict[str, Any]:
    quartiles = equal_quartiles(feature)
    pg_mse, pg_mae = metric_arrays(pg, "complete")
    mr_mse, mr_mae = metric_arrays(mr, "complete")
    mse_benefit = pg_mse - mr_mse
    mae_benefit = pg_mae - mr_mae
    return {
        "q1": group_comparison(pg, mr, quartiles[0], "complete"),
        "q4": group_comparison(pg, mr, quartiles[3], "complete"),
        "feature_q1_mean": float(feature[quartiles[0]].mean()),
        "feature_q4_mean": float(feature[quartiles[3]].mean()),
        "feature_mse_benefit_pearson": safe_correlation(feature, mse_benefit),
        "feature_mae_benefit_pearson": safe_correlation(feature, mae_benefit),
    }


def rarity_gate_checks(seed: int, primary: dict[str, Any]) -> dict[str, bool]:
    q1 = primary["q1"]
    q4 = primary["q4"]
    return {
        f"seed{seed}_rare_q4_mse_gain_positive": q4["mse_gain"] > 0.0,
        f"seed{seed}_rare_q4_mae_gain_positive": q4["mae_gain"] > 0.0,
        f"seed{seed}_rare_q4_mse_gain_above_common_q1": q4["mse_gain"] > q1["mse_gain"],
        f"seed{seed}_rare_q4_mae_gain_above_common_q1": q4["mae_gain"] > q1["mae_gain"],
    }


def analyze_seed(seed: int, runs: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, bool]]:
    pg_run = runs["pg95"]
    mr_run = runs["mrba50"]
    pg = pg_run["samples"]
    mr = mr_run["samples"]
    pg_metrics = pg_run["metrics"]
    mr_metrics = mr_run["metrics"]
    mask_asset = Path(pg_metrics["mask_asset"])
    with np.load(mask_asset) as asset:
        row_missing_mask = asset["row_mask"].astype(np.uint8)
        asset_seed = int(asset["seed"])
    features = mask_features_and_rarity(row_missing_mask, pg["sample_index"])

    checks = {
        f"seed{seed}_pg_artifact_pass": pg_metrics.get("artifact_status") == "PASS",
        f"seed{seed}_mr_artifact_pass": mr_metrics.get("artifact_status") == "PASS",
        f"seed{seed}_validation_only": pg_metrics.get("uses_test_split") is False and mr_metrics.get("uses_test_split") is False,
        f"seed{seed}_sample_index_equal": bool(np.array_equal(pg["sample_index"], mr["sample_index"])),
        f"seed{seed}_sample_index_chronological": bool(np.array_equal(pg["sample_index"], np.arange(len(pg["sample_index"])))),
        f"seed{seed}_packed_mask_equal": bool(np.array_equal(pg["history_mask_packed"], mr["history_mask_packed"])),
        f"seed{seed}_packed_mask_matches_asset": bool(np.array_equal(pg["history_mask_packed"], features["expected_observed_packed"])),
        f"seed{seed}_complete_count_equal": bool(np.array_equal(pg["complete_count"], mr["complete_count"])),
        f"seed{seed}_observed_count_equal": bool(np.array_equal(pg["observed_count"], mr["observed_count"])),
        f"seed{seed}_mask_asset_same": pg_metrics.get("mask_asset_sha256") == mr_metrics.get("mask_asset_sha256") == sha256(mask_asset),
        f"seed{seed}_mask_asset_identity": asset_seed == seed and len(row_missing_mask) == ROWS,
        f"seed{seed}_source_methods": pg_metrics.get("method") == "pg95" and mr_metrics.get("method") == "mrba50",
        f"seed{seed}_source_seed": int(pg_metrics.get("evaluation_seed", -1)) == seed and int(mr_metrics.get("evaluation_seed", -1)) == seed,
    }
    if not all(checks.values()):
        raise RuntimeError(f"stage12 seed{seed} pairing checks failed: {checks}")

    primary = summarize_feature(features["rarity"], pg, mr)
    gate_checks = rarity_gate_checks(seed, primary)
    pg_mse, pg_mae = metric_arrays(pg, "complete")
    secondary = {
        "missing_rate": summarize_feature(features["missing_rate"], pg, mr),
        "transition_rate": summarize_feature(features["transition_rate"], pg, mr),
        "max_missing_run": summarize_feature(features["max_missing_run"], pg, mr),
        "pg95_mse_tail": summarize_feature(pg_mse, pg, mr),
        "pg95_mae_tail": summarize_feature(pg_mae, pg, mr),
    }
    observed_quartiles = equal_quartiles(features["rarity"])
    observed_primary = {
        "q1": group_comparison(pg, mr, observed_quartiles[0], "observed"),
        "q4": group_comparison(pg, mr, observed_quartiles[3], "observed"),
    }
    whole = {
        "complete": group_comparison(pg, mr, np.arange(len(pg["sample_index"])), "complete"),
        "observed": group_comparison(pg, mr, np.arange(len(pg["sample_index"])), "observed"),
    }
    record = {
        "seed": seed,
        "runs": {method: {key: value for key, value in run.items() if key != "samples" and key != "metrics"} for method, run in runs.items()},
        "sample_count": int(len(pg["sample_index"])),
        "mask_asset": str(mask_asset),
        "train_candidate_count": int(features["train_candidate_count"][0]),
        "train_unique_pattern_count": int(features["train_unique_pattern_count"][0]),
        "whole_validation": whole,
        "primary_rarity_complete_target": primary,
        "rarity_observed_target_sensitivity": observed_primary,
        "secondary_descriptive": secondary,
        "pairing_checks": checks,
        "gate_checks": gate_checks,
    }
    return record, gate_checks


def main() -> int:
    if not STATE.is_file():
        raise FileNotFoundError(f"stage12 queue state missing: {STATE}")
    state = json.loads(STATE.read_text(encoding="utf-8"))
    per_seed: dict[str, Any] = {}
    all_gate_checks: dict[str, bool] = {}
    for seed in SEEDS:
        runs = {method: load_run(state, method, seed) for method in METHODS}
        record, gate_checks = analyze_seed(seed, runs)
        per_seed[str(seed)] = record
        all_gate_checks.update(gate_checks)

    gate_pass = all(all_gate_checks.values())
    decision = {
        "code": (
            "STAGE12_RARITY_GATE_PASS_FREEZE_MASK_CONDITIONED_RETRIEVAL"
            if gate_pass
            else "STAGE12_RARITY_GATE_FAILED_TERMINATE_MASK_SPACE"
        ),
        "continue_mask_conditioned_retrieval": gate_pass,
        "summary_zh": (
            "两个seed的稀有形态Q4均双指标改善且优于常见Q1；允许冻结新的掩码条件检索候选。"
            if gate_pass
            else "预注册稀有形态门未在两个seed同时成立；终止mask-space分支，不从次级分组挑正结果。"
        ),
    }
    report = {
        "schema_version": 1,
        "step": 12,
        "artifact_status": "PASS",
        "scope": "Weather validation histories/targets only; frozen PG95 and MRBA checkpoints; no test samples",
        "preregistered_primary": {
            "feature": "mean Laplace-smoothed negative log frequency of six 32-step input-mask blocks under training candidates",
            "metric_scope": "complete validation target",
            "quartiles": "stable equal-count quartiles within each seed",
            "gate": "For both seeds, rare Q4 MSE/MAE gains must be positive and each exceed common Q1 gain.",
        },
        "gate_checks": all_gate_checks,
        "decision": decision,
        "per_seed": per_seed,
    }
    atomic_text(REPORT_JSON, json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        "# 步骤12：MRBA机制—精度错位诊断",
        "",
        "- 范围：冻结PG95/MRBA checkpoint，仅使用Weather validation历史与目标；未读取test样本。",
        "- 主变量：六个32步历史mask块在train候选中的平均平滑负对数频率（越大越稀有）。",
        "- 主门：两个seed的稀有Q4均须双指标改善，且收益均大于常见Q1。",
        "",
        "| seed | group | PG95 MSE | MRBA MSE | MSE gain | PG95 MAE | MRBA MAE | MAE gain |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        item = per_seed[str(seed)]["primary_rarity_complete_target"]
        for label in ("q1", "q4"):
            row = item[label]
            lines.append(
                f"| {seed} | {label.upper()} | {row['pg95_mse']:.9f} | "
                f"{row['mrba_mse']:.9f} | {row['mse_gain']:+.9f} | "
                f"{row['pg95_mae']:.9f} | {row['mrba_mae']:.9f} | "
                f"{row['mae_gain']:+.9f} |"
            )
    lines.extend(
        [
            "",
            f"- 决策：`{decision['code']}`",
            f"- 说明：{decision['summary_zh']}",
            "- 缺失率、转换率、最长缺失段和PG95误差尾部仅保存在JSON作解释，不参与方向选择。",
            "",
        ]
    )
    atomic_text(REPORT_MD, "\n".join(lines))
    print(json.dumps(decision, indent=2, sort_keys=True))
    print(f"REPORT_JSON={REPORT_JSON}")
    print(f"REPORT_MD={REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
