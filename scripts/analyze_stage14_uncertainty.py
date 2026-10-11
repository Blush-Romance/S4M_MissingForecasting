#!/usr/bin/env python3
"""Apply the pre-registered Stage-14 cross-seed uncertainty gate."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime" / "stage14_uncertainty_queue" / "state.json"
REPORT_JSON = ROOT / "reports" / "stage14_uncertainty_analysis.json"
REPORT_MD = ROOT / "reports" / "stage14_uncertainty_analysis.md"


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def group_metrics(indices: np.ndarray, arrays: dict[str, np.ndarray]) -> dict[str, float]:
    count = arrays["complete_count"][indices].sum()
    return {
        "sample_count": int(indices.size),
        "mse": float(arrays["complete_sse_on"][indices].sum() / count),
        "mae": float(arrays["complete_sae_on"][indices].sum() / count),
        "uncertainty_mean": float(arrays["mask_disagreement_mse"][indices].mean()),
        "prototype_disagreement_mean": float(
            arrays["prototype_disagreement_mse"][indices].mean()
        ),
        "harmful_retrieval_rate": float(
            np.mean(
                arrays["complete_sse_on"][indices]
                > arrays["complete_sse_off"][indices]
            )
        ),
    }


def analyze_arrays(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    uncertainty = arrays["mask_disagreement_mse"]
    sample_mse = arrays["complete_sse_on"] / arrays["complete_count"]
    sample_mae = arrays["complete_sae_on"] / arrays["complete_count"]
    order = np.argsort(uncertainty, kind="mergesort")
    groups = np.array_split(order, 4)
    q1 = group_metrics(groups[0], arrays)
    q4 = group_metrics(groups[-1], arrays)
    correlation = spearman(uncertainty, sample_mse)
    mse_ratio = float(q4["mse"] / q1["mse"])
    mae_ratio = float(q4["mae"] / q1["mae"])
    checks = {
        "spearman_at_least_0_15": correlation >= 0.15,
        "q4_q1_mse_ratio_at_least_1_25": mse_ratio >= 1.25,
        "q4_q1_mae_ratio_at_least_1_10": mae_ratio >= 1.10,
    }
    count_total = arrays["complete_count"].sum()
    return {
        "sample_count": int(uncertainty.size),
        "mask_disagreement_error_spearman": correlation,
        "q1": q1,
        "q4": q4,
        "q4_q1_mse_ratio": mse_ratio,
        "q4_q1_mae_ratio": mae_ratio,
        "checks": checks,
        "gate_pass": bool(all(checks.values())),
        "secondary": {
            "prototype_disagreement_error_spearman": spearman(
                arrays["prototype_disagreement_mse"], sample_mse
            ),
            "overall_harmful_retrieval_rate": float(
                np.mean(arrays["complete_sse_on"] > arrays["complete_sse_off"])
            ),
            "prototype_on_mse": float(arrays["complete_sse_on"].sum() / count_total),
            "prototype_off_mse": float(arrays["complete_sse_off"].sum() / count_total),
            "nested_mse": float(arrays["complete_sse_nested"].sum() / count_total),
            "mask_disagreement_mae_spearman": spearman(uncertainty, sample_mae),
        },
    }


def main() -> int:
    state = json.loads(STATE.read_text(encoding="utf-8"))
    results = {}
    integrity = {}
    required_keys = (
        "sample_index", "complete_sse_on", "complete_sae_on",
        "complete_sse_off", "complete_sae_off", "complete_sse_nested",
        "complete_sae_nested", "complete_count", "mask_disagreement_mse",
        "prototype_disagreement_mse", "additional_missing_rate",
    )
    for seed in (2024, 2025):
        job = state["jobs"][f"weather_pg95_seed{seed}_uncertainty_formal"]
        if job.get("state") != "SUCCESS" or not job.get("run_dir"):
            raise RuntimeError(f"Stage-14 formal seed{seed} is not successful")
        run_dir = Path(job["run_dir"])
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        if metrics.get("artifact_status") != "PASS":
            raise RuntimeError(f"Stage-14 seed{seed} artifact gate failed")
        with np.load(run_dir / "validation_uncertainty_samples.npz") as loaded:
            arrays = {key: loaded[key].astype(np.float64, copy=False) for key in required_keys}
            sample_index = loaded["sample_index"]
            history_mask = loaded["history_mask_packed"]
            nested_mask = loaded["nested_mask_packed"]
        checks = {
            "sample_index_chronological": bool(
                np.array_equal(sample_index, np.arange(sample_index.size))
            ),
            "all_arrays_finite": all(np.isfinite(value).all() for value in arrays.values()),
            "nested_mask_shape_matches": nested_mask.shape == history_mask.shape,
            "validation_only": (
                metrics.get("uses_test_split") is False
                and metrics.get("source_split") == "val"
            ),
            "frozen_checkpoint": metrics.get("updates_model_or_memory") is False,
            "protocol_identity": (
                float(metrics.get("nested_target_rate")) == 0.15
                and int(metrics.get("nested_block_length")) == 5
                and int(metrics.get("nested_seed")) == seed
            ),
        }
        if not all(checks.values()):
            raise RuntimeError(f"Stage-14 seed{seed} integrity failed: {checks}")
        integrity[str(seed)] = checks
        results[str(seed)] = analyze_arrays(arrays)

    both_pass = all(results[str(seed)]["gate_pass"] for seed in (2024, 2025))
    if both_pass:
        code = "STAGE14_UNCERTAINTY_GATE_PASS_FREEZE_PERTURBATION_MODULE"
        summary = "两个seed均通过预注册风险门；允许冻结扰动不确定性感知候选。"
    else:
        code = "STAGE14_UNCERTAINTY_GATE_FAILED_STOP_SIGNAL"
        summary = "至少一个seed未通过预注册风险门；停止该风险信号，不从次级指标挑正结果。"
    payload = {
        "scope": "Weather PG95 frozen checkpoints; validation only; no test samples",
        "pre_registered_gate": {
            "spearman_min": 0.15,
            "q4_q1_mse_ratio_min": 1.25,
            "q4_q1_mae_ratio_min": 1.10,
            "all_seeds_required": [2024, 2025],
        },
        "integrity": integrity,
        "results": results,
        "decision": {"code": code, "summary_zh": summary},
    }
    atomic_text(REPORT_JSON, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# 步骤14：跨掩码预测分歧风险诊断", "",
        "- 范围：冻结PG95 checkpoint，仅使用Weather validation；未读取test样本。",
        "- 主门：两个seed均须Spearman≥0.15、Q4/Q1 MSE≥1.25、MAE≥1.10。", "",
        "| seed | Spearman | Q1 MSE | Q4 MSE | MSE ratio | Q1 MAE | Q4 MAE | MAE ratio | PASS |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for seed in (2024, 2025):
        result = results[str(seed)]
        lines.append(
            f"| {seed} | {result['mask_disagreement_error_spearman']:.6f} | "
            f"{result['q1']['mse']:.9f} | {result['q4']['mse']:.9f} | "
            f"{result['q4_q1_mse_ratio']:.4f} | {result['q1']['mae']:.9f} | "
            f"{result['q4']['mae']:.9f} | {result['q4_q1_mae_ratio']:.4f} | "
            f"{result['gate_pass']} |"
        )
    lines.extend(["", f"- 决策：`{code}`", f"- 说明：{summary}"])
    atomic_text(REPORT_MD, "\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
