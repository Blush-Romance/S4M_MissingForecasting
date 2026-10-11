#!/usr/bin/env python3
"""Pair frozen PG95/MRRA-v3 validation caches and diagnose error transfer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = ROOT / "reports" / "stage22_mrra_v3_error_diagnostic.json"
REPORT_MD = ROOT / "reports" / "stage22_mrra_v3_error_diagnostic.md"
FIT = slice(0, 1264)
DEV = slice(1456, 1888)
AUDIT = slice(2080, 2528)


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def latest_formal_run(method: str) -> Path:
    prefix = f"stage22_valcache_{method}__electricity__validation__seed2023__"
    candidates: list[Path] = []
    for path in (ROOT / "runs").glob(prefix + "*"):
        status_path = path / "status.json"
        resolved_path = path / "resolved_config.yaml"
        if not status_path.is_file() or not resolved_path.is_file():
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        if status.get("state") != "SUCCESS":
            continue
        if bool(resolved["experiment"].get("smoke_test")):
            continue
        candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"no successful formal cache found for {method}")
    return sorted(candidates)[-1]


def load_run(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    resolved = yaml.safe_load((path / "resolved_config.yaml").read_text(encoding="utf-8"))
    with np.load(path / "stage22_validation_samples.npz", allow_pickle=False) as source:
        samples = {key: source[key].copy() for key in source.files}
    return metrics, samples, resolved


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


def errors(samples: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    count = samples["complete_count"].astype(np.float64)
    return {
        "mse": samples["complete_sse_original"] / count,
        "mae": samples["complete_sae_original"] / count,
        "sse": samples["complete_sse_original"].astype(np.float64),
        "sae": samples["complete_sae_original"].astype(np.float64),
        "count": count,
    }


def aggregate_choice(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    choose_candidate: np.ndarray,
    indices: slice | np.ndarray,
) -> dict[str, float | int]:
    choose = np.asarray(choose_candidate[indices], dtype=bool)
    ref_sse = reference["sse"][indices]
    ref_sae = reference["sae"][indices]
    cand_sse = candidate["sse"][indices]
    cand_sae = candidate["sae"][indices]
    count = reference["count"][indices]
    mse = float(np.where(choose, cand_sse, ref_sse).sum() / count.sum())
    mae = float(np.where(choose, cand_sae, ref_sae).sum() / count.sum())
    ref_mse = float(ref_sse.sum() / count.sum())
    ref_mae = float(ref_sae.sum() / count.sum())
    return {
        "mse": mse,
        "mae": mae,
        "reference_mse": ref_mse,
        "reference_mae": ref_mae,
        "mse_gain_percent": percent_gain(ref_mse, mse),
        "mae_gain_percent": percent_gain(ref_mae, mae),
        "candidate_fraction": float(choose.mean()),
        "sample_count": int(choose.size),
        "gate_pass": bool(mse <= ref_mse and mae <= ref_mae and (mse < ref_mse or mae < ref_mae)),
    }


def response_routing_screen(
    response: np.ndarray,
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
) -> dict[str, Any]:
    fit_response = response[FIT]
    thresholds = np.unique(np.quantile(fit_response, np.linspace(0.0, 1.0, 21)))
    rules: list[dict[str, Any]] = []
    for direction in ("low", "high"):
        for threshold in thresholds:
            choose = response <= threshold if direction == "low" else response >= threshold
            fit_result = aggregate_choice(reference, candidate, choose, FIT)
            if not fit_result["gate_pass"]:
                continue
            dev_result = aggregate_choice(reference, candidate, choose, DEV)
            if not dev_result["gate_pass"]:
                continue
            score = (
                float(dev_result["mse"]) / float(dev_result["reference_mse"])
                + float(dev_result["mae"]) / float(dev_result["reference_mae"])
            )
            rules.append({
                "direction": direction,
                "threshold": float(threshold),
                "score": score,
                "fit": fit_result,
                "development": dev_result,
                "choose": choose,
            })
    if not rules:
        return {
            "status": "NO_DUAL_NONDEGRADING_RULE_ON_FIT_AND_DEVELOPMENT",
            "candidate_rule_count": 0,
            "audit_gate_pass": False,
            "split": {
                "fit": [0, 1264], "embargo_1": [1264, 1456],
                "development": [1456, 1888], "embargo_2": [1888, 2080],
                "audit": [2080, 2528],
            },
        }
    selected = min(
        rules,
        key=lambda item: (
            item["score"],
            item["development"]["candidate_fraction"],
            item["threshold"],
            item["direction"],
        ),
    )
    audit = aggregate_choice(reference, candidate, selected.pop("choose"), AUDIT)
    selected["audit"] = audit
    selected["audit_gate_pass"] = bool(audit["gate_pass"])
    selected["status"] = (
        "AUDIT_PASS" if audit["gate_pass"] else "AUDIT_FAIL"
    )
    selected["candidate_rule_count"] = len(rules)
    selected["split"] = {
        "fit": [0, 1264], "embargo_1": [1264, 1456],
        "development": [1456, 1888], "embargo_2": [1888, 2080],
        "audit": [2080, 2528],
    }
    return selected


def axis_summary(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    prefix: str,
) -> dict[str, Any]:
    count = reference[f"{prefix}_count"]
    ref_mse = reference[f"{prefix}_sse"] / count
    ref_mae = reference[f"{prefix}_sae"] / count
    cand_mse = candidate[f"{prefix}_sse"] / count
    cand_mae = candidate[f"{prefix}_sae"] / count
    mse_gain = ref_mse - cand_mse
    mae_gain = ref_mae - cand_mae
    best = np.argsort(mse_gain + mae_gain, kind="mergesort")[-10:][::-1]
    worst = np.argsort(mse_gain + mae_gain, kind="mergesort")[:10]
    return {
        "size": int(count.size),
        "mse_improved_fraction": float((mse_gain > 0.0).mean()),
        "mae_improved_fraction": float((mae_gain > 0.0).mean()),
        "both_improved_fraction": float(((mse_gain > 0.0) & (mae_gain > 0.0)).mean()),
        "best_indices": [int(value) for value in best],
        "worst_indices": [int(value) for value in worst],
        "mse_gain": [float(value) for value in mse_gain],
        "mae_gain": [float(value) for value in mae_gain],
    }


def paired_analysis(
    pg95_samples: dict[str, np.ndarray],
    v3_samples: dict[str, np.ndarray],
) -> dict[str, Any]:
    pg95 = errors(pg95_samples)
    v3 = errors(v3_samples)
    mse_gain = pg95["mse"] - v3["mse"]
    mae_gain = pg95["mae"] - v3["mae"]
    both = (mse_gain > 0.0) & (mae_gain > 0.0)
    mse_only = (mse_gain > 0.0) & (mae_gain <= 0.0)
    mae_only = (mse_gain <= 0.0) & (mae_gain > 0.0)
    neither = (mse_gain <= 0.0) & (mae_gain <= 0.0)
    order = np.argsort(pg95["mse"], kind="mergesort")
    deciles = []
    for number, indices in enumerate(np.array_split(order, 10), start=1):
        deciles.append({
            "decile": number,
            "sample_count": int(indices.size),
            "pg95_mse": float(pg95["mse"][indices].mean()),
            "mse_gain": float(mse_gain[indices].mean()),
            "mae_gain": float(mae_gain[indices].mean()),
            "both_improve_fraction": float(both[indices].mean()),
        })
    response = v3_samples["prediction_disagreement_mse"].astype(np.float64)
    response_order = np.argsort(response, kind="mergesort")
    response_deciles = []
    for number, indices in enumerate(np.array_split(response_order, 10), start=1):
        response_deciles.append({
            "decile": number,
            "sample_count": int(indices.size),
            "response_mean": float(response[indices].mean()),
            "mse_gain": float(mse_gain[indices].mean()),
            "mae_gain": float(mae_gain[indices].mean()),
            "both_improve_fraction": float(both[indices].mean()),
        })
    mse_oracle = v3["sse"] < pg95["sse"]
    mae_oracle = v3["sae"] < pg95["sae"]
    oracle = {
        "mse_oracle": aggregate_choice(pg95, v3, mse_oracle, np.arange(mse_gain.size)),
        "mae_oracle": aggregate_choice(pg95, v3, mae_oracle, np.arange(mse_gain.size)),
        "not_deployable_uses_labels": True,
    }
    routing = response_routing_screen(response, pg95, v3)
    return {
        "fractions": {
            "both_improve": float(both.mean()),
            "mse_only_improves": float(mse_only.mean()),
            "mae_only_improves": float(mae_only.mean()),
            "neither_improves": float(neither.mean()),
        },
        "sample_mean_gain": {"mse": float(mse_gain.mean()), "mae": float(mae_gain.mean())},
        "moving_block_bootstrap": {
            "mse_gain": moving_block_mean_ci(mse_gain),
            "mae_gain": moving_block_mean_ci(mae_gain),
        },
        "pg95_mse_deciles": deciles,
        "response": {
            "mean": float(response.mean()),
            "exact_zero_fraction": float((response == 0.0).mean()),
            "spearman_with_mse_gain": spearman(response, mse_gain),
            "spearman_with_mae_gain": spearman(response, mae_gain),
            "spearman_with_pg95_mse": spearman(response, pg95["mse"]),
            "deciles": response_deciles,
        },
        "oracle": oracle,
        "observable_response_routing": routing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pg95-run", default=None)
    parser.add_argument("--mrra-v3-run", default=None)
    args = parser.parse_args()
    pg95_run = Path(args.pg95_run).resolve() if args.pg95_run else latest_formal_run("pg95")
    v3_run = Path(args.mrra_v3_run).resolve() if args.mrra_v3_run else latest_formal_run("mrra_v3")
    pg95_metrics, pg95_samples, pg95_config = load_run(pg95_run)
    v3_metrics, v3_samples, v3_config = load_run(v3_run)
    protocol_checks = {
        "both_artifacts_pass": (
            pg95_metrics.get("artifact_status") == "PASS"
            and v3_metrics.get("artifact_status") == "PASS"
        ),
        "both_validation_only": (
            pg95_metrics.get("uses_test_split") is False
            and v3_metrics.get("uses_test_split") is False
        ),
        "both_frozen": (
            pg95_metrics.get("updates_model_or_memory") is False
            and v3_metrics.get("updates_model_or_memory") is False
        ),
        "methods_match_identity": (
            pg95_metrics.get("method") == "pg95"
            and v3_metrics.get("method") == "mrra_v3"
        ),
        "same_data": pg95_config["data"] == v3_config["data"],
        "sample_index_identical": bool(np.array_equal(pg95_samples["sample_index"], v3_samples["sample_index"])),
        "history_mask_identical": bool(np.array_equal(pg95_samples["history_mask_packed"], v3_samples["history_mask_packed"])),
        "nested_mask_identical": bool(np.array_equal(pg95_samples["nested_mask_packed"], v3_samples["nested_mask_packed"])),
        "complete_count_identical": bool(np.array_equal(pg95_samples["complete_count"], v3_samples["complete_count"])),
        "full_validation": (
            int(pg95_metrics.get("sample_count", 0)) == 2528
            and int(v3_metrics.get("sample_count", 0)) == 2528
            and int(pg95_metrics.get("validation_batches_processed", 0)) == 158
            and int(v3_metrics.get("validation_batches_processed", 0)) == 158
        ),
        "v3_checkpoint_finite_and_contains_mrra": (
            v3_metrics.get("checkpoint_all_model_tensors_finite") is True
            and len(v3_metrics.get("checkpoint_mrra_state_keys", [])) > 0
        ),
    }
    if not all(protocol_checks.values()):
        raise RuntimeError(f"MRRA-v3 paired protocol failed: {protocol_checks}")
    pg95 = errors(pg95_samples)
    v3 = errors(v3_samples)
    pg95_mse = float(pg95["sse"].sum() / pg95["count"].sum())
    pg95_mae = float(pg95["sae"].sum() / pg95["count"].sum())
    v3_mse = float(v3["sse"].sum() / v3["count"].sum())
    v3_mae = float(v3["sae"].sum() / v3["count"].sum())
    paired = paired_analysis(pg95_samples, v3_samples)
    routing_pass = bool(paired["observable_response_routing"]["audit_gate_pass"])
    oracle_pass = any(
        bool(value["gate_pass"])
        for key, value in paired["oracle"].items()
        if key.endswith("_oracle")
    )
    if routing_pass:
        decision = "OBSERVABLE_RESPONSE_GATE_SUPPORTED"
    elif oracle_pass:
        decision = "COMPLEMENTARITY_WITHOUT_RESPONSE_SEPARATION"
    else:
        decision = "MRRA_ACCURACY_BRANCH_STOP"
    payload = {
        "step": "22.14",
        "status": "PASS",
        "scope": "Electricity seed2023 frozen-checkpoint validation-only MRRA-v3 diagnosis",
        "decision": decision,
        "test_accessed": False,
        "runs": {"pg95": str(pg95_run), "mrra_v3": str(v3_run)},
        "protocol_checks": protocol_checks,
        "aggregate": {
            "pg95": {"mse": pg95_mse, "mae": pg95_mae},
            "mrra_v3": {"mse": v3_mse, "mae": v3_mae},
            "mrra_v3_gain_percent": {
                "mse": percent_gain(pg95_mse, v3_mse),
                "mae": percent_gain(pg95_mae, v3_mae),
            },
        },
        "paired_sample_analysis": paired,
        "horizon_analysis": axis_summary(pg95_samples, v3_samples, "horizon"),
        "variable_analysis": axis_summary(pg95_samples, v3_samples, "variable"),
        "limitations": [
            "single seed is diagnostic evidence, not statistical stability",
            "oracle results use validation labels and are not deployable",
            "response routing is audited only inside chronological validation partitions",
        ],
    }
    atomic_text(REPORT_JSON, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    routing = paired["observable_response_routing"]
    lines = [
        "# 步骤22.14：MRRA-v3验证集误差迁移诊断", "",
        f"- 协议：`PASS`", f"- 决策：`{decision}`",
        f"- PG95 MSE/MAE：`{pg95_mse:.9f} / {pg95_mae:.9f}`",
        f"- MRRA-v3 MSE/MAE：`{v3_mse:.9f} / {v3_mae:.9f}`",
        f"- MRRA-v3相对收益：MSE `{percent_gain(pg95_mse, v3_mse):+.6f}%`，MAE `{percent_gain(pg95_mae, v3_mae):+.6f}%`",
        f"- 响应路由审计：`{routing['status']}`",
        f"- test访问：`false`", "",
        "完整逐样本、难度分位、预测步、变量、块自助法、oracle与时间审计见JSON。", "",
    ]
    atomic_text(REPORT_MD, "\n".join(lines))
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"REPORT_JSON={REPORT_JSON}")
    print(f"REPORT_MD={REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
