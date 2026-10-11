#!/usr/bin/env python3
"""Pair Stage-22 S4M/PG95 validation caches and diagnose the risk trade-off."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = ROOT / "reports" / "stage22_validation_tradeoff.json"
REPORT_MD = ROOT / "reports" / "stage22_validation_tradeoff.md"


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(path)


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
        raise FileNotFoundError(f"no successful formal Stage-22 run found for {method}")
    return sorted(candidates)[-1]


def load_run(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    with np.load(path / "stage22_validation_samples.npz") as source:
        samples = {key: source[key].copy() for key in source.files}
    return metrics, samples


def percent_gain(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def moving_block_mean_ci(
    delta: np.ndarray,
    *,
    seed: int = 2023,
    block_length: int = 288,
    repeats: int = 2000,
) -> dict[str, float]:
    values = np.asarray(delta, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("moving-block bootstrap requires a non-empty vector")
    block_length = min(int(block_length), int(values.size))
    starts = np.arange(values.size - block_length + 1)
    blocks_needed = int(np.ceil(values.size / block_length))
    rng = np.random.default_rng(seed)
    estimates = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([values[start:start + block_length] for start in chosen])
        estimates[index] = sample[:values.size].mean()
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "block_length": int(block_length),
        "repeats": int(repeats),
    }


def paired_summary(s4m: dict[str, np.ndarray], pg95: dict[str, np.ndarray]) -> dict[str, Any]:
    count = s4m["complete_count"]
    s4m_mse = s4m["complete_sse_original"] / count
    pg95_mse = pg95["complete_sse_original"] / count
    s4m_mae = s4m["complete_sae_original"] / count
    pg95_mae = pg95["complete_sae_original"] / count
    mse_gain = s4m_mse - pg95_mse
    mae_gain = s4m_mae - pg95_mae
    both = (mse_gain > 0.0) & (mae_gain > 0.0)
    mse_only = (mse_gain > 0.0) & (mae_gain <= 0.0)
    mae_only = (mse_gain <= 0.0) & (mae_gain > 0.0)
    neither = (mse_gain <= 0.0) & (mae_gain <= 0.0)

    order = np.argsort(s4m_mse, kind="mergesort")
    decile_rows = []
    for index, indices in enumerate(np.array_split(order, 10), start=1):
        decile_rows.append({
            "decile": index,
            "sample_count": int(indices.size),
            "s4m_mse": float(s4m_mse[indices].mean()),
            "pg95_mse": float(pg95_mse[indices].mean()),
            "mse_gain": float(mse_gain[indices].mean()),
            "mae_gain": float(mae_gain[indices].mean()),
            "both_improve_fraction": float(both[indices].mean()),
        })

    top_count = max(1, int(np.ceil(0.10 * order.size)))
    top = order[-top_count:]
    body = order[:-top_count]
    total_sse_gain = float(
        (s4m["complete_sse_original"] - pg95["complete_sse_original"]).sum()
    )
    top_sse_gain = float(
        (s4m["complete_sse_original"][top] - pg95["complete_sse_original"][top]).sum()
    )
    total_sae_gain = float(
        (s4m["complete_sae_original"] - pg95["complete_sae_original"]).sum()
    )
    top_sae_gain = float(
        (s4m["complete_sae_original"][top] - pg95["complete_sae_original"][top]).sum()
    )
    return {
        "fractions": {
            "both_improve": float(both.mean()),
            "mse_only_improves": float(mse_only.mean()),
            "mae_only_improves": float(mae_only.mean()),
            "neither_improves": float(neither.mean()),
        },
        "sample_mean_gain": {
            "mse": float(mse_gain.mean()),
            "mae": float(mae_gain.mean()),
        },
        "moving_block_bootstrap": {
            "mse_gain": moving_block_mean_ci(mse_gain),
            "mae_gain": moving_block_mean_ci(mae_gain),
        },
        "baseline_mse_deciles": decile_rows,
        "top10_vs_body": {
            "top_sample_count": int(top.size),
            "body_sample_count": int(body.size),
            "top_mse_gain": float(mse_gain[top].mean()),
            "body_mse_gain": float(mse_gain[body].mean()),
            "top_mae_gain": float(mae_gain[top].mean()),
            "body_mae_gain": float(mae_gain[body].mean()),
            "top_sse_gain_share_of_total": (
                float(top_sse_gain / total_sse_gain) if total_sse_gain != 0.0 else None
            ),
            "top_sae_gain_share_of_total": (
                float(top_sae_gain / total_sae_gain) if total_sae_gain != 0.0 else None
            ),
        },
    }


def aggregate_axis_summary(
    s4m: dict[str, np.ndarray], pg95: dict[str, np.ndarray], prefix: str
) -> dict[str, Any]:
    count = s4m[f"{prefix}_count"]
    s4m_mse = s4m[f"{prefix}_sse"] / count
    pg95_mse = pg95[f"{prefix}_sse"] / count
    s4m_mae = s4m[f"{prefix}_sae"] / count
    pg95_mae = pg95[f"{prefix}_sae"] / count
    s4m_bias = s4m[f"{prefix}_signed_error"] / count
    pg95_bias = pg95[f"{prefix}_signed_error"] / count
    mse_gain = s4m_mse - pg95_mse
    mae_gain = s4m_mae - pg95_mae
    worst_mae = np.argsort(mae_gain, kind="mergesort")[:10]
    best_mae = np.argsort(mae_gain, kind="mergesort")[-10:][::-1]
    return {
        "size": int(count.size),
        "mse_improved_fraction": float((mse_gain > 0.0).mean()),
        "mae_improved_fraction": float((mae_gain > 0.0).mean()),
        "both_improved_fraction": float(((mse_gain > 0.0) & (mae_gain > 0.0)).mean()),
        "mean_absolute_bias_s4m": float(np.abs(s4m_bias).mean()),
        "mean_absolute_bias_pg95": float(np.abs(pg95_bias).mean()),
        "best_mae_indices": [int(index) for index in best_mae],
        "best_mae_gains": [float(mae_gain[index]) for index in best_mae],
        "worst_mae_indices": [int(index) for index in worst_mae],
        "worst_mae_gains": [float(mae_gain[index]) for index in worst_mae],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s4m-run", default=None)
    parser.add_argument("--pg95-run", default=None)
    args = parser.parse_args()
    s4m_run = Path(args.s4m_run).resolve() if args.s4m_run else latest_formal_run("s4m")
    pg95_run = Path(args.pg95_run).resolve() if args.pg95_run else latest_formal_run("pg95")
    s4m_metrics, s4m = load_run(s4m_run)
    pg95_metrics, pg95 = load_run(pg95_run)

    protocol_checks = {
        "both_artifacts_pass": (
            s4m_metrics.get("artifact_status") == "PASS"
            and pg95_metrics.get("artifact_status") == "PASS"
        ),
        "both_validation_only": (
            s4m_metrics.get("uses_test_split") is False
            and pg95_metrics.get("uses_test_split") is False
        ),
        "sample_index_identical": bool(np.array_equal(s4m["sample_index"], pg95["sample_index"])),
        "history_mask_identical": bool(np.array_equal(s4m["history_mask_packed"], pg95["history_mask_packed"])),
        "nested_mask_identical": bool(np.array_equal(s4m["nested_mask_packed"], pg95["nested_mask_packed"])),
        "complete_count_identical": bool(np.array_equal(s4m["complete_count"], pg95["complete_count"])),
        "observed_count_identical": bool(np.array_equal(s4m["observed_count"], pg95["observed_count"])),
        "full_validation_sample_count": int(s4m["sample_index"].size) == 2528,
    }
    if not all(protocol_checks.values()):
        raise RuntimeError(f"Stage-22 paired protocol failed: {protocol_checks}")

    s4m_mse = float(s4m["complete_sse_original"].sum() / s4m["complete_count"].sum())
    s4m_mae = float(s4m["complete_sae_original"].sum() / s4m["complete_count"].sum())
    pg95_mse = float(pg95["complete_sse_original"].sum() / pg95["complete_count"].sum())
    pg95_mae = float(pg95["complete_sae_original"].sum() / pg95["complete_count"].sum())
    paired = paired_summary(s4m, pg95)
    view_response = {}
    for name, metrics, samples in (
        ("s4m", s4m_metrics, s4m), ("pg95", pg95_metrics, pg95)
    ):
        original_mse = float(samples["complete_sse_original"].sum() / samples["complete_count"].sum())
        original_mae = float(samples["complete_sae_original"].sum() / samples["complete_count"].sum())
        nested_mse = float(samples["complete_sse_nested"].sum() / samples["complete_count"].sum())
        nested_mae = float(samples["complete_sae_nested"].sum() / samples["complete_count"].sum())
        ensemble_mse = float(samples["complete_sse_ensemble"].sum() / samples["complete_count"].sum())
        ensemble_mae = float(samples["complete_sae_ensemble"].sum() / samples["complete_count"].sum())
        disagreement = samples["prediction_disagreement_mse"]
        view_response[name] = {
            "prediction_disagreement_mse_mean": float(disagreement.mean()),
            "disagreement_to_forecast_mse_ratio": float(disagreement.mean() / original_mse),
            "exact_equal_sample_fraction": float((disagreement == 0.0).mean()),
            "nested_gain_percent": {
                "mse": percent_gain(original_mse, nested_mse),
                "mae": percent_gain(original_mae, nested_mae),
            },
            "ensemble_gain_percent": {
                "mse": percent_gain(original_mse, ensemble_mse),
                "mae": percent_gain(original_mae, ensemble_mae),
            },
            "source_metric_roundtrip": {
                "mse_abs_error": abs(original_mse - float(metrics["source_validation_mse_reference_only"])),
                "mae_abs_error": abs(original_mae - float(metrics["source_validation_mae_reference_only"])),
            },
        }

    body_mae_gain = paired["top10_vs_body"]["body_mae_gain"]
    top_sse_share = paired["top10_vs_body"]["top_sse_gain_share_of_total"]
    tail_body_conflict = (
        pg95_mse < s4m_mse
        and pg95_mae > s4m_mae
        and body_mae_gain < 0.0
        and top_sse_share is not None
        and top_sse_share > 0.5
    )
    payload = {
        "status": "PASS",
        "scope": "Electricity seed2023 validation-only paired diagnosis",
        "runs": {"s4m": str(s4m_run), "pg95": str(pg95_run)},
        "protocol_checks": protocol_checks,
        "aggregate": {
            "s4m": {"mse": s4m_mse, "mae": s4m_mae},
            "pg95": {"mse": pg95_mse, "mae": pg95_mae},
            "pg95_gain_percent": {
                "mse": percent_gain(s4m_mse, pg95_mse),
                "mae": percent_gain(s4m_mae, pg95_mae),
            },
        },
        "paired_sample_analysis": paired,
        "horizon_analysis": aggregate_axis_summary(s4m, pg95, "horizon"),
        "variable_analysis": aggregate_axis_summary(s4m, pg95, "variable"),
        "view_response": view_response,
        "diagnosis": {
            "tail_body_conflict": bool(tail_body_conflict),
            "interpretation": (
                "PG95 concentrates squared-error gains in the baseline tail while harming body MAE"
                if tail_body_conflict
                else "the pre-registered tail/body conflict signature was not fully supported"
            ),
            "test_accessed": False,
        },
    }
    atomic_text(REPORT_JSON, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    md = [
        "# 步骤22：Electricity验证集误差权衡与视图响应诊断",
        "",
        f"- 协议：`{'PASS' if all(protocol_checks.values()) else 'FAIL'}`",
        f"- S4M MSE/MAE：`{s4m_mse:.9f} / {s4m_mae:.9f}`",
        f"- PG95 MSE/MAE：`{pg95_mse:.9f} / {pg95_mae:.9f}`",
        f"- PG95相对收益：MSE `{percent_gain(s4m_mse, pg95_mse):.6f}%`，MAE `{percent_gain(s4m_mae, pg95_mae):.6f}%`",
        f"- tail/body（尾部/主体）冲突：`{str(tail_body_conflict).lower()}`",
        f"- S4M视图分歧/预测MSE比：`{view_response['s4m']['disagreement_to_forecast_mse_ratio']:.12g}`",
        f"- PG95视图分歧/预测MSE比：`{view_response['pg95']['disagreement_to_forecast_mse_ratio']:.12g}`",
        "- test（测试集）访问：`false`",
        "",
        "完整分组、block bootstrap（块自助法）区间和变量/预测步统计见JSON。",
    ]
    atomic_text(REPORT_MD, "\n".join(md) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"REPORT_JSON={REPORT_JSON}")
    print(f"REPORT_MD={REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
