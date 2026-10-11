#!/usr/bin/env python3
"""Paired three-seed statistics for frozen PG95 versus frozen BMSA50."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import numpy as np
from scipy import stats
import yaml


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime" / "stage10_weather_multiseed_queue" / "state.json"
REPORT_JSON = ROOT / "reports" / "stage10_weather_multiseed_comparison.json"
REPORT_MD = ROOT / "reports" / "stage10_weather_multiseed_comparison.md"
SEED2023_RUNS = {
    "pg95": ROOT / "runs" / "mipp_pg95__weather__point__mr0.06__seed2023__20260717_234738",
    "bmsa50": ROOT / "runs" / "mipp_bmsa50__weather__point__mr0.06__seed2023__20260720_154704",
}
PROTOCOL_KEYS = (
    "is_training", "seed", "model", "data", "features", "target", "freq",
    "seq_len", "pred_len", "label_len", "enc_in", "dec_in", "c_out",
    "per_mem_size", "en_conv_hidden_size", "topK", "topM", "thres1",
    "thres2", "M", "K", "memory_size", "momentum", "short_len", "W",
    "e_layers", "d_model", "d_ff", "n_heads", "num_pattn", "num_dG",
    "encoder_type", "memnet", "mem_type", "batch_size", "learning_rate",
    "optimizer", "train_epochs", "patience", "lradj", "num_workers", "itr",
    "mean_type", "fillna", "plot", "max_train_batches", "max_eval_batches",
    "wandb_mode", "deterministic_cluster", "use_mipp",
    "use_content_mask_factorization", "use_mask_intervention_pair",
    "use_stability_write_gate", "mipp_inv_weight", "mipp_var_weight",
    "mipp_cov_weight", "mipp_dec_weight", "mipp_mask_weight",
    "mipp_variance_target", "mipp_stability_threshold", "mipp_pair_rate_min",
    "mipp_pair_rate_max", "mipp_pair_block_lengths",
)
DATA_KEYS = (
    "missing_csv", "complete_csv", "mask_asset", "mask_metadata",
    "source_manifest", "official_mask_seed", "generation_seed", "split",
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def pick(mapping: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise KeyError(f"missing protocol fields: {missing}")
    return {key: mapping[key] for key in keys}


def validate_run(run_dir: Path) -> dict[str, Any]:
    status = load_json(run_dir / "status.json")
    metrics = load_json(run_dir / "metrics.json")
    config = load_yaml(run_dir / "resolved_config.yaml")
    resources = load_json(run_dir / "resource_usage.json")
    if status.get("state") != "SUCCESS":
        raise RuntimeError(f"run is not successful: {run_dir}")
    if metrics.get("checkpoint_load_verified") is not True:
        raise RuntimeError(f"checkpoint reload not verified: {run_dir}")
    return {"run_dir": str(run_dir), "metrics": metrics, "config": config, "resources": resources}


def run_dirs() -> dict[int, dict[str, Path]]:
    state = load_json(STATE)
    result = {2023: SEED2023_RUNS.copy()}
    for seed in (2024, 2025):
        result[seed] = {}
        for method in ("pg95", "bmsa50"):
            job = state["jobs"][f"weather_{method}_seed{seed}_formal"]
            if job.get("state") != "SUCCESS" or not job.get("run_dir"):
                raise RuntimeError(f"missing successful formal job for {method} seed{seed}")
            result[seed][method] = Path(job["run_dir"])
    return result


def paired_protocol_ok(pg95: dict[str, Any], bmsa: dict[str, Any], seed: int) -> dict[str, bool]:
    pg_cfg = pg95["config"]
    bm_cfg = bmsa["config"]
    pg_args = pg_cfg["official_args"]
    bm_args = bm_cfg["official_args"]
    return {
        "same_data_assets": pick(pg_cfg["data"], DATA_KEYS) == pick(bm_cfg["data"], DATA_KEYS),
        "same_training_and_evaluation_protocol": pick(pg_args, PROTOCOL_KEYS)
        == pick(bm_args, PROTOCOL_KEYS),
        "seed_identity_matches": int(pg_args["seed"]) == seed
        and int(bm_args["seed"]) == seed
        and int(pg_cfg["data"]["generation_seed"]) == seed,
        "bmsa_mode_and_budget_correct": bm_args.get("mipp_write_gate_mode") == "budget_topk"
        and int(bm_args.get("mipp_write_budget", -1)) == int(bm_args["M"]),
        "pg95_uses_absolute_mode": pg_args.get("mipp_write_gate_mode", "absolute") == "absolute",
        "same_parameter_count": int(pg95["metrics"]["parameter_count"])
        == int(bmsa["metrics"]["parameter_count"]),
    }


def bootstrap_ci(values: list[float], samples: int = 100000) -> list[float]:
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(20260721)
    indices = rng.integers(0, len(data), size=(samples, len(data)))
    means = data[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def metric_statistics(pg_values: list[float], bm_values: list[float]) -> dict[str, Any]:
    deltas = [bm - pg for pg, bm in zip(pg_values, bm_values)]
    gains = [(pg - bm) / pg for pg, bm in zip(pg_values, bm_values)]
    ttest = stats.ttest_rel(bm_values, pg_values)
    wilcoxon = stats.wilcoxon(bm_values, pg_values, alternative="less", method="exact")
    wins = sum(delta < 0.0 for delta in deltas)
    sign_test = stats.binomtest(wins, len(deltas), 0.5, alternative="greater")
    return {
        "pg95_mean": mean(pg_values),
        "pg95_sample_std": stdev(pg_values),
        "bmsa50_mean": mean(bm_values),
        "bmsa50_sample_std": stdev(bm_values),
        "paired_deltas_bmsa_minus_pg95": deltas,
        "paired_relative_gains": gains,
        "mean_paired_delta": mean(deltas),
        "median_paired_delta": median(deltas),
        "sample_std_paired_delta": stdev(deltas),
        "mean_relative_gain": mean(gains),
        "wins": wins,
        "losses": sum(delta > 0.0 for delta in deltas),
        "ties": sum(delta == 0.0 for delta in deltas),
        "bootstrap_95_ci_mean_delta": bootstrap_ci(deltas),
        "paired_ttest_two_sided_p": float(ttest.pvalue),
        "wilcoxon_one_sided_bmsa_less_p": float(wilcoxon.pvalue),
        "sign_test_one_sided_win_p": float(sign_test.pvalue),
    }


def main() -> int:
    directories = run_dirs()
    artifacts = {
        seed: {method: validate_run(path) for method, path in methods.items()}
        for seed, methods in directories.items()
    }
    protocol_checks = {
        str(seed): paired_protocol_ok(methods["pg95"], methods["bmsa50"], seed)
        for seed, methods in artifacts.items()
    }
    artifact_status = "PASS" if all(
        all(checks.values()) for checks in protocol_checks.values()
    ) else "FAIL"
    if artifact_status != "PASS":
        raise RuntimeError(f"paired protocol gate failed: {protocol_checks}")

    per_seed = {}
    for seed, methods in artifacts.items():
        pg = methods["pg95"]["metrics"]
        bm = methods["bmsa50"]["metrics"]
        per_seed[str(seed)] = {
            "pg95": {"mse": float(pg["mse"]), "mae": float(pg["mae"]), "run_dir": methods["pg95"]["run_dir"]},
            "bmsa50": {"mse": float(bm["mse"]), "mae": float(bm["mae"]), "run_dir": methods["bmsa50"]["run_dir"]},
            "mse_delta_bmsa_minus_pg95": float(bm["mse"]) - float(pg["mse"]),
            "mae_delta_bmsa_minus_pg95": float(bm["mae"]) - float(pg["mae"]),
            "dual_metric_win": float(bm["mse"]) < float(pg["mse"])
            and float(bm["mae"]) < float(pg["mae"]),
        }
    mse_stats = metric_statistics(
        [per_seed[str(seed)]["pg95"]["mse"] for seed in sorted(artifacts)],
        [per_seed[str(seed)]["bmsa50"]["mse"] for seed in sorted(artifacts)],
    )
    mae_stats = metric_statistics(
        [per_seed[str(seed)]["pg95"]["mae"] for seed in sorted(artifacts)],
        [per_seed[str(seed)]["bmsa50"]["mae"] for seed in sorted(artifacts)],
    )
    dual_wins = sum(item["dual_metric_win"] for item in per_seed.values())
    mean_dual_improvement = mse_stats["mean_paired_delta"] < 0.0 and mae_stats["mean_paired_delta"] < 0.0
    if dual_wins == len(per_seed) and mean_dual_improvement:
        decision = "BMSA_THREE_SEED_CONSISTENT_POSITIVE_EXTEND_TO_FIVE_SEEDS_AND_ETTH1"
        summary = "BMSA在三个预注册seed上均同时改善MSE和MAE；进入五seed补强与ETTh1扩展。"
    elif dual_wins >= 2 and mean_dual_improvement:
        decision = "BMSA_THREE_SEED_MAJORITY_POSITIVE_ADD_TWO_SEEDS_BEFORE_DATASET_EXPANSION"
        summary = "BMSA平均双指标改善且多数seed同向，但一致性不足；先补两seed再决定跨数据集。"
    else:
        decision = "BMSA_THREE_SEED_UNSTABLE_REVIEW_BEFORE_EXPANSION"
        summary = "BMSA三seed证据不稳定；保留全部结果并在扩数据集前分析写入选择方差。"

    report = {
        "schema_version": 1,
        "step": 10,
        "artifact_status": artifact_status,
        "claim_boundary": "Weather point-r0.06; seeds 2023/2024/2025; model seed and fixed mask seed paired; official mask seeds unavailable",
        "protocol_checks": protocol_checks,
        "per_seed": per_seed,
        "statistics": {"mse": mse_stats, "mae": mae_stats, "dual_metric_wins": dual_wins},
        "decision": {"code": decision, "summary_zh": summary},
    }
    atomic_text(REPORT_JSON, json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# 步骤10：Weather三seed配对复核",
        "",
        f"- 工件/协议门禁：`{artifact_status}`",
        "- seed集合在运行前固定为2023/2024/2025；每个seed的PG95与BMSA共享同一mask。",
        "",
        "| seed | PG95 MSE | BMSA MSE | ΔMSE | PG95 MAE | BMSA MAE | ΔMAE | 双指标胜出 |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for seed in sorted(artifacts):
        item = per_seed[str(seed)]
        lines.append(
            f"| {seed} | {item['pg95']['mse']:.9f} | {item['bmsa50']['mse']:.9f} | "
            f"{item['mse_delta_bmsa_minus_pg95']:+.9f} | {item['pg95']['mae']:.9f} | "
            f"{item['bmsa50']['mae']:.9f} | {item['mae_delta_bmsa_minus_pg95']:+.9f} | "
            f"{item['dual_metric_win']} |"
        )
    lines.extend(
        [
            "",
            f"- PG95 MSE：{mse_stats['pg95_mean']:.9f} ± {mse_stats['pg95_sample_std']:.9f}",
            f"- BMSA MSE：{mse_stats['bmsa50_mean']:.9f} ± {mse_stats['bmsa50_sample_std']:.9f}",
            f"- MSE平均相对增益：{100.0 * mse_stats['mean_relative_gain']:.4f}%",
            f"- PG95 MAE：{mae_stats['pg95_mean']:.9f} ± {mae_stats['pg95_sample_std']:.9f}",
            f"- BMSA MAE：{mae_stats['bmsa50_mean']:.9f} ± {mae_stats['bmsa50_sample_std']:.9f}",
            f"- MAE平均相对增益：{100.0 * mae_stats['mean_relative_gain']:.4f}%",
            f"- 双指标逐seed胜出：{dual_wins}/{len(per_seed)}",
            "- 三seed统计功效有限；p值和bootstrap区间只作透明报告，不作为录用或稳定性的单一判据。",
            "",
            f"- 决策：`{decision}`",
            f"- 说明：{summary}",
            "",
        ]
    )
    atomic_text(REPORT_MD, "\n".join(lines))
    print(json.dumps(report["decision"], ensure_ascii=False, sort_keys=True))
    print(f"REPORT_JSON={REPORT_JSON}")
    print(f"REPORT_MD={REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
