#!/usr/bin/env python3
"""Compare frozen MRBA with paired PG95/BMSA anchors on seeds 2024/2025."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime" / "stage11_mrba_anchor_queue" / "state.json"
PREVIOUS = ROOT / "reports" / "stage10_weather_multiseed_comparison.json"
REPORT_JSON = ROOT / "reports" / "stage11_mrba_anchor_comparison.json"
REPORT_MD = ROOT / "reports" / "stage11_mrba_anchor_comparison.md"

PROTOCOL_KEYS = (
    "is_training", "seed", "model", "data", "features", "target", "freq",
    "seq_len", "pred_len", "label_len", "enc_in", "dec_in", "c_out",
    "per_mem_size", "en_conv_hidden_size", "topK", "topM", "thres1",
    "thres2", "M", "K", "memory_size", "momentum", "short_len", "W",
    "e_layers", "d_model", "d_ff", "n_heads", "num_pattn", "num_dG",
    "encoder_type", "memnet", "mem_type", "batch_size", "learning_rate",
    "optimizer", "train_epochs", "patience", "lradj", "num_workers", "itr",
    "mean_type", "fillna", "plot", "max_train_batches", "max_eval_batches",
    "wandb_mode", "deterministic_cluster",
)
MIPP_SHARED_KEYS = (
    "use_mipp", "use_content_mask_factorization", "use_mask_intervention_pair",
    "use_stability_write_gate", "mipp_inv_weight", "mipp_var_weight",
    "mipp_cov_weight", "mipp_dec_weight", "mipp_mask_weight",
    "mipp_variance_target", "mipp_stability_threshold",
    "mipp_pair_rate_min", "mipp_pair_rate_max", "mipp_pair_block_lengths",
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


def load_run(run_dir: Path) -> dict[str, Any]:
    status = load_json(run_dir / "status.json")
    metrics = load_json(run_dir / "metrics.json")
    config = load_yaml(run_dir / "resolved_config.yaml")
    resources = load_json(run_dir / "resource_usage.json")
    if status.get("state") != "SUCCESS":
        raise RuntimeError(f"run is not SUCCESS: {run_dir}")
    if metrics.get("checkpoint_load_verified") is not True:
        raise RuntimeError(f"checkpoint reload is not verified: {run_dir}")
    return {
        "run_dir": str(run_dir),
        "metrics": metrics,
        "config": config,
        "resources": resources,
    }


def metric_change(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    result = {}
    for metric in ("mse", "mae"):
        before = float(reference[metric])
        after = float(candidate[metric])
        result[f"{metric}_delta"] = after - before
        result[f"{metric}_relative_gain"] = (before - after) / before
    result["dual_metric_win"] = bool(
        result["mse_delta"] < 0.0 and result["mae_delta"] < 0.0
    )
    return result


def anchor_decision(
    per_seed: dict[str, Any], mechanism_pass: bool
) -> tuple[bool, str, str]:
    accuracy_pass = all(
        per_seed[str(seed)]["mrba_vs_pg95"]["dual_metric_win"]
        for seed in (2024, 2025)
    )
    if accuracy_pass and mechanism_pass:
        return (
            True,
            "MRBA_TWO_ANCHOR_DUAL_WIN_AND_COVERAGE_RUN_SEED2023",
            "MRBA在两个锚点均双指标优于PG95且mask覆盖机制成立；按预注册规则补跑seed2023。",
        )
    return (
        False,
        "MRBA_TWO_ANCHOR_GATE_FAILED_STOP_CANDIDATE",
        "MRBA未同时通过双锚点精度门与mask覆盖机制门；保留结果并停止该候选，不扩seed或数据集。",
    )


def checks_status(checks: dict[str, bool]) -> str:
    return "PASS" if checks and all(checks.values()) else "FAIL"


def main() -> int:
    state = load_json(STATE)
    previous = load_json(PREVIOUS)
    per_seed = {}
    protocol_checks: dict[str, bool] = {}
    execution_checks: dict[str, bool] = {}
    mechanism_checks: dict[str, bool] = {}
    for seed in (2024, 2025):
        seed_key = str(seed)
        job = state["jobs"][f"weather_mrba50_seed{seed}_formal"]
        if job.get("state") != "SUCCESS" or not job.get("run_dir"):
            raise RuntimeError(f"MRBA seed{seed} formal is not successful")
        paths = {
            "pg95": Path(previous["per_seed"][seed_key]["pg95"]["run_dir"]),
            "bmsa50": Path(previous["per_seed"][seed_key]["bmsa50"]["run_dir"]),
            "mrba50": Path(job["run_dir"]),
        }
        runs = {name: load_run(path) for name, path in paths.items()}
        mrba_cfg = runs["mrba50"]["config"]
        mrba_metrics = runs["mrba50"]["metrics"]
        for reference in ("pg95", "bmsa50"):
            ref_cfg = runs[reference]["config"]
            prefix = f"seed{seed}_{reference}"
            protocol_checks[f"{prefix}_same_data"] = pick(
                ref_cfg["data"], DATA_KEYS
            ) == pick(mrba_cfg["data"], DATA_KEYS)
            protocol_checks[f"{prefix}_same_protocol"] = pick(
                ref_cfg["official_args"], PROTOCOL_KEYS
            ) == pick(mrba_cfg["official_args"], PROTOCOL_KEYS)
            protocol_checks[f"{prefix}_same_mipp_shared"] = pick(
                ref_cfg["official_args"], MIPP_SHARED_KEYS
            ) == pick(mrba_cfg["official_args"], MIPP_SHARED_KEYS)
            ref_budget = int(
                ref_cfg["official_args"].get(
                    "mipp_write_budget", ref_cfg["official_args"]["M"]
                )
            )
            mrba_budget = int(
                mrba_cfg["official_args"].get(
                    "mipp_write_budget", mrba_cfg["official_args"]["M"]
                )
            )
            protocol_checks[f"{prefix}_same_effective_write_budget"] = (
                ref_budget == mrba_budget == 50
            )
            protocol_checks[f"{prefix}_same_parameter_count"] = int(
                runs[reference]["metrics"]["parameter_count"]
            ) == int(mrba_metrics["parameter_count"])
        execution = {
            "mode_correct": mrba_metrics.get("mipp_write_gate_mode")
            == "mask_regime_balanced_rank",
            "budget_correct": int(mrba_metrics.get("mipp_write_budget", -1)) == 50,
            "eligible_within_budget": 0.0
            < float(mrba_metrics.get("mipp_write_eligible_count_mean", 0.0))
            <= 50.0,
            "sampled_within_budget": 0.0
            < float(mrba_metrics.get("mipp_write_sampled_count_mean", 0.0))
            <= 50.0,
        }
        mechanism = {
            "positive_coverage_gain_vs_topk": float(
                mrba_metrics.get(
                    "mipp_write_mask_coverage_gain_vs_topk_mean", float("nan")
                )
            ) > 0.0,
            "coverage_distance_below_topk": float(
                mrba_metrics.get(
                    "mipp_write_mask_coverage_distance_mean", float("nan")
                )
            )
            < float(
                mrba_metrics.get(
                    "mipp_write_mask_topk_coverage_distance_mean", float("nan")
                )
            ),
        }
        for key, value in execution.items():
            execution_checks[f"seed{seed}_{key}"] = bool(value)
        for key, value in mechanism.items():
            mechanism_checks[f"seed{seed}_{key}"] = bool(value)
        per_seed[seed_key] = {
            "runs": {
                name: {
                    "run_dir": item["run_dir"],
                    "mse": float(item["metrics"]["mse"]),
                    "mae": float(item["metrics"]["mae"]),
                    "optimizer_steps": int(item["metrics"].get("optimizer_steps", 0)),
                    "parameter_count": int(item["metrics"]["parameter_count"]),
                    "peak_gpu_memory_mb": item["metrics"].get("peak_gpu_memory_mb"),
                    "elapsed_seconds": item["resources"].get("elapsed_seconds"),
                }
                for name, item in runs.items()
            },
            "mrba_vs_pg95": metric_change(
                runs["pg95"]["metrics"], mrba_metrics
            ),
            "mrba_vs_bmsa50": metric_change(
                runs["bmsa50"]["metrics"], mrba_metrics
            ),
            "mrba_mechanism": {
                key: mrba_metrics.get(key)
                for key in (
                    "mipp_write_gate_mode",
                    "mipp_write_budget",
                    "mipp_stable_write_acceptance_rate_mean",
                    "mipp_write_selected_stability_mean",
                    "mipp_write_mask_candidate_pairwise_distance_mean",
                    "mipp_write_mask_selected_pairwise_distance_mean",
                    "mipp_write_mask_coverage_distance_mean",
                    "mipp_write_mask_topk_coverage_distance_mean",
                    "mipp_write_mask_coverage_gain_vs_topk_mean",
                    "mipp_write_mask_candidate_missing_rate_mean",
                    "mipp_write_mask_selected_missing_rate_mean",
                    "mipp_write_eligible_count_mean",
                    "mipp_write_sampled_count_mean",
                    "mipp_write_updated_count_mean",
                )
            },
        }

    protocol_status = checks_status(protocol_checks)
    execution_status = checks_status(execution_checks)
    artifact_status = (
        "PASS" if protocol_status == "PASS" and execution_status == "PASS" else "FAIL"
    )
    mechanism_status = checks_status(mechanism_checks)
    if artifact_status != "PASS":
        raise RuntimeError(
            "MRBA artifact/protocol gate failed: "
            f"protocol={protocol_checks}, execution={execution_checks}"
        )
    mechanism_pass = mechanism_status == "PASS"
    anchor_pass, code, summary = anchor_decision(per_seed, mechanism_pass)
    report = {
        "schema_version": 2,
        "step": 11,
        "artifact_status": artifact_status,
        "protocol_status": protocol_status,
        "execution_status": execution_status,
        "mechanism_status": mechanism_status,
        "claim_boundary": "Weather point-r0.06 seeds2024/2025 frozen anchor screen",
        "pre_registered_gate": (
            "Both seeds must improve MSE and MAE versus PG95 and both must reduce "
            "mask coverage distance versus stability Top-M."
        ),
        "protocol_checks": protocol_checks,
        "execution_checks": execution_checks,
        "mechanism_checks": mechanism_checks,
        "per_seed": per_seed,
        "decision": {
            "code": code,
            "summary_zh": summary,
            "run_seed2023": anchor_pass,
        },
    }
    atomic_text(REPORT_JSON, json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# 步骤11：MRBA双锚点正式比较",
        "",
        "- MRBA：Mask-Regime Balanced Admission（缺失形态均衡准入）。",
        "- 预注册门槛：两个seed均须双指标改善PG95，且均须降低相对稳定性Top-M的mask覆盖距离。",
        f"- 工件/协议状态：`{artifact_status}`；机制状态：`{mechanism_status}`。",
        "- 机制状态失败仍必须报告精度负结果；它不等同于协议或工件损坏。",
        "",
        "| seed | PG95 MSE | BMSA MSE | MRBA MSE | PG95 MAE | BMSA MAE | MRBA MAE | coverage gain | MRBA双胜PG95 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for seed in (2024, 2025):
        item = per_seed[str(seed)]
        runs = item["runs"]
        lines.append(
            f"| {seed} | {runs['pg95']['mse']:.9f} | {runs['bmsa50']['mse']:.9f} | "
            f"{runs['mrba50']['mse']:.9f} | {runs['pg95']['mae']:.9f} | "
            f"{runs['bmsa50']['mae']:.9f} | {runs['mrba50']['mae']:.9f} | "
            f"{item['mrba_mechanism']['mipp_write_mask_coverage_gain_vs_topk_mean']:.9g} | "
            f"{item['mrba_vs_pg95']['dual_metric_win']} |"
        )
        lines.append(
            f"\n- seed{seed} MRBA相对PG95：MSE "
            f"{100.0 * item['mrba_vs_pg95']['mse_relative_gain']:+.4f}% ，MAE "
            f"{100.0 * item['mrba_vs_pg95']['mae_relative_gain']:+.4f}% 。"
        )
    lines.extend(["", f"- 决策：`{code}`", f"- 说明：{summary}", ""])
    atomic_text(REPORT_MD, "\n".join(lines))
    print(json.dumps(report["decision"], ensure_ascii=False, sort_keys=True))
    print(f"REPORT_JSON={REPORT_JSON}")
    print(f"REPORT_MD={REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
