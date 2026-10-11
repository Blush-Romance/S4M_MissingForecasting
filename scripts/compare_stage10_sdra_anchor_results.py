#!/usr/bin/env python3
"""Compare frozen SDRA with paired PG95/BMSA anchors on seeds 2024/2025."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime" / "stage10_sdra_anchor_queue" / "state.json"
PREVIOUS = ROOT / "reports" / "stage10_weather_multiseed_comparison.json"
REPORT_JSON = ROOT / "reports" / "stage10_sdra_anchor_comparison.json"
REPORT_MD = ROOT / "reports" / "stage10_sdra_anchor_comparison.md"

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


def anchor_decision(per_seed: dict[str, Any]) -> tuple[bool, str, str]:
    anchor_pass = all(
        per_seed[str(seed)]["sdra_vs_pg95"]["dual_metric_win"]
        for seed in (2024, 2025)
    )
    if anchor_pass:
        return (
            True,
            "SDRA_TWO_ANCHOR_DUAL_WIN_RUN_SEED2023",
            "SDRA在正向与失败锚点均双指标优于PG95；按预注册规则补跑seed2023。",
        )
    return (
        False,
        "SDRA_TWO_ANCHOR_GATE_FAILED_STOP_CANDIDATE",
        "SDRA未在两个锚点同时双指标优于PG95；保留结果并停止该候选，不扩seed或数据集。",
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
        job = state["jobs"][f"weather_sdra50_seed{seed}_formal"]
        if job.get("state") != "SUCCESS" or not job.get("run_dir"):
            raise RuntimeError(f"SDRA seed{seed} formal is not successful")
        paths = {
            "pg95": Path(previous["per_seed"][seed_key]["pg95"]["run_dir"]),
            "bmsa50": Path(previous["per_seed"][seed_key]["bmsa50"]["run_dir"]),
            "sdra50": Path(job["run_dir"]),
        }
        runs = {name: load_run(path) for name, path in paths.items()}
        sdra_cfg = runs["sdra50"]["config"]
        sdra_metrics = runs["sdra50"]["metrics"]
        for reference in ("pg95", "bmsa50"):
            ref_cfg = runs[reference]["config"]
            prefix = f"seed{seed}_{reference}"
            protocol_checks[f"{prefix}_same_data"] = pick(
                ref_cfg["data"], DATA_KEYS
            ) == pick(sdra_cfg["data"], DATA_KEYS)
            protocol_checks[f"{prefix}_same_protocol"] = pick(
                ref_cfg["official_args"], PROTOCOL_KEYS
            ) == pick(sdra_cfg["official_args"], PROTOCOL_KEYS)
            protocol_checks[f"{prefix}_same_mipp_shared"] = pick(
                ref_cfg["official_args"], MIPP_SHARED_KEYS
            ) == pick(sdra_cfg["official_args"], MIPP_SHARED_KEYS)
            ref_budget = int(
                ref_cfg["official_args"].get(
                    "mipp_write_budget", ref_cfg["official_args"]["M"]
                )
            )
            sdra_budget = int(
                sdra_cfg["official_args"].get(
                    "mipp_write_budget", sdra_cfg["official_args"]["M"]
                )
            )
            protocol_checks[f"{prefix}_same_effective_write_budget"] = (
                ref_budget == sdra_budget == 50
            )
            protocol_checks[f"{prefix}_same_parameter_count"] = int(
                runs[reference]["metrics"]["parameter_count"]
            ) == int(sdra_metrics["parameter_count"])
        execution = {
            "mode_correct": sdra_metrics.get("mipp_write_gate_mode")
            == "stability_diverse_rank",
            "budget_correct": int(sdra_metrics.get("mipp_write_budget", -1)) == 50,
            "eligible_within_budget": 0.0
            < float(sdra_metrics.get("mipp_write_eligible_count_mean", 0.0))
            <= 50.0,
            "sampled_within_budget": 0.0
            < float(sdra_metrics.get("mipp_write_sampled_count_mean", 0.0))
            <= 50.0,
        }
        mechanism = {
            "positive_diversity_gain": float(
                sdra_metrics.get("mipp_write_diversity_gain_mean", float("nan"))
            ) > 0.0,
            "selected_pairwise_below_candidate": float(
                sdra_metrics.get("mipp_write_selected_pairwise_cosine_mean", float("nan"))
            )
            < float(
                sdra_metrics.get("mipp_write_candidate_pairwise_cosine_mean", float("nan"))
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
            "sdra_vs_pg95": metric_change(
                runs["pg95"]["metrics"], sdra_metrics
            ),
            "sdra_vs_bmsa50": metric_change(
                runs["bmsa50"]["metrics"], sdra_metrics
            ),
            "sdra_mechanism": {
                key: sdra_metrics.get(key)
                for key in (
                    "mipp_write_gate_mode",
                    "mipp_write_budget",
                    "mipp_stable_write_acceptance_rate_mean",
                    "mipp_write_candidate_pairwise_cosine_mean",
                    "mipp_write_selected_pairwise_cosine_mean",
                    "mipp_write_selected_stability_mean",
                    "mipp_write_diversity_gain_mean",
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
            "SDRA artifact/protocol gate failed: "
            f"protocol={protocol_checks}, execution={execution_checks}"
        )
    anchor_pass, code, summary = anchor_decision(per_seed)
    report = {
        "schema_version": 2,
        "step": 10,
        "artifact_status": artifact_status,
        "protocol_status": protocol_status,
        "execution_status": execution_status,
        "mechanism_status": mechanism_status,
        "claim_boundary": "Weather point-r0.06 seeds2024/2025 frozen anchor screen",
        "pre_registered_gate": (
            "Both seed2024 and seed2025 must improve both MSE and MAE versus paired PG95."
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
        "# 步骤10：SDRA双锚点正式比较",
        "",
        "- SDRA：Stability-Diversity Rank Admission（稳定性—多样性秩准入）。",
        "- 预注册门槛：seed2024和seed2025都必须同时改善PG95的MSE与MAE。",
        f"- 工件/协议状态：`{artifact_status}`；机制状态：`{mechanism_status}`。",
        "- 机制状态失败仍必须报告精度负结果；它不等同于协议或工件损坏。",
        "",
        "| seed | PG95 MSE | BMSA MSE | SDRA MSE | PG95 MAE | BMSA MAE | SDRA MAE | diversity gain | SDRA双胜PG95 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for seed in (2024, 2025):
        item = per_seed[str(seed)]
        runs = item["runs"]
        lines.append(
            f"| {seed} | {runs['pg95']['mse']:.9f} | {runs['bmsa50']['mse']:.9f} | "
            f"{runs['sdra50']['mse']:.9f} | {runs['pg95']['mae']:.9f} | "
            f"{runs['bmsa50']['mae']:.9f} | {runs['sdra50']['mae']:.9f} | "
            f"{item['sdra_mechanism']['mipp_write_diversity_gain_mean']:.9g} | "
            f"{item['sdra_vs_pg95']['dual_metric_win']} |"
        )
        lines.append(
            f"\n- seed{seed} SDRA相对PG95：MSE "
            f"{100.0 * item['sdra_vs_pg95']['mse_relative_gain']:+.4f}% ，MAE "
            f"{100.0 * item['sdra_vs_pg95']['mae_relative_gain']:+.4f}% 。"
        )
    lines.extend(["", f"- 决策：`{code}`", f"- 说明：{summary}", ""])
    atomic_text(REPORT_MD, "\n".join(lines))
    print(json.dumps(report["decision"], ensure_ascii=False, sort_keys=True))
    print(f"REPORT_JSON={REPORT_JSON}")
    print(f"REPORT_MD={REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
