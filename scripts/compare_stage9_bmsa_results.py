#!/usr/bin/env python3
"""Compare Step-9 BMSA with the fixed Weather baseline and PG95."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASELINE_RUN = ROOT / "runs" / "s4m_paper_adam_reprofix__weather__point__mr0.06__seed2023__20260717_201045"
PG95_RUN = ROOT / "runs" / "mipp_pg95__weather__point__mr0.06__seed2023__20260717_234738"
STATE = ROOT / "runtime" / "stage9_bmsa_queue" / "state.json"
REPORT_JSON = ROOT / "reports" / "stage9_bmsa_comparison.json"
REPORT_MD = ROOT / "reports" / "stage9_bmsa_comparison.md"

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


def changes(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in ("mse", "mae"):
        before = float(reference[metric])
        after = float(candidate[metric])
        result[f"{metric}_absolute_change"] = after - before
        result[f"{metric}_relative_gain"] = (before - after) / before
    return result


def validate_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    status = load_json(run_dir / "status.json")
    metrics = load_json(run_dir / "metrics.json")
    config = load_yaml(run_dir / "resolved_config.yaml")
    resources = load_json(run_dir / "resource_usage.json")
    if status.get("state") != "SUCCESS":
        raise RuntimeError(f"run is not successful: {run_dir}")
    if metrics.get("checkpoint_load_verified") is not True:
        raise RuntimeError(f"checkpoint reload not verified: {run_dir}")
    return status, metrics, config, resources


def main() -> int:
    state = load_json(STATE)
    formal = state.get("jobs", {}).get("weather_bmsa50_formal", {})
    if formal.get("state") != "SUCCESS" or not formal.get("run_dir"):
        raise RuntimeError("Step-9 BMSA formal run is not successful")
    bmsa_run = Path(formal["run_dir"])

    runs = {
        "baseline": BASELINE_RUN,
        "pg95": PG95_RUN,
        "bmsa50": bmsa_run,
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for name, run_dir in runs.items():
        status, metrics, config, resources = validate_run(run_dir)
        artifacts[name] = {
            "run_dir": str(run_dir),
            "status": status,
            "metrics": metrics,
            "config": config,
            "resources": resources,
        }

    baseline_cfg = artifacts["baseline"]["config"]
    pg95_cfg = artifacts["pg95"]["config"]
    bmsa_cfg = artifacts["bmsa50"]["config"]
    baseline_protocol = {
        "data": pick(baseline_cfg["data"], DATA_KEYS),
        "official_args": pick(baseline_cfg["official_args"], PROTOCOL_KEYS),
    }
    pg95_protocol = {
        "data": pick(pg95_cfg["data"], DATA_KEYS),
        "official_args": pick(pg95_cfg["official_args"], PROTOCOL_KEYS),
    }
    bmsa_protocol = {
        "data": pick(bmsa_cfg["data"], DATA_KEYS),
        "official_args": pick(bmsa_cfg["official_args"], PROTOCOL_KEYS),
    }
    protocol_match_baseline = baseline_protocol == bmsa_protocol
    protocol_match_pg95 = pg95_protocol == bmsa_protocol
    mipp_shared_match = pick(pg95_cfg["official_args"], MIPP_SHARED_KEYS) == pick(
        bmsa_cfg["official_args"], MIPP_SHARED_KEYS
    )

    bmsa_args = bmsa_cfg["official_args"]
    bmsa_metrics = artifacts["bmsa50"]["metrics"]
    mechanism_checks = {
        "protocol_matches_baseline": protocol_match_baseline,
        "protocol_matches_pg95": protocol_match_pg95,
        "pg95_mipp_settings_unchanged": mipp_shared_match,
        "fusr_disabled": not bool(bmsa_args.get("use_fusr", False)),
        "budget_topk_enabled": bmsa_args.get("mipp_write_gate_mode") == "budget_topk",
        "budget_matches_existing_M": int(bmsa_args.get("mipp_write_budget", -1))
        == int(bmsa_args["M"]),
        "metrics_report_budget_topk": bmsa_metrics.get("mipp_write_gate_mode")
        == "budget_topk",
        "checkpoint_verified": bmsa_metrics.get("checkpoint_load_verified") is True,
        "same_parameter_count_as_pg95": int(bmsa_metrics["parameter_count"])
        == int(artifacts["pg95"]["metrics"]["parameter_count"]),
    }
    artifact_status = "PASS" if all(mechanism_checks.values()) else "FAIL"
    if artifact_status != "PASS":
        raise RuntimeError(f"Step-9 BMSA artifact gate failed: {mechanism_checks}")

    baseline_metrics = artifacts["baseline"]["metrics"]
    pg95_metrics = artifacts["pg95"]["metrics"]
    versus_baseline = changes(baseline_metrics, bmsa_metrics)
    versus_pg95 = changes(pg95_metrics, bmsa_metrics)
    positive_vs_pg95 = (
        versus_pg95["mse_relative_gain"] > 0.0
        and versus_pg95["mae_relative_gain"] > 0.0
    )
    if positive_vs_pg95:
        decision_code = "BMSA_SINGLE_SEED_DIRECTIONAL_POSITIVE"
        summary = "BMSA在同协议单seed下同时优于PG95的MSE和MAE；值得进入复核与多seed门禁。"
    elif (
        versus_pg95["mse_relative_gain"] > 0.0
        or versus_pg95["mae_relative_gain"] > 0.0
    ):
        decision_code = "BMSA_SINGLE_SEED_MIXED_ITERATE_STEP9"
        summary = "BMSA相对PG95仅单指标改善；保留结果并在步骤9内诊断，不扩多seed。"
    else:
        decision_code = "BMSA_SINGLE_SEED_NEGATIVE_ITERATE_STEP9"
        summary = "BMSA相对PG95两项均未改善；保留负结果并在步骤9内更换最小机制。"

    report = {
        "schema_version": 1,
        "step": 9,
        "artifact_status": artifact_status,
        "claim_boundary": "Weather point-r0.06 seed2023 fast formal screen; official mask seed unavailable",
        "mechanism_checks": mechanism_checks,
        "runs": {
            name: {
                "run_dir": item["run_dir"],
                "mse": float(item["metrics"]["mse"]),
                "mae": float(item["metrics"]["mae"]),
                "parameter_count": int(item["metrics"]["parameter_count"]),
                "optimizer_steps": int(item["metrics"].get("optimizer_steps", 0)),
                "elapsed_seconds": item["resources"].get("elapsed_seconds"),
                "peak_gpu_memory_mb": item["metrics"].get("peak_gpu_memory_mb"),
                "peak_gpu_reserved_mb": item["metrics"].get("peak_gpu_reserved_mb"),
            }
            for name, item in artifacts.items()
        },
        "bmsa_mechanism": {
            key: bmsa_metrics.get(key)
            for key in (
                "mipp_write_gate_mode",
                "mipp_write_budget",
                "mipp_stable_write_acceptance_rate_mean",
                "mipp_write_raw_absolute_acceptance_rate_mean",
                "mipp_write_stability_mean",
                "mipp_write_stability_std_mean",
                "mipp_write_effective_cutoff_mean",
                "mipp_write_eligible_count_mean",
                "mipp_write_sampled_count_mean",
                "mipp_write_updated_count_mean",
            )
        },
        "versus_baseline": versus_baseline,
        "versus_pg95": versus_pg95,
        "decision": {
            "code": decision_code,
            "summary_zh": summary,
            "expand_multi_seed": bool(positive_vs_pg95),
        },
    }
    atomic_text(REPORT_JSON, json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# 步骤9：BMSA同协议快速比较",
        "",
        f"- 工件/协议门禁：`{artifact_status}`",
        "- BMSA：Budget-Matched Stability Admission（预算匹配稳定性准入）。",
        "- 只改变PG95写入准入模式；测试mask、seed、训练预算和评价代码保持一致。",
        "",
        "| 模型 | MSE | MAE | 参数量 | optimizer steps | 峰值显存MiB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("baseline", "pg95", "bmsa50"):
        item = report["runs"][name]
        lines.append(
            f"| {name} | {item['mse']:.9f} | {item['mae']:.9f} | "
            f"{item['parameter_count']} | {item['optimizer_steps']} | "
            f"{item['peak_gpu_memory_mb']} |"
        )
    lines.extend(
        [
            "",
            f"- 相对PG95 MSE增益：{100.0 * versus_pg95['mse_relative_gain']:.4f}%",
            f"- 相对PG95 MAE增益：{100.0 * versus_pg95['mae_relative_gain']:.4f}%",
            f"- 相对基线 MSE增益：{100.0 * versus_baseline['mse_relative_gain']:.4f}%",
            f"- 相对基线 MAE增益：{100.0 * versus_baseline['mae_relative_gain']:.4f}%",
            f"- BMSA平均有效接受率：{bmsa_metrics.get('mipp_stable_write_acceptance_rate_mean')}",
            f"- 原0.95阈值平均接受率：{bmsa_metrics.get('mipp_write_raw_absolute_acceptance_rate_mean')}",
            "",
            f"- 决策：`{decision_code}`",
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
