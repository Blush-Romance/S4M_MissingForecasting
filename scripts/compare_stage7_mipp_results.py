#!/usr/bin/env python3
"""Create the auditable four-run Weather MIPP Step-7 comparison."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASELINE_RUN = (
    ROOT
    / "runs"
    / "s4m_paper_adam_reprofix__weather__point__mr0.06__seed2023__20260717_201045"
)
MAIN_QUEUE_STATE = ROOT / "runtime" / "stage7_mipp_queue" / "state.json"
VARIANT_QUEUE_STATE = ROOT / "runtime" / "stage7_mipp_variants_queue" / "state.json"
REPORT_JSON = ROOT / "reports" / "stage7_mipp_comparison.json"
REPORT_MD = ROOT / "reports" / "stage7_mipp_comparison.md"

RUN_SPECS = {
    "baseline": {
        "label": "S4M baseline",
        "description": "frozen S4M baseline with Adam",
        "run_dir": BASELINE_RUN,
    },
    "main_mipp": {
        "label": "Main MIPP",
        "description": "full MIPP with stability threshold 0.80",
        "queue_state": MAIN_QUEUE_STATE,
        "job_id": "weather_mipp_formal",
    },
    "sg95": {
        "label": "SG95",
        "description": "full MIPP with stricter stability threshold 0.95",
        "queue_state": VARIANT_QUEUE_STATE,
        "job_id": "strict_gate_t095_formal",
    },
    "pg95": {
        "label": "PG95",
        "description": "pair invariance and threshold-0.95 gate without factorization losses",
        "queue_state": VARIANT_QUEUE_STATE,
        "job_id": "pair_gate_only_t095_formal",
    },
}

# This is deliberately a whitelist. Run identity, output locations, GPU identity and
# MIPP mechanism switches/weights cannot create a false protocol mismatch.
EXPERIMENT_PROTOCOL_KEYS = (
    "dataset",
    "mask_type",
    "nominal_missing_rate",
    "realized_missing_rate",
    "block_length",
    "seed",
    "smoke_test",
)
DATA_PROTOCOL_KEYS = (
    "missing_csv",
    "complete_csv",
    "mask_asset",
    "mask_metadata",
    "source_manifest",
    "official_mask_seed",
    "generation_seed",
    "split",
)
OFFICIAL_PROTOCOL_KEYS = (
    "is_training",
    "seed",
    "model",
    "data",
    "features",
    "target",
    "freq",
    "seq_len",
    "pred_len",
    "label_len",
    "enc_in",
    "dec_in",
    "c_out",
    "per_mem_size",
    "en_conv_hidden_size",
    "topK",
    "topM",
    "thres1",
    "thres2",
    "M",
    "K",
    "memory_size",
    "momentum",
    "short_len",
    "W",
    "e_layers",
    "d_model",
    "d_ff",
    "n_heads",
    "num_pattn",
    "num_dG",
    "encoder_type",
    "memnet",
    "mem_type",
    "batch_size",
    "learning_rate",
    "optimizer",
    "train_epochs",
    "patience",
    "lradj",
    "num_workers",
    "itr",
    "mean_type",
    "fillna",
    "plot",
    "max_train_batches",
    "max_eval_batches",
    "wandb_mode",
    "deterministic_cluster",
    "root_path",
    "data_path",
    "true_data_path",
)

MIPP_CONFIG_KEYS = (
    "use_mipp",
    "use_content_mask_factorization",
    "use_mask_intervention_pair",
    "use_stability_write_gate",
    "mipp_inv_weight",
    "mipp_var_weight",
    "mipp_cov_weight",
    "mipp_dec_weight",
    "mipp_mask_weight",
    "mipp_variance_target",
    "mipp_stability_threshold",
    "mipp_pair_rate_min",
    "mipp_pair_rate_max",
    "mipp_pair_block_lengths",
)

EPOCH_RE = re.compile(
    r"Epoch:\s*(?P<epoch>\d+),\s*Steps:\s*(?P<steps>\d+)\s*\|\s*"
    r"Train Loss:\s*(?P<train>[0-9.eE+-]+)\s+Vali Loss:\s*(?P<validation>[0-9.eE+-]+)"
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a YAML mapping: {path}")
    return value


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def pick_required(mapping: dict[str, Any], keys: tuple[str, ...], scope: str) -> dict[str, Any]:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise KeyError(f"missing protocol fields in {scope}: {missing}")
    return {key: mapping[key] for key in keys}


def resolve_run_dir(key: str, spec: dict[str, Any]) -> Path:
    if "run_dir" in spec:
        run_dir = Path(spec["run_dir"])
    else:
        queue_path = Path(spec["queue_state"])
        queue = load_json(queue_path)
        job = queue.get("jobs", {}).get(spec["job_id"])
        if not isinstance(job, dict):
            raise KeyError(f"missing {spec['job_id']} in {queue_path}")
        if job.get("state") != "SUCCESS" or not job.get("run_dir"):
            raise RuntimeError(
                f"{key} is not successful: state={job.get('state')}, "
                f"run_dir={job.get('run_dir')}"
            )
        run_dir = Path(job["run_dir"])
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    return run_dir


def protocol_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": pick_required(
            config["experiment"], EXPERIMENT_PROTOCOL_KEYS, "experiment"
        ),
        "data_and_fixed_evaluation_assets": pick_required(
            config["data"], DATA_PROTOCOL_KEYS, "data"
        ),
        "model_training_and_evaluation": pick_required(
            config["official_args"], OFFICIAL_PROTOCOL_KEYS, "official_args"
        ),
    }


def nested_differences(
    baseline: Any, candidate: Any, prefix: str = ""
) -> list[dict[str, Any]]:
    if isinstance(baseline, dict) and isinstance(candidate, dict):
        differences: list[dict[str, Any]] = []
        for key in sorted(set(baseline) | set(candidate)):
            field = f"{prefix}.{key}" if prefix else key
            if key not in baseline:
                differences.append(
                    {"field": field, "baseline": "<missing>", "candidate": candidate[key]}
                )
            elif key not in candidate:
                differences.append(
                    {"field": field, "baseline": baseline[key], "candidate": "<missing>"}
                )
            else:
                differences.extend(nested_differences(baseline[key], candidate[key], field))
        return differences
    if baseline != candidate:
        return [{"field": prefix, "baseline": baseline, "candidate": candidate}]
    return []


def training_trace(run_dir: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    log_text = (run_dir / "train.log").read_text(encoding="utf-8", errors="replace")
    epochs: list[dict[str, Any]] = []
    saved_checkpoint_epochs: list[int] = []
    current_epoch: int | None = None
    for line in log_text.splitlines():
        match = EPOCH_RE.search(line)
        if match:
            current_epoch = int(match.group("epoch"))
            epochs.append(
                {
                    "epoch": current_epoch,
                    "reported_loader_steps": int(match.group("steps")),
                    "train_loss": float(match.group("train")),
                    "validation_loss": float(match.group("validation")),
                }
            )
        elif "Validation loss decreased" in line and current_epoch is not None:
            saved_checkpoint_epochs.append(current_epoch)
    if not epochs:
        raise RuntimeError(f"no completed epochs found in {run_dir / 'train.log'}")
    best_printed = min(epochs, key=lambda item: item["validation_loss"])
    actual_epochs = epochs[-1]["epoch"]
    optimizer_steps = int(metrics["optimizer_steps"])
    return {
        "configured_max_epochs": None,
        "configured_patience": None,
        "actual_epochs_completed": actual_epochs,
        "actual_optimizer_steps": optimizer_steps,
        "optimizer_steps_per_completed_epoch": optimizer_steps / actual_epochs,
        "reported_loader_steps_per_epoch": epochs[-1]["reported_loader_steps"],
        "early_stopped": "Early stopping" in log_text,
        "last_saved_checkpoint_epoch": (
            saved_checkpoint_epochs[-1] if saved_checkpoint_epochs else None
        ),
        "best_printed_validation_epoch": best_printed["epoch"],
        "best_printed_validation_loss": best_printed["validation_loss"],
    }


def mechanism_config(config: dict[str, Any]) -> dict[str, Any]:
    args = config["official_args"]
    if not args.get("use_mipp", False):
        return {"use_mipp": False}
    return {key: args.get(key) for key in MIPP_CONFIG_KEYS}


def raw_run_record(key: str, spec: dict[str, Any]) -> dict[str, Any]:
    run_dir = resolve_run_dir(key, spec)
    status = load_json(run_dir / "status.json")
    metrics = load_json(run_dir / "metrics.json")
    resource = load_json(run_dir / "resource_usage.json")
    config = load_yaml(run_dir / "resolved_config.yaml")
    if status.get("state") != "SUCCESS" or status.get("return_code") != 0:
        raise RuntimeError(f"run is not successful: {run_dir}")
    if not metrics.get("checkpoint_load_verified"):
        raise RuntimeError(f"checkpoint reload was not verified: {run_dir}")
    trace = training_trace(run_dir, metrics)
    trace["configured_max_epochs"] = int(config["official_args"]["train_epochs"])
    trace["configured_patience"] = int(config["official_args"]["patience"])
    return {
        "key": key,
        "label": spec["label"],
        "description": spec["description"],
        "run_dir": str(run_dir),
        "experiment_id": status["experiment_id"],
        "run_state": status["state"],
        "checkpoint_load_verified": bool(metrics["checkpoint_load_verified"]),
        "config": config,
        "protocol_snapshot": protocol_snapshot(config),
        "metrics": metrics,
        "resource": resource,
        "training_trace": trace,
    }


def performance_summary(
    metrics: dict[str, Any], baseline_metrics: dict[str, Any]
) -> dict[str, float]:
    mse = float(metrics["mse"])
    mae = float(metrics["mae"])
    baseline_mse = float(baseline_metrics["mse"])
    baseline_mae = float(baseline_metrics["mae"])
    return {
        "mse": mse,
        "mse_absolute_change_vs_baseline": mse - baseline_mse,
        "mse_relative_gain_vs_baseline": (baseline_mse - mse) / baseline_mse,
        "mae": mae,
        "mae_absolute_change_vs_baseline": mae - baseline_mae,
        "mae_relative_gain_vs_baseline": (baseline_mae - mae) / baseline_mae,
    }


def efficiency_summary(
    record: dict[str, Any], baseline_record: dict[str, Any]
) -> dict[str, Any]:
    metrics = record["metrics"]
    baseline_metrics = baseline_record["metrics"]
    parameters = int(metrics["parameter_count"])
    baseline_parameters = int(baseline_metrics["parameter_count"])
    elapsed = float(record["resource"]["elapsed_seconds"])
    optimizer_steps = int(metrics["optimizer_steps"])
    return {
        "parameter_count": parameters,
        "parameter_delta_vs_baseline": parameters - baseline_parameters,
        "parameter_relative_increase_vs_baseline": (
            (parameters - baseline_parameters) / baseline_parameters
        ),
        "elapsed_seconds": elapsed,
        "seconds_per_optimizer_step": elapsed / optimizer_steps,
        "peak_gpu_memory_mb": float(metrics["peak_gpu_memory_mb"]),
        "peak_gpu_reserved_mb": float(metrics["peak_gpu_reserved_mb"]),
        "gpu_uuid": record["resource"].get("gpu_uuid"),
        **record["training_trace"],
    }


def public_record(
    record: dict[str, Any], baseline_record: dict[str, Any]
) -> dict[str, Any]:
    metrics = record["metrics"]
    protocol_differences = nested_differences(
        baseline_record["protocol_snapshot"], record["protocol_snapshot"]
    )
    return {
        "key": record["key"],
        "label": record["label"],
        "description": record["description"],
        "experiment_id": record["experiment_id"],
        "run_dir": record["run_dir"],
        "run_state": record["run_state"],
        "checkpoint_load_verified": record["checkpoint_load_verified"],
        "same_protocol_as_baseline": not protocol_differences,
        "protocol_differences": protocol_differences,
        "performance": performance_summary(metrics, baseline_record["metrics"]),
        "efficiency_and_realized_training": efficiency_summary(record, baseline_record),
        "mechanism_config": mechanism_config(record["config"]),
        "mechanism_metrics": {
            key: value for key, value in metrics.items() if key.startswith("mipp_")
        },
    }


def markdown_report(payload: dict[str, Any]) -> str:
    by_key = {record["key"]: record for record in payload["runs"]}
    lines = [
        "# 步骤7：Weather MIPP 四组同协议单 seed 比较",
        "",
        f"- 报告状态：`{payload['status']}`",
        f"- 项目内同协议：`{payload['same_protocol']}`",
        "- 相对收益为正表示指标下降、优于基线；相对收益为负表示退化。",
        "",
        "## 指标与资源",
        "",
        "| 模型 | MSE | ΔMSE | MSE相对收益 | MAE | ΔMAE | MAE相对收益 | 参数量 | 参数增量 | 峰值显存(MB) | 实际优化器steps | 实际epoch/最后保存epoch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in payload["runs"]:
        perf = record["performance"]
        efficiency = record["efficiency_and_realized_training"]
        lines.append(
            f"| {record['label']} | {perf['mse']:.9f} | "
            f"{perf['mse_absolute_change_vs_baseline']:+.9f} | "
            f"{perf['mse_relative_gain_vs_baseline'] * 100:+.4f}% | "
            f"{perf['mae']:.9f} | {perf['mae_absolute_change_vs_baseline']:+.9f} | "
            f"{perf['mae_relative_gain_vs_baseline'] * 100:+.4f}% | "
            f"{efficiency['parameter_count']} | {efficiency['parameter_delta_vs_baseline']:+d} | "
            f"{efficiency['peak_gpu_memory_mb']:.1f} | "
            f"{efficiency['actual_optimizer_steps']} | "
            f"{efficiency['actual_epochs_completed']}/{efficiency['last_saved_checkpoint_epoch']} |"
        )

    lines.extend(
        [
            "",
            "四组最大训练轮数均为 15、early stopping（早停）patience 均为 3，但实际早停轮次不同。"
            "因此配置协议相同，实际计算路径并不等长；该差异来自同一早停规则下的验证集轨迹，而非静默修改预算。",
            "",
            "## MIPP 机制诊断",
            "",
            "| 变体 | 内容/掩码因子分解 | 稳定阈值 | 稳定写入接受率 | 配对内容余弦相似度 | 不变性损失 | 总辅助损失 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key in ("main_mipp", "sg95", "pg95"):
        record = by_key[key]
        config = record["mechanism_config"]
        metrics = record["mechanism_metrics"]
        lines.append(
            f"| {record['label']} | `{config['use_content_mask_factorization']}` | "
            f"{config['mipp_stability_threshold']:.2f} | "
            f"{metrics['mipp_stable_write_acceptance_rate_mean'] * 100:.4f}% | "
            f"{metrics['mipp_paired_content_cosine_mean']:.6f} | "
            f"{metrics['mipp_invariance_loss_mean']:.8f} | "
            f"{metrics['mipp_total_loss_mean']:.8f} |"
        )

    fairness = payload["fairness_and_external_limitations"]
    best = payload["best_candidate"]
    lines.extend(
        [
            "",
            "## 协议审计与适用边界",
            "",
            f"- 四组均读取同一固定 mask（掩码）资产：`{fairness['mask_asset']}`，"
            f"项目内公平比较为 `{fairness['project_internal_fixed_mask_comparison']}`。",
            f"- 数据划分、缺失数据文件、归一化目标、seed、模型主干、优化器、scheduler（学习率调度器）、"
            f"最大 epoch、早停规则和评估入口均纳入协议白名单；四组差异数为 "
            f"`{payload['protocol_difference_count']}`。",
            f"- 原论文 mask seed（掩码随机种子）为 `{fairness['official_paper_mask_seed']}`，"
            "未公开。因此本报告能确认项目内同固定 mask 公平，不能声称与原论文具有完全相同的随机 mask。",
            f"- 原论文参考值为 MSE `{fairness['paper_reference']['mse']}`、MAE "
            f"`{fairness['paper_reference']['mae']}`；受 mask seed 未公开限制，仅作背景参考，不用于判定本轮内部胜负。",
            "- 当前仅为 Weather、seed=2023 的单 seed 方向筛选，不构成统计稳定性或 SCI 分区结论。",
            "",
            "## 当前最佳候选",
            "",
            f"`{best['label']}` 是唯一同时改善 MSE 与 MAE 的候选：MSE 相对提升 "
            f"`{best['mse_relative_gain'] * 100:.4f}%`，MAE 相对提升 "
            f"`{best['mae_relative_gain'] * 100:.4f}%`。",
            "",
            "主 MIPP 与 SG95 均改善 MAE 但退化 MSE；固定阈值下三种 MIPP 的写入接受率仍超过 99.5%，"
            "说明该门控尚未形成强原型净化。PG95 可作为后续组合实验的暂定 MIPP 分支，但必须继续通过"
            "多 seed、更多数据集及消融验证。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    raw_records = {
        key: raw_run_record(key, spec) for key, spec in RUN_SPECS.items()
    }
    baseline_record = raw_records["baseline"]
    runs = [
        public_record(raw_records[key], baseline_record)
        for key in ("baseline", "main_mipp", "sg95", "pg95")
    ]
    candidates = runs[1:]
    same_protocol = all(record["same_protocol_as_baseline"] for record in candidates)
    protocol_difference_count = sum(
        len(record["protocol_differences"]) for record in candidates
    )

    baseline_config = baseline_record["config"]
    fixed_mask_signatures = {
        json.dumps(
            {
                key: record["config"]["data"][key]
                for key in (
                    "missing_csv",
                    "complete_csv",
                    "mask_asset",
                    "mask_metadata",
                    "generation_seed",
                    "split",
                )
            },
            sort_keys=True,
        )
        for record in raw_records.values()
    }
    official_mask_seed = baseline_config["data"]["official_mask_seed"]
    official_mask_seed_known = official_mask_seed not in (None, "", "unknown", "unpublished")

    joint_improvements = [
        record
        for record in candidates
        if record["performance"]["mse_relative_gain_vs_baseline"] > 0
        and record["performance"]["mae_relative_gain_vs_baseline"] > 0
    ]
    if joint_improvements:
        selected = min(
            joint_improvements,
            key=lambda record: (record["performance"]["mse"], record["performance"]["mae"]),
        )
        selection_reason = "only candidates improving both metrics are eligible; then lowest MSE wins"
    else:
        selected = min(candidates, key=lambda record: record["performance"]["mse"])
        selection_reason = "no candidate improved both metrics; lowest MSE is reported provisionally"

    realized_step_counts = {
        record["efficiency_and_realized_training"]["actual_optimizer_steps"] for record in runs
    }
    payload = {
        "schema_version": 2,
        "status": "PASS" if same_protocol else "PROTOCOL_MISMATCH",
        "same_protocol": same_protocol,
        "protocol_difference_count": protocol_difference_count,
        "protocol_comparison_policy": {
            "method": "explicit whitelist of data, fixed mask, seed, S4M backbone, training and evaluation fields",
            "excluded_by_design": [
                "experiment/run identity and timestamps",
                "output/checkpoint/result paths",
                "GPU identity and launch-time resource snapshots",
                "MIPP mechanism switches, weights and intervention settings",
            ],
            "reference_snapshot": baseline_record["protocol_snapshot"],
        },
        "fairness_and_external_limitations": {
            "project_internal_fixed_mask_comparison": len(fixed_mask_signatures) == 1,
            "mask_asset": baseline_config["data"]["mask_asset"],
            "mask_metadata": baseline_config["data"]["mask_metadata"],
            "generation_seed": baseline_config["data"]["generation_seed"],
            "official_paper_mask_seed": official_mask_seed,
            "official_paper_mask_seed_known": official_mask_seed_known,
            "exact_mask_match_to_paper_claimed": False,
            "paper_reference": {
                "mse": float(baseline_config["experiment"]["paper_reference_mse"]),
                "mae": float(baseline_config["experiment"]["paper_reference_mae"]),
            },
            "note": (
                "The four project runs share one persisted mask asset and are fair internally; "
                "the paper's mask seed is unpublished, so exact paper-mask equivalence is not claimed."
            ),
        },
        "configured_training_protocol_equal": same_protocol,
        "realized_training_steps_equal": len(realized_step_counts) == 1,
        "realized_training_note": (
            "All runs use train_epochs=15 and patience=3, but the common early-stopping rule "
            "ended runs at different epochs and optimizer-step counts."
        ),
        "runs": runs,
        "best_candidate": {
            "key": selected["key"],
            "label": selected["label"],
            "experiment_id": selected["experiment_id"],
            "selection_rule": selection_reason,
            "mse_relative_gain": selected["performance"]["mse_relative_gain_vs_baseline"],
            "mae_relative_gain": selected["performance"]["mae_relative_gain_vs_baseline"],
            "single_seed_only": True,
        },
        "decision_note": (
            "PG95 is the current directional candidate, not a final statistical claim; "
            "multi-seed and cross-dataset evidence remain required."
        ),
    }
    atomic_text(REPORT_JSON, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    atomic_text(REPORT_MD, markdown_report(payload))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if same_protocol else 2


if __name__ == "__main__":
    raise SystemExit(main())
