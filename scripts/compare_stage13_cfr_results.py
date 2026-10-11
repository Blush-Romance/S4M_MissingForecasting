#!/usr/bin/env python3
"""Fair single-seed PG95 versus CFR comparison for Step 13."""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime" / "stage13_cfr_queue" / "state.json"
BASELINE_RUN = ROOT / "runs" / "mipp_pg95__weather__point__mr0.06__seed2023__20260717_234738"
REPORT_JSON = ROOT / "reports" / "stage13_cfr_comparison.json"
REPORT_MD = ROOT / "reports" / "stage13_cfr_comparison.md"


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def load_run(run_dir: Path) -> tuple[dict, dict, dict]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    if status.get("state") != "SUCCESS":
        raise RuntimeError(f"run is not successful: {run_dir}")
    return metrics, status, config


def main() -> int:
    state = json.loads(STATE.read_text(encoding="utf-8"))
    formal = state["jobs"]["weather_pg95_cfr_seed2023_formal"]
    if formal.get("state") != "SUCCESS" or not formal.get("run_dir"):
        return 0
    cfr_run = Path(formal["run_dir"])
    baseline, _, baseline_config = load_run(BASELINE_RUN)
    cfr, _, cfr_config = load_run(cfr_run)

    base_args = baseline_config["official_args"]
    cfr_args = cfr_config["official_args"]
    ignored = {
        "model_id", "des", "use_cfr", "checkpoints", "results_root",
        "test_results_root", "metrics_path",
    }
    protocol_differences = {
        key: {"pg95": base_args.get(key, "<missing>"), "cfr": cfr_args.get(key, "<missing>")}
        for key in sorted(set(base_args) | set(cfr_args))
        if key not in ignored and base_args.get(key, "<missing>") != cfr_args.get(key, "<missing>")
    }
    same_assets = baseline_config["data"] == cfr_config["data"]
    same_seed = int(baseline_config["experiment"]["seed"]) == int(cfr_config["experiment"]["seed"])
    artifact_gate = all(
        (
            metrics.get("checkpoint_load_verified") is True
            and metrics.get("mipp_enabled") is True
        )
        for metrics in (baseline, cfr)
    ) and cfr.get("cfr_enabled") is True
    protocol_gate = not protocol_differences and same_assets and same_seed
    mse_gain = float(baseline["mse"]) - float(cfr["mse"])
    mae_gain = float(baseline["mae"]) - float(cfr["mae"])
    mse_relative = mse_gain / float(baseline["mse"])
    mae_relative = mae_gain / float(baseline["mae"])
    accuracy_gate = mse_gain > 0.0 and mae_gain > 0.0
    if not artifact_gate or not protocol_gate:
        code = "CFR_ARTIFACT_OR_PROTOCOL_GATE_FAILED"
        summary = "工件或公平协议不完整；不能解释精度差异。"
    elif accuracy_gate:
        code = "CFR_ACCURACY_GATE_PASS_REQUIRES_VALIDATION_MECHANISM"
        summary = "CFR单seed双指标优于PG95；下一步只做冻结validation机制复核，再决定第二seed。"
    else:
        code = "CFR_ACCURACY_GATE_FAILED_STOP_OR_MINIMAL_REVIEW"
        summary = "CFR未在单seed同时改善MSE与MAE；不扩第二seed，先按机制证据决定停止或一次最小修正。"

    payload = {
        "decision": {"code": code, "summary_zh": summary},
        "gates": {
            "artifact_gate": artifact_gate,
            "protocol_gate": protocol_gate,
            "accuracy_gate": accuracy_gate,
        },
        "protocol_differences_excluding_identity_and_output_paths": protocol_differences,
        "pg95": {"run_dir": str(BASELINE_RUN), "mse": baseline["mse"], "mae": baseline["mae"]},
        "cfr": {
            "run_dir": str(cfr_run),
            "mse": cfr["mse"],
            "mae": cfr["mae"],
            "training_positive_regret": cfr.get("cfr_positive_regret_mean"),
            "training_harmful_retrieval_rate": cfr.get("cfr_harmful_retrieval_rate_mean"),
            "elapsed_seconds": cfr.get("elapsed_seconds"),
            "peak_gpu_memory_mb": cfr.get("peak_gpu_memory_mb"),
        },
        "gain": {
            "mse_absolute": mse_gain,
            "mae_absolute": mae_gain,
            "mse_relative": mse_relative,
            "mae_relative": mae_relative,
        },
    }
    atomic_text(REPORT_JSON, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# 步骤13：PG95 + CFR 单seed正式对照",
        "",
        "- 公平口径：Weather seed2023，同一数据、固定mask、训练预算、优化器和评价脚本。",
        "- CFR只在训练期增加反事实后悔正则；测试仍为单路prototype-on。",
        "",
        "| 模型 | MSE | MAE |",
        "|---|---:|---:|",
        f"| PG95 | {float(baseline['mse']):.9f} | {float(baseline['mae']):.9f} |",
        f"| PG95+CFR | {float(cfr['mse']):.9f} | {float(cfr['mae']):.9f} |",
        "",
        f"- MSE相对收益：{100.0 * mse_relative:+.4f}%",
        f"- MAE相对收益：{100.0 * mae_relative:+.4f}%",
        f"- 训练期平均正后悔：{cfr.get('cfr_positive_regret_mean')}",
        f"- 训练期有害检索率：{cfr.get('cfr_harmful_retrieval_rate_mean')}",
        f"- 决策：`{code}`",
        f"- 说明：{summary}",
    ]
    atomic_text(REPORT_MD, "\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
