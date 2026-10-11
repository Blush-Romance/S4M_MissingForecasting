#!/usr/bin/env python3
"""Frozen three-seed gate for Stage-16 CNM-SE."""

from __future__ import annotations

import json
import os
from pathlib import Path

from analyze_stage15_cnm_se import decide, evaluate_seed


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime" / "stage15_cnm_se_queue" / "state.json"
REPORT_JSON = ROOT / "reports" / "stage16_cnm_se_three_seed_analysis.json"
REPORT_MD = ROOT / "reports" / "stage16_cnm_se_three_seed_analysis.md"
SEEDS = (2023, 2024, 2025)


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    state = json.loads(STATE.read_text(encoding="utf-8"))
    results = {}
    runs = {}
    for seed in SEEDS:
        job_name = f"weather_cnm_se_seed{seed}_formal"
        job = state["jobs"][job_name]
        if job.get("state") != "SUCCESS" or not job.get("run_dir"):
            raise RuntimeError(f"{job_name} is not successful")
        run_dir = Path(job["run_dir"])
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        results[str(seed)] = evaluate_seed(metrics)
        runs[str(seed)] = str(run_dir)
    summary = decide(results)
    decision = (
        "CNM_SE_THREE_SEED_GATE_PASS_START_ABLATIONS"
        if summary["gate_pass"]
        else "CNM_SE_THREE_SEED_GATE_FAILED_RETURN_TO_VALIDATION"
    )
    payload = {
        "status": "PASS" if summary["gate_pass"] else "FAIL",
        "scope": "Weather fixed point-missing test assets; three frozen PG95 checkpoints",
        "pre_registered_gate": {
            "all_seeds": list(SEEDS),
            "each_seed_mse_and_mae_must_improve": True,
            "mean_mse_gain_min_pct": 1.0,
            "mean_mae_gain_min_pct": 0.5,
        },
        "results": results,
        "summary": summary,
        "runs": runs,
        "decision": decision,
    }
    atomic_text(REPORT_JSON, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# 步骤16：CNM-SE三seed正式测试",
        "",
        "| seed | PG95 MSE | CNM-SE MSE | MSE gain | PG95 MAE | CNM-SE MAE | MAE gain | PASS |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for seed in SEEDS:
        result = results[str(seed)]
        lines.append(
            f"| {seed} | {result['mse_on']:.9f} | {result['mse_cnm_se']:.9f} | "
            f"{result['mse_gain_pct']:.4f}% | {result['mae_on']:.9f} | "
            f"{result['mae_cnm_se']:.9f} | {result['mae_gain_pct']:.4f}% | "
            f"{result['seed_pass']} |"
        )
    lines.extend([
        "",
        f"- 三seed平均MSE改善：`{summary['mean_mse_gain_pct']:.4f}%`",
        f"- 三seed平均MAE改善：`{summary['mean_mae_gain_pct']:.4f}%`",
        f"- 决策：`{decision}`",
        "",
    ])
    atomic_text(REPORT_MD, "\n".join(lines))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if summary["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
