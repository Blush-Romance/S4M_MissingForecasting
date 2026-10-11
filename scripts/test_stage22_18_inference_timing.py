#!/usr/bin/env python3
"""Static and configuration gates for Step-22.18 inference timing."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_stage22_18_inference_timing import project_path, validate_config


def main() -> int:
    config_dir = ROOT / "configs" / "stage22_ttrr"
    configs = {}
    for method in ("pg95", "mcpr_mse", "ttrr_mcpr"):
        for smoke in (True, False):
            suffix = "_smoke" if smoke else ""
            path = config_dir / f"timing_{method}_electricity_seed2024{suffix}.yaml"
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            source_path = project_path(config["source"]["resolved_config"])
            source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
            validate_config(config, source)
            configs[(method, smoke)] = config

    formal = [configs[(method, False)] for method in (
        "pg95", "mcpr_mse", "ttrr_mcpr"
    )]
    smoke = [configs[(method, True)] for method in (
        "pg95", "mcpr_mse", "ttrr_mcpr"
    )]
    experiment_source = (
        ROOT / "baselines" / "S4M" / "s4m" / "experiments" / "exp_pretrain1.py"
    ).read_text(encoding="utf-8")
    run_source = (ROOT / "baselines" / "S4M" / "s4m" / "run.py").read_text(
        encoding="utf-8"
    )
    queue_source = (
        ROOT / "scripts" / "run_stage22_18_inference_timing_queue.py"
    ).read_text(encoding="utf-8")
    start_source = (
        ROOT / "scripts" / "start_stage22_18_inference_timing_queue.sh"
    ).read_text(encoding="utf-8")
    checks = {
        "six_configs_loaded": len(configs) == 6,
        "formal_assets_identical": len({
            (
                item["data"]["missing_csv"],
                item["data"]["complete_csv"],
                item["data"]["mask_asset"],
            ) for item in formal
        }) == 1,
        "formal_budget_identical": len({
            (
                item["diagnostic"]["max_eval_batches"],
                item["diagnostic"]["timing_warmup_batches"],
                item["diagnostic"]["timing_repeats"],
            ) for item in formal
        }) == 1,
        "smoke_budget_identical": len({
            (
                item["diagnostic"]["max_eval_batches"],
                item["diagnostic"]["timing_warmup_batches"],
                item["diagnostic"]["timing_repeats"],
            ) for item in smoke
        }) == 1,
        "all_source_assets_exist": all(
            project_path(item[section][key]).is_file()
            for item in formal
            for section, key in (
                ("source", "resolved_config"),
                ("source", "metrics"),
                ("source", "checkpoint"),
                ("data", "missing_csv"),
                ("data", "complete_csv"),
                ("data", "mask_asset"),
            )
        ),
        "chronological_loader_hardcoded": "shuffle=False" in experiment_source,
        "cuda_event_timing_present": "torch.cuda.Event(enable_timing=True)" in experiment_source,
        "timing_excludes_transfer": "host-to-device transfer" in experiment_source,
        "validation_only_evidence_present": "'test_split_accessed': False" in experiment_source,
        "diagnostic_default_off": "--stage22_inference_timing" in run_source,
        "diagnostic_exclusive": "+ int(args.stage22_inference_timing)" in run_source,
        "queue_dependency_chain_serial": all(
            token in queue_source for token in (
                '"depends_on": ["pg95_timing_smoke"]',
                '"depends_on": ["mcpr_mse_timing_smoke"]',
                '"depends_on": ["ttrr_mcpr_timing_smoke"]',
                '"depends_on": ["pg95_timing_formal"]',
                '"depends_on": ["mcpr_mse_timing_formal"]',
            )
        ),
        "launcher_max_jobs_one": "--max-jobs 1" in start_source,
        "launcher_single_uuid_guard": 'if [[ "$GPU_CANDIDATES" == *,* ]]' in start_source,
    }
    report = {
        "step": "22.18.7",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    report_path = ROOT / "reports" / "stage22_18_inference_timing_test.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
