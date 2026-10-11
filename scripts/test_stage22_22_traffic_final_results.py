#!/usr/bin/env python3
"""Independent final audit for Step-22.22 Traffic three-seed results."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime/stage22_22_traffic_pipeline/state.json"
RESULTS = ROOT / "reports/stage22_22_traffic_multiseed_results.json"
RESULTS_MD = ROOT / "reports/stage22_22_traffic_multiseed_results.md"
RESULTS_CSV = ROOT / "reports/stage22_22_traffic_multiseed_results.csv"
ASSET_AUDIT = ROOT / "reports/stage22_22_traffic_asset_test.json"
PIPELINE_TEST = ROOT / "reports/stage22_22_traffic_pipeline_test.json"
PROTOCOL = ROOT / "docs/STAGE22_22_TRAFFIC_MULTISEED_PROTOCOL_ZH.md"
FINAL_JSON = ROOT / "reports/stage22_22_traffic_final_audit.json"
FINAL_MD = ROOT / "reports/stage22_22_traffic_final_audit.md"
SEEDS = (2026, 2027, 2028)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def main() -> int:
    required = (
        STATE, RESULTS, RESULTS_MD, RESULTS_CSV, ASSET_AUDIT,
        PIPELINE_TEST, PROTOCOL,
    )
    absent_final_files = [str(path) for path in required if not path.is_file()]
    if absent_final_files:
        raise FileNotFoundError(absent_final_files)
    state, results = load(STATE), load(RESULTS)
    jobs, per_seed = state.get("jobs", {}), results.get("per_seed", [])

    run_file_failures, run_status_failures = [], []
    run_dirs = []
    for job_id, job in jobs.items():
        run_dir = Path(job["run_dir"]).resolve()
        run_dirs.append(run_dir)
        required_names = (
            "resolved_config.yaml", "command.sh", "env.txt", "git_state.txt",
            "code.diff", "train.log", "metrics.json", "status.json",
            "resource_usage.json",
        )
        absent = [name for name in required_names if not (run_dir / name).is_file()]
        if absent:
            run_file_failures.append(f"{job_id}:{','.join(absent)}")
            continue
        status, resource = load(run_dir / "status.json"), load(
            run_dir / "resource_usage.json"
        )
        if (
            status.get("state") != "SUCCESS"
            or int(status.get("return_code", -1)) != 0
            or status.get("verification_error") is not None
            or int(resource.get("return_code", -1)) != 0
        ):
            run_status_failures.append(job_id)

    handoff_failures, validation_roundtrip_failures = [], []
    for item in per_seed:
        seed, records = int(item["seed"]), item["runs"]
        metrics = {
            name: load(Path(record["run_dir"]) / "metrics.json")
            for name, record in records.items()
        }
        for method, eval_names in (
            ("s4m", ("s4m_test",)),
            ("pg95", ("pg95_validation", "pg95_test")),
        ):
            checkpoint = Path(metrics[f"{method}_train"]["checkpoint"]).resolve()
            for eval_name in eval_names:
                if Path(metrics[eval_name]["source_checkpoint"]).resolve() != checkpoint:
                    handoff_failures.append(f"seed{seed}:{eval_name}")
        if not (
            close(metrics["pg95_train"]["mse"],
                  metrics["pg95_validation"]["complete_target_mse_on"], 1e-5)
            and close(metrics["pg95_train"]["mae"],
                      metrics["pg95_validation"]["complete_target_mae_on"], 1e-5)
        ):
            validation_roundtrip_failures.append(seed)

    aggregate, aggregate_ok, paired_ok = results["aggregate"], True, True
    for split in ("validation", "test"):
        for method in ("s4m_original", "pg95_original", "pg95_cnm_se_multiscale"):
            for metric in ("mse", "mae"):
                values = [float(item[split][method][metric]) for item in per_seed]
                stored = aggregate[split][method][metric]
                aggregate_ok &= close(statistics.mean(values), stored["mean"])
                aggregate_ok &= close(statistics.stdev(values), stored["sample_std"])
        for comparison in ("pg95_vs_s4m", "cnm_se_vs_pg95", "cnm_se_vs_s4m"):
            for metric in ("mse", "mae"):
                values = [
                    float(item[split][comparison][metric]
                          ["relative_improvement_percent"])
                    for item in per_seed
                ]
                wins = sum(
                    item[split][comparison][metric]["absolute_reduction"] > 0
                    for item in per_seed
                )
                stored = aggregate[split][comparison][metric]
                paired_ok &= close(
                    statistics.mean(values),
                    stored["relative_improvement_percent"]["mean"],
                )
                paired_ok &= close(
                    statistics.stdev(values),
                    stored["relative_improvement_percent"]["sample_std"],
                ) and wins == int(stored["win_count"])

    with RESULTS_CSV.open("r", encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    csv_map = {
        (int(row["seed"]), row["split"], row["method"]): row for row in csv_rows
    }
    csv_ok = len(csv_rows) == 18 and all(
        close(csv_map[(int(item["seed"]), split, method)][metric],
              item[split][method][metric])
        for item in per_seed for split in ("validation", "test")
        for method in ("s4m_original", "pg95_original", "pg95_cnm_se_multiscale")
        for metric in ("mse", "mae")
    )
    cnm_test_wins = all(
        item["test"]["cnm_se_vs_pg95"][metric]["absolute_reduction"] > 0
        for item in per_seed for metric in ("mse", "mae")
    )
    pg95_test_wins = all(
        item["test"]["pg95_vs_s4m"][metric]["absolute_reduction"] > 0
        for item in per_seed for metric in ("mse", "mae")
    )
    ratios = [
        float(item["cost"]["pg95_test_timing"]["serial_over_original_ratio"])
        for item in per_seed
    ]
    markdown = RESULTS_MD.read_text(encoding="utf-8")
    checks = {
        "queue_success_exact_22": state.get("queue_state") == "SUCCESS"
        and len(jobs) == 22 and all(j.get("state") == "SUCCESS" for j in jobs.values()),
        "first_attempt_no_errors": all(
            int(j.get("attempt_count", -1)) == 1
            and int(j.get("oom_retry_count", -1)) == 0
            and not j.get("last_error_type")
            for j in jobs.values()
        ),
        "unique_run_directories": len(set(run_dirs)) == 22,
        "complete_run_files": not run_file_failures,
        "successful_run_status_and_resources": not run_status_failures,
        "prerequisite_audits_pass": load(ASSET_AUDIT).get("status") == "PASS"
        and load(PIPELINE_TEST).get("status") == "PASS",
        "protocol_predates_replication_assets": all(
            PROTOCOL.stat().st_mtime <= (
                ROOT / f"artifacts/masks/traffic/point/rate_0.06/seed_{seed}.json"
            ).stat().st_mtime for seed in (2027, 2028)
        ),
        "exact_precommitted_seed_set": results.get("precommitted_seeds") == list(SEEDS)
        and [int(x["seed"]) for x in per_seed] == list(SEEDS)
        and results.get("all_seeds_included") is True and not results.get("missing_sources"),
        "per_seed_protocol_checks_pass": all(
            all(item["protocol_checks"].values()) for item in per_seed
        ),
        "exact_checkpoint_handoff": not handoff_failures,
        "validation_metric_roundtrip": not validation_roundtrip_failures,
        "aggregate_mean_std_roundtrip": aggregate_ok,
        "paired_gain_win_roundtrip": paired_ok,
        "csv_18_rows_roundtrip": csv_ok,
        "cnm_se_beats_pg95_all_test_seeds": cnm_test_wins,
        "pg95_not_strong_chain_vs_s4m": not pg95_test_wins,
        "preregistered_decision_matches": results.get("protocol_pass") is True
        and results.get("decision") == "TRAFFIC_MULTISEED_CONFIRMED"
        and "`TRAFFIC_MULTISEED_CONFIRMED`" in markdown,
        "timing_cost_finite": all(math.isfinite(x) and x > 1 for x in ratios),
        "timing_excludes_l1_audit_only_route": all(
            item["cost"]["pg95_test_timing"].get(
                "excludes_l1_audit_only_route"
            ) is True for item in per_seed
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    failures = [name for name, passed in checks.items() if not passed]
    payload = {
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
        "run_file_failures": run_file_failures,
        "run_status_failures": run_status_failures,
        "handoff_failures": handoff_failures,
        "validation_roundtrip_failures": validation_roundtrip_failures,
        "test_cnm_se_vs_pg95_mean_improvement_percent": {
            metric: aggregate["test"]["cnm_se_vs_pg95"][metric]
            ["relative_improvement_percent"]["mean"] for metric in ("mse", "mae")
        },
        "test_cnm_se_vs_s4m_mean_improvement_percent": {
            metric: aggregate["test"]["cnm_se_vs_s4m"][metric]
            ["relative_improvement_percent"]["mean"] for metric in ("mse", "mae")
        },
        "serial_over_original_ratios": ratios,
        "decision": "TRAFFIC_MULTISEED_CONFIRMED" if not failures
        else "TRAFFIC_MULTISEED_FINAL_AUDIT_FAILED",
    }
    FINAL_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
    FINAL_MD.write_text(
        "# 步骤22.22：Traffic三seed终态审计\n\n"
        f"- 审计：`{payload['status']}`\n"
        f"- 决策：`{payload['decision']}`\n"
        "- 队列：22/22 SUCCESS，全部首次attempt。\n"
        f"- CNM-SE相对PG95 test平均提升：MSE "
        f"{payload['test_cnm_se_vs_pg95_mean_improvement_percent']['mse']:.6f}%、"
        f"MAE {payload['test_cnm_se_vs_pg95_mean_improvement_percent']['mae']:.6f}%。\n"
        "- 多尺度/单次前向：" + ", ".join(f"{x:.3f}x" for x in ratios) + "。\n"
        "- 边界：PG95相对S4M不是3/3 seed双指标胜出。\n\n"
        + "\n".join(f"- [{'x' if ok else ' '}] {name}" for name, ok in checks.items())
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
