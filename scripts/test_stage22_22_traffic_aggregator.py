#!/usr/bin/env python3
"""End-to-end synthetic test for the Step-22.22 incremental aggregator."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import yaml

import aggregate_stage22_22_traffic as aggregator


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_run(
    root: Path,
    job_id: str,
    metrics: dict,
    *,
    elapsed: float = 10.0,
    peak: float = 1000.0,
) -> str:
    run = root / job_id
    run.mkdir(parents=True)
    write_json(run / "metrics.json", metrics)
    write_json(run / "resource_usage.json", {
        "elapsed_seconds": elapsed, "peak_gpu_memory_mb": peak, "return_code": 0,
    })
    write_json(run / "status.json", {"state": "SUCCESS", "return_code": 0})
    (run / "resolved_config.yaml").write_text(
        yaml.safe_dump({"experiment": {"id": job_id}}), encoding="utf-8"
    )
    return str(run)


def train_metrics(mse: float, mae: float, parameters: int) -> dict:
    return {
        "mse": mse, "mae": mae, "parameter_count": parameters,
        "validation_samples_processed": 1648,
        "test_split_accessed": False,
    }


def eval_metrics(
    original: tuple[float, float],
    multiscale: tuple[float, float],
    *,
    split: str,
) -> dict:
    payload = {
        "complete_target_mse_on": original[0],
        "complete_target_mae_on": original[1],
        "complete_target_mse_multiscale": multiscale[0],
        "complete_target_mae_multiscale": multiscale[1],
        "sample_count": 1648 if split == "val" else 3408,
        "uses_test_split": split == "test",
        "artifact_checks": {"synthetic_contract": True},
    }
    if split == "test":
        names = (
            "original_forward_seconds",
            "l1_mask_seconds", "l1_preprocess_seconds", "l1_forward_seconds",
            "l1_fusion_seconds", "l5_mask_seconds", "l5_preprocess_seconds",
            "l5_forward_seconds", "l5_fusion_seconds", "l20_mask_seconds",
            "l20_preprocess_seconds", "l20_forward_seconds",
            "l20_fusion_seconds", "multiscale_fusion_seconds",
        )
        payload["timing_components"] = {
            name: {"mean_ms": 1.0} for name in names
        }
        payload["timing_measured_batches"] = 203
    return payload


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="stage22_22_aggregator_") as temporary:
        root = Path(temporary)
        current_runs = root / "current_runs"
        old_runs = root / "old_runs"
        current_jobs: dict[str, dict] = {}
        old_jobs: dict[str, dict] = {}

        for seed, offset in ((2026, 0.00), (2027, 0.01), (2028, 0.02)):
            target = old_jobs if seed == 2026 else current_jobs
            run_root = old_runs if seed == 2026 else current_runs
            s4m_train_id = f"traffic_s4m_seed{seed}_formal"
            pg_train_id = f"traffic_pg95_seed{seed}_formal"
            pg_val_id = (
                "traffic_pg95_multiscale_validation_seed2026_formal"
                if seed == 2026 else
                f"traffic_pg95_multiscale_validation_seed{seed}_formal"
            )
            pg_test_id = (
                "traffic_conditional_single_test_seed2026_formal"
                if seed == 2026 else
                f"traffic_pg95_multiscale_test_seed{seed}_formal"
            )
            for job_id, metrics in (
                (s4m_train_id, train_metrics(0.60 + offset, 0.40 + offset, 100)),
                (pg_train_id, train_metrics(0.57 + offset, 0.38 + offset, 110)),
                (pg_val_id, eval_metrics(
                    (0.57 + offset, 0.38 + offset),
                    (0.55 + offset, 0.36 + offset), split="val",
                )),
                (pg_test_id, eval_metrics(
                    (0.70 + offset, 0.45 + offset),
                    (0.68 + offset, 0.43 + offset), split="test",
                )),
            ):
                target[job_id] = {
                    "state": "SUCCESS",
                    "run_dir": make_run(run_root, job_id, metrics),
                }

            s4m_test_id = f"traffic_s4m_multiscale_test_seed{seed}_formal"
            current_jobs[s4m_test_id] = {
                "state": "SUCCESS",
                "run_dir": make_run(
                    current_runs,
                    s4m_test_id,
                    eval_metrics(
                        (0.74 + offset, 0.48 + offset),
                        (0.73 + offset, 0.47 + offset), split="test",
                    ),
                ),
            }

        state_path = root / "state.json"
        old_state_path = root / "old_state.json"
        report_json = root / "results.json"
        report_md = root / "results.md"
        report_csv = root / "results.csv"
        write_json(state_path, {"queue_state": "SUCCESS", "jobs": current_jobs})
        write_json(old_state_path, {"queue_state": "SUCCESS", "jobs": old_jobs})

        aggregator.STATE = state_path
        aggregator.OLD_STATE = old_state_path
        aggregator.REPORT_JSON = report_json
        aggregator.REPORT_MD = report_md
        aggregator.REPORT_CSV = report_csv
        aggregator.subprocess.run = lambda *args, **kwargs: None
        exit_code = aggregator.main()
        payload = json.loads(report_json.read_text(encoding="utf-8"))
        csv_rows = report_csv.read_text(encoding="utf-8").strip().splitlines()
        checks = {
            "main_exit_zero": exit_code == 0,
            "all_outputs_exist": all(
                path.is_file() for path in (report_json, report_md, report_csv)
            ),
            "all_three_seeds_included": (
                payload["all_seeds_included"] is True
                and [item["seed"] for item in payload["per_seed"]]
                == [2026, 2027, 2028]
            ),
            "protocol_pass": payload["protocol_pass"] is True,
            "strong_decision": (
                payload["decision"]
                == "TRAFFIC_MULTISEED_STRONG_CHAIN_CONFIRMED"
            ),
            "sample_standard_deviation_reported": (
                payload["aggregate"]["test"]["s4m_original"]["mse"]
                ["sample_std"] > 0
            ),
            "csv_has_header_plus_18_rows": len(csv_rows) == 19,
            "timing_ratio_roundtrip": all(
                abs(item["cost"]["pg95_test_timing"]
                    ["serial_over_original_ratio"] - 10.0) < 1e-12
                and item["cost"]["pg95_test_timing"]
                    ["excludes_l1_audit_only_route"] is True
                for item in payload["per_seed"]
            ),
        }
        failures = [name for name, passed in checks.items() if not passed]
        result = {
            "status": "PASS" if not failures else "FAIL",
            "checks": checks,
            "failures": failures,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
