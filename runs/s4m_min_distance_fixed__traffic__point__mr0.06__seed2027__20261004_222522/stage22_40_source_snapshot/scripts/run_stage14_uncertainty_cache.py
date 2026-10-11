#!/usr/bin/env python3
"""Build one auditable Stage-14 validation-only uncertainty cache."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import yaml

from run_s4m_experiment import (
    atomic_json,
    atomic_text,
    cli_from_mapping,
    git_evidence,
    resolve_gpu_metadata,
    run_capture,
    safe_environment_text,
    verify_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "baselines" / "S4M"
S4M_ROOT = BASELINE_ROOT / "s4m"
STRUCTURED_KERNEL_ROOT = S4M_ROOT / "model" / "S4" / "s4" / "extensions" / "kernels"
TEST_VARIANTS = {"stage15_test", "stage16_ablation"}
STAGE18_VARIANT = "stage18_validation"
STAGE19_VARIANT = "stage19_external_test"
STAGE20_VARIANT = "stage20_pg95_external_test"
STAGE21_VALIDATION_VARIANT = "stage21_electricity_validation"
STAGE21_TEST_VARIANT = "stage21_electricity_test"
STAGE22_21_VALIDATION_VARIANT = "stage22_21_traffic_validation"
STAGE22_21_TEST_VARIANT = "stage22_21_traffic_test"
STAGE22_22_S4M_TEST_VARIANT = "stage22_22_traffic_s4m_test"
STAGE22_34_S4M_VALIDATION_VARIANT = "stage22_34_traffic_s4m_validation"
STAGE22_24_BATCH_PARALLEL_VARIANT = "stage22_24_traffic_batch_parallel_validation"
STAGE22_25_CUDA_STREAM_VARIANT = "stage22_25_traffic_cuda_stream_validation"


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_rng_reference_checks(metrics, samples):
    checks = {}
    count = samples['complete_count']
    checks['fixed_rng_protocol'] = (
        metrics.get('evaluation_rng_protocol') == 'stage22_34_crn_v1'
        and metrics.get('reference_pass') == 'separate_original_only_chronological_loader'
        and metrics.get('reference_consistency_tolerance') == 1e-5)
    for metric, accumulator in (('mse', 'sse'), ('mae', 'sae')):
        ref = samples[f'reference_{accumulator}_on']
        on = samples[f'complete_{accumulator}_on']
        checks[f'fixed_rng_{metric}_reference_roundtrip'] = bool(
            ref.shape == count.shape == on.shape and np.isfinite(ref).all()
            and abs(float(ref.sum() / count.sum()) - metrics[f'fixed_rng_reference_{metric}']) <= 1e-12
            and abs(float(np.max(np.abs(ref-on) / count))
                    - metrics[f'fixed_rng_max_window_{metric}_difference']) <= 1e-12)
        checks[f'fixed_rng_{metric}_paired_consistency'] = bool(
            np.max(np.abs(ref-on) / count) <= 1e-5)
    return checks


def validate_config(config: dict[str, Any], source: dict[str, Any]) -> None:
    experiment = config["experiment"]
    diagnostic = config["diagnostic"]
    official = source["official_args"]
    variant = str(diagnostic.get("variant", "stage14_validation"))
    if variant not in {
        "stage14_validation",
        STAGE18_VARIANT,
        STAGE19_VARIANT,
        STAGE20_VARIANT,
        STAGE21_VALIDATION_VARIANT,
        STAGE21_TEST_VARIANT,
        STAGE22_21_VALIDATION_VARIANT,
        STAGE22_21_TEST_VARIANT,
        STAGE22_22_S4M_TEST_VARIANT,
        STAGE22_34_S4M_VALIDATION_VARIANT,
        STAGE22_24_BATCH_PARALLEL_VARIANT,
        STAGE22_25_CUDA_STREAM_VARIANT,
        *TEST_VARIANTS,
    }:
        raise ValueError(f"unsupported diagnostic variant: {variant}")
    is_test_variant = variant in TEST_VARIANTS
    is_stage18 = variant == STAGE18_VARIANT
    is_stage19 = variant == STAGE19_VARIANT
    is_stage20 = variant == STAGE20_VARIANT
    is_stage21_validation = variant == STAGE21_VALIDATION_VARIANT
    is_stage21_test = variant == STAGE21_TEST_VARIANT
    is_stage21 = is_stage21_validation or is_stage21_test
    is_traffic_validation = variant == STAGE22_21_VALIDATION_VARIANT
    is_traffic_test = variant == STAGE22_21_TEST_VARIANT
    is_traffic_s4m_test = variant == STAGE22_22_S4M_TEST_VARIANT
    is_traffic_s4m_validation = variant == STAGE22_34_S4M_VALIDATION_VARIANT
    is_batch_parallel = variant == STAGE22_24_BATCH_PARALLEL_VARIANT
    is_stream_parallel = variant == STAGE22_25_CUDA_STREAM_VARIANT
    is_traffic = (
        is_traffic_validation or is_traffic_test or is_traffic_s4m_test
        or is_batch_parallel or is_stream_parallel or is_traffic_s4m_validation
    )
    is_gated_validation = (
        is_stage21_validation or is_traffic_validation
        or is_batch_parallel or is_stream_parallel or is_traffic_s4m_validation
    )
    is_gated_test = is_stage21_test or is_traffic_test or is_traffic_s4m_test
    is_gated_extension = is_stage21 or is_traffic
    is_external_test = is_stage19 or is_stage20 or is_gated_test
    is_equal_rate = is_stage18 or is_gated_validation or is_external_test
    if diagnostic.get('fixed_eval_rng', False):
        if (not is_gated_validation or is_batch_parallel or is_stream_parallel
                or experiment.get('dataset') != 'traffic'
                or diagnostic.get('rng_protocol') != 'stage22_34_crn_v1'):
            raise ValueError('fixed RNG recovery requires Traffic serial validation CRN v1')
    if is_stage19:
        if (
            experiment.get("dataset") != "etth1"
            or experiment.get("method") != "s4m_baseline"
        ):
            raise ValueError("Stage-19 is frozen to the ETTh1 S4M baseline")
    elif is_stage20:
        if (
            experiment.get("dataset") != "etth1"
            or experiment.get("method") != "pg95"
        ):
            raise ValueError("Stage-20 is frozen to ETTh1 PG95")
    elif is_stage21:
        if (
            experiment.get("dataset") != "electricity"
            or experiment.get("method") != "pg95"
        ):
            raise ValueError("Stage-21 is frozen to Electricity PG95")
    elif is_traffic:
        expected_method = "s4m_baseline" if (is_traffic_s4m_test or is_traffic_s4m_validation) else "pg95"
        if (
            experiment.get("dataset") != "traffic"
            or experiment.get("method") != expected_method
        ):
            raise ValueError(
                f"Traffic evaluation is frozen to {expected_method}"
            )
    elif (
        experiment.get("dataset") != "weather"
        or experiment.get("method") != "pg95"
    ):
        raise ValueError("Stage-14/18 is frozen to Weather PG95")
    if is_traffic:
        allowed_seeds = (2026, 2027, 2028)
        # Step 22.37 adds only paired-training, serial CRN validation.
        # Existing test variants and historical protocols remain unchanged.
        if (
            experiment.get("protocol_scope") == "stage22_37_paired_training_CRN_validation"
            and (is_traffic_validation or is_traffic_s4m_validation)
            and diagnostic.get("split") == "val"
            and diagnostic.get("fixed_eval_rng") is True
            and diagnostic.get("rng_protocol") == "stage22_34_crn_v1"
            and official.get("stage22_36_paired_training_rng") is True
        ):
            allowed_seeds = (2028, 2029)
    elif is_external_test or is_stage21_validation:
        allowed_seeds = (2023,)
    else:
        allowed_seeds = (2023, 2024, 2025) if is_test_variant else (2024, 2025)
    if int(experiment["seed"]) not in allowed_seeds:
        raise ValueError(f"{variant} seed must be one of {allowed_seeds}")
    expected_split = "test" if (is_test_variant or is_external_test) else "val"
    if diagnostic.get("split") != expected_split:
        raise ValueError(f"{variant} must use {expected_split} only")
    if is_equal_rate:
        if float(diagnostic["target_rate"]) != 0.15:
            raise ValueError("equal-rate target rate is frozen at 0.15")
        if [int(value) for value in diagnostic["block_lengths"]] != [1, 5, 20]:
            raise ValueError("equal-rate block lengths are frozen at [1, 5, 20]")
        if int(diagnostic["nested_seed"]) != int(experiment["seed"]):
            raise ValueError("equal-rate nested seed must equal the evaluation seed")
        weights = diagnostic["multiscale_weights"]
        if (
            float(weights["original"]) != 0.5
            or float(weights["l5"]) != 0.25
            or float(weights["l20"]) != 0.25
        ):
            raise ValueError(
                "equal-rate multiscale weights are frozen at 0.5/0.25/0.25"
            )
        timing_warmup = int(diagnostic["timing_warmup_batches"])
        if timing_warmup < 0:
            raise ValueError("equal-rate timing warmup must be non-negative")
        max_batches = int(diagnostic["max_eval_batches"])
        if max_batches > 0 and timing_warmup >= max_batches:
            raise ValueError("equal-rate timing warmup must leave a measured batch")
    else:
        if float(diagnostic["nested_target_rate"]) != 0.15:
            raise ValueError("Stage-14 nested target rate is frozen at 0.15")
        nested_block_length = int(diagnostic["nested_block_length"])
        expected_block_lengths = {1, 20} if variant == "stage16_ablation" else {5}
        if nested_block_length not in expected_block_lengths:
            raise ValueError(
                f"{variant} nested block length must be one of "
                f"{sorted(expected_block_lengths)}"
            )
        if int(diagnostic["nested_seed"]) != int(experiment["seed"]):
            raise ValueError("Stage-14 nested seed must equal the evaluation seed")
        save_equal_ensemble = bool(diagnostic.get("save_equal_ensemble", False))
        if is_test_variant and not save_equal_ensemble:
            raise ValueError("CNM-SE test variants require the fixed equal ensemble")
        if save_equal_ensemble and float(diagnostic.get("ensemble_weight", -1.0)) != 0.5:
            raise ValueError("Stage-14 equal ensemble weight is frozen at 0.5")
    max_batches = int(diagnostic["max_eval_batches"])
    if max_batches == 0 or max_batches < -1:
        raise ValueError("max_eval_batches must be -1 or positive")
    if int(official["seed"]) != int(experiment["seed"]):
        raise ValueError("source checkpoint seed mismatch")
    is_s4m_source = is_stage19 or is_traffic_s4m_test or is_traffic_s4m_validation
    if is_s4m_source:
        if bool(official.get("use_mipp", False)):
            raise ValueError("S4M source must remain the original baseline")
    else:
        if not bool(official.get("use_mipp")):
            raise ValueError("Stage-14 source must use PG95/MIPP")
        if str(official.get("mipp_write_gate_mode", "absolute")) != "absolute":
            raise ValueError("Stage-14 source must be PG95 absolute admission")
        if bool(official.get("use_fusr", False)) or bool(official.get("use_cfr", False)):
            raise ValueError("Stage-14 source must not use FUSR or CFR")
        if is_stage20 or is_gated_extension:
            frozen_pg95 = {
                "use_content_mask_factorization": False,
                "use_mask_intervention_pair": True,
                "use_stability_write_gate": True,
                "mipp_inv_weight": 0.05,
                "mipp_var_weight": 0.0,
                "mipp_cov_weight": 0.0,
                "mipp_dec_weight": 0.0,
                "mipp_mask_weight": 0.0,
                "mipp_stability_threshold": 0.95,
                "mipp_pair_rate_min": 0.15,
                "mipp_pair_rate_max": 0.50,
                "mipp_pair_block_lengths": [1, 5, 20],
            }
            mismatches = {
                key: {"expected": expected, "actual": official.get(key)}
                for key, expected in frozen_pg95.items()
                if official.get(key) != expected
            }
            if mismatches:
                raise ValueError(
                    f"{variant} source does not match frozen PG95: {mismatches}"
                )
    if is_gated_extension:
        if official.get("post_train_evaluation") != "validation":
            raise ValueError(
                "gated source must be trained with validation-only post-evaluation"
            )
        expected_variables = 321 if is_stage21 else 862
        if (
            int(official.get("enc_in", -1)) != expected_variables
            or int(official.get("dec_in", -1)) != expected_variables
            or int(official.get("c_out", -1)) != expected_variables
        ):
            raise ValueError(
                f"gated source must use all {expected_variables} variables"
            )
        if (
            int(official.get("d_model", -1)) != 512
            or int(official.get("d_ff", -1)) != 512
            or int(official.get("en_conv_hidden_size", -1)) != 512
        ):
            raise ValueError("gated source must use frozen hidden width 512")
        if int(official.get("e_layers", -1)) != 8:
            raise ValueError("gated source must use the pre-frozen 8-layer architecture")
        if is_traffic and str(official.get("target")) != "OT":
            raise ValueError("Traffic source target must be the existing final column OT")


def run_id(config: dict[str, Any], timestamp: str) -> str:
    experiment = config["experiment"]
    smoke = "__smoke" if experiment.get("smoke_test") else ""
    diagnostic = config["diagnostic"]
    variant = diagnostic.get("variant")
    if diagnostic.get('fixed_eval_rng', False):
        family = 's4m' if experiment['method'] == 's4m_baseline' else 'pg95'
        return f'cnm_se_crn_v1_{family}__traffic__validation__seed{experiment["seed"]}{smoke}__{timestamp}'
    if variant == STAGE19_VARIANT:
        prefix = "cnm_se_equal_rate_multiscale_s4m"
        split = "test"
    elif variant == STAGE20_VARIANT:
        prefix = "cnm_se_equal_rate_multiscale_pg95"
        split = "test"
    elif variant == STAGE18_VARIANT:
        prefix = "cnm_se_equal_rate_multiscale_pg95"
        split = "validation"
    elif variant in {
        STAGE21_VALIDATION_VARIANT, STAGE21_TEST_VARIANT,
        STAGE22_21_VALIDATION_VARIANT, STAGE22_21_TEST_VARIANT,
    }:
        prefix = "cnm_se_equal_rate_multiscale_pg95"
        split = (
            "validation"
            if variant in {
                STAGE21_VALIDATION_VARIANT, STAGE22_21_VALIDATION_VARIANT,
            }
            else "test"
        )
    elif variant == STAGE22_24_BATCH_PARALLEL_VARIANT:
        prefix = "cnm_se_batch_parallel_compare_pg95"
        split = "validation"
    elif variant == STAGE22_25_CUDA_STREAM_VARIANT:
        prefix = "cnm_se_cuda_stream_compare_pg95"
        split = "validation"
    elif variant == STAGE22_22_S4M_TEST_VARIANT:
        prefix = "cnm_se_equal_rate_multiscale_s4m"
        split = "test"
    elif variant == STAGE22_34_S4M_VALIDATION_VARIANT:
        prefix = "cnm_se_equal_rate_multiscale_s4m"
        split = "validation"
    elif variant == "stage16_ablation":
        prefix = f"cnm_se_ablation_l{int(diagnostic['nested_block_length'])}_pg95"
        split = "test"
    elif variant == "stage15_test":
        prefix = "cnm_se_pg95"
        split = "test"
    else:
        prefix = (
            "stage14_equal_ensemble_pg95"
            if diagnostic.get("save_equal_ensemble", False)
            else "stage14_uncertainty_pg95"
        )
        split = "validation"
    return (
        f"{prefix}__{experiment['dataset']}__{split}"
        f"__seed{experiment['seed']}{smoke}__{timestamp}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-uuid", default=None)
    parser.add_argument("--scheduler-attempt", type=int, default=0)
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_config_path = resolve_path(config["source"]["resolved_config"])
    source = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    checkpoint = resolve_path(config["source"]["checkpoint"])
    source_metrics_path = resolve_path(config["source"]["metrics"])
    missing_csv = resolve_path(config["data"]["missing_csv"])
    complete_csv = resolve_path(config["data"]["complete_csv"])
    mask_asset = resolve_path(config["data"]["mask_asset"])
    for path in (
        config_path, source_config_path, checkpoint, source_metrics_path,
        missing_csv, complete_csv, mask_asset,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    validate_config(config, source)

    if args.gpu_uuid:
        config["runtime"]["gpu_uuid"] = args.gpu_uuid
    if config["runtime"].get("device") == "cuda":
        config["runtime"].update(
            resolve_gpu_metadata(str(config["runtime"]["gpu_uuid"]))
        )
    config["runtime"]["scheduler_attempt"] = int(args.scheduler_attempt)

    identifier = run_id(config, dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir = ROOT / "runs" / identifier
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    metrics_path = run_dir / "metrics.json"
    variant = str(config["diagnostic"].get("variant", "stage14_validation"))
    is_test_variant = variant in TEST_VARIANTS
    is_stage18 = variant == STAGE18_VARIANT
    is_stage19 = variant == STAGE19_VARIANT
    is_stage20 = variant == STAGE20_VARIANT
    is_stage21_validation = variant == STAGE21_VALIDATION_VARIANT
    is_stage21_test = variant == STAGE21_TEST_VARIANT
    is_stage21 = is_stage21_validation or is_stage21_test
    is_traffic_validation = variant == STAGE22_21_VALIDATION_VARIANT
    is_traffic_test = variant == STAGE22_21_TEST_VARIANT
    is_traffic_s4m_test = variant == STAGE22_22_S4M_TEST_VARIANT
    is_traffic_s4m_validation = variant == STAGE22_34_S4M_VALIDATION_VARIANT
    is_batch_parallel = variant == STAGE22_24_BATCH_PARALLEL_VARIANT
    is_stream_parallel = variant == STAGE22_25_CUDA_STREAM_VARIANT
    is_traffic = (
        is_traffic_validation or is_traffic_test or is_traffic_s4m_test
        or is_batch_parallel or is_stream_parallel or is_traffic_s4m_validation
    )
    is_gated_validation = (
        is_stage21_validation or is_traffic_validation
        or is_batch_parallel or is_stream_parallel or is_traffic_s4m_validation
    )
    is_gated_test = is_stage21_test or is_traffic_test or is_traffic_s4m_test
    is_gated_extension = is_stage21 or is_traffic
    is_external_test = is_stage19 or is_stage20 or is_gated_test
    is_equal_rate = is_stage18 or is_gated_validation or is_external_test
    samples_path = run_dir / (
        (
            "validation_equal_rate_multiscale_samples.npz"
            if is_stage18 or is_gated_validation
            else "test_equal_rate_multiscale_samples.npz"
        )
        if is_equal_rate
        else (
            "test_nested_ensemble_samples.npz"
            if is_test_variant
            else "validation_uncertainty_samples.npz"
        )
    )
    checkpoint_hash = sha256(checkpoint)
    mask_hash = sha256(mask_asset)
    checkpoint_verification = verify_checkpoint(
        checkpoint,
        require_mipp=not (is_stage19 or is_traffic_s4m_test or is_traffic_s4m_validation),
    )
    source_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8"))
    if is_gated_extension:
        source_identity = {
            "evaluation_split": source_metrics.get("evaluation_split"),
            "post_train_evaluation": source_metrics.get("post_train_evaluation"),
            "test_split_accessed": source_metrics.get("test_split_accessed"),
            "chronological_loader": source_metrics.get("chronological_loader"),
        }
        expected_source_identity = {
            "evaluation_split": "validation",
            "post_train_evaluation": "validation",
            "test_split_accessed": False,
            "chronological_loader": True,
        }
        if source_identity != expected_source_identity:
            raise ValueError(
                "gated source metrics are not an auditable validation-only "
                f"artifact: {source_identity}"
            )

    official = copy.deepcopy(source["official_args"])
    diagnostic = config["diagnostic"]
    save_equal_ensemble = bool(diagnostic.get("save_equal_ensemble", False))
    official.update({
        "is_training": 0,
        "model_id": identifier,
        "root_path": str(missing_csv.parent),
        "data_path": missing_csv.name,
        "true_data_path": str(complete_csv),
        "checkpoints": str(run_dir / "checkpoints"),
        "results_root": str(run_dir / "results"),
        "test_results_root": str(run_dir / "test_results"),
        "metrics_path": str(metrics_path),
        "max_train_batches": -1,
        "max_eval_batches": int(diagnostic["max_eval_batches"]),
        "wandb_mode": "disabled",
        "gate_h_diagnostic": False,
        "hcup_diagnostic": False,
        "stage12_validation_diagnostic": False,
        "stage14_uncertainty_diagnostic": not is_test_variant,
        "stage15_nested_ensemble_test": is_test_variant,
        "stage18_equal_rate_multiscale_validation": is_stage18,
        "stage19_equal_rate_multiscale_test": is_stage19 or is_stage20,
        "stage21_equal_rate_multiscale_evaluation": (
            is_gated_extension and not is_batch_parallel and not is_stream_parallel
        ),
        "stage22_24_batch_parallel_comparison": is_batch_parallel,
        "stage22_25_cuda_stream_comparison": is_stream_parallel,
        "gpu": 0,
    })
    if is_equal_rate:
        weights = diagnostic["multiscale_weights"]
        argument_prefix = (
            "stage21"
            if is_gated_extension
            else ("stage19" if is_external_test else "stage18")
        )
        official.update({
            "stage14_uncertainty_diagnostic": False,
            "stage15_nested_ensemble_test": False,
            "stage18_equal_rate_multiscale_validation": is_stage18,
            "stage19_equal_rate_multiscale_test": is_stage19 or is_stage20,
            "stage21_equal_rate_multiscale_evaluation": (
                is_gated_extension and not is_batch_parallel
                and not is_stream_parallel
            ),
            "stage22_24_batch_parallel_comparison": is_batch_parallel,
            "stage22_25_cuda_stream_comparison": is_stream_parallel,
            f"{argument_prefix}_checkpoint": str(checkpoint),
            f"{argument_prefix}_metrics_path": str(metrics_path),
            f"{argument_prefix}_samples_path": str(samples_path),
            f"{argument_prefix}_target_rate": float(diagnostic["target_rate"]),
            f"{argument_prefix}_block_lengths": [
                int(value) for value in diagnostic["block_lengths"]
            ],
            f"{argument_prefix}_nested_seed": int(diagnostic["nested_seed"]),
            f"{argument_prefix}_original_weight": float(weights["original"]),
            f"{argument_prefix}_l5_weight": float(weights["l5"]),
            f"{argument_prefix}_l20_weight": float(weights["l20"]),
            f"{argument_prefix}_timing_warmup_batches": int(
                diagnostic["timing_warmup_batches"]
            ),
        })
        if is_gated_extension:
            official["stage21_split"] = "test" if is_gated_test else "val"
        if diagnostic.get('fixed_eval_rng', False):
            official['stage22_34_fixed_eval_rng'] = True
    elif is_test_variant:
        official.update({
            "stage15_checkpoint": str(checkpoint),
            "stage15_metrics_path": str(metrics_path),
            "stage15_samples_path": str(samples_path),
            "stage15_nested_target_rate": float(diagnostic["nested_target_rate"]),
            "stage15_nested_block_length": int(diagnostic["nested_block_length"]),
            "stage15_nested_seed": int(diagnostic["nested_seed"]),
            "stage15_ensemble_weight": float(diagnostic["ensemble_weight"]),
        })
    else:
        official.update({
            "stage14_checkpoint": str(checkpoint),
            "stage14_metrics_path": str(metrics_path),
            "stage14_samples_path": str(samples_path),
            "stage14_nested_target_rate": float(diagnostic["nested_target_rate"]),
            "stage14_nested_block_length": int(diagnostic["nested_block_length"]),
            "stage14_nested_seed": int(diagnostic["nested_seed"]),
        })

    resolved = copy.deepcopy(config)
    resolved["experiment"]["id"] = identifier
    resolved["experiment"]["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    resolved["source"].update({
        "resolved_config": str(source_config_path),
        "checkpoint": str(checkpoint),
        "metrics": str(source_metrics_path),
        "checkpoint_sha256": checkpoint_hash,
    })
    resolved["data"]["mask_asset"] = str(mask_asset)
    resolved["data"]["mask_asset_sha256"] = mask_hash
    resolved["official_args"] = official
    resolved["paths"] = {
        "run_dir": str(run_dir),
        "config_source": str(config_path),
        "baseline_repo": str(BASELINE_ROOT),
        "upstream_repo": str(ROOT / "upstream" / "S4M"),
    }
    atomic_text(
        run_dir / "resolved_config.yaml",
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True),
    )
    atomic_text(run_dir / "env.txt", safe_environment_text(config))
    atomic_text(
        run_dir / "git_state.txt",
        "[baseline_adapter]\n" + git_evidence(BASELINE_ROOT)
        + "\n[upstream_snapshot]\n" + git_evidence(ROOT / "upstream" / "S4M"),
    )
    _, diff = run_capture(["git", "diff", "--binary"], BASELINE_ROOT)
    atomic_text(run_dir / "code.diff", diff)
    (run_dir / "checkpoints").mkdir()
    atomic_json(
        run_dir / "checkpoints" / "source_checkpoint.json",
        {"path": str(checkpoint), "sha256": checkpoint_hash, **checkpoint_verification},
    )

    command = [sys.executable, str(S4M_ROOT / "run.py"), *cli_from_mapping(official)]
    python_path = os.pathsep.join([
        str(S4M_ROOT),
        str(S4M_ROOT / "model"),
        str(S4M_ROOT / "model" / "S4" / "s4"),
        str(STRUCTURED_KERNEL_ROOT),
    ])
    gpu_export = ""
    if config["runtime"].get("gpu_uuid"):
        gpu_export = "export CUDA_VISIBLE_DEVICES=" + shlex.quote(
            str(config["runtime"]["gpu_uuid"])
        ) + "\n"
    command_text = (
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {shlex.quote(str(S4M_ROOT))}\n"
        f"export PYTHONPATH={shlex.quote(python_path)}\n"
        + gpu_export
        + "export PYTHONDONTWRITEBYTECODE=1\n"
        + "export WANDB_MODE=disabled\n"
        + shlex.join(command) + "\n"
    )
    atomic_text(run_dir / "command.sh", command_text)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = python_path
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["WANDB_MODE"] = "disabled"
    environment["MPLCONFIGDIR"] = str(run_dir / ".matplotlib")
    if config["runtime"].get("gpu_uuid"):
        environment["CUDA_VISIBLE_DEVICES"] = str(config["runtime"]["gpu_uuid"])

    status = {
        "experiment_id": identifier,
        "state": "RUNNING",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": str(config_path),
        "evaluation_seed": int(config["experiment"]["seed"]),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": checkpoint_hash,
    }
    atomic_json(run_dir / "status.json", status)
    print(f"run_dir={run_dir}", flush=True)
    started = time.monotonic()
    with (run_dir / "train.log").open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=S4M_ROOT,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        status["pid"] = process.pid
        atomic_json(run_dir / "status.json", status)
        return_code = process.wait()
    elapsed = time.monotonic() - started

    metrics: dict[str, Any] = {}
    verification_error = None
    if return_code == 0 and metrics_path.is_file() and samples_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            with np.load(samples_path) as samples:
                count = int(metrics["sample_count"])
                if is_batch_parallel or is_stream_parallel:
                    parallel_timing_key = (
                        "timing_parallel_stream_forward_seconds"
                        if is_stream_parallel
                        else "timing_parallel_concat_forward_split_seconds"
                    )
                    vector_keys = (
                        "sample_index", "complete_count", "target_removed_count",
                        "removed_count_l5", "removed_count_l20",
                        "additional_missing_rate_l5",
                        "additional_missing_rate_l20",
                        "complete_sse_serial_multiscale",
                        "complete_sae_serial_multiscale",
                        "complete_sse_parallel_multiscale",
                        "complete_sae_parallel_multiscale",
                        *[
                            f"serial_parallel_{kind}_{route}"
                            for route in ("original", "l5", "l20", "multiscale")
                            for kind in ("mse", "max_abs")
                        ],
                    )
                    timing_keys = (
                        "timing_l5_mask_seconds",
                        "timing_l5_preprocess_seconds",
                        "timing_l20_mask_seconds",
                        "timing_l20_preprocess_seconds",
                        "timing_serial_forward_seconds",
                        "timing_serial_fusion_seconds",
                        parallel_timing_key,
                        "timing_parallel_fusion_seconds",
                        "timing_serial_pipeline_seconds",
                        "timing_parallel_pipeline_seconds",
                        "timing_paired_reduction_percent",
                        "timing_execution_order_parallel_first",
                    )
                    original = np.unpackbits(
                        samples["history_mask_packed"], axis=1
                    )[:, :192]
                    nested_masks = {
                        block: np.unpackbits(
                            samples[f"l{block}_mask_packed"], axis=1
                        )[:, :192]
                        for block in (5, 20)
                    }
                    expected_removed = np.rint(
                        0.15 * original.sum(axis=1)
                    ).astype(np.int64)
                    count_total = samples["complete_count"].sum()
                    recalculated = {
                        f"complete_target_{metric}_{route}_multiscale": float(
                            samples[
                                f"complete_s{'s' if metric == 'mse' else 'a'}e_"
                                f"{route}_multiscale"
                            ].sum() / count_total
                        )
                        for metric in ("mse", "mae")
                        for route in ("serial", "parallel")
                    }
                    prediction_tolerance = float(
                        metrics["prediction_equivalence_tolerance"]
                    )
                    metric_tolerance = float(
                        metrics["metric_equivalence_tolerance"]
                    )
                    expected_equivalence = (
                        prediction_tolerance == 1e-4
                        and metric_tolerance == 1e-6
                        and all(
                            float(
                                samples[f"serial_parallel_max_abs_{route}"].max()
                            ) <= prediction_tolerance
                            for route in ("original", "l5", "l20", "multiscale")
                        )
                        and abs(
                            recalculated["complete_target_mse_serial_multiscale"]
                            - recalculated["complete_target_mse_parallel_multiscale"]
                        ) <= metric_tolerance
                        and abs(
                            recalculated["complete_target_mae_serial_multiscale"]
                            - recalculated["complete_target_mae_parallel_multiscale"]
                        ) <= metric_tolerance
                    )
                    measured_batches = int(metrics["timing_measured_batches"])
                    order = samples["timing_execution_order_parallel_first"]
                    serial_pipeline = samples["timing_serial_pipeline_seconds"]
                    parallel_pipeline = samples["timing_parallel_pipeline_seconds"]
                    reduction = (
                        (serial_pipeline - parallel_pipeline)
                        / serial_pipeline * 100.0
                    )
                    checks = {
                        "sample_count_positive": count > 0,
                        "all_vectors_match": all(
                            samples[key].size == count for key in vector_keys
                        ),
                        "packed_shapes_match": (
                            samples["history_mask_packed"].shape == (count, 24)
                            and all(
                                samples[f"l{block}_mask_packed"].shape == (count, 24)
                                for block in (5, 20)
                            )
                        ),
                        "chronological_indices": bool(
                            np.array_equal(samples["sample_index"], np.arange(count))
                        ),
                        "nested_masks_are_subsets": all(
                            bool((nested_masks[block] <= original).all())
                            for block in (5, 20)
                        ),
                        "strict_equal_removed_count_per_sample": all(
                            np.array_equal(
                                samples[f"removed_count_l{block}"],
                                samples["target_removed_count"],
                            )
                            for block in (5, 20)
                        ),
                        "target_count_roundtrip": bool(
                            np.array_equal(
                                samples["target_removed_count"], expected_removed
                            )
                        ),
                        "finite_vectors": all(
                            np.isfinite(samples[key]).all()
                            for key in vector_keys[1:]
                        ),
                        "positive_counts": bool(
                            (samples["complete_count"] > 0).all()
                        ),
                        "validation_only": (
                            metrics.get("uses_test_split") is False
                            and metrics.get("source_split") == "val"
                            and metrics.get("updates_model_or_memory") is False
                        ),
                        "frozen_batch_parallel_protocol": (
                            metrics.get("execution_optimization_only") is True
                            and metrics.get("parallel_execution_mode")
                            == (
                                "cuda_streams"
                                if is_stream_parallel else "batch_concat"
                            )
                            and metrics.get("block_lengths") == [5, 20]
                            and metrics.get("multiscale_weights") == {
                                "original": 0.5, "l5": 0.25, "l20": 0.25,
                            }
                            and int(metrics["parallel_effective_model_batch_size"])
                            == (
                                int(official["batch_size"])
                                if is_stream_parallel
                                else 3 * int(official["batch_size"])
                            )
                            and int(metrics["parallel_aggregate_inflight_batch_size"])
                            == 3 * int(official["batch_size"])
                            and int(metrics["parallel_model_call_count"])
                            == (3 if is_stream_parallel else 1)
                            and int(metrics["parallel_stream_count"])
                            == (3 if is_stream_parallel else 1)
                        ),
                        "metric_roundtrip": all(
                            abs(recalculated[key] - float(metrics[key])) <= 1e-12
                            for key in recalculated
                        ),
                        "equivalence_report_consistent": (
                            bool(metrics["prediction_equivalence_pass"])
                            is bool(expected_equivalence)
                            and all(
                                abs(
                                    float(metrics["maximum_absolute_prediction_difference"][route])
                                    - float(samples[f"serial_parallel_max_abs_{route}"].max())
                                ) <= 1e-12
                                for route in ("original", "l5", "l20", "multiscale")
                            )
                        ),
                        "paired_same_gpu_timing_protocol": (
                            metrics.get("timing_device_same_gpu") is True
                            and metrics.get("timing_execution_order_alternates") is True
                            and measured_batches
                            == int(metrics["validation_batches_processed"])
                            - int(metrics["timing_warmup_batches"])
                            and set(order.tolist()) == {0, 1}
                        ),
                        "timing_arrays_match": all(
                            samples[key].size == measured_batches
                            for key in timing_keys
                        ),
                        "timing_finite": all(
                            np.isfinite(samples[key]).all() for key in timing_keys
                        ),
                        "positive_pipeline_timings": bool(
                            (serial_pipeline > 0).all()
                            and (parallel_pipeline > 0).all()
                        ),
                        "paired_reduction_roundtrip": bool(
                            np.allclose(
                                samples["timing_paired_reduction_percent"],
                                reduction, rtol=0.0, atol=1e-12,
                            )
                            and abs(
                                float(reduction.mean())
                                - float(metrics["paired_latency_reduction_percent_mean"])
                            ) <= 1e-12
                        ),
                        "source_is_validation_only": (
                            source_metrics.get("evaluation_split") == "validation"
                            and source_metrics.get("post_train_evaluation") == "validation"
                            and source_metrics.get("test_split_accessed") is False
                            and source_metrics.get("chronological_loader") is True
                        ),
                    }
                elif is_equal_rate:
                    labels = (
                        "on", "l1", "l5", "l20",
                        "ensemble_l1", "ensemble_l5", "ensemble_l20", "multiscale",
                    )
                    vector_keys = (
                        "sample_index", "complete_count", "target_removed_count",
                        *[
                            key
                            for label in labels
                            for key in (
                                f"complete_sse_{label}",
                                f"complete_sae_{label}",
                            )
                        ],
                        *[
                            key
                            for block_length in (1, 5, 20)
                            for key in (
                                f"removed_count_l{block_length}",
                                f"additional_missing_rate_l{block_length}",
                                f"prediction_difference_mse_l{block_length}",
                            )
                        ],
                    )
                    timing_keys = (
                        "timing_original_forward_seconds",
                        *[
                            f"timing_l{block_length}_{component}_seconds"
                            for block_length in (1, 5, 20)
                            for component in ("mask", "preprocess", "forward", "fusion")
                        ],
                        "timing_multiscale_fusion_seconds",
                    )
                    original = np.unpackbits(
                        samples["history_mask_packed"], axis=1
                    )[:, :192]
                    nested_masks = {
                        block_length: np.unpackbits(
                            samples[f"l{block_length}_mask_packed"], axis=1
                        )[:, :192]
                        for block_length in (1, 5, 20)
                    }
                    expected_removed = np.rint(
                        0.15 * original.sum(axis=1)
                    ).astype(np.int64)
                    recalculated = {
                        f"complete_target_{metric}_{label}": float(
                            samples[
                                f"complete_s{'s' if metric == 'mse' else 'a'}e_{label}"
                            ].sum() / samples["complete_count"].sum()
                        )
                        for metric in ("mse", "mae")
                        for label in labels
                    }
                    counts_equal = all(
                        np.array_equal(
                            samples[f"removed_count_l{block_length}"],
                            samples["target_removed_count"],
                        )
                        for block_length in (1, 5, 20)
                    )
                    rates_roundtrip = all(
                        np.allclose(
                            samples[f"additional_missing_rate_l{block_length}"],
                            samples[f"removed_count_l{block_length}"]
                            / np.maximum(original.sum(axis=1), 1),
                            rtol=0.0,
                            atol=1e-7,
                        )
                        for block_length in (1, 5, 20)
                    )
                    prediction_difference_roundtrip = all(
                        abs(
                            float(
                                samples[
                                    f"prediction_difference_mse_l{block_length}"
                                ].mean()
                            )
                            - float(
                                metrics[
                                    f"prediction_difference_mse_l{block_length}_mean"
                                ]
                            )
                        ) <= 1e-12
                        and abs(
                            float(
                                metrics[
                                    "prediction_difference_to_original_mse_ratio_"
                                    f"l{block_length}"
                                ]
                            )
                            - float(
                                metrics[
                                    f"prediction_difference_mse_l{block_length}_mean"
                                ]
                                / metrics["complete_target_mse_on"]
                            )
                        ) <= 1e-12
                        for block_length in (1, 5, 20)
                    )
                    measured_batches = int(metrics["timing_measured_batches"])
                    checks = {
                        "sample_count_positive": count > 0,
                        "all_vectors_match": all(
                            samples[key].size == count for key in vector_keys
                        ),
                        "packed_shapes_match": (
                            samples["history_mask_packed"].shape == (count, 24)
                            and all(
                                samples[f"l{block_length}_mask_packed"].shape
                                == (count, 24)
                                for block_length in (1, 5, 20)
                            )
                        ),
                        "chronological_indices": bool(
                            np.array_equal(samples["sample_index"], np.arange(count))
                        ),
                        "nested_masks_are_subsets": all(
                            bool((nested_masks[block_length] <= original).all())
                            for block_length in (1, 5, 20)
                        ),
                        "all_scales_remove_rows": all(
                            bool((nested_masks[block_length] < original).any())
                            for block_length in (1, 5, 20)
                        ),
                        "target_count_roundtrip": bool(
                            np.array_equal(
                                samples["target_removed_count"], expected_removed
                            )
                        ),
                        "strict_equal_removed_count_per_sample": counts_equal,
                        "additional_rates_roundtrip": rates_roundtrip,
                        "prediction_difference_roundtrip": (
                            prediction_difference_roundtrip
                        ),
                        "scales_change_mask_organization": (
                            not np.array_equal(nested_masks[1], nested_masks[5])
                            and not np.array_equal(nested_masks[5], nested_masks[20])
                        ),
                        "finite_vectors": all(
                            np.isfinite(samples[key]).all()
                            for key in vector_keys[1:]
                        ),
                        "positive_counts": bool(
                            (samples["complete_count"] > 0).all()
                        ),
                        **{
                            (
                                "expected_external_test_split"
                                if is_external_test
                                else "expected_validation_split"
                            ): (
                                (
                                    metrics.get("uses_test_split") is True
                                    and metrics.get("source_split") == "test"
                                )
                                if is_external_test
                                else (
                                    metrics.get("uses_test_split") is False
                                    and metrics.get("source_split") == "val"
                                )
                            )
                        },
                        "frozen": metrics.get("updates_model_or_memory") is False,
                        "metric_roundtrip": all(
                            abs(recalculated[key] - float(metrics[key])) <= 1e-12
                            for key in recalculated
                        ),
                        "frozen_equal_rate_protocol": (
                            float(metrics["target_additional_missing_rate"]) == 0.15
                            and metrics["block_lengths"] == [1, 5, 20]
                            and int(metrics["nested_seed"])
                            == int(config["experiment"]["seed"])
                            and metrics["target_count_rounding"]
                            == "numpy_rint_ties_to_even"
                        ),
                        **(fixed_rng_reference_checks(metrics, samples)
                           if diagnostic.get('fixed_eval_rng', False) else {}),
                        "fixed_multiscale_weights": metrics["multiscale_weights"] == {
                            "original": 0.5, "l5": 0.25, "l20": 0.25
                        },
                        "serial_timing_protocol": (
                            metrics.get("timing_device_serial") is True
                            and metrics.get("timing_fixed_forward_order")
                            == ["original", "l1", "l5", "l20"]
                            and measured_batches
                            == int(
                                metrics[
                                    (
                                        "test_batches_processed"
                                        if is_external_test
                                        else "validation_batches_processed"
                                    )
                                ]
                            )
                            - int(metrics["timing_warmup_batches"])
                        ),
                        "timing_arrays_match": all(
                            samples[key].size == measured_batches
                            for key in timing_keys
                        ),
                        "timing_finite_positive": all(
                            np.isfinite(samples[key]).all()
                            and bool((samples[key] > 0).all())
                            for key in timing_keys
                        ),
                        **(
                            {
                                "matched_source_baseline": (
                                    bool(config["experiment"].get("smoke_test"))
                                    or (
                                        abs(
                                            float(
                                                metrics[
                                                    "complete_target_mse_on"
                                                ]
                                            )
                                            - float(source_metrics["mse"])
                                        ) <= 1e-5
                                        and abs(
                                            float(
                                                metrics[
                                                    "complete_target_mae_on"
                                                ]
                                            )
                                            - float(source_metrics["mae"])
                                        ) <= 1e-5
                                    )
                                )
                            }
                            if is_external_test and not is_gated_test
                            else {}
                        ),
                        **(
                            {
                                "matched_source_validation_baseline": (
                                    (
                                        metrics.get('evaluation_rng_protocol') == 'stage22_34_crn_v1'
                                        and all(
                                            abs(float(metrics[f'complete_target_{m}_on'])
                                                - float(metrics[f'fixed_rng_reference_{m}'])) <= 1e-5
                                            and float(metrics[f'fixed_rng_max_window_{m}_difference']) <= 1e-5
                                            for m in ('mse', 'mae')
                                        )
                                    ) if diagnostic.get('fixed_eval_rng', False) else (
                                    bool(config["experiment"].get("smoke_test"))
                                    or (
                                        abs(
                                            float(metrics["complete_target_mse_on"])
                                            - float(source_metrics["mse"])
                                        ) <= 1e-5
                                        and abs(
                                            float(metrics["complete_target_mae_on"])
                                            - float(source_metrics["mae"])
                                        ) <= 1e-5
                                    )
                                    )
                                ),
                                "source_is_validation_only": (
                                    source_metrics.get("evaluation_split")
                                    == "validation"
                                    and source_metrics.get("post_train_evaluation")
                                    == "validation"
                                    and source_metrics.get("test_split_accessed") is False
                                    and source_metrics.get("chronological_loader") is True
                                ),
                            }
                            if is_gated_validation
                            else {}
                        ),
                        **(
                            {
                                "source_is_validation_only": (
                                    source_metrics.get("evaluation_split")
                                    == "validation"
                                    and source_metrics.get("post_train_evaluation")
                                    == "validation"
                                    and source_metrics.get("test_split_accessed") is False
                                    and source_metrics.get("chronological_loader") is True
                                )
                            }
                            if is_gated_test
                            else {}
                        ),
                    }
                else:
                    vector_keys = (
                        "sample_index", "complete_sse_on", "complete_sae_on",
                        "complete_sse_nested", "complete_sae_nested", "complete_count",
                        "mask_disagreement_mse", "prototype_disagreement_mse",
                        "additional_missing_rate",
                    )
                    if not is_test_variant:
                        vector_keys = vector_keys[:3] + (
                            "complete_sse_off", "complete_sae_off",
                        ) + vector_keys[3:]
                    else:
                        vector_keys = tuple(
                            key for key in vector_keys
                            if key != "prototype_disagreement_mse"
                        )
                    if save_equal_ensemble:
                        vector_keys = vector_keys[:-3] + (
                            "complete_sse_ensemble", "complete_sae_ensemble",
                        ) + vector_keys[-3:]
                    original = np.unpackbits(
                        samples["history_mask_packed"], axis=1
                    )[:, :192]
                    nested = np.unpackbits(
                        samples["nested_mask_packed"], axis=1
                    )[:, :192]
                    recalculated = {
                        f"complete_target_{metric}_{mode}": float(
                            samples[
                                f"complete_s{'s' if metric == 'mse' else 'a'}e_{mode}"
                            ].sum() / samples["complete_count"].sum()
                        )
                        for metric in ("mse", "mae")
                        for mode in (
                            (
                                ("on", "nested", "ensemble")
                                if is_test_variant
                                else ("on", "off", "nested", "ensemble")
                            )
                            if save_equal_ensemble else ("on", "off", "nested")
                        )
                    }
                    checks = {
                        "sample_count_positive": count > 0,
                        "all_vectors_match": all(
                            samples[key].size == count for key in vector_keys
                        ),
                        "packed_shapes_match": (
                            samples["history_mask_packed"].shape == (count, 24)
                            and samples["nested_mask_packed"].shape == (count, 24)
                        ),
                        "chronological_indices": bool(
                            np.array_equal(samples["sample_index"], np.arange(count))
                        ),
                        "nested_is_subset": bool((nested <= original).all()),
                        "nested_removes_observed_rows": bool(
                            (nested < original).any()
                        ),
                        "finite_vectors": all(
                            np.isfinite(samples[key]).all()
                            for key in vector_keys[1:]
                        ),
                        "positive_counts": bool(
                            (samples["complete_count"] > 0).all()
                        ),
                        "expected_split": (
                            (
                                metrics.get("uses_test_split") is True
                                and metrics.get("source_split") == "test"
                            )
                            if is_test_variant
                            else (
                                metrics.get("uses_test_split") is False
                                and metrics.get("source_split") == "val"
                            )
                        ),
                        "frozen": metrics.get("updates_model_or_memory") is False,
                        "metric_roundtrip": all(
                            abs(recalculated[key] - float(metrics[key])) <= 1e-12
                            for key in recalculated
                        ),
                        "frozen_nested_protocol": (
                            float(metrics["nested_target_rate"]) == 0.15
                            and int(metrics["nested_block_length"])
                            == int(diagnostic["nested_block_length"])
                            and int(metrics["nested_seed"])
                            == int(config["experiment"]["seed"])
                        ),
                        "fixed_equal_ensemble": (
                            not save_equal_ensemble
                            or (
                                float(metrics["ensemble_weight_on"]) == 0.5
                                and float(metrics["ensemble_weight_nested"]) == 0.5
                            )
                        ),
                        "matched_source_baseline": (
                            not is_test_variant
                            or bool(config["experiment"].get("smoke_test"))
                            or (
                                abs(
                                    float(metrics["complete_target_mse_on"])
                                    - float(source_metrics["mse"])
                                ) <= 1e-5
                                and abs(
                                    float(metrics["complete_target_mae_on"])
                                    - float(source_metrics["mae"])
                                ) <= 1e-5
                            )
                        ),
                    }
            if not all(checks.values()):
                raise RuntimeError(
                    f"{'equal-rate multiscale' if is_equal_rate else 'Stage-14'} "
                    f"artifact checks failed: {checks}"
                )
            metrics.update({
                "experiment_id": identifier,
                "dataset": config["experiment"]["dataset"],
                "method": (
                    (
                        "cnm_se_cuda_stream_compare_pg95"
                        if is_stream_parallel
                        else "cnm_se_batch_parallel_compare_pg95"
                    )
                    if is_batch_parallel or is_stream_parallel
                    else
                    (
                        "cnm_se_equal_rate_multiscale_s4m"
                        if is_stage19 or is_traffic_s4m_validation
                        else (
                            "cnm_se_equal_rate_multiscale_pg95"
                            if is_stage20 or is_gated_extension
                            else "cnm_se_equal_rate_multiscale"
                        )
                    )
                    if is_equal_rate
                    else (
                        "cnm_se"
                        if is_test_variant
                        else (
                            "pg95_equal_nested_ensemble"
                            if save_equal_ensemble else "pg95"
                        )
                    )
                ),
                "evaluation_seed": int(config["experiment"]["seed"]),
                "gpu_uuid": config["runtime"].get("gpu_uuid"),
                "scope": (
                    "smoke_test_not_formal_result"
                    if config["experiment"].get("smoke_test")
                    else (
                        (
                            "formal_validation_cuda_stream_equivalence_and_paired_timing"
                            if is_stream_parallel
                            else "formal_validation_batch_parallel_equivalence_and_paired_timing"
                        )
                        if is_batch_parallel or is_stream_parallel
                        else (
                        "formal_validation_equal_rate_multiscale_and_serial_timing"
                        if is_stage18 or is_gated_validation
                        else (
                            "formal_external_test_equal_rate_multiscale_and_serial_timing"
                            if is_external_test
                        else (
                            "formal_test_causal_nested_mask_self_ensemble"
                            if variant == "stage15_test"
                            else (
                                f"formal_test_cnm_se_block_length_"
                                f"{int(diagnostic['nested_block_length'])}_ablation"
                                if variant == "stage16_ablation"
                                else (
                                    "formal_validation_only_equal_ensemble"
                                    if save_equal_ensemble
                                    else "formal_validation_only_uncertainty_cache"
                                )
                            )
                        )))
                    )
                ),
                "source_formal_test_mse_reference_only": source_metrics.get("mse"),
                "source_formal_test_mae_reference_only": source_metrics.get("mae"),
                "source_validation_mse_reference_only": (
                    source_metrics.get("mse") if is_gated_extension else None
                ),
                "source_validation_mae_reference_only": (
                    source_metrics.get("mae") if is_gated_extension else None
                ),
                **({
                    'historical_validation_difference_reference_only': {
                        m: float(metrics[f'complete_target_{m}_on']) - float(source_metrics[m])
                        for m in ('mse', 'mae')},
                    'baseline_consistency_reference': 'new_separate_original_only_CRN_v1',
                    'timing_is_isolated_benchmark': False,
                } if diagnostic.get('fixed_eval_rng', False) else {}),
                "source_checkpoint": str(checkpoint),
                "source_checkpoint_sha256": checkpoint_hash,
                "mask_asset": str(mask_asset),
                "mask_asset_sha256": mask_hash,
                "evaluation_samples_sha256": sha256(samples_path),
                "artifact_checks": checks,
                "artifact_status": "PASS",
                **checkpoint_verification,
            })
            atomic_json(metrics_path, metrics)
        except Exception as exc:
            verification_error = f"{type(exc).__name__}: {exc}"

    log_text = (run_dir / "train.log").read_text(encoding="utf-8", errors="replace").lower()
    failure_state = "OOM" if return_code != 0 and "out of memory" in log_text else "CODE_ERROR"
    success = (
        return_code == 0 and metrics_path.is_file() and samples_path.is_file()
        and verification_error is None and metrics.get("artifact_status") == "PASS"
    )
    atomic_json(run_dir / "resource_usage.json", {
        "elapsed_seconds": elapsed,
        "device": config["runtime"].get("device"),
        "gpu_uuid": config["runtime"].get("gpu_uuid"),
        "peak_gpu_memory_mb": metrics.get("peak_gpu_memory_mb"),
        "peak_gpu_reserved_mb": metrics.get("peak_gpu_reserved_mb"),
        "return_code": return_code,
    })
    status.update({
        "state": "SUCCESS" if success else failure_state,
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "return_code": return_code,
        "metrics_path": str(metrics_path) if metrics_path.is_file() else None,
        "samples_path": str(samples_path) if samples_path.is_file() else None,
        "verification_error": verification_error,
        **checkpoint_verification,
    })
    atomic_json(run_dir / "status.json", status)
    print(json.dumps(status, indent=2, sort_keys=True))
    print(f"run_dir={run_dir}")
    return 0 if success else (return_code or 2)


if __name__ == "__main__":
    raise SystemExit(main())
