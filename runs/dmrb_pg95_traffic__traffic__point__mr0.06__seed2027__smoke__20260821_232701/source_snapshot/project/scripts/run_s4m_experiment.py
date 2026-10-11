#!/usr/bin/env python3
"""Run one auditable S4M experiment without modifying the upstream snapshot."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any

# The launcher only orchestrates a separate training process. Hiding GPUs here
# prevents its own PyTorch import/checkpoint verification from reserving a CUDA
# context; the child receives the selected UUID explicitly below.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = PROJECT_ROOT / "baselines" / "S4M"
S4M_ROOT = BASELINE_ROOT / "s4m"
STRUCTURED_KERNEL_ROOT = S4M_ROOT / "model" / "S4" / "s4" / "extensions" / "kernels"

FUSR_REQUIRED_CONFIG_KEYS = (
    "use_utility_read_gate",
    "use_fusr_spectral_uncertainty",
    "fusr_hidden_dim",
    "fusr_util_weight",
    "fusr_sign_weight",
    "fusr_gate_temperature",
    "fusr_uncertainty_penalty",
    "fusr_delta",
    "fusr_utility_clip",
    "fusr_neutral_margin",
    "fusr_eval_paired_diagnostics",
)

FUSR_V2_REQUIRED_CONFIG_KEYS = (
    "fusr_objective_version",
    "fusr_feature_set",
    "fusr_gate_score",
    "fusr_gate_prior_mode",
    "fusr_router_only",
    "fusr_backbone_checkpoint",
    "fusr_router_epochs",
    "fusr_router_patience",
    "fusr_router_batch_size",
    "fusr_router_lr",
    "fusr_router_weight_decay",
    "fusr_rank_weight",
    "fusr_fit_fraction",
    "fusr_dev_fraction",
    "fusr_embargo_windows",
    "fusr_cache_split",
    "fusr_cache_max_batches",
    "fusr_calibration_mode",
    "fusr_calibration_tie_break",
)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_capture(command: list[str], cwd: Path) -> tuple[int, str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.returncode, completed.stdout


def resolve_gpu_metadata(gpu_uuid: str) -> dict[str, Any]:
    code, output = run_capture(
        [
            "nvidia-smi",
            "--query-gpu=uuid,index,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        PROJECT_ROOT,
    )
    if code != 0:
        raise RuntimeError(f"nvidia-smi failed while resolving {gpu_uuid}: {output.strip()}")
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 6 and fields[0] == gpu_uuid:
            return {
                "gpu_uuid": fields[0],
                "gpu_physical_index": int(fields[1]),
                "gpu_name": fields[2],
                "gpu_total_mib": int(fields[3]),
                "gpu_used_mib_at_resolution": int(fields[4]),
                "gpu_free_mib_at_resolution": int(fields[5]),
            }
    raise RuntimeError(f"requested GPU UUID was not reported by nvidia-smi: {gpu_uuid}")


def git_evidence(repo: Path) -> str:
    sections: list[str] = []
    for label, command in (
        ("commit", ["git", "rev-parse", "HEAD"]),
        ("status", ["git", "status", "--short"]),
        ("diff_stat", ["git", "diff", "--stat"]),
    ):
        code, output = run_capture(command, repo)
        sections.append(f"[{label}] exit={code}\n{output.rstrip()}\n")
    return "\n".join(sections)


def cli_from_mapping(arguments: dict[str, Any]) -> list[str]:
    cli: list[str] = []
    for key, value in arguments.items():
        flag = "--" + key
        if isinstance(value, bool):
            if value:
                cli.append(flag)
            continue
        if isinstance(value, list):
            cli.append(flag)
            cli.extend(str(item) for item in value)
            continue
        cli.extend([flag, str(value)])
    return cli


def experiment_id(config: dict[str, Any], timestamp: str) -> str:
    exp = config["experiment"]
    smoke = "__smoke" if exp.get("smoke_test") else ""
    return (
        f"{exp['model']}__{exp['dataset']}__{exp['mask_type']}"
        f"__mr{exp['nominal_missing_rate']}__seed{exp['seed']}{smoke}__{timestamp}"
    )


def safe_environment_text(config: dict[str, Any]) -> str:
    runtime = config["runtime"]
    kernel_artifacts = sorted(STRUCTURED_KERNEL_ROOT.glob("structured_kernels*.so"))
    if kernel_artifacts:
        kernel_artifact = kernel_artifacts[-1]
        kernel_sha256 = hashlib.sha256(kernel_artifact.read_bytes()).hexdigest()
    else:
        kernel_artifact = None
        kernel_sha256 = ""
    lines = [
        f"created_at={dt.datetime.now(dt.timezone.utc).isoformat()}",
        f"hostname={platform.node()}",
        f"platform={platform.platform()}",
        f"python_executable={sys.executable}",
        f"python_version={platform.python_version()}",
        f"torch_version={torch.__version__}",
        f"torch_cuda_runtime={torch.version.cuda}",
        "launcher_cuda_probe=skipped_to_avoid_parent_cuda_context",
        f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
        "gpu_mapping=logical_0_to_physical_1;logical_1_to_physical_0",
        f"requested_device={runtime['device']}",
        f"requested_gpu_uuid={runtime.get('gpu_uuid', '')}",
        f"resolved_gpu_physical_index={runtime.get('gpu_physical_index', '')}",
        f"resolved_gpu_name={runtime.get('gpu_name', '')}",
        f"resolved_gpu_total_mib={runtime.get('gpu_total_mib', '')}",
        f"resolved_gpu_free_mib={runtime.get('gpu_free_mib_at_resolution', '')}",
        f"structured_kernel_artifact={kernel_artifact or ''}",
        f"structured_kernel_sha256={kernel_sha256}",
        "conda_environment=s4m_missing_forecasting",
    ]
    return "\n".join(lines) + "\n"


def validate_experiment_config(config: dict[str, Any]) -> None:
    """Reject internally inconsistent innovation configs before creating a run."""
    experiment = config.get("experiment")
    runtime = config.get("runtime")
    official_args = config.get("official_args")
    if not isinstance(experiment, dict) or not isinstance(runtime, dict) or not isinstance(official_args, dict):
        raise ValueError("config must contain experiment, runtime, and official_args mappings")

    if "seed" in official_args and int(official_args["seed"]) != int(experiment["seed"]):
        raise ValueError("experiment.seed and official_args.seed must match")
    for key in ("max_train_batches", "max_eval_batches"):
        if key in runtime and key in official_args and int(runtime[key]) != int(official_args[key]):
            raise ValueError(f"runtime.{key} and official_args.{key} must match")
    post_train_evaluation = str(
        official_args.get("post_train_evaluation", "test")
    )
    if post_train_evaluation not in {"test", "validation"}:
        raise ValueError(
            "official_args.post_train_evaluation must be test or validation"
        )

    use_mipp = bool(official_args.get("use_mipp", False))
    use_stability_gate = bool(official_args.get("use_stability_write_gate", False))
    mipp_write_gate_mode = str(
        official_args.get("mipp_write_gate_mode", "absolute")
    )
    budgeted_write_modes = {
        "budget_topk", "stability_diverse_rank", "mask_regime_balanced_rank"
    }
    if mipp_write_gate_mode not in {"absolute", *budgeted_write_modes}:
        raise ValueError(f"unsupported MIPP write gate mode: {mipp_write_gate_mode}")
    mipp_write_budget = int(
        official_args.get("mipp_write_budget", official_args.get("M", 50))
    )
    if mipp_write_budget < 1:
        raise ValueError("MIPP write budget must be positive")
    if mipp_write_gate_mode in budgeted_write_modes:
        if not use_mipp or not use_stability_gate:
            raise ValueError(
                "MIPP budgeted write modes require use_mipp and use_stability_write_gate"
            )
        if mipp_write_budget != int(official_args.get("M", -1)):
            raise ValueError("MIPP budgeted write modes must equal the existing M write budget")

    use_cfr = bool(official_args.get("use_cfr", False))
    if use_cfr and not use_mipp:
        raise ValueError("CFR requires official_args.use_mipp=true")
    if "cfr" in str(experiment.get("model", "")).lower() and not use_cfr:
        raise ValueError("CFR experiment identity requires official_args.use_cfr=true")

    use_mrra = bool(official_args.get("use_mrra", False))
    use_radr = bool(official_args.get("use_radr", False))
    use_mcpr = bool(official_args.get("use_mcpr", False))
    use_ttrr = bool(official_args.get("use_ttrr", False))
    use_ard = bool(official_args.get("use_ard", False))
    use_bf_cmvd = bool(official_args.get("use_bf_cmvd", False))
    use_dmrb = bool(official_args.get("use_dmrb", False))
    if use_dmrb and not use_mcpr:
        raise ValueError("DMRB requires official_args.use_mcpr=true")
    if not use_dmrb and (
        bool(official_args.get("dmrb_l5_enabled", False))
        or bool(official_args.get("dmrb_l20_enabled", False))
    ):
        raise ValueError("DMRB branch switches require official_args.use_dmrb=true")
    model_identity = str(experiment.get("model", "")).lower()
    if "mrra" in model_identity and not use_mrra:
        raise ValueError("MRRA experiment identity requires official_args.use_mrra=true")
    if "radr" in model_identity and not use_radr:
        raise ValueError("RADR experiment identity requires official_args.use_radr=true")
    if "mcpr" in model_identity and not use_mcpr:
        raise ValueError("MCPR experiment identity requires official_args.use_mcpr=true")
    if "ttrr" in model_identity and not use_ttrr:
        raise ValueError("TTRR experiment identity requires official_args.use_ttrr=true")
    if "ard" in model_identity and not use_ard:
        raise ValueError("ARD experiment identity requires official_args.use_ard=true")
    if "bf_cmvd" in model_identity and not use_bf_cmvd:
        raise ValueError(
            "BF-CMVD experiment identity requires official_args.use_bf_cmvd=true"
        )
    if "dmrb" in model_identity and not use_dmrb:
        raise ValueError("DMRB experiment identity requires official_args.use_dmrb=true")
    if use_ttrr and not use_mcpr:
        raise ValueError("TTRR requires official_args.use_mcpr=true")
    if use_ard and not use_mcpr:
        raise ValueError("ARD requires official_args.use_mcpr=true")
    if use_ard and use_ttrr:
        raise ValueError("ARD and TTRR are mutually exclusive")
    if use_mrra:
        mrra_version = str(official_args.get("mrra_version", "v1"))
        if mrra_version not in {
            "v1",
            "v2_normalized_bounded",
            "v3_gradient_isolated_two_timescale",
        }:
            raise ValueError(f"unsupported MRRA version={mrra_version!r}")
        if "mrra_v2" in model_identity and mrra_version != "v2_normalized_bounded":
            raise ValueError(
                "MRRA-v2 experiment identity requires "
                "official_args.mrra_version=v2_normalized_bounded"
            )
        if (
            "mrra_v3" in model_identity
            and mrra_version != "v3_gradient_isolated_two_timescale"
        ):
            raise ValueError(
                "MRRA-v3 experiment identity requires official_args.mrra_version="
                "v3_gradient_isolated_two_timescale"
            )
        required_mrra = (
            "mrra_rank", "mrra_kernel_size", "mrra_reconstruction_weight"
        )
        missing_mrra = [key for key in required_mrra if key not in official_args]
        if missing_mrra:
            raise ValueError(
                f"MRRA config must explicitly resolve all fields; missing={missing_mrra}"
            )
        if int(official_args["mrra_rank"]) < 1:
            raise ValueError("MRRA rank must be positive")
        kernel_size = int(official_args["mrra_kernel_size"])
        if kernel_size < 1 or kernel_size > int(official_args.get("seq_len", 0)):
            raise ValueError("MRRA kernel size must lie in [1, seq_len]")
        reconstruction_weight = float(
            official_args["mrra_reconstruction_weight"]
        )
        if not math.isfinite(reconstruction_weight) or reconstruction_weight < 0.0:
            raise ValueError("MRRA reconstruction weight must be finite and non-negative")
        lr_multiplier = float(official_args.get("mrra_lr_multiplier", 1.0))
        if not math.isfinite(lr_multiplier) or lr_multiplier <= 0.0:
            raise ValueError("MRRA learning-rate multiplier must be finite and positive")
        if mrra_version == "v3_gradient_isolated_two_timescale":
            if "mrra_lr_multiplier" not in official_args:
                raise ValueError("MRRA-v3 must explicitly resolve mrra_lr_multiplier")
            if not math.isclose(lr_multiplier, 0.1, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("MRRA-v3 requires mrra_lr_multiplier=0.1")
    if use_radr and official_args.get("loss", "MSE") != "MSE":
        raise ValueError("RADR comparison requires the baseline loss identity MSE")
    if (use_mrra or use_radr) and (use_cfr or bool(official_args.get("use_fusr", False))):
        raise ValueError("Step-22 MRRA/RADR candidates cannot be mixed with CFR or FUSR")
    if use_mcpr:
        required_mcpr = (
            "mcpr_objective", "mcpr_rank", "mcpr_kernel_size",
            "mcpr_learning_rate", "mcpr_backbone_checkpoint",
        )
        missing_mcpr = [key for key in required_mcpr if key not in official_args]
        if missing_mcpr:
            raise ValueError(f"MCPR config fields are missing: {missing_mcpr}")
        objective = str(official_args["mcpr_objective"])
        if objective not in {
            "mse_only", "pareto_common_descent", "tail_trust_mse",
            "analytic_decision_distill", "benefit_filtered_multiview_distill",
            "token_benefit_filtered_multiview_distill",
            "dual_scale_residual_basis_distill",
        }:
            raise ValueError("unsupported MCPR objective")
        if int(official_args["mcpr_rank"]) < 1:
            raise ValueError("MCPR rank must be positive")
        kernel_size = int(official_args["mcpr_kernel_size"])
        if kernel_size < 1 or kernel_size > int(official_args.get("seq_len", 0)):
            raise ValueError("MCPR kernel size must lie in [1, seq_len]")
        mcpr_lr = float(official_args["mcpr_learning_rate"])
        if not math.isfinite(mcpr_lr) or mcpr_lr <= 0.0:
            raise ValueError("MCPR learning rate must be finite and positive")
        if not use_mipp:
            raise ValueError("MCPR source identity requires PG95/MIPP enabled")
        if use_mrra or use_radr or use_cfr or bool(official_args.get("use_fusr", False)):
            raise ValueError("MCPR cannot be combined with MRRA, RADR, CFR, or FUSR")
        if use_ttrr:
            required_ttrr = ("ttrr_tail_fraction", "ttrr_tail_weight")
            missing_ttrr = [key for key in required_ttrr if key not in official_args]
            if missing_ttrr:
                raise ValueError(f"TTRR config fields are missing: {missing_ttrr}")
            if objective != "tail_trust_mse":
                raise ValueError("TTRR requires mcpr_objective=tail_trust_mse")
            tail_fraction = float(official_args["ttrr_tail_fraction"])
            tail_weight = float(official_args["ttrr_tail_weight"])
            if not math.isfinite(tail_fraction) or not 0.0 < tail_fraction <= 1.0:
                raise ValueError("TTRR tail fraction must lie in (0, 1]")
            if not math.isfinite(tail_weight) or tail_weight <= 0.0:
                raise ValueError("TTRR tail weight must be finite and positive")
        elif objective == "tail_trust_mse":
            raise ValueError("tail_trust_mse requires official_args.use_ttrr=true")
        if use_ard:
            required_ard = (
                "ard_hidden_dim", "ard_distill_warmup_epochs",
                "ard_energy_epsilon", "ard_action_epsilon",
                "ard_action_weight", "ard_scale_weight",
            )
            missing_ard = [key for key in required_ard if key not in official_args]
            if missing_ard:
                raise ValueError(f"ARD config fields are missing: {missing_ard}")
            if objective != "analytic_decision_distill":
                raise ValueError("ARD requires analytic_decision_distill objective")
            frozen_ard = {
                "ard_hidden_dim": 16,
                "ard_distill_warmup_epochs": 1,
                "ard_energy_epsilon": 1e-8,
                "ard_action_epsilon": 1e-6,
                "ard_action_weight": 1.0,
                "ard_scale_weight": 1.0,
            }
            for key, expected in frozen_ard.items():
                value = official_args[key]
                if isinstance(expected, int):
                    matches = int(value) == expected
                else:
                    matches = math.isclose(
                        float(value), expected, rel_tol=0.0,
                        abs_tol=1e-15 if expected < 1e-5 else 1e-12,
                    )
                if not matches:
                    raise ValueError(f"Step22.19 ARD field {key} is frozen at {expected}")
        elif objective == "analytic_decision_distill":
            raise ValueError("analytic_decision_distill requires official_args.use_ard=true")
        if use_bf_cmvd:
            required_bf_cmvd = (
                "bf_cmvd_target_rate", "bf_cmvd_block_lengths",
                "bf_cmvd_nested_seed", "bf_cmvd_original_weight",
                "bf_cmvd_l5_weight", "bf_cmvd_l20_weight",
            )
            missing_bf_cmvd = [
                key for key in required_bf_cmvd if key not in official_args
            ]
            if missing_bf_cmvd:
                raise ValueError(
                    f"BF-CMVD config fields are missing: {missing_bf_cmvd}"
                )
            if objective not in {
                "benefit_filtered_multiview_distill",
                "token_benefit_filtered_multiview_distill",
                "dual_scale_residual_basis_distill",
            }:
                raise ValueError(
                    "BF-CMVD requires benefit_filtered_multiview_distill objective"
                )
            if use_ttrr or use_ard:
                raise ValueError("BF-CMVD cannot be combined with TTRR or ARD")
            if not math.isclose(
                float(official_args["bf_cmvd_target_rate"]),
                0.15, rel_tol=0.0, abs_tol=1e-12,
            ):
                raise ValueError("BF-CMVD target rate is frozen at 0.15")
            if [int(value) for value in official_args["bf_cmvd_block_lengths"]] != [5, 20]:
                raise ValueError("BF-CMVD block lengths are frozen at [5, 20]")
            if int(official_args["bf_cmvd_nested_seed"]) != int(official_args["seed"]):
                raise ValueError("BF-CMVD nested seed must equal the experiment seed")
            frozen_weights = {
                "bf_cmvd_original_weight": 0.5,
                "bf_cmvd_l5_weight": 0.25,
                "bf_cmvd_l20_weight": 0.25,
            }
            for key, expected in frozen_weights.items():
                if not math.isclose(
                    float(official_args[key]), expected, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError(f"BF-CMVD field {key} is frozen at {expected}")
        elif objective in {
            "benefit_filtered_multiview_distill",
            "token_benefit_filtered_multiview_distill",
            "dual_scale_residual_basis_distill",
        }:
            raise ValueError(
                "benefit_filtered_multiview_distill requires official_args.use_bf_cmvd=true"
            )
        if use_dmrb:
            required_dmrb = ("dmrb_l5_enabled", "dmrb_l20_enabled")
            missing_dmrb = [key for key in required_dmrb if key not in official_args]
            if missing_dmrb:
                raise ValueError(f"DMRB config fields are missing: {missing_dmrb}")
            if not use_bf_cmvd:
                raise ValueError("DMRB requires official_args.use_bf_cmvd=true")
            if objective != "dual_scale_residual_basis_distill":
                raise ValueError("DMRB requires dual_scale_residual_basis_distill objective")
            if not (
                bool(official_args["dmrb_l5_enabled"])
                or bool(official_args["dmrb_l20_enabled"])
            ):
                raise ValueError("DMRB requires at least one enabled scale basis")
            if use_ttrr or use_ard:
                raise ValueError("DMRB cannot be combined with TTRR or ARD")
            if int(official_args["mcpr_rank"]) != 32 or kernel_size != 5:
                raise ValueError("Step22.28 DMRB rank/kernel are frozen at 32/5")
            if official_args.get("mcpr_delta_checkpoint") is not True:
                raise ValueError("Step22.28 DMRB requires mcpr_delta_checkpoint=true")
            if post_train_evaluation != "validation":
                raise ValueError("Step22.28 DMRB is frozen to validation-only evaluation")
        elif (
            bool(official_args.get("dmrb_l5_enabled", False))
            or bool(official_args.get("dmrb_l20_enabled", False))
            or objective == "dual_scale_residual_basis_distill"
        ):
            raise ValueError("DMRB objective/branch switches require official_args.use_dmrb=true")
        elif bool(official_args.get("mcpr_delta_checkpoint", False)):
            raise ValueError("MCPR delta checkpoints are currently restricted to DMRB")
        backbone_value = str(official_args["mcpr_backbone_checkpoint"]).strip()
        backbone_path = Path(backbone_value).expanduser()
        if not backbone_path.is_absolute():
            backbone_path = (PROJECT_ROOT / backbone_path).resolve()
        if not backbone_path.is_file():
            raise FileNotFoundError(f"MCPR backbone checkpoint does not exist: {backbone_path}")
        official_args["mcpr_backbone_checkpoint"] = str(backbone_path)

    use_fusr = bool(official_args.get("use_fusr", False))
    if use_cfr and use_fusr:
        raise ValueError("CFR and FUSR are mutually exclusive")
    objective_version = str(official_args.get("fusr_objective_version", "v1_joint"))
    if objective_version not in {"v1_joint", "v2_frozen_balanced_rank"}:
        raise ValueError(f"unsupported fusr_objective_version={objective_version!r}")
    router_only = official_args.get("fusr_router_only", False)
    if not isinstance(router_only, bool):
        raise TypeError("fusr_router_only must be boolean")
    if router_only and objective_version != "v2_frozen_balanced_rank":
        raise ValueError("fusr_router_only=true is only valid for v2_frozen_balanced_rank")
    if objective_version == "v2_frozen_balanced_rank" and not use_fusr:
        raise ValueError("FUSR V2 requires official_args.use_fusr=true")
    fusr_subswitches = (
        bool(official_args.get("use_utility_read_gate", False)),
        bool(official_args.get("use_fusr_spectral_uncertainty", False)),
        bool(official_args.get("fusr_eval_paired_diagnostics", False)),
        router_only,
    )
    if any(fusr_subswitches) and not use_fusr:
        raise ValueError("FUSR sub-switches require official_args.use_fusr=true")
    if "fusr" in str(experiment.get("model", "")).lower() and not use_fusr:
        raise ValueError("FUSR experiment identity requires official_args.use_fusr=true")
    if not use_fusr:
        return

    missing = [key for key in FUSR_REQUIRED_CONFIG_KEYS if key not in official_args]
    if missing:
        raise ValueError(f"FUSR config must explicitly resolve all FUSR fields; missing={missing}")
    if not bool(official_args["use_utility_read_gate"]):
        raise ValueError("Step-8 FUSR requires use_utility_read_gate=true")
    if not isinstance(official_args["use_fusr_spectral_uncertainty"], bool):
        raise TypeError("use_fusr_spectral_uncertainty must be boolean")
    if not isinstance(official_args["fusr_eval_paired_diagnostics"], bool):
        raise TypeError("fusr_eval_paired_diagnostics must be boolean")

    hidden_dim = int(official_args["fusr_hidden_dim"])
    if hidden_dim < 1:
        raise ValueError("fusr_hidden_dim must be positive")
    numeric = {
        key: float(official_args[key])
        for key in (
            "fusr_util_weight",
            "fusr_sign_weight",
            "fusr_gate_temperature",
            "fusr_uncertainty_penalty",
            "fusr_delta",
            "fusr_utility_clip",
            "fusr_neutral_margin",
        )
    }
    invalid = [key for key, value in numeric.items() if not math.isfinite(value)]
    if invalid:
        raise ValueError(f"FUSR numeric fields must be finite: {invalid}")
    if numeric["fusr_util_weight"] < 0.0 or numeric["fusr_sign_weight"] < 0.0:
        raise ValueError("FUSR loss weights must be non-negative")
    if numeric["fusr_util_weight"] + numeric["fusr_sign_weight"] <= 0.0:
        raise ValueError("at least one FUSR supervision weight must be positive")
    if numeric["fusr_gate_temperature"] <= 0.0:
        raise ValueError("fusr_gate_temperature must be positive")
    if numeric["fusr_uncertainty_penalty"] < 0.0:
        raise ValueError("fusr_uncertainty_penalty must be non-negative")
    if numeric["fusr_utility_clip"] <= 0.0:
        raise ValueError("fusr_utility_clip must be positive")
    if numeric["fusr_neutral_margin"] < 0.0:
        raise ValueError("fusr_neutral_margin must be non-negative")

    if objective_version != "v2_frozen_balanced_rank":
        return

    missing_v2 = [key for key in FUSR_V2_REQUIRED_CONFIG_KEYS if key not in official_args]
    if missing_v2:
        raise ValueError(f"FUSR V2 config must explicitly resolve all V2 fields; missing={missing_v2}")
    if official_args["fusr_router_only"] is not True:
        raise ValueError("FUSR V2 requires fusr_router_only=true")
    required_identity = {
        "fusr_feature_set": "intervention14",
        "fusr_gate_score": "sign_logit",
        "fusr_gate_prior_mode": "none",
        "fusr_cache_split": "val",
        "fusr_calibration_mode": "validation_min_mse",
        "fusr_calibration_tie_break": "prefer_on",
    }
    mismatches = {
        key: official_args[key]
        for key, expected in required_identity.items()
        if str(official_args[key]) != expected
    }
    if mismatches:
        raise ValueError(f"FUSR V2 protocol identity mismatch: {mismatches}")

    backbone_value = str(official_args["fusr_backbone_checkpoint"]).strip()
    if not backbone_value:
        raise ValueError("FUSR V2 requires a non-empty fusr_backbone_checkpoint")
    backbone_path = Path(backbone_value).expanduser()
    if not backbone_path.is_absolute():
        backbone_path = (PROJECT_ROOT / backbone_path).resolve()
    if not backbone_path.is_file():
        raise FileNotFoundError(f"FUSR V2 backbone checkpoint does not exist: {backbone_path}")
    official_args["fusr_backbone_checkpoint"] = str(backbone_path)

    fit_fraction = float(official_args["fusr_fit_fraction"])
    dev_fraction = float(official_args["fusr_dev_fraction"])
    if not all(math.isfinite(value) and value > 0.0 for value in (fit_fraction, dev_fraction)):
        raise ValueError("FUSR V2 fit/dev fractions must be finite and positive")
    if fit_fraction + dev_fraction >= 1.0:
        raise ValueError("FUSR V2 fit_fraction + dev_fraction must be less than 1")
    integer_fields = (
        "fusr_embargo_windows",
        "fusr_router_epochs",
        "fusr_router_patience",
        "fusr_router_batch_size",
        "fusr_cache_max_batches",
    )
    invalid_integer_fields = [
        key
        for key in integer_fields
        if isinstance(official_args[key], bool) or not isinstance(official_args[key], int)
    ]
    if invalid_integer_fields:
        raise TypeError(f"FUSR V2 integer fields must be integers: {invalid_integer_fields}")
    if official_args["fusr_embargo_windows"] < 0:
        raise ValueError("FUSR V2 embargo windows must be non-negative")
    if official_args["fusr_router_epochs"] <= 0:
        raise ValueError("FUSR V2 router epochs must be positive")
    if official_args["fusr_router_batch_size"] <= 0:
        raise ValueError("FUSR V2 router batch size must be positive")
    if official_args["fusr_router_patience"] < 0:
        raise ValueError("FUSR V2 router patience must be non-negative")
    router_lr = float(official_args["fusr_router_lr"])
    if not math.isfinite(router_lr) or router_lr <= 0.0:
        raise ValueError("FUSR V2 router learning rate must be finite and positive")
    for key in ("fusr_router_weight_decay", "fusr_rank_weight"):
        value = float(official_args[key])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"FUSR V2 {key} must be finite and non-negative")
    cache_max_batches = official_args["fusr_cache_max_batches"]
    if cache_max_batches == 0 or cache_max_batches < -1:
        raise ValueError("FUSR V2 cache_max_batches must be -1 or a positive integer")


def verify_checkpoint(
    path: Path,
    require_mipp: bool = False,
    require_fusr: bool = False,
    require_mrra: bool = False,
    require_mcpr: bool = False,
    require_ttrr: bool = False,
    require_ard: bool = False,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    checkpoint_format = str(payload.get("checkpoint_format", "full_model_v1"))
    source_checkpoint = None
    source_checkpoint_sha256 = None
    adapter_model_key_count = None
    if checkpoint_format == "mcpr_delta_v1":
        required = {
            "checkpoint_format", "model", "source_checkpoint",
            "source_checkpoint_sha256",
        }
        missing = sorted(required.difference(payload))
        if missing:
            raise RuntimeError(f"delta checkpoint is missing keys: {missing}")
        source_checkpoint = Path(str(payload["source_checkpoint"])).resolve()
        if not source_checkpoint.is_file():
            raise FileNotFoundError(
                f"delta checkpoint source does not exist: {source_checkpoint}"
            )
        source_checkpoint_sha256 = sha256_file(source_checkpoint)
        if source_checkpoint_sha256 != str(payload["source_checkpoint_sha256"]):
            raise RuntimeError("delta checkpoint source SHA-256 mismatch")
        source_payload = torch.load(source_checkpoint, map_location="cpu")
        source_required = {"model", "memory", "repre", "num_dic"}
        source_missing = sorted(source_required.difference(source_payload))
        if source_missing:
            raise RuntimeError(
                f"delta checkpoint source is missing keys: {source_missing}"
            )
        adapter_state = payload["model"]
        if not isinstance(adapter_state, dict) or not adapter_state:
            raise RuntimeError("delta checkpoint adapter state is empty or invalid")
        adapter_model_key_count = len(adapter_state)
        model_state = dict(source_payload["model"])
        model_state.update(adapter_state)
    elif checkpoint_format == "full_model_v1":
        required = {"model", "memory", "repre", "num_dic"}
        missing = sorted(required.difference(payload))
        if missing:
            raise RuntimeError(f"checkpoint is missing keys: {missing}")
        model_state = payload["model"]
    else:
        raise RuntimeError(f"unsupported checkpoint format: {checkpoint_format}")
    if not isinstance(model_state, dict):
        raise RuntimeError("checkpoint model state is not a mapping")
    nonfinite_state_keys = sorted(
        str(key)
        for key, value in model_state.items()
        if torch.is_tensor(value)
        and (value.is_floating_point() or value.is_complex())
        and not bool(torch.isfinite(value).all().item())
    )
    if nonfinite_state_keys:
        raise RuntimeError(
            "checkpoint contains non-finite model tensors: "
            f"{nonfinite_state_keys[:8]}"
        )
    mipp_state_keys = sorted(key for key in model_state if ".mipp." in key)
    fusr_state_keys = sorted(key for key in model_state if "fusr." in key.lower())
    mrra_state_keys = sorted(
        key for key in model_state
        if key.lower().startswith("mrra.") or ".mrra." in key.lower()
    )
    mcpr_state_keys = sorted(
        key for key in model_state
        if key.lower().startswith("mcpr.") or ".mcpr." in key.lower()
    )
    ttrr_state_keys = sorted(
        key for key in model_state
        if "mcpr.trust_gate." in key.lower()
    )
    ard_state_keys = sorted(
        key for key in model_state
        if "mcpr.ard_gate." in key.lower()
    )
    if require_mipp and not mipp_state_keys:
        raise RuntimeError("MIPP run checkpoint contains no MIPP parameters")
    if require_fusr and not fusr_state_keys:
        raise RuntimeError("FUSR run checkpoint contains no FUSR parameters")
    if require_mrra and not mrra_state_keys:
        raise RuntimeError("MRRA run checkpoint contains no MRRA parameters")
    if require_mcpr and not mcpr_state_keys:
        raise RuntimeError("MCPR run checkpoint contains no MCPR parameters")
    if require_ttrr and not ttrr_state_keys:
        raise RuntimeError("TTRR run checkpoint contains no trust-gate parameters")
    if require_ard and not ard_state_keys:
        raise RuntimeError("ARD run checkpoint contains no decision-gate parameters")
    return {
        "checkpoint_load_verified": True,
        "checkpoint_keys": sorted(payload.keys()),
        "checkpoint_format": checkpoint_format,
        "checkpoint_source_checkpoint": (
            str(source_checkpoint) if source_checkpoint is not None else None
        ),
        "checkpoint_source_checkpoint_sha256": source_checkpoint_sha256,
        "checkpoint_adapter_model_key_count": adapter_model_key_count,
        "checkpoint_model_key_count": len(model_state),
        "checkpoint_all_model_tensors_finite": True,
        "checkpoint_mipp_state_keys": mipp_state_keys,
        "checkpoint_fusr_state_keys": fusr_state_keys,
        "checkpoint_mrra_state_keys": mrra_state_keys,
        "checkpoint_mcpr_state_keys": mcpr_state_keys,
        "checkpoint_ttrr_state_keys": ttrr_state_keys,
        "checkpoint_ard_state_keys": ard_state_keys,
        "checkpoint_size_bytes": path.stat().st_size,
    }


def verify_mrra_metrics(metrics: dict[str, Any], smoke_test: bool) -> None:
    if metrics.get("mrra_enabled") is not True:
        raise RuntimeError("MRRA config completed without mrra_enabled=true evidence")
    if int(metrics.get("mrra_steps", 0)) < 1:
        raise RuntimeError("MRRA config completed without an optimizer step")
    if int(metrics.get("mrra_missing_elements", 0)) < 1:
        raise RuntimeError("MRRA config observed no missing history elements")
    required = (
        "mrra_reconstruction_loss_mean",
        "mrra_context_residual_energy_mean",
        "mrra_reconstruction_correction_energy_mean",
        "mrra_reconstruction_weight",
    )
    invalid = [
        key for key in required
        if key not in metrics or not math.isfinite(float(metrics[key]))
    ]
    if invalid:
        raise RuntimeError(f"MRRA metrics are missing or non-finite: {invalid}")
    if float(metrics["mrra_reconstruction_loss_mean"]) <= 0.0:
        raise RuntimeError("MRRA reconstruction objective did not activate")
    if float(metrics["mrra_reconstruction_correction_energy_mean"]) <= 0.0:
        raise RuntimeError("MRRA reconstruction correction is identically zero")
    if smoke_test and float(metrics["mrra_context_residual_energy_mean"]) <= 0.0:
        raise RuntimeError("MRRA context residual did not leave zero initialization")
    if metrics.get("mrra_version") == "v3_gradient_isolated_two_timescale":
        v3_checks = {
            "gradient_isolated": metrics.get("mrra_gradient_isolated") is True,
            "normalization_non_affine": (
                metrics.get("mrra_hidden_normalization_affine") is False
            ),
            "two_optimizer_groups": int(
                metrics.get("mrra_optimizer_group_count", 0)
            ) == 2,
            "fixed_lr_multiplier": math.isclose(
                float(metrics.get("mrra_learning_rate_multiplier", -1.0)),
                0.1,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "diagnostics_executed": int(
                metrics.get("mrra_diagnostic_steps", 0)
            ) > 0,
            "finite_positive_gradient_norm": (
                math.isfinite(float(metrics.get("mrra_gradient_l2_max", float("nan"))))
                and float(metrics.get("mrra_gradient_l2_max", 0.0)) > 0.0
            ),
            "finite_parameter_scale": math.isfinite(
                float(metrics.get("mrra_parameter_abs_max", float("nan")))
            ),
        }
        if not all(v3_checks.values()):
            raise RuntimeError(f"MRRA-v3 mechanism verification failed: {v3_checks}")
        initial_lrs = metrics.get("mrra_initial_optimizer_group_lrs", {})
        final_lrs = metrics.get("mrra_final_optimizer_group_lrs", {})
        for label, group_lrs in (("initial", initial_lrs), ("final", final_lrs)):
            backbone_lr = float(group_lrs.get("backbone", float("nan")))
            mrra_lr = float(group_lrs.get("mrra", float("nan")))
            if not (
                math.isfinite(backbone_lr)
                and backbone_lr > 0.0
                and math.isclose(
                    mrra_lr / backbone_lr, 0.1, rel_tol=0.0, abs_tol=1e-10
                )
            ):
                raise RuntimeError(
                    f"MRRA-v3 {label} optimizer learning-rate ratio is invalid"
                )


def verify_mcpr_metrics(
    metrics: dict[str, Any],
    smoke_test: bool,
    official_args: dict[str, Any] | None = None,
) -> None:
    checks = {
        "enabled": metrics.get("mcpr_enabled") is True,
        "steps": int(metrics.get("mcpr_steps", 0)) > 0,
        "backbone_frozen": metrics.get("mcpr_backbone_frozen") is True,
        "backbone_hash_equal": (
            metrics.get("mcpr_backbone_hash_before")
            == metrics.get("mcpr_backbone_hash_after")
        ),
        "memory_frozen": metrics.get("mcpr_memory_frozen") is True,
        "memory_hash_equal": (
            metrics.get("mcpr_memory_hash_before")
            == metrics.get("mcpr_memory_hash_after")
        ),
        "trainable_parameters": int(
            metrics.get("mcpr_trainable_parameter_count", 0)
        ) > 0,
        "finite_positive_residual": (
            math.isfinite(float(metrics.get("mcpr_residual_energy_mean", float("nan"))))
            and float(metrics.get("mcpr_residual_energy_mean", 0.0)) > 0.0
        ),
        "finite_positive_gradient": (
            math.isfinite(float(metrics.get("mcpr_gradient_norm_max", float("nan"))))
            and float(metrics.get("mcpr_gradient_norm_max", 0.0)) > 0.0
        ),
        "source_mipp_frozen": metrics.get("mipp_frozen_source_only") is True,
    }
    objective = str(metrics.get("mcpr_objective", ""))
    if objective == "pareto_common_descent":
        checks.update({
            "pareto_steps": int(metrics.get("mcpr_pareto_diagnostic_steps", 0)) > 0,
            "finite_gradient_cosine": math.isfinite(float(
                metrics.get("mcpr_gradient_cosine_mean", float("nan"))
            )),
            "mse_common_descent": float(
                metrics.get("mcpr_common_mse_dot_min", float("-inf"))
            ) >= -1e-10,
            "mae_common_descent": float(
                metrics.get("mcpr_common_mae_dot_min", float("-inf"))
            ) >= -1e-10,
        })
    elif objective == "tail_trust_mse":
        official_args = official_args or {}
        tail_fraction = float(official_args.get("ttrr_tail_fraction", float("nan")))
        tail_weight = float(official_args.get("ttrr_tail_weight", float("nan")))
        gate_mean = float(metrics.get("ttrr_gate_mean", float("nan")))
        gate_min = float(metrics.get("ttrr_gate_min", float("nan")))
        gate_max = float(metrics.get("ttrr_gate_max", float("nan")))
        raw_energy = float(
            metrics.get("ttrr_raw_residual_energy_mean", float("nan"))
        )
        applied_energy = float(
            metrics.get("ttrr_applied_residual_energy_mean", float("nan"))
        )
        checks.update({
            "ttrr_enabled": metrics.get("ttrr_enabled") is True,
            "ttrr_steps": int(metrics.get("ttrr_diagnostic_steps", 0)) > 0,
            "tail_fraction_matches": math.isclose(
                float(metrics.get("ttrr_tail_fraction", float("nan"))),
                tail_fraction, rel_tol=0.0, abs_tol=1e-12,
            ),
            "tail_weight_matches": math.isclose(
                float(metrics.get("ttrr_tail_weight", float("nan"))),
                tail_weight, rel_tol=0.0, abs_tol=1e-12,
            ),
            "label_free_initial_gate": math.isclose(
                float(metrics.get("ttrr_initial_gate_mean", float("nan"))),
                0.5, rel_tol=0.0, abs_tol=1e-12,
            ),
            "finite_tail_cvar": (
                math.isfinite(float(metrics.get("ttrr_tail_cvar_mean", float("nan"))))
                and float(metrics.get("ttrr_tail_cvar_mean", -1.0)) >= 0.0
            ),
            "finite_positive_harm_fraction": (
                math.isfinite(float(metrics.get(
                    "ttrr_positive_harm_fraction_mean", float("nan")
                )))
                and 0.0 <= float(metrics.get(
                    "ttrr_positive_harm_fraction_mean", -1.0
                )) <= 1.0
            ),
            "gate_inside_open_interval": (
                math.isfinite(gate_mean)
                and math.isfinite(gate_min)
                and math.isfinite(gate_max)
                and 0.0 < gate_min <= gate_mean <= gate_max < 1.0
            ),
            "trust_scaling_does_not_increase_energy": (
                math.isfinite(raw_energy)
                and math.isfinite(applied_energy)
                and raw_energy > 0.0
                and 0.0 < applied_energy <= raw_energy + 1e-12
            ),
        })
    elif objective == "analytic_decision_distill":
        official_args = official_args or {}
        gate_mean = float(metrics.get("ard_gate_mean", float("nan")))
        gate_std = float(metrics.get("ard_gate_std_mean", float("nan")))
        gate_min = float(metrics.get("ard_gate_min", float("nan")))
        gate_max = float(metrics.get("ard_gate_max", float("nan")))
        checks.update({
            "ard_enabled": metrics.get("ard_enabled") is True,
            "ard_distillation_steps": int(
                metrics.get("ard_distillation_steps", 0)
            ) > 0,
            "ard_teacher_samples": int(
                metrics.get("ard_teacher_valid_samples", 0)
            ) > 0,
            "ard_two_teacher_actions": sum(
                int(metrics.get(key, 0)) > 0
                for key in (
                    "ard_teacher_off_count", "ard_teacher_partial_count",
                    "ard_teacher_full_count",
                )
            ) >= 2,
            "ard_frozen_hidden": int(
                metrics.get("ard_hidden_dim", -1)
            ) == int(official_args.get("ard_hidden_dim", -2)) == 16,
            "ard_frozen_warmup": int(
                metrics.get("ard_distill_warmup_epochs", -1)
            ) == int(official_args.get("ard_distill_warmup_epochs", -2)) == 1,
            "ard_initial_gate": math.isclose(
                float(metrics.get("ard_initial_gate_mean", float("nan"))),
                0.5, rel_tol=0.0, abs_tol=1e-12,
            ),
            "ard_gate_finite_bounded": (
                all(math.isfinite(value) for value in (
                    gate_mean, gate_std, gate_min, gate_max
                ))
                and 0.0 <= gate_min <= gate_mean <= gate_max <= 1.0
                and gate_std > 0.0
            ),
            "ard_losses_finite": all(math.isfinite(float(
                metrics.get(key, float("nan"))
            )) for key in (
                "ard_action_loss_mean", "ard_scale_loss_mean",
                "ard_teacher_alpha_mean",
            )),
        })
    elif objective in {
        "benefit_filtered_multiview_distill",
        "token_benefit_filtered_multiview_distill",
        "dual_scale_residual_basis_distill",
    }:
        official_args = official_args or {}
        beneficial_fraction = float(
            metrics.get("bf_cmvd_beneficial_fraction", float("nan"))
        )
        checks.update({
            "bf_cmvd_enabled": metrics.get("bf_cmvd_enabled") is True,
            "bf_cmvd_filter_granularity": metrics.get(
                "bf_cmvd_filter_granularity"
            ) == (
                "token"
                if objective in {
                    "token_benefit_filtered_multiview_distill",
                    "dual_scale_residual_basis_distill",
                }
                else "sample"
            ),
            "bf_cmvd_steps": int(
                metrics.get("bf_cmvd_diagnostic_steps", 0)
            ) > 0,
            "bf_cmvd_beneficial_samples": int(
                metrics.get("bf_cmvd_beneficial_samples", 0)
            ) > 0,
            "bf_cmvd_non_beneficial_samples": int(
                metrics.get("bf_cmvd_total_samples", 0)
            ) > int(metrics.get("bf_cmvd_beneficial_samples", 0)),
            "bf_cmvd_fraction_open_interval": (
                math.isfinite(beneficial_fraction)
                and 0.0 < beneficial_fraction < 1.0
            ),
            "bf_cmvd_target_rate": math.isclose(
                float(metrics.get("bf_cmvd_target_rate", float("nan"))),
                float(official_args.get("bf_cmvd_target_rate", float("nan"))),
                rel_tol=0.0, abs_tol=1e-12,
            ),
            "bf_cmvd_block_lengths": metrics.get("bf_cmvd_block_lengths") == [5, 20],
            "bf_cmvd_nested_seed": int(
                metrics.get("bf_cmvd_nested_seed", -1)
            ) == int(official_args.get("bf_cmvd_nested_seed", -2)),
            "bf_cmvd_teacher_weights": metrics.get("bf_cmvd_teacher_weights") == {
                "original": 0.5, "l5": 0.25, "l20": 0.25,
            },
            "bf_cmvd_distill_positive": (
                math.isfinite(float(
                    metrics.get("bf_cmvd_distill_l1_mean", float("nan"))
                ))
                and float(metrics.get("bf_cmvd_distill_l1_mean", 0.0)) > 0.0
            ),
            "bf_cmvd_teacher_delta_positive": (
                math.isfinite(float(
                    metrics.get("bf_cmvd_teacher_delta_energy_mean", float("nan"))
                ))
                and float(
                    metrics.get("bf_cmvd_teacher_delta_energy_mean", 0.0)
                ) > 0.0
            ),
        })
        if objective == "dual_scale_residual_basis_distill":
            official_args = official_args or {}
            l5_enabled = bool(official_args.get("dmrb_l5_enabled", False))
            l20_enabled = bool(official_args.get("dmrb_l20_enabled", False))
            expected_parameters = 45182 * (int(l5_enabled) + int(l20_enabled))
            checks.update({
                "dmrb_enabled": metrics.get("dmrb_enabled") is True,
                "dmrb_l5_switch": (
                    metrics.get("dmrb_l5_enabled") is l5_enabled
                ),
                "dmrb_l20_switch": (
                    metrics.get("dmrb_l20_enabled") is l20_enabled
                ),
                "dmrb_exact_parameter_count": int(
                    metrics.get("mcpr_trainable_parameter_count", -1)
                ) == expected_parameters,
                "dmrb_single_backbone_forward": int(
                    metrics.get("dmrb_inference_backbone_forward_count", -1)
                ) == 1,
                "dmrb_fixed_sum_fusion": (
                    metrics.get("dmrb_fixed_residual_fusion") == "sum"
                ),
                "dmrb_teacher_reconstruction": (
                    math.isfinite(float(metrics.get(
                        "dmrb_teacher_reconstruction_max_abs", float("nan")
                    )))
                    and float(metrics.get(
                        "dmrb_teacher_reconstruction_max_abs", float("inf")
                    )) <= 1e-6
                ),
                "dmrb_l5_mechanism": (
                    (not l5_enabled)
                    or all(
                        math.isfinite(float(metrics.get(key, float("nan"))))
                        and float(metrics.get(key, 0.0)) > 0.0
                        for key in (
                            "dmrb_l5_distill_l1_mean",
                            "dmrb_l5_residual_energy_mean",
                            "dmrb_l5_target_energy_mean",
                            "dmrb_l5_gradient_norm_max",
                        )
                    )
                ),
                "dmrb_l20_mechanism": (
                    (not l20_enabled)
                    or all(
                        math.isfinite(float(metrics.get(key, float("nan"))))
                        and float(metrics.get(key, 0.0)) > 0.0
                        for key in (
                            "dmrb_l20_distill_l1_mean",
                            "dmrb_l20_residual_energy_mean",
                            "dmrb_l20_target_energy_mean",
                            "dmrb_l20_gradient_norm_max",
                        )
                    )
                ),
            })
    elif objective != "mse_only":
        checks["known_objective"] = False
    if not all(checks.values()):
        raise RuntimeError(f"MCPR mechanism verification failed: {checks}")


def classify_run_failure(log_text: str) -> str:
    lowered = log_text.lower()
    if "out of memory" in lowered or "cuda error: out of memory" in lowered:
        return "OOM"
    if "nan_or_divergence" in lowered or (
        "non-finite" in lowered and ("loss" in lowered or "objective" in lowered)
    ):
        return "NAN_OR_DIVERGENCE"
    return "CODE_ERROR"


def verify_radr_metrics(metrics: dict[str, Any], smoke_test: bool) -> None:
    if metrics.get("radr_enabled") is not True:
        raise RuntimeError("RADR config completed without radr_enabled=true evidence")
    if int(metrics.get("radr_steps", 0)) < 1:
        raise RuntimeError("RADR config completed without an optimizer step")
    if int(metrics.get("radr_samples", 0)) < 1:
        raise RuntimeError("RADR config observed no valid forecast samples")
    required = (
        "radr_total_loss_mean",
        "radr_tail_mse_mean",
        "radr_body_mse_mean",
        "radr_body_mae_mean",
        "radr_normalized_tail_risk_mean",
        "radr_normalized_body_risk_mean",
        "radr_difficulty_rank_std_mean",
        "radr_hard_body_mse_gap_mean",
    )
    invalid = [
        key for key in required
        if key not in metrics or not math.isfinite(float(metrics[key]))
    ]
    if invalid:
        raise RuntimeError(f"RADR metrics are missing or non-finite: {invalid}")
    if float(metrics["radr_difficulty_rank_std_mean"]) <= 0.0:
        raise RuntimeError("RADR difficulty ranking is identically constant")
    if float(metrics["radr_hard_body_mse_gap_mean"]) < -1e-8:
        raise RuntimeError("RADR tail weighting does not emphasize harder samples")
    if smoke_test and int(metrics.get("optimizer_steps", 0)) < 1:
        raise RuntimeError("RADR smoke completed without an optimizer step")


def verify_mipp_metrics(
    metrics: dict[str, Any],
    smoke_test: bool,
    official_args: dict[str, Any] | None = None,
) -> None:
    if metrics.get("mipp_enabled") is not True:
        raise RuntimeError("MIPP config completed without mipp_enabled=true evidence")
    if int(metrics.get("mipp_steps", 0)) < 1:
        raise RuntimeError("MIPP config completed without a paired-objective optimizer step")
    if int(metrics.get("mipp_pair_views_generated", 0)) < 1:
        raise RuntimeError("MIPP config generated no historical mask-intervention pair")
    required = (
        "mipp_total_loss_mean",
        "mipp_invariance_loss_mean",
        "mipp_variance_loss_mean",
        "mipp_covariance_loss_mean",
        "mipp_decorrelation_loss_mean",
        "mipp_mask_reconstruction_loss_mean",
        "mipp_paired_content_cosine_mean",
        "mipp_teacher_pair_cosine_mean",
        "mipp_stable_write_acceptance_rate_mean",
        "mipp_pair_missing_rate_mean",
    )
    invalid = []
    for key in required:
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            invalid.append(key)
    if invalid:
        raise RuntimeError(f"MIPP metrics are missing or non-finite: {invalid}")
    acceptance = float(metrics["mipp_stable_write_acceptance_rate_mean"])
    if not 0.0 <= acceptance <= 1.0:
        raise RuntimeError(f"MIPP write acceptance is outside [0, 1]: {acceptance}")
    if smoke_test and int(metrics.get("memory_bank_warmup_count", 0)) < 1:
        raise RuntimeError("MIPP smoke did not warm its purified prototype bank")

    official_args = official_args or {}
    expected_mode = str(official_args.get("mipp_write_gate_mode", "absolute"))
    reported_mode = str(metrics.get("mipp_write_gate_mode", "absolute"))
    if reported_mode != expected_mode:
        raise RuntimeError(
            f"MIPP write gate mode mismatch: expected={expected_mode} observed={reported_mode}"
        )
    if expected_mode not in {
        "budget_topk", "stability_diverse_rank", "mask_regime_balanced_rank"
    }:
        return

    expected_budget = int(official_args["mipp_write_budget"])
    if int(metrics.get("mipp_write_budget", -1)) != expected_budget:
        raise RuntimeError("MIPP write-budget evidence does not match the resolved config")
    budgeted_required = (
        "mipp_write_stability_mean",
        "mipp_write_stability_std_mean",
        "mipp_write_effective_cutoff_mean",
        "mipp_write_raw_absolute_acceptance_rate_mean",
        "mipp_write_eligible_count_mean",
        "mipp_write_sampled_count_mean",
        "mipp_write_updated_count_mean",
    )
    invalid_budgeted = []
    for key in budgeted_required:
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            invalid_budgeted.append(key)
    if invalid_budgeted:
        raise RuntimeError(
            f"budgeted MIPP metrics are missing or non-finite: {invalid_budgeted}"
        )
    eligible = float(metrics["mipp_write_eligible_count_mean"])
    sampled = float(metrics["mipp_write_sampled_count_mean"])
    if not 0.0 < eligible <= expected_budget:
        raise RuntimeError(
            f"MIPP eligible count violates budget: eligible={eligible}, budget={expected_budget}"
        )
    if not 0.0 < sampled <= expected_budget:
        raise RuntimeError(
            f"MIPP sampled count violates budget: sampled={sampled}, budget={expected_budget}"
        )
    if expected_mode == "stability_diverse_rank":
        diversity_required = (
            "mipp_write_candidate_pairwise_cosine_mean",
            "mipp_write_selected_pairwise_cosine_mean",
            "mipp_write_selected_stability_mean",
            "mipp_write_diversity_gain_mean",
        )
        invalid_diversity = []
        for key in diversity_required:
            value = metrics.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                invalid_diversity.append(key)
        if invalid_diversity:
            raise RuntimeError(
                "stability-diversity metrics are missing or non-finite: "
                f"{invalid_diversity}"
            )
    if expected_mode == "mask_regime_balanced_rank":
        mask_coverage_required = (
            "mipp_write_mask_candidate_pairwise_distance_mean",
            "mipp_write_mask_selected_pairwise_distance_mean",
            "mipp_write_mask_coverage_distance_mean",
            "mipp_write_mask_topk_coverage_distance_mean",
            "mipp_write_mask_coverage_gain_vs_topk_mean",
            "mipp_write_mask_candidate_missing_rate_mean",
            "mipp_write_mask_selected_missing_rate_mean",
            "mipp_write_selected_stability_mean",
        )
        invalid_mask_coverage = []
        for key in mask_coverage_required:
            value = metrics.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                invalid_mask_coverage.append(key)
        if invalid_mask_coverage:
            raise RuntimeError(
                "mask-regime coverage metrics are missing or non-finite: "
                f"{invalid_mask_coverage}"
            )


def verify_fusr_metrics(
    metrics: dict[str, Any],
    smoke_test: bool,
    objective_version: str = "v1_joint",
) -> None:
    if metrics.get("fusr_enabled") is not True:
        raise RuntimeError("FUSR config completed without fusr_enabled=true evidence")
    reported_version = str(metrics.get("fusr_objective_version", objective_version))
    if reported_version != objective_version:
        raise RuntimeError(
            "FUSR objective-version evidence mismatch: "
            f"expected={objective_version} observed={reported_version}"
        )
    router_steps = int(metrics.get("fusr_router_gradient_steps", 0))
    if max(int(metrics.get("fusr_steps", 0)), router_steps) < 1:
        raise RuntimeError("FUSR config completed without a utility-supervised optimizer step")
    cache_samples = int(metrics.get("fusr_cache_samples", 0))
    if max(int(metrics.get("fusr_utility_pairs_generated", 0)), cache_samples) < 1:
        raise RuntimeError("FUSR config generated no paired prototype-on/off utility target")

    required_finite = (
        "fusr_total_loss_mean",
        "fusr_gate_mean",
        "fusr_abstention_rate",
    )
    optional_finite = (
        "fusr_utility_loss_mean",
        "fusr_utility_nll_mean",
        "fusr_sign_loss_mean",
        "fusr_predicted_utility_mean",
        "fusr_utility_std_mean",
        "fusr_spectral_uncertainty_mean",
        "fusr_sign_accuracy",
    )
    invalid = []
    for key in required_finite:
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            invalid.append(key)
    for key in optional_finite:
        value = metrics.get(key)
        if value is not None and (
            not isinstance(value, (int, float)) or not math.isfinite(float(value))
        ):
            invalid.append(key)
    if invalid:
        raise RuntimeError(f"FUSR metrics are missing or non-finite: {invalid}")

    for key in ("fusr_gate_mean", "fusr_abstention_rate"):
        value = float(metrics[key])
        if not 0.0 <= value <= 1.0:
            raise RuntimeError(f"{key} is outside [0, 1]: {value}")
    if "fusr_sign_accuracy" in metrics:
        value = float(metrics["fusr_sign_accuracy"])
        if not 0.0 <= value <= 1.0:
            raise RuntimeError(f"fusr_sign_accuracy is outside [0, 1]: {value}")
    if max(int(metrics.get("fusr_valid_utility_samples", 0)), cache_samples) < 1:
        raise RuntimeError("FUSR produced no valid utility-supervision samples")
    if smoke_test and max(int(metrics.get("optimizer_steps", 0)), router_steps) < 1:
        raise RuntimeError("FUSR smoke completed without an optimizer step")

    if objective_version != "v2_frozen_balanced_rank":
        return

    if metrics.get("fusr_calibration_split") != "validation":
        raise RuntimeError("FUSR V2 calibration_split must be validation")
    if metrics.get("fusr_used_test_targets") is not False:
        raise RuntimeError("FUSR V2 must record used_test_targets=false")
    threshold = metrics.get("fusr_calibrated_threshold")
    if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
        raise RuntimeError("FUSR V2 calibrated threshold is missing or non-finite")
    if cache_samples < 1:
        raise RuntimeError("FUSR V2 frozen cache contains no samples")
    if int(metrics.get("fusr_cache_train_samples", 0)) < 1:
        raise RuntimeError("FUSR V2 frozen cache contains no chronological fit samples")
    if int(metrics.get("fusr_cache_validation_samples", 0)) < 1:
        raise RuntimeError("FUSR V2 frozen cache contains no validation-calibration samples")
    if int(metrics.get("fusr_cache_positive_samples", 0)) < 1:
        raise RuntimeError("FUSR V2 frozen cache contains no positive-utility samples")
    if int(metrics.get("fusr_cache_negative_samples", 0)) < 1:
        raise RuntimeError("FUSR V2 frozen cache contains no negative-utility samples")
    if int(metrics.get("fusr_rank_pairs_generated", 0)) < 1:
        raise RuntimeError("FUSR V2 generated no beneficial-versus-harmful rank pair")
    if router_steps < 1:
        raise RuntimeError("FUSR V2 completed without a router gradient step")
    if metrics.get("fusr_backbone_frozen") is not True:
        raise RuntimeError("FUSR V2 did not prove that the backbone remained frozen")
    if metrics.get("fusr_memory_unchanged") is not True:
        raise RuntimeError("FUSR V2 did not prove that prototype memory remained unchanged")
    if metrics.get("fusr_calibration_state") != "CALIBRATION_LOCKED":
        raise RuntimeError("FUSR V2 calibration did not reach CALIBRATION_LOCKED")


def verify_cfr_metrics(metrics: dict[str, Any], smoke_test: bool) -> None:
    if metrics.get("cfr_enabled") is not True:
        raise RuntimeError("CFR config completed without cfr_enabled=true evidence")
    if int(metrics.get("cfr_steps", 0)) < 1:
        raise RuntimeError("CFR config completed without a regret-supervised optimizer step")
    if int(metrics.get("cfr_samples", 0)) < 1:
        raise RuntimeError("CFR config produced no valid paired samples")
    if metrics.get("cfr_reference_detached") is not True:
        raise RuntimeError("CFR no-prototype reference was not recorded as detached")
    if metrics.get("cfr_inference_path") != "prototype_on_single_path":
        raise RuntimeError("CFR inference path must remain the single prototype-on path")
    required = (
        "cfr_total_loss_mean",
        "cfr_prototype_on_mse_mean",
        "cfr_prototype_off_mse_mean",
        "cfr_signed_regret_mean",
        "cfr_positive_regret_mean",
        "cfr_harmful_retrieval_rate_mean",
    )
    invalid = []
    for key in required:
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            invalid.append(key)
    if invalid:
        raise RuntimeError(f"CFR metrics are missing or non-finite: {invalid}")
    harmful_rate = float(metrics["cfr_harmful_retrieval_rate_mean"])
    if not 0.0 <= harmful_rate <= 1.0:
        raise RuntimeError(f"CFR harmful retrieval rate is outside [0, 1]: {harmful_rate}")
    if float(metrics["cfr_positive_regret_mean"]) < 0.0:
        raise RuntimeError("CFR positive regret cannot be negative")
    if not math.isclose(
        float(metrics["cfr_total_loss_mean"]),
        float(metrics["cfr_positive_regret_mean"]),
        rel_tol=1e-7,
        abs_tol=1e-9,
    ):
        raise RuntimeError("CFR total loss must equal the frozen positive-regret term")
    if smoke_test and int(metrics.get("optimizer_steps", 0)) < 1:
        raise RuntimeError("CFR smoke completed without an optimizer step")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-uuid", default=None)
    parser.add_argument("--scheduler-attempt", type=int, default=0)
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_experiment_config(config)
    if args.gpu_uuid:
        config["runtime"]["gpu_uuid"] = args.gpu_uuid
    if config["runtime"].get("device") == "cuda":
        config["runtime"].update(resolve_gpu_metadata(str(config["runtime"]["gpu_uuid"])))
    config["runtime"]["scheduler_attempt"] = args.scheduler_attempt
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = experiment_id(config, timestamp)
    run_dir = PROJECT_ROOT / "runs" / exp_id
    if run_dir.exists():
        raise FileExistsError(f"experiment ID already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    resolved = copy.deepcopy(config)
    resolved["experiment"]["id"] = exp_id
    resolved["experiment"]["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    resolved["paths"] = {
        "run_dir": str(run_dir),
        "config_source": str(config_path),
        "baseline_repo": str(BASELINE_ROOT),
        "upstream_repo": str(PROJECT_ROOT / "upstream" / "S4M"),
    }

    missing_csv = (PROJECT_ROOT / config["data"]["missing_csv"]).resolve()
    complete_csv = (PROJECT_ROOT / config["data"]["complete_csv"]).resolve()
    if not missing_csv.is_file() or not complete_csv.is_file():
        raise FileNotFoundError(f"data asset missing: missing={missing_csv} complete={complete_csv}")

    official_args = copy.deepcopy(config["official_args"])
    official_args.update(
        {
            "root_path": str(missing_csv.parent),
            "data_path": missing_csv.name,
            "true_data_path": str(complete_csv),
            "checkpoints": str(run_dir / "checkpoints"),
            "results_root": str(run_dir / "results"),
            "test_results_root": str(run_dir / "test_results"),
            "metrics_path": str(run_dir / "metrics.json"),
        }
    )
    resolved["official_args"] = official_args
    atomic_text(run_dir / "resolved_config.yaml", yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True))
    atomic_text(run_dir / "env.txt", safe_environment_text(config))
    atomic_text(
        run_dir / "git_state.txt",
        "[baseline_adapter]\n" + git_evidence(BASELINE_ROOT) + "\n[upstream_snapshot]\n" + git_evidence(PROJECT_ROOT / "upstream" / "S4M"),
    )
    _, code_diff = run_capture(["git", "diff", "--binary"], BASELINE_ROOT)
    atomic_text(run_dir / "code.diff", code_diff)
    source_snapshot_root = run_dir / "source_snapshot"
    critical_sources = {
        "baseline/s4m/model/MCPR.py": S4M_ROOT / "model/MCPR.py",
        "baseline/s4m/model/S4M.py": S4M_ROOT / "model/S4M.py",
        "baseline/s4m/experiments/exp_pretrain1.py": (
            S4M_ROOT / "experiments/exp_pretrain1.py"
        ),
        "baseline/s4m/run.py": S4M_ROOT / "run.py",
        "baseline/s4m/utils/tools.py": S4M_ROOT / "utils/tools.py",
        "project/scripts/run_s4m_experiment.py": Path(__file__).resolve(),
        "project/config_source.yaml": config_path,
    }
    source_snapshot_manifest = {}
    for snapshot_name, source_path in critical_sources.items():
        if not source_path.is_file():
            raise FileNotFoundError(f"critical source snapshot missing: {source_path}")
        destination = source_snapshot_root / snapshot_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        source_snapshot_manifest[snapshot_name] = {
            "source": str(source_path),
            "sha256": sha256_file(source_path),
            "size_bytes": source_path.stat().st_size,
        }
    atomic_json(
        run_dir / "source_snapshot_manifest.json", source_snapshot_manifest
    )

    command = [sys.executable, str(S4M_ROOT / "run.py"), *cli_from_mapping(official_args)]
    s4_source_root = S4M_ROOT / "model" / "S4" / "s4"
    structured_kernels_root = STRUCTURED_KERNEL_ROOT
    python_path = os.pathsep.join(
        [
            str(S4M_ROOT),
            str(S4M_ROOT / "model"),
            str(s4_source_root),
            str(structured_kernels_root),
        ]
    )
    gpu_export = ""
    if config["runtime"].get("gpu_uuid"):
        gpu_export = f"export CUDA_VISIBLE_DEVICES={shlex.quote(str(config['runtime']['gpu_uuid']))}\n"
    command_text = (
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {shlex.quote(str(S4M_ROOT))}\n"
        f"export PYTHONPATH={shlex.quote(python_path)}\n"
        + gpu_export
        + "export PYTHONDONTWRITEBYTECODE=1\n"
        + "export WANDB_MODE=disabled\n"
        + shlex.join(command)
        + "\n"
    )
    atomic_text(run_dir / "command.sh", command_text)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = python_path
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["WANDB_MODE"] = "disabled"
    environment["MPLCONFIGDIR"] = str(run_dir / ".matplotlib")
    if config["runtime"].get("gpu_uuid"):
        environment["CUDA_VISIBLE_DEVICES"] = str(config["runtime"]["gpu_uuid"])

    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    start = time.monotonic()
    status = {
        "experiment_id": exp_id,
        "state": "RUNNING",
        "started_at": started_at,
        "config": str(config_path),
        "device": config["runtime"]["device"],
    }
    atomic_json(run_dir / "status.json", status)
    print(f"run_dir={run_dir}", flush=True)

    with (run_dir / "train.log").open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(command, cwd=S4M_ROOT, env=environment, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        status["pid"] = process.pid
        atomic_json(run_dir / "status.json", status)
        return_code = process.wait()

    elapsed = time.monotonic() - start
    resource_usage = {
        "elapsed_seconds": elapsed,
        "device": config["runtime"]["device"],
        "peak_gpu_memory_mb": None,
        "return_code": return_code,
    }

    metrics_path = run_dir / "metrics.json"
    verification: dict[str, Any] = {}
    verification_error: str | None = None
    metrics: dict[str, Any] = {}
    if return_code == 0 and metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            checkpoint = Path(metrics["checkpoint"])
            if not checkpoint.is_file():
                raise FileNotFoundError(f"reported checkpoint does not exist: {checkpoint}")
            mipp_enabled = bool(official_args.get("use_mipp", False))
            fusr_enabled = bool(official_args.get("use_fusr", False))
            cfr_enabled = bool(official_args.get("use_cfr", False))
            mrra_enabled = bool(official_args.get("use_mrra", False))
            radr_enabled = bool(official_args.get("use_radr", False))
            mcpr_enabled = bool(official_args.get("use_mcpr", False))
            ttrr_enabled = bool(official_args.get("use_ttrr", False))
            ard_enabled = bool(official_args.get("use_ard", False))
            dmrb_enabled = bool(official_args.get("use_dmrb", False))
            verification = verify_checkpoint(
                checkpoint,
                require_mipp=mipp_enabled,
                require_fusr=fusr_enabled,
                require_mrra=mrra_enabled,
                require_mcpr=mcpr_enabled,
                require_ttrr=ttrr_enabled,
                require_ard=ard_enabled,
            )
            if dmrb_enabled:
                checkpoint_mcpr_keys = verification["checkpoint_mcpr_state_keys"]
                if bool(official_args.get("dmrb_l5_enabled", False)) and not any(
                    "mcpr.l5_basis." in key for key in checkpoint_mcpr_keys
                ):
                    raise RuntimeError("DMRB checkpoint contains no L5 basis parameters")
                if bool(official_args.get("dmrb_l20_enabled", False)) and not any(
                    "mcpr.l20_basis." in key for key in checkpoint_mcpr_keys
                ):
                    raise RuntimeError("DMRB checkpoint contains no L20 basis parameters")
                verification["checkpoint_dmrb_branches_verified"] = True
            effective_optimizer_steps = max(
                int(metrics.get("optimizer_steps", 0)),
                int(metrics.get("fusr_router_gradient_steps", 0)),
            )
            if str(official_args.get("post_train_evaluation", "test")) == "validation":
                if metrics.get("evaluation_split") != "validation":
                    raise RuntimeError(
                        "validation-only training did not report evaluation_split=validation"
                    )
                if metrics.get("post_train_evaluation") != "validation":
                    raise RuntimeError(
                        "validation-only training did not preserve its evaluation identity"
                    )
                if metrics.get("test_split_accessed") is not False:
                    raise RuntimeError(
                        "validation-only training did not prove test_split_accessed=false"
                    )
                if metrics.get("chronological_loader") is not True:
                    raise RuntimeError(
                        "validation-only training requires a chronological loader"
                    )
            if config["experiment"].get("smoke_test") and effective_optimizer_steps < 1:
                raise RuntimeError("smoke test completed without an optimizer step")
            if mipp_enabled and not mcpr_enabled:
                verify_mipp_metrics(
                    metrics,
                    bool(config["experiment"].get("smoke_test")),
                    official_args,
                )
            if fusr_enabled:
                verify_fusr_metrics(
                    metrics,
                    bool(config["experiment"].get("smoke_test")),
                    str(official_args.get("fusr_objective_version", "v1_joint")),
                )
            if cfr_enabled:
                verify_cfr_metrics(
                    metrics,
                    bool(config["experiment"].get("smoke_test")),
                )
            if mrra_enabled:
                verify_mrra_metrics(
                    metrics,
                    bool(config["experiment"].get("smoke_test")),
                )
            if radr_enabled:
                verify_radr_metrics(
                    metrics,
                    bool(config["experiment"].get("smoke_test")),
                )
            if mcpr_enabled:
                verify_mcpr_metrics(
                    metrics,
                    bool(config["experiment"].get("smoke_test")),
                    official_args,
                )
            required_warmups = int(config["experiment"].get("required_memory_bank_warmups", 0))
            if int(metrics.get("memory_bank_warmup_count", 0)) < required_warmups:
                raise RuntimeError(
                    "memory-bank lifecycle regression: "
                    f"required {required_warmups} warmups, observed "
                    f"{metrics.get('memory_bank_warmup_count', 0)}"
                )
            metrics.update(verification)
            metrics.update(
                {
                    "experiment_id": exp_id,
                    "scope": "smoke_test_not_formal_result" if config["experiment"].get("smoke_test") else "formal",
                    "nominal_missing_rate": config["experiment"]["nominal_missing_rate"],
                    "realized_missing_rate": config["experiment"]["realized_missing_rate"],
                    "seed": config["experiment"]["seed"],
                    "protocol_match": config["experiment"].get("protocol_match", "unknown"),
                    "mask_asset": config["data"].get("mask_asset", "unknown"),
                }
            )
            atomic_json(metrics_path, metrics)
        except Exception as exc:
            verification_error = f"{type(exc).__name__}: {exc}"

    resource_usage["peak_gpu_memory_mb"] = metrics.get("peak_gpu_memory_mb")
    resource_usage["peak_gpu_reserved_mb"] = metrics.get("peak_gpu_reserved_mb")
    resource_usage["gpu_uuid"] = config["runtime"].get("gpu_uuid")
    atomic_json(run_dir / "resource_usage.json", resource_usage)

    success = return_code == 0 and metrics_path.is_file() and verification.get("checkpoint_load_verified", False) and verification_error is None
    failure_state = "CODE_ERROR"
    if return_code != 0:
        log_text = (run_dir / "train.log").read_text(encoding="utf-8", errors="replace").lower()
        failure_state = classify_run_failure(log_text)
    status.update(
        {
            "state": "SUCCESS" if success else failure_state,
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "return_code": return_code,
            "metrics_path": str(metrics_path) if metrics_path.is_file() else None,
            "verification_error": verification_error,
            **verification,
        }
    )
    atomic_json(run_dir / "status.json", status)
    print(json.dumps(status, indent=2, sort_keys=True))
    print(f"run_dir={run_dir}")
    return 0 if success else (return_code or 2)


if __name__ == "__main__":
    raise SystemExit(main())
