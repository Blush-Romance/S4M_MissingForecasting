"""Read-only, CPU-only audit of trusted local S4M checkpoint pairs.

This module does not import the baseline, instantiate a model, perform inference,
or create output files. Checkpoints contain pickle data: only pass trusted local
project artifacts. Geometry is descriptive, not a diagnosis of bank collapse.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import torch


_EPOCH = re.compile(
    r"Epoch:\s*(\d+),\s*Steps:\s*(\d+)\s*\|\s*"
    r"Train Loss:\s*([-+\deE.]+)\s+Vali Loss:\s*([-+\deE.]+)"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _distribution(values):
    values = [int(value) for value in values]
    return {
        "values": values,
        "sum": sum(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "histogram": {str(key): value for key, value in sorted(Counter(values).items())},
    }


def audit_training_log(log_path, metrics):
    """Return printed curves and only log-corroborated selected best epochs."""
    path = Path(log_path)
    epochs, save_epochs = [], []
    most_recent = None
    text = path.read_text(errors="replace") if path.is_file() else ""
    for line in text.splitlines():
        match = _EPOCH.search(line)
        if match:
            epoch, steps, train_loss, validation_loss = match.groups()
            most_recent = int(epoch)
            epochs.append({
                "epoch": most_recent,
                "steps": int(steps),
                "train_loss_printed": _finite_number(train_loss),
                "observed_validation_loss_printed": _finite_number(validation_loss),
            })
        if "Validation loss decreased" in line and "Saving model" in line:
            if most_recent is not None:
                save_epochs.append(most_recent)
    finite_epochs = [item for item in epochs if item["observed_validation_loss_printed"] is not None]
    duplicate_epochs = len({item["epoch"] for item in epochs}) != len(epochs)
    best_printed = min(finite_epochs, key=lambda item: item["observed_validation_loss_printed"]) if finite_epochs else None
    persisted = metrics.get("best_observed_validation_loss")
    persisted_finite = _finite_number(persisted) if persisted is not None else None
    matches_metric = bool(
        best_printed is not None
        and persisted_finite is not None
        and abs(best_printed["observed_validation_loss_printed"] - persisted_finite) <= 5e-7
    )
    corroborated = bool(
        best_printed is not None and matches_metric and not duplicate_epochs
        and save_epochs and save_epochs[-1] == best_printed["epoch"]
    )
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path) if path.is_file() else None,
        "epochs": epochs,
        "logged_save_epochs": save_epochs,
        "best_observed_validation_loss": persisted_finite,
        "best_printed_validation_loss": best_printed["observed_validation_loss_printed"] if best_printed else None,
        "printed_best_matches_metrics_with_rounding": matches_metric,
        "duplicate_epoch_numbers": duplicate_epochs,
        "best_epoch": best_printed["epoch"] if corroborated else None,
        "best_epoch_source": "printed_epoch_curve_and_last_save_event_and_metrics" if corroborated else "unknown_not_corroborated",
        "checkpoint_epoch_metadata_present": False,
        "selection_target": "mean_across_batches_of_MSE_on_seq_y_mask_equal_1_observed_forecast_positions",
        "not_complete_target_mse": True,
        "not_global_element_weighted_mse": True,
        "limitations": "The checkpoint has no epoch metadata; best_epoch is corroborated by the log, not encoded in the checkpoint.",
    }


def _geometry(representatives):
    x = representatives.detach().to(device="cpu", dtype=torch.float64)
    if x.ndim != 2:
        raise ValueError("repre must have shape [centers, feature_dimension]")
    n, dimension = x.shape
    finite = bool(torch.isfinite(x).all().item())
    result = {
        "finite": finite,
        "centered_rank_cap": min(max(n - 1, 0), dimension),
        "effective_rank_definition": "exp(entropy(normalized_squared_singular_values_of_centered_repre))",
        "interpretation": "Rank is limited by min(n_centers-1, dimension). Do not label cross-count differences as collapse or infer causality.",
    }
    if not finite or n == 0:
        return result
    norms = torch.linalg.vector_norm(x, dim=1)
    result["center_norm_min"] = float(norms.min())
    result["center_norm_mean"] = float(norms.mean())
    result["center_norm_max"] = float(norms.max())
    valid = norms > 0
    result["zero_norm_centers"] = int((~valid).sum())
    normalized = x[valid] / norms[valid, None]
    if normalized.shape[0] >= 2:
        cosine = normalized @ normalized.T
        mask = ~torch.eye(cosine.shape[0], dtype=torch.bool)
        values = cosine[mask]
        result["pairwise_cosine"] = {
            "ordered_nonself_pairs": int(values.numel()),
            "mean": float(values.mean()),
            "std_population": float(values.std(unbiased=False)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    else:
        result["pairwise_cosine"] = None
    singular = torch.linalg.svdvals(x - x.mean(dim=0, keepdim=True))
    energies = singular.square()
    total = float(energies.sum())
    tolerance = max(n, dimension) * torch.finfo(torch.float64).eps * (float(singular.max()) if singular.numel() else 0.0)
    numerical_rank = int((singular > tolerance).sum()) if singular.numel() else 0
    if total > 0:
        probabilities = energies[energies > 0] / total
        effective = float(torch.exp(-(probabilities * probabilities.log()).sum()))
    else:
        effective = 0.0
    result.update({
        "centered_numerical_rank": numerical_rank,
        "centered_effective_rank": effective,
        "effective_rank_fraction_of_count_cap": effective / result["centered_rank_cap"] if result["centered_rank_cap"] else None,
        "centered_total_energy": total,
    })
    return result


def _bank_summary(checkpoint, top_m, top_k):
    memory, repre, num_dic = checkpoint["memory"], checkpoint["repre"], checkpoint["num_dic"]
    if not isinstance(memory, (list, tuple)) or not torch.is_tensor(repre):
        raise TypeError("Expected list/tuple memory and tensor repre")
    n = len(memory)
    if repre.ndim != 2 or repre.shape[0] != n:
        raise ValueError("memory center count and repre shape do not match")
    if len(num_dic) < n:
        raise ValueError("num_dic has fewer entries than bank centers")
    if not all(torch.is_tensor(item) and item.ndim == 2 and item.shape[1] == repre.shape[1] for item in memory):
        raise ValueError("Each memory entry must be a resident [samples, feature_dimension] tensor")
    resident = [item.shape[0] for item in memory]
    tracked = [int(value) for value in num_dic[:n]]
    memory_finite = all(bool(torch.isfinite(item).all()) for item in memory)
    max_discrepancy = None
    if n and all(resident) and memory_finite and bool(torch.isfinite(repre).all()):
        centroids = torch.stack([item.double().mean(0) for item in memory])
        max_discrepancy = float((centroids - repre.double()).abs().max())
    candidates = min(n, top_m)
    return {
        "center_count": n,
        "repre_shape": list(repre.shape),
        "memory_shapes": [list(item.shape) for item in memory],
        "resident_sample_counts": _distribution(resident),
        "num_dic_active_tracked_counts": _distribution(tracked),
        "num_dic_total_slots": len(num_dic),
        "num_dic_inactive_values": [int(value) for value in num_dic[n:]],
        "resident_vs_num_dic_mismatching_centers": sum(a != b for a, b in zip(resident, tracked)),
        "occupancy_caveat": "Resident rows measure actual checkpoint samples; num_dic is the implementation's tracked count and can exceed resident rows. These are not interchangeable occupancy measures.",
        "memory_finite": memory_finite,
        "bank_tensor_bytes_cpu": sum(item.numel() * item.element_size() for item in memory) + repre.numel() * repre.element_size(),
        "centroid_vs_resident_mean_abs_max": max_discrepancy,
        "retrieval_branch": {
            "top_m": top_m,
            "top_k": top_k,
            "random_center_subsampling": n > top_m,
            "candidate_centers": candidates,
            "top_k_selection": top_k < candidates,
            "weighted_centers": min(top_k, candidates),
            "mode": "top_k_softmax" if top_k < candidates else "all_candidate_centers_softmax",
            "score": "raw_dot_product_in_Bank.py_not_the_descriptive_cosine_metric",
        },
        "geometry": _geometry(repre),
    }


def _group(key):
    pieces = key.split(".")
    return ".".join(pieces[:2]) if pieces[0] == "mem_net" and len(pieces) > 2 else pieces[0]


def _state_changes(original, fixed):
    common = sorted(original.keys() & fixed.keys())
    accumulators, shape_mismatches, nonfinite_keys = {}, [], []
    floating_keys, static_keys = [], []
    for key in common:
        a, b = original[key], fixed[key]
        if not torch.is_tensor(a) or not torch.is_tensor(b):
            continue
        if a.shape != b.shape:
            shape_mismatches.append(key)
            continue
        if not (a.is_floating_point() or a.is_complex()) or not (b.is_floating_point() or b.is_complex()):
            static_keys.append(key)
            continue
        if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
            nonfinite_keys.append(key)
            continue
        dtype = torch.complex128 if a.is_complex() or b.is_complex() else torch.float64
        a64, b64 = a.to(device="cpu", dtype=dtype), b.to(device="cpu", dtype=dtype)
        base_squared = float(a64.abs().square().sum())
        delta = b64 - a64
        delta_squared = float(delta.abs().square().sum())
        entry = accumulators.setdefault(_group(key), {"base_squared": 0.0, "delta_squared": 0.0, "elements": 0, "keys": 0, "max_abs_delta": 0.0})
        entry["base_squared"] += base_squared
        entry["delta_squared"] += delta_squared
        entry["elements"] += a.numel()
        entry["keys"] += 1
        entry["max_abs_delta"] = max(entry["max_abs_delta"], float(delta.abs().max()) if delta.numel() else 0.0)
        floating_keys.append(key)
    groups = {}
    for name, item in accumulators.items():
        groups[name] = {
            "matched_floating_state_keys": item["keys"],
            "matched_elements": item["elements"],
            "original_l2": math.sqrt(item["base_squared"]),
            "delta_l2": math.sqrt(item["delta_squared"]),
            "relative_l2": math.sqrt(item["delta_squared"] / item["base_squared"]) if item["base_squared"] else None,
            "max_abs_delta": item["max_abs_delta"],
        }
    return {
        "common_state_key_count": len(common),
        "matched_floating_state_key_count": len(floating_keys),
        "static_nonfloating_keys_excluded": static_keys,
        "shape_mismatches_excluded": shape_mismatches,
        "nonfinite_keys_excluded": nonfinite_keys,
        "original_only_keys": sorted(original.keys() - fixed.keys()),
        "fixed_only_keys": sorted(fixed.keys() - original.keys()),
        "all_model_tensors_finite_original": all(bool(torch.isfinite(value).all()) for value in original.values() if torch.is_tensor(value)),
        "all_model_tensors_finite_fixed": all(bool(torch.isfinite(value).all()) for value in fixed.values() if torch.is_tensor(value)),
        "groups": groups,
        "caveat": "Same-key same-shape floating state tensors only. State tensors can include buffers; relative L2 is descriptive parameter/state divergence, not causal attribution or an activation statistic.",
    }


def _read_run(run_dir):
    path = Path(run_dir).resolve()
    metrics_path = path / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    if metrics.get("test_split_accessed") is not False or metrics.get("evaluation_split") != "validation":
        raise ValueError("Only explicitly validation-only training artifacts are accepted")
    checkpoint_path = Path(metrics["checkpoint"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = path / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    before = _sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not {"model", "memory", "repre", "num_dic"}.issubset(checkpoint):
        raise ValueError("Expected full S4M checkpoint with model, memory, repre, num_dic")
    return path, metrics_path, metrics, checkpoint_path, before, checkpoint


def audit_checkpoint_pair(original_run, fixed_run, top_m=100, top_k=20):
    """Return a small JSON-compatible audit without writing or using CUDA."""
    if not isinstance(top_m, int) or not isinstance(top_k, int) or top_m <= 0 or top_k <= 0:
        raise ValueError("top_m and top_k must be positive integers")
    started = time.monotonic()
    old = _read_run(original_run)
    new = _read_run(fixed_run)
    if old[2].get("seed") is None or old[2].get("seed") != new[2].get("seed"):
        raise ValueError("Checkpoint pairs must have the same seed")
    result = {
        "seed": old[2].get("seed"),
        "cpu_only": True,
        "model_instantiated": False,
        "new_training": False,
        "test_accessed": False,
        "original": {},
        "fixed": {},
        "model_state_changes": _state_changes(old[5]["model"], new[5]["model"]),
        "causal_limitations": "Observational comparison of saved best checkpoints. Initialization, artificial missing masks, and optimization trajectories vary across seeds; bank statistics do not establish why gains differ.",
    }
    for name, record in (("original", old), ("fixed", new)):
        run, metrics_path, metrics, checkpoint_path, before, checkpoint = record
        bank = _bank_summary(checkpoint, top_m, top_k)
        log = audit_training_log(run / "train.log", metrics)
        after = _sha256(checkpoint_path)
        if before != after:
            raise RuntimeError("Checkpoint changed during read-only audit")
        result[name] = {
            "run_dir": str(run),
            "metrics_path": str(metrics_path),
            "metrics_sha256": _sha256(metrics_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256_before": before,
            "checkpoint_sha256_after": after,
            "checkpoint_unchanged": True,
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "bank": bank,
            "training_log": log,
        }
    result["elapsed_seconds"] = time.monotonic() - started
    json.dumps(result, allow_nan=False)
    return result
