#!/usr/bin/env python3
"""Mechanically derive the frozen Step-22.22 Traffic configurations."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "configs/stage22_21_traffic"
OUTPUT_ROOT = ROOT / "configs/stage22_22_traffic"
RATES = {
    2026: 0.2598609211126311,
    2027: 0.2600889192886457,
    2028: 0.2599749202006384,
}
PROTOCOL_SCOPE = "stage22_22_traffic_frozen_multiseed_replication"


def load(name: str) -> dict:
    return yaml.safe_load((SOURCE_ROOT / name).read_text(encoding="utf-8"))


def atomic_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def seed_paths(seed: int) -> dict[str, str]:
    stem = f"traffic_point_external_r0.06_l5_seed{seed}.csv"
    return {
        "missing_csv": f"data/traffic/{stem}",
        "complete_csv": "data/traffic/traffic.csv",
        "mask_asset": f"artifacts/masks/traffic/point/rate_0.06/seed_{seed}.npz",
        "mask_metadata": f"artifacts/masks/traffic/point/rate_0.06/seed_{seed}.json",
        "source_manifest": "artifacts/manifests/s4m_data_assets.yaml",
        "official_mask_seed": "unavailable",
        "generation_seed": seed,
        "split": [0.7, 0.1, 0.2],
    }


def training_config(family: str, seed: int, smoke: bool) -> dict:
    template = load(
        f"{family}_traffic_seed2026{'_smoke' if smoke else ''}.yaml"
    )
    config = copy.deepcopy(template)
    experiment = config["experiment"]
    experiment.update({
        "realized_missing_rate": RATES[seed],
        "seed": seed,
        "smoke_test": smoke,
        "protocol_scope": PROTOCOL_SCOPE + ("_smoke" if smoke else "_formal"),
        "purpose": (
            f"engineering_smoke_before_{family}_seed{seed}_formal"
            if smoke
            else f"frozen_{family}_traffic_multiseed_replication_seed{seed}"
        ),
    })
    if smoke:
        experiment["required_memory_bank_warmups"] = 1
    config["data"] = seed_paths(seed)
    config["runtime"]["max_train_batches"] = 2 if smoke else -1
    config["runtime"]["max_eval_batches"] = 3 if smoke else -1
    official = config["official_args"]
    official.update({
        "seed": seed,
        "model_id": f"t22_{'s4' if family == 's4m' else 'p95'}_{seed}_{'s' if smoke else 'f'}",
        "des": f"T22{'S4' if family == 's4m' else 'P95'}{seed}{'S' if smoke else 'F'}",
        "train_epochs": 1 if smoke else 15,
        "max_train_batches": 2 if smoke else -1,
        "max_eval_batches": 3 if smoke else -1,
    })
    return config


def evaluation_config(
    method: str,
    seed: int,
    split: str,
    smoke: bool,
) -> dict:
    if split == "val":
        template = load(
            "pg95_traffic_multiscale_validation_seed2026"
            + ("_smoke.yaml" if smoke else ".yaml")
        )
    else:
        template = load("pg95_traffic_multiscale_test_seed2026.yaml")
    config = copy.deepcopy(template)
    experiment = config["experiment"]
    experiment.update({
        "model": (
            "cnm_se_equal_rate_multiscale"
            if method == "pg95"
            else "s4m_original_readonly_via_equal_rate_evaluator"
        ),
        "method": "pg95" if method == "pg95" else "s4m_baseline",
        "seed": seed,
        "smoke_test": smoke,
        "domain": (
            "validation_point_r006" if split == "val" else "test_point_r006"
        ),
        "purpose": (
            f"engineering_smoke_before_{method}_{split}_seed{seed}"
            if smoke
            else f"frozen_{method}_{split}_traffic_multiseed_replication_seed{seed}"
        ),
    })
    source_job = f"traffic_{method}_seed{seed}_formal"
    config["source"] = {"job_id": source_job}
    if seed == 2026:
        config["source"]["queue_state"] = (
            "runtime/stage22_21_traffic_pipeline/state.json"
        )
    config["data"] = {
        key: value
        for key, value in seed_paths(seed).items()
        if key in {"missing_csv", "complete_csv", "mask_asset"}
    }
    diagnostic = config["diagnostic"]
    diagnostic.update({
        "variant": (
            "stage22_22_traffic_s4m_test"
            if method == "s4m"
            else (
                "stage22_21_traffic_validation"
                if split == "val"
                else "stage22_21_traffic_test"
            )
        ),
        "split": split,
        "max_eval_batches": 3 if smoke else -1,
        "nested_seed": seed,
        "timing_warmup_batches": 1 if smoke else 10,
    })
    return config


def main() -> int:
    written: list[Path] = []
    for seed in (2027, 2028):
        for family in ("s4m", "pg95"):
            for smoke in (True, False):
                suffix = "_smoke" if smoke else ""
                path = OUTPUT_ROOT / f"{family}_traffic_seed{seed}{suffix}.yaml"
                atomic_yaml(path, training_config(family, seed, smoke))
                written.append(path)
        for smoke in (True, False):
            suffix = "_smoke" if smoke else ""
            path = OUTPUT_ROOT / (
                f"pg95_traffic_multiscale_validation_seed{seed}{suffix}.yaml"
            )
            atomic_yaml(path, evaluation_config("pg95", seed, "val", smoke))
            written.append(path)
        for method in ("s4m", "pg95"):
            for smoke in (True, False):
                suffix = "_smoke" if smoke else ""
                path = OUTPUT_ROOT / (
                    f"{method}_traffic_multiscale_test_seed{seed}{suffix}.yaml"
                )
                atomic_yaml(path, evaluation_config(method, seed, "test", smoke))
                written.append(path)

    for smoke in (True, False):
        suffix = "_smoke" if smoke else ""
        path = OUTPUT_ROOT / f"s4m_traffic_multiscale_test_seed2026{suffix}.yaml"
        atomic_yaml(path, evaluation_config("s4m", 2026, "test", smoke))
        written.append(path)

    if len(written) != 22:
        raise AssertionError(f"expected 22 configs, got {len(written)}")
    print(f"generated={len(written)} root={OUTPUT_ROOT}")
    for path in written:
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
