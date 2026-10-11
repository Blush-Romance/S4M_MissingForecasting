#!/usr/bin/env python3
"""Static, negative, routing, and decision tests for Step-22.24."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import yaml

from aggregate_stage22_24_batch_parallel import scientific_decision
from run_stage14_uncertainty_cache import validate_config
import run_stage22_24_traffic_queue as queue


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/stage22_24_traffic"
SOURCE_CONFIG = (
    ROOT / "configs/stage22_22_traffic/pg95_traffic_seed2027.yaml"
)


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def rejected(callback) -> bool:
    try:
        callback()
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError):
        return True
    return False


def main() -> int:
    source = load(SOURCE_CONFIG)
    smoke = load(CONFIG_ROOT / "batch_parallel_traffic_validation_seed2027_smoke.yaml")
    formal = load(CONFIG_ROOT / "batch_parallel_traffic_validation_seed2027.yaml")
    validate_config(smoke, source)
    validate_config(formal, source)

    bad_weight = copy.deepcopy(formal)
    bad_weight["diagnostic"]["multiscale_weights"]["l20"] = 0.30
    bad_split = copy.deepcopy(formal)
    bad_split["diagnostic"]["split"] = "test"
    bad_seed = copy.deepcopy(formal)
    bad_seed["experiment"]["seed"] = 2029
    bad_seed["diagnostic"]["nested_seed"] = 2029
    bad_source = copy.deepcopy(source)
    bad_source["official_args"]["seed"] = 2029

    batch_size = 4
    routes = [torch.randn(batch_size, 3, 2) for _ in range(3)]
    serial = [value.square() + 2.0 * value for value in routes]
    combined = torch.cat(routes, dim=0)
    parallel = list(torch.split(combined.square() + 2.0 * combined, batch_size))

    definitions = queue.scheduler.JOB_DEFINITIONS
    checks = {
        "smoke_and_formal_validate": True,
        "frozen_variant": all(
            config["diagnostic"]["variant"]
            == "stage22_24_traffic_batch_parallel_validation"
            for config in (smoke, formal)
        ),
        "smoke_budget": (
            smoke["diagnostic"]["max_eval_batches"] == 3
            and smoke["diagnostic"]["timing_warmup_batches"] == 1
        ),
        "formal_budget": (
            formal["diagnostic"]["max_eval_batches"] == -1
            and formal["diagnostic"]["timing_warmup_batches"] == 10
        ),
        "changed_weight_rejected": rejected(lambda: validate_config(bad_weight, source)),
        "test_split_rejected": rejected(lambda: validate_config(bad_split, source)),
        "unregistered_seed_rejected": rejected(
            lambda: validate_config(bad_seed, bad_source)
        ),
        "batch_concat_split_routes_exact": all(
            torch.equal(left, right) for left, right in zip(serial, parallel)
        ),
        "queue_smoke_then_formal": (
            len(definitions) == 2
            and definitions[0]["depends_on"] == []
            and definitions[1]["depends_on"]
            == ["traffic_batch_parallel_validation_seed2027_smoke"]
        ),
        "decision_equivalence_fail": scientific_decision(
            complete=True, protocol_pass=True,
            equivalence_pass=False, speed_gate_pass=True,
        ) == "BATCH_PARALLEL_EQUIVALENCE_FAIL",
        "decision_speed_fail": scientific_decision(
            complete=True, protocol_pass=True,
            equivalence_pass=True, speed_gate_pass=False,
        ) == "BATCH_PARALLEL_SPEED_GATE_FAIL",
        "decision_authorizes_replication": scientific_decision(
            complete=True, protocol_pass=True,
            equivalence_pass=True, speed_gate_pass=True,
        ) == "BATCH_PARALLEL_REPLICATION_SEED2028_AUTHORIZED",
    }
    failures = [name for name, passed in checks.items() if not passed]
    print(json.dumps({
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
    }, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
