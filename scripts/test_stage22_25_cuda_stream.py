#!/usr/bin/env python3
"""Static, negative, and scientific-dependency tests for Step-22.25."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile

import yaml

from run_stage14_uncertainty_cache import validate_config
import run_stage22_25_traffic_queue as queue


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/stage22_25_traffic"
SOURCE = ROOT / "configs/stage22_22_traffic/pg95_traffic_seed2027.yaml"


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> int:
    source = load(SOURCE)
    smoke = load(CONFIG_ROOT / "cuda_stream_traffic_validation_seed2027_smoke.yaml")
    formal = load(CONFIG_ROOT / "cuda_stream_traffic_validation_seed2027.yaml")
    validate_config(smoke, source)
    validate_config(formal, source)

    bad = copy.deepcopy(formal)
    bad["diagnostic"]["multiscale_weights"]["l5"] = 0.30
    rejected = False
    try:
        validate_config(bad, source)
    except ValueError:
        rejected = True

    with tempfile.TemporaryDirectory(prefix="stage22_25_gate_") as temporary:
        original_append_event = queue.scheduler.append_event
        queue.scheduler.append_event = lambda *args, **kwargs: None
        root = Path(temporary)
        failed_run = root / "failed"
        failed_run.mkdir()
        (failed_run / "metrics.json").write_text(json.dumps({
            "artifact_status": "PASS", "prediction_equivalence_pass": False,
        }), encoding="utf-8")
        failed_state = {"jobs": {
            queue.SMOKE_JOB: {"state": "SUCCESS", "run_dir": str(failed_run)},
            queue.FORMAL_JOB: {
                "state": "PENDING", "depends_on": [queue.SMOKE_JOB],
            },
        }}
        queue.enforce_scientific_gate(failed_state)

        passed_run = root / "passed"
        passed_run.mkdir()
        (passed_run / "metrics.json").write_text(json.dumps({
            "artifact_status": "PASS", "prediction_equivalence_pass": True,
        }), encoding="utf-8")
        passed_state = {"jobs": {
            queue.SMOKE_JOB: {"state": "SUCCESS", "run_dir": str(passed_run)},
            queue.FORMAL_JOB: {
                "state": "PENDING", "depends_on": [queue.SMOKE_JOB],
            },
        }}
        queue.enforce_scientific_gate(passed_state)
        queue.scheduler.append_event = original_append_event

    checks = {
        "configs_validate": True,
        "frozen_variant": all(
            item["diagnostic"]["variant"]
            == "stage22_25_traffic_cuda_stream_validation"
            for item in (smoke, formal)
        ),
        "changed_weight_rejected": rejected,
        "failed_smoke_blocks_formal": (
            failed_state["jobs"][queue.FORMAL_JOB]["state"]
            == "BLOCKED_DEPENDENCY"
            and failed_state["jobs"][queue.FORMAL_JOB]
            ["scientific_dependency"]["passed"] is False
        ),
        "passed_smoke_leaves_formal_pending": (
            passed_state["jobs"][queue.FORMAL_JOB]["state"] == "PENDING"
            and passed_state["jobs"][queue.FORMAL_JOB]
            ["scientific_dependency"]["passed"] is True
        ),
        "queue_smoke_then_formal": (
            len(queue.scheduler.JOB_DEFINITIONS) == 2
            and queue.scheduler.JOB_DEFINITIONS[1]["depends_on"]
            == [queue.SMOKE_JOB]
        ),
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
