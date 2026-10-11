#!/usr/bin/env python3
"""Test Stage-20 dependency-to-checkpoint materialization without GPU."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile

import yaml

from run_stage20_etth1_pg95_pipeline import (
    materialize_evaluation_config,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (
    ROOT / "configs" / "stage20_etth1_pg95"
    / "s4m_etth1_pg95_multiscale_seed2023_smoke.yaml"
)
REPORT = ROOT / "reports" / "stage20_dispatcher_test.json"


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="stage20_dispatch_") as temporary:
        root = Path(temporary)
        run = root / "formal_run"
        checkpoint = run / "checkpoints" / "setting" / "checkpoint.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        (run / "resolved_config.yaml").write_text(
            "official_args:\n  use_mipp: true\n",
            encoding="utf-8",
        )
        write_json(run / "metrics.json", {"checkpoint": str(checkpoint)})
        write_json(run / "status.json", {"state": "SUCCESS"})
        state = {
            "jobs": {
                "etth1_pg95_seed2023_formal": {
                    "state": "SUCCESS",
                    "run_dir": str(run),
                }
            }
        }
        state_path = root / "state.json"
        write_json(state_path, state)
        destination = root / "materialized.yaml"
        materialize_evaluation_config(
            TEMPLATE,
            state_path=state_path,
            destination=destination,
        )
        materialized = yaml.safe_load(
            destination.read_text(encoding="utf-8")
        )
        source = materialized["source"]
        expected = {
            "job_id": "etth1_pg95_seed2023_formal",
            "run_dir": str(run.resolve()),
            "resolved_config": str((run / "resolved_config.yaml").resolve()),
            "metrics": str((run / "metrics.json").resolve()),
            "checkpoint": str(checkpoint.resolve()),
        }
        if source != expected:
            raise AssertionError({"source": source, "expected": expected})

        failed_state = copy.deepcopy(state)
        failed_state["jobs"]["etth1_pg95_seed2023_formal"]["state"] = (
            "CODE_ERROR"
        )
        write_json(state_path, failed_state)
        try:
            materialize_evaluation_config(
                TEMPLATE,
                state_path=state_path,
                destination=root / "should_not_exist.yaml",
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("failed dependency was accepted")

        write_json(state_path, state)
        outside_checkpoint = root / "outside_checkpoint.pth"
        outside_checkpoint.write_bytes(b"outside")
        write_json(
            run / "metrics.json",
            {"checkpoint": str(outside_checkpoint)},
        )
        try:
            materialize_evaluation_config(
                TEMPLATE,
                state_path=state_path,
                destination=root / "outside_should_not_exist.yaml",
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("checkpoint outside source run was accepted")

    payload = {
        "status": "PASS",
        "source_handoff": "exact",
        "failed_dependency_rejected": True,
        "outside_checkpoint_rejected": True,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
