#!/usr/bin/env python3
"""Start the MIPP queue only after the default-off baseline closure passes."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = (
    ROOT
    / "runs"
    / "s4m_gate_h__weather__id_point_r006__seed2023__smoke__20260717_212748"
    / "metrics.json"
)
STATE = ROOT / "runtime" / "stage7_mipp_queue" / "baseline_closure_gate.json"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    status_path = run_dir / "status.json"
    atomic_json(
        STATE,
        {
            "state": "WAITING_BASELINE_CLOSURE",
            "run_dir": str(run_dir),
            "reference_metrics": str(REFERENCE),
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )
    while True:
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(max(5, args.poll_seconds))
            continue
        state = status.get("state")
        if state == "SUCCESS":
            break
        if state not in {"RUNNING", None}:
            atomic_json(
                STATE,
                {
                    "state": "BASELINE_CLOSURE_FAILED",
                    "run_dir": str(run_dir),
                    "run_state": state,
                    "verification_error": status.get("verification_error"),
                    "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                },
            )
            return 2
        time.sleep(max(5, args.poll_seconds))

    current = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    keys = (
        "prototype_on_mse",
        "prototype_off_mse",
        "prototype_on_mae",
        "prototype_off_mae",
        "top1_similarity_mean",
        "selected_similarity_mean",
        "retrieval_entropy_mean",
    )
    differences = {
        key: abs(float(current[key]) - float(reference[key])) for key in keys
    }
    tolerance = 1e-12
    passed = all(value <= tolerance for value in differences.values())
    if not passed:
        atomic_json(
            STATE,
            {
                "state": "BASELINE_NUMERICAL_CLOSURE_FAILED",
                "run_dir": str(run_dir),
                "reference_metrics": str(REFERENCE),
                "absolute_differences": differences,
                "tolerance": tolerance,
                "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        )
        return 3

    start = subprocess.run(
        ["bash", str(ROOT / "scripts" / "start_stage7_mipp_queue.sh")],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env={**os.environ, "MAX_JOBS": "3"},
    )
    atomic_json(
        STATE,
        {
            "state": "PASS_QUEUE_START_REQUESTED" if start.returncode == 0 else "QUEUE_START_FAILED",
            "run_dir": str(run_dir),
            "reference_metrics": str(REFERENCE),
            "absolute_differences": differences,
            "tolerance": tolerance,
            "queue_start_return_code": start.returncode,
            "queue_start_output": start.stdout.strip(),
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )
    print(start.stdout, end="")
    return start.returncode


if __name__ == "__main__":
    raise SystemExit(main())
