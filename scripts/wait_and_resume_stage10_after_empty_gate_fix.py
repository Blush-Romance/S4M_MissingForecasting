#!/usr/bin/env python3
"""Resume only PG95 jobs affected by the Step-10 empty-gate diagnostic bug."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime" / "stage10_weather_multiseed_queue"
STATE = RUNTIME / "state.json"
HANDOFF = RUNTIME / "empty_gate_fix_handoff.json"
START = ROOT / "scripts" / "start_stage10_weather_multiseed_queue.sh"


def iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def scheduler_is_live(pid: int) -> bool:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False
    return (
        len(fields) > 2
        and fields[2] != "Z"
        and b"run_stage10_weather_multiseed_queue.py" in command
    )


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def jobs_to_reset(state: dict) -> list[str]:
    reset = []
    jobs = state.get("jobs", {})
    for seed in (2024, 2025):
        smoke_id = f"weather_pg95_seed{seed}_smoke"
        formal_id = f"weather_pg95_seed{seed}_formal"
        smoke_state = jobs.get(smoke_id, {}).get("state")
        formal_state = jobs.get(formal_id, {}).get("state")
        if smoke_state != "SUCCESS":
            reset.extend((smoke_id, formal_id))
        elif formal_state != "SUCCESS":
            reset.append(formal_id)
    return reset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-hours", type=float, default=120.0)
    args = parser.parse_args()
    deadline = time.time() + args.max_hours * 3600.0
    while scheduler_is_live(args.scheduler_pid):
        if time.time() >= deadline:
            atomic_json(
                HANDOFF,
                {
                    "state": "TIMEOUT",
                    "at": iso_now(),
                    "scheduler_pid": args.scheduler_pid,
                },
            )
            return 3
        time.sleep(max(10, args.poll_seconds))

    state = json.loads(STATE.read_text(encoding="utf-8"))
    reset = jobs_to_reset(state)
    if not reset:
        atomic_json(
            HANDOFF,
            {
                "state": "NO_REPAIR_NEEDED",
                "at": iso_now(),
                "previous_queue_state": state.get("queue_state"),
            },
        )
        return 0

    environment = os.environ.copy()
    environment.update(
        {
            "RESET_JOBS": ",".join(reset),
            "RESET_REASON": "fix_absolute_gate_empty_selection_diagnostics",
            "MAX_JOBS": "5",
        }
    )
    completed = subprocess.run(
        ["bash", str(START)],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    atomic_json(
        HANDOFF,
        {
            "state": "RESTARTED" if completed.returncode == 0 else "RESTART_FAILED",
            "at": iso_now(),
            "previous_queue_state": state.get("queue_state"),
            "reset_jobs": reset,
            "return_code": completed.returncode,
            "output": completed.stdout,
        },
    )
    print(completed.stdout, end="")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
