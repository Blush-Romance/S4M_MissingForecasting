#!/usr/bin/env python3
"""Wait for real GPU memory, launch one full-architecture smoke, and requeue OOM."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PROJECT_ROOT / "runtime" / "fullarch_gpu_smoke_waiter"
STATE_PATH = RUNTIME_ROOT / "state.json"
STOP_PATH = RUNTIME_ROOT / "stop.requested"
LOCK_PATH = PROJECT_ROOT / "runtime" / "locks" / "gpu_selection.lock"
ATTEMPT_LOG_ROOT = PROJECT_ROOT / "logs" / "fullarch_gpu_smoke_attempts"
RUNNER = PROJECT_ROOT / "scripts" / "run_s4m_experiment.py"


def iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def epoch_now() -> float:
    return time.time()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def query_gpus() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi exit={completed.returncode}: {completed.stdout.strip()}")
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            continue
        rows.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "total_mib": int(fields[3]),
                "used_mib": int(fields[4]),
                "free_mib": int(fields[5]),
            }
        )
    return rows


def choose_gpu(candidates: set[str], required_mib: int, last_launch_free_mib: int) -> dict[str, Any] | None:
    eligible = [
        gpu
        for gpu in query_gpus()
        if gpu["uuid"] in candidates
        and gpu["free_mib"] >= required_mib
        and gpu["free_mib"] > last_launch_free_mib
        and gpu["total_mib"] >= required_mib
    ]
    return max(eligible, key=lambda gpu: gpu["free_mib"], default=None)


def parse_run_dir(log_path: Path) -> Path | None:
    if not log_path.is_file():
        return None
    matches = re.findall(r"^run_dir=(.+)$", log_path.read_text(encoding="utf-8", errors="replace"), flags=re.MULTILINE)
    return Path(matches[-1]) if matches else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-candidates", required=True, help="comma-separated GPU UUIDs")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--initial-required-memory-mib", type=int, default=9000)
    parser.add_argument("--initial-oom-retry-count", type=int, default=1)
    parser.add_argument("--initial-last-launch-free-mib", type=int, default=4914)
    parser.add_argument("--initial-last-oom-time", default="")
    parser.add_argument("--resource-estimate-note", default="")
    args = parser.parse_args()

    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    ATTEMPT_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    candidates = {item.strip() for item in args.gpu_candidates.split(",") if item.strip()}
    state: dict[str, Any] = {
        "state": "STARTING",
        "waiter_pid": os.getpid(),
        "config": str((PROJECT_ROOT / args.config).resolve()),
        "gpu_candidates": sorted(candidates),
        "required_memory_estimate_mib": args.initial_required_memory_mib,
        "oom_retry_count": args.initial_oom_retry_count,
        "last_oom_time": args.initial_last_oom_time or iso_now(),
        "last_launch_free_mib": args.initial_last_launch_free_mib,
        "resource_estimate_note": args.resource_estimate_note,
        "next_eligible_epoch": epoch_now(),
        "updated_at": iso_now(),
    }
    atomic_json(STATE_PATH, state)

    while True:
        if STOP_PATH.exists():
            state.update({"state": "STOPPED", "updated_at": iso_now(), "note": "stop requested; no running experiment was killed"})
            atomic_json(STATE_PATH, state)
            return 0

        if epoch_now() < float(state["next_eligible_epoch"]):
            state.update({"state": "BACKOFF", "updated_at": iso_now()})
            atomic_json(STATE_PATH, state)
            time.sleep(min(args.poll_seconds, max(1, int(state["next_eligible_epoch"] - epoch_now()))))
            continue

        try:
            gpu = choose_gpu(candidates, int(state["required_memory_estimate_mib"]), int(state["last_launch_free_mib"]))
        except Exception as exc:
            state.update({"state": "WAITING_RESOURCE", "updated_at": iso_now(), "last_query_error": f"{type(exc).__name__}: {exc}"})
            atomic_json(STATE_PATH, state)
            time.sleep(args.poll_seconds)
            continue

        if gpu is None:
            snapshot = query_gpus()
            state.update({"state": "WAITING_RESOURCE", "updated_at": iso_now(), "gpu_snapshot": snapshot})
            atomic_json(STATE_PATH, state)
            time.sleep(args.poll_seconds)
            continue

        with LOCK_PATH.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            gpu = choose_gpu(candidates, int(state["required_memory_estimate_mib"]), int(state["last_launch_free_mib"]))
            if gpu is None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                time.sleep(args.poll_seconds)
                continue

            attempt = int(state["oom_retry_count"]) + 1
            attempt_log = ATTEMPT_LOG_ROOT / f"attempt_{attempt:02d}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            command = [
                sys.executable,
                str(RUNNER),
                "--config",
                args.config,
                "--gpu-uuid",
                gpu["uuid"],
                "--scheduler-attempt",
                str(attempt),
            ]
            log_handle = attempt_log.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            state.update(
                {
                    "state": "RUNNING",
                    "updated_at": iso_now(),
                    "experiment_pid": process.pid,
                    "selected_gpu": gpu,
                    "last_launch_free_mib": gpu["free_mib"],
                    "attempt": attempt,
                    "attempt_log": str(attempt_log),
                }
            )
            atomic_json(STATE_PATH, state)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

        while process.poll() is None:
            state.update({"state": "RUNNING", "updated_at": iso_now()})
            atomic_json(STATE_PATH, state)
            time.sleep(min(10, args.poll_seconds))
        log_handle.close()

        run_dir = parse_run_dir(attempt_log)
        run_status: dict[str, Any] = {}
        if run_dir and (run_dir / "status.json").is_file():
            run_status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        state.update(
            {
                "updated_at": iso_now(),
                "last_run_dir": str(run_dir) if run_dir else None,
                "last_run_status": run_status,
                "last_exit_code": process.returncode,
                "experiment_pid": None,
            }
        )

        if process.returncode == 0 and run_status.get("state") == "SUCCESS":
            state["state"] = "SUCCESS"
            atomic_json(STATE_PATH, state)
            return 0

        if run_status.get("state") == "OOM":
            retry_count = int(state["oom_retry_count"]) + 1
            previous_required = int(state["required_memory_estimate_mib"])
            launch_free = int(state["last_launch_free_mib"])
            increased = max(previous_required * 1.25, launch_free + 2048)
            state["oom_retry_count"] = retry_count
            state["last_oom_time"] = iso_now()
            state["required_memory_estimate_mib"] = int(math.ceil(increased / 512.0) * 512)
            backoff_seconds = min(300 * (2 ** max(0, retry_count - 1)), 3600)
            state["next_eligible_epoch"] = epoch_now() + backoff_seconds
            state["state"] = "BACKOFF"
            state["backoff_seconds"] = backoff_seconds
            atomic_json(STATE_PATH, state)
            continue

        state["state"] = "FAILED"
        state["failure_type"] = run_status.get("state", "UNKNOWN_ERROR")
        atomic_json(STATE_PATH, state)
        return process.returncode or 2


if __name__ == "__main__":
    raise SystemExit(main())
