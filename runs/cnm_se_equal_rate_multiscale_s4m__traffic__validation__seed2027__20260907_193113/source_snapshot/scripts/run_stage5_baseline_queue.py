#!/usr/bin/env python3
"""Durable dependency-aware GPU queue for the Step-5 S4M baselines."""

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
import tempfile
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PROJECT_ROOT / "runtime" / "stage5_baseline_queue"
STATE_PATH = RUNTIME_ROOT / "state.json"
EVENTS_PATH = RUNTIME_ROOT / "events.jsonl"
STOP_PATH = RUNTIME_ROOT / "stop.requested"
SCHEDULER_LOCK_PATH = RUNTIME_ROOT / "scheduler.lock"
GPU_LOCK_PATH = PROJECT_ROOT / "runtime" / "locks" / "gpu_selection.lock"
LOG_ROOT = PROJECT_ROOT / "logs" / "stage5_baseline_queue"
RUNNER = PROJECT_ROOT / "scripts" / "run_s4m_experiment.py"
AGGREGATOR = PROJECT_ROOT / "scripts" / "aggregate_results.py"
QUEUE_NAME = "stage5_weather_etth1_single_seed_baselines"
MAX_CONCURRENT_JOBS = 5


JOB_DEFINITIONS = [
    {
        "id": "weather_smoke",
        "config": "configs/baseline/s4m_weather_point_r006_paper_adam_smoke.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 6144,
    },
    {
        "id": "etth1_smoke",
        "config": "configs/baseline/s4m_etth1_point_r006_paper_adam_smoke.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 6144,
    },
    {
        "id": "weather_formal",
        "config": "configs/baseline/s4m_weather_point_r006_paper_adam_seed2023.yaml",
        "depends_on": ["weather_smoke"],
        "required_memory_estimate_mib": 7168,
    },
    {
        "id": "etth1_formal",
        "config": "configs/baseline/s4m_etth1_point_r006_paper_adam_seed2023.yaml",
        "depends_on": ["etth1_smoke"],
        "required_memory_estimate_mib": 7168,
    },
]

WAITING_STATES = {"PENDING", "RETRYING", "WAITING_RESOURCE"}
TERMINAL_FAILURE_STATES = {
    "DATA_ERROR",
    "CONFIG_ERROR",
    "CODE_ERROR",
    "NAN_OR_DIVERGENCE",
    "UNKNOWN_ERROR",
    "BLOCKED_DEPENDENCY",
}


def iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_event(kind: str, **payload: Any) -> None:
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    event = {"at": iso_now(), "event": kind, **payload}
    with EVENTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def query_gpus() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi exit={completed.returncode}: {completed.stdout.strip()}")
    gpus: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 7:
            continue
        gpus.append(
            {
                "physical_index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "total_mib": int(fields[3]),
                "used_mib": int(fields[4]),
                "free_mib": int(fields[5]),
                "utilization_percent": int(fields[6]),
            }
        )
    return gpus


def query_gpu_process_memory() -> dict[tuple[str, int], int]:
    """Return live per-process GPU memory without relying on logical indices."""
    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi process query exit={completed.returncode}: {completed.stdout.strip()}")
    usage: dict[tuple[str, int], int] = {}
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or not fields[1].isdigit():
            continue
        key = (fields[0], int(fields[1]))
        try:
            usage[key] = usage.get(key, 0) + int(fields[2])
        except ValueError:
            continue
    return usage


def outstanding_reservations_by_gpu(
    jobs: dict[str, dict[str, Any]],
    process_memory: dict[tuple[str, int], int],
) -> dict[str, int]:
    """Keep launch reservations until each training process materializes them.

    ``nvidia-smi`` free memory already reflects allocations that have happened.
    Therefore only the unmaterialized part of each reservation is subtracted.
    This closes the data-loading/startup race where two jobs both observe the
    same free memory before the first job creates its CUDA context.
    """
    outstanding: dict[str, int] = {}
    for job in jobs.values():
        if job.get("state") != "RUNNING" or not job.get("gpu"):
            continue
        gpu_uuid = str(job["gpu"]["uuid"])
        reserved = int(job.get("launch_reservation_mib") or 0)
        training_pid = int(job.get("training_pid") or 0)
        materialized = process_memory.get((gpu_uuid, training_pid), 0) if training_pid else 0
        outstanding[gpu_uuid] = outstanding.get(gpu_uuid, 0) + max(0, reserved - materialized)
    return outstanding


def process_is_live(pid: int | None) -> bool:
    if not pid:
        return False
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        fields = stat_path.read_text(encoding="utf-8").split()
        return len(fields) > 2 and fields[2] != "Z"
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False


def parse_run_dir(log_path: Path) -> Path | None:
    if not log_path.is_file():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"^run_dir=(.+)$", text, flags=re.MULTILINE)
    return Path(matches[-1]) if matches else None


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def classify_failure(log_text: str, run_status: dict[str, Any]) -> str:
    reported = str(run_status.get("state", "")).upper()
    if reported in {
        "OOM",
        "DATA_ERROR",
        "CONFIG_ERROR",
        "CODE_ERROR",
        "NAN_OR_DIVERGENCE",
        "EXTERNAL_KILL",
        "UNKNOWN_ERROR",
    }:
        return reported
    lowered = log_text.lower()
    if "out of memory" in lowered or "cuda error: out of memory" in lowered:
        return "OOM"
    if "filenotfounderror" in lowered or "no such file or directory" in lowered:
        return "DATA_ERROR"
    if "config_error" in lowered or "yaml" in lowered and "error" in lowered:
        return "CONFIG_ERROR"
    if "nan" in lowered and ("loss" in lowered or "diverg" in lowered):
        return "NAN_OR_DIVERGENCE"
    if "traceback" in lowered or "runtimeerror" in lowered or "syntaxerror" in lowered:
        return "CODE_ERROR"
    if "killed" in lowered or "signal" in lowered:
        return "EXTERNAL_KILL"
    return "UNKNOWN_ERROR"


def classify_finished_attempt(
    log_text: str, run_status: dict[str, Any], return_code: int | None
) -> str:
    """Classify a stopped attempt, including scheduler-resume loss of Popen state."""
    if return_code is None and run_status.get("state") == "RUNNING":
        return "EXTERNAL_KILL"
    return classify_failure(log_text, run_status)


def initial_state(candidate_uuids: list[str], max_jobs: int) -> dict[str, Any]:
    jobs = {}
    for definition in JOB_DEFINITIONS:
        jobs[definition["id"]] = {
            **definition,
            "state": "PENDING",
            "attempt_count": 0,
            "oom_retry_count": 0,
            "next_eligible_epoch": 0,
            "pid": None,
            "gpu": None,
            "last_launch_free_mib": None,
            "last_launch_total_mib": None,
            "last_oom_time": None,
            "run_dir": None,
            "attempt_log": None,
            "history": [],
        }
    return {
        "queue": QUEUE_NAME,
        "queue_state": "STARTING",
        "scheduler_pid": os.getpid(),
        "created_at": iso_now(),
        "updated_at": iso_now(),
        "candidate_gpu_uuids": candidate_uuids,
        "max_concurrent_jobs": max_jobs,
        "jobs": jobs,
        "gpu_snapshot": [],
    }


def load_or_create_state(
    candidate_uuids: list[str],
    max_jobs: int,
    reset_jobs: str = "",
    reset_reason: str = "verified_code_fix",
) -> dict[str, Any]:
    if not STATE_PATH.is_file():
        return initial_state(candidate_uuids, max_jobs)
    state = read_json(STATE_PATH)
    if not state.get("jobs"):
        return initial_state(candidate_uuids, max_jobs)
    state["scheduler_pid"] = os.getpid()
    state["queue"] = QUEUE_NAME
    state["candidate_gpu_uuids"] = candidate_uuids
    state["max_concurrent_jobs"] = max_jobs
    state["updated_at"] = iso_now()
    for definition in JOB_DEFINITIONS:
        if definition["id"] not in state["jobs"]:
            state["jobs"][definition["id"]] = initial_state(candidate_uuids, max_jobs)["jobs"][definition["id"]]
    requested = {item.strip() for item in reset_jobs.split(",") if item.strip()}
    if "all" in requested:
        requested = set(state["jobs"])
    unknown = requested.difference(state["jobs"])
    if unknown:
        raise ValueError(f"unknown --reset-jobs entries: {sorted(unknown)}")
    for job_id in sorted(requested):
        job = state["jobs"][job_id]
        if job.get("state") == "RUNNING" and process_is_live(job.get("pid")):
            raise RuntimeError(f"cannot reset live job: {job_id} pid={job.get('pid')}")
        job.update(
            {
                "state": "PENDING",
                "pid": None,
                "gpu": None,
                "attempt_log": None,
                "run_dir": None,
                "runner_status": None,
                "training_pid": None,
                "next_eligible_epoch": 0,
                "oom_retry_count": 0,
                "last_oom_time": None,
                "requeue_reason": reset_reason,
                "requeued_at": iso_now(),
            }
        )
        for key in ("failure_summary", "last_failure_type", "finished_at"):
            job.pop(key, None)
        append_event(
            "manual_requeue_after_code_fix",
            job_id=job_id,
            preserved_attempt_count=job.get("attempt_count", 0),
            reason=reset_reason,
        )
    if requested:
        state.pop("finished_at", None)
        state["queue_state"] = "STARTING"
    return state


def update_dependency_states(state: dict[str, Any]) -> None:
    jobs = state["jobs"]
    for job in jobs.values():
        if job["state"] not in WAITING_STATES:
            continue
        dependency_states = [jobs[item]["state"] for item in job["depends_on"]]
        if any(item in TERMINAL_FAILURE_STATES for item in dependency_states):
            job["state"] = "BLOCKED_DEPENDENCY"
            job["failure_summary"] = f"dependency states: {dependency_states}"
            job["finished_at"] = iso_now()
            append_event("blocked_dependency", job_id=job["id"], dependencies=dependency_states)


def dependency_ready(job: dict[str, Any], jobs: dict[str, Any]) -> bool:
    return all(jobs[item]["state"] == "SUCCESS" for item in job["depends_on"])


def refresh_formal_memory_estimate(job: dict[str, Any], jobs: dict[str, Any]) -> None:
    if not job["id"].endswith("_formal") or not job["depends_on"]:
        return
    smoke = jobs[job["depends_on"][0]]
    run_dir = Path(smoke["run_dir"]) if smoke.get("run_dir") else None
    if not run_dir:
        return
    metrics = read_json(run_dir / "metrics.json")
    peak_reserved = metrics.get("peak_gpu_reserved_mb")
    if peak_reserved is None:
        return
    profiled = int(math.ceil((float(peak_reserved) + 1536) / 512.0) * 512)
    job["required_memory_estimate_mib"] = max(6144, profiled)
    job["memory_estimate_source"] = {
        "smoke_job": smoke["id"],
        "smoke_peak_reserved_mb": peak_reserved,
        "safety_margin_mib": 1536,
    }


def choose_gpu(
    job: dict[str, Any],
    gpus: list[dict[str, Any]],
    candidate_uuids: set[str],
    gpu_uuids_launched_this_poll: set[str],
    outstanding_reservation_mib_by_uuid: dict[str, int] | None = None,
) -> dict[str, Any] | None:
    required = int(job["required_memory_estimate_mib"])
    outstanding_reservation_mib_by_uuid = outstanding_reservation_mib_by_uuid or {}
    eligible = []
    for gpu in gpus:
        if (
            gpu["uuid"] not in candidate_uuids
            or gpu["uuid"] in gpu_uuids_launched_this_poll
        ):
            continue
        outstanding = int(outstanding_reservation_mib_by_uuid.get(gpu["uuid"], 0))
        effective_free = int(gpu["free_mib"]) - outstanding
        if effective_free < required or gpu["total_mib"] < required:
            continue
        if int(job.get("oom_retry_count", 0)) > 0:
            last_free = int(job.get("last_launch_free_mib") or 0)
            last_total = int(job.get("last_launch_total_mib") or 0)
            if gpu["free_mib"] <= last_free and gpu["total_mib"] <= last_total:
                continue
        eligible.append(
            {
                **gpu,
                "outstanding_reservation_mib": outstanding,
                "effective_free_mib": effective_free,
            }
        )
    return max(eligible, key=lambda item: item["effective_free_mib"], default=None)


def launch_job(job: dict[str, Any], gpu: dict[str, Any], state: dict[str, Any]) -> subprocess.Popen[str]:
    job["attempt_count"] = int(job["attempt_count"]) + 1
    attempt = int(job["attempt_count"])
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{job['id']}__attempt{attempt:02d}__{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    runner = Path(job.get("runner", RUNNER))
    command = [
        sys.executable,
        str(runner),
        "--config",
        job["config"],
        "--gpu-uuid",
        gpu["uuid"],
        "--scheduler-attempt",
        str(attempt),
    ]
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    job.update(
        {
            "state": "RUNNING",
            "pid": process.pid,
            "gpu": gpu,
            "attempt_log": str(log_path),
            "started_at": iso_now(),
            "last_launch_free_mib": gpu["free_mib"],
            "last_launch_total_mib": gpu["total_mib"],
            "launch_reservation_mib": int(job["required_memory_estimate_mib"]),
            "run_dir": None,
        }
    )
    job["history"].append(
        {
            "attempt": attempt,
            "started_at": job["started_at"],
            "pid": process.pid,
            "gpu": gpu,
            "attempt_log": str(log_path),
        }
    )
    append_event("launched", job_id=job["id"], pid=process.pid, gpu=gpu, attempt=attempt)
    atomic_json(STATE_PATH, state)
    return process


def finish_job(job: dict[str, Any], return_code: int | None) -> None:
    log_path = Path(job["attempt_log"]) if job.get("attempt_log") else Path("/nonexistent")
    run_dir = parse_run_dir(log_path)
    if run_dir:
        job["run_dir"] = str(run_dir)
    run_status = read_json(run_dir / "status.json") if run_dir else {}
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    success = run_status.get("state") == "SUCCESS" and (return_code in (0, None))

    history = job["history"][-1]
    history.update(
        {
            "finished_at": iso_now(),
            "return_code": return_code,
            "run_dir": str(run_dir) if run_dir else None,
            "run_state": run_status.get("state"),
        }
    )
    job["pid"] = None
    job["finished_at"] = history["finished_at"]

    if success:
        job["state"] = "SUCCESS"
        append_event("succeeded", job_id=job["id"], run_dir=job.get("run_dir"))
        return

    # A resumed scheduler has no Popen handle for jobs launched by the previous
    # scheduler process.  If that recorded PID is gone while the run artifact
    # still says RUNNING, the process hierarchy was externally terminated.
    # Classify it explicitly so the job is requeued instead of being frozen as
    # UNKNOWN_ERROR.
    failure = classify_finished_attempt(log_text, run_status, return_code)
    history["failure_type"] = failure
    job["last_failure_type"] = failure
    job["failure_summary"] = run_status.get("verification_error") or failure

    if failure in {"OOM", "EXTERNAL_KILL"}:
        if failure == "OOM":
            job["oom_retry_count"] = int(job["oom_retry_count"]) + 1
            job["last_oom_time"] = iso_now()
            required = max(
                float(job["required_memory_estimate_mib"]) * 1.25,
                float(job.get("last_launch_free_mib") or 0) + 1536,
            )
            job["required_memory_estimate_mib"] = int(math.ceil(required / 512.0) * 512)
        if failure == "EXTERNAL_KILL":
            job["external_kill_retry_count"] = int(job.get("external_kill_retry_count", 0)) + 1
            retries = int(job["external_kill_retry_count"])
            backoff = min(30 * (2 ** max(0, retries - 1)), 300)
        else:
            retries = int(job["oom_retry_count"])
            backoff = min(300 * (2 ** max(0, retries - 1)), 3600)
        job["next_eligible_epoch"] = time.time() + backoff
        job["state"] = "WAITING_RESOURCE" if retries >= 4 else "RETRYING"
        append_event(
            "requeued",
            job_id=job["id"],
            failure_type=failure,
            backoff_seconds=backoff,
            required_memory_estimate_mib=job["required_memory_estimate_mib"],
        )
        return

    job["state"] = failure
    append_event("failed", job_id=job["id"], failure_type=failure)


def aggregate_results() -> None:
    subprocess.run([sys.executable, str(AGGREGATOR)], cwd=PROJECT_ROOT, check=False)


def fake_worker(kind: str, marker: Path) -> int:
    if kind == "oom_once" and not marker.exists():
        marker.write_text("first attempt failed\n", encoding="utf-8")
        print("RuntimeError: CUDA out of memory")
        return 17
    if kind == "data_error":
        print("FileNotFoundError: fake missing asset")
        return 2
    time.sleep(0.12)
    print("SUCCESS")
    return 0


def self_test() -> int:
    root = Path(tempfile.mkdtemp(prefix="s4m_stage5_scheduler_selftest_"))
    marker = root / "oom.marker"
    jobs = {
        "fast_a": {"state": "PENDING", "depends": [], "kind": "success", "attempts": 0},
        "fast_b": {"state": "PENDING", "depends": [], "kind": "success", "attempts": 0},
        "fast_c": {"state": "PENDING", "depends": [], "kind": "success", "attempts": 0},
        "fast_d": {"state": "PENDING", "depends": [], "kind": "success", "attempts": 0},
        "fast_e": {"state": "PENDING", "depends": [], "kind": "success", "attempts": 0},
        "fast_f": {"state": "PENDING", "depends": [], "kind": "success", "attempts": 0},
        "oom_once": {"state": "PENDING", "depends": [], "kind": "oom_once", "attempts": 0},
        "dependent": {"state": "PENDING", "depends": ["fast_a"], "kind": "success", "attempts": 0},
        "data_error": {"state": "PENDING", "depends": [], "kind": "data_error", "attempts": 0},
    }
    running: dict[str, tuple[subprocess.Popen[str], Path]] = {}
    max_observed = 0
    refill_while_running = False
    events: list[dict[str, Any]] = []
    while True:
        for job_id, (process, log_path) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            text = log_path.read_text(encoding="utf-8", errors="replace")
            failure = classify_failure(text, {}) if code else None
            job = jobs[job_id]
            if failure == "OOM" and job["attempts"] < 2:
                job["state"] = "PENDING"
                events.append({"job": job_id, "event": "OOM_REQUEUE"})
            elif code == 0:
                job["state"] = "SUCCESS"
                events.append({"job": job_id, "event": "SUCCESS"})
            else:
                job["state"] = failure or "UNKNOWN_ERROR"
                events.append({"job": job_id, "event": job["state"]})
            del running[job_id]

        for job_id, job in jobs.items():
            if len(running) >= MAX_CONCURRENT_JOBS:
                break
            if job["state"] != "PENDING":
                continue
            if not all(jobs[dep]["state"] == "SUCCESS" for dep in job["depends"]):
                continue
            if running:
                refill_while_running = True
            job["attempts"] += 1
            log_path = root / f"{job_id}.{job['attempts']}.log"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--fake-worker",
                job["kind"],
                "--fake-marker",
                str(marker),
            ]
            with log_path.open("w", encoding="utf-8") as handle:
                process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
            job["state"] = "RUNNING"
            running[job_id] = (process, log_path)
            events.append({"job": job_id, "event": "LAUNCH", "attempt": job["attempts"]})
        max_observed = max(max_observed, len(running))

        terminal = {"SUCCESS", "DATA_ERROR", "CONFIG_ERROR", "CODE_ERROR", "NAN_OR_DIVERGENCE", "UNKNOWN_ERROR"}
        if not running and all(job["state"] in terminal for job in jobs.values()):
            break
        time.sleep(0.03)

    state_path = root / "resume_state.json"
    atomic_json(state_path, {"jobs": jobs, "events": events})
    reloaded = read_json(state_path)
    checks = {
        "queue_and_max_concurrency": max_observed == MAX_CONCURRENT_JOBS,
        "configured_max_jobs_supports_user_override": MAX_CONCURRENT_JOBS == 5,
        "dynamic_refill": refill_while_running,
        "oom_requeued_once_then_succeeded": jobs["oom_once"]["state"] == "SUCCESS" and jobs["oom_once"]["attempts"] == 2,
        "deterministic_error_not_retried": jobs["data_error"]["state"] == "DATA_ERROR" and jobs["data_error"]["attempts"] == 1,
        "dependency_order_completed": jobs["dependent"]["state"] == "SUCCESS",
        "resume_state_preserves_success": reloaded["jobs"]["oom_once"]["state"] == "SUCCESS",
        "stale_running_resume_is_external_kill": classify_finished_attempt(
            "", {"state": "RUNNING"}, None
        )
        == "EXTERNAL_KILL",
    }
    reservation_jobs = {
        "already_running": {
            "state": "RUNNING",
            "gpu": {"uuid": "GPU-test"},
            "launch_reservation_mib": 7168,
            "training_pid": 123,
        }
    }
    outstanding = outstanding_reservations_by_gpu(
        reservation_jobs, {("GPU-test", 123): 5632}
    )
    reservation_race_gpu = choose_gpu(
        {"required_memory_estimate_mib": 7168, "oom_retry_count": 0},
        [
            {
                "physical_index": 1,
                "uuid": "GPU-test",
                "name": "test",
                "total_mib": 24564,
                "used_mib": 16227,
                "free_mib": 8337,
                "utilization_percent": 99,
            }
        ],
        {"GPU-test"},
        set(),
        outstanding,
    )
    checks["unmaterialized_reservation_blocks_startup_race"] = (
        outstanding == {"GPU-test": 1536} and reservation_race_gpu is None
    )
    report = {
        "self_test": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "max_observed_running": max_observed,
        "artifact_root": str(root),
        "jobs": jobs,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 2


def scheduler_main(args: argparse.Namespace) -> int:
    if not 1 <= args.max_jobs <= MAX_CONCURRENT_JOBS:
        raise ValueError(
            f"--max-jobs must be in [1, {MAX_CONCURRENT_JOBS}]"
        )
    candidates = [item.strip() for item in args.gpu_candidates.split(",") if item.strip()]
    if not candidates:
        raise ValueError("at least one GPU UUID is required")

    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    GPU_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    scheduler_lock = SCHEDULER_LOCK_PATH.open("a+")
    try:
        fcntl.flock(scheduler_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another stage5 scheduler already holds the scheduler lock", file=sys.stderr)
        return 3

    state = load_or_create_state(
        candidates, args.max_jobs, args.reset_jobs, args.reset_reason
    )
    state["queue_state"] = "RUNNING"
    if STOP_PATH.exists() and args.resume:
        STOP_PATH.unlink()
    atomic_json(STATE_PATH, state)
    append_event("scheduler_started", pid=os.getpid(), resume=args.resume)

    live_processes: dict[str, subprocess.Popen[str]] = {}
    while True:
        state["scheduler_pid"] = os.getpid()
        state["updated_at"] = iso_now()
        jobs = state["jobs"]

        for job in jobs.values():
            if job["state"] != "RUNNING":
                continue
            log_path = Path(job["attempt_log"]) if job.get("attempt_log") else None
            if log_path:
                run_dir = parse_run_dir(log_path)
                if run_dir:
                    job["run_dir"] = str(run_dir)
                    run_status = read_json(run_dir / "status.json")
                    job["runner_status"] = run_status.get("state")
                    job["training_pid"] = run_status.get("pid")
            process = live_processes.get(job["id"])
            return_code = process.poll() if process else None
            live = return_code is None and process_is_live(job.get("pid"))
            if live:
                continue
            finish_job(job, return_code)
            live_processes.pop(job["id"], None)
            aggregate_results()

        update_dependency_states(state)

        stop_requested = STOP_PATH.exists()
        running_jobs = [job for job in jobs.values() if job["state"] == "RUNNING"]
        if not stop_requested:
            try:
                gpu_snapshot = query_gpus()
                state["gpu_snapshot"] = gpu_snapshot
                state.pop("last_gpu_query_error", None)
            except Exception as exc:
                gpu_snapshot = []
                state["last_gpu_query_error"] = f"{type(exc).__name__}: {exc}"

            # Allow safe same-GPU coexistence across polling cycles.  At most
            # one new task is placed on a GPU in each cycle, so its CUDA
            # allocation has time to appear in the next nvidia-smi snapshot
            # before another task can be admitted to the same card.
            gpu_uuids_launched_this_poll: set[str] = set()
            for job in jobs.values():
                if len(running_jobs) >= args.max_jobs:
                    break
                if job["state"] not in WAITING_STATES:
                    continue
                if time.time() < float(job.get("next_eligible_epoch", 0)):
                    continue
                if not dependency_ready(job, jobs):
                    continue
                refresh_formal_memory_estimate(job, jobs)
                with GPU_LOCK_PATH.open("a+") as gpu_lock:
                    fcntl.flock(gpu_lock.fileno(), fcntl.LOCK_EX)
                    locked_snapshot = query_gpus()
                    process_memory = query_gpu_process_memory()
                    outstanding_reservations = outstanding_reservations_by_gpu(
                        jobs, process_memory
                    )
                    gpu = choose_gpu(
                        job,
                        locked_snapshot,
                        set(candidates),
                        gpu_uuids_launched_this_poll,
                        outstanding_reservations,
                    )
                    if gpu is not None:
                        process = launch_job(job, gpu, state)
                        live_processes[job["id"]] = process
                        gpu_uuids_launched_this_poll.add(gpu["uuid"])
                        running_jobs.append(job)
                    else:
                        job["state"] = "WAITING_RESOURCE"
                        job["last_resource_check_at"] = iso_now()
                    fcntl.flock(gpu_lock.fileno(), fcntl.LOCK_UN)

        running_jobs = [job for job in jobs.values() if job["state"] == "RUNNING"]
        if all(job["state"] == "SUCCESS" for job in jobs.values()):
            state["queue_state"] = "SUCCESS"
            state["finished_at"] = iso_now()
            atomic_json(STATE_PATH, state)
            append_event("queue_succeeded")
            aggregate_results()
            return 0
        if stop_requested and not running_jobs:
            state["queue_state"] = "STOPPED"
            state["finished_at"] = iso_now()
            atomic_json(STATE_PATH, state)
            append_event("scheduler_stopped")
            return 0
        if not running_jobs and any(job["state"] in TERMINAL_FAILURE_STATES for job in jobs.values()):
            if all(job["state"] not in WAITING_STATES for job in jobs.values()):
                state["queue_state"] = "FAILED"
                state["finished_at"] = iso_now()
                atomic_json(STATE_PATH, state)
                append_event("queue_failed")
                aggregate_results()
                return 2

        waiting = any(job["state"] in WAITING_STATES for job in jobs.values())
        state["queue_state"] = "RUNNING" if running_jobs else ("WAITING_RESOURCE" if waiting else "FAILED")
        atomic_json(STATE_PATH, state)
        time.sleep(max(5, args.poll_seconds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-candidates", default="")
    parser.add_argument("--max-jobs", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reset-jobs",
        default="",
        help="comma-separated stopped jobs to requeue after a verified code fix; use all for this queue",
    )
    parser.add_argument(
        "--reset-reason",
        default="verified_code_fix",
        help="auditable reason stored when --reset-jobs requeues failed tasks",
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--fake-worker", choices=("success", "oom_once", "data_error"))
    parser.add_argument("--fake-marker", type=Path, default=Path("/tmp/s4m_fake_marker"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fake_worker:
        return fake_worker(args.fake_worker, args.fake_marker)
    if args.self_test:
        return self_test()
    return scheduler_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
