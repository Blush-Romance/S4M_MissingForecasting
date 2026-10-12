"""Tiny resumable job queue for a single GPU.

A job file holds one job per line:  <job_name> | <space separated key=value overrides>
Finished jobs (results/<name>.json exists) are skipped, so the queue can be
restarted at any time.  Failed jobs are logged and retried once on the next pass.

    python -m mtsf.queue jobs/main.txt --workers 1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

RESULTS = "results_mtsf"


def parse(path):
    jobs = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, ov = [s.strip() for s in line.split("|", 1)]
        jobs.append((name, ov.split()))
    return jobs


def sync_results(jobs, running):
    """Copy metrics.json of finished jobs into results_mtsf/."""
    for name in [j[0] for j in jobs]:
        src = f"runs_mtsf/{name}/metrics.json"
        dst = f"{RESULTS}/{name}.json"
        if os.path.exists(src) and not os.path.exists(dst) and name not in running:
            with open(src) as f:
                json.dump(json.load(f), open(dst, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobfile")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--config", nargs="+", default=["configs_mtsf/base.yaml"])
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs("logs_mtsf", exist_ok=True)
    running = {}
    failed = {}
    while True:
        jobs = parse(a.jobfile)  # re-read so jobs can be appended while running
        todo = [j for j in jobs if not os.path.exists(f"{RESULTS}/{j[0]}.json")
                and j[0] not in running and failed.get(j[0], 0) < 2]
        for name, p in list(running.items()):
            if p.poll() is not None:
                del running[name]
                if p.returncode != 0:
                    failed[name] = failed.get(name, 0) + 1
                    print(f"[queue] FAILED {name} rc={p.returncode}", flush=True)
                else:
                    print(f"[queue] done {name}", flush=True)
        if not todo and not running:
            sync_results(jobs, running)
            break
        while todo and len(running) < a.workers:
            name, ov = todo.pop(0)
            out_dir = f"runs_mtsf/{name}"
            cmd = [sys.executable, "-m", "mtsf.cli", "--config", *a.config, "--set", *ov,
                   f"out_dir={out_dir}", f"tag={name}"]
            log = open(f"logs_mtsf/{name}.log", "w")
            print(f"[queue] start {name}", flush=True)
            running[name] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        time.sleep(5)
        sync_results(jobs, running)
    print("[queue] all done", flush=True)


if __name__ == "__main__":
    main()
