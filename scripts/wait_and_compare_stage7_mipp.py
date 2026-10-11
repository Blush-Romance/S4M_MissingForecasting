#!/usr/bin/env python3
"""Wait without GPU use, then compare the formal Step-7 MIPP result."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime" / "stage7_mipp_queue" / "state.json"
LOG = ROOT / "logs" / "stage7_mipp_queue" / "comparison.log"


def main() -> int:
    while True:
        try:
            state = json.loads(STATE.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(60)
            continue
        queue_state = state.get("queue_state")
        if queue_state == "SUCCESS":
            break
        if queue_state in {"FAILED", "STOPPED"}:
            LOG.parent.mkdir(parents=True, exist_ok=True)
            LOG.write_text(
                json.dumps({"comparison": "NOT_RUN", "queue_state": queue_state}, indent=2)
                + "\n",
                encoding="utf-8",
            )
            return 2
        time.sleep(60)
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "compare_stage7_mipp_results.py")],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(completed.stdout, encoding="utf-8")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
