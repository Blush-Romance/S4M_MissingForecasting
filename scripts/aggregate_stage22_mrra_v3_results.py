#!/usr/bin/env python3
"""Refresh results and run the combined Stage22 gate after MRRA-v3."""

from __future__ import annotations

import json
from pathlib import Path

import aggregate_results
import analyze_stage22_innovation


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime" / "stage22_mrra_v3" / "state.json"
FORMAL = "mrra_v3_pg95_electricity_seed2023_formal"


def main() -> int:
    aggregate_results.main()
    if not STATE.is_file():
        print("stage22_mrra_v3_gate=WAITING_STATE")
        return 0
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if state.get("jobs", {}).get(FORMAL, {}).get("state") != "SUCCESS":
        print("stage22_mrra_v3_gate=WAITING_FORMAL")
        return 0
    return analyze_stage22_innovation.main()


if __name__ == "__main__":
    raise SystemExit(main())
