#!/usr/bin/env python3
"""Use the frozen BF-CMVD comparison logic for the v2 token-filter queue."""

from __future__ import annotations

from pathlib import Path

import aggregate_stage22_26_bf_cmvd as base


ROOT = Path(__file__).resolve().parents[1]
base.STATE = ROOT / "runtime/stage22_26_traffic_v2/state.json"
base.REPORT_JSON = ROOT / "reports/stage22_26_bf_cmvd_v2_seed2027.json"
base.REPORT_MD = ROOT / "reports/stage22_26_bf_cmvd_v2_seed2027.md"
base.SMOKE_JOB = "bf_cmvd_v2_traffic_validation_seed2027_smoke"
base.FORMAL_JOB = "bf_cmvd_v2_traffic_validation_seed2027_formal"


if __name__ == "__main__":
    raise SystemExit(base.main())
