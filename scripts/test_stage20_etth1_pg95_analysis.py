#!/usr/bin/env python3
"""Unit checks for the frozen Stage-20 scientific decision table."""

from __future__ import annotations

import json
from pathlib import Path

from analyze_stage20_etth1_pg95 import decision


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "stage20_etth1_pg95_analysis_test.json"


def main() -> int:
    cases = {
        "full": decision(True, True, True),
        "incremental_fail": decision(True, True, False),
        "pg95_only": decision(True, False, False),
        "cnm_se_only": decision(False, True, True),
        "fail": decision(False, False, False),
    }
    expected = {
        "full": "FULL_METHOD_DIRECTIONAL_PASS",
        "incremental_fail": "MULTISCALE_INCREMENT_FAIL",
        "pg95_only": "PG95_ONLY_PASS",
        "cnm_se_only": "CNM_SE_ON_PG95_ONLY_PASS",
        "fail": "FULL_METHOD_FAIL",
    }
    if cases != expected:
        raise AssertionError({"actual": cases, "expected": expected})
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps({"status": "PASS", "cases": cases}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "cases": cases}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
