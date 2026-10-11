#!/usr/bin/env python3
"""Decision-table unit test for the Stage-19 preregistered gates."""

from __future__ import annotations

import json

from analyze_stage19_external_validation import decision


def main() -> int:
    checks = {
        "full_pass": (
            decision(True, True) == "EXTERNAL_TRANSFER_FULL_PASS"
        ),
        "partial_pass": (
            decision(True, False) == "EXTERNAL_TRANSFER_PARTIAL_PASS"
        ),
        "external_fail_even_if_incremental": (
            decision(False, True) == "EXTERNAL_TRANSFER_FAIL"
        ),
        "full_fail": (
            decision(False, False) == "EXTERNAL_TRANSFER_FAIL"
        ),
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
