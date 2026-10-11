#!/usr/bin/env python3
"""Materialize and dispatch Step-22.24 frozen evaluation configs."""

from __future__ import annotations

import run_stage22_22_traffic_pipeline as pipeline


pipeline.MATERIALIZED_ROOT = (
    pipeline.ROOT / "runtime/stage22_24_traffic_pipeline/materialized"
)


if __name__ == "__main__":
    raise SystemExit(pipeline.main())
