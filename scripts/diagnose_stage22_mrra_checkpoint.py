#!/usr/bin/env python3
"""Read-only finite-value audit for Stage22 MRRA checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def tensor_audit(name: str, value: torch.Tensor) -> dict[str, Any]:
    floating = value.is_floating_point() or value.is_complex()
    if not floating:
        return {
            "name": name,
            "shape": list(value.shape),
            "elements": int(value.numel()),
            "nonfinite_elements": 0,
            "finite_abs_max": None,
        }
    finite = torch.isfinite(value)
    finite_values = value[finite]
    return {
        "name": name,
        "shape": list(value.shape),
        "elements": int(value.numel()),
        "nonfinite_elements": int((~finite).sum().item()),
        "finite_abs_max": (
            float(finite_values.abs().max().item()) if finite_values.numel() else None
        ),
    }


def audit_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint state is not a mapping: {path}")

    tensors = [
        tensor_audit(str(name), value)
        for name, value in state.items()
        if torch.is_tensor(value)
    ]
    bad_tensors = [item for item in tensors if item["nonfinite_elements"] > 0]
    def is_mrra_key(name: str) -> bool:
        return name.startswith("mrra.") or ".mrra." in name

    mrra_tensors = [item for item in tensors if is_mrra_key(item["name"])]
    non_mrra_tensors = [item for item in tensors if not is_mrra_key(item["name"])]

    auxiliary = []
    if isinstance(checkpoint, dict):
        for name in ("memory", "repre", "memory1", "repre1", "memory2", "repre2"):
            value = checkpoint.get(name)
            if torch.is_tensor(value):
                auxiliary.append(tensor_audit(name, value))

    def summarize(items: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "tensor_count": len(items),
            "elements": sum(item["elements"] for item in items),
            "nonfinite_elements": sum(item["nonfinite_elements"] for item in items),
            "nonfinite_tensor_count": sum(
                item["nonfinite_elements"] > 0 for item in items
            ),
        }

    return {
        "checkpoint": str(path.resolve()),
        "all_model_tensors": summarize(tensors),
        "mrra_model_tensors": summarize(mrra_tensors),
        "non_mrra_model_tensors": summarize(non_mrra_tensors),
        "auxiliary_tensors": auxiliary,
        "nonfinite_model_tensors": bad_tensors,
        "mrra_tensor_statistics": mrra_tensors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"checkpoints": [audit_checkpoint(path) for path in args.checkpoints]}
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
