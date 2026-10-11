#!/usr/bin/env python3
"""Build a read-only, lossless classification index from one persisted CSV snapshot.

No resolved YAML, checkpoint, target cache, GPU, model, training, or evaluation
is read or invoked. SUCCESS is a lifecycle label, not a scientific quality gate.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "group_meeting_20261010"
SOURCE = ROOT / "reports" / "latest_results.csv"
FULL_CSV = OUT / "完整运行工件索引.csv"
SHORT_CSV = OUT / "完整运行工件短表.csv"
REGISTER = OUT / "_working" / "artifact_register.json"

FAILED_STATUSES = {
    "OOM", "CODE_ERROR", "DATA_ERROR", "CONFIG_ERROR", "NAN_OR_DIVERGENCE",
    "EXTERNAL_KILL", "UNKNOWN_ERROR", "FAILED", "FAIL", "ERROR",
}
NONTERMINAL_STATUSES = {
    "RUNNING", "PENDING", "QUEUED", "RETRYING", "WAITING_RESOURCE",
}
KINDS = ("SMOKE", "DIAGNOSTIC", "EFFICIENCY", "TRAIN_OR_EVAL", "OTHER")


def classify(row: dict[str, str]) -> tuple[str, str]:
    """Classification is only an objective artifact-purpose label, not a ranking."""
    eid = row["experiment_id"].lower()
    scope = row["protocol_scope"].lower()
    protocol = row["same_evaluation_protocol"].lower()
    if "__smoke__" in eid or "smoke" in scope or "smoke" in protocol:
        return "SMOKE", "ID/protocol explicitly marks smoke"
    if "skipped" in scope:
        return "OTHER", "protocol explicitly marks conditional skipped test"
    # Mixed multiscale evaluation-and-serial-timing is not timing-only; keep
    # those rows as TRAIN_OR_EVAL rather than silently counting extra tests.
    if (
        "same_gpu" in scope and "timing" in scope
        or "batch_parallel_equivalence_and_paired_timing" in scope
        or "cuda_stream_equivalence_and_paired_timing" in scope
        or "same_gpu_timing" in eid
    ):
        return "EFFICIENCY", "protocol explicitly marks timing/equivalence experiment"
    diagnostics = (
        "diagnostic", "diagnosis", "failure_attribution", "information_diagnostic",
        "temporal_audit", "descriptive", "cache_audit", "factorial",
        "path_intervention", "frozen_operational_fusion", "frozen_fusion",
        "gate_h", "valcache", "hcup", "uncertainty_cache", "validation_only_cache",
        "formal_validation_only_cache", "psavs_offline",
    )
    if any(token in scope or token in eid for token in diagnostics):
        return "DIAGNOSTIC", "ID/protocol explicitly marks cache, intervention, attribution, audit or diagnostic"
    if any(token in scope for token in ("formal", "training", "validation", "baseline")):
        return "TRAIN_OR_EVAL", "protocol explicitly marks training or prediction evaluation; not independently certified formal evidence"
    return "OTHER", "insufficient ID/protocol evidence for a narrower artifact-purpose label"


def main() -> None:
    source_stat = SOURCE.stat()
    raw = SOURCE.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    fields = list(reader.fieldnames or [])
    rows = list(reader)
    required = {"experiment_id", "dataset", "seed", "protocol_scope", "status", "mse", "mae"}
    assert required.issubset(fields), (required - set(fields))
    assert len(rows) == 302, f"Expected the requested 302-row snapshot, got {len(rows)}"
    assert all(None not in row and set(row) == set(fields) for row in rows), "Malformed CSV row"
    ids = [row["experiment_id"] for row in rows]
    assert all(ids), "Empty experiment ID"
    duplicates = sorted(eid for eid, count in Counter(ids).items() if count > 1)
    assert not duplicates, f"Duplicate IDs: {duplicates}"

    appended = ["report_category", "artifact_kind", "classification_basis", "source_row_number"]
    assert not set(fields) & set(appended), "Would overwrite an original field"
    enriched: list[dict[str, str | int]] = []
    for line_number, row in enumerate(rows, start=2):
        kind, basis = classify(row)
        status = row["status"].upper()
        category = "FAILED" if status in FAILED_STATUSES else kind
        if category == "FAILED":
            basis = f"persisted status={row['status']}; underlying artifact purpose: {basis}"
        enriched.append(dict(row, report_category=category, artifact_kind=kind,
                             classification_basis=basis, source_row_number=line_number))

    short_fields = ["experiment_id", "dataset", "seed", "scope", "status", "mse", "mae"]
    short_rows = [
        {name: row["protocol_scope"] if name == "scope" else row[name] for name in short_fields}
        for row in rows
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    with FULL_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + appended)
        writer.writeheader()
        writer.writerows(enriched)
    with SHORT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=short_fields)
        writer.writeheader()
        writer.writerows(short_rows)

    with FULL_CSV.open(encoding="utf-8-sig", newline="") as handle:
        roundtrip_reader = csv.DictReader(handle)
        roundtrip_fields = list(roundtrip_reader.fieldnames or [])
        roundtrip = list(roundtrip_reader)
    assert roundtrip_fields == fields + appended
    assert len(roundtrip) == len(rows)
    assert [{field: row[field] for field in fields} for row in roundtrip] == rows
    with SHORT_CSV.open(encoding="utf-8-sig", newline="") as handle:
        assert list(csv.DictReader(handle)) == short_rows
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_sha, "Source changed during build"

    status_counts = dict(sorted(Counter(row["status"] for row in rows).items()))
    nonterminal_rows = [row for row in rows if row["status"].upper() in NONTERMINAL_STATUSES]
    failed_rows = [row for row in rows if row["status"].upper() in FAILED_STATUSES]
    terminal_rows = [row for row in rows if row["status"].upper() not in NONTERMINAL_STATUSES]
    assert len(terminal_rows) == 301 and len(nonterminal_rows) == 1
    categories = Counter(row["report_category"] for row in enriched)
    underlying = Counter(row["artifact_kind"] for row in enriched)
    category_by_status: dict[str, dict[str, int]] = {}
    for category in (*KINDS, "FAILED"):
        category_by_status[category] = dict(sorted(Counter(
            row["status"] for row in enriched if row["report_category"] == category
        ).items()))

    register = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(SOURCE), "sha256": source_sha, "size_bytes": len(raw),
            "mtime_ns": source_stat.st_mtime_ns, "original_columns": fields,
            "original_column_count": len(fields),
            "snapshot_note": "Persisted CSV only; lifecycle states are not live PID/GPU verification.",
        },
        "counts": {
            "source_data_rows": len(rows), "output_data_rows": len(roundtrip),
            "physical_csv_rows_including_header": len(rows) + 1,
            "unique_experiment_ids": len(set(ids)), "duplicate_ids": duplicates,
            "terminal_status_snapshot_rows": len(terminal_rows),
            "nonterminal_status_snapshot_rows": len(nonterminal_rows),
            "successful_status_snapshot_rows": status_counts.get("SUCCESS", 0),
            "failed_status_snapshot_rows": len(failed_rows),
            "status_counts": status_counts,
            "report_category_counts": {name: categories[name] for name in (*KINDS, "FAILED")},
            "underlying_artifact_kind_counts": {name: underlying[name] for name in KINDS},
            "report_category_by_status": category_by_status,
        },
        "classification": {
            "appended_columns": appended,
            "priority": ["identify SMOKE/OTHER-skipped/EFFICIENCY/DIAGNOSTIC/TRAIN_OR_EVAL/OTHER purpose", "override report_category with FAILED when persisted failure status is explicit"],
            "failed_statuses": sorted(FAILED_STATUSES),
            "success_does_not_mean_formal_or_scientific_gate_pass": True,
            "train_or_eval_does_not_count_independent_accuracy_experiments": True,
            "purpose_is_identified_only_from_existing_id_and_protocol_fields": True,
            "excludes_model_or_yaml_inspection": True,
            "mixed_evaluation_and_timing": "TRAIN_OR_EVAL unless explicitly a timing/equivalence experiment",
            "failed_rows_retain_underlying_purpose": "artifact_kind column",
        },
        "integrity": {
            "all_302_source_rows_preserved": True,
            "all_original_columns_preserved": True,
            "all_original_field_values_and_row_order_roundtrip_exact": True,
            "all_ids_nonempty_and_unique": True,
            "source_unchanged_after_read": True,
            "short_table_roundtrip_exact": True,
        },
        "nonterminal_snapshot_rows": [
            {key: row[key] for key in ("experiment_id", "protocol_scope", "status", "mse", "mae")}
            for row in nonterminal_rows
        ],
        "important_count_note": "302 is the complete artifact snapshot; 301 is the terminal-status subset because one historical row is labeled RUNNING. Neither is an independent formal accuracy-experiment count. This task did not check live processes or update stale states.",
        "interpretation_boundaries": [
            "Do not rank smoke, diagnostic, timing or skipped artifacts as independent precision experiments.",
            "Do not rank across datasets, split/audit subsets, fixed assets, metrics or evaluation/RNG protocols.",
            "Multiple views/candidates of one checkpoint/cache are dependent artifacts.",
            "Every failure remains in the index; NA and original metric strings are retained without imputation.",
            "302 short-table rows are a machine-readable supplement, not a recommendation to print all in the main PDF.",
        ],
        "outputs": {
            "full_csv": str(FULL_CSV),
            "full_csv_sha256": hashlib.sha256(FULL_CSV.read_bytes()).hexdigest(),
            "short_csv": str(SHORT_CSV),
            "short_csv_sha256": hashlib.sha256(SHORT_CSV.read_bytes()).hexdigest(),
        },
        "short_table_columns": short_fields,
        "short_table_rows": short_rows,
    }
    with REGISTER.open("w", encoding="utf-8") as handle:
        json.dump(register, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    with REGISTER.open(encoding="utf-8") as handle:
        checked = json.load(handle)
    assert len(checked["short_table_rows"]) == 302
    assert sum(checked["counts"]["report_category_counts"].values()) == 302
    assert sum(checked["counts"]["status_counts"].values()) == 302
    print(json.dumps({"source_sha256": source_sha, "counts": register["counts"],
                      "integrity": register["integrity"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
