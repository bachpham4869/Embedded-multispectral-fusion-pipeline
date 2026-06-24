#!/usr/bin/env python3
"""Audit raw-data recoverability for SmartBinocular ML JSONL records."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ml_metadata_utils import count_by, markdown_table, read_jsonl, row_sha256, write_text
from tools.rebuild_training_jsonl_with_metadata import SOURCE_SPECS, build_source_index, load_mapping


TRACE_FIELDS = [
    "source",
    "source_dataset",
    "original_image_path",
    "image_path",
    "filename",
    "frame_idx",
    "ts",
    "label",
    "label_source",
    "label_confidence",
]


def audit_jsonl(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    fields = sorted({field for row in rows for field in row})
    trace_present = [field for field in TRACE_FIELDS if field in fields]
    return {
        "path": str(path),
        "row_count": len(rows),
        "fields": fields,
        "trace_present": trace_present,
        "source_counts": count_by(rows, "source"),
        "metadata_status_counts": count_by(rows, "metadata_status"),
    }


def source_recoverability(rows: list[dict[str, Any]], source_root: Path, mapping_path: Path) -> list[dict[str, Any]]:
    mapping = load_mapping(mapping_path)
    index = build_source_index(source_root, mapping)
    counts = count_by(rows, "source")
    out: list[dict[str, Any]] = []
    for source, row_count in counts.items():
        spec = SOURCE_SPECS.get(source, {})
        root = source_root / spec.get("rel", "")
        candidates = index.get(source, [])
        deterministic = bool(spec.get("deterministic", False))
        if not spec:
            status = "unresolved"
            reason = "source not recognized by Phase 3 rebuilder"
        elif not root.exists():
            status = "unresolved"
            reason = "local raw dataset directory missing"
        elif deterministic:
            status = "verified"
            reason = "source plus frame_idx can replay deterministic raw image order"
        else:
            status = "inferred_low_confidence"
            reason = "raw images exist but original JSONL lacks path/hash; source used shuffled/subsampled order"
        out.append(
            {
                "source": source,
                "rows": row_count,
                "dataset": spec.get("dataset", "unknown"),
                "raw_root": str(root) if spec else "",
                "raw_root_exists": root.exists() if spec else False,
                "candidate_images": len(candidates),
                "metadata_status": status,
                "reason": reason,
            }
        )
    return sorted(out, key=lambda item: item["source"])


def write_audit_table(path: Path, source_rows: list[dict[str, Any]]) -> None:
    rows = [
        [
            item["source"],
            item["rows"],
            item["dataset"],
            item["raw_root_exists"],
            item["candidate_images"],
            item["metadata_status"],
            item["reason"],
        ]
        for item in source_rows
    ]
    body = [
        "# Raw Data Recoverability",
        "",
        "Feature-vector/hash matches are not identity evidence. `verified` means path/filename/frame identity is replayable or file SHA is available.",
        "",
        markdown_table(
            ["Source", "Rows", "Dataset", "Raw root exists", "Candidate images", "Metadata status", "Reason"],
            rows,
        ),
    ]
    write_text(path, "\n".join(body) + "\n")


def write_doc(path: Path, audits: list[dict[str, Any]], source_rows: list[dict[str, Any]]) -> None:
    file_rows = [
        [
            audit["path"],
            audit["row_count"],
            ", ".join(audit["trace_present"]),
            len(audit["fields"]),
        ]
        for audit in audits
    ]
    unresolved = [row for row in source_rows if row["metadata_status"] == "unresolved"]
    inferred = [row for row in source_rows if row["metadata_status"] == "inferred_low_confidence"]
    body = [
        "# Data Recoverability Audit",
        "",
        "Status: Phase 3 audit for the optical RGB-proxy baseline. This document does not claim live NIR/LWIR validation.",
        "",
        "## JSONL Field Coverage",
        "",
        markdown_table(["JSONL", "Rows", "Trace fields present", "Field count"], file_rows),
        "",
        "## Source-Level Recoverability",
        "",
        markdown_table(
            ["Source", "Rows", "Status", "Reason"],
            [[row["source"], row["rows"], row["metadata_status"], row["reason"]] for row in source_rows],
        ),
        "",
        "## Conclusion",
        "",
        "- `verified` is limited to sources where the raw path identity can be replayed from source plus frame index.",
        "- `inferred_low_confidence` sources keep candidate metadata separate from verified `file_sha256`/`dhash` fields.",
        "- Feature-vector/hash matching remains only a weak hint and is never treated as original-image identity.",
        f"- Unresolved source count: {len(unresolved)}.",
        f"- Inferred-low-confidence source count: {len(inferred)}.",
    ]
    write_text(path, "\n".join(body) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit JSONL/raw-image metadata recoverability.")
    p.add_argument("--jsonl", type=Path, action="append", required=True)
    p.add_argument("--logs-glob", default=None)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--mapping", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--doc", type=Path, required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    jsonl_paths = list(args.jsonl)
    if args.logs_glob:
        jsonl_paths.extend(sorted(Path().glob(args.logs_glob)))
    audits = [audit_jsonl(path) for path in jsonl_paths if path.exists()]
    combined_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in args.jsonl:
        if path.exists():
            for row in read_jsonl(path):
                fp = row_sha256(row)
                if fp in seen:
                    continue
                seen.add(fp)
                combined_rows.append(row)
    source_rows = source_recoverability(combined_rows, args.source_root, args.mapping)
    write_audit_table(args.out, source_rows)
    write_doc(args.doc, audits, source_rows)
    print(f"Wrote {args.out}")
    print(f"Wrote {args.doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
