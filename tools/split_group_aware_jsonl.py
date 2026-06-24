#!/usr/bin/env python3
"""Create a group-aware train/test split for SmartBinocular ML JSONL."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ml_metadata_utils import (
    count_by,
    effective_label,
    git_branch_name,
    git_commit_hash,
    label_counts,
    manifest_for_paths,
    manifest_hash,
    markdown_table,
    optical_feature_hash,
    package_versions,
    read_jsonl,
    row_sha256,
    write_jsonl,
    write_text,
)


def _group_id(row: dict[str, Any], group_field: str) -> str:
    value = row.get(group_field)
    if value not in (None, ""):
        return str(value)
    source = row.get("source") or "unknown"
    return f"unresolved::{source}::{optical_feature_hash(row) or row_sha256(row)}"


def group_aware_split(
    rows: list[dict[str, Any]],
    *,
    train_ratio: float,
    seed: int,
    group_field: str = "split_group_id",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group = _group_id(row, group_field)
        out = dict(row)
        out[group_field] = group
        groups[group].append(out)

    by_label: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    for group, group_rows in groups.items():
        primary_label = Counter(effective_label(row) for row in group_rows).most_common(1)[0][0]
        by_label[primary_label].append((group, group_rows))

    rng = random.Random(seed)
    train_groups: set[str] = set()
    test_groups: set[str] = set()
    for label in sorted(by_label):
        entries = by_label[label][:]
        rng.shuffle(entries)
        total = sum(len(group_rows) for _, group_rows in entries)
        target_train = int(round(total * train_ratio))
        if total >= 2:
            target_train = max(1, min(total - 1, target_train))
        seen = 0
        for group, group_rows in entries:
            if seen < target_train:
                train_groups.add(group)
                seen += len(group_rows)
            else:
                test_groups.add(group)

    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for group, group_rows in groups.items():
        if group in train_groups:
            train_rows.extend(group_rows)
        elif group in test_groups:
            test_rows.extend(group_rows)
        else:
            train_rows.extend(group_rows)

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    train_group_values = {row[group_field] for row in train_rows}
    test_group_values = {row[group_field] for row in test_rows}
    source_overlap = sorted(
        {str(row.get("source")) for row in train_rows if row.get("source")}
        & {str(row.get("source")) for row in test_rows if row.get("source")}
    )
    summary = {
        "split_name": "group-aware split",
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "group_field": group_field,
        "group_count": len(groups),
        "group_overlap_count": len(train_group_values & test_group_values),
        "source_overlap_count": len(source_overlap),
        "source_overlap": source_overlap,
        "train_label_counts": label_counts(train_rows),
        "test_label_counts": label_counts(test_rows),
        "train_source_counts": count_by(train_rows, "source"),
        "test_source_counts": count_by(test_rows, "source"),
        "unresolved_train_rows": sum(1 for row in train_rows if str(row.get("metadata_status")) == "unresolved"),
        "unresolved_test_rows": sum(1 for row in test_rows if str(row.get("metadata_status")) == "unresolved"),
    }
    return train_rows, test_rows, summary


def write_distribution(path: Path, summary: dict[str, Any]) -> None:
    class_rows: list[list[Any]] = []
    labels = sorted(set(summary["train_label_counts"]) | set(summary["test_label_counts"]))
    for label in labels:
        class_rows.append(
            [
                label,
                summary["train_label_counts"].get(label, 0),
                summary["test_label_counts"].get(label, 0),
            ]
        )
    source_rows: list[list[Any]] = []
    sources = sorted(set(summary["train_source_counts"]) | set(summary["test_source_counts"]))
    for source in sources:
        source_rows.append(
            [
                source,
                summary["train_source_counts"].get(source, 0),
                summary["test_source_counts"].get(source, 0),
            ]
        )
    body = [
        "# Group-Aware Split Distribution",
        "",
        "This is a group-aware split, not a source-held-out split unless source overlap is zero.",
        "",
        "## Summary",
        "",
        markdown_table(
            ["Metric", "Value"],
            [
                ["train_rows", summary["train_rows"]],
                ["test_rows", summary["test_rows"]],
                ["group_field", summary["group_field"]],
                ["group_count", summary["group_count"]],
                ["group_overlap_count", summary["group_overlap_count"]],
                ["source_overlap_count", summary["source_overlap_count"]],
                ["unresolved_train_rows", summary["unresolved_train_rows"]],
                ["unresolved_test_rows", summary["unresolved_test_rows"]],
            ],
        ),
        "",
        "## Per-Class Support",
        "",
        markdown_table(["Class", "Train", "Test"], class_rows),
        "",
        "## Source Distribution",
        "",
        markdown_table(["Source", "Train", "Test"], source_rows),
        "",
        "## Remaining Source Overlap",
        "",
        ", ".join(summary["source_overlap"]) if summary["source_overlap"] else "No source-name overlap.",
    ]
    write_text(path, "\n".join(body) + "\n")


def write_manifest(path: Path, *, command: str, input_path: Path, outputs: list[Path], summary: dict[str, Any], seed: int, train_ratio: float) -> None:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_hash(),
        "branch": git_branch_name(),
        "command": command,
        "input_manifest": manifest_for_paths([input_path]),
        "output_manifest": manifest_for_paths([p for p in outputs if p.exists()]),
        "split_policy": "group-aware split by split_group_id; source overlap reported separately",
        "train_ratio": train_ratio,
        "seed": seed,
        "feature_set_id": "optical_12_baseline",
        "summary": summary,
        "versions": package_versions(),
    }
    manifest["manifest_hash"] = manifest_hash(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a group-aware SmartBinocular JSONL split.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--train-out", type=Path, required=True)
    p.add_argument("--test-out", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--distribution", type=Path, required=True)
    p.add_argument("--train-ratio", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--group-field", default="split_group_id")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0 < args.train_ratio < 1:
        print("ERROR: --train-ratio must be between 0 and 1.", file=sys.stderr)
        return 2
    rows = read_jsonl(args.input)
    train_rows, test_rows, summary = group_aware_split(
        rows,
        train_ratio=args.train_ratio,
        seed=args.seed,
        group_field=args.group_field,
    )
    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.test_out, test_rows)
    write_distribution(args.distribution, summary)
    write_manifest(
        args.manifest,
        command=" ".join([sys.executable, *sys.argv]),
        input_path=args.input,
        outputs=[args.train_out, args.test_out],
        summary=summary,
        seed=args.seed,
        train_ratio=args.train_ratio,
    )
    print(f"Wrote {args.train_out}")
    print(f"Wrote {args.test_out}")
    print(f"Group overlap count: {summary['group_overlap_count']}")
    print(f"Source overlap count: {summary['source_overlap_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
