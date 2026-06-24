#!/usr/bin/env python3
"""Create a duplicate-cluster-aware group split for SmartBinocular ML JSONL."""

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


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def _fallback_id(row: dict[str, Any]) -> str:
    return optical_feature_hash(row) or row_sha256(row)


def _super_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    dsu = DisjointSet(len(rows))
    by_cluster: dict[str, list[int]] = defaultdict(list)
    by_split_group: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        duplicate_cluster = row.get("duplicate_cluster_id") or f"missing_duplicate_cluster::{_fallback_id(row)}"
        split_group = row.get("split_group_id") or f"missing_split_group::{_fallback_id(row)}"
        by_cluster[str(duplicate_cluster)].append(idx)
        by_split_group[str(split_group)].append(idx)
    for buckets in (by_cluster, by_split_group):
        for indices in buckets.values():
            first = indices[0]
            for idx in indices[1:]:
                dsu.union(first, idx)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        out = dict(row)
        root = dsu.find(idx)
        out["cluster_aware_split_group_id"] = f"phase4_supergroup::{root:06d}"
        groups[out["cluster_aware_split_group_id"]].append(out)
    return groups


def duplicate_cluster_aware_split(
    rows: list[dict[str, Any]],
    *,
    train_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    groups = _super_groups(rows)
    by_label: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    for group, group_rows in groups.items():
        primary_label = Counter(effective_label(row) for row in group_rows).most_common(1)[0][0]
        by_label[primary_label].append((group, group_rows))

    rng = random.Random(seed)
    train_group_ids: set[str] = set()
    test_group_ids: set[str] = set()
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
                train_group_ids.add(group)
                seen += len(group_rows)
            else:
                test_group_ids.add(group)

    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for group, group_rows in groups.items():
        if group in train_group_ids:
            train_rows.extend(group_rows)
        elif group in test_group_ids:
            test_rows.extend(group_rows)
        else:
            train_rows.extend(group_rows)
    rng.shuffle(train_rows)
    rng.shuffle(test_rows)

    train_clusters = {str(row.get("duplicate_cluster_id")) for row in train_rows if row.get("duplicate_cluster_id")}
    test_clusters = {str(row.get("duplicate_cluster_id")) for row in test_rows if row.get("duplicate_cluster_id")}
    train_split_groups = {str(row.get("split_group_id")) for row in train_rows if row.get("split_group_id")}
    test_split_groups = {str(row.get("split_group_id")) for row in test_rows if row.get("split_group_id")}
    train_sources = {str(row.get("source")) for row in train_rows if row.get("source")}
    test_sources = {str(row.get("source")) for row in test_rows if row.get("source")}
    labels = sorted({effective_label(row) for row in rows})
    train_label_counts = label_counts(train_rows)
    test_label_counts = label_counts(test_rows)
    low_support_warnings = [
        label
        for label in labels
        if train_label_counts.get(label, 0) == 0 or test_label_counts.get(label, 0) == 0 or test_label_counts.get(label, 0) < 30
    ]
    summary = {
        "split_name": "duplicate-cluster-aware group split",
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "test_ratio": len(test_rows) / len(rows) if rows else 0.0,
        "super_group_count": len(groups),
        "duplicate_cluster_overlap_count": len(train_clusters & test_clusters),
        "split_group_overlap_count": len(train_split_groups & test_split_groups),
        "source_overlap_count": len(train_sources & test_sources),
        "source_overlap": sorted(train_sources & test_sources),
        "train_label_counts": train_label_counts,
        "test_label_counts": test_label_counts,
        "train_source_counts": count_by(train_rows, "source"),
        "test_source_counts": count_by(test_rows, "source"),
        "low_support_warnings": low_support_warnings,
        "selected_for_benchmark_feasible": (
            0.10 <= (len(test_rows) / len(rows) if rows else 0.0) <= 0.25
            and all(train_label_counts.get(label, 0) > 0 and test_label_counts.get(label, 0) > 0 for label in labels)
        ),
    }
    return train_rows, test_rows, summary


def write_distribution(path: Path, summary: dict[str, Any]) -> None:
    labels = sorted(set(summary["train_label_counts"]) | set(summary["test_label_counts"]))
    class_rows = [[label, summary["train_label_counts"].get(label, 0), summary["test_label_counts"].get(label, 0)] for label in labels]
    sources = sorted(set(summary["train_source_counts"]) | set(summary["test_source_counts"]))
    source_rows = [
        [source, summary["train_source_counts"].get(source, 0), summary["test_source_counts"].get(source, 0)]
        for source in sources
    ]
    body = [
        "# Cluster-Aware Split Distribution",
        "",
        "This is a duplicate-cluster-aware group split, not a source-held-out split unless source overlap is zero.",
        "",
        "## Summary",
        "",
        markdown_table(
            ["Metric", "Value"],
            [
                ["train_rows", summary["train_rows"]],
                ["test_rows", summary["test_rows"]],
                ["test_ratio", f"{summary['test_ratio']:.4f}"],
                ["super_group_count", summary["super_group_count"]],
                ["duplicate_cluster_overlap_count", summary["duplicate_cluster_overlap_count"]],
                ["split_group_overlap_count", summary["split_group_overlap_count"]],
                ["source_overlap_count", summary["source_overlap_count"]],
                ["selected_for_benchmark_feasible", summary["selected_for_benchmark_feasible"]],
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
        "## Low-Support Warnings",
        "",
        ", ".join(summary["low_support_warnings"]) if summary["low_support_warnings"] else "No zero-support or test<30 class warning.",
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
        "split_policy": "duplicate-cluster-aware group split by duplicate_cluster_id + split_group_id; source overlap reported separately",
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
    p = argparse.ArgumentParser(description="Create a duplicate-cluster-aware SmartBinocular JSONL split.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--train-out", type=Path, required=True)
    p.add_argument("--test-out", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--distribution", type=Path, required=True)
    p.add_argument("--train-ratio", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0 < args.train_ratio < 1:
        print("ERROR: --train-ratio must be between 0 and 1.", file=sys.stderr)
        return 2
    rows = read_jsonl(args.input)
    train_rows, test_rows, summary = duplicate_cluster_aware_split(rows, train_ratio=args.train_ratio, seed=args.seed)
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
    print(f"Duplicate cluster overlap count: {summary['duplicate_cluster_overlap_count']}")
    print(f"Split group overlap count: {summary['split_group_overlap_count']}")
    print(f"Source overlap count: {summary['source_overlap_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
