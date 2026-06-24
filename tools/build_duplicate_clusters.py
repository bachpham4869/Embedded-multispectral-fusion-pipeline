#!/usr/bin/env python3
"""Build duplicate clusters over a full metadata-enriched SmartBinocular JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_dhash_pairs import classify_pair_category
from tools.ml_metadata_utils import (
    count_by,
    effective_label,
    git_branch_name,
    git_commit_hash,
    hamming_distance_hex,
    label_counts,
    manifest_for_paths,
    manifest_hash,
    markdown_table,
    package_versions,
    read_jsonl,
    row_sha256,
    sha256_json,
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


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _row_id(row: dict[str, Any]) -> str:
    for key in ("file_sha256", "relative_image_id", "candidate_relative_image_id", "split_group_id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return row_sha256(row)


def _cluster_id(mode: str, member_indices: list[int], rows: list[dict[str, Any]]) -> str:
    member_ids = sorted(_row_id(rows[idx]) for idx in member_indices)
    digest = sha256_json({"mode": mode, "members": member_ids})[:24]
    return f"dupcluster::{mode}::{digest}"


def build_duplicate_clusters(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    dhash_threshold: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    dsu = DisjointSet(len(rows))
    edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[int, int]] = set()

    by_sha: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        sha = row.get("file_sha256")
        if sha:
            by_sha[str(sha)].append(idx)
    for sha, indices in by_sha.items():
        if len(indices) < 2:
            continue
        for a, b in combinations(indices, 2):
            key = _edge_key(a, b)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            dsu.union(a, b)
            edges.append(
                {
                    "edge_type": "exact_file_duplicate",
                    "category": "exact_file_duplicate",
                    "row_a": a,
                    "row_b": b,
                    "hamming_distance": "",
                    "file_sha256": sha,
                    "label_a": effective_label(rows[a]),
                    "label_b": effective_label(rows[b]),
                    "source_a": rows[a].get("source", "unknown"),
                    "source_b": rows[b].get("source", "unknown"),
                }
            )

    dhash_rows = [(idx, row, str(row["dhash"])) for idx, row in enumerate(rows) if row.get("dhash")]
    for left_pos, (idx_a, row_a, hash_a) in enumerate(dhash_rows):
        for idx_b, row_b, hash_b in dhash_rows[left_pos + 1 :]:
            distance = hamming_distance_hex(hash_a, hash_b)
            if distance > dhash_threshold:
                continue
            key = _edge_key(idx_a, idx_b)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            dsu.union(idx_a, idx_b)
            category = classify_pair_category(row_a, row_b, hamming_distance=distance)
            edges.append(
                {
                    "edge_type": "dhash_screening",
                    "category": category,
                    "row_a": idx_a,
                    "row_b": idx_b,
                    "hamming_distance": distance,
                    "file_sha256": "",
                    "label_a": effective_label(row_a),
                    "label_b": effective_label(row_b),
                    "source_a": row_a.get("source", "unknown"),
                    "source_b": row_b.get("source", "unknown"),
                    "metadata_status_a": row_a.get("metadata_status", ""),
                    "metadata_status_b": row_b.get("metadata_status", ""),
                }
            )

    components: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(rows)):
        components[dsu.find(idx)].append(idx)

    out_rows = [dict(row) for row in rows]
    cluster_sizes: Counter[int] = Counter()
    cluster_label_counts: dict[str, dict[str, int]] = {}
    cluster_source_counts: dict[str, dict[str, int]] = {}
    for member_indices in components.values():
        cluster = _cluster_id(mode, member_indices, rows)
        cluster_sizes[len(member_indices)] += 1
        cluster_label_counts[cluster] = label_counts(rows[idx] for idx in member_indices)
        cluster_source_counts[cluster] = count_by((rows[idx] for idx in member_indices), "source")
        for idx in member_indices:
            out_rows[idx]["duplicate_cluster_id"] = cluster
            out_rows[idx]["duplicate_cluster_mode"] = mode
            out_rows[idx]["duplicate_cluster_dhash_threshold"] = dhash_threshold

    category_counts = Counter(str(edge["category"]) for edge in edges)
    edge_type_counts = Counter(str(edge["edge_type"]) for edge in edges)
    cluster_ids = [str(row["duplicate_cluster_id"]) for row in out_rows]
    cluster_counts = Counter(cluster_ids)
    largest = cluster_counts.most_common(20)
    summary = {
        "mode": mode,
        "dhash_threshold": dhash_threshold,
        "row_count": len(rows),
        "dhash_coverage_rows": len(dhash_rows),
        "file_sha256_coverage_rows": sum(1 for row in rows if row.get("file_sha256")),
        "cluster_count": len(cluster_counts),
        "non_singleton_cluster_count": sum(1 for _, size in cluster_counts.items() if size > 1),
        "max_cluster_size": max(cluster_counts.values()) if cluster_counts else 0,
        "edge_count": len(edges),
        "exact_file_sha_edges": edge_type_counts.get("exact_file_duplicate", 0),
        "dhash_edges": edge_type_counts.get("dhash_screening", 0),
        "edge_category_counts": dict(sorted(category_counts.items())),
        "cluster_size_distribution": dict(sorted(cluster_sizes.items())),
        "largest_clusters": [
            {
                "duplicate_cluster_id": cluster,
                "size": size,
                "labels": cluster_label_counts.get(cluster, {}),
                "sources": cluster_source_counts.get(cluster, {}),
            }
            for cluster, size in largest
        ],
    }
    return out_rows, summary, edges


def write_edges_csv(path: Path, edges: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output: list[dict[str, Any]] = []
    for edge in edges:
        cluster_a = rows[int(edge["row_a"])].get("duplicate_cluster_id", "")
        output.append({**edge, "duplicate_cluster_id": cluster_a})
    fieldnames = sorted({key for row in output for key in row.keys()}) if output else ["edge_type"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output)


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    largest_rows = [
        [
            item["duplicate_cluster_id"],
            item["size"],
            json.dumps(item["labels"], sort_keys=True),
            json.dumps(item["sources"], sort_keys=True),
        ]
        for item in summary["largest_clusters"]
    ]
    body = [
        f"# Duplicate Cluster Summary ({summary['mode']})",
        "",
        "Clusters are duplicate-screening groups, not proof that every member is the same image.",
        "",
        markdown_table(
            ["Metric", "Value"],
            [
                ["mode", summary["mode"]],
                ["dhash_threshold", summary["dhash_threshold"]],
                ["row_count", summary["row_count"]],
                ["dhash_coverage_rows", summary["dhash_coverage_rows"]],
                ["file_sha256_coverage_rows", summary["file_sha256_coverage_rows"]],
                ["cluster_count", summary["cluster_count"]],
                ["non_singleton_cluster_count", summary["non_singleton_cluster_count"]],
                ["max_cluster_size", summary["max_cluster_size"]],
                ["edge_count", summary["edge_count"]],
                ["exact_file_sha_edges", summary["exact_file_sha_edges"]],
                ["dhash_edges", summary["dhash_edges"]],
            ],
        ),
        "",
        "## Edge Category Counts",
        "",
        markdown_table(["Category", "Count"], [[k, v] for k, v in summary["edge_category_counts"].items()]),
        "",
        "## Cluster Size Distribution",
        "",
        markdown_table(["Cluster size", "Cluster count"], [[k, v] for k, v in summary["cluster_size_distribution"].items()]),
        "",
        "## Largest Clusters",
        "",
        markdown_table(["Cluster", "Size", "Labels", "Sources"], largest_rows),
    ]
    write_text(path, "\n".join(body) + "\n")


def write_manifest(path: Path, *, command: str, input_path: Path, out_jsonl: Path, clusters_csv: Path, summary: dict[str, Any]) -> None:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_hash(),
        "branch": git_branch_name(),
        "command": command,
        "input_manifest": manifest_for_paths([input_path]),
        "output_manifest": manifest_for_paths([out_jsonl, clusters_csv]),
        "feature_set_id": "optical_12_baseline",
        "cluster_policy": "connected components over exact file SHA edges and full-dataset dHash screening edges",
        "summary": summary,
        "versions": package_versions(),
    }
    manifest["manifest_hash"] = manifest_hash(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build full-dataset duplicate clusters for metadata-enriched JSONL.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--mode", choices=["strict", "conservative"], required=True)
    p.add_argument("--dhash-threshold", type=int, required=True)
    p.add_argument("--out-jsonl", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--clusters-csv", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_jsonl(args.input)
    clustered_rows, summary, edges = build_duplicate_clusters(rows, mode=args.mode, dhash_threshold=args.dhash_threshold)
    write_jsonl(args.out_jsonl, clustered_rows)
    write_edges_csv(args.clusters_csv, edges, clustered_rows)
    write_summary(args.summary, summary)
    write_manifest(
        args.manifest,
        command=" ".join([sys.executable, *sys.argv]),
        input_path=args.input,
        out_jsonl=args.out_jsonl,
        clusters_csv=args.clusters_csv,
        summary=summary,
    )
    print(f"Wrote {args.out_jsonl}")
    print(f"Wrote {args.summary}")
    print(f"Clusters: {summary['cluster_count']} edges: {summary['edge_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
