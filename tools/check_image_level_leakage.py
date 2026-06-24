#!/usr/bin/env python3
"""Image-level leakage checks for metadata-enriched SmartBinocular JSONL."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ml_metadata_utils import (
    effective_label,
    hamming_distance_hex,
    markdown_table,
    optical_feature_hash,
    read_jsonl,
    row_sha256,
    write_text,
)


def _index_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[tuple[int, dict[str, Any]]]]:
    out: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        value = row.get(key)
        if value not in (None, ""):
            out[str(value)].append((idx, row))
    return out


def find_near_duplicate_pairs(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    threshold: int,
) -> list[dict[str, Any]]:
    train_hashes = [(idx, row, str(row["dhash"])) for idx, row in enumerate(train_rows) if row.get("dhash")]
    test_hashes = [(idx, row, str(row["dhash"])) for idx, row in enumerate(test_rows) if row.get("dhash")]
    pairs: list[dict[str, Any]] = []
    for train_idx, train_row, train_hash in train_hashes:
        for test_idx, test_row, test_hash in test_hashes:
            distance = hamming_distance_hex(train_hash, test_hash)
            if distance <= threshold:
                pairs.append(
                    {
                        "method": "dhash",
                        "threshold": threshold,
                        "hamming_distance": distance,
                        "train_index": train_idx,
                        "test_index": test_idx,
                        "train_label": effective_label(train_row),
                        "test_label": effective_label(test_row),
                        "train_source": train_row.get("source", "unknown"),
                        "test_source": test_row.get("source", "unknown"),
                        "train_path": train_row.get("original_image_path", ""),
                        "test_path": test_row.get("original_image_path", ""),
                        "train_dhash": train_hash,
                        "test_dhash": test_hash,
                    }
                )
    return pairs


def _pair_rows_by_key(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    key: str,
    pair_type: str,
) -> list[dict[str, Any]]:
    train_index = _index_by(train_rows, key)
    pairs: list[dict[str, Any]] = []
    for test_idx, test_row in enumerate(test_rows):
        value = test_row.get(key)
        if value in (None, ""):
            continue
        for train_idx, train_row in train_index.get(str(value), []):
            pairs.append(
                {
                    "pair_type": pair_type,
                    "key": key,
                    "value": value,
                    "train_index": train_idx,
                    "test_index": test_idx,
                    "train_label": effective_label(train_row),
                    "test_label": effective_label(test_row),
                    "train_source": train_row.get("source", "unknown"),
                    "test_source": test_row.get("source", "unknown"),
                    "train_metadata_status": train_row.get("metadata_status", ""),
                    "test_metadata_status": test_row.get("metadata_status", ""),
                }
            )
    return pairs


def _feature_hint_pairs(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    train_index: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for idx, row in enumerate(train_rows):
        fp = optical_feature_hash(row)
        if fp:
            train_index[fp].append((idx, row))
    pairs: list[dict[str, Any]] = []
    for test_idx, test_row in enumerate(test_rows):
        fp = optical_feature_hash(test_row)
        if not fp:
            continue
        for train_idx, train_row in train_index.get(fp, []):
            pairs.append(
                {
                    "pair_type": "feature_vector_weak_hint",
                    "key": "optical_12_feature_hash",
                    "value": fp,
                    "train_index": train_idx,
                    "test_index": test_idx,
                    "train_label": effective_label(train_row),
                    "test_label": effective_label(test_row),
                    "train_source": train_row.get("source", "unknown"),
                    "test_source": test_row.get("source", "unknown"),
                    "train_metadata_status": train_row.get("metadata_status", ""),
                    "test_metadata_status": test_row.get("metadata_status", ""),
                }
            )
    return pairs


def check_leakage(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    dhash_threshold: int,
) -> dict[str, Any]:
    exact_row_pairs: list[dict[str, Any]] = []
    train_row_hashes: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for idx, row in enumerate(train_rows):
        train_row_hashes[row_sha256(row)].append((idx, row))
    for test_idx, test_row in enumerate(test_rows):
        for train_idx, train_row in train_row_hashes.get(row_sha256(test_row), []):
            exact_row_pairs.append(
                {
                    "pair_type": "exact_json_row",
                    "key": "row_sha256",
                    "value": row_sha256(test_row),
                    "train_index": train_idx,
                    "test_index": test_idx,
                    "train_label": effective_label(train_row),
                    "test_label": effective_label(test_row),
                    "train_source": train_row.get("source", "unknown"),
                    "test_source": test_row.get("source", "unknown"),
                    "train_metadata_status": train_row.get("metadata_status", ""),
                    "test_metadata_status": test_row.get("metadata_status", ""),
                }
            )
    pair_sets = {
        "exact_json_row": exact_row_pairs,
        "file_sha256": _pair_rows_by_key(train_rows, test_rows, "file_sha256", "file_sha256_overlap"),
        "split_group_id": _pair_rows_by_key(train_rows, test_rows, "split_group_id", "split_group_id_overlap"),
        "duplicate_cluster_id": _pair_rows_by_key(
            train_rows,
            test_rows,
            "duplicate_cluster_id",
            "duplicate_cluster_id_overlap",
        ),
        "session_id": _pair_rows_by_key(train_rows, test_rows, "session_id", "session_overlap"),
        "feature_vector_weak_hint": _feature_hint_pairs(train_rows, test_rows),
    }
    near_pairs = find_near_duplicate_pairs(train_rows, test_rows, threshold=dhash_threshold)
    source_overlap = sorted(
        {str(row.get("source")) for row in train_rows if row.get("source")}
        & {str(row.get("source")) for row in test_rows if row.get("source")}
    )
    coverage = {
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "train_file_sha256_rows": sum(1 for row in train_rows if row.get("file_sha256")),
        "test_file_sha256_rows": sum(1 for row in test_rows if row.get("file_sha256")),
        "train_dhash_rows": sum(1 for row in train_rows if row.get("dhash")),
        "test_dhash_rows": sum(1 for row in test_rows if row.get("dhash")),
        "dhash_threshold": dhash_threshold,
        "source_overlap_count": len(source_overlap),
        "source_overlap": source_overlap,
        "duplicate_cluster_id_overlap_pairs": len(pair_sets["duplicate_cluster_id"]),
    }
    return {"coverage": coverage, "pair_sets": pair_sets, "near_duplicate_pairs": near_pairs}


def write_pairs_csv(path: Path, pairs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for pair in pairs for key in pair.keys()}) if pairs else ["pair_type"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            writer.writerow(pair)


def write_near_pairs_md(path: Path, pairs: list[dict[str, Any]]) -> None:
    rows = [
        [
            pair["train_index"],
            pair["test_index"],
            pair["hamming_distance"],
            pair["train_label"],
            pair["test_label"],
            pair["train_source"],
            pair["test_source"],
            pair.get("train_path", ""),
            pair.get("test_path", ""),
        ]
        for pair in pairs[:200]
    ]
    body = [
        "# dHash Near-Duplicate Pairs",
        "",
        "dHash is a screening method only. Absence of rows here is not proof that no visual duplicates exist.",
        "",
        markdown_table(
            [
                "Train index",
                "Test index",
                "Hamming",
                "Train label",
                "Test label",
                "Train source",
                "Test source",
                "Train path",
                "Test path",
            ],
            rows,
        ),
    ]
    write_text(path, "\n".join(body) + "\n")


def write_summary(path: Path, result: dict[str, Any]) -> None:
    coverage = result["coverage"]
    pair_sets = result["pair_sets"]
    rows = [
        ["train_rows", coverage["train_rows"]],
        ["test_rows", coverage["test_rows"]],
        ["train_file_sha256_rows", coverage["train_file_sha256_rows"]],
        ["test_file_sha256_rows", coverage["test_file_sha256_rows"]],
        ["train_dhash_rows", coverage["train_dhash_rows"]],
        ["test_dhash_rows", coverage["test_dhash_rows"]],
        ["dhash_threshold", coverage["dhash_threshold"]],
        ["source_overlap_count", coverage["source_overlap_count"]],
        ["exact_json_row_pairs", len(pair_sets["exact_json_row"])],
        ["file_sha256_overlap_pairs", len(pair_sets["file_sha256"])],
        ["split_group_id_overlap_pairs", len(pair_sets["split_group_id"])],
        ["duplicate_cluster_id_overlap_pairs", len(pair_sets["duplicate_cluster_id"])],
        ["session_overlap_pairs", len(pair_sets["session_id"])],
        ["feature_vector_weak_hint_pairs", len(pair_sets["feature_vector_weak_hint"])],
        ["dhash_near_duplicate_pairs", len(result["near_duplicate_pairs"])],
    ]
    claim = (
        "No absolute visual-duplicate claim is made. If dHash near-duplicate pairs are zero, "
        "that means only that no dHash pairs were found within records with dHash coverage at the configured threshold."
    )
    body = [
        "# Image-Level Leakage Summary",
        "",
        "Status: preliminary until reviewed with raw-image coverage and split policy.",
        "",
        claim,
        "",
        "Feature-vector/hash matching is a weak hint only and is not used as original-image identity evidence.",
        "",
        "## Coverage and Pair Counts",
        "",
        markdown_table(["Metric", "Value"], rows),
        "",
        "## Source Overlap",
        "",
        ", ".join(coverage["source_overlap"]) if coverage["source_overlap"] else "No source-name overlap.",
    ]
    write_text(path, "\n".join(body) + "\n")


def write_remaining_risks(path: Path, result: dict[str, Any]) -> None:
    coverage = result["coverage"]
    pair_sets = result["pair_sets"]
    risk_rows = [
        [
            "exact_json_row",
            len(pair_sets["exact_json_row"]),
            "Definite JSON record overlap if nonzero.",
        ],
        [
            "file_sha256",
            len(pair_sets["file_sha256"]),
            "Definite original-file overlap within file SHA coverage if nonzero.",
        ],
        [
            "split_group_id",
            len(pair_sets["split_group_id"]),
            "Group split violation if nonzero.",
        ],
        [
            "duplicate_cluster_id",
            len(pair_sets["duplicate_cluster_id"]),
            "Duplicate-cluster split violation if nonzero.",
        ],
        [
            "dhash_screening",
            len(result["near_duplicate_pairs"]),
            "Screening only; zero only means no pairs within dHash-covered rows at this threshold.",
        ],
        [
            "source_name_overlap",
            coverage["source_overlap_count"],
            "Not leakage by itself, but prevents source-held-out wording if nonzero.",
        ],
    ]
    body = [
        "# Cluster-Aware Remaining Risks",
        "",
        "No absolute visual-duplicate claim is made. dHash is a screening method only.",
        "",
        markdown_table(["Risk", "Count", "Interpretation"], risk_rows),
        "",
        "## Source Overlap",
        "",
        ", ".join(coverage["source_overlap"]) if coverage["source_overlap"] else "No source-name overlap.",
    ]
    write_text(path, "\n".join(body) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check image-level leakage on metadata-enriched JSONL.")
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--dhash-threshold", type=int, default=4)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--pairs-csv", type=Path, required=True)
    p.add_argument("--near-pairs-md", type=Path, required=True)
    p.add_argument("--near-pairs-csv", type=Path, required=True)
    p.add_argument("--remaining-risks", type=Path)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_rows = read_jsonl(args.train)
    test_rows = read_jsonl(args.test)
    result = check_leakage(train_rows, test_rows, dhash_threshold=args.dhash_threshold)
    all_pairs: list[dict[str, Any]] = []
    for pairs in result["pair_sets"].values():
        all_pairs.extend(pairs)
    write_pairs_csv(args.pairs_csv, all_pairs)
    write_pairs_csv(args.near_pairs_csv, result["near_duplicate_pairs"])
    write_near_pairs_md(args.near_pairs_md, result["near_duplicate_pairs"])
    write_summary(args.summary, result)
    if args.remaining_risks:
        write_remaining_risks(args.remaining_risks, result)
    print(f"Wrote {args.summary}")
    print(f"Wrote {args.pairs_csv}")
    print(f"Wrote {args.near_pairs_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
