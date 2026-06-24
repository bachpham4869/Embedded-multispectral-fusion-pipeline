#!/usr/bin/env python3
"""Analyze Phase 3 dHash screening pairs without treating them as proof of leakage."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ml_metadata_utils import effective_label, markdown_table, read_jsonl, write_text

LOW_SUPPORT_LABELS = {"glare", "backlight", "transition"}


def _has_verified_identity(row: dict[str, Any]) -> bool:
    return str(row.get("metadata_status", "")) == "verified" and any(
        row.get(key) for key in ("file_sha256", "original_image_path", "relative_image_id")
    )


def classify_pair_category(train_row: dict[str, Any], test_row: dict[str, Any], *, hamming_distance: int) -> str:
    """Classify a dHash pair conservatively.

    dHash matches are screening evidence. Only exact file SHA equality is a
    definite duplicate; all other categories are weaker evidence.
    """

    train_sha = train_row.get("file_sha256")
    test_sha = test_row.get("file_sha256")
    if train_sha and test_sha and str(train_sha) == str(test_sha):
        return "exact_file_duplicate"
    if not (_has_verified_identity(train_row) and _has_verified_identity(test_row)):
        return "unresolved"
    if effective_label(train_row) != effective_label(test_row):
        return "dhash_false_positive_candidate"
    if hamming_distance <= 2 or train_row.get("source") == test_row.get("source"):
        return "likely_near_duplicate"
    return "likely_near_duplicate"


def _load_pair_rows(pairs_path: Path, train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    analyzed: list[dict[str, Any]] = []
    with pairs_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for pair in reader:
            train_idx = int(pair["train_index"])
            test_idx = int(pair["test_index"])
            train_row = train_rows[train_idx]
            test_row = test_rows[test_idx]
            distance = int(pair.get("hamming_distance", 0))
            train_sha = train_row.get("file_sha256") or ""
            test_sha = test_row.get("file_sha256") or ""
            train_group = train_row.get("split_group_id") or ""
            test_group = test_row.get("split_group_id") or ""
            category = classify_pair_category(train_row, test_row, hamming_distance=distance)
            analyzed.append(
                {
                    "split_side": "train-test",
                    "train_index": train_idx,
                    "test_index": test_idx,
                    "hamming_distance": distance,
                    "category": category,
                    "train_label": effective_label(train_row),
                    "test_label": effective_label(test_row),
                    "label_pair": f"{effective_label(train_row)} -> {effective_label(test_row)}",
                    "train_source": train_row.get("source", "unknown"),
                    "test_source": test_row.get("source", "unknown"),
                    "source_pair": f"{train_row.get('source', 'unknown')} -> {test_row.get('source', 'unknown')}",
                    "train_metadata_status": train_row.get("metadata_status", ""),
                    "test_metadata_status": test_row.get("metadata_status", ""),
                    "metadata_status_pair": f"{train_row.get('metadata_status', '')} -> {test_row.get('metadata_status', '')}",
                    "train_file_sha256": train_sha,
                    "test_file_sha256": test_sha,
                    "file_sha256_relation": "same" if train_sha and train_sha == test_sha else "different_or_missing",
                    "train_split_group_id": train_group,
                    "test_split_group_id": test_group,
                    "split_group_relation": "same" if train_group and train_group == test_group else "different_or_missing",
                    "train_path": train_row.get("original_image_path") or train_row.get("candidate_original_image_path") or "",
                    "test_path": test_row.get("original_image_path") or test_row.get("candidate_original_image_path") or "",
                    "train_dhash": train_row.get("dhash", pair.get("train_dhash", "")),
                    "test_dhash": test_row.get("dhash", pair.get("test_dhash", "")),
                    "low_support_pair": bool(
                        effective_label(train_row) in LOW_SUPPORT_LABELS or effective_label(test_row) in LOW_SUPPORT_LABELS
                    ),
                }
            )
    return analyzed


def write_analysis_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else ["category"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _counter_rows(counter: Counter[Any], limit: int = 20) -> list[list[Any]]:
    return [[key, value] for key, value in counter.most_common(limit)]


def write_analysis_md(path: Path, rows: list[dict[str, Any]]) -> None:
    category_counts = Counter(row["category"] for row in rows)
    distance_counts = Counter(str(row["hamming_distance"]) for row in rows)
    label_counts = Counter(row["label_pair"] for row in rows)
    source_counts = Counter(row["source_pair"] for row in rows)
    status_counts = Counter(row["metadata_status_pair"] for row in rows)
    low_support_count = sum(1 for row in rows if row["low_support_pair"])
    body = [
        "# dHash Pair Analysis",
        "",
        "dHash pairs are screening evidence only. This table does not classify all dHash pairs as confirmed leakage.",
        "",
        "## Summary",
        "",
        markdown_table(
            ["Metric", "Value"],
            [
                ["total_pairs", len(rows)],
                ["low_support_label_pairs", low_support_count],
                ["exact_file_duplicate", category_counts.get("exact_file_duplicate", 0)],
                ["likely_near_duplicate", category_counts.get("likely_near_duplicate", 0)],
                ["dhash_false_positive_candidate", category_counts.get("dhash_false_positive_candidate", 0)],
                ["unresolved", category_counts.get("unresolved", 0)],
            ],
        ),
        "",
        "## Category Counts",
        "",
        markdown_table(["Category", "Count"], _counter_rows(category_counts)),
        "",
        "## Distance Counts",
        "",
        markdown_table(["Hamming distance", "Count"], _counter_rows(distance_counts)),
        "",
        "## Top Label Pairs",
        "",
        markdown_table(["Label pair", "Count"], _counter_rows(label_counts)),
        "",
        "## Top Source Pairs",
        "",
        markdown_table(["Source pair", "Count"], _counter_rows(source_counts)),
        "",
        "## Metadata Status Pairs",
        "",
        markdown_table(["Metadata status pair", "Count"], _counter_rows(status_counts)),
    ]
    write_text(path, "\n".join(body) + "\n")


def _resolve_image(path_text: str) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path if path.exists() else None


def _read_thumb(path_text: str, label: str, size: tuple[int, int] = (180, 120)) -> np.ndarray:
    canvas = np.full((size[1], size[0], 3), 240, dtype=np.uint8)
    path = _resolve_image(path_text)
    if path is not None:
        img = cv.imread(str(path), cv.IMREAD_COLOR)
        if img is None:
            data = np.fromfile(str(path), dtype=np.uint8)
            img = cv.imdecode(data, cv.IMREAD_COLOR)
        if img is not None:
            h, w = img.shape[:2]
            scale = min(size[0] / max(w, 1), (size[1] - 24) / max(h, 1))
            resized = cv.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv.INTER_AREA)
            y = 0
            x = (size[0] - resized.shape[1]) // 2
            canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    cv.putText(canvas, label[:28], (4, size[1] - 8), cv.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1, cv.LINE_AA)
    return canvas


def _priority(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
    verified = int(row["train_metadata_status"] == "verified" and row["test_metadata_status"] == "verified")
    different_label = int(row["train_label"] != row["test_label"])
    low_support = int(row["low_support_pair"])
    return (-verified, int(row["hamming_distance"]), -different_label, -low_support, 0)


def write_contact_sheets(
    out_dir: Path,
    rows: list[dict[str, Any]],
    *,
    max_contact_sheets: int,
    max_pairs_per_sheet: int,
) -> None:
    if max_contact_sheets <= 0 or max_pairs_per_sheet <= 0:
        return
    selected = sorted(rows, key=_priority)[: max_contact_sheets * max_pairs_per_sheet]
    if not selected:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for sheet_idx in range(max_contact_sheets):
        chunk = selected[sheet_idx * max_pairs_per_sheet : (sheet_idx + 1) * max_pairs_per_sheet]
        if not chunk:
            break
        pair_images: list[np.ndarray] = []
        for row in chunk:
            left = _read_thumb(str(row["train_path"]), f"T {row['train_label']} d={row['hamming_distance']}")
            right = _read_thumb(str(row["test_path"]), f"E {row['test_label']} {row['category'][:8]}")
            pair_images.append(np.concatenate([left, right], axis=1))
        sheet = np.concatenate(pair_images, axis=0)
        cv.imwrite(str(out_dir / f"dhash_pairs_{sheet_idx + 1:02d}.png"), sheet)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze Phase 3 dHash pair diagnostics.")
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--out-md", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--contact-sheet-dir", type=Path)
    p.add_argument("--max-contact-sheets", type=int, default=4)
    p.add_argument("--max-pairs-per-sheet", type=int, default=12)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_rows = read_jsonl(args.train)
    test_rows = read_jsonl(args.test)
    analyzed = _load_pair_rows(args.pairs, train_rows, test_rows)
    write_analysis_csv(args.out_csv, analyzed)
    write_analysis_md(args.out_md, analyzed)
    if args.contact_sheet_dir:
        write_contact_sheets(
            args.contact_sheet_dir,
            analyzed,
            max_contact_sheets=args.max_contact_sheets,
            max_pairs_per_sheet=args.max_pairs_per_sheet,
        )
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
