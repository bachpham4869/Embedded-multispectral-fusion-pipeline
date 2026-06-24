#!/usr/bin/env python3
"""Build non-production optical_v2/optical_21 candidate feature JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.extract_sensor_frame_features import extract_feature_record_for_frame
from tools.ml_metadata_utils import effective_label, markdown_table, read_jsonl, write_text
from tools.optical_candidate_features import compute_candidate_features


def _read_image(path: Path) -> np.ndarray | None:
    img = cv.imread(str(path))
    if img is None and path.is_file():
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv.imdecode(data, cv.IMREAD_COLOR)
    return img


def enrich_row_with_candidates(row: dict[str, Any], *, temporal_brightness_std: float | None = None) -> dict[str, Any] | None:
    path_value = row.get("original_image_path") or row.get("frame_path")
    if not path_value:
        return None
    path = Path(str(path_value))
    if not path.is_absolute():
        path = REPO_ROOT / path
    img = _read_image(path)
    if img is None:
        return None
    enriched = dict(row)
    start = time.perf_counter()
    enriched.update(compute_candidate_features(img, temporal_brightness_std=temporal_brightness_std))
    enriched["feature_extraction_candidate_ms"] = (time.perf_counter() - start) * 1000.0
    enriched["feature_set_candidates"] = ["optical_v2_candidate", "optical_21_candidate_still"]
    if temporal_brightness_std is not None:
        enriched["feature_set_candidates"].append("optical_21_candidate_temporal")
    return enriched


def build_verified_feature_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out: list[dict[str, Any]] = []
    skipped = 0
    for row in rows:
        enriched = enrich_row_with_candidates(row, temporal_brightness_std=None)
        if enriched is None:
            skipped += 1
            continue
        out.append(enriched)
    labels = defaultdict(int)
    for row in out:
        labels[effective_label(row)] += 1
    return out, {"input_rows": len(rows), "output_rows": len(out), "skipped_missing_image": skipped, "label_counts": dict(sorted(labels.items()))}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_frame_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_sensor_v2_rows(frames_manifest: Path) -> list[dict[str, Any]]:
    manifest = _read_frame_manifest(frames_manifest)
    brightness_by_video: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=10))
    rows: list[dict[str, Any]] = []
    for item in manifest:
        frame_path = Path(item["frame_path"])
        base = extract_feature_record_for_frame(
            frame_path=frame_path,
            video_id=item.get("video_id") or Path(item["video_path"]).stem,
            frame_idx=int(item["frame_idx"]),
            timestamp_sec=float(item["timestamp_sec"]),
            modality_guess=item.get("modality_guess") or "unknown optical",
        )
        video_id = str(base["video_id"])
        hist = brightness_by_video[video_id]
        temporal_std = float(np.std(hist)) if len(hist) >= 3 else None
        hist.append(float(base["nir_mean_brightness"]))
        img = _read_image(frame_path)
        if img is None:
            continue
        start = time.perf_counter()
        base.update(compute_candidate_features(img, temporal_brightness_std=temporal_std))
        base["feature_extraction_candidate_ms"] = (time.perf_counter() - start) * 1000.0
        base["feature_set_candidates"] = ["optical_v2_candidate", "optical_21_candidate_still"]
        if temporal_std is not None:
            base["feature_set_candidates"].append("optical_21_candidate_temporal")
        rows.append(base)
    return rows


def write_summary(path: Path, summaries: dict[str, dict[str, Any]], sensor_rows: int) -> None:
    body = [
        "# Feature Candidate Table Build Summary",
        "",
        "Status: non-production feature tables. Production schema and loader are unchanged.",
        "",
        markdown_table(
            ["Split", "Input rows", "Output rows", "Skipped missing image", "Labels"],
            [
                [name, s["input_rows"], s["output_rows"], s["skipped_missing_image"], json.dumps(s["label_counts"], sort_keys=True)]
                for name, s in summaries.items()
            ],
        ),
        "",
        markdown_table(["Sensor candidate rows", "Value"], [["raw_sensor_features_v2", sensor_rows]]),
        "",
        "`temporal_brightness_std` is emitted only when sequential frame history exists. Still-image training rows do not receive zero-imputed temporal values.",
        "`optical_21_candidate_still` excludes temporal-only fields and is emitted for verified still-image rows as a 21-derived still-compatible feature set.",
    ]
    write_text(path, "\n".join(body) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build non-production optical candidate feature JSONL files.")
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--sensor-frames", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--sensor-out", type=Path)
    p.add_argument("--summary", type=Path, default=Path("docs/tables/ml/feature_candidate_build_summary.md"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_rows, train_summary = build_verified_feature_rows(read_jsonl(args.train))
    test_rows, test_summary = build_verified_feature_rows(read_jsonl(args.test))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.out_dir / "cluster_aware_conservative_train_optical_v2_verified.jsonl", train_rows)
    _write_jsonl(args.out_dir / "cluster_aware_conservative_test_optical_v2_verified.jsonl", test_rows)
    sensor_count = 0
    if args.sensor_frames and args.sensor_out:
        sensor_rows = build_sensor_v2_rows(args.sensor_frames)
        _write_jsonl(args.sensor_out, sensor_rows)
        sensor_count = len(sensor_rows)
    write_summary(args.summary, {"train": train_summary, "test": test_summary}, sensor_count)
    print(f"Wrote {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
