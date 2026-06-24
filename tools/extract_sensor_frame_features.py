#!/usr/bin/env python3
"""Extract optical_12_baseline feature JSONL from sampled raw sensor frames."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smartbinocular.feature_extractor import FeatureExtractor
from smartbinocular.utils import build_frame_cache
from tools.ml_metadata_utils import markdown_table, write_text


def extract_feature_record_for_frame(
    *,
    frame_path: Path,
    video_id: str,
    frame_idx: int,
    timestamp_sec: float,
    modality_guess: str,
) -> dict[str, Any]:
    img = cv.imread(str(frame_path))
    if img is None:
        data = np.fromfile(str(frame_path), dtype=np.uint8)
        img = cv.imdecode(data, cv.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode frame: {frame_path}")
    extractor = FeatureExtractor()
    # FeatureRecord.ts is a Unix timestamp for hour-of-day encoding. Raw video
    # manifests only provide relative timestamps, so use 0.0 and preserve the
    # relative time separately as timestamp_sec.
    cache = build_frame_cache(img, np.zeros((62, 80), dtype=np.uint8), ts=0.0, skip_nir_160=True)
    rec = extractor.extract(
        cache,
        nir_channel="unknown",
        thermal_channel="none",
        ts=0.0,
        source="raw_sensor_capture",
        frame_idx=int(frame_idx),
        has_motion=False,
        has_temporal=False,
        label=None,
        label_source="unlabeled_sensor_raw",
        label_confidence=None,
    ).to_dict()
    rec.update(
        {
            "video_id": video_id,
            "timestamp_sec": float(timestamp_sec),
            "frame_path": str(frame_path),
            "original_image_path": str(frame_path),
            "relative_image_id": f"{video_id}/{frame_path.name}",
            "source_dataset": "raw_sensor_capture",
            "capture_device": "unknown",
            "modality_guess": modality_guess,
            "metadata_status": "verified",
            "metadata_verification": "sampled_frame_file",
            "label": None,
            "label_source": "unlabeled_sensor_raw",
            "label_confidence": None,
            "nir_channel": "unknown",
            "thermal_channel": "none",
        }
    )
    return rec


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(path: Path, processed: int, skipped: list[str]) -> None:
    body = [
        "# Raw Sensor Feature Summary",
        "",
        "Status: unlabeled raw sensor features. These are not sensor-real accuracy labels.",
        "",
        markdown_table(
            ["Metric", "Value"],
            [
                ["feature_set", "optical_12_baseline"],
                ["processed_frames", processed],
                ["skipped_frames", len(skipped)],
                ["label_source", "unlabeled_sensor_raw"],
                ["modality", "unknown optical"],
            ],
        ),
    ]
    if skipped:
        body.extend(["", "## Skipped", "", markdown_table(["Reason"], [[s] for s in skipped[:100]])])
    write_text(path, "\n".join(body) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract optical_12_baseline features from sampled sensor frames.")
    p.add_argument("--frames-manifest", type=Path, required=True)
    p.add_argument("--out-jsonl", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--feature-set", default="optical_12_baseline", choices=["optical_12_baseline"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records: list[dict[str, Any]] = []
    skipped: list[str] = []
    for row in read_manifest(args.frames_manifest):
        try:
            records.append(
                extract_feature_record_for_frame(
                    frame_path=Path(row["frame_path"]),
                    video_id=row.get("video_id") or Path(row["video_path"]).stem,
                    frame_idx=int(row["frame_idx"]),
                    timestamp_sec=float(row["timestamp_sec"]),
                    modality_guess=row.get("modality_guess") or "unknown optical",
                )
            )
        except Exception as exc:
            skipped.append(f"{row.get('frame_path', '')}: {exc}")
    write_jsonl(args.out_jsonl, records)
    write_summary(args.summary, len(records), skipped)
    print(f"Wrote {args.out_jsonl}")
    print(f"Wrote {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
