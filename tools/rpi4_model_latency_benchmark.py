#!/usr/bin/env python3
"""RPi4-ready latency benchmark for SmartBinocular feature extraction and model inference."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.extract_sensor_frame_features import extract_feature_record_for_frame
from tools.ml_metadata_utils import git_branch_name, git_commit_hash, package_versions, read_jsonl, write_text


MEASUREMENT_KEYS = [
    "feature_extraction_latency_ms",
    "model_inference_latency_ms",
    "feature_plus_predict_latency_ms",
    "model_load_time_ms",
    "model_size_bytes",
    "ram_peak_bytes",
]


def build_manifest(*, model_path: Path, feature_source: str, repeats: int, hardware_label: str) -> dict[str, Any]:
    return {
        "git_commit": git_commit_hash(),
        "git_branch": git_branch_name(),
        "model_path": str(model_path),
        "feature_source": feature_source,
        "repeats": repeats,
        "hardware_label": hardware_label,
        "metric_status": "requires RPi4 run for deployment claim",
        "versions": package_versions(),
        "measurements": {
            "feature_extraction_latency_ms": None,
            "model_inference_latency_ms": None,
            "feature_plus_predict_latency_ms": None,
            "model_load_time_ms": None,
            "model_size_bytes": model_path.stat().st_size if model_path.exists() else None,
            "ram_peak_bytes": None,
        },
    }


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": math.nan, "median": math.nan, "p95": math.nan, "runs": 0}
    arr = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(arr)), "median": float(np.median(arr)), "p95": float(np.percentile(arr, 95)), "runs": int(len(values))}


def _read_frame_manifest(path: Path, limit: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit > 0 else rows


def _matrix(rows: list[dict[str, Any]], feature_set: list[str]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in feature_set] for row in rows], dtype=np.float32)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    manifest = build_manifest(
        model_path=args.model,
        feature_source=str(args.features or args.frames_manifest or "none"),
        repeats=args.repeats,
        hardware_label=args.hardware_label,
    )
    tracemalloc.start()
    load_start = time.perf_counter()
    bundle = joblib.load(args.model)
    manifest["measurements"]["model_load_time_ms"] = (time.perf_counter() - load_start) * 1000.0
    feature_set = list(bundle["feature_set"])
    scaler = bundle["scalers"][args.scaler_group]
    model = bundle["rf"]

    feature_rows = read_jsonl(args.features) if args.features else []
    if feature_rows:
        X = scaler.transform(_matrix(feature_rows, feature_set))
        inference_vals = []
        sample = X[:1]
        for _ in range(args.repeats):
            start = time.perf_counter()
            model.predict(sample)
            inference_vals.append((time.perf_counter() - start) * 1000.0)
        manifest["measurements"]["model_inference_latency_ms"] = _summary(inference_vals)

    if args.frames_manifest:
        frames = _read_frame_manifest(args.frames_manifest, args.max_frames)
        feature_vals = []
        combined_vals = []
        for item in frames[: args.repeats]:
            frame_path = Path(item["frame_path"])
            start = time.perf_counter()
            record = extract_feature_record_for_frame(
                frame_path=frame_path,
                video_id=item.get("video_id") or Path(item.get("video_path", "video")).stem,
                frame_idx=int(item.get("frame_idx") or 0),
                timestamp_sec=float(item.get("timestamp_sec") or 0.0),
                modality_guess=item.get("modality_guess") or "unknown optical",
            )
            extracted_ms = (time.perf_counter() - start) * 1000.0
            feature_vals.append(extracted_ms)
            start = time.perf_counter()
            X = scaler.transform(_matrix([record], feature_set))
            model.predict(X)
            combined_vals.append(extracted_ms + ((time.perf_counter() - start) * 1000.0))
        manifest["measurements"]["feature_extraction_latency_ms"] = _summary(feature_vals)
        manifest["measurements"]["feature_plus_predict_latency_ms"] = _summary(combined_vals)

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    manifest["measurements"]["ram_peak_bytes"] = peak
    return manifest


def write_summary(path: Path, manifest: dict[str, Any]) -> None:
    rows = []
    for key in MEASUREMENT_KEYS:
        value = manifest["measurements"].get(key)
        rows.append([key, json.dumps(value, sort_keys=True)])
    write_text(
        path,
        "\n".join(
            [
                "# RPi4 Model Latency Summary",
                "",
                "Status: run this on Raspberry Pi 4 CPU before claiming the 20 FPS deployment target. Non-RPi runs are proxy evidence only.",
                "",
                "| Measurement | Value |",
                "| --- | --- |",
                *[f"| {k} | `{v}` |" for k, v in rows],
            ]
        )
        + "\n",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark feature extraction and model latency on target hardware.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--features", type=Path, help="Existing feature JSONL for model-only latency.")
    parser.add_argument("--frames-manifest", type=Path, help="Frame manifest for feature extraction and feature+predict latency.")
    parser.add_argument("--scaler-group", default="rgb")
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--hardware-label", default="Raspberry Pi 4 CPU")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_benchmark(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    write_summary(args.summary, manifest)
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
