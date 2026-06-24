#!/usr/bin/env python3
"""Evaluate no-reference image quality metrics for existing local artifacts."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from fusion_eval_manifest import markdown_table, read_csv_rows, run_manifest, write_csv, write_json, write_markdown  # type: ignore[import]
from fusion_eval_metrics import compute_image_metrics, summarize_metric_rows  # type: ignore[import]


METRICS = [
    "brightness_mean",
    "brightness_p5",
    "brightness_p50",
    "brightness_p95",
    "rms_contrast",
    "log_rms_contrast",
    "entropy",
    "laplacian_variance",
    "tenengrad",
    "edge_density",
    "pct_dark_clipped",
    "pct_highlight_saturated",
    "noise_proxy",
    "spatial_frequency",
    "average_gradient",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute no-reference IQA metrics for NIR, thermal, and fusion artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image-folder", type=Path, action="append", default=None)
    parser.add_argument("--pairing-manifest", type=Path, default=Path("artifacts/fusion_eval/pairing_manifest.csv"))
    parser.add_argument("--thermal-seq-dir", type=Path, default=Path("data/eval/thermal_seq"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/fusion_eval"))
    parser.add_argument("--summary-dir", type=Path, default=Path("docs/tables/fusion"))
    parser.add_argument("--max-thermal-frames-per-seq", type=int, default=3)
    return parser.parse_args()


def _read_image(path: Path) -> np.ndarray | None:
    return cv.imread(str(path), cv.IMREAD_COLOR)


def _add_image_row(rows: list[dict[str, object]], *, path: Path, algorithm: str, evidence_label: str, source: str, condition: str) -> None:
    img = _read_image(path)
    if img is None:
        return
    metrics = compute_image_metrics(img)
    rows.append(
        {
            "path": str(path),
            "algorithm": algorithm,
            "evidence_label": evidence_label,
            "bucket_or_condition": condition,
            "source_or_session": source,
            **{k: round(float(v), 6) for k, v in metrics.items() if v is not None},
        }
    )


def _collect_from_manifest(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pair in read_csv_rows(path):
        source = pair.get("source_or_session", "")
        condition = pair.get("bucket_or_condition", "")
        evidence = pair.get("evidence_label", "") or "unknown"
        for column, algorithm in (
            ("nir_path", "nir_capture"),
            ("thermal_path", "thermal_capture"),
            ("fusion_path", "fusion_capture"),
        ):
            value = pair.get(column, "")
            if value and Path(value).exists():
                _add_image_row(rows, path=Path(value), algorithm=algorithm, evidence_label=evidence, source=source, condition=condition)
    return rows


def _collect_from_folders(folders: list[Path] | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for folder in folders or []:
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                continue
            _add_image_row(
                rows,
                path=path,
                algorithm="folder_image",
                evidence_label="unknown",
                source=str(folder),
                condition="",
            )
    return rows


def _collect_thermal_sequences(seq_dir: Path, max_frames: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not seq_dir.exists():
        return rows
    for seq_path in sorted(seq_dir.glob("*.npy")):
        arr = np.load(seq_path)
        if arr.ndim == 2:
            frames = [arr]
        elif arr.ndim == 3:
            indices = np.linspace(0, arr.shape[0] - 1, min(max_frames, arr.shape[0]), dtype=int)
            frames = [arr[int(i)] for i in indices]
        else:
            continue
        for idx, frame in enumerate(frames):
            metrics = compute_image_metrics(frame)
            rows.append(
                {
                    "path": f"{seq_path}#frame={idx}",
                    "algorithm": "thermal_sequence_raw",
                    "evidence_label": "proxy",
                    "bucket_or_condition": "thermal_sequence",
                    "source_or_session": str(seq_path.parent),
                    **{k: round(float(v), 6) for k, v in metrics.items() if v is not None},
                }
            )
    return rows


def _summary_rows(rows: list[dict[str, object]], modality_filter: str) -> list[dict[str, object]]:
    filtered = [r for r in rows if modality_filter in str(r.get("algorithm", ""))]
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in filtered:
        for metric in METRICS:
            if metric in row:
                key = (
                    str(row.get("algorithm", "")),
                    str(row.get("evidence_label", "")),
                    str(row.get("bucket_or_condition", "")),
                    str(row.get("source_or_session", "")),
                    metric,
                )
                grouped[key].append({"value": row[metric]})
    out = []
    for (algorithm, evidence, condition, source, metric), values in sorted(grouped.items()):
        s = summarize_metric_rows(values)
        out.append(
            {
                "algorithm": algorithm,
                "evidence_label": evidence,
                "bucket_or_condition": condition,
                "source_or_session": source,
                "metric": metric,
                **{k: "" if v is None else round(float(v), 6) for k, v in s.items()},
            }
        )
    return out


def _write_summary(path: Path, title: str, rows: list[dict[str, object]]) -> None:
    columns = [
        "algorithm",
        "evidence_label",
        "bucket_or_condition",
        "metric",
        "n",
        "mean",
        "median",
        "std",
        "p25",
        "p75",
        "p95",
        "bootstrap_ci95_low",
        "bootstrap_ci95_high",
    ]
    body = "Tier 2 no-reference IQA metrics unless explicitly noted. These are not proof of real task quality.\n\n"
    body += markdown_table(rows[:200], columns)
    write_markdown(path, title, body)


def main() -> None:
    args = parse_args()
    out_dir = (ROOT / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    summary_dir = (ROOT / args.summary_dir).resolve() if not args.summary_dir.is_absolute() else args.summary_dir
    manifest_path = (ROOT / args.pairing_manifest).resolve() if not args.pairing_manifest.is_absolute() else args.pairing_manifest
    seq_dir = (ROOT / args.thermal_seq_dir).resolve() if not args.thermal_seq_dir.is_absolute() else args.thermal_seq_dir

    rows = []
    rows.extend(_collect_from_manifest(manifest_path))
    rows.extend(_collect_from_folders(args.image_folder))
    rows.extend(_collect_thermal_sequences(seq_dir, args.max_thermal_frames_per_seq))

    write_csv(out_dir / "image_quality_metrics.csv", rows)
    nir_rows = _summary_rows(rows, "nir")
    thermal_rows = _summary_rows(rows, "thermal")
    fusion_rows = _summary_rows(rows, "fusion")
    write_csv(out_dir / "nir_quality_summary.csv", nir_rows)
    write_csv(out_dir / "thermal_quality_summary.csv", thermal_rows)
    write_csv(out_dir / "fusion_quality_summary.csv", fusion_rows)
    _write_summary(summary_dir / "nir_quality_summary.md", "NIR Quality Summary", nir_rows)
    _write_summary(summary_dir / "thermal_quality_summary.md", "Thermal Quality Summary", thermal_rows)
    _write_summary(summary_dir / "fusion_quality_summary.md", "Fusion Quality Summary", fusion_rows)
    write_json(
        out_dir / "image_quality_run_manifest.json",
        run_manifest(
            "python3 tools/evaluate_image_quality.py",
            [str(manifest_path), str(seq_dir)],
            {"max_thermal_frames_per_seq": args.max_thermal_frames_per_seq},
        ),
    )
    print(f"image_metric_rows={len(rows)}")


if __name__ == "__main__":
    main()
