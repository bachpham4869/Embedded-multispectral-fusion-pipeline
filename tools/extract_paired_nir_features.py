#!/usr/bin/env python3
"""Extract optical ML features from paired IMX/NIR frames.

This tool is non-production evidence tooling. It reads paired capture videos and
manifests, writes derived JSONL/tables, and never mutates raw data or production
models.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
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
from smartbinocular.feature_schema import ENV_CLASSES
from smartbinocular.utils import build_frame_cache
from tools.ml_metadata_utils import markdown_table, write_jsonl, write_text


PAIRING_TIERS = (
    "frame_strict",
    "time_strict_100ms",
    "protocol_strict_1s",
    "near_paired",
    "weak_paired",
    "unpaired",
)
TRUSTED_LABEL_SOURCES = {"manual", "verified_sidecar", "user_provided", "human_verified"}
MIN_TRUSTED_LABEL_CONFIDENCE = 0.8


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["pair_id"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _repo_rel(path: Path | str) -> str:
    text = str(path)
    if "#" in text:
        base, suffix = text.split("#", 1)
        return _repo_rel(Path(base)) + "#" + suffix
    try:
        return str(Path(text).resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return text


def trusted_label_from_manifest_row(
    row: dict[str, Any],
    *,
    min_confidence: float = MIN_TRUSTED_LABEL_CONFIDENCE,
) -> tuple[str | None, str]:
    label = str(row.get("env_label") or row.get("label") or "").strip()
    source = str(row.get("label_source") or "").strip()
    confidence = _safe_float(row.get("label_confidence"))
    if not label or label == "unknown":
        return None, "missing_or_unknown_label"
    if label not in ENV_CLASSES:
        return None, "invalid_taxonomy_label"
    if source not in TRUSTED_LABEL_SOURCES:
        return None, "untrusted_label_source"
    if confidence is None or confidence < min_confidence:
        return None, "low_or_invalid_label_confidence"
    return label, "trusted_label"


def _nir_modality(row: dict[str, Any]) -> str:
    explicit = str(row.get("nir_modality") or "").strip()
    if explicit:
        return explicit
    raw = str(row.get("nir_raw_path") or row.get("imx_video_path") or "").lower()
    if "imx" in raw or "nir" in raw:
        return "unknown_optical"
    return "unknown_optical"


def _thermal_channel(row: dict[str, Any]) -> str:
    modality = str(row.get("thermal_modality") or "").strip()
    if modality == "raw_numeric_thermal":
        return "lwir"
    if modality == "display_heatmap_like":
        return "thermal_display"
    return "unknown_thermal"


def _evidence_type(row: dict[str, Any], trusted_label: str | None) -> str:
    if trusted_label:
        return "labeled"
    source = str(row.get("label_source") or "")
    if "weak" in source:
        return "weak_label"
    if source and source not in {"none", "unknown", ""}:
        return "proxy"
    return "unlabeled"


def feature_record_from_pair_frame(bgr: np.ndarray, row: dict[str, Any]) -> dict[str, Any]:
    if bgr is None or bgr.size == 0:
        raise ValueError("paired frame is empty")
    trusted_label, label_status = trusted_label_from_manifest_row(row)
    frame_idx = int(float(row.get("frame_idx") or row.get("video_frame_index") or 0))
    timestamp_sec = _safe_float(row.get("timestamp_sec")) or 0.0
    extractor = FeatureExtractor()
    cache = build_frame_cache(bgr, np.zeros((62, 80), dtype=np.uint8), ts=0.0, skip_nir_160=True)
    rec = extractor.extract(
        cache,
        nir_channel="unknown",
        thermal_channel=_thermal_channel(row),
        ts=0.0,
        source="paired_sensor_capture",
        frame_idx=frame_idx,
        has_motion=False,
        has_temporal=False,
        label=trusted_label,
        label_source=str(row.get("label_source") or "none"),
        label_confidence=_safe_float(row.get("label_confidence")) if trusted_label else None,
    ).to_dict()
    rec.update(
        {
            "pair_id": row.get("pair_id", ""),
            "session_id": row.get("session_id", row.get("source_or_session", "paired_data")),
            "frame_idx": frame_idx,
            "timestamp_sec": timestamp_sec,
            "timestamp_iso": row.get("timestamp_iso", ""),
            "nir_raw_path": row.get("nir_raw_path", ""),
            "thermal_raw_path": row.get("thermal_raw_path") or row.get("thermal_heatmap_path", ""),
            "thermal_heatmap_path": row.get("thermal_heatmap_path", ""),
            "fusion_output_path": row.get("fusion_output_path", ""),
            "pairing_tier": row.get("pairing_tier", "unpaired"),
            "pairing_gap_ms": row.get("pairing_gap_ms", ""),
            "source_dataset": "paired_sensor_capture",
            "capture_device": row.get("capture_device", "imx+mi48"),
            "nir_modality": _nir_modality(row),
            "thermal_modality": row.get("thermal_modality", "unknown_thermal"),
            "nir_channel": "unknown_optical",
            "thermal_channel": _thermal_channel(row),
            "label": trusted_label,
            "env_label": row.get("env_label", "unknown"),
            "label_source": row.get("label_source", "none"),
            "label_confidence": _safe_float(row.get("label_confidence")) if trusted_label else None,
            "label_trust_status": label_status,
            "evidence_type": _evidence_type(row, trusted_label),
            "metadata_status": "verified",
            "metadata_verification": "paired_timestamp_csv_video_frame_identity",
            "frame_path": row.get("nir_raw_path", ""),
            "relative_image_id": f"{row.get('session_id', 'paired_data')}/{row.get('pair_id', '')}",
        }
    )
    return rec


def decode_video_frame(video_path: Path, frame_index: int) -> np.ndarray:
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    try:
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise ValueError(f"Could not decode frame {frame_index} from {video_path}")
    return frame


def _path_without_fragment(path_text: str) -> Path:
    return Path(path_text.split("#", 1)[0])


def _frame_index(row: dict[str, Any]) -> int:
    for key in ("video_frame_index", "imx_frame_id", "frame_idx"):
        value = row.get(key)
        if value not in (None, ""):
            return int(float(value))
    return 0


def extract_records_from_manifest(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    skipped: list[str] = []
    for row in rows:
        try:
            video_path = _path_without_fragment(row.get("imx_video_path") or row.get("nir_raw_path") or "")
            if not video_path.is_absolute():
                video_path = REPO_ROOT / video_path
            frame = decode_video_frame(video_path, _frame_index(row))
            records.append(feature_record_from_pair_frame(frame, row))
        except Exception as exc:
            skipped.append(f"{row.get('pair_id', '')}: {exc}")
    return records, skipped


def _ml_manifest_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        trusted_label, label_status = trusted_label_from_manifest_row(row)
        out.append(
            {
                "pair_id": row.get("pair_id", ""),
                "session_id": row.get("session_id", ""),
                "frame_idx": row.get("frame_idx", ""),
                "timestamp_sec": row.get("timestamp_sec", ""),
                "pairing_tier": row.get("pairing_tier", ""),
                "nir_raw_path": row.get("nir_raw_path", ""),
                "thermal_raw_path": row.get("thermal_raw_path") or row.get("thermal_heatmap_path", ""),
                "fusion_output_path": row.get("fusion_output_path", ""),
                "nir_modality": _nir_modality(row),
                "thermal_modality": row.get("thermal_modality", "unknown_thermal"),
                "label_source": row.get("label_source", "none"),
                "env_label": row.get("env_label", "unknown"),
                "label_trust_status": label_status,
                "trusted_label": trusted_label or "",
                "evidence_type": _evidence_type(row, trusted_label),
            }
        )
    return out


def write_ml_inventory(path: Path, manifest_rows: list[dict[str, Any]], feature_rows: list[dict[str, Any]]) -> None:
    tier_counts = Counter(str(row.get("pairing_tier", "unknown")) for row in manifest_rows)
    rows = [
        ["manifest_rows", len(manifest_rows)],
        ["feature_rows", len(feature_rows)],
        ["trusted_labeled_rows", sum(1 for row in feature_rows if row.get("evidence_type") == "labeled")],
        ["nir_modality", dict(Counter(str(row.get("nir_modality", "unknown")) for row in manifest_rows))],
        ["thermal_modality", dict(Counter(str(row.get("thermal_modality", "unknown")) for row in manifest_rows))],
        ["pairing_tier_counts", dict(sorted(tier_counts.items()))],
    ]
    write_text(
        path,
        "\n".join(
            [
                "# Paired Data ML Inventory",
                "",
                "Evidence scope: paired optical/NIR candidate frames for ML domain shift and labeling. Accuracy is measured only when trusted labels exist.",
                "",
                markdown_table(["Metric", "Value"], rows),
            ]
        )
        + "\n",
    )


def write_audit(path: Path, manifest_rows: list[dict[str, Any]], skipped: list[str]) -> None:
    trusted = sum(1 for row in manifest_rows if row.get("trusted_label"))
    write_text(
        path,
        "\n".join(
            [
                "# Paired Data ML Audit",
                "",
                "Status: paired data is used for ML domain-shift, confidence, abstention, prediction distribution, and manual labeling evidence.",
                "",
                "No paired-data accuracy, model-selection-by-accuracy, training integration, production migration, schema migration, or class migration is allowed without trusted labels and user confirmation.",
                "",
                markdown_table(
                    ["Field", "Value"],
                    [
                        ["pair_rows", len(manifest_rows)],
                        ["trusted_label_rows", trusted],
                        ["skipped_feature_rows", len(skipped)],
                        ["label_policy", "taxonomy-valid + trusted source + label_confidence >= 0.8"],
                        ["inference_scope", "RGB-scaler proxy inference, not validated NIR classifier accuracy"],
                    ],
                ),
            ]
        )
        + "\n",
    )


def write_feature_summary(path: Path, records: list[dict[str, Any]], skipped: list[str]) -> None:
    write_text(
        path,
        "\n".join(
            [
                "# Paired NIR Feature Summary",
                "",
                "Feature set: `optical_12_baseline`. Status: paired optical/NIR candidate features for domain shift; not training data.",
                "",
                markdown_table(
                    ["Metric", "Value"],
                    [
                        ["processed_rows", len(records)],
                        ["skipped_rows", len(skipped)],
                        ["trusted_label_rows", sum(1 for row in records if row.get("evidence_type") == "labeled")],
                        ["pairing_tier_counts", dict(Counter(str(row.get("pairing_tier", "unknown")) for row in records))],
                        ["label_source_counts", dict(Counter(str(row.get("label_source", "unknown")) for row in records))],
                    ],
                ),
            ]
        )
        + "\n",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract optical_12_baseline features from paired IMX/NIR capture frames.")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/paired_eval/strict_paired_manifest.csv"))
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--feature-set", choices=["optical_12_baseline"], default="optical_12_baseline")
    parser.add_argument("--ml-manifest", type=Path, default=Path("artifacts/paired_eval/paired_ml_manifest.csv"))
    parser.add_argument("--inventory", type=Path, default=Path("docs/tables/ml/paired_data_ml_inventory.md"))
    parser.add_argument("--audit-doc", type=Path, default=Path("docs/ml/PAIRED_DATA_ML_AUDIT.md"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = _read_csv(args.manifest)
    records, skipped = extract_records_from_manifest(rows)
    manifest_rows = _ml_manifest_rows(rows)
    write_jsonl(args.out_jsonl, records)
    _write_csv(args.ml_manifest, manifest_rows)
    write_feature_summary(args.summary, records, skipped)
    write_ml_inventory(args.inventory, manifest_rows, records)
    write_audit(args.audit_doc, manifest_rows, skipped)
    print(f"Wrote {args.out_jsonl}")
    print(f"Wrote {args.summary}")
    print(f"Wrote {args.ml_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
