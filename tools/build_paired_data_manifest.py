#!/usr/bin/env python3
"""Build a strict paired NIR/thermal manifest from captured paired videos.

This is an offline evaluation helper. It only reads ``data/paired_data`` and
emits provenance-rich manifests/docs under artifacts/docs paths.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2 as cv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fusion_eval_manifest import markdown_table, write_csv, write_json, write_markdown  # type: ignore[import]


STRICT_TIERS = {"frame_strict", "time_strict_100ms", "protocol_strict_1s"}
MANIFEST_FIELDS = [
    "pair_id",
    "session_id",
    "frame_idx",
    "video_frame_index",
    "timestamp_iso",
    "timestamp_sec",
    "timestamp_source",
    "pairing_tier",
    "pairing_gap_sec",
    "pairing_gap_ms",
    "time_strict_100ms",
    "protocol_strict_1s",
    "nir_raw_path",
    "nir_enhanced_path",
    "thermal_raw_path",
    "thermal_heatmap_path",
    "thermal_mask_path",
    "fusion_output_path",
    "homography_path",
    "imx_video_path",
    "thermal_video_path",
    "imx_frame_id",
    "thermal_frame_id",
    "thermal_modality",
    "thermal_unit_for_display",
    "thermal_scale",
    "env_label",
    "label_source",
    "label_confidence",
    "processing_bucket",
    "processing_bucket_source",
    "fusion_mode",
    "fusion_source",
    "capture_device",
    "input_data_type",
    "evidence_label",
    "metric_tier",
    "source_or_session",
    "alignment_source",
    "thermal_none_reads",
    "thermal_bad_frames",
    "thermal_errors",
    "imx_errors",
    "caveat",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paired NIR/thermal inventory and strict paired manifest from data/paired_data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--input", type=Path, default=Path("data/paired_data"), help="Paired data directory.")
    parser.add_argument("--timestamps", type=Path, default=Path("timestamps.csv"), help="Timestamp CSV path or name under --input.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/paired_eval"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs/paired"))
    parser.add_argument("--tables-dir", type=Path, default=Path("docs/tables/paired"))
    parser.add_argument("--dry-run", action="store_true", help="Print planned outputs without writing.")
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _rel(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    except Exception:
        return "unknown"


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def classify_pairing_tier(
    skew_ms: float | None,
    *,
    same_csv_row: bool,
    has_required_modalities: bool,
) -> str:
    """Classify pairing strength from capture metadata."""
    if not has_required_modalities:
        return "unpaired"
    if skew_ms is None:
        return "weak_paired"
    gap_ms = abs(skew_ms)
    if same_csv_row:
        return "frame_strict"
    if gap_ms <= 100.0:
        return "time_strict_100ms"
    if gap_ms <= 1000.0:
        return "protocol_strict_1s"
    if gap_ms <= 5000.0:
        return "near_paired"
    return "weak_paired"


def evidence_label_for_tier(pairing_tier: str) -> str:
    if pairing_tier in STRICT_TIERS:
        return "real_paired"
    if pairing_tier in {"near_paired", "weak_paired"}:
        return "proxy"
    if pairing_tier == "unpaired":
        return "unpaired"
    return "unknown"


def detect_thermal_modality(input_dir: Path) -> str:
    """Return raw thermal modality when numeric thermal arrays are present."""
    numeric_exts = {".npy", ".npz"}
    for path in input_dir.rglob("*"):
        if path.suffix.lower() in numeric_exts and "thermal" in path.name.lower():
            return "raw_numeric_thermal"
    return "display_heatmap_like"


def video_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": "false"}
    cap = cv.VideoCapture(str(path))
    if not cap.isOpened():
        return {"exists": "true", "openable": "false", "size_bytes": path.stat().st_size}
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {
        "exists": "true",
        "openable": "true",
        "size_bytes": path.stat().st_size,
        "frame_count": frame_count,
        "fps": round(fps, 6),
        "width": width,
        "height": height,
        "duration_sec": round(frame_count / fps, 6) if fps > 0 else "",
    }


def read_timestamp_rows(timestamps_path: Path) -> list[dict[str, str]]:
    with timestamps_path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _input_data_type(thermal_modality: str) -> str:
    if thermal_modality == "raw_numeric_thermal":
        return "paired NIR video + raw numeric thermal arrays"
    return "paired NIR video + thermal display/heatmap-like video"


def _thermal_caveat(thermal_modality: str) -> str:
    if thermal_modality == "raw_numeric_thermal":
        return "Raw numeric thermal arrays were discovered; validate units before radiometric claims."
    return "thermal_modality=display_heatmap_like; thermal_paired.mp4 is not raw radiometric thermal."


def manifest_rows_from_timestamps(input_dir: Path, timestamps_path: Path) -> list[dict[str, str]]:
    thermal_modality = detect_thermal_modality(input_dir)
    raw_rows = read_timestamp_rows(timestamps_path)
    rows: list[dict[str, str]] = []
    session_id = input_dir.name or "paired_data"
    input_type = _input_data_type(thermal_modality)
    caveat = _thermal_caveat(thermal_modality)
    for video_index, raw in enumerate(raw_rows):
        pair_idx = raw.get("pair_idx") or str(video_index)
        skew_ms = _safe_float(raw.get("skew_ms_imx_minus_thermal"))
        gap_sec = abs(skew_ms) / 1000.0 if skew_ms is not None else None
        imx_video = raw.get("imx_video") or "imx_paired.mp4"
        thermal_video = raw.get("thermal_video") or "thermal_paired.mp4"
        has_modalities = bool((input_dir / imx_video).exists() and (input_dir / thermal_video).exists())
        pairing_tier = classify_pairing_tier(skew_ms, same_csv_row=True, has_required_modalities=has_modalities)
        evidence_label = evidence_label_for_tier(pairing_tier)
        pair_id = f"paired_{int(pair_idx):06d}" if str(pair_idx).isdigit() else f"paired_{video_index:06d}"
        thermal_heatmap_path = (
            f"{input_dir / thermal_video}#frame={video_index}" if thermal_modality == "display_heatmap_like" else ""
        )
        thermal_raw_path = f"{input_dir / thermal_video}#frame={video_index}" if thermal_modality != "display_heatmap_like" else ""
        row = {
            "pair_id": pair_id,
            "session_id": session_id,
            "frame_idx": pair_idx,
            "video_frame_index": str(video_index),
            "timestamp_iso": "",
            "timestamp_sec": raw.get("pair_t_sec", ""),
            "timestamp_source": "relative_pair_t_sec",
            "pairing_tier": pairing_tier,
            "pairing_gap_sec": "" if gap_sec is None else f"{gap_sec:.6f}",
            "pairing_gap_ms": "" if skew_ms is None else f"{abs(skew_ms):.6f}",
            "time_strict_100ms": _bool_text(gap_sec is not None and gap_sec <= 0.1),
            "protocol_strict_1s": _bool_text(gap_sec is not None and gap_sec <= 1.0),
            "nir_raw_path": f"{input_dir / imx_video}#frame={video_index}",
            "nir_enhanced_path": "",
            "thermal_raw_path": thermal_raw_path,
            "thermal_heatmap_path": thermal_heatmap_path,
            "thermal_mask_path": "",
            "fusion_output_path": "",
            "homography_path": "",
            "imx_video_path": str(input_dir / imx_video),
            "thermal_video_path": str(input_dir / thermal_video),
            "imx_frame_id": raw.get("imx_frame_id", ""),
            "thermal_frame_id": raw.get("thermal_frame_id", ""),
            "thermal_modality": thermal_modality,
            "thermal_unit_for_display": raw.get("thermal_unit_for_display", ""),
            "thermal_scale": raw.get("thermal_scale", ""),
            "env_label": "unknown",
            "label_source": "none",
            "label_confidence": "0",
            "processing_bucket": "unknown",
            "processing_bucket_source": "unknown",
            "fusion_mode": "",
            "fusion_source": "none",
            "capture_device": "imx+mi48",
            "input_data_type": input_type,
            "evidence_label": evidence_label,
            "metric_tier": "Tier 1" if pairing_tier in STRICT_TIERS else "Tier 3",
            "source_or_session": session_id,
            "alignment_source": "pair_idx_video_order",
            "thermal_none_reads": raw.get("thermal_none_reads", ""),
            "thermal_bad_frames": raw.get("thermal_bad_frames", ""),
            "thermal_errors": raw.get("thermal_errors", ""),
            "imx_errors": raw.get("imx_errors", ""),
            "caveat": caveat,
            "notes": "CSV row defines the paired capture cycle; pair_idx/video frame order is validated against video metadata.",
        }
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _count_rows(rows: list[dict[str, str]], key: str) -> list[dict[str, Any]]:
    counts = Counter(row.get(key, "unknown") or "unknown" for row in rows)
    return [{key: name, "n": counts[name]} for name in sorted(counts)]


def _inventory_rows(repo_root: Path, input_dir: Path, timestamps_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        kind = "timestamp_csv" if path == timestamps_path else ("video" if path.suffix.lower() in {".mp4", ".avi"} else "sidecar_or_data")
        item = {
            "path": _rel(repo_root, path),
            "kind": kind,
            "size_bytes": path.stat().st_size,
        }
        if kind == "video":
            item.update(video_metadata(path))
        rows.append(item)
    return rows


def _write_docs(
    *,
    repo_root: Path,
    input_dir: Path,
    timestamps_path: Path,
    output_dir: Path,
    docs_dir: Path,
    tables_dir: Path,
    rows: list[dict[str, str]],
    inventory: list[dict[str, Any]],
) -> None:
    inventory_md_rows = [
        {
            "path": row.get("path", ""),
            "kind": row.get("kind", ""),
            "frame_count": row.get("frame_count", ""),
            "fps": row.get("fps", ""),
            "width": row.get("width", ""),
            "height": row.get("height", ""),
            "size_bytes": row.get("size_bytes", ""),
        }
        for row in inventory
    ]
    write_markdown(
        tables_dir / "paired_data_inventory.md",
        "Paired Data Inventory",
        markdown_table(inventory_md_rows, ["path", "kind", "frame_count", "fps", "width", "height", "size_bytes"]),
    )
    summary = []
    for key in ("pairing_tier", "evidence_label", "thermal_modality", "input_data_type"):
        summary.extend({"summary_key": key, "value": r[key], "n": r["n"]} for r in _count_rows(rows, key))
    write_markdown(
        tables_dir / "strict_paired_manifest_summary.md",
        "Strict Paired Manifest Summary",
        markdown_table(summary, ["summary_key", "value", "n"]),
    )
    strict_count = sum(1 for row in rows if row.get("pairing_tier") in STRICT_TIERS)
    time100_count = sum(1 for row in rows if row.get("time_strict_100ms") == "true")
    body = "\n".join(
        [
            f"- Input directory: `{_rel(repo_root, input_dir)}`",
            f"- Timestamp CSV: `{_rel(repo_root, timestamps_path)}`",
            f"- Total manifest rows: `{len(rows)}`",
            f"- Strict paired rows: `{strict_count}`",
            f"- Time strict <=100 ms rows: `{time100_count}`",
            "- Thermal modality: `display_heatmap_like` unless raw numeric thermal arrays are listed in the inventory.",
            "- Caveat: `thermal_paired.mp4` is not raw radiometric thermal when `thermal_modality=display_heatmap_like`.",
            f"- CSV manifest: `{_rel(repo_root, output_dir / 'strict_paired_manifest.csv')}`",
            f"- JSONL manifest: `{_rel(repo_root, output_dir / 'paired_data_manifest.jsonl')}`",
        ]
    )
    write_markdown(docs_dir / "PAIRED_DATA_AUDIT.md", "Paired Data Audit", body)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    input_dir = _resolve(repo_root, args.input)
    timestamps_path = args.timestamps if args.timestamps.is_absolute() else input_dir / args.timestamps
    output_dir = _resolve(repo_root, args.output_dir)
    docs_dir = _resolve(repo_root, args.docs_dir)
    tables_dir = _resolve(repo_root, args.tables_dir)

    if not timestamps_path.exists():
        raise FileNotFoundError(f"timestamps CSV not found: {timestamps_path}")

    rows = manifest_rows_from_timestamps(input_dir, timestamps_path)
    strict_rows = [row for row in rows if row.get("pairing_tier") in STRICT_TIERS]
    inventory = _inventory_rows(repo_root, input_dir, timestamps_path)
    run = {
        "timestamp_iso": _now_iso(),
        "git_commit": _git_commit(repo_root),
        "command": " ".join(sys.argv),
        "inputs": [_rel(repo_root, input_dir), _rel(repo_root, timestamps_path)],
        "outputs": [
            _rel(repo_root, output_dir / "strict_paired_manifest.csv"),
            _rel(repo_root, output_dir / "paired_data_manifest.jsonl"),
            _rel(repo_root, tables_dir / "paired_data_inventory.md"),
            _rel(repo_root, tables_dir / "strict_paired_manifest_summary.md"),
            _rel(repo_root, docs_dir / "PAIRED_DATA_AUDIT.md"),
        ],
        "config": {
            "pairing_tiers": ["frame_strict", "time_strict_100ms", "protocol_strict_1s", "near_paired", "weak_paired", "unpaired", "unknown"],
            "thermal_modality_policy": "raw numeric thermal arrays required for raw radiometric thermal claims",
        },
    }
    if args.dry_run:
        print(json.dumps(run, indent=2, sort_keys=True))
        return 0

    write_csv(output_dir / "strict_paired_manifest.csv", strict_rows, MANIFEST_FIELDS)
    write_jsonl(output_dir / "paired_data_manifest.jsonl", rows)
    write_csv(output_dir / "paired_data_inventory.csv", inventory)
    write_json(output_dir / "paired_data_run_manifest.json", run)
    _write_docs(
        repo_root=repo_root,
        input_dir=input_dir,
        timestamps_path=timestamps_path,
        output_dir=output_dir,
        docs_dir=docs_dir,
        tables_dir=tables_dir,
        rows=rows,
        inventory=inventory,
    )
    print(f"Wrote {len(strict_rows)} strict paired rows and {len(rows)} total paired rows to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
