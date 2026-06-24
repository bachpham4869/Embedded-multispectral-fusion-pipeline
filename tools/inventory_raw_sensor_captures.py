#!/usr/bin/env python3
"""Inventory raw SmartBinocular sensor capture files without modifying them."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ml_metadata_utils import markdown_table, sha256_file, write_text

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def _rate_to_float(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    if "/" in value:
        num, den = value.split("/", 1)
        try:
            den_f = float(den)
            return float(num) / den_f if den_f else None
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_ffprobe_payload(payload: dict[str, Any]) -> dict[str, Any]:
    streams = payload.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    fmt = payload.get("format") or {}
    fps = _rate_to_float(video.get("avg_frame_rate")) or _rate_to_float(video.get("r_frame_rate")) or 0.0
    duration = float(fmt.get("duration") or 0.0)
    nb_frames = video.get("nb_frames")
    try:
        approx_frame_count = int(nb_frames)
    except (TypeError, ValueError):
        approx_frame_count = int(round(duration * fps)) if duration and fps else 0
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    return {
        "duration_sec": duration,
        "fps": fps,
        "resolution": f"{width}x{height}" if width and height else "unknown",
        "width": width,
        "height": height,
        "codec": video.get("codec_name") or "unknown",
        "approx_frame_count": approx_frame_count,
        "bit_rate": fmt.get("bit_rate") or "",
        "modality_guess": "unknown optical",
    }


def ffprobe_payload(path: Path) -> dict[str, Any]:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size,bit_rate:stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    return json.loads(out)


def inventory_file(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": str(path),
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "timestamp_if_available": "",
        "source_notes": "",
    }
    if path.suffix.lower() in VIDEO_EXTS:
        try:
            row.update(parse_ffprobe_payload(ffprobe_payload(path)))
            row["usable_for_frame_extract"] = True
        except Exception as exc:
            row.update(
                {
                    "duration_sec": 0.0,
                    "fps": 0.0,
                    "resolution": "unknown",
                    "width": 0,
                    "height": 0,
                    "codec": "unknown",
                    "approx_frame_count": 0,
                    "bit_rate": "",
                    "modality_guess": "unknown optical",
                    "usable_for_frame_extract": False,
                    "source_notes": f"ffprobe failed: {exc}",
                }
            )
    else:
        row.update(
            {
                "duration_sec": "",
                "fps": "",
                "resolution": "",
                "width": "",
                "height": "",
                "codec": "",
                "approx_frame_count": "",
                "bit_rate": "",
                "modality_guess": "unknown",
                "usable_for_frame_extract": False,
            }
        )
    return row


def inventory_tree(root: Path) -> list[dict[str, Any]]:
    return [inventory_file(path) for path in sorted(p for p in root.rglob("*") if p.is_file())]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "path",
        "filename",
        "bytes",
        "sha256",
        "duration_sec",
        "fps",
        "resolution",
        "codec",
        "approx_frame_count",
        "bit_rate",
        "timestamp_if_available",
        "modality_guess",
        "usable_for_frame_extract",
        "source_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_inventory_md(path: Path, rows: list[dict[str, Any]]) -> None:
    table_rows = [
        [
            r["filename"],
            r["bytes"],
            f"{float(r.get('duration_sec') or 0):.2f}" if r.get("duration_sec") != "" else "",
            f"{float(r.get('fps') or 0):.3f}" if r.get("fps") != "" else "",
            r.get("resolution", ""),
            r.get("codec", ""),
            r.get("approx_frame_count", ""),
            r.get("modality_guess", ""),
            r.get("usable_for_frame_extract", ""),
        ]
        for r in rows
    ]
    body = [
        "# Raw Sensor Capture Inventory",
        "",
        "Status: raw capture inventory only. Modality is `unknown optical` until user/manual review confirms NIR, RGB, fusion, or thermal.",
        "",
        markdown_table(
            ["Filename", "Bytes", "Duration s", "FPS", "Resolution", "Codec", "Frames", "Modality", "Usable"],
            table_rows,
        ),
    ]
    write_text(path, "\n".join(body) + "\n")


def write_audit_md(path: Path, rows: list[dict[str, Any]]) -> None:
    usable = sum(1 for r in rows if r.get("usable_for_frame_extract"))
    body = [
        "# Raw Sensor Capture Audit",
        "",
        "Raw files were scanned read-only. No raw video was modified.",
        "",
        markdown_table(
            ["Metric", "Value"],
            [
                ["raw_capture_count", len(rows)],
                ["usable_video_count", usable],
                ["modality_status", "unknown optical until user confirmation"],
                ["sensor_accuracy_status", "not measured; raw capture is unlabeled"],
            ],
        ),
    ]
    write_text(path, "\n".join(body) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inventory raw sensor captures with ffprobe metadata.")
    p.add_argument("--input", type=Path, required=True, help="Raw sensor capture directory")
    p.add_argument("--out-md", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--audit-md", type=Path, required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = inventory_tree(args.input)
    write_csv(args.out_csv, rows)
    write_inventory_md(args.out_md, rows)
    write_audit_md(args.audit_md, rows)
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.audit_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
