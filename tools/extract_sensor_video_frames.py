#!/usr/bin/env python3
"""Sample representative frames from raw sensor videos with a manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.inventory_raw_sensor_captures import VIDEO_EXTS, parse_ffprobe_payload
from tools.ml_metadata_utils import dhash_image, markdown_table, sha256_file, write_text


def sample_frame_indices(total_frames: int, fps: float, sample_fps: float, max_frames: int) -> list[int]:
    if total_frames <= 0 or fps <= 0 or sample_fps <= 0 or max_frames <= 0:
        return []
    step = max(1, int(fps / sample_fps))
    indices = list(range(0, total_frames, step))
    return indices[:max_frames]


def build_frame_manifest_row(
    *,
    video_path: Path,
    frame_path: Path,
    frame_idx: int,
    fps: float,
    modality_guess: str,
    command: str,
) -> dict[str, Any]:
    img = cv.imread(str(frame_path))
    h, w = img.shape[:2] if img is not None else (0, 0)
    return {
        "video_path": str(video_path),
        "video_id": video_path.stem,
        "frame_path": str(frame_path),
        "frame_id": f"{video_path.stem}:{frame_idx}",
        "frame_idx": int(frame_idx),
        "timestamp_sec": float(frame_idx) / float(fps) if fps else 0.0,
        "width": int(w),
        "height": int(h),
        "file_sha256": sha256_file(frame_path),
        "dhash": dhash_image(frame_path),
        "modality_guess": modality_guess,
        "extraction_command": command,
        "extract_status": "ok",
    }


def write_modality_review_sheet(frame_paths: list[Path], out_path: Path, *, title: str) -> None:
    thumbs: list[np.ndarray] = []
    for path in frame_paths:
        img = cv.imread(str(path))
        if img is None:
            continue
        thumb = cv.resize(img, (240, 135), interpolation=cv.INTER_AREA)
        cv.putText(
            thumb,
            path.name[:28],
            (8, 124),
            cv.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )
        thumbs.append(thumb)
    if not thumbs:
        canvas = np.full((160, 320, 3), 32, dtype=np.uint8)
    else:
        cols = min(3, len(thumbs))
        rows = int(math.ceil(len(thumbs) / cols))
        canvas = np.full((rows * 135 + 40, cols * 240, 3), 24, dtype=np.uint8)
        for i, thumb in enumerate(thumbs):
            r, c = divmod(i, cols)
            canvas[40 + r * 135 : 40 + (r + 1) * 135, c * 240 : (c + 1) * 240] = thumb
    cv.putText(canvas, f"Modality review: {title}", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 1, cv.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(out_path), canvas)


def _probe(path: Path) -> dict[str, Any]:
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
    return parse_ffprobe_payload(json.loads(out))


def extract_video_frames(video_path: Path, frames_dir: Path, *, sample_fps: float, max_frames: int, jpeg_quality: int, command: str) -> list[dict[str, Any]]:
    meta = _probe(video_path)
    fps = float(meta["fps"] or 0.0)
    total = int(meta["approx_frame_count"] or 0)
    indices = set(sample_frame_indices(total, fps, sample_fps, max_frames))
    if not indices:
        return []
    out_dir = frames_dir / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv.VideoCapture(str(video_path))
    rows: list[dict[str, Any]] = []
    for frame_idx in sorted(indices):
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame_path = out_dir / f"{video_path.stem}_frame_{frame_idx:06d}.jpg"
        cv.imwrite(str(frame_path), frame, [int(cv.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        rows.append(
            build_frame_manifest_row(
                video_path=video_path,
                frame_path=frame_path,
                frame_idx=frame_idx,
                fps=fps,
                modality_guess="unknown optical",
                command=command,
            )
        )
    cap.release()
    return rows


def write_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "video_path",
        "video_id",
        "frame_path",
        "frame_id",
        "frame_idx",
        "timestamp_sec",
        "width",
        "height",
        "file_sha256",
        "dhash",
        "modality_guess",
        "extraction_command",
        "extract_status",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_manifest_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    by_video: dict[str, int] = {}
    for row in rows:
        by_video[row["video_id"]] = by_video.get(row["video_id"], 0) + 1
    body = [
        "# Raw Sensor Frame Extraction Summary",
        "",
        "Status: sampled-frame extraction only. Raw videos were not modified. Modality remains `unknown optical`.",
        "",
        markdown_table(["Metric", "Value"], [["frames_extracted", len(rows)], ["video_count", len(by_video)]]),
        "",
        markdown_table(["Video", "Frames"], [[k, v] for k, v in sorted(by_video.items())]),
    ]
    write_text(path, "\n".join(body) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract sampled frames from raw sensor videos.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--frames-dir", type=Path, required=True)
    p.add_argument("--manifest-csv", type=Path, required=True)
    p.add_argument("--manifest-jsonl", type=Path)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--modality-review", type=Path, default=Path("docs/figures/ml/raw_sensor_modality_review.png"))
    p.add_argument("--fps", type=float, default=0.5, help="Sampling FPS")
    p.add_argument("--max-frames-per-video", type=int, default=600)
    p.add_argument("--jpeg-quality", type=int, default=90)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    videos = sorted(p for p in args.input.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    command = " ".join([sys.executable, *sys.argv])
    rows: list[dict[str, Any]] = []
    for video in videos:
        rows.extend(
            extract_video_frames(
                video,
                args.frames_dir,
                sample_fps=args.fps,
                max_frames=args.max_frames_per_video,
                jpeg_quality=args.jpeg_quality,
                command=command,
            )
        )
    write_manifest_csv(args.manifest_csv, rows)
    if args.manifest_jsonl:
        write_manifest_jsonl(args.manifest_jsonl, rows)
    write_summary(args.summary, rows)
    review_frames = [Path(row["frame_path"]) for row in rows[:: max(1, len(rows) // 9)]][:9]
    write_modality_review_sheet(review_frames, args.modality_review, title="unknown optical")
    print(f"Wrote {args.manifest_csv}")
    print(f"Wrote {args.summary}")
    print(f"Wrote {args.modality_review}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
