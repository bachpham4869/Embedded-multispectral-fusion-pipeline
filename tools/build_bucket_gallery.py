"""build_bucket_gallery.py — Produce a raw→bucket-A-through-F grid figure.

Selects up to --n-per-class representative frames per env_class from the manifest,
runs each through all six bucket processors, and tiles the results into a PNG grid.

  Rows:    one per selected frame (sorted by env_class)
  Columns: Raw | Bucket A | Bucket B | Bucket C | Bucket D | Bucket E | Bucket F

Bucket surrogate modes (cold-start / single-frame / EMA proxy) are noted in the
column header. Output is NOT from a live pipeline — it is a still-image cold-start
representative run.

Usage:
    python tools/build_bucket_gallery.py
    python tools/build_bucket_gallery.py \\
        --manifest data/eval/nir_val/manifest_v2.csv \\
        --n-per-class 2 \\
        --out docs/thesis_eval/bucket_dispatch/figures/bucket_gallery.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from smartbinocular.nir_pipeline import (
    HybridNIREnhancer,
    RainTemporalMedian,
    nir_anti_glare_bgr,
    nir_dehaze_lite,
    nir_nir_night_clahe,
    nir_transition_blend,
)

THUMB_W = 160
THUMB_H = 120
LABEL_H = 18
FONT = cv.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.32
FONT_THICK = 1

COLUMNS = [
    ("Raw", None),
    ("Bucket A\n(HybridNIR)", "A"),
    ("Bucket B\n(CLAHE clip)", "B"),
    ("Bucket C\n(Anti-glare)", "C"),
    ("Bucket D\n(Dehaze)", "D"),
    ("Bucket E\n(Rain median)", "E"),
    ("Bucket F\n(A+C blend)", "F"),
]


def _load_manifest(path: Path) -> List[dict]:
    rows = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            img_path = row.get("path", "")
            if img_path and Path(img_path).exists():
                rows.append(row)
    return rows


def _select_frames(manifest: List[dict], n_per_class: int) -> List[dict]:
    by_class: dict = {}
    for r in manifest:
        cls = r.get("env_class", "unknown")
        by_class.setdefault(cls, []).append(r)
        
    # Prioritize specific frames used in Chapter 2 for consistent visual comparison
    priority_files = {
        "input_000027.png",  # Glare with strong effect
        "input_000040.png",  # Glare with mild effect
    }
    
    selected = []
    for cls in sorted(by_class):
        class_rows = sorted(
            by_class[cls],
            key=lambda r: 0 if Path(r.get("path", "")).name in priority_files else 1
        )
        selected.extend(class_rows[:n_per_class])
    return selected


def _thumb(bgr: np.ndarray) -> np.ndarray:
    return cv.resize(bgr, (THUMB_W, THUMB_H), interpolation=cv.INTER_AREA)


def _label_cell(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.zeros((LABEL_H, THUMB_W, 3), dtype=np.uint8)
    cv.putText(bar, text[:22], (2, LABEL_H - 4), FONT, FONT_SCALE, (200, 200, 200), FONT_THICK, cv.LINE_AA)
    return np.vstack([bar, img])


def _col_header(text: str) -> np.ndarray:
    lines = text.split("\n")
    h = LABEL_H * len(lines) + 4
    bar = np.ones((h, THUMB_W, 3), dtype=np.uint8) * 50
    for i, line in enumerate(lines):
        cv.putText(bar, line, (2, (i + 1) * LABEL_H - 4), FONT, FONT_SCALE, (255, 220, 100), FONT_THICK, cv.LINE_AA)
    return bar


def _make_dispatch():
    enhancer = HybridNIREnhancer()
    rain = RainTemporalMedian()

    def run_bucket(bucket: str, bgr: np.ndarray) -> np.ndarray:
        if bucket == "A":
            enhancer.reset()
            return enhancer.process(bgr)
        if bucket == "B":
            return nir_nir_night_clahe(bgr)
        if bucket == "C":
            return nir_anti_glare_bgr(bgr)
        if bucket == "D":
            return nir_dehaze_lite(bgr)
        if bucket == "E":
            rain.reset()
            rain.process(bgr)
            rain.process(bgr)
            return rain.process(bgr)
        if bucket == "F":
            green = bgr[:, :, 1].astype(np.float32)
            ema = float(green.mean()) / 255.0
            return nir_transition_blend(bgr, enhancer, ema)
        return bgr

    return run_bucket


def build_gallery(frames: List[dict], run_bucket, n_cols: int = 7) -> np.ndarray:
    col_headers = [_col_header(col_name) for col_name, _ in COLUMNS]
    header_h = max(h.shape[0] for h in col_headers)
    padded_headers = []
    for h in col_headers:
        pad = np.ones((header_h - h.shape[0], THUMB_W, 3), dtype=np.uint8) * 50
        padded_headers.append(np.vstack([pad, h]))
    header_row = np.hstack(padded_headers)

    rows_imgs = []
    for frame in frames:
        img_path = frame["path"]
        env_cls = frame.get("env_class", "?")
        bgr_raw = cv.imread(img_path, cv.IMREAD_COLOR)
        if bgr_raw is None:
            print(f"  Skip unreadable: {img_path}", file=sys.stderr)
            continue
        bgr_raw = cv.resize(bgr_raw, (THUMB_W * 2, THUMB_H * 2))

        cells = []
        for col_name, bucket in COLUMNS:
            if bucket is None:
                cell = _thumb(bgr_raw)
            else:
                try:
                    out = run_bucket(bucket, bgr_raw)
                except Exception as e:
                    print(f"  Bucket {bucket} error on {Path(img_path).name}: {e}", file=sys.stderr)
                    out = bgr_raw
                cell = _thumb(out)
            cell = _label_cell(cell, env_cls if bucket is None else "")
            cells.append(cell)

        row_img = np.hstack(cells)
        rows_imgs.append(row_img)

    if not rows_imgs:
        raise RuntimeError("No frames processed")

    body = np.vstack(rows_imgs)
    # Pad header to match body width
    if header_row.shape[1] < body.shape[1]:
        pad = np.ones((header_row.shape[0], body.shape[1] - header_row.shape[1], 3), dtype=np.uint8) * 50
        header_row = np.hstack([header_row, pad])

    return np.vstack([header_row, body])


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=ROOT / "data/eval/nir_val/manifest_v2.csv")
    p.add_argument("--n-per-class", type=int, default=2,
                   help="Representative frames per env_class (default: 2)")
    p.add_argument("--out", type=Path, default=ROOT / "docs/thesis_eval/bucket_dispatch/figures/bucket_gallery.png")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.manifest.exists():
        print(f"THESIS_SKIP: manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(0)

    manifest = _load_manifest(args.manifest)
    if not manifest:
        print(f"THESIS_SKIP: no readable frames in manifest: {args.manifest}", file=sys.stderr)
        sys.exit(0)

    frames = _select_frames(manifest, args.n_per_class)
    print(f"Selected {len(frames)} frames ({args.n_per_class} per class) from {len(manifest)} total")

    run_bucket = _make_dispatch()
    gallery = build_gallery(frames, run_bucket)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(args.out), gallery)
    print(f"Saved: {args.out}  ({gallery.shape[1]}×{gallery.shape[0]} px)")
    print("Caption: still-image cold-start bucket outputs — not live-pipeline behavior.")


if __name__ == "__main__":
    main()
