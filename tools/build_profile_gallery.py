"""build_profile_gallery.py — Produce a Raw vs Quality vs Throughput grid figure.

Usage:
    python tools/build_profile_gallery.py
    python tools/build_profile_gallery.py \
        --manifest data/eval/nir_val/manifest_v2.csv \
        --n-per-class 1 \
        --out docs/thesis_eval/bucket_dispatch/figures/profile_gallery.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Dict

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
    resolve_optical_bucket,
)

THUMB_W = 240
THUMB_H = 180
LABEL_H = 18
FONT = cv.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4
FONT_THICK = 1

COLUMNS = [
    ("Raw Sensor Feed", "raw"),
    ("Throughput Profile\n(Embedded Defaults)", "throughput"),
    ("Quality Profile\n(Full Processing)", "quality"),
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
    by_class: Dict[str, list] = {}
    for r in manifest:
        cls = r.get("env_class", "unknown")
        by_class.setdefault(cls, []).append(r)
        
    priority_files = {
        "input_000027.png",  # Glare with strong effect
        "input_000040.png",  # Glare with mild effect
    }
    
    selected = []
    # Only select a few representative classes to fit nicely
    target_classes = ["night_clear", "glare", "fog"]
    for cls in target_classes:
        if cls not in by_class:
            continue
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
    cv.putText(bar, text[:25], (2, LABEL_H - 4), FONT, FONT_SCALE, (200, 200, 200), FONT_THICK, cv.LINE_AA)
    return np.vstack([bar, img])

def _col_header(text: str) -> np.ndarray:
    lines = text.split("\n")
    h = LABEL_H * len(lines) + 4
    bar = np.ones((h, THUMB_W, 3), dtype=np.uint8) * 50
    for i, line in enumerate(lines):
        cv.putText(bar, line, (2, (i + 1) * LABEL_H - 4), FONT, FONT_SCALE, (255, 220, 100), FONT_THICK, cv.LINE_AA)
    return bar

def _make_dispatch():
    # Maintain instances to avoid cold-start penalties if needed
    quality_enhancer = HybridNIREnhancer(patch_size=5, detail_strength=0.25)
    throughput_enhancer = HybridNIREnhancer(patch_size=3, detail_strength=0.0)
    
    def run_profile(profile: str, env_class: str, bgr: np.ndarray) -> np.ndarray:
        if profile == "raw":
            return bgr.copy()
            
        lite = (profile == "throughput")
        bucket = resolve_optical_bucket(env_class, lite=lite)
        
        if bucket == "A":
            enh = throughput_enhancer if lite else quality_enhancer
            enh.reset()
            return enh.process(bgr)
        if bucket == "B":
            # Throughput/raw might use lower clip, but nir_nir_night_clahe defaults to clip=3.0
            return nir_nir_night_clahe(bgr)
        if bucket == "C":
            return nir_anti_glare_bgr(bgr)
        if bucket == "D":
            return nir_dehaze_lite(bgr)
        
        # Default fallback
        enh = throughput_enhancer if lite else quality_enhancer
        enh.reset()
        return enh.process(bgr)

    return run_profile

def build_gallery(frames: List[dict], run_profile) -> np.ndarray:
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
            continue
        # Upscale a bit for processing
        bgr_raw = cv.resize(bgr_raw, (320, 240))

        cells = []
        for _, profile in COLUMNS:
            try:
                out = run_profile(profile, env_cls, bgr_raw)
            except Exception as e:
                out = bgr_raw
            cell = _thumb(out)
            cell = _label_cell(cell, f"ENV: {env_cls}" if profile == "raw" else "")
            cells.append(cell)

        row_img = np.hstack(cells)
        rows_imgs.append(row_img)

    if not rows_imgs:
        raise RuntimeError("No frames processed")

    body = np.vstack(rows_imgs)
    if header_row.shape[1] < body.shape[1]:
        pad = np.ones((header_row.shape[0], body.shape[1] - header_row.shape[1], 3), dtype=np.uint8) * 50
        header_row = np.hstack([header_row, pad])

    return np.vstack([header_row, body])

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/eval/nir_val/manifest_v2.csv")
    parser.add_argument("--n-per-class", type=int, default=1)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/thesis_eval/bucket_dispatch/figures/profile_gallery.png")
    args = parser.parse_args()

    if not args.manifest.exists():
        sys.exit(0)

    manifest = _load_manifest(args.manifest)
    frames = _select_frames(manifest, args.n_per_class)
    
    if not frames:
        sys.exit(0)

    run_profile = _make_dispatch()
    gallery = build_gallery(frames, run_profile)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(args.out), gallery)

if __name__ == "__main__":
    main()
