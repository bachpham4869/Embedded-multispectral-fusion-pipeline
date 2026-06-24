"""gen_ch5_bucket_figures.py — Produce a Raw vs Throughput vs Quality grid figure for each bucket.

Usage:
    python tools/gen_ch5_bucket_figures.py
"""

from __future__ import annotations

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
    nir_transition_blend,
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

BUCKET_MAPPING = {
    "A": "night_clear",
    "B": "nir_night",
    "C": "glare",
    "D": "fog",
    "E": "rain",
    "F": "transition"
}

def _load_manifest(path: Path) -> List[dict]:
    rows = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            img_path = row.get("path", "")
            if img_path and Path(img_path).exists():
                rows.append(row)
    return rows

def _select_frame_for_bucket(manifest: List[dict], bucket: str) -> dict | None:
    # Hardcode specific representative frames based on user feedback
    if bucket == "C":
        for r in manifest:
            if "input_000040.png" in r.get("path", ""):
                return r
    elif bucket == "F":
        # transition: use an actual transition frame from the mwd dataset
        return {"path": "data/weather/mwd/dataset2/dataset2/sunrise10.jpg", "env_class": "transition"}
                
    env_class = BUCKET_MAPPING.get(bucket)
    if not env_class:
        return None
    
    # Try to find a representative frame for the target class
    for r in manifest:
        if r.get("env_class") == env_class:
            return r
            
    # Fallback if specific class not found
    for r in manifest:
        return r
    return None

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
    quality_enhancer = HybridNIREnhancer(patch_size=5, detail_strength=0.25)
    throughput_enhancer = HybridNIREnhancer(patch_size=3, detail_strength=0.0)
    rain = RainTemporalMedian()
    
    def run_profile(profile: str, bucket: str, bgr: np.ndarray) -> np.ndarray:
        if profile == "raw":
            return bgr.copy()
            
        lite = (profile == "throughput")
        
        if bucket == "A":
            enh = throughput_enhancer if lite else quality_enhancer
            enh.reset()
            return enh.process(bgr)
        if bucket == "B":
            return nir_nir_night_clahe(bgr)
        if bucket == "C":
            if lite:
                return nir_nir_night_clahe(bgr)
            return nir_anti_glare_bgr(bgr)
        if bucket == "D":
            return nir_dehaze_lite(bgr)
        if bucket == "E":
            rain.reset()
            rain.process(bgr)
            rain.process(bgr)
            return rain.process(bgr)
        if bucket == "F":
            # Just do a simple blend approximation for visual display
            green = bgr[:, :, 1].astype(np.float32)
            ema = float(green.mean()) / 255.0
            enh = throughput_enhancer if lite else quality_enhancer
            enh.reset()
            return nir_transition_blend(bgr, enh, ema)
            
        enh = throughput_enhancer if lite else quality_enhancer
        enh.reset()
        return enh.process(bgr)

    return run_profile

def build_single_gallery(frame: dict, bucket: str, run_profile) -> np.ndarray:
    col_headers = [_col_header(col_name) for col_name, _ in COLUMNS]
    header_h = max(h.shape[0] for h in col_headers)
    padded_headers = []
    for h in col_headers:
        pad = np.ones((header_h - h.shape[0], THUMB_W, 3), dtype=np.uint8) * 50
        padded_headers.append(np.vstack([pad, h]))
    header_row = np.hstack(padded_headers)

    img_path = frame["path"]
    env_cls = frame.get("env_class", "?")
    bgr_raw = cv.imread(img_path, cv.IMREAD_COLOR)
    if bgr_raw is None:
        return header_row
    
    bgr_raw = cv.resize(bgr_raw, (320, 240))

    cells = []
    for _, profile in COLUMNS:
        try:
            out = run_profile(profile, bucket, bgr_raw)
        except Exception as e:
            out = bgr_raw
        cell = _thumb(out)
        cell = _label_cell(cell, f"ENV: {env_cls}" if profile == "raw" else "")
        cells.append(cell)

    body = np.hstack(cells)
    
    if header_row.shape[1] < body.shape[1]:
        pad = np.ones((header_row.shape[0], body.shape[1] - header_row.shape[1], 3), dtype=np.uint8) * 50
        header_row = np.hstack([header_row, pad])

    return np.vstack([header_row, body])

def main() -> None:
    manifest_path = ROOT / "data/eval/nir_val/manifest_v2.csv"
    out_dir = ROOT / "Thesis_report/figures/ch5_implementation"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(0)

    manifest = _load_manifest(manifest_path)
    if not manifest:
        sys.exit(0)

    run_profile = _make_dispatch()
    
    for bucket in ["A", "B", "C", "D", "E", "F"]:
        frame = _select_frame_for_bucket(manifest, bucket)
        if frame:
            gallery = build_single_gallery(frame, bucket, run_profile)
            out_file = out_dir / f"bucket_{bucket.lower()}_profiles.png"
            cv.imwrite(str(out_file), gallery)
            print(f"Generated {out_file}")

if __name__ == "__main__":
    main()
