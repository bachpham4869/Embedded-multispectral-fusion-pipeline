"""Generate thermal .npy sequences from the scaled MI48 PNG dataset.

Converts the PNG images in data/thermal/scaled_mi48_80x62/ into .npy arrays
suitable for the sweep_3dnr_alpha.py and sweep_kalman_qr.py tools.

The PNGs are from different sources (indicated by filename prefixes), so we
group them into mini-sequences of 30 frames each for meaningful temporal
analysis. These are NOT consecutive real MI48 frames — they are a surrogate
dataset for parameter sensitivity analysis. This is documented as a proxy
evaluation.

Usage:
    python tools/gen_thermal_sweep_sequences.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "data" / "thermal" / "scaled_mi48_80x62"
OUT_3DNR = ROOT / "data" / "eval" / "thermal_seq"
OUT_KALMAN = ROOT / "data" / "thermal_sequences"
FRAMES_PER_SEQ = 30


def main() -> None:
    pngs = sorted(SRC_DIR.glob("*.png"))
    if not pngs:
        print(f"ERROR: No PNGs in {SRC_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pngs)} thermal PNGs in {SRC_DIR}")

    # Load all frames as float32 grayscale
    all_frames = []
    for p in pngs:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None and img.shape == (62, 80):
            all_frames.append(img.astype(np.float32))

    print(f"Loaded {len(all_frames)} valid frames (62x80)")

    # Group into sequences for 3DNR sweep
    OUT_3DNR.mkdir(parents=True, exist_ok=True)
    seq_count = 0
    for start in range(0, len(all_frames) - FRAMES_PER_SEQ + 1, FRAMES_PER_SEQ):
        chunk = np.stack(all_frames[start:start + FRAMES_PER_SEQ], axis=0)
        out_path = OUT_3DNR / f"thermal_seq_{seq_count:03d}.npy"
        np.save(out_path, chunk)
        seq_count += 1
    print(f"Saved {seq_count} sequences to {OUT_3DNR} (3DNR sweep)")

    # Also create a single stacked sequence for Kalman sweep
    OUT_KALMAN.mkdir(parents=True, exist_ok=True)
    big_seq = np.stack(all_frames[:min(200, len(all_frames))], axis=0)
    kalman_path = OUT_KALMAN / "thermal_surrogate_200frames.npy"
    np.save(kalman_path, big_seq)
    print(f"Saved Kalman sequence: {kalman_path} (shape={big_seq.shape})")


if __name__ == "__main__":
    main()
