"""sweep_3dnr_alpha.py — Sweep thermal_3dnr_alpha (IIR temporal smoothing).

Applies the 3DNR temporal IIR filter at various alpha values to a sequence of
thermal frames (.npy arrays, shape [H, W] float32 or uint8) and measures
frame-level residual noise and PSNR against a reference (first frame).

THESIS_SKIP guard: If data/eval/thermal_seq/ is empty or absent, this script
exits 0 with a THESIS_SKIP note. Run after recording MI48 .npy sequences on RPi4:
    mkdir -p data/eval/thermal_seq/
    # Transfer .npy frames from RPi4 (MI48 80×62 thermal sensor)
    rsync -avz pi@raspberrypi.local:~/smartBinocular/thermal_captures/*.npy data/eval/thermal_seq/

Usage:
    python tools/sweep_3dnr_alpha.py
    python tools/sweep_3dnr_alpha.py \\
        --seq-dir data/eval/thermal_seq/ \\
        --alphas 0.40,0.50,0.60,0.65,0.70,0.80,0.85 \\
        --out docs/thesis_eval/thermal/tables/3dnr_alpha_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SEQ_DIR = ROOT / "data/eval/thermal_seq"
DEFAULT_ALPHAS = [0.40, 0.50, 0.60, 0.65, 0.70, 0.80, 0.85]
DEFAULT_OUT = ROOT / "docs/thesis_eval/thermal/tables/3dnr_alpha_sweep.csv"
PRODUCTION_ALPHA = 0.65


def _find_sequences(seq_dir: Path) -> List[Path]:
    return sorted(seq_dir.glob("*.npy"))


def _psnr(ref: np.ndarray, img: np.ndarray, max_val: float = 255.0) -> float:
    mse = float(np.mean((ref.astype(float) - img.astype(float)) ** 2))
    if mse < 1e-10:
        return 100.0
    return 10 * math.log10(max_val ** 2 / mse)


def _residual_noise(frames: List[np.ndarray]) -> float:
    """Mean frame-to-frame absolute difference (proxy for residual noise)."""
    if len(frames) < 2:
        return 0.0
    diffs = [float(np.mean(np.abs(frames[i].astype(float) - frames[i-1].astype(float))))
             for i in range(1, len(frames))]
    return float(np.mean(diffs))


def _apply_3dnr(frames: List[np.ndarray], alpha: float) -> List[np.ndarray]:
    """Apply IIR temporal filter: out[t] = alpha * out[t-1] + (1-alpha) * in[t]."""
    out = []
    prev = frames[0].astype(float)
    for f in frames:
        smoothed = alpha * prev + (1.0 - alpha) * f.astype(float)
        out.append(np.clip(smoothed, 0, 255).astype(np.uint8))
        prev = smoothed
    return out


def run_sweep(seq_files: List[Path], alphas: List[float]) -> List[dict]:
    rows = []
    for seq_file in seq_files:
        try:
            data = np.load(seq_file)
        except Exception as e:
            print(f"  Skip {seq_file.name}: {e}", file=sys.stderr)
            continue

        # Handle (N, H, W) or (H, W) arrays
        if data.ndim == 2:
            frames = [data]
        elif data.ndim == 3:
            frames = [data[i] for i in range(data.shape[0])]
        else:
            print(f"  Skip {seq_file.name}: unexpected shape {data.shape}", file=sys.stderr)
            continue

        if len(frames) < 3:
            print(f"  Skip {seq_file.name}: too few frames ({len(frames)})", file=sys.stderr)
            continue

        ref_frame = frames[0].astype(float)
        raw_noise = _residual_noise(frames)

        for alpha in alphas:
            smoothed = _apply_3dnr(frames, alpha)
            sm_noise = _residual_noise(smoothed)
            sm_psnr = float(np.mean([_psnr(ref_frame, s.astype(float)) for s in smoothed[1:]]))
            rows.append({
                "seq_file": seq_file.name,
                "n_frames": len(frames),
                "alpha": alpha,
                "is_production_default": alpha == PRODUCTION_ALPHA,
                "raw_residual_noise": round(raw_noise, 4),
                "smoothed_residual_noise": round(sm_noise, 4),
                "noise_reduction_pct": round(100 * (1 - sm_noise / (raw_noise + 1e-9)), 2),
                "psnr_vs_first_frame_db": round(sm_psnr, 3),
            })

    return rows


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq-dir", type=Path, default=DEFAULT_SEQ_DIR)
    p.add_argument("--alphas", default=",".join(str(a) for a in DEFAULT_ALPHAS))
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]

    seq_files = _find_sequences(args.seq_dir) if args.seq_dir.exists() else []
    if not seq_files:
        print(f"THESIS_SKIP: no .npy sequence files in {args.seq_dir}", file=sys.stderr)
        print("Record MI48 thermal sequences on RPi4 and transfer to data/eval/thermal_seq/", file=sys.stderr)
        sys.exit(0)

    print(f"Sweep: {len(seq_files)} sequences × {len(alphas)} alpha values")
    rows = run_sweep(seq_files, alphas)
    if not rows:
        print("No rows produced.", file=sys.stderr)
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {args.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
