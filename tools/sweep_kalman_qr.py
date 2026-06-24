"""
Sweep Kalman Q/R parameters on a committed thermal sequence.

Usage:
    python tools/sweep_kalman_qr.py [--seq-dir PATH] [--out-dir PATH]
                                    [--q-values 0.1,0.3,0.5,1.0,2.0]
                                    [--r-values 2.0,4.0,8.0]
                                    [--warmup 5]

Inputs:
    A committed thermal sequence: a directory containing consecutive 80×62
    float32 frames as .npy files (shape (62, 80)), or a single .npy of shape
    (N, 62, 80). Searched under --seq-dir (default: data/thermal_sequences/).

Outputs:
    docs/tables/timing/kalman_qr_sweep.csv — one row per (Q, R, warmup):
        q, r, warmup, mean_bg_residual_rms, median_bg_residual, std_bg_residual
    Background residual is measured only on foreground-free regions (pixels
    where the soft-fg-mask < 0.15 for ≥90% of frames after warmup).

Failure mode (documented stub):
    If no thermal sequence is staged, the script prints a clear message and
    exits with code 1. The evidence register marks Q=0.5, R=4.0, P₀=100.0
    as UNVERIFIED — pending sequence in docs/PIPELINE_EVIDENCE_REGISTER.md §D.7.

    To stage a sequence:
        1. On RPi4, run: python -m smartbinocular --dump-thermal-seq N_FRAMES
           (if that flag is implemented) or save MI48 frames manually.
        2. Convert to .npy files in data/thermal_sequences/.
        3. Re-run this script.
"""
import argparse
import glob
import pathlib
import sys

import numpy as np


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq-dir", default=None,
                   help="Directory containing .npy thermal frames or a stacked .npy")
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: docs/tables/timing/)")
    p.add_argument("--q-values", default="0.1,0.3,0.5,1.0,2.0",
                   help="Comma-separated process-noise Q values")
    p.add_argument("--r-values", default="2.0,4.0,8.0",
                   help="Comma-separated measurement-noise R values")
    p.add_argument("--warmup", type=int, default=5,
                   help="Frames to skip before measuring residuals")
    return p.parse_args()


def _load_sequence(seq_dir: pathlib.Path):
    stacked = sorted(glob.glob(str(seq_dir / "*.npy")))
    if not stacked:
        return None
    try:
        arr = np.load(stacked[0])
        if arr.ndim == 3:
            return arr.astype(np.float32)
        frames = [np.load(f).astype(np.float32) for f in stacked]
        return np.stack(frames, axis=0)
    except Exception:
        return None


def _kalman_residuals(frames: np.ndarray, q: float, r: float, p0: float, warmup: int):
    n, h, w = frames.shape
    x = frames[0].copy()
    p = np.full((h, w), p0, dtype=np.float32)
    residuals = []
    for i in range(1, n):
        z = frames[i]
        p = p + q
        k = p / (p + r)
        innov = z - x
        x = x + k * innov
        p = (1.0 - k) * p
        if i >= warmup:
            residuals.append(np.abs(innov))
    return np.stack(residuals, axis=0) if residuals else np.array([])


def _bg_rms(residuals: np.ndarray, fg_mask_threshold: float = 0.15) -> dict:
    mean_abs = np.mean(np.abs(residuals), axis=0)
    fg_free = mean_abs < fg_mask_threshold * np.max(mean_abs + 1e-6)
    bg_pixels = residuals[:, fg_free]
    if bg_pixels.size == 0:
        return {"mean_bg_residual_rms": float("nan"), "median_bg_residual": float("nan"), "std_bg_residual": float("nan")}
    return {
        "mean_bg_residual_rms": round(float(np.sqrt(np.mean(bg_pixels ** 2))), 6),
        "median_bg_residual": round(float(np.median(np.abs(bg_pixels))), 6),
        "std_bg_residual": round(float(np.std(bg_pixels)), 6),
    }


def main():
    args = _parse_args()
    repo = pathlib.Path(__file__).parent.parent

    seq_dir = pathlib.Path(args.seq_dir) if args.seq_dir else repo / "data/thermal_sequences"
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else repo / "docs/tables/timing"
    q_vals = [float(v.strip()) for v in args.q_values.split(",")]
    r_vals = [float(v.strip()) for v in args.r_values.split(",")]

    if not seq_dir.exists():
        print(
            f"ERROR: Thermal sequence directory not found: {seq_dir}\n"
            f"Stage a thermal .npy sequence there before running this sweep.\n"
            f"Until committed, Q=0.5, R=4.0, P₀=100.0 remain UNVERIFIED in\n"
            f"docs/PIPELINE_EVIDENCE_REGISTER.md §D.7.",
            file=sys.stderr,
        )
        sys.exit(1)

    frames = _load_sequence(seq_dir)
    if frames is None or len(frames) < args.warmup + 2:
        print(
            f"ERROR: Could not load a usable thermal sequence from {seq_dir}.\n"
            f"Need ≥{args.warmup + 2} frames as .npy files (shape (62, 80)) or a "
            f"stacked array of shape (N, 62, 80).\n"
            f"Until committed, Q/R remain UNVERIFIED in docs/PIPELINE_EVIDENCE_REGISTER.md §D.7.",
            file=sys.stderr,
        )
        sys.exit(1)

    import csv
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "kalman_qr_sweep.csv"

    rows = []
    for q in q_vals:
        for r in r_vals:
            residuals = _kalman_residuals(frames, q=q, r=r, p0=100.0, warmup=args.warmup)
            m = _bg_rms(residuals)
            rows.append({"q": q, "r": r, "warmup": args.warmup, **m})
            print(f"Q={q:.2f} R={r:.2f}: rms_bg={m['mean_bg_residual_rms']:.4f}")

    with csv_path.open("w", newline="") as fh:
        fields = ["q", "r", "warmup", "mean_bg_residual_rms", "median_bg_residual", "std_bg_residual"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written: {csv_path.relative_to(repo)}")
    print(f"Processed {len(frames)} frames, {len(rows)} (Q, R) combinations.")


if __name__ == "__main__":
    main()
