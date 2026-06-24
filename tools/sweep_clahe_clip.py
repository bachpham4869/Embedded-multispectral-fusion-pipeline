"""
Proxy sweep of CLAHE clip values on committed NIR sample frames.

Usage:
    python tools/sweep_clahe_clip.py [--frame-dir PATH] [--out-dir PATH]
                                     [--clips 0.5,2.0,4.0,6.0,8.0]
                                     [--bucket B]

Inputs:
    NIR sample frames (grayscale or BGR PNG/JPG) under --frame-dir
    (default: data/training/nir_samples/ or data/training/).
    Frames are expected to be representative nir_night scenes for Bucket B.
    For other buckets, change --bucket and point to the appropriate sub-directory.

Outputs:
    docs/tables/iqa/clahe_clip_sweep.csv  — one row per (frame_name, clip_value):
        frame, clip, log_rms_contrast, pct_saturated_new, pct_crushed_new
    docs/tables/iqa/clahe_clip_sweep_summary.md — per-clip aggregated means

Metrics:
    log_rms_contrast  = std(log(L + 1)) after CLAHE — quantifies global contrast lift
    pct_saturated_new = mean(L_out >= 250) - mean(L_in >= 250) — new highlight clipping
    pct_crushed_new   = mean(L_out <= 5) - mean(L_in <= 5) — new shadow crush

Proxy note: these metrics require no ground-truth labels and can be computed on any
    committed NIR sample. They are NOT a subjective quality metric — they measure
    whether a clip value over-amplifies (saturated_new > 2%) or under-amplifies
    (log_rms_contrast < baseline × 1.1). Use them to narrow the clip search range,
    then validate visually.

Failure mode:
    If no frames are found, the script exits 1 with a clear message.
    The evidence register marks Bucket B clip [0.5, 8.0] as
    PARTIAL — proxy sweep pending input frames.
"""
import argparse
import csv
import glob
import pathlib
import sys

import cv2 as cv
import numpy as np


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frame-dir", default=None,
                   help="Directory containing NIR sample frames (PNG/JPG)")
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: docs/tables/iqa/)")
    p.add_argument("--clips", default="0.5,1.0,2.0,3.0,4.0,6.0,8.0",
                   help="Comma-separated CLAHE clip values to sweep")
    p.add_argument("--bucket", default="B",
                   help="Which bucket this sweep targets (for CSV metadata; default: B)")
    return p.parse_args()


def _find_frames(frame_dir: pathlib.Path) -> list:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG")
    files = []
    for ext in exts:
        files.extend(sorted(frame_dir.glob(ext)))
    return files


def _extract_l_channel(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        return bgr.astype(np.float32)
    lab = cv.cvtColor(bgr, cv.COLOR_BGR2Lab)
    return lab[:, :, 0].astype(np.float32)


def _apply_clahe(gray_u8: np.ndarray, clip: float) -> np.ndarray:
    clahe = cv.createCLAHE(clipLimit=clip, tileGridSize=(4, 4))
    if gray_u8.dtype != np.uint8:
        gray_u8 = np.clip(gray_u8, 0, 255).astype(np.uint8)
    return clahe.apply(gray_u8).astype(np.float32)


def _metrics(l_in: np.ndarray, l_out: np.ndarray) -> dict:
    log_rms = float(np.std(np.log(l_out + 1.0)))
    pct_sat_new = float(np.mean(l_out >= 250) - np.mean(l_in >= 250))
    pct_crush_new = float(np.mean(l_out <= 5) - np.mean(l_in <= 5))
    return {
        "log_rms_contrast": round(log_rms, 6),
        "pct_saturated_new": round(pct_sat_new, 6),
        "pct_crushed_new": round(pct_crush_new, 6),
    }


def main():
    args = _parse_args()
    repo = pathlib.Path(__file__).parent.parent

    if args.frame_dir:
        frame_dir = pathlib.Path(args.frame_dir)
    else:
        candidates = [
            repo / "data/training/nir_samples",
            repo / "data/training",
            repo / "data",
        ]
        frame_dir = next((d for d in candidates if d.exists()), candidates[0])

    out_dir = pathlib.Path(args.out_dir) if args.out_dir else repo / "docs/tables/iqa"
    clips = [float(c.strip()) for c in args.clips.split(",")]

    frames = _find_frames(frame_dir)
    if not frames:
        print(
            f"ERROR: No image frames found in {frame_dir}.\n"
            f"Stage NIR sample frames there before running this sweep.\n"
            f"Example:\n"
            f"  cp /path/to/nir_samples/*.png {frame_dir}/\n"
            f"Until frames are committed, Bucket B CLAHE clip [0.5, 8.0] remains\n"
            f"PARTIAL evidence in docs/PIPELINE_EVIDENCE_REGISTER.md §D.5.",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "clahe_clip_sweep.csv"
    md_path = out_dir / "clahe_clip_sweep_summary.md"

    rows = []
    for fpath in frames:
        bgr = cv.imread(str(fpath))
        if bgr is None:
            print(f"WARNING: could not read {fpath.name}, skipping", file=sys.stderr)
            continue
        l_in = _extract_l_channel(bgr)
        l_in_u8 = np.clip(l_in, 0, 255).astype(np.uint8)
        for clip in clips:
            l_out = _apply_clahe(l_in_u8, clip)
            m = _metrics(l_in, l_out)
            rows.append({"frame": fpath.name, "clip": clip, "bucket": args.bucket, **m})

    with csv_path.open("w", newline="") as fh:
        fields = ["frame", "clip", "bucket", "log_rms_contrast", "pct_saturated_new", "pct_crushed_new"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    per_clip = {}
    for r in rows:
        c = r["clip"]
        if c not in per_clip:
            per_clip[c] = []
        per_clip[c].append(r)

    with md_path.open("w") as fh:
        fh.write(f"# CLAHE Clip Sweep Summary (Bucket {args.bucket})\n\n")
        fh.write(f"Frames: {len(frames)} | Clips tested: {clips}\n\n")
        fh.write("| clip | mean_log_rms_contrast | mean_pct_saturated_new | mean_pct_crushed_new |\n")
        fh.write("|------|-----------------------|------------------------|---------------------|\n")
        for clip in sorted(per_clip):
            vals = per_clip[clip]
            fh.write(
                f"| {clip} "
                f"| {round(sum(r['log_rms_contrast'] for r in vals)/len(vals), 6)} "
                f"| {round(sum(r['pct_saturated_new'] for r in vals)/len(vals), 6)} "
                f"| {round(sum(r['pct_crushed_new'] for r in vals)/len(vals), 6)} |\n"
            )

    try:
        print(f"Written: {csv_path.relative_to(repo)}")
        print(f"Written: {md_path.relative_to(repo)}")
    except ValueError:
        print(f"Written: {csv_path}")
        print(f"Written: {md_path}")
    print(f"Processed {len(frames)} frame(s) × {len(clips)} clip values = {len(rows)} rows.")


if __name__ == "__main__":
    main()
