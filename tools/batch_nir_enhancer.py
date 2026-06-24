"""batch_nir_enhancer.py — Offline still-image IQA for NIR buckets A–F.

Routes each image through one or more bucket functions and measures proxy IQA
metrics (log_rms_contrast, pct_saturated, pct_crushed, hist_entropy,
local_contrast_std).

IMPORTANT — still_image_cold_start_mode (Bucket A):
    HybridNIREnhancer is stateful. This tool forces update_rate=1 and calls
    reset() per image so every image uses the heavy path. This does NOT replicate
    live-video temporal EMA behavior. All output is labelled
    run_mode=still_image_cold_start_mode.

Bucket E — single-frame surrogate:
    RainTemporalMedian requires N≥3 frames. In normal mode, each image is fed as
    a single still; the first n_frames-1 stills return the input unchanged (buffer
    not yet full). For a more meaningful evaluation use --rain-sequence-mode which
    feeds each image 3× (with noise/jitter) to trigger the median. Flagged in
    surrogate_mode column.

Bucket F — single-frame EMA proxy:
    nir_transition_blend uses nir_b_ema_norm (live EMA). Here we proxy it as
    mean(green_channel)/255 per still. This saturates to 0 or 1 more often than
    live EMA. Flagged in surrogate_mode column.

Usage:
    python tools/batch_nir_enhancer.py \\
        --manifest data/eval/nir_val/manifest_v2.csv \\
        --bucket A B C D E F \\
        --out docs/tables/iqa/batch_nir_iqa.csv

    # Mis-dispatch matrix (all buckets × all images ignoring env routing):
    python tools/batch_nir_enhancer.py \\
        --manifest data/eval/nir_val/manifest_v2.csv \\
        --matrix A B C D E F \\
        --out docs/tables/iqa/mis_dispatch_matrix.csv

    # Dry run (first 5 images):
    python tools/batch_nir_enhancer.py \\
        --manifest data/eval/nir_val/manifest_v2.csv --dry-run

Manifest CSV format: path, env_class [, edge_case, source_dir, mean_L, pct_sat, sha256]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from _iqa_metrics import compute_iqa_metrics  # type: ignore[import]
from smartbinocular.nir_pipeline import (
    HybridNIREnhancer,
    RainTemporalMedian,
    nir_anti_glare_bgr,
    nir_dehaze_lite,
    nir_nir_night_clahe,
    nir_transition_blend,
)

_ALL_BUCKETS = list("ABCDEF")

_BUCKET_FN_NAMES = {
    "A": "HybridNIREnhancer",
    "B": "nir_nir_night_clahe",
    "C": "nir_anti_glare_bgr",
    "D": "nir_dehaze_lite",
    "E": "RainTemporalMedian",
    "F": "nir_transition_blend",
}

_OUTPUT_FIELDS = [
    "path", "env_class", "edge_case", "bucket", "bucket_fn", "run_mode", "surrogate_mode",
    "before_log_rms_contrast", "before_pct_saturated", "before_pct_crushed",
    "before_hist_entropy", "before_local_contrast_std",
    "after_log_rms_contrast", "after_pct_saturated", "after_pct_crushed",
    "after_hist_entropy", "after_local_contrast_std",
    "proc_ms",
]


def _load_manifest(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = row.get("path") or row.get("image_path", "")
            env = row.get("env_class") or row.get("label", "unknown")
            if img_path:
                rows.append({
                    "path": img_path,
                    "env_class": env,
                    "edge_case": row.get("edge_case", ""),
                })
    return rows


def _load_bgr(img_path: str) -> Optional[np.ndarray]:
    bgr = cv.imread(img_path, cv.IMREAD_COLOR)
    return bgr  # None if unreadable


def _make_bucket_dispatch(
    enhancer: HybridNIREnhancer,
    rain_processor: RainTemporalMedian,
    rain_sequence_mode: bool,
) -> dict[str, Callable[[np.ndarray], tuple[np.ndarray, str]]]:
    """Return {letter: fn(bgr) -> (out_bgr, surrogate_mode)} for each bucket."""

    def bucket_a(bgr: np.ndarray) -> tuple[np.ndarray, str]:
        enhancer.reset()
        return enhancer.process(bgr), "still_image_cold_start_mode"

    def bucket_b(bgr: np.ndarray) -> tuple[np.ndarray, str]:
        return nir_nir_night_clahe(bgr), ""

    def bucket_c(bgr: np.ndarray) -> tuple[np.ndarray, str]:
        return nir_anti_glare_bgr(bgr), ""

    def bucket_d(bgr: np.ndarray) -> tuple[np.ndarray, str]:
        return nir_dehaze_lite(bgr), ""

    def bucket_e(bgr: np.ndarray) -> tuple[np.ndarray, str]:
        if rain_sequence_mode:
            # Feed 3-frame pseudo-sequence: original + 2 noise/jitter variants.
            rain_processor.reset()
            noise1 = bgr.copy().astype(np.int16)
            noise1 += np.random.randint(-2, 3, bgr.shape, dtype=np.int16)
            f1 = np.clip(noise1, 0, 255).astype(np.uint8)
            M = np.float32([[1, 0, 1], [0, 1, 0]])
            f2 = cv.warpAffine(bgr, M, (bgr.shape[1], bgr.shape[0]))
            rain_processor.process(bgr)
            rain_processor.process(f1)
            out = rain_processor.process(f2)
            surrogate = "synthetic_3stack"
        else:
            rain_processor.reset()
            out = rain_processor.process(bgr)
            surrogate = "single_still_prefill"
        return out, surrogate

    def bucket_f(bgr: np.ndarray) -> tuple[np.ndarray, str]:
        # Single-frame EMA proxy: nir_b_ema_norm = mean(green)/255.
        green = bgr[:, :, 1].astype(np.float32)
        nir_b_ema_norm = float(green.mean()) / 255.0
        out = nir_transition_blend(bgr, enhancer, nir_b_ema_norm)
        return out, "single_frame_ema_proxy"

    return {
        "A": bucket_a,
        "B": bucket_b,
        "C": bucket_c,
        "D": bucket_d,
        "E": bucket_e,
        "F": bucket_f,
    }


def run_batch(
    manifest: list[dict],
    buckets: list[str],
    dispatch: dict[str, Callable],
    matrix_mode: bool,
) -> list[dict]:
    """Process each manifest entry through each requested bucket.

    In matrix_mode, all buckets run on every image regardless of env_class.
    In normal mode, only the requested buckets run; env_class is preserved.
    """
    out_rows = []
    total = len(manifest) * len(buckets)
    done = 0

    for entry in manifest:
        img_path = entry["path"]
        env_class = entry["env_class"]
        edge_case = entry.get("edge_case", "")

        bgr = _load_bgr(img_path)
        if bgr is None:
            print(f"  SKIP (cannot read): {img_path}", file=sys.stderr)
            done += len(buckets)
            continue

        gray_before = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
        metrics_before = compute_iqa_metrics(gray_before)

        for bucket in buckets:
            fn = dispatch[bucket]
            try:
                t0 = time.perf_counter()
                out_bgr, surrogate_mode = fn(bgr)
                proc_ms = (time.perf_counter() - t0) * 1000.0
            except Exception as exc:
                print(f"  ERROR bucket {bucket} on {img_path}: {exc}", file=sys.stderr)
                done += 1
                continue

            gray_after = cv.cvtColor(out_bgr, cv.COLOR_BGR2GRAY)
            metrics_after = compute_iqa_metrics(gray_after)

            # run_mode label: A always has cold-start caveat; E/F have surrogate labels.
            if bucket == "A":
                run_mode = "still_image_cold_start_mode"
            elif bucket in ("E", "F"):
                run_mode = f"surrogate:{surrogate_mode}"
            else:
                run_mode = "stateless"

            out_rows.append({
                "path": img_path,
                "env_class": env_class,
                "edge_case": edge_case,
                "bucket": bucket,
                "bucket_fn": _BUCKET_FN_NAMES[bucket],
                "run_mode": run_mode,
                "surrogate_mode": surrogate_mode,
                "before_log_rms_contrast": metrics_before["log_rms_contrast"],
                "before_pct_saturated": metrics_before["pct_saturated"],
                "before_pct_crushed": metrics_before["pct_crushed"],
                "before_hist_entropy": metrics_before["hist_entropy"],
                "before_local_contrast_std": metrics_before["local_contrast_std"],
                "after_log_rms_contrast": metrics_after["log_rms_contrast"],
                "after_pct_saturated": metrics_after["pct_saturated"],
                "after_pct_crushed": metrics_after["pct_crushed"],
                "after_hist_entropy": metrics_after["hist_entropy"],
                "after_local_contrast_std": metrics_after["local_contrast_std"],
                "proc_ms": round(proc_ms, 3),
            })
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{total} rows...", file=sys.stderr)

    return out_rows


def _print_summary(out_rows: list[dict]) -> None:
    by_bucket: dict = defaultdict(list)
    for r in out_rows:
        by_bucket[r["bucket"]].append(r)
    print("\n--- per-bucket summary ---", file=sys.stderr)
    for b in _ALL_BUCKETS:
        rows = by_bucket.get(b)
        if not rows:
            continue
        delta_rms = [r["after_log_rms_contrast"] - r["before_log_rms_contrast"] for r in rows]
        mean_delta = sum(delta_rms) / len(delta_rms)
        pct_sat = sum(r["after_pct_saturated"] for r in rows) / len(rows)
        pct_crush = sum(r["after_pct_crushed"] for r in rows) / len(rows)
        print(
            f"  Bucket {b} ({_BUCKET_FN_NAMES[b]}): n={len(rows)}"
            f"  Δlog_rms={mean_delta:+.4f}  sat_after={pct_sat:.4f}"
            f"  crush_after={pct_crush:.4f}",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="CSV with columns: path, env_class")
    parser.add_argument(
        "--bucket", nargs="+", choices=_ALL_BUCKETS, default=None,
        help="Buckets to run in normal mode (default: A B D for legacy compat)",
    )
    parser.add_argument(
        "--matrix", nargs="+", choices=_ALL_BUCKETS, default=None,
        help="Run mis-dispatch matrix: all listed buckets × all images",
    )
    parser.add_argument("--out", default=None, help="Write results to this CSV file")
    parser.add_argument("--proc-w", type=int, default=320, help="HybridNIREnhancer proc width")
    parser.add_argument("--proc-h", type=int, default=240, help="HybridNIREnhancer proc height")
    parser.add_argument(
        "--rain-sequence-mode", action="store_true",
        help="Bucket E: feed synthetic 3-frame sequence instead of single still",
    )
    parser.add_argument("--dry-run", action="store_true", help="Process first 5 images only")
    args = parser.parse_args()

    if args.matrix and args.bucket:
        print("ERROR: --matrix and --bucket are mutually exclusive", file=sys.stderr)
        sys.exit(1)

    buckets = args.matrix or args.bucket or ["A", "B", "D"]
    matrix_mode = bool(args.matrix)

    manifest = _load_manifest(args.manifest)
    if not manifest:
        print(f"ERROR: no rows in manifest {args.manifest}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        manifest = manifest[:5]
        print(f"[dry-run] {len(manifest)} images × {len(buckets)} buckets", file=sys.stderr)

    enhancer = HybridNIREnhancer(
        proc_w=args.proc_w,
        proc_h=args.proc_h,
        update_rate=1,
    )
    rain_processor = RainTemporalMedian(n_frames=3)

    dispatch = _make_bucket_dispatch(enhancer, rain_processor, args.rain_sequence_mode)

    mode_label = "matrix" if matrix_mode else "normal"
    print(
        f"[batch_nir_enhancer] mode={mode_label}  "
        f"{len(manifest)} images × {len(buckets)} buckets = "
        f"{len(manifest) * len(buckets)} rows",
        file=sys.stderr,
    )

    out_rows = run_batch(manifest, buckets, dispatch, matrix_mode)

    if not out_rows:
        print("No output rows produced.", file=sys.stderr)
        sys.exit(1)

    writer = csv.DictWriter(sys.stdout, fieldnames=_OUTPUT_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(out_rows)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not out_path.exists()
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_OUTPUT_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nWrote {len(out_rows)} rows to {out_path}", file=sys.stderr)

    _print_summary(out_rows)


if __name__ == "__main__":
    main()
