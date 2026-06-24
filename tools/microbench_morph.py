"""microbench_morph.py — Compare scipy.ndimage vs cv2 morphology on 320×240 NIR.

Usage:
    python tools/microbench_morph.py [--n N] [--ksize K] [--out PATH]

Outputs a CSV row to stdout and optionally appends to docs/tables/timing/morph_microbench.csv.
Result informs whether nir_pipeline.py should prefer OpenCV erode/dilate over
scipy.ndimage minimum_filter/maximum_filter (the current primary path).
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import sys
import time
from pathlib import Path


def _bench(func, arr, n: int) -> float:
    """Return mean wall-clock ms over n iterations (first is warm-up, excluded)."""
    func(arr)  # warm-up
    t0 = time.perf_counter()
    for _ in range(n):
        func(arr)
    return (time.perf_counter() - t0) / n * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200, help="Iterations per backend (default 200)")
    parser.add_argument("--ksize", type=int, default=5, help="Kernel / filter size (default 5)")
    parser.add_argument("--out", default=None, help="Append CSV row to this path (optional)")
    args = parser.parse_args()

    n: int = args.n
    ksize: int = args.ksize

    import numpy as np
    import cv2 as cv

    rng = np.random.default_rng(42)
    frame = rng.integers(0, 256, size=(240, 320), dtype=np.uint8).astype(np.float32)
    kernel = np.ones((ksize, ksize), dtype=np.uint8)

    # OpenCV paths (always available)
    def cv_min(a: np.ndarray) -> np.ndarray:
        return cv.erode(a.astype(np.float32), kernel)

    def cv_max(a: np.ndarray) -> np.ndarray:
        return cv.dilate(a.astype(np.float32), kernel)

    cv_min_ms = _bench(cv_min, frame, n)
    cv_max_ms = _bench(cv_max, frame, n)

    # SciPy path (optional)
    scipy_min_ms: float | None = None
    scipy_max_ms: float | None = None
    has_scipy = False
    try:
        from scipy.ndimage import minimum_filter, maximum_filter
        has_scipy = True

        def sp_min(a: np.ndarray) -> np.ndarray:
            return minimum_filter(a, size=(ksize, ksize), mode="reflect")

        def sp_max(a: np.ndarray) -> np.ndarray:
            return maximum_filter(a, size=(ksize, ksize), mode="reflect")

        scipy_min_ms = _bench(sp_min, frame, n)
        scipy_max_ms = _bench(sp_max, frame, n)
    except ImportError:
        pass

    recommendation = "N/A (scipy not installed)"
    if has_scipy and scipy_min_ms is not None and scipy_max_ms is not None:
        if (cv_min_ms + cv_max_ms) < (scipy_min_ms + scipy_max_ms):
            recommendation = "OpenCV"
        else:
            recommendation = "SciPy"

    row = {
        "platform": platform.machine(),
        "python": platform.python_version(),
        "ksize": ksize,
        "n_iters": n,
        "frame_shape": "240x320",
        "cv_erode_ms": round(cv_min_ms, 4),
        "cv_dilate_ms": round(cv_max_ms, 4),
        "scipy_min_ms": round(scipy_min_ms, 4) if scipy_min_ms is not None else "N/A",
        "scipy_max_ms": round(scipy_max_ms, 4) if scipy_max_ms is not None else "N/A",
        "has_scipy": has_scipy,
        "recommendation": recommendation,
    }

    header = list(row.keys())
    writer = csv.DictWriter(sys.stdout, fieldnames=header)
    writer.writeheader()
    writer.writerow(row)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not out_path.exists()
        with open(out_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if write_header:
                w.writeheader()
            w.writerow(row)
        print(f"\nAppended to {out_path}", file=sys.stderr)

    print(f"\nRecommendation for nir_pipeline.py: use {recommendation}", file=sys.stderr)


if __name__ == "__main__":
    main()
