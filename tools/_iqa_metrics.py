"""_iqa_metrics.py — Shared proxy IQA metrics for offline NIR evaluation.

These metrics are free from BRISQUE's natural-image-statistics assumption (Risk R1)
and are suitable for NIR night imagery. Each function accepts a uint8 grayscale array.

Metrics:
    log_rms_contrast    — log10 of RMS local contrast (higher = more texture)
    pct_saturated       — fraction of pixels above sat_thresh (default 250)
    pct_crushed         — fraction of pixels below crush_thresh (default 5)
    hist_entropy        — Shannon entropy of the 256-bin histogram (higher = richer tonal range)
    local_contrast_std  — std of local (8×8 tile) mean brightness (higher = more spatial variation)
"""

from __future__ import annotations

import math
from typing import Dict

import cv2 as cv
import numpy as np


def compute_iqa_metrics(
    gray: np.ndarray,
    *,
    sat_thresh: int = 250,
    crush_thresh: int = 5,
    tile_size: int = 8,
) -> Dict[str, float]:
    """Return a dict of proxy IQA metrics for a uint8 grayscale image.

    Args:
        gray: H×W uint8 grayscale array.
        sat_thresh: pixel values >= this count as saturated.
        crush_thresh: pixel values <= this count as crushed.
        tile_size: tile side for local_contrast_std.

    Returns:
        Dict with keys: log_rms_contrast, pct_saturated, pct_crushed,
        hist_entropy, local_contrast_std.
    """
    if gray.ndim != 2:
        raise ValueError(f"Expected 2-D grayscale, got shape {gray.shape}")
    arr = gray.astype(np.float32)
    n = arr.size

    # --- RMS local contrast (Laplacian response) ---
    lap = cv.Laplacian(gray, cv.CV_32F)
    rms = float(np.sqrt(np.mean(lap ** 2)))
    log_rms = math.log10(rms + 1e-6)

    # --- Saturation / crush fractions ---
    pct_sat = float(np.sum(gray >= sat_thresh)) / n
    pct_crush = float(np.sum(gray <= crush_thresh)) / n

    # --- Histogram entropy ---
    hist = cv.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    prob = hist / (hist.sum() + 1e-9)
    entropy = float(-np.sum(prob[prob > 0] * np.log2(prob[prob > 0])))

    # --- Local contrast std (tile means) ---
    h, w = gray.shape
    tile_means = []
    for r in range(0, h - tile_size + 1, tile_size):
        for c in range(0, w - tile_size + 1, tile_size):
            tile_means.append(float(np.mean(arr[r:r + tile_size, c:c + tile_size])))
    lc_std = float(np.std(tile_means)) if len(tile_means) > 1 else 0.0

    return {
        "log_rms_contrast": round(log_rms, 5),
        "pct_saturated": round(pct_sat, 6),
        "pct_crushed": round(pct_crush, 6),
        "hist_entropy": round(entropy, 5),
        "local_contrast_std": round(lc_std, 5),
    }
