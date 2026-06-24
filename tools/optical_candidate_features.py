#!/usr/bin/env python3
"""Non-production candidate optical features for SmartBinocular ML ablations."""

from __future__ import annotations

import cv2 as cv
import numpy as np

from tools.ml_metadata_utils import FEATURE_KEYS

OPTICAL_V2_FEATURES = [
    "local_contrast_mean",
    "local_contrast_std",
    "edge_density",
    "dark_channel_mean",
    "dark_channel_p95",
    "highlight_connected_component_area",
    "saturated_component_count",
    "p99_p1_dynamic_range",
]

OPTICAL_21_FEATURES = [
    *FEATURE_KEYS,
    "local_contrast_mean",
    "local_contrast_std",
    "edge_density",
    "dark_channel_mean",
    "dark_channel_p95",
    "highlight_connected_component_area",
    "saturated_component_count",
    "p99_p1_dynamic_range",
    "temporal_brightness_std",
]

OPTICAL_21_STILL_FEATURES = [
    *FEATURE_KEYS,
    *OPTICAL_V2_FEATURES,
]

OPTICAL_21_TEMPORAL_FEATURES = list(OPTICAL_21_FEATURES)

FEATURE_SET_DEFINITIONS = {
    "optical_12_baseline": {
        "features": list(FEATURE_KEYS),
        "actual_feature_count": len(FEATURE_KEYS),
        "applicability": "still image and runtime frame rows",
        "status": "production-compatible baseline",
    },
    "optical_v2_candidate": {
        "features": [*FEATURE_KEYS, *OPTICAL_V2_FEATURES],
        "actual_feature_count": len(FEATURE_KEYS) + len(OPTICAL_V2_FEATURES),
        "applicability": "still image rows with verified image paths",
        "status": "non-production research candidate",
    },
    "optical_21_candidate_still": {
        "features": list(OPTICAL_21_STILL_FEATURES),
        "actual_feature_count": len(OPTICAL_21_STILL_FEATURES),
        "applicability": "still image rows with verified image paths; temporal-only fields excluded",
        "status": "non-production 21-derived still-compatible candidate; not the full temporal 21-feature set",
    },
    "optical_21_candidate_temporal": {
        "features": list(OPTICAL_21_TEMPORAL_FEATURES),
        "actual_feature_count": len(OPTICAL_21_TEMPORAL_FEATURES),
        "applicability": "sequential video/runtime rows only",
        "status": "analysis/future labeled sequence candidate, not supervised still-image classifier evidence",
    },
}

FEATURE_GROUPS = {
    "brightness": ["nir_mean_brightness", "nir_std", "nir_p95", "nir_dark_fraction", "p99_p1_dynamic_range", "temporal_brightness_std"],
    "contrast": ["local_contrast_mean", "local_contrast_std", "p99_p1_dynamic_range"],
    "texture": ["nir_sharpness", "edge_density"],
    "haze": ["dark_channel_mean", "dark_channel_p95"],
    "glare": ["nir_glare_score", "highlight_connected_component_area", "saturated_component_count"],
    "color": ["nir_saturation_mean", "nir_blue_mean_ema"],
    "temporal": ["hour_of_day_sin", "hour_of_day_cos", "prev_env_class", "temporal_brightness_std"],
    "metadata_context": ["hour_of_day_sin", "hour_of_day_cos", "prev_env_class"],
}


def compute_candidate_features(bgr: np.ndarray, *, temporal_brightness_std: float | None = None) -> dict[str, float | None]:
    if bgr is None or bgr.size == 0:
        raise ValueError("bgr image is empty")
    small = bgr
    h, w = bgr.shape[:2]
    if max(h, w) > 256:
        scale = 256.0 / max(h, w)
        small = cv.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv.INTER_AREA)
    gray = cv.cvtColor(small, cv.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32)

    blur = cv.GaussianBlur(gray_f, (9, 9), 0)
    local_abs = np.abs(gray_f - blur)
    edges = cv.Canny(gray, 50, 150)
    dark_channel = np.min(small, axis=2).astype(np.float32)
    saturated = (gray >= 245).astype(np.uint8)
    n_labels, labels, stats, _ = cv.connectedComponentsWithStats(saturated, connectivity=8)
    areas = [int(stats[i, cv.CC_STAT_AREA]) for i in range(1, n_labels)]
    total = float(gray.size)
    p99 = float(np.percentile(gray_f, 99))
    p1 = float(np.percentile(gray_f, 1))

    return {
        "local_contrast_mean": float(local_abs.mean()),
        "local_contrast_std": float(local_abs.std()),
        "edge_density": float(np.count_nonzero(edges) / total),
        "dark_channel_mean": float(dark_channel.mean()),
        "dark_channel_p95": float(np.percentile(dark_channel, 95)),
        "highlight_connected_component_area": float(max(areas) / total) if areas else 0.0,
        "saturated_component_count": float(len([a for a in areas if a >= 4])),
        "p99_p1_dynamic_range": p99 - p1,
        "temporal_brightness_std": temporal_brightness_std,
    }
