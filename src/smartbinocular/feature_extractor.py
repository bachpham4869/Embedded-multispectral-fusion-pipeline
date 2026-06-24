"""Optical-first per-frame features for environment classification and ML logging.

``main`` and offline tools build :class:`~smartbinocular.feature_schema.FeatureRecord`
instances via :class:`FeatureExtractor` and :class:`~smartbinocular.utils.FrameCache`.

Design constraints:
  - Pure optical path; no import from hardware.py or thermal_pipeline.py.
  - Shared between offline pipeline (Mac) and runtime (RPi).
  - Thermal features use simple stats from raw thermal_80 array when available.
    Runtime integration in ``main`` may override ``thm_*`` fields with
    pre-computed :class:`~smartbinocular.thermal_pipeline.ThermalProcessor` outputs
    for better background alignment.
  - C7: Motion and temporal features are computed only when has_motion/has_temporal
    is explicitly asserted True by the caller (video sequences only).
  - C8: No zero-imputation; missing optional features remain None.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import cv2 as cv
import numpy as np

from smartbinocular.feature_schema import (
    ENV_CLASS_TO_INT,
    ENV_INT_UNKNOWN,
    FeatureRecord,
    encode_hour,
)
from smartbinocular.utils import FrameCache


# ── Optical stat helpers ───────────────────────────────────────────────────────

def _compute_entropy(gray: np.ndarray) -> float:
    """Shannon entropy of the intensity histogram (bits)."""
    hist = cv.calcHist([gray], [0], None, [256], [0.0, 256.0]).flatten()
    hist = hist / (hist.sum() + 1e-10)
    nz = hist[hist > 0]
    return float(-np.sum(nz * np.log2(nz)))


def _nir_saturation_mean_bgr(small_bgr: np.ndarray) -> float:
    """Mean OpenCV HSV S-channel on small BGR (same resize as nir_gray). Range [0, 255]."""
    hsv = cv.cvtColor(small_bgr, cv.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean())


def _optical_stats(gray: np.ndarray) -> tuple[float, float, float, float, float, float, float]:
    """Return (mean, std, entropy, p95, glare_score, sharpness, dark_fraction).

    gray: uint8 single-channel image (any size; smaller = faster).
    All values are float.
    """
    gray_f = gray.astype(np.float32)
    mean = float(gray_f.mean())
    std = float(gray_f.std())
    p95 = float(np.percentile(gray_f, 95))
    total = gray.size
    glare_score = float(np.count_nonzero(gray > 240) / total)
    dark_fraction = float(np.count_nonzero(gray < 30) / total)
    entropy = _compute_entropy(gray)
    # Laplacian variance — focus/sharpness proxy
    lap = cv.Laplacian(gray, cv.CV_64F)
    sharpness = float(lap.var())
    return mean, std, entropy, p95, glare_score, sharpness, dark_fraction


# ── Thermal stat helpers ───────────────────────────────────────────────────────

def _thermal_stats(
    thermal_raw: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """Return (mean, std, max, p95_p05_delta, fg_fraction, anomaly_score).

    thermal_raw: uint8 or float32 array from sensor (e.g., 80×62).
    These are simple offline-compatible stats; runtime may use ThermalProcessor
    outputs instead when the live pipeline supplies them.
    """
    t = thermal_raw.astype(np.float32).flatten()
    mean = float(t.mean())
    std = float(t.std())
    tmax = float(t.max())
    p95 = float(np.percentile(t, 95))
    p05 = float(np.percentile(t, 5))
    p95_p05_delta = p95 - p05
    # Foreground fraction: pixels significantly above mean (simple BG model)
    fg_threshold = mean + 1.5 * std
    fg_fraction = float(np.count_nonzero(t > fg_threshold) / t.size)
    # Anomaly score: how extreme is the hottest pixel relative to robust center?
    median = float(np.median(t))
    anomaly_score = float((tmax - median) / (std + 1e-6))
    return mean, std, tmax, p95_p05_delta, fg_fraction, anomaly_score


# ── Motion helpers ─────────────────────────────────────────────────────────────

_LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
)
_FEATURE_PARAMS = dict(maxCorners=50, qualityLevel=0.3, minDistance=7, blockSize=7)


def _motion_magnitude(gray_prev: np.ndarray, gray_curr: np.ndarray) -> float:
    """Mean sparse LK optical flow magnitude (px/frame). Returns 0.0 on failure."""
    pts = cv.goodFeaturesToTrack(gray_prev, mask=None, **_FEATURE_PARAMS)
    if pts is None or len(pts) < 4:
        return 0.0
    pts_next, status, _ = cv.calcOpticalFlowPyrLK(
        gray_prev, gray_curr, pts, None, **_LK_PARAMS
    )
    if pts_next is None:
        return 0.0
    good_mask = (status.flatten() == 1)
    if good_mask.sum() < 2:
        return 0.0
    delta = (pts_next[good_mask] - pts[good_mask]).squeeze()
    magnitudes = np.linalg.norm(delta.reshape(-1, 2), axis=1)
    return float(magnitudes.mean())


# ── FeatureExtractor ───────────────────────────────────────────────────────────

class FeatureExtractor:
    """Stateful feature extractor shared between offline and runtime pipelines.

    Maintains rolling windows for temporal features and previous frame state
    for motion computation. Call reset_sequence() between unrelated sequences
    (e.g., different video files or dataset sources).
    """

    TEMPORAL_WINDOW: int = 10

    # EMA smoothing factor for nir_blue_mean_ema (B-channel mean over nir_small_bgr).
    # Matches ml_posterior_ema_alpha default for consistency; not the same as NIR_B_EMA in main.py.
    BLUE_EMA_ALPHA: float = 0.55

    def __init__(self) -> None:
        self._nir_brightness_window: deque[float] = deque(maxlen=self.TEMPORAL_WINDOW)
        self._thm_mean_window: deque[float] = deque(maxlen=self.TEMPORAL_WINDOW)
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_motion_mag: float = 0.0
        self._nir_blue_mean_ema: Optional[float] = None  # seeded on first frame

    def reset_sequence(self) -> None:
        """Reset all stateful windows. Call at sequence boundaries (C7)."""
        self._nir_brightness_window.clear()
        self._thm_mean_window.clear()
        self._prev_gray = None
        self._prev_motion_mag = 0.0
        self._nir_blue_mean_ema = None  # reset EMA between unrelated sequences

    def extract(
        self,
        cache: FrameCache,
        nir_channel: str,
        thermal_channel: str,
        ts: float,
        *,
        prev_env_class: int = ENV_INT_UNKNOWN,
        source: str = "rpi",
        frame_idx: int = 0,
        has_motion: bool = False,
        has_temporal: bool = False,
        label: Optional[str] = None,
        label_source: Optional[str] = None,
        weak_label: Optional[str] = None,
        label_confidence: Optional[float] = None,
    ) -> FeatureRecord:
        """Extract a FeatureRecord from a FrameCache.

        Parameters
        ----------
        cache:            Precomputed FrameCache (nir_gray + nir_small_bgr for optical stats).
        nir_channel:      "nir" or "rgb" — determines normalization group.
        thermal_channel:  "lwir" or "none" — gates thermal feature extraction.
        ts:               Unix timestamp (0.0 for offline with unknown time).
        prev_env_class:   Previous ENV class int (0 = unknown).
        source:           Source tag (e.g. "offline_mwd", "rpi").
        has_motion:       If True, compute LK optical flow. MUST be False for
                          still images (C7 — no temporal/motion simulation).
        has_temporal:     If True, compute brightness delta from rolling window.
                          MUST be False unless ≥TEMPORAL_WINDOW frames seen.
        """
        gray = cache.nir_gray  # uint8, max_side=128

        # ── CORE optical stats ────────────────────────────────────────────────
        mean, std, entropy, p95, glare, sharpness, dark_frac = _optical_stats(gray)
        sat_mean = _nir_saturation_mean_bgr(cache.nir_small_bgr)
        hour_sin, hour_cos = encode_hour(ts)

        # ── Blue-channel EMA (nir_blue_mean_ema — feature #12) ───────────────
        # B channel = index 0 in BGR. DISTINCT from main.py's nir_b_ema (brightness EMA).
        blue_mean = float(cache.nir_small_bgr[:, :, 0].mean())
        if self._nir_blue_mean_ema is None:
            self._nir_blue_mean_ema = blue_mean  # seed on first frame
        else:
            a = self.BLUE_EMA_ALPHA
            self._nir_blue_mean_ema = a * blue_mean + (1.0 - a) * self._nir_blue_mean_ema

        # ── THERMAL (optional) ────────────────────────────────────────────────
        thm_mean = thm_std = thm_max = thm_p95p05 = thm_fg = thm_anom = None
        thermal_available = False
        if thermal_channel == "lwir":
            t = cache.thermal_80
            # Only extract if thermal data looks valid (not all-zero dummy)
            if t is not None and t.max() > 0:
                thm_mean, thm_std, thm_max, thm_p95p05, thm_fg, thm_anom = _thermal_stats(t)
                thermal_available = True

        # ── MOTION (optional, video only — C7) ────────────────────────────────
        motion_mag = motion_jerk = None
        motion_available = False
        if has_motion:
            if self._prev_gray is not None:
                motion_mag = _motion_magnitude(self._prev_gray, gray)
                motion_jerk = abs(motion_mag - self._prev_motion_mag)
                motion_available = True
            self._prev_gray = gray.copy()
            self._prev_motion_mag = motion_mag if motion_mag is not None else 0.0
        else:
            # Reset prev_gray when motion is not being tracked (C7: no leakage)
            self._prev_gray = None

        # ── Update NIR brightness window (always, for future temporal use) ────
        self._nir_brightness_window.append(mean)
        if thm_mean is not None:
            self._thm_mean_window.append(thm_mean)

        # ── TEMPORAL (optional, video with ≥TEMPORAL_WINDOW frames — C7) ─────
        nir_delta = thm_delta = None
        temporal_available = False
        if has_temporal and len(self._nir_brightness_window) >= self.TEMPORAL_WINDOW:
            window_mean = float(np.mean(list(self._nir_brightness_window)[:-1]))
            nir_delta = mean - window_mean
            temporal_available = True
            if thm_mean is not None and len(self._thm_mean_window) >= self.TEMPORAL_WINDOW:
                thm_window_mean = float(np.mean(list(self._thm_mean_window)[:-1]))
                thm_delta = thm_mean - thm_window_mean

        return FeatureRecord(
            # CORE
            nir_mean_brightness=mean,
            nir_std=std,
            nir_entropy=entropy,
            nir_p95=p95,
            nir_glare_score=glare,
            nir_sharpness=sharpness,
            nir_dark_fraction=dark_frac,
            nir_saturation_mean=sat_mean,
            hour_of_day_sin=hour_sin,
            hour_of_day_cos=hour_cos,
            prev_env_class=prev_env_class,
            nir_blue_mean_ema=self._nir_blue_mean_ema,
            # THERMAL
            thm_mean=thm_mean,
            thm_std=thm_std,
            thm_max=thm_max,
            thm_p95_p05_delta=thm_p95p05,
            thm_fg_fraction=thm_fg,
            thm_anomaly_score=thm_anom,
            # MOTION
            motion_magnitude=motion_mag,
            motion_jerk=motion_jerk,
            # TEMPORAL
            nir_brightness_delta_10f=nir_delta,
            thm_mean_delta_10f=thm_delta,
            # RUNTIME ONLY — always None in extracted records (set by main.py)
            skew_ms=None,
            fusion_alpha=None,
            # METADATA
            ts=ts,
            frame_idx=frame_idx,
            source=source,
            nir_channel=nir_channel,
            thermal_channel=thermal_channel,
            thermal_available=thermal_available,
            motion_available=motion_available,
            temporal_available=temporal_available,
            # LABELS
            label=label,
            label_source=label_source,
            weak_label=weak_label,
            label_confidence=label_confidence,
        )
