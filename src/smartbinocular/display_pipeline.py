"""
Display post-processing after fusion and shake reduction in ``main``.

Functions here run in LAB space for Tier-S luminance caps and optional ENV-driven
grading. :class:`DisplayTemporalGlareBlend` temporally smooths highlights before
final grading. Outputs feed ``cv.imshow`` in the main loop.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import cv2 as cv


def display_luminance_cap_bgr(frame_bgr: np.ndarray, l_max: int = 240) -> np.ndarray:
    """Clamp LAB L channel to reduce screen glare (Tier S4).

    Used in ``luma_only`` display_grade_mode; for grading + cap in a single pass use
    :func:`display_grade_and_cap_bgr`.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr
    lab = cv.cvtColor(frame_bgr, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    np.minimum(l, min(int(l_max), 255), out=l)
    return cv.cvtColor(cv.merge([l, a, b]), cv.COLOR_LAB2BGR)


def display_grade_and_cap_bgr(
    frame_bgr: np.ndarray,
    *,
    l_max: int = 240,
    brightness: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    warmth: float = 0.0,
    shadows: float = 0.0,
    highlights: float = 0.0,
) -> np.ndarray:
    """Single BGR↔LAB pass: ENV-driven color grading and luminance cap combined.

    Saves one BGR↔LAB round trip per frame versus separate grading and cap steps.
    ``l_max`` is the Tier S4 L ceiling; all other parameters shift the respective
    LAB channel. See :func:`display_luminance_cap_bgr` for cap-only mode.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr
    lab = cv.cvtColor(frame_bgr, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    lf = l.astype(np.float32)
    if abs(shadows) > 1e-6:
        mask = np.clip((128.0 - lf) / 128.0, 0.0, 1.0)
        lf = lf + shadows * 40.0 * mask
    if abs(highlights) > 1e-6:
        hm = np.clip((lf - 200.0) / 55.0, 0.0, 1.0)
        lf = lf - highlights * 35.0 * hm
    lf = lf + brightness * 255.0
    lf = (lf - 128.0) * float(contrast) + 128.0
    lf = np.minimum(lf, float(l_max))
    lf = np.clip(lf, 0.0, 255.0)
    af = (a.astype(np.float32) - 128.0) * float(saturation) + 128.0 + warmth * 40.0
    bf = (b.astype(np.float32) - 128.0) * float(saturation) + 128.0
    return cv.cvtColor(
        cv.merge([
            lf.astype(np.uint8),
            np.clip(af, 0, 255).astype(np.uint8),
            np.clip(bf, 0, 255).astype(np.uint8),
        ]),
        cv.COLOR_LAB2BGR,
    )


class DisplayTemporalGlareBlend:
    """Tier S3 temporal IIR blend to soften harsh highlights; resets on jerk events."""

    def __init__(self, prev_weight: float = 0.42):
        """``prev_weight`` is the previous-frame weight in the exponential blend."""
        self.prev_weight = float(prev_weight)
        self._prev: Optional[np.ndarray] = None

    def process(self, frame_bgr: np.ndarray, apply_blend: bool, reset: bool) -> np.ndarray:
        """Blend with the previous display frame when ``apply_blend``; clear history if ``reset``."""
        if frame_bgr is None:
            return frame_bgr
        if reset:
            self._prev = None
        if not apply_blend:
            self._prev = frame_bgr.copy()
            return frame_bgr
        if self._prev is None or self._prev.shape != frame_bgr.shape:
            self._prev = frame_bgr.copy()
            return frame_bgr
        a0 = 1.0 - self.prev_weight
        out = cv.addWeighted(frame_bgr, a0, self._prev, self.prev_weight, 0)
        self._prev = out.copy()
        return out
