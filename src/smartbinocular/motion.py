"""
Motion estimation and display stabilization used from ``main``.

:class:`JerkGate` flags sharp motion on NIR for HUD and gating.
:class:`DisplayShakeReducerLite` smooths the final BGR before ``imshow``.
:class:`SparseOpticalFlowMotion` feeds ML feature vectors (optional; lean mode skips it).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import cv2 as cv

from smartbinocular.nir_pipeline import _percentile_u8_from_hist


@dataclass
class OpsecLocalOnly:
    """Local-only opsec stub: when enabled, outbound networking should be denied."""

    enabled: bool = False

    def allow_outbound_network(self) -> bool:
        """Return False if local-only mode is active."""
        return not self.enabled

    def toggle(self) -> bool:
        """Flip ``enabled`` and return the new value."""
        self.enabled = not self.enabled
        return self.enabled


class OneEuroFilter1D:
    """1D One Euro filter for smoothing shift estimates (no IMU required)."""

    __slots__ = ("min_cutoff", "beta", "d_cutoff", "x_prev", "dx_prev", "t_prev", "_initialized")

    def __init__(self, min_cutoff: float = 1.15, beta: float = 0.018, d_cutoff: float = 1.0):
        """Create filter with One Euro cutoff parameters."""
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = 0.0
        self.dx_prev = 0.0
        self.t_prev = 0.0
        self._initialized = False

    def reset(self) -> None:
        """Clear state so the next sample re-initializes the filter."""
        self._initialized = False

    @staticmethod
    def _smoothing_factor(dt: float, cutoff: float) -> float:
        """Interpolation alpha from timestep and cutoff frequency."""
        if dt <= 0:
            dt = 1e-6
        r = 2.0 * math.pi * cutoff * dt
        return r / (r + 1.0)

    def __call__(self, x: float, t: float) -> float:
        """Filter measurement ``x`` at monotonic time ``t``; returns smoothed value."""
        x = float(x)
        t = float(t)
        if not self._initialized:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            self._initialized = True
            return x
        dt = t - self.t_prev
        if dt <= 0:
            dt = 1e-6
        dx = (x - self.x_prev) / dt
        a_d = self._smoothing_factor(dt, self.d_cutoff)
        self.dx_prev += a_d * (dx - self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(self.dx_prev)
        a = self._smoothing_factor(dt, cutoff)
        x_hat = self.x_prev + a * (x - self.x_prev)
        self.x_prev = x_hat
        self.t_prev = t
        return x_hat


class JerkGate:
    """Image-based jerk detector on consecutive NIR frames (IMU placeholder).

    Uses a downscaled green-channel proxy, a high percentile of frame difference,
    and consecutive-frame confirmation to limit false positives.
    """

    def __init__(
        self,
        diff_threshold: float = 8.5,
        hold_frames: int = 12,
        consecutive_frames: int = 2,
        percentile: float = 94.0,
        max_side: int = 128,
        near_active_ratio: float = 0.62,
        lite_mode: bool = True,
        perf_benchmark_samples: int = 24,
    ):
        """Configure thresholds, hold duration, and optional lite-mode percentile path."""
        self.diff_threshold = diff_threshold
        self.hold_frames = hold_frames
        self.consecutive_frames = max(1, int(consecutive_frames))
        self.percentile = float(percentile)
        self.max_side = int(max_side)
        self.near_active_ratio = float(near_active_ratio)
        self.lite_mode = bool(lite_mode)
        self.perf_benchmark_samples = int(max(0, perf_benchmark_samples))
        self._prev_gray = None
        self._hold = 0
        self._strike = 0
        self.active = False
        self.near_active = False
        self.last_score = 0.0
        self._perf_old_ms_ema = 0.0
        self._perf_new_ms_ema = 0.0
        self._perf_n = 0

    def reset(self) -> None:
        """Clear differencing history after mode switches (invoked from ``main``)."""
        self._prev_gray = None
        self._hold = 0
        self._strike = 0
        self.active = False
        self.near_active = False
        self.last_score = 0.0

    @staticmethod
    def _ema_update(cur: float, val: float, alpha: float = 0.12) -> float:
        """Exponential moving average helper for micro-benchmark EMAs."""
        if cur <= 0.0:
            return float(val)
        return float((1.0 - alpha) * cur + alpha * float(val))

    def get_saved_ms_estimate(self) -> float:
        """Estimated ms/frame saved when lite percentile path is used (perf logging)."""
        if self._perf_n <= 0:
            return 0.0
        return max(0.0, self._perf_old_ms_ema - self._perf_new_ms_ema)

    def _to_small_gray(self, nir_bgr: np.ndarray) -> np.ndarray:
        """Downscale green channel to at most ``max_side`` for differencing."""
        gray = nir_bgr[:, :, 1]  # green channel as NIR luminance proxy
        h, w = gray.shape[:2]
        m = max(h, w)
        if m <= self.max_side:
            return gray
        scale = self.max_side / float(m)
        return cv.resize(
            gray,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv.INTER_AREA,
        )

    def _percentile_fast_u8(self, arr_u8: np.ndarray, pct: float) -> float:
        """Histogram-based percentile for uint8 ``absdiff`` (lite mode)."""
        hist = np.bincount(arr_u8.ravel(), minlength=256).astype(np.int32)
        return _percentile_u8_from_hist(hist, pct)

    def update(
        self,
        nir_bgr: Optional[np.ndarray],
        *,
        precomputed_gray: Optional[np.ndarray] = None,
    ) -> None:
        """Update jerk state from ``nir_bgr`` or a precomputed small gray from ``FrameCache``.

        Passing ``precomputed_gray`` skips internal resize and saves ~0.4 ms/frame.
        """
        if nir_bgr is None and precomputed_gray is None:
            if self._hold > 0:
                self._hold -= 1
            self.active = self._hold > 0
            self.near_active = False
            self.last_score = 0.0
            return
        gray = precomputed_gray if precomputed_gray is not None else self._to_small_gray(nir_bgr)
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            df_u8 = cv.absdiff(gray, self._prev_gray)
            t0 = time.perf_counter()
            m = float(np.mean(df_u8))
            if self.lite_mode:
                p = float(self._percentile_fast_u8(df_u8, self.percentile))
            else:
                p = float(np.percentile(df_u8.astype(np.float32), self.percentile))
            t1 = time.perf_counter()
            new_ms = (t1 - t0) * 1000.0
            self._perf_new_ms_ema = self._ema_update(self._perf_new_ms_ema, new_ms)
            if self.lite_mode and self._perf_n < self.perf_benchmark_samples:
                t2 = time.perf_counter()
                _ = float(np.percentile(df_u8.astype(np.float32), self.percentile))
                t3 = time.perf_counter()
                old_ms = (t3 - t2) * 1000.0
                self._perf_old_ms_ema = self._ema_update(self._perf_old_ms_ema, old_ms)
                self._perf_n += 1
            score = 0.48 * m + 0.52 * p
            self.last_score = score
            self.near_active = score > (self.diff_threshold * self.near_active_ratio)
            if score > self.diff_threshold:
                self._strike += 1
                if self._strike >= self.consecutive_frames:
                    self._hold = self.hold_frames
                    self._strike = 0
            else:
                self._strike = 0
        else:
            self._strike = 0
            self.near_active = False
            self.last_score = 0.0
        self._prev_gray = gray.copy()
        if self._hold > 0:
            self._hold -= 1
            self.active = True
        else:
            self.active = False


class DisplayShakeReducerLite:
    """Stabilize displayed frames via temporal blend or sub-pixel phase-correlation shift."""

    def __init__(
        self,
        mode: str = "blend",
        blend_current_weight: float = 0.50,
        shift_ema: float = 0.42,
        max_shift_px: float = 14.0,
        small_max_side: int = 160,
        *,
        use_one_euro_shift: bool = True,
        adaptive_blend: bool = True,
        one_euro_min_cutoff: float = 1.15,
        one_euro_beta: float = 0.018,
        one_euro_d_cutoff: float = 1.0,
    ):
        """``mode`` is ``off``, ``blend``, or ``shift``; see :meth:`process` for behavior."""
        self.mode = mode
        self.blend_current_weight = blend_current_weight
        self.shift_ema = shift_ema
        self.max_shift_px = max_shift_px
        self.small_max_side = small_max_side
        self.use_one_euro_shift = use_one_euro_shift
        self.adaptive_blend = adaptive_blend
        self._euro_x = OneEuroFilter1D(one_euro_min_cutoff, one_euro_beta, one_euro_d_cutoff)
        self._euro_y = OneEuroFilter1D(one_euro_min_cutoff, one_euro_beta, one_euro_d_cutoff)
        self._prev_bgr = None
        self._prev_gray_small = None
        self._sx = 0.0
        self._sy = 0.0

    def set_mode(self, mode: str) -> None:
        """Clamp unknown modes to ``off`` and reset internal state."""
        if mode not in ("off", "blend", "shift"):
            mode = "off"
        self.mode = mode
        self.reset()

    def reset(self) -> None:
        """Drop previous-frame buffers and reset One Euro filters."""
        self._prev_bgr = None
        self._prev_gray_small = None
        self._sx = self._sy = 0.0
        self._euro_x.reset()
        self._euro_y.reset()

    def process(
        self,
        frame_bgr: np.ndarray,
        jerk_active: bool = False,
        soft_motion_active: bool = False,
    ) -> np.ndarray:
        """Return stabilized ``frame_bgr``; pass-through on jerk or soft-motion guard."""
        if self.mode == "off" or frame_bgr is None:
            return frame_bgr
        if jerk_active or soft_motion_active:
            self._prev_bgr = frame_bgr.copy()
            g = frame_bgr[:, :, 1]  # green channel as NIR luminance proxy
            h, w = g.shape[:2]
            scale = min(self.small_max_side / max(w, h), 1.0)
            self._prev_gray_small = cv.resize(g, (int(w * scale), int(h * scale)), interpolation=cv.INTER_AREA).astype(np.float32)
            self._sx = self._sy = 0.0
            self._euro_x.reset()
            self._euro_y.reset()
            return frame_bgr

        h, w = frame_bgr.shape[:2]

        if self.mode == "blend":
            if self._prev_bgr is None or self._prev_bgr.shape != frame_bgr.shape:
                self._prev_bgr = frame_bgr.copy()
                return frame_bgr
            a0 = self.blend_current_weight
            if self.adaptive_blend:
                g0 = self._prev_bgr[:, :, 1]  # green channel as NIR luminance proxy
                g1 = frame_bgr[:, :, 1]
                md = float(np.mean(cv.absdiff(g1, g0)))
                a = float(np.clip(a0 + 0.14 * math.tanh(md / 4.0) - 0.07, 0.36, 0.9))
            else:
                a = a0
            out = cv.addWeighted(frame_bgr, a, self._prev_bgr, 1.0 - a, 0)
            self._prev_bgr = out.copy()
            return out

        g = frame_bgr[:, :, 1]  # green channel as NIR luminance proxy
        scale = min(self.small_max_side / max(w, h), 1.0)
        sw, sh = int(w * scale), int(h * scale)
        small = cv.resize(g, (sw, sh), interpolation=cv.INTER_AREA).astype(np.float32)

        if self._prev_gray_small is None or self._prev_gray_small.shape != small.shape:
            self._prev_gray_small = small.copy()
            self._prev_bgr = frame_bgr.copy()
            return frame_bgr

        try:
            (dx, dy), _resp = cv.phaseCorrelate(self._prev_gray_small, small)
        except cv.error:
            self._prev_gray_small = small.copy()
            return frame_bgr

        fx = dx * (w / float(sw))
        fy = dy * (h / float(sh))
        if self.use_one_euro_shift:
            t_mono = time.monotonic()
            self._sx = self._euro_x(fx, t_mono)
            self._sy = self._euro_y(fy, t_mono)
        else:
            self._sx = self.shift_ema * fx + (1.0 - self.shift_ema) * self._sx
            self._sy = self.shift_ema * fy + (1.0 - self.shift_ema) * self._sy
        tx = float(np.clip(self._sx, -self.max_shift_px, self.max_shift_px))
        ty = float(np.clip(self._sy, -self.max_shift_px, self.max_shift_px))

        M = np.float32([[1.0, 0.0, -tx], [0.0, 1.0, -ty]])
        out = cv.warpAffine(frame_bgr, M, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)

        self._prev_gray_small = 0.88 * self._prev_gray_small + 0.12 * small
        self._prev_bgr = out.copy()
        return out

    def cycle_mode(self) -> str:
        """Advance ``off`` → ``blend`` → ``shift`` → ``off``; return the new mode name."""
        order = ("off", "blend", "shift")
        try:
            i = order.index(self.mode)
        except ValueError:
            i = 0
        self.set_mode(order[(i + 1) % len(order)])
        return self.mode


# ─── Sparse optical flow (ML motion features) ───────────────────────────────

class SparseOpticalFlowMotion:
    """Lucas–Kanade sparse flow on downscaled NIR (parallel to :class:`JerkGate`).

    Exposes ``motion_magnitude``, ``motion_direction``, and ``motion_jerk`` for
    :class:`~smartbinocular.feature_extractor.FeatureExtractor` and JSONL logging.

    Cost: about 2 ms on Raspberry Pi 4 at 160×120 with ~20 corners.
    """

    def __init__(
        self,
        max_points: int = 20,
        refresh_interval: int = 15,    # refresh corner set every N frames
        small_size: tuple = (160, 120),
        lk_win: int = 11,
        lk_levels: int = 2,
    ):
        """Configure corner count, LK window, and refresh cadence."""
        self.max_points = int(max_points)
        self.refresh_interval = int(max(1, refresh_interval))
        self.small_size = tuple(small_size)
        self._lk_params = dict(
            winSize=(lk_win, lk_win),
            maxLevel=lk_levels,
            criteria=(cv.TERM_CRITERIA_COUNT | cv.TERM_CRITERIA_EPS, 10, 0.03),
        )
        self._feat_params = dict(
            maxCorners=max_points, qualityLevel=0.1, minDistance=8, blockSize=7
        )
        self._prev_gray: Optional[np.ndarray] = None
        self._pts: Optional[np.ndarray] = None
        self._frame_idx: int = 0
        # Public state — read by ML feature extractor
        self.motion_magnitude: float = 0.0
        self.motion_direction: float = 0.0   # degrees, circular mean of flow vectors
        self.motion_jerk: float = 0.0         # frame-to-frame Δ magnitude

    def reset(self) -> None:
        """Clear corners and public motion scalars (e.g. after stream reconfigure)."""
        self._prev_gray = None
        self._pts = None
        self._frame_idx = 0
        self.motion_magnitude = 0.0
        self.motion_direction = 0.0
        self.motion_jerk = 0.0

    def _to_gray(self, nir_bgr: np.ndarray) -> np.ndarray:
        """Resize to ``small_size`` and take green channel as luminance proxy."""
        return cv.resize(nir_bgr, self.small_size, interpolation=cv.INTER_AREA)[:, :, 1]

    def update(
        self,
        nir_bgr: Optional[np.ndarray],
        *,
        nir_gray_small: Optional[np.ndarray] = None,
    ) -> None:
        """Advance flow state; pass ``nir_gray_small`` to skip internal resize."""
        if nir_bgr is None and nir_gray_small is None:
            return
        curr = nir_gray_small if nir_gray_small is not None else self._to_gray(nir_bgr)
        self._frame_idx += 1

        # First frame or stale corners — detect
        if self._prev_gray is None or self._pts is None or len(self._pts) < 4:
            self._pts = cv.goodFeaturesToTrack(curr, **self._feat_params)
            self._prev_gray = curr.copy()
            return

        # Periodic corner refresh in current frame (no flow this cycle)
        if self._frame_idx % self.refresh_interval == 0:
            new_pts = cv.goodFeaturesToTrack(curr, **self._feat_params)
            if new_pts is not None:
                self._pts = new_pts
            self._prev_gray = curr.copy()
            return

        pts_new, status, _ = cv.calcOpticalFlowPyrLK(
            self._prev_gray, curr, self._pts, None, **self._lk_params
        )
        if status is None or pts_new is None:
            self._prev_gray = curr.copy()
            return

        good_old = self._pts[status.ravel() == 1]
        good_new = pts_new[status.ravel() == 1]

        if len(good_old) < 3:
            self._pts = None
            self._prev_gray = curr.copy()
            return

        flow_vecs = good_new - good_old                           # (N, 1, 2)
        magnitudes = np.sqrt(np.sum(flow_vecs ** 2, axis=2)).ravel()
        angles = np.arctan2(flow_vecs[:, 0, 1], flow_vecs[:, 0, 0])

        prev_mag = self.motion_magnitude
        self.motion_magnitude = float(np.median(magnitudes))
        self.motion_jerk = abs(self.motion_magnitude - prev_mag)

        # Circular mean for direction (handles 0°/360° wrap correctly)
        self.motion_direction = float(np.degrees(
            np.arctan2(float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles))))
        ))

        self._pts = good_new.reshape(-1, 1, 2)
        self._prev_gray = curr.copy()
