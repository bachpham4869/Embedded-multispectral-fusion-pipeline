"""
Low-level thermal processing for ``main``: 3DNR, background model, heat maps,
and optional anomaly detectors.

:class:`ThermalProcessor` is the primary entry from the frame loop; detector
classes support HUD overlays in thermal and fusion modes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import cv2 as cv


class _NoopProfiler:
    """Drop-in no-op for StageProfiler — used when thermal sub-timing is disabled."""
    def __call__(self, _: str) -> "_NoopProfiler": return self
    def __enter__(self) -> "_NoopProfiler": return self
    def __exit__(self, *_: object) -> None: pass

_NOOP_PROFILER = _NoopProfiler()


def thermal_highlight_need_compress(
    frame_u8: np.ndarray,
    high_pct: float = 99.0,
    saturate_at: float = 240.0,
) -> bool:
    """True when the high-tail percentile of the frame meets or exceeds the saturation threshold."""
    f = frame_u8.astype(np.float32)
    p_hi = float(np.percentile(f, high_pct))
    return p_hi >= saturate_at


def thermal_glare_hud_should_apply(
    frame_u8: np.ndarray,
    high_pct: float = 99.5,
    saturate_at: float = 252.0,
    min_max_u8: float = 253.0,
) -> bool:
    """True for thermal HUD GLARE when the high tail is hot and near-saturation pixels exist."""
    f = frame_u8.astype(np.float32)
    p_hi = float(np.percentile(f, high_pct))
    mx = float(np.max(f))
    return p_hi >= saturate_at and mx >= min_max_u8


def thermal_anti_glare(
    frame_u8: np.ndarray,
    high_pct: float = 99.0,
    saturate_at: float = 240.0,
    gamma: float = 0.74,
) -> np.ndarray:
    """Per-pixel gamma compression on thermal uint8 (geometry unchanged)."""
    if not thermal_highlight_need_compress(frame_u8, high_pct, saturate_at):
        return frame_u8
    f = frame_u8.astype(np.float32)
    out = np.clip(np.power(f / 255.0, gamma) * 255.0, 0, 255)
    return out.astype(np.uint8)


def thermal_agc(frame: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
    """Automatic gain: percentile stretch to 0–255."""
    f = frame.astype(np.float32)
    lo = np.percentile(f, low_pct)
    hi = np.percentile(f, high_pct)
    if hi - lo < 1:
        return frame
    return np.clip((f - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def thermal_edge_enhance(frame: np.ndarray, strength: float = 0.2) -> np.ndarray:
    """Light Laplacian edge enhancement on thermal grayscale."""
    if abs(strength) < 1e-6:
        return frame
    lap = cv.Laplacian(frame.astype(np.float32), cv.CV_32F, ksize=3)
    return np.clip(frame.astype(np.float32) + strength * np.abs(lap), 0, 255).astype(np.uint8)


class ThermalTemporalFilter:
    """Temporal EMA denoise (3DNR-style) on small thermal frames."""

    def __init__(self, alpha: float = 0.65):
        """``alpha`` weights the current frame in the EMA blend."""
        self.alpha = alpha
        self.ema = None

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Blend ``frame`` with the running EMA; return uint8 clipped output."""
        f = frame.astype(np.float32)
        if self.ema is None:
            self.ema = f.copy()
        else:
            self.ema = self.alpha * f + (1 - self.alpha) * self.ema
        return np.clip(self.ema, 0, 255).astype(np.uint8)


class ThermalBackgroundModel:
    """Legacy cold-frame background: warmup average then slow adaptation.

    Produces a continuous heat map (float32) for gradient visualization.
    """

    def __init__(self, warmup_frames: int = 40, adaptive_rate: float = 0.005):
        """``warmup_frames`` buffers median background; ``adaptive_rate`` controls drift."""
        self.warmup_frames = warmup_frames
        self.adaptive_rate = adaptive_rate
        self.cold_frame = None
        self._buf: List[np.ndarray] = []
        self.is_ready = False
        self.frame_count = 0

    def update(self, frame: np.ndarray) -> None:
        f = frame.astype(np.float32)
        self.frame_count += 1
        if not self.is_ready:
            self._buf.append(f)
            if len(self._buf) >= self.warmup_frames:
                self.cold_frame = np.mean(self._buf, axis=0)
                self._buf = []
                self.is_ready = True
        else:
            self.cold_frame = (1 - self.adaptive_rate) * self.cold_frame + self.adaptive_rate * f

    def get_heat_map(self, frame: np.ndarray, floor: float = 3.0) -> Optional[np.ndarray]:
        """Continuous heat map 0–255; ``floor`` suppresses small background residuals."""
        if not self.is_ready:
            return None
        diff = frame.astype(np.float32) - self.cold_frame
        diff = np.maximum(diff - floor, 0)
        mx = diff.max()
        if mx < 1:
            return np.zeros_like(frame, dtype=np.uint8)
        return np.clip(diff * (255.0 / mx), 0, 255).astype(np.uint8)

    def get_foreground_mask(
        self,
        frame: np.ndarray,
        threshold: float = 18.0,
        min_area: int = 12,
        max_fg_ratio: float = 0.5,
        morph_kernel_size: int = 5,
        open_iterations: int = 2,
        close_iterations: int = 1,
    ) -> Optional[np.ndarray]:
        """Morph-cleaned binary mask; returns zeros if foreground covers more than ``max_fg_ratio``."""
        if not self.is_ready:
            return None
        diff = frame.astype(np.float32) - self.cold_frame
        raw_mask = (diff > threshold).astype(np.uint8) * 255
        k = int(max(3, morph_kernel_size))
        if k % 2 == 0:
            k += 1
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))
        cleaned = cv.morphologyEx(raw_mask, cv.MORPH_OPEN, kernel, iterations=max(0, int(open_iterations)))
        cleaned = cv.morphologyEx(cleaned, cv.MORPH_CLOSE, kernel, iterations=max(0, int(close_iterations)))
        fg_ratio = np.count_nonzero(cleaned) / cleaned.size
        if fg_ratio > max_fg_ratio:
            return np.zeros_like(cleaned)
        contours, _ = cv.findContours(cleaned, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        result = np.zeros_like(cleaned)
        for c in contours:
            if cv.contourArea(c) >= min_area:
                cv.drawContours(result, [c], -1, 255, -1)
        return result


# ─── Kalman background model (per-pixel 1D filter) ───────────────────────────

class KalmanThermalBackground:
    """Per-pixel 1D Kalman filter for thermal background estimation.

    Replaces the EMA-based ThermalBackgroundModel. No warmup period needed —
    initialises with high uncertainty and converges naturally within ~5 frames.

    Vectorised across all 80×62 = 4960 pixels simultaneously → ~0.5 ms/frame.

    Interface is a drop-in superset of ThermalBackgroundModel:
    same update() / get_heat_map() / get_foreground_mask() / is_ready / warmup_pct.
    """

    def __init__(
        self,
        warmup_frames: int = 1,        # kept for API compat; Kalman needs only 1
        adaptive_rate: float = 0.005,  # unused; kept for API compat
        process_noise: float = 0.5,    # Q: expected BG temperature drift per frame
        measurement_noise: float = 4.0,  # R: thermal sensor noise variance (~±2°C)
        init_uncertainty: float = 100.0, # P₀: large → fast initial convergence
    ):
        self.warmup_frames = int(max(1, warmup_frames))
        self._Q = float(process_noise)
        self._R = float(measurement_noise)
        self._init_P = float(init_uncertainty)
        self._x: Optional[np.ndarray] = None   # background estimate (float32)
        self._P: Optional[np.ndarray] = None   # per-pixel covariance   (float32)
        self.is_ready: bool = False
        self.frame_count: int = 0

    def update(self, frame: np.ndarray) -> None:
        z = frame.astype(np.float32)
        self.frame_count += 1
        if self._x is None:
            self._x = z.copy()
            self._P = np.full_like(z, self._init_P, dtype=np.float32)
            self.is_ready = True
            return
        K = self._P / (self._P + self._R)          # Kalman gain (element-wise)
        self._x = self._x + K * (z - self._x)      # state update
        self._P = (1.0 - K) * self._P + self._Q    # covariance update

    @property
    def cold_frame(self) -> Optional[np.ndarray]:
        """Alias: background estimate (used by legacy code)."""
        return self._x

    @property
    def warmup_pct(self) -> int:
        return 100 if self.is_ready else 0

    def get_heat_map(self, frame: np.ndarray, floor: float = 3.0) -> Optional[np.ndarray]:
        if self._x is None:
            return None
        diff = np.maximum(frame.astype(np.float32) - self._x - floor, 0.0)
        mx = float(diff.max())
        if mx < 1.0:
            return np.zeros_like(frame, dtype=np.uint8)
        return np.clip(diff * (255.0 / mx), 0, 255).astype(np.uint8)

    def get_foreground_mask(
        self,
        frame: np.ndarray,
        threshold: float = 18.0,
        min_area: int = 12,
        max_fg_ratio: float = 0.5,
        morph_kernel_size: int = 5,
        open_iterations: int = 2,
        close_iterations: int = 1,
    ) -> Optional[np.ndarray]:
        if self._x is None:
            return None
        diff = frame.astype(np.float32) - self._x
        raw_mask = (diff > threshold).astype(np.uint8) * 255
        k = int(max(3, morph_kernel_size))
        if k % 2 == 0:
            k += 1
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k, k))
        cleaned = cv.morphologyEx(raw_mask, cv.MORPH_OPEN, kernel, iterations=max(0, int(open_iterations)))
        cleaned = cv.morphologyEx(cleaned, cv.MORPH_CLOSE, kernel, iterations=max(0, int(close_iterations)))
        if np.count_nonzero(cleaned) / cleaned.size > max_fg_ratio:
            return np.zeros_like(cleaned)
        contours, _ = cv.findContours(cleaned, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        result = np.zeros_like(cleaned)
        for c in contours:
            if cv.contourArea(c) >= min_area:
                cv.drawContours(result, [c], -1, 255, -1)
        return result


class ThermalProcessor:
    """Full thermal stack: 3DNR → background → heat map / fg mask → display branch.

    Matches ``legacy/py/final_fusion.py`` when ``use_kalman_background`` is False: EMA cold-frame
    :class:`ThermalBackgroundModel`, and heat/foreground from **denoised** only (not anti-glare
    or detail-mix paths).
    """

    def __init__(
        self,
        warmup: int = 40,
        anti_glare: bool = True,
        *,
        use_kalman_background: bool = False,
        display_anti_glare: bool = True,
        thermal_high_pct: float = 99.0,
        thermal_saturate_at: float = 240.0,
        thermal_gamma: float = 0.74,
        thermal_floor: float = 3.0,
        thermal_fg_threshold: float = 18.0,
        thermal_fg_max_ratio: float = 0.5,
        thermal_edge_strength: float = 0.2,
        thermal_agc_low_pct: float = 2.0,
        thermal_agc_high_pct: float = 98.0,
        detail_preserve_detect: bool = False,
        detect_raw_mix: float = 0.55,
        detail_grad_threshold: float = 9.0,
        detail_threshold_relax: float = 3.0,
        detect_use_anti_glare: bool = False,
        thermal_3dnr_alpha: float = 0.65,
        bilateral_display_enable: bool = True,
        bilateral_d: int = 5,
        bilateral_sigma_color: float = 15.0,
        bilateral_sigma_space: float = 5.0,
    ):
        self.temporal = ThermalTemporalFilter(alpha=float(thermal_3dnr_alpha))
        self._use_kalman_bg = bool(use_kalman_background)
        if self._use_kalman_bg:
            self.bg = KalmanThermalBackground(warmup_frames=warmup, adaptive_rate=0.005)
        else:
            self.bg = ThermalBackgroundModel(
                warmup_frames=int(max(1, warmup)),
                adaptive_rate=0.005,
            )
        self.anti_glare_enabled = anti_glare
        # When False, skip per-pixel gamma on the AGC display path only (3DNR + edge still run).
        self.display_anti_glare = bool(display_anti_glare)
        self.thermal_high_pct = float(thermal_high_pct)
        self.thermal_saturate_at = float(thermal_saturate_at)
        self.thermal_gamma = float(thermal_gamma)
        self.thermal_floor = float(thermal_floor)
        self.thermal_fg_threshold = float(thermal_fg_threshold)
        self.thermal_fg_max_ratio = float(np.clip(thermal_fg_max_ratio, 0.05, 0.95))
        self.thermal_edge_strength = float(thermal_edge_strength)
        self.thermal_agc_low_pct = float(thermal_agc_low_pct)
        self.thermal_agc_high_pct = float(thermal_agc_high_pct)
        self.detail_preserve_detect = bool(detail_preserve_detect)
        self.detect_raw_mix = float(np.clip(detect_raw_mix, 0.0, 1.0))
        self.detail_grad_threshold = float(max(0.0, detail_grad_threshold))
        self.detail_threshold_relax = float(max(0.0, detail_threshold_relax))
        self.detect_use_anti_glare = bool(detect_use_anti_glare)
        self.bilateral_display_enable = bool(bilateral_display_enable)
        self.bilateral_d = int(max(1, bilateral_d))
        if self.bilateral_d % 2 == 0:
            self.bilateral_d += 1
        self.bilateral_sigma_color = float(bilateral_sigma_color)
        self.bilateral_sigma_space = float(bilateral_sigma_space)
        self.last_detail_mode_active = False
        self.last_detail_grad_mean = 0.0
        # Optional sub-stage profiler (StageProfiler or _NoopProfiler). Set from main after construction.
        self.stage_profiler: Any = _NOOP_PROFILER

    def apply_env_runtime(
        self,
        merged_opt_cfg: Dict[str, Any],
        thermal_extra: Optional[Dict[str, Any]] = None,
        thermal_3dnr_alpha: Optional[float] = None,
    ) -> None:
        """Apply merged ENV ``opt_cfg`` and preset extras without resetting the background model."""
        m = merged_opt_cfg
        self.thermal_high_pct = float(m["thermal_high_pct"])
        self.thermal_saturate_at = float(m["thermal_saturate_at"])
        self.thermal_gamma = float(m["thermal_gamma"])
        self.thermal_floor = float(m["thermal_floor"])
        self.thermal_fg_threshold = float(m["thermal_fg_threshold"])
        if "thermal_fg_max_ratio" in m:
            self.thermal_fg_max_ratio = float(np.clip(float(m["thermal_fg_max_ratio"]), 0.05, 0.95))
        self.thermal_edge_strength = float(m["thermal_edge_strength"])
        self.thermal_agc_low_pct = float(m["thermal_agc_low_pct"])
        self.thermal_agc_high_pct = float(m["thermal_agc_high_pct"])
        te = thermal_extra or {}
        if "thermal_detect_raw_mix" in te:
            self.detect_raw_mix = float(np.clip(float(te["thermal_detect_raw_mix"]), 0.0, 1.0))
        if "thermal_detail_grad_threshold" in te:
            self.detail_grad_threshold = float(max(0.0, float(te["thermal_detail_grad_threshold"])))
        if "thermal_detail_threshold_relax" in te:
            self.detail_threshold_relax = float(max(0.0, float(te["thermal_detail_threshold_relax"])))
        # Default False each preset apply; glare_heavy may set True via thermal_extra.
        self.detect_use_anti_glare = bool(te.get("thermal_detect_use_anti_glare", False))
        if "thermal_display_glare_suppression" in te:
            self.display_anti_glare = bool(te["thermal_display_glare_suppression"])
        if "thermal_fg_max_ratio" in te:
            self.thermal_fg_max_ratio = float(np.clip(float(te["thermal_fg_max_ratio"]), 0.05, 0.95))
        if thermal_3dnr_alpha is not None:
            self.temporal.alpha = float(np.clip(float(thermal_3dnr_alpha), 0.35, 0.85))

    def update_runtime_params(self, opt_cfg: Dict[str, Any]) -> None:
        """Update bilateral filter params from opt_cfg_runtime on preset-stable change.

        Called at most once per stable-preset transition (not per frame). Safe to call
        when keys are absent — uses .get() pattern so missing keys leave values unchanged.
        """
        if "thermal_bilateral_d" in opt_cfg:
            d = int(opt_cfg["thermal_bilateral_d"])
            if d % 2 == 0:
                d += 1
            self.bilateral_d = max(1, d)
        if "thermal_bilateral_sigma_color" in opt_cfg:
            self.bilateral_sigma_color = float(opt_cfg["thermal_bilateral_sigma_color"])
        if "thermal_bilateral_sigma_space" in opt_cfg:
            self.bilateral_sigma_space = float(opt_cfg["thermal_bilateral_sigma_space"])

    @property
    def is_ready(self) -> bool:
        return self.bg.is_ready

    @property
    def warmup_pct(self) -> int:
        wf = int(getattr(self.bg, "warmup_frames", 40))
        if wf < 1:
            wf = 1
        return min(100, int(self.bg.frame_count / wf * 100))

    def process(
        self, raw_frame: np.ndarray, *, compute_enhanced: bool = True
    ) -> tuple:
        """Return ``(denoised, enhanced, heat_map, fg_mask)``.

        - ``denoised``: 3DNR output only (smooth, low background noise).
        - ``enhanced``: bilateral→AGC→edge display path; ``None`` when ``compute_enhanced=False``.
        - ``heat_map`` / ``fg_mask``: from **denoised** vs. cold background (``final_fusion`` parity).
          Optional detail branch (when ``detail_preserve_detect``) only updates HUD telemetry.

        Pass ``compute_enhanced=False`` in fusion mode — the display path is unused there.
        """
        _sp = self.stage_profiler
        with _sp("thermal_3dnr"):
            denoised = self.temporal.process(raw_frame)
        with _sp("thermal_kalman"):
            self.bg.update(denoised)

        if not compute_enhanced:
            enhanced = None
        else:
            with _sp("thermal_agc"):
                if self.bilateral_display_enable:
                    display_src = cv.bilateralFilter(
                        denoised,
                        self.bilateral_d,
                        self.bilateral_sigma_color,
                        self.bilateral_sigma_space,
                    )
                else:
                    display_src = denoised
                agc = thermal_agc(display_src, low_pct=self.thermal_agc_low_pct, high_pct=self.thermal_agc_high_pct)
                agc = (
                    thermal_anti_glare(
                        agc,
                        high_pct=self.thermal_high_pct,
                        saturate_at=self.thermal_saturate_at,
                        gamma=self.thermal_gamma,
                    )
                    if self.anti_glare_enabled and self.display_anti_glare
                    else agc
                )
            with _sp("thermal_edge"):
                enhanced = thermal_edge_enhance(agc, strength=self.thermal_edge_strength)
        # Heat and fg match legacy: same denoised frame as background update (not anti-glare / detail mix).
        with _sp("thermal_heatmap"):
            heat_map = self.bg.get_heat_map(denoised, floor=float(self.thermal_floor))
        with _sp("thermal_fgmask"):
            fg_mask = self.bg.get_foreground_mask(
            denoised,
            threshold=self.thermal_fg_threshold,
            min_area=12,
            max_fg_ratio=self.thermal_fg_max_ratio,
            morph_kernel_size=5,
            open_iterations=2,
            close_iterations=1,
        )
        if self.detail_preserve_detect:
            detect_mix = cv.addWeighted(
                raw_frame.astype(np.float32),
                self.detect_raw_mix,
                denoised.astype(np.float32),
                1.0 - self.detect_raw_mix,
                0,
            )
            detect_src = np.clip(detect_mix, 0, 255).astype(np.uint8)
            gx = cv.Sobel(detect_src, cv.CV_32F, 1, 0, ksize=3)
            gy = cv.Sobel(detect_src, cv.CV_32F, 0, 1, ksize=3)
            self.last_detail_grad_mean = float(np.mean(cv.magnitude(gx, gy)))
            self.last_detail_mode_active = self.last_detail_grad_mean >= self.detail_grad_threshold
        else:
            self.last_detail_grad_mean = 0.0
            self.last_detail_mode_active = False
        return denoised, enhanced, heat_map, fg_mask


class ThermalAnomalyDetectorLite:
    """[FEATURE] E1-lite thermal detection.

    Input: thermal_raw, thermal_denoised, heat_map, fg_mask, jerk_active
    Output: candidate_mask (uint8), blobs [{cx, cy, area, score}]
    """

    def __init__(
        self,
        *,
        local_kernel: int = 5,
        z_thresh: float = 1.25,
        heat_thresh: float = 46.0,
        min_area: int = 10,
        raw_mix: float = 0.45,
    ):
        k = int(max(3, local_kernel))
        if k % 2 == 0:
            k += 1
        self.local_kernel = k
        self.z_thresh = float(z_thresh)
        self.heat_thresh = float(heat_thresh)
        self.min_area = int(max(3, min_area))
        self.raw_mix = float(np.clip(raw_mix, 0.0, 1.0))

    def process(
        self,
        thermal_raw: np.ndarray,
        thermal_denoised: np.ndarray,
        heat_map: Optional[np.ndarray],
        fg_mask: Optional[np.ndarray],
        *,
        jerk_active: bool = False,
    ) -> tuple:
        if thermal_raw is None or thermal_denoised is None:
            return np.zeros((1, 1), dtype=np.uint8), []

        src_f = cv.addWeighted(
            thermal_raw.astype(np.float32),
            self.raw_mix,
            thermal_denoised.astype(np.float32),
            1.0 - self.raw_mix,
            0,
        )
        src_u8 = np.clip(src_f, 0, 255).astype(np.uint8)
        sf = src_u8.astype(np.float32)
        k = self.local_kernel
        local_mean = cv.blur(sf, (k, k))
        local_sq_mean = cv.blur(sf * sf, (k, k))
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean * local_mean, 0.0) + 1.0)
        z_map = (sf - local_mean) / (local_std + 1e-6)

        z_thr = self.z_thresh + (0.30 if jerk_active else 0.0)
        z_mask = z_map > z_thr
        if heat_map is not None:
            hm_thr = self.heat_thresh + (8.0 if jerk_active else 0.0)
            hm_mask = heat_map.astype(np.float32) > hm_thr
        else:
            hm_mask = np.zeros_like(z_mask, dtype=bool)

        if fg_mask is not None and fg_mask.shape == src_u8.shape:
            base_mask = fg_mask > 0
        else:
            base_mask = hm_mask | (z_map > (z_thr + 0.35))

        cand = (base_mask & (z_mask | hm_mask)).astype(np.uint8) * 255
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        cand = cv.morphologyEx(cand, cv.MORPH_OPEN, kernel, iterations=1)
        cand = cv.morphologyEx(cand, cv.MORPH_CLOSE, kernel, iterations=1)
        if jerk_active:
            cand = cv.erode(cand, kernel, iterations=1)

        contours, _ = cv.findContours(cand, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        blobs: List[Dict[str, float]] = []
        for c in contours:
            area = float(cv.contourArea(c))
            if area < self.min_area:
                continue
            m = cv.moments(c)
            if abs(m["m00"]) < 1e-6:
                continue
            cx = float(m["m10"] / m["m00"])
            cy = float(m["m01"] / m["m00"])
            obj_mask = np.zeros_like(cand)
            cv.drawContours(obj_mask, [c], -1, 255, -1)
            idx = obj_mask > 0
            z_mean = float(np.mean(z_map[idx])) if np.any(idx) else 0.0
            hm_mean = float(np.mean(heat_map[idx])) if (heat_map is not None and np.any(idx)) else 0.0
            z_norm = float(np.clip((z_mean - self.z_thresh) / 3.0, 0.0, 1.0))
            hm_norm = float(np.clip(hm_mean / 255.0, 0.0, 1.0))
            score = 0.65 * z_norm + 0.35 * hm_norm
            blobs.append({"cx": cx, "cy": cy, "area": area, "score": score})
        blobs.sort(key=lambda b: (b["score"], b["area"]), reverse=True)
        return cand, blobs


# ─── MAD-based thermal anomaly detector (E1 upgrade path) ────────────────────

class ThermalMADAnomalyDetector:
    """E1 upgrade: Median Absolute Deviation anomaly detection (Leys et al., 2013).

    More robust than local z-score to asymmetric thermal distributions and
    isolated hot pixels. Includes a temporal integrator to prevent single-frame
    noise triggers.

    Cost: ~0.3 ms on 80×62.
    Returns: (anomaly_score: float, anomaly_active: bool, blobs: list)
    """

    def __init__(
        self,
        *,
        mad_z_thresh: float = 3.5,
        min_fg_pixels: int = 5,
        temporal_window: int = 3,   # fire only after N consecutive anomaly frames
        min_area: int = 10,
        heat_weight: float = 0.35,  # weight of heat_map score vs MAD z-score
    ):
        self.mad_z_thresh = float(mad_z_thresh)
        self.min_fg_pixels = int(min_fg_pixels)
        self.temporal_window = int(max(1, temporal_window))
        self.min_area = int(max(3, min_area))
        self.heat_weight = float(np.clip(heat_weight, 0.0, 1.0))
        self._consecutive: int = 0
        self.anomaly_score: float = 0.0
        self.anomaly_active: bool = False

    def process(
        self,
        thermal_frame: np.ndarray,
        fg_mask: Optional[np.ndarray],
        heat_map: Optional[np.ndarray] = None,
        *,
        jerk_active: bool = False,
    ) -> tuple:
        if fg_mask is not None and np.any(fg_mask > 0):
            fg_pixels = thermal_frame.astype(np.float32)[fg_mask > 0]
        else:
            fg_pixels = thermal_frame.astype(np.float32).ravel()

        if len(fg_pixels) < self.min_fg_pixels:
            self.anomaly_score = 0.0
            self._consecutive = 0
            self.anomaly_active = False
            return 0.0, False, []

        med = float(np.median(fg_pixels))
        mad = float(np.median(np.abs(fg_pixels - med)))
        modified_z = 0.6745 * (fg_pixels - med) / (mad + 1e-6)
        thresh = self.mad_z_thresh + (0.5 if jerk_active else 0.0)
        self.anomaly_score = float(np.mean(modified_z > thresh))

        if self.anomaly_score > 0.05:
            self._consecutive += 1
        else:
            self._consecutive = 0
        self.anomaly_active = self._consecutive >= self.temporal_window

        blobs: List[Dict[str, float]] = []
        if fg_mask is not None and self.anomaly_active:
            contours, _ = cv.findContours(
                (fg_mask > 0).astype(np.uint8) * 255,
                cv.RETR_EXTERNAL,
                cv.CHAIN_APPROX_SIMPLE,
            )
            tf = thermal_frame.astype(np.float32)
            for c in contours:
                area = float(cv.contourArea(c))
                if area < self.min_area:
                    continue
                m = cv.moments(c)
                if abs(m["m00"]) < 1e-6:
                    continue
                cx = float(m["m10"] / m["m00"])
                cy = float(m["m01"] / m["m00"])
                obj_mask = np.zeros_like(fg_mask)
                cv.drawContours(obj_mask, [c], -1, 255, -1)
                idx = obj_mask > 0
                if np.any(idx):
                    zvals = 0.6745 * (tf[idx] - med) / (mad + 1e-6)
                    z_norm = float(np.clip((float(np.mean(zvals)) - self.mad_z_thresh) / 3.0, 0.0, 1.0))
                else:
                    z_norm = 0.0
                hm_norm = float(np.clip(np.mean(heat_map.astype(np.float32)[idx]) / 255.0, 0.0, 1.0)) \
                    if (heat_map is not None and np.any(idx)) else 0.0
                score = (1.0 - self.heat_weight) * z_norm + self.heat_weight * hm_norm
                blobs.append({"cx": cx, "cy": cy, "area": area, "score": score})
            blobs.sort(key=lambda b: (b["score"], b["area"]), reverse=True)

        return self.anomaly_score, self.anomaly_active, blobs
