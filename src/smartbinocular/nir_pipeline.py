"""
NIR (IMX) processing: glare metrics, bucket-dispatch helpers, and HybridNIREnhancer.

The main loop routes frames through bucket-specific functions using OPTICAL_BUCKET_DISPATCH.
FrameCache supplies shared downscaled BGR for enhancer paths.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np
import cv2 as cv

# Guided filter (optional): requires opencv-contrib ``cv.ximgproc``
try:
    _cv_ximgproc = cv.ximgproc  # noqa: F821
    _HAS_XIMGPROC = True
except AttributeError:
    _HAS_XIMGPROC = False

# BRISQUE no-reference scoring — requires opencv-contrib + model YAML files.
# Model files ship in the OpenCV source tree but NOT in the pip wheel.
# Place brisque_model_live.yml and brisque_range_live.yml in models/brisque/.
import os as _os
_BRISQUE_DIR = _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "models", "brisque")
_BRISQUE_MODEL = _os.path.normpath(_os.path.join(_BRISQUE_DIR, "brisque_model_live.yml"))
_BRISQUE_RANGE = _os.path.normpath(_os.path.join(_BRISQUE_DIR, "brisque_range_live.yml"))
_HAS_QUALITY = False
_brisque_obj = None
try:
    if _os.path.isfile(_BRISQUE_MODEL) and _os.path.isfile(_BRISQUE_RANGE):
        _brisque_obj = cv.quality.QualityBRISQUE_create(_BRISQUE_MODEL, _BRISQUE_RANGE)
        _HAS_QUALITY = True
except (AttributeError, cv.error):
    pass  # opencv-contrib not installed or model files absent — silent fallback
_brisque_warned = False


def compute_brisque_score(frame: np.ndarray) -> Optional[float]:
    """Return BRISQUE score for *frame* or None if the model files are absent.

    Lower = better perceptual quality. Used as a comparative metric across
    night-relevant optical buckets A/B/D/E; not a ground-truth quality oracle.
    """
    global _brisque_warned
    if not _HAS_QUALITY:
        if not _brisque_warned:
            import warnings
            warnings.warn(
                "[IQA] BRISQUE unavailable — place brisque_model_live.yml and "
                "brisque_range_live.yml in models/brisque/ (opencv-contrib required).",
                UserWarning, stacklevel=2,
            )
            _brisque_warned = True
        return None
    try:
        return float(_brisque_obj.compute(frame)[0])
    except Exception:
        return None


def _min_filter(arr: np.ndarray, ksize: int) -> np.ndarray:
    """Morphological min filter: OpenCV erode (primary — ~30× faster than scipy on aarch64).
    SciPy path retained for reference but not used; see docs/tables/timing/morph_microbench.csv."""
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (ksize, ksize))
    return cv.erode(arr.astype(np.float32), kernel)


def _max_filter(arr: np.ndarray, ksize: int) -> np.ndarray:
    """Morphological max filter: OpenCV dilate (primary — ~30× faster than scipy on aarch64)."""
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (ksize, ksize))
    return cv.dilate(arr.astype(np.float32), kernel)


def _nir_mean_brightness_bgr(nir_bgr: np.ndarray, subsample: int) -> float:
    """Mean NIR luminance proxy (green channel) for HUD / metrics.

    ``subsample == 1``: full-resolution mean. Larger ``k``: downscale by ~k before mean.
    """
    k = int(subsample)
    if k <= 1:
        return float(np.mean(nir_bgr[:, :, 1]))
    h, w = nir_bgr.shape[:2]
    tw = max(1, w // k)
    th = max(1, h // k)
    small = cv.resize(nir_bgr, (tw, th), interpolation=cv.INTER_AREA)
    return float(np.mean(small[:, :, 1]))


def _nir_gray_std_quick(nir_bgr: np.ndarray, max_side: int = 128) -> float:
    g = nir_bgr[:, :, 1]  # green channel as NIR luminance proxy
    h, w = g.shape[:2]
    m = max(h, w)
    if m > max_side:
        s = max_side / float(m)
        g = cv.resize(g, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv.INTER_AREA)
    return float(np.std(g.astype(np.float32)))


def _nir_gray_for_stats(frame_bgr: np.ndarray, max_side: int = 112) -> np.ndarray:
    gray = frame_bgr[:, :, 1]  # green channel as NIR luminance proxy
    h, w = gray.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return gray
    scale = max_side / float(m)
    sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
    return cv.resize(gray, (sw, sh), interpolation=cv.INTER_AREA)


def _percentile_u8_from_hist(hist: np.ndarray, pct: float) -> float:
    """Percentile for uint8 data via histogram CDF (faster than ``np.percentile`` on small frames)."""
    total = int(hist.sum())
    if total <= 0:
        return 0.0
    p = float(np.clip(pct, 0.0, 100.0))
    rank = int(math.ceil((p / 100.0) * total)) - 1
    rank = max(0, min(rank, total - 1))
    cdf = np.cumsum(hist)
    idx = int(np.searchsorted(cdf, rank, side="left"))
    return float(idx)


def _nir_glare_metrics_from_gray(
    gray_u8: np.ndarray,
    *,
    high_pct: float = 95.0,
    saturate_at: float = 233.0,
    use_fast: bool = True,
) -> Tuple[bool, bool, float, float, float]:
    if use_fast:
        hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.int32)
        p_hi = _percentile_u8_from_hist(hist, high_pct)
        p99 = _percentile_u8_from_hist(hist, 99.0)
        mx = float(np.max(gray_u8))
    else:
        g = gray_u8.astype(np.float32)
        p_hi = float(np.percentile(g, high_pct))
        p99 = float(np.percentile(g, 99.0))
        mx = float(np.max(g))
    need = bool(p_hi >= saturate_at or p99 >= 242.0 or mx >= 249.0)
    hud = bool(p_hi >= 248.0 or (p99 >= 254.0 and mx >= 254.0))
    return need, hud, p_hi, p99, mx


def nir_compute_gray_cached(
    nir_bgr: np.ndarray,
    subsample: int = 4,
    max_side: int = 128,
) -> Tuple[np.ndarray, float]:
    """Compute downscaled gray once; return (gray_small, mean_brightness).

    Standardizes to max_side=128 for use in std/glare, also returns brightness.
    The brightness uses subsample for speed (same as _nir_mean_brightness_bgr).
    """
    k = max(1, int(subsample))
    h, w = nir_bgr.shape[:2]
    # Resize to max_side for std/glare usage
    m = max(h, w)
    if m > max_side:
        scale = max_side / float(m)
        tw, th = max(1, int(w * scale)), max(1, int(h * scale))
    else:
        tw, th = w, h
    small = cv.resize(nir_bgr, (tw, th), interpolation=cv.INTER_AREA)
    gray = small[:, :, 1]  # green channel as NIR luminance proxy
    brightness = float(np.mean(gray))
    return gray, brightness


def nir_skip_second_anti_glare_bgr(
    *,
    bucket: str,
    use_raw_nir: bool,
    nir_optical_lite: bool,
) -> bool:
    """True when display must not run ``nir_anti_glare_bgr`` again (bucket C non-lite)."""
    if use_raw_nir or nir_optical_lite:
        return False
    return bucket == "C"


def nir_glare_eval(
    frame_bgr: np.ndarray,
    *,
    high_pct: float = 95.0,
    saturate_at: float = 233.0,
    use_fast: bool = True,
) -> Tuple[bool, bool]:
    g = _nir_gray_for_stats(frame_bgr)
    need, hud, _p_hi, _p99, _mx = _nir_glare_metrics_from_gray(
        g,
        high_pct=high_pct,
        saturate_at=saturate_at,
        use_fast=use_fast,
    )
    return need, hud


def nir_highlight_need_compress(
    frame_bgr: np.ndarray,
    high_pct: float = 95.0,
    saturate_at: float = 233.0,
) -> bool:
    need, _hud = nir_glare_eval(
        frame_bgr,
        high_pct=high_pct,
        saturate_at=saturate_at,
        use_fast=True,
    )
    return need


def nir_glare_hud_should_show(frame_bgr: np.ndarray) -> bool:
    _need, hud = nir_glare_eval(
        frame_bgr,
        high_pct=95.0,
        saturate_at=233.0,
        use_fast=True,
    )
    return hud


def nir_anti_glare_should_apply_bgr(
    frame_bgr: np.ndarray,
    high_pct: float = 95.0,
    saturate_at: float = 233.0,
) -> bool:
    return nir_highlight_need_compress(frame_bgr, high_pct, saturate_at)


def nir_a1_lite_tone_map_bgr(
    frame_bgr: np.ndarray,
    gamma: float = 0.72,
    roll_t: float = 0.92,
    roll_scale: float = 0.38,
    shadow_floor: float = 0.17,
    shadow_ceil: float = 0.48,
) -> np.ndarray:
    """A1-lite tone mapping."""
    gray = frame_bgr[:, :, 1].astype(np.float32)  # green channel as luminance proxy
    g0 = np.clip(gray / 255.0, 1e-6, 1.0)
    g = np.power(g0, gamma)
    g = np.where(g > roll_t, roll_t + (g - roll_t) * roll_scale, g)
    ratio = np.clip(g / g0, 0.0, 1.55)
    span = max(shadow_ceil - shadow_floor, 1e-6)
    w = np.clip((g0 - shadow_floor) / span, 0.0, 1.0)
    ratio_eff = 1.0 + w * (ratio - 1.0)
    return np.clip(frame_bgr.astype(np.float32) * ratio_eff[..., None], 0, 255).astype(np.uint8)


def nir_anti_glare_bgr(
    frame_bgr: np.ndarray,
    high_pct: float = 95.0,
    saturate_at: float = 233.0,
    gamma: float = 0.72,
) -> np.ndarray:
    """C4 NIR: highlight gate + A1-lite tone map in one pass (no post-CLAHE)."""
    # guard: C — require mean brightness > 60/255 to avoid darkening deep-night scenes
    # see mis_dispatch_matrix_summary.md (bucket=C, passthrough confirmed on night classes)
    if float(np.mean(frame_bgr)) < 60.0:
        return frame_bgr
    if not nir_highlight_need_compress(frame_bgr, high_pct, saturate_at):
        return frame_bgr
    return nir_a1_lite_tone_map_bgr(frame_bgr, gamma=gamma)


# ─── Bucket B: nir_night — single CLAHE pass ─────────────────────────────────

def nir_nir_night_clahe(frame: np.ndarray, clahe_clip_scale: float = 1.0) -> np.ndarray:
    """Bucket B: NIR-night — one CLAHE at clip=3.0×scale on LAB L channel. ~1-2 ms."""
    lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    # guard: B — clamp clip_scale to 0.5 when input is bright to prevent sat residue
    # see mis_dispatch_matrix_summary.md (bucket=B, env=backlight/extreme_bright)
    pct_sat = float((frame > 250).mean())
    effective_scale = min(clahe_clip_scale, 0.5) if pct_sat > 0.10 else clahe_clip_scale
    clip = float(np.clip(3.0 * effective_scale, 0.5, 8.0))
    clahe = cv.createCLAHE(clipLimit=clip, tileGridSize=(4, 4))
    return cv.cvtColor(cv.merge([clahe.apply(l), a, b]), cv.COLOR_LAB2BGR)


# ─── Bucket D: fog dehaze-lite ────────────────────────────────────────────────

def nir_dehaze_lite(frame: np.ndarray, omega: float = 0.85) -> np.ndarray:
    """Bucket D: DCP dehazing at 160×120 downsample. ~4-6 ms on RPi4.

    omega controls transmission strength: 0.5 (subtle) – 0.95 (aggressive).
    """
    orig_h, orig_w = frame.shape[:2]
    # guard: D — return input unchanged on dark scenes (DCP adds crush on night imagery)
    # see mis_dispatch_matrix_summary.md (bucket=D, env=night_clear/normal_night/mixed_edge)
    small_check = cv.resize(frame, (160, 120), interpolation=cv.INTER_AREA)
    I_check = small_check.astype(np.float32) / 255.0
    if float(I_check.mean()) < 0.18 or float((I_check < 0.05).mean()) > 0.20:
        return frame
    small = small_check
    I = small.astype(np.float32) / 255.0
    dark = np.min(I, axis=2)
    dark = _min_filter(dark, 5)
    top_mask = dark >= np.quantile(dark, 0.99)
    A_val = float(I[top_mask].mean()) if np.any(top_mask) else 0.85
    A_val = float(np.clip(A_val, 0.5, 1.0))
    t = np.clip(1.0 - float(omega) * (dark / max(A_val, 1e-6)), 0.1, 1.0)
    J = np.clip((I - A_val) / t[..., None] + A_val, 0.0, 1.0)
    dehazed = (J * 255.0).astype(np.uint8)
    return cv.resize(dehazed, (orig_w, orig_h), interpolation=cv.INTER_LINEAR)


# ─── Bucket E: rain temporal median ──────────────────────────────────────────

class RainTemporalMedian:
    """Bucket E: N-frame temporal median for rain destreaking. ~4-6 ms on RPi4."""

    def __init__(self, n_frames: int = 3) -> None:
        self.n = max(2, int(n_frames))
        self._buf: deque = deque(maxlen=self.n)
        self._static_count: int = 0  # consecutive near-static frames

    def process(self, frame: np.ndarray) -> np.ndarray:
        # guard: E — skip median on static scenes (saves ~4 ms for stationary camera)
        # see mis_dispatch_matrix_summary.md (bucket=E, near-passthrough on non-rain classes)
        if self._buf:
            prev_mean = float(self._buf[-1].astype(np.float32).mean())
            curr_mean = float(frame.astype(np.float32).mean())
            if abs(curr_mean - prev_mean) < 0.5:
                self._static_count += 1
            else:
                self._static_count = 0
        if self._static_count >= 3:
            self._buf.append(frame.copy())
            return frame

        self._buf.append(frame.copy())
        if len(self._buf) < self.n:
            return frame
        stack = np.stack(list(self._buf), axis=0).astype(np.float32)
        return np.median(stack, axis=0).astype(np.uint8)

    def reset(self) -> None:
        self._buf.clear()
        self._static_count = 0


# ─── Bucket F: transition blend (A + C) ──────────────────────────────────────

def nir_transition_blend(
    frame: np.ndarray,
    nir_enhancer: "HybridNIREnhancer",
    nir_b_ema_norm: float,
    *,
    lo: float = 0.15,
    hi: float = 0.45,
    precomputed_small: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Bucket F: blend Bucket A (night enhance) + Bucket C (anti-glare) by brightness EMA.

    w=0 (dark) → full Bucket A; w=1 (bright) → full Bucket C.
    """
    w = float(np.clip((nir_b_ema_norm - lo) / max(hi - lo, 1e-6), 0.0, 1.0))
    # guard: F — short-circuit when w is degenerate to avoid A+C compute cost
    # see mis_dispatch_matrix_summary.md (bucket=F, w saturates on extreme scenes)
    if w < 0.02:
        return nir_enhancer.process(frame, precomputed_small=precomputed_small)
    if w > 0.98:
        return nir_anti_glare_bgr(frame)
    a_out = nir_enhancer.process(frame, precomputed_small=precomputed_small)
    c_out = nir_anti_glare_bgr(frame)
    return cv.addWeighted(a_out, 1.0 - w, c_out, w, 0.0)


# ─── Optical bucket dispatch table ───────────────────────────────────────────

OPTICAL_BUCKET_DISPATCH: Dict[str, str] = {
    # ENV_CLASS → bucket key
    "night_clear":  "A",  # HybridNIREnhancer (dark channel + adaptive CLAHE)
    "normal_night": "A",  # HybridNIREnhancer
    "nir_night":    "B",  # single CLAHE at clip=3.0 (lower-power night)
    "glare":        "C",  # nir_anti_glare_bgr tone-map / passthrough
    "backlight":    "C",  # nir_anti_glare_bgr
    "normal_day":   "C",  # nir_anti_glare_bgr (passthrough when no compression needed)
    "fog":          "D",  # nir_dehaze_lite DCP at 160×120
    "rain":         "E",  # RainTemporalMedian 3-frame median
    "transition":   "F",  # blend A + C by brightness EMA
    "default":      "A",  # safe fallback
}

# Cheaper substitutes for C/D/E/F when ``nir_optical_lite`` is True in config.
OPTICAL_BUCKET_DISPATCH_LITE: Dict[str, str] = {
    "night_clear": "A",
    "normal_night": "A",
    "nir_night": "B",
    "glare": "B",
    "backlight": "B",
    "normal_day": "B",
    "fog": "B",
    "rain": "E",
    "transition": "A",
    "default": "A",
}


def resolve_optical_bucket(env_class: str, *, lite: bool) -> str:
    """Return bucket key for ``env_class``; ``lite`` uses OPTICAL_BUCKET_DISPATCH_LITE."""
    table = OPTICAL_BUCKET_DISPATCH_LITE if lite else OPTICAL_BUCKET_DISPATCH
    return table.get(env_class, table.get("default", "A"))


class HybridNIREnhancer:
    """NIR pipeline: dark/bright channel, atmosphere estimate, adaptive CLAHE, and detail.

    Operates in BGR end-to-end (no forced grayscale; no Schmitt trigger)."""

    def __init__(
        self,
        proc_w: int = 320,
        proc_h: int = 240,
        patch_size: int = 5,
        update_rate: int = 8,
        *,
        detail_strength: float = 0.25,
        clahe_clip_scale: float = 1.0,
    ):
        self.proc_w = proc_w
        self.proc_h = proc_h
        self.patch = patch_size if patch_size % 2 == 1 else patch_size + 1
        self.update_rate = update_rate

        self.frame_count = 0
        self.last_A = np.array([0.7, 0.7, 0.7], dtype=np.float32)
        self.last_weight = None
        self.A_buffer: deque = deque(maxlen=10)
        self.A_buffer.append(self.last_A)
        self.brightness_buffer: deque = deque(maxlen=15)

        _cs = float(np.clip(float(clahe_clip_scale), 0.25, 2.0))
        # Three fixed CLAHE objects selected by brightness — matches night_vision.py pattern.
        # Pre-created once, reused every frame; no per-frame clip recalculation.
        self.clahe_levels = {
            "very_dark": cv.createCLAHE(clipLimit=3.0 * _cs, tileGridSize=(4, 4)),
            "dark": cv.createCLAHE(clipLimit=2.0 * _cs, tileGridSize=(6, 6)),
            "medium": cv.createCLAHE(clipLimit=1.5 * _cs, tileGridSize=(8, 8)),
        }
        self.detail_strength = float(max(0.0, detail_strength))
        self.min_boost = 1.1
        self.max_boost = 1.8
        self.p = 0.05
        self._cc_buf = np.empty((proc_h, proc_w, 3), dtype=np.float32)

    def update_runtime_params(self, opt_cfg: Dict) -> None:
        """Update detail_strength and CLAHE clip scale from opt_cfg_runtime on preset-stable change.

        Called at most once per stable-preset transition (not per frame). Safe when keys are absent.
        """
        if "nir_enhancer_detail_strength" in opt_cfg:
            self.detail_strength = float(max(0.0, float(opt_cfg["nir_enhancer_detail_strength"])))
        if "nir_enhancer_clahe_clip_scale" in opt_cfg:
            _cs = float(np.clip(float(opt_cfg["nir_enhancer_clahe_clip_scale"]), 0.25, 2.0))
            self.clahe_levels = {
                "very_dark": cv.createCLAHE(clipLimit=3.0 * _cs, tileGridSize=(4, 4)),
                "dark":      cv.createCLAHE(clipLimit=2.0 * _cs, tileGridSize=(6, 6)),
                "medium":    cv.createCLAHE(clipLimit=1.5 * _cs, tileGridSize=(8, 8)),
            }

    def reset(self) -> None:
        """Reset all temporal EMA state. Required for still-image batch processing.

        Without this, frame_count gating (% update_rate) silently uses stale channel
        estimates for most images. Call reset() + construct with update_rate=1 for
        batch stills (still_image_cold_start_mode).
        """
        self.frame_count = 0
        self.last_A = np.array([0.7, 0.7, 0.7], dtype=np.float32)
        self.last_weight = None
        self.A_buffer.clear()
        self.A_buffer.append(self.last_A)
        self.brightness_buffer.clear()

    def _compute_channels(self, I_float: np.ndarray):
        min_ch = np.min(I_float, axis=2)
        max_ch = np.max(I_float, axis=2)
        dark = _min_filter(min_ch, self.patch)
        bright = _max_filter(max_ch, self.patch)
        return dark, bright

    def _estimate_atmosphere(self, I_float: np.ndarray, bright_map: np.ndarray) -> np.ndarray:
        threshold = np.quantile(bright_map, 1 - self.p)
        mask = bright_map >= threshold
        if np.any(mask):
            A = np.mean(I_float[mask], axis=0)
        else:
            idx = np.argmax(bright_map)
            y, x = np.unravel_index(idx, bright_map.shape)
            A = I_float[y, x]
        A = np.clip(A, 0.2, 0.9)
        if len(self.A_buffer) > 0:
            A = 0.7 * A + 0.3 * np.mean(self.A_buffer, axis=0)
        return A

    def _make_weight_map(self, dark_map: np.ndarray, bright_map: np.ndarray, A: np.ndarray) -> np.ndarray:
        weight = 1.0 - dark_map
        weight += 0.5 * np.abs(bright_map - np.mean(A))
        weight[bright_map > 0.7] *= 0.3
        weight = np.clip(weight, 0, 1)
        weight = cv.GaussianBlur(weight.astype(np.float32), (5, 5), 1.0)
        return weight

    def _apply_clahe_boost(
        self,
        bgr_small: np.ndarray,
        weight_map: Optional[np.ndarray],
        avg_brightness: float,
        *,
        cur_bright: float = 0.5,
    ) -> np.ndarray:
        """Single LAB pass: CLAHE + weight-map boost + unsharp + optional dark-scene L boost.

        CLAHE selection follows night_vision.py: 3 fixed pre-created objects selected by
        avg_brightness. No per-frame entropy computation (entropy mode removed to reduce
        ~0.2 ms/frame overhead and match the simpler, predictable cost of night_vision.py).

        Dark-scene L boost (when cur_bright is low) replaces the former gray + Schmitt path.
        """
        lab = cv.cvtColor(bgr_small, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(lab)

        # Select CLAHE by brightness — identical to night_vision.py logic.
        if avg_brightness < 0.25:
            clahe = self.clahe_levels["very_dark"]
            base_boost = self.max_boost
        elif avg_brightness < 0.45:
            clahe = self.clahe_levels["dark"]
            base_boost = (self.max_boost + self.min_boost) / 2
        else:
            clahe = self.clahe_levels["medium"]
            base_boost = self.min_boost

        l_enh = clahe.apply(l).astype(np.float32) / 255.0
        if weight_map is not None:
            wm = cv.resize(weight_map, (l_enh.shape[1], l_enh.shape[0])) if weight_map.shape != l_enh.shape else weight_map
            l_enh = l_enh * (1.0 + (base_boost - 1.0) * wm)

        if self.detail_strength > 0:
            blur = cv.GaussianBlur(l_enh, (3, 3), 0.5)
            l_enh = l_enh + self.detail_strength * (l_enh - blur)

        # Dark-scene L boost on BGR path (no grayscale conversion).
        # guard: A.bright — skip night_boost on medium/bright inputs to prevent oversaturation
        # Threshold 0.30 (76/255) avoids 1.3× boost on intermediate-brightness NIR scenes
        # (e.g. nir_night at mean green 0.235–0.431) while preserving 2.2×/1.6× for dark.
        # see mis_dispatch_matrix_summary.md (bucket=A, env=nir_night/backlight/mixed_edge)
        if cur_bright < 0.30:
            night_boost = 2.2 if cur_bright < 0.15 else 1.6
            l_enh = l_enh * night_boost

        l_out = np.clip(l_enh * 255, 0, 255).astype(np.uint8)
        return cv.cvtColor(cv.merge([l_out, a, b]), cv.COLOR_LAB2BGR)

    def _color_correct(self, bgr: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Shift colors 30% toward atmosphere vector ``A`` (dehaze helper)."""
        np.copyto(self._cc_buf, bgr, casting="unsafe")
        self._cc_buf /= 255.0
        avg = np.mean(self._cc_buf, axis=(0, 1))
        self._cc_buf += (A - avg) * 0.3
        np.clip(self._cc_buf, 0.0, 1.0, out=self._cc_buf)
        self._cc_buf *= 255.0
        return self._cc_buf.astype(np.uint8)

    def process(
        self,
        frame: np.ndarray,
        thermal_guide: Optional[np.ndarray] = None,
        precomputed_small: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Enhance one NIR BGR frame at working resolution (CLAHE + brightness-guided fusion).

        ``thermal_guide``: raw thermal 80×62 for optional guided filter.
        ``precomputed_small``: BGR at ``(proc_h, proc_w)`` from ``FrameCache.nir_320`` when
        ``nir_enhancer_proc_w/h`` match; otherwise a resize from full frame.
        """
        orig_h, orig_w = frame.shape[:2]
        if precomputed_small is not None and precomputed_small.shape[:2] == (self.proc_h, self.proc_w):
            small = precomputed_small
        else:
            small = cv.resize(frame, (self.proc_w, self.proc_h))

        # Guided filter: upsampled thermal guides NIR denoising when contrib is available.
        # Requires opencv-contrib (ximgproc). Gracefully skipped if not available.
        if _HAS_XIMGPROC and thermal_guide is not None:
            guide_up = cv.resize(
                thermal_guide if thermal_guide.dtype == np.uint8
                else np.clip(thermal_guide, 0, 255).astype(np.uint8),
                (self.proc_w, self.proc_h),
                interpolation=cv.INTER_LINEAR,
            )
            guide_gray = cv.cvtColor(guide_up, cv.COLOR_GRAY2BGR) \
                if guide_up.ndim == 2 else guide_up
            # Apply guided filter on BGR small; radius=8, eps=1e-4 (~0.01²)
            small = cv.ximgproc.guidedFilter(
                guide=guide_gray, src=small, radius=8, eps=1e-4
            )

        self.frame_count += 1
        # Brightness from green channel (luminance proxy); ``small`` stays BGR for the pipeline.
        cur_bright = float(np.mean(small[:, :, 1])) / 255.0
        self.brightness_buffer.append(cur_bright)

        # guard: A.dark — suppress detail_strength on extreme-dark inputs to avoid noise amp
        # see mis_dispatch_matrix_summary.md (bucket=A, env=mixed_edge/extreme_dark)
        _effective_detail = 0.0 if cur_bright < 15.0 / 255.0 else self.detail_strength

        need_update = (
            self.frame_count % self.update_rate == 0
            or self.last_weight is None
            or len(self.brightness_buffer) < 5
        )

        if need_update:
            I_float = small.astype(np.float32) / 255.0
            dark, bright = self._compute_channels(I_float)
            A = self._estimate_atmosphere(I_float, bright)
            weight = self._make_weight_map(dark, bright, A)
            self.last_A = A
            self.last_weight = weight
            self.A_buffer.append(A)
        else:
            weight = self.last_weight

        avg_b = float(np.mean(self.brightness_buffer)) if self.brightness_buffer else cur_bright
        _saved_detail = self.detail_strength
        self.detail_strength = _effective_detail  # apply A.dark guard for this frame
        enhanced = self._apply_clahe_boost(small, weight, avg_b, cur_bright=cur_bright)
        self.detail_strength = _saved_detail
        enhanced = self._color_correct(enhanced, self.last_A)

        return cv.resize(enhanced, (orig_w, orig_h))
