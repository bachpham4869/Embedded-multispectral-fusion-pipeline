#!/usr/bin/env python3
"""
fusion_live_unified.py — MỘT FILE THỐNG NHẤT (3 MODE + TOÀN CỤC + MỞ RỘNG)
==========================================================================
Bản **độc lập**: gồm toàn bộ pipeline fusion (giống ``fusion_live_standalone.py``)
**và** khối thuật toán **toàn cục** (D1/D2/C4) + **metrics JSON (luận văn)** **nhúng trong file** —
**không** import ``fusion_live_standalone``, ``fusion_advanced``, ``live_global_services``, hay module metrics riêng.

Mục đích:
  • Một chỗ để phát triển: thêm tracking, IMU, alert, … mà không phải nhảy nhiều file.
  • ``fusion_live_standalone.py`` + ``live_global_services.py`` vẫn giữ để tách module / tái sử dụng;
    khi ổn định có thể đồng bộ ngược patch sang các file đó.

Cấu trúc:
  SECTION 0 — Toàn cục (D1, D2, C4) + placeholder mở rộng
  SECTION 1 — Thermal → NIR → Fusion → capture → main

Phím: 1/2/3, R, S, A, +/-, Q. **+/-** chỉnh **alpha** pha trộn NIR/thermal ở mode Fusion (0.05–1.0). Toàn cục §1b tự chạy (C4, A6b blend, D2).

GLARE (HUD): chỉ khi đuôi histogram **rất** sáng (ít nhạy). Ảnh vẫn có thể được **nén highlight** sớm hơn qua ``nir_highlight_need_compress`` / thermal tương ứng (gamma + roll-off NIR).
"""

from __future__ import annotations

import sys
import os
import json
import uuid
import time
import threading
import signal as sig_module
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import math
import platform
import numpy as np
import cv2 as cv

# ── Metrics phiên / manifest (JSON) — nhúng trong file, không import module ngoài ─

_METRICS_SCHEMA_VERSION = "1.1"


def _metrics_write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _metrics_mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _metrics_std_sample(xs: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return float(math.sqrt(v))


def metrics_write_run_manifest(
    path: str,
    *,
    session_id: str,
    homography_path: str,
    display_size: tuple,
    fov_deg: tuple,
    tier_s_flags: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    import subprocess

    git_rev = None
    _git_cwd = os.path.dirname(os.path.abspath(homography_path))
    try:
        git_rev = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=_git_cwd,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            .decode()
            .strip()
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass

    data: Dict[str, Any] = {
        "schema_version": _METRICS_SCHEMA_VERSION,
        "kind": "run_manifest",
        "session_id": session_id,
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "unix_time": time.time(),
        "homography_file": homography_path,
        "display_px": [int(display_size[0]), int(display_size[1])],
        "fov_deg": {"horizontal": float(fov_deg[0]), "vertical": float(fov_deg[1])},
        "tier_s_integrated": tier_s_flags,
        "git_revision": git_rev,
        "platform": platform.platform(),
    }
    if extra:
        data["extra"] = extra
    _metrics_write_json(path, data)


@dataclass
class ThesisRunMetrics:
    """Gom thống kê theo frame; ``finalize()`` xuất báo cáo phiên."""

    session_id: str = ""
    homography_path: str = ""
    display_size: tuple = (0, 0)
    fusion_alpha: float = 0.55
    nir_schmitt_raw_on: float = 30.0
    nir_schmitt_dim_on: float = 18.0
    nir_b_ema_coef: float = 0.18

    t0: float = field(default_factory=time.perf_counter)
    frame_count: int = 0
    frames_by_mode: Dict[str, int] = field(default_factory=dict)
    fps_samples: deque = field(default_factory=lambda: deque(maxlen=900))
    nir_brightness_samples: deque = field(default_factory=lambda: deque(maxlen=2000))
    jerk_frames: int = 0
    glare_nir_frames: int = 0
    glare_th_frames: int = 0
    night_mode_frames: int = 0
    thermal_not_ready_frames: int = 0
    stream_skew_ms_samples: deque = field(default_factory=lambda: deque(maxlen=1200))

    def record_frame(
        self,
        *,
        mode: str,
        fps: float,
        nir_brightness: Optional[float] = None,
        nir_use_raw_auto: Optional[bool] = None,
        is_night_mode: bool = False,
        jerk_active: bool = False,
        glare_nir: bool = False,
        glare_th: bool = False,
        thermal_is_ready: bool = True,
        stream_skew_ms: Optional[float] = None,
    ) -> None:
        self.frame_count += 1
        self.frames_by_mode[mode] = self.frames_by_mode.get(mode, 0) + 1
        self.fps_samples.append(float(fps))
        if nir_brightness is not None:
            self.nir_brightness_samples.append(float(nir_brightness))
        if is_night_mode and mode in ("imx", "fusion"):
            self.night_mode_frames += 1
        if jerk_active:
            self.jerk_frames += 1
        if glare_nir:
            self.glare_nir_frames += 1
        if glare_th:
            self.glare_th_frames += 1
        if not thermal_is_ready and mode in ("thermal", "fusion"):
            self.thermal_not_ready_frames += 1
        if stream_skew_ms is not None:
            self.stream_skew_ms_samples.append(float(stream_skew_ms))

    def build_capture_meta(
        self,
        *,
        image_filename: str,
        trigger: str,
        mode: str,
        fps: float,
        alpha: float,
        nir_brightness: Optional[float] = None,
        nir_b_ema: Optional[float] = None,
        nir_use_raw_auto: Optional[bool] = None,
        show_raw_toggle: bool = False,
        is_night_mode: bool = False,
        jerk_active: bool = False,
        glare_nir: bool = False,
        glare_th: bool = False,
        tags_hud: Optional[List[str]] = None,
        bearing_h_deg: Optional[float] = None,
        bearing_v_deg: Optional[float] = None,
        stream_skew_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "schema_version": _METRICS_SCHEMA_VERSION,
            "kind": "capture",
            "session_id": self.session_id,
            "trigger": trigger,
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "unix_time": time.time(),
            "image_file": image_filename,
            "mode": mode,
            "display_px": [int(self.display_size[0]), int(self.display_size[1])],
            "fusion_alpha": float(alpha),
            "fps_instant": round(float(fps), 3),
            "homography_file": self.homography_path,
            "nir_mean_gray": None if nir_brightness is None else round(float(nir_brightness), 4),
            "nir_brightness_ema": None if nir_b_ema is None else round(float(nir_b_ema), 4),
            "nir_use_raw_auto": nir_use_raw_auto,
            "show_raw_manual": bool(show_raw_toggle),
            "nir_night_mode": bool(is_night_mode),
            "d2_jerk_active": bool(jerk_active),
            "glare_nir_hud": bool(glare_nir),
            "glare_thermal_hud": bool(glare_th),
            "hud_tags": tags_hud or [],
            "platform": platform.platform(),
        }
        if bearing_h_deg is not None:
            d["bearing_offset_h_deg"] = round(float(bearing_h_deg), 4)
        if bearing_v_deg is not None:
            d["bearing_offset_v_deg"] = round(float(bearing_v_deg), 4)
        if stream_skew_ms is not None:
            d["nir_thermal_stream_skew_ms"] = round(float(stream_skew_ms), 3)
        return d

    def finalize(self) -> Dict[str, Any]:
        t1 = time.perf_counter()
        dur = t1 - self.t0
        fps_list = list(self.fps_samples)
        nir_list = list(self.nir_brightness_samples)
        n = max(1, self.frame_count)
        skew_list = list(self.stream_skew_ms_samples)
        return {
            "schema_version": _METRICS_SCHEMA_VERSION,
            "kind": "session_summary",
            "software": "fusion_live_unified",
            "session_id": self.session_id,
            "session_start_perf_s": self.t0,
            "duration_wall_s": round(dur, 3),
            "frames_total": self.frame_count,
            "frames_by_mode": dict(self.frames_by_mode),
            "fps_mean": round(_metrics_mean(fps_list), 3) if fps_list else 0.0,
            "fps_std_sample": round(_metrics_std_sample(fps_list), 3) if fps_list else 0.0,
            "fps_sample_count": len(fps_list),
            "homography_file": self.homography_path,
            "display_px": [int(self.display_size[0]), int(self.display_size[1])],
            "fusion_alpha_config": float(self.fusion_alpha),
            "nir_pipeline": {
                "schmitt_raw_on": self.nir_schmitt_raw_on,
                "schmitt_dim_on": self.nir_schmitt_dim_on,
                "brightness_ema_coef": self.nir_b_ema_coef,
            },
            "nir_brightness_mean": round(_metrics_mean(nir_list), 4) if nir_list else None,
            "nir_brightness_std": round(_metrics_std_sample(nir_list), 4) if len(nir_list) > 1 else None,
            "nir_brightness_sample_count": len(nir_list),
            "rate_jerk_frames": round(self.jerk_frames / n, 5),
            "rate_glare_nir_frames": round(self.glare_nir_frames / n, 5),
            "rate_glare_thermal_frames": round(self.glare_th_frames / n, 5),
            "rate_nir_night_mode_frames": round(self.night_mode_frames / n, 5),
            "thermal_warmup_frames_observed": self.thermal_not_ready_frames,
            "nir_thermal_stream_skew_ms_mean": round(_metrics_mean(skew_list), 4) if skew_list else None,
            "nir_thermal_stream_skew_ms_std": round(_metrics_std_sample(skew_list), 4) if len(skew_list) > 1 else None,
            "nir_thermal_stream_skew_sample_count": len(skew_list),
            "platform": platform.platform(),
        }

    def summary_line(self, report: Optional[Dict[str, Any]] = None) -> str:
        r = report if report is not None else self.finalize()
        return (
            f"[metrics] frames={r['frames_total']} duration_s={r['duration_wall_s']:.1f} "
            f"fps_mean={r['fps_mean']:.1f}±{r['fps_std_sample']:.2f} "
            f"jerk_rate={r['rate_jerk_frames']:.3f} glare_nir_rate={r['rate_glare_nir_frames']:.3f}"
        )


# Tên công khai giống module metrics cũ (test / script ngoài)
SCHEMA_VERSION = _METRICS_SCHEMA_VERSION
write_metrics_json = _metrics_write_json


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — Thuật toán TOÀN CỤC (§1b) + chỗ thêm sau (tracking, IMU, GNSS, …)
# Metrics JSON (khối trên) nhúng trong file — có thể copy sang module riêng nếu cần.
# ═══════════════════════════════════════════════════════════════════════════════


def thermal_highlight_need_compress(
    frame_u8: np.ndarray,
    high_pct: float = 99.0,
    saturate_at: float = 240.0,
) -> bool:
    f = frame_u8.astype(np.float32)
    p_hi = float(np.percentile(f, high_pct))
    return p_hi >= saturate_at


def thermal_glare_hud_should_apply(
    frame_u8: np.ndarray,
    high_pct: float = 99.5,
    saturate_at: float = 252.0,
    min_max_u8: float = 253.0,
) -> bool:
    """HUD GLARE thermal: chỉ khi đuôi histogram **rất** cao *và* có pixel gần bão hòa.

    Tránh báo giả từ nhiễu nền 80×62: **không** dùng ảnh sau AGC (dải bị kéo, p99 dễ cao).
    """
    f = frame_u8.astype(np.float32)
    p_hi = float(np.percentile(f, high_pct))
    mx = float(np.max(f))
    return p_hi >= saturate_at and mx >= min_max_u8


def thermal_anti_glare_should_apply(
    frame_u8: np.ndarray,
    high_pct: float = 99.0,
    saturate_at: float = 240.0,
) -> bool:
    return thermal_highlight_need_compress(frame_u8, high_pct, saturate_at)


def thermal_anti_glare(
    frame_u8: np.ndarray,
    high_pct: float = 99.0,
    saturate_at: float = 240.0,
    gamma: float = 0.74,
) -> np.ndarray:
    """C4 thermal: gamma trên pixel; không đổi geometry."""
    if not thermal_highlight_need_compress(frame_u8, high_pct, saturate_at):
        return frame_u8
    f = frame_u8.astype(np.float32)
    out = np.clip(np.power(f / 255.0, gamma) * 255.0, 0, 255)
    return out.astype(np.uint8)


def nir_glare_allowed(
    nir_brightness_mean: float,
    nir_raw_threshold: float,
    is_night_mode: bool,
    *,
    display_raw: bool = False,
) -> bool:
    """NIR anti-glare chỉ khi cảnh sáng / flare; night grayscale → không (tránh báo giả sau enhance).

    ``display_raw``: đang hiển thị buffer raw (phím R hoặc auto raw sau hysteresis) → luôn cho phép
    (tránh stale ``is_night_mode`` khi không gọi ``process()``).
    """
    if display_raw or nir_brightness_mean >= nir_raw_threshold:
        return True
    return not is_night_mode


def _nir_gray_for_stats(frame_bgr: np.ndarray, max_side: int = 112) -> np.ndarray:
    gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return gray
    scale = max_side / float(m)
    sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
    return cv.resize(gray, (sw, sh), interpolation=cv.INTER_AREA)


def nir_highlight_need_compress(
    frame_bgr: np.ndarray,
    high_pct: float = 95.0,
    saturate_at: float = 233.0,
) -> bool:
    g = _nir_gray_for_stats(frame_bgr)
    p_hi = float(np.percentile(g, high_pct))
    if p_hi >= saturate_at:
        return True
    p99 = float(np.percentile(g, 99.0))
    mx = float(np.max(g))
    return p99 >= 242.0 or mx >= 249.0


def nir_glare_hud_should_show(frame_bgr: np.ndarray) -> bool:
    g = _nir_gray_for_stats(frame_bgr)
    p_hi = float(np.percentile(g, 95))
    p99 = float(np.percentile(g, 99.0))
    mx = float(np.max(g))
    return p_hi >= 248.0 or (p99 >= 254.0 and mx >= 254.0)


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
    """A1-lite — đồng bộ ``live_global_services``."""
    gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY).astype(np.float32)
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
    """C4 NIR: cổng nén + A1-lite một pass; không CLAHE sau nén (ít nhiễu, nhẹ hơn trên RPi)."""
    if not nir_highlight_need_compress(frame_bgr, high_pct, saturate_at):
        return frame_bgr
    return nir_a1_lite_tone_map_bgr(frame_bgr, gamma=gamma)


class JerkGate:
    """D2: proxy rung từ chênh lệch hai frame NIR **raw** (thay IMU sau).

    Dùng ảnh thu nhỏ + ngưỡng cao + N khung liên tiếp vượt ngưỡng để giảm báo giả (nhiễu AE/CLAHE).
    """

    def __init__(
        self,
        diff_threshold: float = 8.5,
        hold_frames: int = 12,
        consecutive_frames: int = 2,
        percentile: float = 94.0,
        max_side: int = 128,
    ):
        self.diff_threshold = diff_threshold
        self.hold_frames = hold_frames
        self.consecutive_frames = max(1, int(consecutive_frames))
        self.percentile = float(percentile)
        self.max_side = int(max_side)
        self._prev_gray = None
        self._hold = 0
        self._strike = 0
        self.active = False

    def reset(self) -> None:
        self._prev_gray = None
        self._hold = 0
        self._strike = 0
        self.active = False

    def _to_small_gray(self, nir_bgr: np.ndarray) -> np.ndarray:
        gray = cv.cvtColor(nir_bgr, cv.COLOR_BGR2GRAY)
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

    def update(self, nir_bgr):
        if nir_bgr is None:
            if self._hold > 0:
                self._hold -= 1
            self.active = self._hold > 0
            return
        gray = self._to_small_gray(nir_bgr)
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            df = cv.absdiff(gray, self._prev_gray).astype(np.float32)
            m = float(np.mean(df))
            p = float(np.percentile(df, self.percentile))
            score = 0.48 * m + 0.52 * p
            if score > self.diff_threshold:
                self._strike += 1
                if self._strike >= self.consecutive_frames:
                    self._hold = self.hold_frames
                    self._strike = 0
            else:
                self._strike = 0
        else:
            self._strike = 0
        self._prev_gray = gray.copy()
        if self._hold > 0:
            self._hold -= 1
            self.active = True
        else:
            self.active = False


@dataclass
class OpsecLocalOnly:
    """D1: chỉ local; khi bật, hook mạng/stream kiểm tra allow_outbound_network()."""

    enabled: bool = False

    def allow_outbound_network(self) -> bool:
        return not self.enabled

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        return self.enabled


# ── A6b: đồng bộ live_global_services (One Euro + blend adaptive) ─


class OneEuroFilter1D:
    """One Euro 1D — không cần sensor; chỉ cần đo và thời gian (đồng bộ live_global_services)."""

    __slots__ = ("min_cutoff", "beta", "d_cutoff", "x_prev", "dx_prev", "t_prev", "_initialized")

    def __init__(self, min_cutoff: float = 1.15, beta: float = 0.018, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = 0.0
        self.dx_prev = 0.0
        self.t_prev = 0.0
        self._initialized = False

    def reset(self) -> None:
        self._initialized = False

    @staticmethod
    def _smoothing_factor(dt: float, cutoff: float) -> float:
        if dt <= 0:
            dt = 1e-6
        r = 2.0 * math.pi * cutoff * dt
        return r / (r + 1.0)

    def __call__(self, x: float, t: float) -> float:
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


class DisplayShakeReducerLite:
    """Giảm rung hiển thị: blend (IIR + adaptive) hoặc shift (phaseCorrelate + One Euro)."""

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
        if mode not in ("off", "blend", "shift"):
            mode = "off"
        self.mode = mode
        self.reset()

    def reset(self) -> None:
        self._prev_bgr = None
        self._prev_gray_small = None
        self._sx = self._sy = 0.0
        self._euro_x.reset()
        self._euro_y.reset()

    def process(self, frame_bgr: np.ndarray, jerk_active: bool = False) -> np.ndarray:
        if self.mode == "off" or frame_bgr is None:
            return frame_bgr
        if jerk_active:
            self._prev_bgr = frame_bgr.copy()
            g = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
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
                g0 = cv.cvtColor(self._prev_bgr, cv.COLOR_BGR2GRAY)
                g1 = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
                md = float(np.mean(cv.absdiff(g1, g0)))
                a = float(np.clip(a0 + 0.14 * math.tanh(md / 4.0) - 0.07, 0.36, 0.9))
            else:
                a = a0
            out = cv.addWeighted(frame_bgr, a, self._prev_bgr, 1.0 - a, 0)
            self._prev_bgr = out.copy()
            return out

        g = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
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
        order = ("off", "blend", "shift")
        try:
            i = order.index(self.mode)
        except ValueError:
            i = 0
        self.set_mode(order[(i + 1) % len(order)])
        return self.mode


# --- Gợi ý mở rộng (stub — triển khai dần): ------------------------------------
# class ImuJerkReader: ...   # thay JerkGate.update bằng gyro
# class GlobalHudTags: ...    # gom [LOCAL][JERK]…
# -----------------------------------------------------------------------------


# ═══════════════════════════════════════════════════════════════════════════════
# Phần dưới: pipeline fusion (thermal / NIR / fusion / main) — như bản standalone
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from scipy.ndimage import minimum_filter, maximum_filter
    _HAS_SCIPY_NDIMAGE = True
except ImportError:
    _HAS_SCIPY_NDIMAGE = False

# --- NumPy 2.x check ---
try:
    _nv = tuple(int(x) for x in np.__version__.split(".")[:2])
    if _nv >= (2, 0):
        print("Cảnh báo: NumPy 2.x. Nếu lỗi khi import senxor: pip3 install 'numpy<2'")
except Exception:
    pass

# --- Import phần cứng (RPi) ---
try:
    from smbus import SMBus
    from spidev import SpiDev
    from gpiozero import DigitalInputDevice, DigitalOutputDevice
    from senxor.mi48 import MI48, DATA_READY
    from senxor.utils import data_to_frame
    from senxor.interfaces import SPI_Interface, I2C_Interface
    from picamera2 import Picamera2
    _HAS_HARDWARE = True
except (ImportError, AttributeError) as e:
    err = str(e).lower()
    if "numpy" in err or "_array_api" in err or "multiarray" in err or "numpy.core" in err:
        print("NumPy 2.x không tương thích. Sửa: pip3 install 'numpy<2'")
        sys.exit(1)
    _HAS_HARDWARE = False
    SMBus = SpiDev = DigitalInputDevice = DigitalOutputDevice = None
    MI48 = DATA_READY = data_to_frame = SPI_Interface = I2C_Interface = Picamera2 = None


# ═══════════════════════════════════════════════════════════════════════════════
# THERMAL PIPELINE
# Trọng tâm: mượt (3DNR + BG model) + gradient nhiều mức nhiệt (không binary)
# ═══════════════════════════════════════════════════════════════════════════════

class ThermalTemporalFilter:
    """3DNR EMA: giảm nhiễu theo thời gian, rất rẻ trên 80x62."""
    def __init__(self, alpha=0.65):
        self.alpha = alpha
        self.ema = None

    def process(self, frame):
        f = frame.astype(np.float32)
        if self.ema is None:
            self.ema = f.copy()
        else:
            self.ema = self.alpha * f + (1 - self.alpha) * self.ema
        return np.clip(self.ema, 0, 255).astype(np.uint8)


class ThermalBackgroundModel:
    """Cold frame background: trung bình N frame warmup, cập nhật chậm.
    Trả về heat_map liên tục (float32, không binary) để hiện gradient."""
    def __init__(self, warmup_frames=40, adaptive_rate=0.005):
        self.warmup_frames = warmup_frames
        self.adaptive_rate = adaptive_rate
        self.cold_frame = None
        self._buf = []
        self.is_ready = False
        self.frame_count = 0

    def update(self, frame):
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

    def get_heat_map(self, frame, floor=3.0):
        """Trả về heat map liên tục (0–255). floor = bỏ nhiễu nhỏ hơn ngưỡng."""
        if not self.is_ready:
            return None
        diff = frame.astype(np.float32) - self.cold_frame
        diff = np.maximum(diff - floor, 0)
        mx = diff.max()
        if mx < 1:
            return np.zeros_like(frame, dtype=np.uint8)
        return np.clip(diff * (255.0 / mx), 0, 255).astype(np.uint8)

    def get_foreground_mask(self, frame, threshold=18.0, min_area=12, max_fg_ratio=0.5):
        """Mask nhị phân đã lọc nhiễu mạnh. Nếu foreground > max_fg_ratio thì coi như nhiễu → trả zeros."""
        if not self.is_ready:
            return None
        diff = frame.astype(np.float32) - self.cold_frame
        raw_mask = (diff > threshold).astype(np.uint8) * 255
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        cleaned = cv.morphologyEx(raw_mask, cv.MORPH_OPEN, kernel, iterations=2)
        cleaned = cv.morphologyEx(cleaned, cv.MORPH_CLOSE, kernel, iterations=1)
        fg_ratio = np.count_nonzero(cleaned) / cleaned.size
        if fg_ratio > max_fg_ratio:
            return np.zeros_like(cleaned)
        contours, _ = cv.findContours(cleaned, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        result = np.zeros_like(cleaned)
        for c in contours:
            if cv.contourArea(c) >= min_area:
                cv.drawContours(result, [c], -1, 255, -1)
        return result


def thermal_agc(frame, low_pct=2, high_pct=98):
    """AGC: percentile stretch ra 0–255."""
    f = frame.astype(np.float32)
    lo = np.percentile(f, low_pct)
    hi = np.percentile(f, high_pct)
    if hi - lo < 1:
        return frame
    return np.clip((f - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def thermal_edge_enhance(frame, strength=0.2):
    """Laplacian edge enhance cho thermal."""
    lap = cv.Laplacian(frame.astype(np.float32), cv.CV_32F, ksize=3)
    return np.clip(frame.astype(np.float32) + strength * np.abs(lap), 0, 255).astype(np.uint8)


class ThermalProcessor:
    """Pipeline thermal hoàn chỉnh: 3DNR → BG model → heat map gradient → AGC → EE."""
    def __init__(self, warmup=40, anti_glare=True):
        self.temporal = ThermalTemporalFilter(alpha=0.65)
        self.bg = ThermalBackgroundModel(warmup_frames=warmup, adaptive_rate=0.005)
        self.anti_glare_enabled = anti_glare

    @property
    def is_ready(self):
        return self.bg.is_ready

    @property
    def warmup_pct(self):
        return min(100, int(self.bg.frame_count / self.bg.warmup_frames * 100))

    def process(self, raw_frame):
        """Returns (denoised, enhanced, heat_map, fg_mask).
        denoised: chỉ 3DNR (mode 7 style, mượt, ít nhiễu nền).
        enhanced: 3DNR + AGC + EE (dùng fallback fusion).
        heat_map: gradient vùng nóng hơn nền (fusion).
        fg_mask: mask nhị phân cho Surveillance."""
        denoised = self.temporal.process(raw_frame)
        self.bg.update(denoised)
        d_for_detect = thermal_anti_glare(denoised) if self.anti_glare_enabled else denoised
        agc = thermal_agc(denoised)
        agc = thermal_anti_glare(agc) if self.anti_glare_enabled else agc
        enhanced = thermal_edge_enhance(agc, strength=0.2)
        heat_map = self.bg.get_heat_map(d_for_detect, floor=3.0)
        fg_mask = self.bg.get_foreground_mask(d_for_detect, threshold=18.0, min_area=12)
        return denoised, enhanced, heat_map, fg_mask


# ═══════════════════════════════════════════════════════════════════════════════
# NIR PIPELINE (tích hợp hybrid_night_vision + adaptive grayscale)
# Trọng tâm: Dark/Bright Channel + Atmosphere Light + CLAHE + detail + grayscale
# ═══════════════════════════════════════════════════════════════════════════════

def _min_filter(arr, ksize):
    """Min filter: dùng scipy nếu có, fallback cv.erode."""
    if _HAS_SCIPY_NDIMAGE:
        return minimum_filter(arr, size=(ksize, ksize), mode='reflect')
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (ksize, ksize))
    return cv.erode(arr.astype(np.float32), kernel)


def _max_filter(arr, ksize):
    """Max filter: dùng scipy nếu có, fallback cv.dilate."""
    if _HAS_SCIPY_NDIMAGE:
        return maximum_filter(arr, size=(ksize, ksize), mode='reflect')
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (ksize, ksize))
    return cv.dilate(arr.astype(np.float32), kernel)


class HybridNIREnhancer:
    """NIR pipeline: Dark/Bright Channel + Atmosphere Light + Adaptive CLAHE + detail.
    Khi tối (night mode): chuyển grayscale (SNR boost sqrt(3)) + brightness boost mạnh."""

    def __init__(self, proc_w=320, proc_h=240, patch_size=5, update_rate=8):
        self.proc_w = proc_w
        self.proc_h = proc_h
        self.patch = patch_size if patch_size % 2 == 1 else patch_size + 1
        self.update_rate = update_rate

        self.frame_count = 0
        self.last_A = np.array([0.7, 0.7, 0.7], dtype=np.float32)
        self.last_weight = None
        self.A_buffer = deque(maxlen=10)
        self.A_buffer.append(self.last_A)
        self.brightness_buffer = deque(maxlen=15)
        self.is_night_mode = False
        self._night_hyst = False
        self._night_hyst_ready = False

        self.clahe_levels = {
            'very_dark': cv.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)),
            'dark': cv.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6)),
            'medium': cv.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)),
        }
        self.detail_strength = 0.25
        self.min_boost = 1.1
        self.max_boost = 1.8
        self.p = 0.05

    def _compute_channels(self, I_float):
        min_ch = np.min(I_float, axis=2)
        max_ch = np.max(I_float, axis=2)
        dark = _min_filter(min_ch, self.patch)
        bright = _max_filter(max_ch, self.patch)
        return dark, bright

    def _estimate_atmosphere(self, I_float, bright_map):
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

    def _make_weight_map(self, dark_map, bright_map, A):
        weight = 1.0 - dark_map
        weight += 0.5 * np.abs(bright_map - np.mean(A))
        weight[bright_map > 0.7] *= 0.3
        weight = np.clip(weight, 0, 1)
        weight = cv.GaussianBlur(weight.astype(np.float32), (5, 5), 1.0)
        return weight

    def _apply_clahe_boost(self, bgr_small, weight_map, avg_brightness):
        lab = cv.cvtColor(bgr_small, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(lab)
        if avg_brightness < 0.25:
            clahe = self.clahe_levels['very_dark']
            base_boost = self.max_boost
        elif avg_brightness < 0.45:
            clahe = self.clahe_levels['dark']
            base_boost = (self.max_boost + self.min_boost) / 2
        else:
            clahe = self.clahe_levels['medium']
            base_boost = self.min_boost

        l_enh = clahe.apply(l).astype(np.float32) / 255.0
        if weight_map is not None:
            wm = cv.resize(weight_map, (l_enh.shape[1], l_enh.shape[0])) if weight_map.shape != l_enh.shape else weight_map
            l_enh = l_enh * (1.0 + (base_boost - 1.0) * wm)

        if self.detail_strength > 0:
            blur = cv.GaussianBlur(l_enh, (3, 3), 0.5)
            l_enh = l_enh + self.detail_strength * (l_enh - blur)

        l_out = np.clip(l_enh * 255, 0, 255).astype(np.uint8)
        return cv.cvtColor(cv.merge([l_out, a, b]), cv.COLOR_LAB2BGR)

    def _color_correct(self, bgr, A):
        """Giống hybrid_night_vision: 30% shift về atmosphere light."""
        f = bgr.astype(np.float32) / 255.0
        avg = np.mean(f, axis=(0, 1))
        f += (A - avg) * 0.3
        return np.clip(f * 255, 0, 255).astype(np.uint8)

    def process(self, frame):
        """Xử lý 1 frame NIR. Khi tối: grayscale (3ch merge tăng SNR) + brightness boost."""
        orig_h, orig_w = frame.shape[:2]
        small = cv.resize(frame, (self.proc_w, self.proc_h))

        self.frame_count += 1
        gray_small = cv.cvtColor(small, cv.COLOR_BGR2GRAY)
        cur_bright = float(np.mean(gray_small)) / 255.0
        self.brightness_buffer.append(cur_bright)
        # Schmitt quanh 0.45: tránh bật/tắt grayscale khi sáng lỡ cỡ
        _off, _on = 0.48, 0.40
        if not self._night_hyst_ready:
            self._night_hyst = cur_bright < 0.45
            self._night_hyst_ready = True
        elif self._night_hyst:
            if cur_bright > _off:
                self._night_hyst = False
        else:
            if cur_bright < _on:
                self._night_hyst = True
        self.is_night_mode = self._night_hyst

        if self.is_night_mode:
            small = cv.cvtColor(gray_small, cv.COLOR_GRAY2BGR)

        I_float = small.astype(np.float32) / 255.0

        need_update = (
            self.frame_count % self.update_rate == 0 or
            self.last_weight is None or
            len(self.brightness_buffer) < 5
        )

        if need_update:
            dark, bright = self._compute_channels(I_float)
            A = self._estimate_atmosphere(I_float, bright)
            weight = self._make_weight_map(dark, bright, A)
            self.last_A = A
            self.last_weight = weight
            self.A_buffer.append(A)
        else:
            weight = self.last_weight

        avg_b = float(np.mean(self.brightness_buffer)) if self.brightness_buffer else cur_bright
        enhanced = self._apply_clahe_boost(small, weight, avg_b)
        enhanced = self._color_correct(enhanced, self.last_A)

        if self.is_night_mode:
            lab = cv.cvtColor(enhanced, cv.COLOR_BGR2LAB)
            l, a, b = cv.split(lab)
            boost = 2.2 if cur_bright < 0.15 else (1.6 if cur_bright < 0.25 else 1.3)
            l = np.clip(l.astype(np.float32) * boost, 0, 255).astype(np.uint8)
            enhanced = cv.cvtColor(cv.merge([l, a, b]), cv.COLOR_LAB2BGR)

        return cv.resize(enhanced, (orig_w, orig_h))


# ═══════════════════════════════════════════════════════════════════════════════
# FUSION
# Gradient thermal overlay trên enhanced NIR (confidence-weighted)
# ═══════════════════════════════════════════════════════════════════════════════

class GradientThermalFusion:
    """Fusion: overlay thermal gradient (nhiều mức màu) lên NIR enhanced.
    heat_map (0-255 liên tục) → colormap → warp → blend với alpha theo cường độ nhiệt."""

    def __init__(self, base_alpha=0.6, colormap=cv.COLORMAP_JET):
        self.base_alpha = base_alpha
        self.colormap = colormap

    def fuse(self, nir_enhanced, heat_map, thermal_size, H, nir_w, nir_h, user_alpha=0.55):
        """Overlay gradient thermal lên NIR.
        heat_map: uint8 0-255 liên tục (0=nền, 255=nóng nhất).
        Vùng heat_map=0 hoàn toàn trong suốt, heat_map=255 alpha tối đa."""
        resized_heat = cv.resize(heat_map, thermal_size, interpolation=cv.INTER_LINEAR)
        resized_heat = cv.rotate(resized_heat, cv.ROTATE_180)

        colored = cv.applyColorMap(resized_heat, self.colormap)
        warped_color = cv.warpPerspective(colored, H, (nir_w, nir_h))

        heat_norm = resized_heat.astype(np.float32) / 255.0
        warped_alpha = cv.warpPerspective(heat_norm, H, (nir_w, nir_h))

        alpha_map = warped_alpha * self.base_alpha * user_alpha
        m3 = np.stack([alpha_map] * 3, axis=-1)

        result = nir_enhanced.astype(np.float32) * (1 - m3) + warped_color.astype(np.float32) * m3
        return np.clip(result, 0, 255).astype(np.uint8)

    def fuse_fallback(self, nir_enhanced, thermal_processed, thermal_size, H, nir_w, nir_h, user_alpha=0.55):
        """Fallback khi BG model chưa sẵn: dùng AGC thermal thay heat_map."""
        agc = thermal_agc(thermal_processed)
        return self.fuse(nir_enhanced, agc, thermal_size, H, nir_w, nir_h, user_alpha)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════════════════════

class FPSCounter:
    def __init__(self, window=30):
        self.times = deque(maxlen=window)
        self.fps = 0.0

    def tick(self):
        now = time.perf_counter()
        self.times.append(now)
        if len(self.times) >= 2:
            dt = self.times[-1] - self.times[0]
            self.fps = (len(self.times) - 1) / dt if dt > 0 else 0
        return self.fps


def bearing_hv_deg_from_uv(
    u: float,
    v: float,
    w: int,
    h: int,
    fov_h_deg: float,
    fov_v_deg: float,
) -> tuple:
    """Góc lệch (độ) từ tâm quang học tới pixel *(u,v)* — pinhole, FOV toàn khung (Tier S1 / A1).

    Đặt *(u,v)* ở **tâm** → góc ~(0,0)°; *(u,v)* lệch tâm → dH/dV = góc tới pixel đó (đo “điểm cần xem”).
    """
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    fh = math.radians(float(fov_h_deg)) * 0.5
    fv = math.radians(float(fov_v_deg)) * 0.5
    fx = cx / math.tan(fh) if fh > 1e-6 else float(w)
    fy = cy / math.tan(fv) if fv > 1e-6 else float(h)
    th = math.degrees(math.atan((u - cx) / fx))
    tv = math.degrees(math.atan((v - cy) / fy))
    return (float(th), float(tv))


def display_luminance_cap_bgr(frame_bgr: np.ndarray, l_max: int = 240) -> np.ndarray:
    """Trần độ sáng kênh L (LAB) — giảm chói màn Tier S4; không thay thế bảo vệ mắt quang học."""
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr
    lab = cv.cvtColor(frame_bgr, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    l2 = np.minimum(l.astype(np.int32), int(l_max)).astype(np.uint8)
    return cv.cvtColor(cv.merge([l2, a, b]), cv.COLOR_LAB2BGR)


class DisplayTemporalGlareBlend:
    """Tier S3: làm mượt khung hiển thị khi có highlight mạnh; reset khi Jerk."""

    def __init__(self, prev_weight: float = 0.42):
        self.prev_weight = float(prev_weight)
        self._prev: np.ndarray | None = None

    def process(self, frame_bgr: np.ndarray, apply_blend: bool, reset: bool) -> np.ndarray:
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


# ═══════════════════════════════════════════════════════════════════════════════
# HARDWARE CAPTURE THREADS
# ═══════════════════════════════════════════════════════════════════════════════

RPI_GPIO_I2C_CHANNEL = 1
RPI_GPIO_SPI_BUS = 0
RPI_GPIO_SPI_CE_MI48 = 0
MI48_I2C_ADDRESS = 0x40
MI48_SPI_MAX_SPEED_HZ = 7800000
MI48_SPI_CS_DELAY = 0.0001
SPI_XFER_SIZE_BYTES = 160
THERMAL_COLORMAP = cv.COLORMAP_JET


class ThermalCapture(threading.Thread):
    def __init__(self, flip_h=False):
        super().__init__(daemon=True)
        self.latest = None
        self.last_mono: float | None = None
        self.lock = threading.Lock()
        self.running = True
        self.mi48 = None
        self.mi48_spi_cs_n = None
        self.flip_h = flip_h

    def run(self):
        if not _HAS_HARDWARE:
            return
        i2c = I2C_Interface(SMBus(RPI_GPIO_I2C_CHANNEL), MI48_I2C_ADDRESS)
        spi = SPI_Interface(SpiDev(RPI_GPIO_SPI_BUS, RPI_GPIO_SPI_CE_MI48), xfer_size=SPI_XFER_SIZE_BYTES)
        spi.device.mode = 0
        spi.device.max_speed_hz = MI48_SPI_MAX_SPEED_HZ
        spi.device.bits_per_word = 8
        spi.device.lsbfirst = False
        spi.device.cshigh = True
        spi.device.no_cs = True
        self.mi48_spi_cs_n = DigitalOutputDevice("BCM7", active_high=False, initial_value=False)
        mi48_data_ready = DigitalInputDevice("BCM24", pull_up=False)
        mi48_reset_n = DigitalOutputDevice("BCM23", active_high=False, initial_value=True)

        class Reset:
            def __init__(s, pin):
                s.pin = pin
            def __call__(s):
                s.pin.on()
                time.sleep(0.000035)
                s.pin.off()
                time.sleep(0.05)

        self.mi48 = MI48([i2c, spi], data_ready=mi48_data_ready, reset_handler=Reset(mi48_reset_n))
        if int(self.mi48.fw_version[0]) >= 2:
            self.mi48.enable_filter(f1=True, f2=True, f3=False)
        self.mi48.set_fps(9)
        self.mi48.start(stream=True, with_header=True)

        while self.running:
            try:
                mi48_data_ready.wait_for_active()
                self.mi48_spi_cs_n.on()
                time.sleep(MI48_SPI_CS_DELAY)
                data, _ = self.mi48.read()
                time.sleep(MI48_SPI_CS_DELAY)
                self.mi48_spi_cs_n.off()
                if data is None:
                    continue
                img = data_to_frame(data, self.mi48.fpa_shape)
                img8u = cv.normalize(img.astype("uint8"), None, 255, 0, cv.NORM_MINMAX, dtype=cv.CV_8U)
                img8u = cv.GaussianBlur(img8u, (3, 3), 0)
                if self.flip_h:
                    img8u = cv.flip(img8u, 1)
                with self.lock:
                    self.latest = img8u.copy()
                    self.last_mono = time.monotonic()
            except Exception as e:
                if self.running:
                    print("Thermal error:", e)
        try:
            self.mi48.stop(stop_timeout=0.5)
        except Exception:
            pass

    def get_latest(self):
        with self.lock:
            return self.latest.copy() if self.latest is not None else None

    def get_last_mono(self) -> float | None:
        with self.lock:
            return self.last_mono

    def stop(self):
        self.running = False


class NIRCapture(threading.Thread):
    def __init__(self, no_rgb2bgr=True):
        super().__init__(daemon=True)
        self.latest = None
        self.last_mono: float | None = None
        self.lock = threading.Lock()
        self.running = True
        self.camera = None
        self.no_rgb2bgr = no_rgb2bgr

    def run(self):
        if not _HAS_HARDWARE:
            return
        self.camera = Picamera2()
        config = self.camera.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            controls={
                "FrameRate": 60,
                "ExposureTime": 40000,
                "AnalogueGain": 2.5,
                "AwbEnable": True,
                "AeEnable": True,
                "Brightness": 0.1,
                "Contrast": 1.1,
                "Saturation": 1.0,
            },
        )
        self.camera.configure(config)
        self.camera.start()
        while self.running:
            try:
                frame = self.camera.capture_array()
                frame = cv.flip(frame, 1)
                if not self.no_rgb2bgr:
                    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
                with self.lock:
                    self.latest = frame.copy()
                    self.last_mono = time.monotonic()
            except Exception as e:
                if self.running:
                    print("NIR error:", e)
        try:
            self.camera.stop()
        except Exception:
            pass

    def get_latest(self):
        with self.lock:
            return self.latest.copy() if self.latest is not None else None

    def get_last_mono(self) -> float | None:
        with self.lock:
            return self.last_mono

    def stop(self):
        self.running = False


# ═══════════════════════════════════════════════════════════════════════════════
# HOMOGRAPHY + Tier S (FOV / chống chói màn)
# ═══════════════════════════════════════════════════════════════════════════════
#
# FOV (°) cho HUD bearing (S1): ghi trong ``homography.json`` → ``meta["fov_deg"]`` = [ngang, dọc]
# theo ống kính/màn thật; thiếu thì dùng hằng dưới (stack tham chiếu ``docs/CONTEXT.md``:
# RPi + IMX290 640×480 + màn 800×480).
#
# Trần L (S4) và blend temporal loá (S3): cố định trong code, không qua CLI.

DEFAULT_DISPLAY_FOV_H_DEG = 52.0
DEFAULT_DISPLAY_FOV_V_DEG = 40.0
TIER_S_DISPLAY_L_MAX = 240
# Khi HUD [GLARE] (nhiệt/NIR rất sáng): hạ thêm trần L màn để dễ nhìn (S4 + phối hợp cảnh báo).
TIER_S_DISPLAY_L_MAX_WHEN_GLARE = 208
TIER_S_GLARE_TEMPORAL_PREV_WEIGHT = 0.42


def load_homography(path):
    with open(path) as f:
        data = json.load(f)
    H = np.array(data["homography"], dtype=np.float32)
    meta = data.get("meta", {})
    thermal_size = tuple(meta.get("thermal_size", [320, 248]))
    nir_size = tuple(meta.get("nir_size", [640, 480]))
    # Góc nhìn hiển thị (°); nên khớp phần cứng / datasheet khi hiệu chỉnh.
    fov = meta.get("fov_deg")
    if fov is not None and len(fov) >= 2:
        fov_h, fov_v = float(fov[0]), float(fov[1])
    else:
        fov_h, fov_v = None, None
    return H, thermal_size, nir_size, fov_h, fov_v


def ensure_fusion_capture_dirs() -> tuple[str, str]:
    """Trả về ``(save_dir, metrics_dir)`` — thư mục ghi được ảnh + JSON metrics.

    Tránh *Permission denied* (errno 13) khi cwd là ``/``, thư mục chỉ đọc, hoặc
    ``fusion_captures`` do user khác/sudo tạo: thử ``./fusion_captures``, không được
    thì ``~/fusion_captures``.
    """
    metrics_name = "metrics"
    candidates: List[str] = [os.path.join(os.getcwd(), "fusion_captures")]
    home = os.path.expanduser("~")
    if home and os.path.abspath(home) != os.path.abspath(os.getcwd()):
        candidates.append(os.path.join(home, "fusion_captures"))

    last_err: OSError | None = None
    for base in candidates:
        md = os.path.join(base, metrics_name)
        try:
            os.makedirs(md, exist_ok=True)
            probe = os.path.join(md, ".probe_write")
            with open(probe, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(probe)
            if base != candidates[0]:
                print(f"[fusion] Lưu fusion_captures tại (cwd không ghi được): {base}")
            return base, md
        except OSError as e:
            last_err = e
            continue

    raise PermissionError(
        "Không tạo/ghi được fusion_captures (đã thử cwd và ~). "
        f"Kiểm tra quyền thư mục hoặc chạy từ Desktop/home. Lỗi: {last_err}"
    ) from last_err


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN (3 MODE: IMX, THERMAL, FUSION)
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    p = argparse.ArgumentParser(description="fusion_live_unified — Thermal-NIR + toàn cục (1 file)")
    p.add_argument("-H", "--homography", default="homography.json")
    p.add_argument("-a", "--alpha", type=float, default=0.55)
    p.add_argument("-w", "--width", type=int, default=800)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--thermal-flip", action="store_true")
    p.add_argument("--nir-rgb2bgr", action="store_true")
    args = p.parse_args()

    if not _HAS_HARDWARE:
        print("Thiếu phần cứng. Chỉ chạy trên RPi + senxor + picamera2.")
        sys.exit(1)

    if not os.path.exists(args.homography):
        print(f"Không tìm thấy {args.homography}. Chạy calibrate trước.")
        sys.exit(1)

    H, thermal_size, nir_size, fov_h_meta, fov_v_meta = load_homography(args.homography)
    nir_w, nir_h = nir_size
    display_size = (args.width, args.height)
    alpha = args.alpha

    fov_h = float(fov_h_meta) if fov_h_meta is not None else DEFAULT_DISPLAY_FOV_H_DEG
    fov_v = float(fov_v_meta) if fov_v_meta is not None else DEFAULT_DISPLAY_FOV_V_DEG
    display_l_max = TIER_S_DISPLAY_L_MAX
    temporal_glare = DisplayTemporalGlareBlend(prev_weight=TIER_S_GLARE_TEMPORAL_PREV_WEIGHT)

    thermal_proc = ThermalProcessor(warmup=40, anti_glare=True)
    nir_enhancer = HybridNIREnhancer(proc_w=320, proc_h=240, patch_size=5, update_rate=8)
    fusion = GradientThermalFusion(base_alpha=0.6, colormap=THERMAL_COLORMAP)
    fps_counter = FPSCounter()
    jerk_gate = JerkGate(diff_threshold=8.5, hold_frames=12, consecutive_frames=2, percentile=94.0, max_side=128)
    opsec = OpsecLocalOnly(enabled=False)
    shake = DisplayShakeReducerLite(mode="blend", blend_current_weight=0.50)

    # Camera threads
    thermal_cap = ThermalCapture(flip_h=args.thermal_flip)
    nir_cap = NIRCapture(no_rgb2bgr=not args.nir_rgb2bgr)

    def on_signal(sig, frame):
        thermal_cap.stop()
        nir_cap.stop()
        sys.exit(0)
    sig_module.signal(sig_module.SIGINT, on_signal)
    sig_module.signal(sig_module.SIGTERM, on_signal)

    print("Đang khởi động camera...")
    thermal_cap.start()
    nir_cap.start()
    time.sleep(2)

    save_dir, metrics_dir = ensure_fusion_capture_dirs()
    session_id = time.strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:10]
    metrics_write_run_manifest(
        os.path.join(metrics_dir, f"manifest_{session_id}.json"),
        session_id=session_id,
        homography_path=os.path.abspath(args.homography),
        display_size=(display_size[0], display_size[1]),
        fov_deg=(fov_h, fov_v),
        tier_s_flags={
            "S1_bearing_hud": True,
            "S2_run_manifest": True,
            "S3_temporal_glare": True,
            "S4_display_luminance_cap": True,
            "S6_stream_timestamp_skew": True,
            "S5_battery": False,
        },
    )
    cv.namedWindow("SmartBinocular", cv.WINDOW_NORMAL)
    cv.resizeWindow("SmartBinocular", *display_size)

    # S1: mốc góc mặc định = tâm màn; **click trái** = chọn pixel đo dH/dV (OpenCV trên Pi ổn với click hơn move).
    a1_probe_xy = [display_size[0] // 2, display_size[1] // 2]

    def _on_mouse_a1(event, x, y, _flags, _param):
        if event == cv.EVENT_LBUTTONDOWN:
            a1_probe_xy[0] = x
            a1_probe_xy[1] = y

    cv.setMouseCallback("SmartBinocular", _on_mouse_a1)

    # Raw ↔ enhance (Schmitt + EMA): tránh nhấp nháy khi độ sáng lỡ cỡ (mean gray 0–255)
    # Đủ sáng: auto hiển thị raw (không cần phím R). Vùng giữa 18–30: giữ trạng thái (Schmitt).
    NIR_BRIGHT_RAW_ON = 30.0   # mean mượt >= này → raw / tắt pipeline enhance
    NIR_DIM_ENHANCE_ON = 18.0  # mean mượt <= này → enhance + grayscale trong pipeline
    NIR_B_EMA = 0.18           # làm mượt đo sáng trước khi so Schmitt
    SWITCH_FRAMES = 28

    thesis_metrics = ThesisRunMetrics(
        session_id=session_id,
        homography_path=os.path.abspath(args.homography),
        display_size=(display_size[0], display_size[1]),
        fusion_alpha=alpha,
        nir_schmitt_raw_on=NIR_BRIGHT_RAW_ON,
        nir_schmitt_dim_on=NIR_DIM_ENHANCE_ON,
        nir_b_ema_coef=NIR_B_EMA,
    )

    mode = "imx"
    nir_b_ema = None
    nir_hyst_initialized = False
    nir_use_raw_auto = True
    show_raw = {"imx": False, "thermal": False}
    mode_switch_pending = None
    switch_frames_left = 0
    auto_start = None
    AUTO_DELAY = 5
    thermal_denoised = thermal_enhanced = heat_map = fg_mask = nir_enhanced = None

    print("=" * 60)
    print("  1 = IMX (NIR)   2 = Thermal (3DNR)   3 = Fusion (NIR+Thermal)")
    print("  NIR: raw khi sáng (Schmitt), grayscale+enhance khi tối")
    print("  R = raw/processed   S = Save   A = Auto   +/- = trọng số pha trộn (Fusion)   Q = Thoát")
    print("  (C4 + A6b blend + D2 tự động; HUD [JERK]/[GLARE] khi có)")
    print(f"  Metrics → JSON cạnh ảnh; báo cáo phiên: {metrics_dir}/session_*.json (khi Q)")
    print(f"  Tier S: A1 bearing @tâm (FOV≈{fov_h:.1f}°×{fov_v:.1f}°) | temporal loá | trần L≤{display_l_max} | S6 skew khi đủ 2 luồng")
    print(f"  Manifest: {metrics_dir}/manifest_{session_id}.json")
    print("=" * 60)

    while True:
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("1") and mode != "imx":
            mode_switch_pending = "imx"
            switch_frames_left = SWITCH_FRAMES
        elif key == ord("2") and mode != "thermal":
            mode_switch_pending = "thermal"
            switch_frames_left = SWITCH_FRAMES
        elif key == ord("3") and mode != "fusion":
            mode_switch_pending = "fusion"
            switch_frames_left = SWITCH_FRAMES
        elif key in (ord("r"), ord("R")) and mode in show_raw:
            show_raw[mode] = not show_raw[mode]
            print(f"{mode}: {'raw' if show_raw[mode] else 'xử lý'}")
        elif key == ord("a"):
            auto_start = time.time()
        elif key == ord("+") or key == ord("="):
            alpha = min(alpha + 0.05, 1.0)
            print(f"Fusion blend α={alpha:.2f} (pha trộn lớp nhiệt lên NIR)")
        elif key == ord("-"):
            alpha = max(alpha - 0.05, 0.05)
            print(f"Fusion blend α={alpha:.2f} (pha trộn lớp nhiệt lên NIR)")
        if mode_switch_pending is not None:
            nir_raw = nir_cap.get_latest()
            thermal_raw = thermal_cap.get_latest()
            if (mode_switch_pending == "imx" and nir_raw is None) or (mode_switch_pending == "thermal" and thermal_raw is None) or (mode_switch_pending == "fusion" and (nir_raw is None or thermal_raw is None)):
                time.sleep(0.02)
                continue
            t = mode_switch_pending
            if t == "imx" and nir_raw is not None:
                nir_enhanced = nir_enhancer.process(nir_raw)
            elif t == "thermal" and thermal_raw is not None:
                thermal_denoised, thermal_enhanced, heat_map, fg_mask = thermal_proc.process(thermal_raw)
            elif t == "fusion" and nir_raw is not None and thermal_raw is not None:
                thermal_denoised, thermal_enhanced, heat_map, fg_mask = thermal_proc.process(thermal_raw)
            switch_frames_left -= 1
            msg = {"imx": "Preparing NIR...", "thermal": "Preparing Thermal...", "fusion": "Preparing Fusion..."}.get(t, "")
            out = np.zeros((display_size[1], display_size[0], 3), dtype=np.uint8)
            out[:] = (40, 40, 40)
            cv.putText(out, msg, (display_size[0] // 2 - 100, display_size[1] // 2 - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1)
            cv.putText(out, f"{switch_frames_left} frames", (display_size[0] // 2 - 40, display_size[1] // 2 + 20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv.imshow("SmartBinocular", out)
            if switch_frames_left <= 0:
                entering = mode_switch_pending
                if entering in ("imx", "fusion") and mode not in ("imx", "fusion"):
                    nir_b_ema = None
                    nir_hyst_initialized = False
                mode = entering
                mode_switch_pending = None
                shake.reset()
                jerk_gate.reset()
            fps_counter.tick()
            continue

        nir_raw = nir_cap.get_latest()
        thermal_raw = thermal_cap.get_latest()
        glare_nir = glare_th = False
        if mode == "imx" and nir_raw is None:
            time.sleep(0.02)
            continue
        if mode == "thermal" and thermal_raw is None:
            time.sleep(0.02)
            continue
        if mode == "fusion" and (nir_raw is None or thermal_raw is None):
            time.sleep(0.02)
            continue

        nir_brightness = None
        nir_need_compress = False

        if mode == "imx":
            thermal_denoised = thermal_enhanced = heat_map = fg_mask = None
        elif mode == "thermal":
            nir_enhanced = None

        if mode in ("imx", "fusion"):
            nir_brightness = float(np.mean(cv.cvtColor(nir_raw, cv.COLOR_BGR2GRAY)))
            if nir_b_ema is None:
                nir_b_ema = nir_brightness
            else:
                nir_b_ema = (1.0 - NIR_B_EMA) * nir_b_ema + NIR_B_EMA * nir_brightness
            b = nir_b_ema
            if not nir_hyst_initialized:
                nir_use_raw_auto = b >= 0.5 * (NIR_BRIGHT_RAW_ON + NIR_DIM_ENHANCE_ON)
                nir_hyst_initialized = True
            elif nir_use_raw_auto:
                if b <= NIR_DIM_ENHANCE_ON:
                    nir_use_raw_auto = False
            else:
                if b >= NIR_BRIGHT_RAW_ON:
                    nir_use_raw_auto = True
            nir_enhanced = None if nir_use_raw_auto else nir_enhancer.process(nir_raw)

        if mode == "thermal":
            thermal_denoised, thermal_enhanced, heat_map, fg_mask = thermal_proc.process(thermal_raw)
        elif mode == "fusion":
            thermal_denoised, thermal_enhanced, heat_map, fg_mask = thermal_proc.process(thermal_raw)

        jerk_gate.update(nir_raw)

        fps = fps_counter.tick()

        if mode == "imx":
            use_raw = show_raw.get("imx") or nir_use_raw_auto
            if use_raw:
                src = nir_raw
            else:
                src = nir_enhanced if nir_enhanced is not None else nir_raw
            nir_ok_glare = nir_glare_allowed(
                nir_brightness, NIR_BRIGHT_RAW_ON, nir_enhancer.is_night_mode, display_raw=use_raw
            )
            src_in = src
            nir_need_compress = bool(nir_ok_glare and nir_highlight_need_compress(src_in))
            glare_nir = False
            if thermal_proc.anti_glare_enabled and nir_ok_glare:
                if nir_need_compress:
                    src = nir_anti_glare_bgr(src_in)
                glare_nir = nir_glare_hud_should_show(src_in)
            out = cv.resize(src, display_size)
        elif mode == "thermal":
            # Mode 7 style: hiển thị denoised (3DNR) → mượt, ít nhiễu nền
            if show_raw.get("thermal"):
                cm = cv.applyColorMap(thermal_raw, THERMAL_COLORMAP)
            elif thermal_denoised is not None:
                cm = cv.applyColorMap(thermal_denoised, THERMAL_COLORMAP)
            else:
                cm = cv.applyColorMap(thermal_enhanced if thermal_enhanced is not None else thermal_raw, THERMAL_COLORMAP)
            cm = cv.resize(cm, thermal_size)
            cm = cv.rotate(cm, cv.ROTATE_180)
            out = cv.resize(cm, display_size)
            if not thermal_proc.is_ready:
                cv.putText(out, f"Warming up... {thermal_proc.warmup_pct}%",
                           (10, 48), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            if thermal_proc.anti_glare_enabled and thermal_denoised is not None:
                glare_th = thermal_glare_hud_should_apply(thermal_denoised)

        else:  # fusion (NIR bg + heat map overlay chỉ ở vùng có vật nóng)
            use_raw = show_raw.get("imx") or nir_use_raw_auto
            nir_src = nir_raw if use_raw else (nir_enhanced if nir_enhanced is not None else nir_raw)
            nir_ok_glare = nir_glare_allowed(
                nir_brightness, NIR_BRIGHT_RAW_ON, nir_enhancer.is_night_mode, display_raw=use_raw
            )
            nir_in = nir_src
            nir_need_compress = bool(nir_ok_glare and nir_highlight_need_compress(nir_in))
            glare_nir = False
            if thermal_proc.anti_glare_enabled and nir_ok_glare:
                if nir_need_compress:
                    nir_src = nir_anti_glare_bgr(nir_in)
                glare_nir = nir_glare_hud_should_show(nir_in)
            if thermal_proc.anti_glare_enabled and thermal_denoised is not None:
                glare_th = thermal_glare_hud_should_apply(thermal_denoised)
            nir_resized = cv.resize(nir_src, (nir_w, nir_h))
            if heat_map is not None and fg_mask is not None and np.any(fg_mask > 0):
                hm_r = cv.resize(heat_map, thermal_size, interpolation=cv.INTER_LINEAR)
                hm_r = cv.rotate(hm_r, cv.ROTATE_180)
                hm_color = cv.applyColorMap(hm_r, THERMAL_COLORMAP)
                hm_warped = cv.warpPerspective(hm_color, H, (nir_w, nir_h))
                fg_r = cv.resize(fg_mask, thermal_size)
                fg_r = cv.rotate(fg_r, cv.ROTATE_180)
                fg_w = cv.warpPerspective(fg_r, H, (nir_w, nir_h), flags=cv.INTER_NEAREST)
                mask_f = fg_w.astype(np.float32) / 255.0
                mask_f = cv.GaussianBlur(mask_f, (7, 7), 2.0)
                m3 = np.stack([mask_f] * 3, axis=-1) * alpha
                blended = nir_resized.astype(np.float32) * (1 - m3) + hm_warped.astype(np.float32) * m3
                out = cv.resize(np.clip(blended, 0, 255).astype(np.uint8), display_size)
            else:
                out = cv.resize(nir_resized, display_size)
            if not thermal_proc.is_ready:
                cv.putText(out, f"BG warming... {thermal_proc.warmup_pct}%",
                           (10, 48), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if mode in ("imx", "fusion"):
            out = temporal_glare.process(out, nir_need_compress and not jerk_gate.active, reset=jerk_gate.active)
        out = shake.process(out, jerk_active=jerk_gate.active)
        _l_cap = int(display_l_max)
        if glare_nir or glare_th:
            _l_cap = min(_l_cap, TIER_S_DISPLAY_L_MAX_WHEN_GLARE)
        out = display_luminance_cap_bgr(out, l_max=_l_cap)

        skew_ms = None
        _tm = thermal_cap.get_last_mono()
        _nm = nir_cap.get_last_mono()
        # S6: chênh lệch thời gian lấy mẫu NIR vs thermal (ms) — đánh giá đồng bộ hai luồng (fusion/độ trễ cảm nhận).
        if _tm is not None and _nm is not None:
            skew_ms = abs(_tm - _nm) * 1000.0
        _wd, _hd = int(display_size[0]), int(display_size[1])
        _u = float(np.clip(a1_probe_xy[0], 0, max(0, _wd - 1)))
        _v = float(np.clip(a1_probe_xy[1], 0, max(0, _hd - 1)))
        bear_h, bear_v = bearing_hv_deg_from_uv(_u, _v, _wd, _hd, fov_h, fov_v)

        thesis_metrics.fusion_alpha = alpha
        thesis_metrics.record_frame(
            mode=mode,
            fps=fps,
            nir_brightness=nir_brightness,
            is_night_mode=nir_enhancer.is_night_mode if mode in ("imx", "fusion") else False,
            jerk_active=jerk_gate.active,
            glare_nir=glare_nir,
            glare_th=glare_th,
            thermal_is_ready=thermal_proc.is_ready if mode in ("thermal", "fusion") else True,
            stream_skew_ms=skew_ms,
        )

        def _write_capture_meta(path_png: str, trigger: str) -> None:
            if mode == "imx":
                srt = bool(show_raw.get("imx"))
            elif mode == "fusion":
                srt = bool(show_raw.get("imx"))
            else:
                srt = bool(show_raw.get("thermal"))
            tag_list = []
            if jerk_gate.active:
                tag_list.append("JERK")
            if glare_nir or glare_th:
                tag_list.append("GLARE")
            meta = thesis_metrics.build_capture_meta(
                image_filename=os.path.basename(path_png),
                trigger=trigger,
                mode=mode,
                fps=fps,
                alpha=alpha,
                nir_brightness=nir_brightness,
                nir_b_ema=nir_b_ema if mode in ("imx", "fusion") else None,
                nir_use_raw_auto=nir_use_raw_auto if mode in ("imx", "fusion") else None,
                show_raw_toggle=srt,
                is_night_mode=nir_enhancer.is_night_mode if mode in ("imx", "fusion") else False,
                jerk_active=jerk_gate.active,
                glare_nir=glare_nir,
                glare_th=glare_th,
                tags_hud=tag_list,
                bearing_h_deg=bear_h,
                bearing_v_deg=bear_v,
                stream_skew_ms=skew_ms,
            )
            _metrics_write_json(path_png.replace(".png", ".json"), meta)

        # ── Auto capture ──
        if auto_start is not None:
            elapsed = time.time() - auto_start
            countdown = max(0, AUTO_DELAY - int(elapsed))
            if countdown > 0:
                cv.putText(out, f"Chụp sau {countdown}s...", (display_size[0] // 2 - 80, 48),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            else:
                ts = time.strftime("%Y%m%d-%H%M%S")
                path = os.path.join(save_dir, f"auto_{mode}_{ts}.png")
                cv.imwrite(path, out)
                _write_capture_meta(path, "timer_auto")
                print("Đã lưu:", path, "+ meta JSON")
                auto_start = None

        if key == ord("s"):
            ts = time.strftime("%Y%m%d-%H%M%S")
            path = os.path.join(save_dir, f"{mode}_{ts}.png")
            cv.imwrite(path, out)
            _write_capture_meta(path, "key_s")
            print("Đã lưu:", path, "+ meta JSON")

        # ── HUD (800x480 5-inch) ──
        labels = {"imx": "IMX (NIR)", "thermal": "Thermal (3DNR)", "fusion": "Fusion"}
        raw_tag = " [RAW]" if show_raw.get(mode) else ""
        cv.putText(out, f"{labels[mode]}{raw_tag}", (8, 22), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        cv.putText(out, f"FPS:{fps:.0f} a:{alpha:.2f}", (display_size[0] - 155, 22),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        if mode == "imx" and not show_raw.get("imx") and nir_enhancer.is_night_mode:
            cv.putText(out, "[NIGHT]", (8, 44), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

        hud_y = 64 if (mode == "imx" and not show_raw.get("imx") and nir_enhancer.is_night_mode) else 44
        tag_parts = []
        if jerk_gate.active:
            tag_parts.append("JERK")
        if glare_nir or glare_th:
            tag_parts.append("GLARE")
        if tag_parts:
            cv.putText(out, "[" + " ".join(tag_parts) + "]", (8, hud_y), cv.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1)

        _hh, _ww = out.shape[:2]
        _cx, _cy = _ww // 2, _hh // 2
        cv.line(out, (_cx - 18, _cy), (_cx + 18, _cy), (0, 220, 255), 1)
        cv.line(out, (_cx, _cy - 18), (_cx, _cy + 18), (0, 220, 255), 1)
        _pu, _pv = int(np.clip(a1_probe_xy[0], 0, _ww - 1)), int(np.clip(a1_probe_xy[1], 0, _hh - 1))
        cv.circle(out, (_pu, _pv), 5, (0, 255, 255), 1)
        _bh = f"{bear_h:+.2f}"
        _bv = f"{bear_v:+.2f}"
        if skew_ms is not None:
            _sk_disp = f"{skew_ms:.2f} ms"
        else:
            _sk_disp = "n/a"
        _hud_font = cv.FONT_HERSHEY_SIMPLEX
        _hud_fs = 0.42
        _hud_th = 1

        def _put_right(label: str, y_px: int, bgr: tuple) -> None:
            (tw, _), _bl = cv.getTextSize(label, _hud_font, _hud_fs, _hud_th)
            x_px = max(8, _ww - tw - 10)
            cv.putText(out, label, (x_px, y_px), _hud_font, _hud_fs, bgr, _hud_th, cv.LINE_AA)

        _put_right(f"FPS:{fps:.0f}", 22, (0, 255, 0))
        _put_right(f"A1 dH={_bh} dV={_bv}", 44, (200, 230, 255))
        _put_right(f"S6 skew {_sk_disp}", 66, (160, 255, 200))

        cv.imshow("SmartBinocular", out)

    thermal_cap.stop()
    nir_cap.stop()
    cv.destroyAllWindows()
    thesis_metrics.fusion_alpha = alpha
    rep = thesis_metrics.finalize()
    _ts = time.strftime("%Y%m%d-%H%M%S")
    _session_path = os.path.join(metrics_dir, f"session_{_ts}.json")
    _metrics_write_json(_session_path, rep)
    print(thesis_metrics.summary_line(rep))
    print("Báo cáo phiên (metrics):", _session_path)
    print("Thoát.")


if __name__ == "__main__":
    main()
