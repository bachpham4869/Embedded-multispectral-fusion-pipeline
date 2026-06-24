#!/usr/bin/env python3
"""
fusion_live_optimized.py — BẢN THỬ NGHIỆM AN TOÀN TỪ unified
==============================================================
Bản **độc lập**: gồm toàn bộ pipeline fusion (giống ``fusion_live_standalone.py``)
**và** khối thuật toán **toàn cục** (D1/D2/C4) + **metrics JSON (luận văn)** **nhúng trong file** —
**không** import ``fusion_live_standalone``, ``fusion_advanced``, ``live_global_services``, hay module metrics riêng.

Mục đích:
  • Một chỗ để phát triển detection/alert nhẹ, IMU/GNSS tương lai… mà không phải nhảy nhiều file.
  • ``fusion_live_standalone.py`` + ``live_global_services.py`` vẫn giữ để tách module / tái sử dụng;
    khi ổn định có thể đồng bộ ngược patch sang các file đó.

Tối ưu (CONFIG, không cần benchmark để bật): ``opencv_num_threads``, ``nir_capture_fps``,
``nir_brightness_subsample``, ``skip_thermal_reprocess_when_same_timestamp`` — xem docstring
``_configure_opencv_runtime`` / ``_nir_mean_brightness_bgr`` / khối ThermalProcessor trong ``main``.

Cấu trúc:
  SECTION 0 — Toàn cục (D1, D2, C4) + placeholder mở rộng
  SECTION 1 — Thermal → NIR → Fusion → capture → main

Phím: 1/2/3, R, S, A, +/-, Q. **+/-** chỉnh **alpha** pha trộn NIR/thermal ở mode Fusion (0.05–1.0). Toàn cục §1b tự chạy (C4, A6b blend, D2).

GLARE (HUD): chỉ khi đuôi histogram **rất** sáng (ít nhạy). Ảnh vẫn có thể được **nén highlight** sớm hơn qua
``nir_highlight_need_compress`` / thermal tương ứng (gamma + roll-off NIR).

Lưu ý an toàn:
  • Không sửa file stable ``fusion_live_unified.py``.
  • Các tối ưu đều có fallback/config để giữ hành vi gần bản stable khi cần đối chiếu.
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
from typing import Any, Dict, List, Optional, Set, Tuple

import math
import platform
import numpy as np
import cv2 as cv

# ── Metrics phiên / manifest (JSON) — nhúng trong file, không import module ngoài ─

_METRICS_SCHEMA_VERSION = "1.2-opt"


def _opt_log(enabled: bool, msg: str) -> None:
    if enabled:
        print(f"[opt] {msg}")


def _configure_opencv_runtime(cfg: Dict[str, Any]) -> None:
    """Tối ưu runtime OpenCV; chi tiết và bối cảnh: ``docs/PERFORMANCE_FUSION_LIVE.md``.

    Trước đây: không gọi ``setNumThreads`` — số thread nội bộ phụ thuộc build OpenCV.
    Giờ: ``opencv_num_threads > 0`` → ``cv.setNumThreads(n)`` để các op TBB/OpenMP
    (resize, filter, v.v.) dùng rõ ràng tới N lõi trên Pi 4.

    ``opencv_disable_opencl`` (mặc định True): tắt OpenCL — VideoCore không có
    OpenCL compute chuẩn cho cv; tránh overhead kiểm tra thiết bị mỗi lần gọi.
    """
    n = int(cfg.get("opencv_num_threads", 4))
    if n > 0:
        cv.setNumThreads(n)
    if bool(cfg.get("opencv_disable_opencl", True)):
        try:
            cv.ocl.setUseOpenCL(False)
        except Exception:
            pass


def _nir_mean_brightness_bgr(nir_bgr: np.ndarray, subsample: int) -> float:
    """Ước lượng độ sáng trung bình NIR cho Schmitt / HUD (mục 4.C + 4.E).

    Cũ: ``mean(gray)`` trên toàn bộ 640×480 mỗi frame.

    Mới: ``nir_brightness_subsample`` = k ≥ 2 → resize về ~(w/k)×(h/k) bằng
    ``INTER_AREA`` rồi mới mean — giảm ~k² phép trên pixel, gần như không đổi
    thống kê cho Schmitt (EMA vẫn làm mượt).

    k = 1: giữ hành vi cũ (mean trên full-resolution gray).
    """
    k = int(subsample)
    if k <= 1:
        return float(np.mean(cv.cvtColor(nir_bgr, cv.COLOR_BGR2GRAY)))
    h, w = nir_bgr.shape[:2]
    tw = max(1, w // k)
    th = max(1, h // k)
    small = cv.resize(nir_bgr, (tw, th), interpolation=cv.INTER_AREA)
    return float(np.mean(cv.cvtColor(small, cv.COLOR_BGR2GRAY)))


# ===== DEFAULT OPTIMIZED CONFIG =====
# Chạy trực tiếp: python3 fusion_live_optimized.py
# Không cần truyền CLI argument.
#
# Nhóm ảnh hưởng ghosting NIR:
# - fix_nir_motion_gating
# - nir_soft_motion_ratio
# - opt_profile (temporal_prev_weight trong profile)
#
# Nhóm ảnh hưởng blur thermal/mất detect:
# - thermal_capture_blur_ksize
# - fix_thermal_detail_preserve
# - thermal_detect_raw_mix
# - thermal_detail_grad_threshold
# - thermal_detail_threshold_relax
# - thermal_display_raw_mix
# - thermal_detect_use_anti_glare
# - thermal_display_agc_low_pct / thermal_display_agc_high_pct
# - thermal_heatmap_agc_low_pct / thermal_heatmap_agc_high_pct
CONFIG: Dict[str, Any] = {
    "homography_path": "homography.json",
    "display_width": 800,
    "display_height": 480,
    "fusion_alpha": 0.55,
    "thermal_flip": False,
    "nir_rgb2bgr": False,
    "debug": False,
    "opt_profile": "handheld_pan",   # baseline | static_scan | handheld_pan | high_glare
    "opt_disable_profiles": False,
    "opt_haze_preset": False,
    "opt_disable_s6_quality_gate": False,
    "fix_nir_motion_gating": True,
    "nir_soft_motion_ratio": 0.62,
    "fix_thermal_detail_preserve": True,
    "thermal_detect_raw_mix": 0.55,
    "thermal_detail_grad_threshold": 9.0,
    "thermal_detail_threshold_relax": 3.0,
    "thermal_capture_blur_ksize": 1,  # 1 = tắt Gaussian blur ở capture thermal
    "thermal_display_raw_mix": 0.28,  # tăng chi tiết thermal bằng cách pha nhẹ raw vào 3DNR
    "thermal_detect_use_anti_glare": False,  # mặc định tắt; ENV preset glare_heavy có thể bật
    "nir_enhancer_detail_strength": 0.25,
    "nir_enhancer_clahe_clip_scale": 1.0,  # nhân clipLimit CLAHE (very_dark/dark/medium), không thêm chi phí
    "thermal_display_agc_low_pct": 1.5,  # tăng tương phản thermal hiển thị
    "thermal_display_agc_high_pct": 99.2,
    "thermal_heatmap_agc_low_pct": 1.0,  # tăng dynamic range heatmap overlay
    "thermal_heatmap_agc_high_pct": 99.5,
    "perf_glare_lite": True,
    "perf_jerk_lite": True,
    "perf_report_interval": 180,
    "perf_benchmark_samples": 24,
    # --- Mục 4: tối ưu có thể bật ngay (không phụ thuộc benchmark trước/sau) ---
    # OpenCV: đa luồng nội bộ cho một số hàm; 0 = không gọi setNumThreads (để mặc định build).
    "opencv_num_threads": 4,
    "opencv_disable_opencl": True,
    # NIR (Picamera2 / libcamera): giảm FPS yêu cầu → ít tải bus/ISP/copy khi không cần 60 FPS.
    "nir_capture_fps": 60.0,
    # Mean độ sáng cho Schmitt: 1 = full-res (hành vi cũ); 4 = ~1/4 kích thước mỗi chiều (mặc định mới).
    "nir_brightness_subsample": 4,
    # Khi NIR ~60 FPS mà thermal ~9 FPS: không gọi lại ThermalProcessor trên cùng mẫu nhiệt
    # (tránh lặp 3DNR/BG trên một frame; đúng với tốc độ cảm biến). False = luôn xử lý như bản cũ.
    "skip_thermal_reprocess_when_same_timestamp": True,
    # Mode 1/2/3: tạm dừng đọc phần cứng ở thread không dùng → giảm bus/SPI/ISP tranh CPU với luồng chính.
    # True = mode IMX (1) không đọc MI48. NIR khi chỉ thermal (2): mặc định False (vẫn cần cho ENV auto_rule + S6).
    "pause_thermal_capture_when_imx_only": True,
    "pause_nir_capture_when_thermal_only": False,
    # Phase-1 features (detection-only)
    "feature_e1_enable": True,
    "feature_e1_detect_interval": 2,
    "feature_e1_local_kernel": 5,
    "feature_e1_z_thresh": 1.25,
    "feature_e1_heat_thresh": 46.0,
    "feature_e1_min_area": 10,
    "feature_e1_raw_mix": 0.45,
    "feature_alert_enable": False,  # alert theo detection frame-level (không tracking)
    "feature_sector_center_deg": 6.0,
    "feature_sector_hold_frames": 8,
    # [ENV] Adaptive environment presets (rule-based; ML có thể gắn sau)
    "env_mode": "off",  # off | manual | auto_rule
    "env_manual_preset": "default",
    "env_fallback_preset": "default",
    "env_hysteresis_frames": 18,
    "env_auto_nir_gray_std_low": 20.0,   # std thấp → cảnh phẳng / sương mù nhẹ
    "env_auto_nir_gray_std_high": 52.0,  # std cao → nền phức tạp
    "nir_schmitt_raw_on": 30.0,
    "nir_schmitt_dim_on": 18.0,
    "thermal_3dnr_alpha": 0.65,
    "experiment": {
        "scenario_type": "unspecified",
        "environment": "unspecified",
        "lighting_level": "unspecified",
        "operator_id": "",
        "build_label": "fusion_live_optimized",
        "notes_short": "",
    },
}


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
    experiment_context: Dict[str, Any] = field(default_factory=dict)
    optimization_config: Dict[str, Any] = field(default_factory=dict)

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
    skew_quality_counts: Dict[str, int] = field(default_factory=lambda: {"GOOD": 0, "DEGRADED": 0, "BAD": 0})

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
        skew_quality: Optional[str] = None,
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
        if skew_quality in self.skew_quality_counts:
            self.skew_quality_counts[skew_quality] += 1

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
        skew_quality: Optional[str] = None,
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
            "experiment_context": self.experiment_context,
            "optimization_config": self.optimization_config,
        }
        if bearing_h_deg is not None:
            d["bearing_offset_h_deg"] = round(float(bearing_h_deg), 4)
        if bearing_v_deg is not None:
            d["bearing_offset_v_deg"] = round(float(bearing_v_deg), 4)
        if stream_skew_ms is not None:
            d["nir_thermal_stream_skew_ms"] = round(float(stream_skew_ms), 3)
        if skew_quality is not None:
            d["nir_thermal_skew_quality"] = str(skew_quality)
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
            "software": "fusion_live_optimized",
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
            "nir_thermal_skew_quality_counts": dict(self.skew_quality_counts),
            "experiment_context": self.experiment_context,
            "optimization_config": self.optimization_config,
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


# [OPTIMIZATION] S2 - Metadata & reproducibility context
# Purpose:
#   Chuẩn hóa ngữ cảnh thí nghiệm để so sánh phiên chạy trong điều kiện thực địa handheld.
# Change:
#   Bổ sung experiment_context và optimization_config vào manifest/capture/session.
# Expected impact:
#   Tăng tái lập và khả năng audit cho luận văn/demo, không ảnh hưởng realtime.
def build_experiment_context(cfg: Dict[str, Any]) -> Dict[str, Any]:
    exp = cfg.get("experiment", {}) if isinstance(cfg, dict) else {}
    return {
        "scenario_type": str(exp.get("scenario_type", "unspecified")),
        "environment": str(exp.get("environment", "unspecified")),
        "lighting_level": str(exp.get("lighting_level", "unspecified")),
        "operator_id": str(exp.get("operator_id", "")),
        "build_label": str(exp.get("build_label", "fusion_live_optimized")),
        "notes_short": str(exp.get("notes_short", "")),
    }


# [OPTIMIZATION] C4/S3 - Profile policy tuning (safe fallback)
# Purpose:
#   Điều chỉnh nhẹ anti-glare/temporal/display-cap theo bối cảnh vận hành để giảm flicker và mỏi mắt.
# Change:
#   Thêm profile runtime; có thể tắt để quay về baseline.
# Expected impact:
#   UX ổn định hơn khi handheld pan hoặc gặp cảnh chói; chi phí CPU thấp.
def resolve_optimization_profile(profile: str, haze_preset: bool, profiles_enabled: bool) -> Dict[str, Any]:
    baseline: Dict[str, Any] = {
        "name": "baseline",
        "nir_high_pct": 95.0,
        "nir_saturate_at": 233.0,
        "nir_gamma": 0.72,
        "thermal_high_pct": 99.0,
        "thermal_saturate_at": 240.0,
        "thermal_gamma": 0.74,
        "display_l_max": int(TIER_S_DISPLAY_L_MAX),
        "display_l_max_when_glare": int(TIER_S_DISPLAY_L_MAX_WHEN_GLARE),
        "temporal_prev_weight": float(TIER_S_GLARE_TEMPORAL_PREV_WEIGHT),
        "fusion_alpha_boost": 0.0,
        "thermal_floor": 3.0,
        "thermal_fg_threshold": 18.0,
        "thermal_edge_strength": 0.2,
        "thermal_agc_low_pct": 2.0,
        "thermal_agc_high_pct": 98.0,
    }
    if not profiles_enabled:
        profile = "baseline"
    p = profile.strip().lower()
    if p == "static_scan":
        baseline.update(
            {
                "name": "static_scan",
                "display_l_max": 236,
                "display_l_max_when_glare": 200,
                "temporal_prev_weight": 0.50,
                "nir_gamma": 0.70,
            }
        )
    elif p == "handheld_pan":
        baseline.update(
            {
                "name": "handheld_pan",
                "display_l_max": 242,
                "display_l_max_when_glare": 210,
                "temporal_prev_weight": 0.34,
                "nir_high_pct": 96.0,
                "nir_saturate_at": 236.0,
            }
        )
    elif p == "high_glare":
        baseline.update(
            {
                "name": "high_glare",
                "display_l_max": 232,
                "display_l_max_when_glare": 194,
                "temporal_prev_weight": 0.48,
                "nir_high_pct": 93.0,
                "nir_saturate_at": 228.0,
                "thermal_saturate_at": 236.0,
            }
        )
    else:
        baseline["name"] = "baseline"

    # [OPTIMIZATION] C1 - Haze/smoke preset (thermal-first, lightweight)
    # Purpose:
    #   Cải thiện tính dùng được trong cảnh tương phản thấp/sương khói mà không thêm model nặng.
    # Change:
    #   Điều chỉnh nhẹ tham số thermal + alpha fusion, giữ nguyên kiến trúc.
    # Expected impact:
    #   Quan sát ổn định hơn trong môi trường xấu, chi phí CPU gần như không đổi.
    if haze_preset:
        baseline.update(
            {
                "fusion_alpha_boost": 0.12,
                "thermal_floor": 4.0,
                "thermal_fg_threshold": 20.0,
                "thermal_edge_strength": 0.14,
                "thermal_agc_low_pct": 1.0,
                "thermal_agc_high_pct": 96.0,
            }
        )
    return baseline


# [ENV] Preset apply logic
# Purpose: map environment labels to safe runtime parameters (reuse opt_profile keys + thermal/E1).
# Notes:
#   - opt_overrides: chỉ các key sau (merge lên opt_cfg_base): nir_high_pct, nir_saturate_at, nir_gamma,
#     temporal_prev_weight, thermal_floor, thermal_fg_threshold, fusion_alpha_boost,
#     display_l_max, display_l_max_when_glare. Không override thermal_saturate/edge/agc tại đây (giữ profile + haze).
#   - thermal_extra: chỉ thermal_detect_use_anti_glare khi cần; mặc định {} — chỉ glare_heavy bật True.
#   - e1_overrides: chỉ feature_e1_z_thresh, feature_e1_min_area (restore từ snapshot trước khi apply).
#   - display_grade: CHỈ display path (sau shake); không ảnh hưởng detect/fusion.
#   - thermal_3dnr_alpha: None = giữ alpha hiện tại; else clamp trong apply_env_runtime.
#   - Không thêm exposure/brilliance/vibrance/blackpoint runtime ở pipeline chính.


def _format_env_apply_log(preset_name: str, pobj: Dict[str, Any]) -> str:
    """Một dòng gọn: preset + các override chính (không spam frame)."""
    ov = pobj.get("opt_overrides") or {}
    te = pobj.get("thermal_extra") or {}
    e1 = pobj.get("e1_overrides") or {}
    a3 = pobj.get("thermal_3dnr_alpha")
    parts: List[str] = []
    for k in (
        "nir_high_pct",
        "nir_saturate_at",
        "nir_gamma",
        "temporal_prev_weight",
        "thermal_floor",
        "thermal_fg_threshold",
        "fusion_alpha_boost",
        "display_l_max",
        "display_l_max_when_glare",
    ):
        if k in ov:
            v = ov[k]
            parts.append(f"{k}={v}" if not isinstance(v, float) else f"{k}={v:.3g}")
    if a3 is not None:
        parts.append(f"thermal_3dnr_alpha={float(a3):.2f}")
    if te.get("thermal_detect_use_anti_glare"):
        parts.append("thermal_detect_use_anti_glare=True")
    if e1:
        parts.append("e1=" + ",".join(f"{kk}={e1[kk]}" for kk in sorted(e1.keys())))
    dg = pobj.get("display_grade") or {}
    if dg:
        parts.append("display_grade=on")
    return f"[ENV] apply preset={preset_name} " + (" ".join(parts) if parts else "(defaults)")


def build_env_presets() -> Dict[str, Any]:
    """Preset môi trường: chỉ override các tham số đã chuẩn hóa (xem docstring khối ENV)."""
    return {
        "default": {
            "label": "default",
            "opt_overrides": {},
            "thermal_extra": {},
            "e1_overrides": {},
            "display_grade": {},
            "thermal_3dnr_alpha": None,
        },
        "night": {
            "label": "night",
            "opt_overrides": {
                "temporal_prev_weight": 0.50,
                "display_l_max": 236,
                "display_l_max_when_glare": 200,
                "nir_gamma": 0.70,
            },
            "thermal_extra": {},
            "e1_overrides": {
                "feature_e1_z_thresh": 1.05,
                "feature_e1_min_area": 8,
            },
            "display_grade": {"brightness": 0.04, "contrast": 1.05, "saturation": 1.0, "warmth": 0.0},
            "thermal_3dnr_alpha": 0.72,
        },
        "low_light": {
            "label": "low_light",
            "opt_overrides": {
                "temporal_prev_weight": 0.46,
                "display_l_max": 238,
                "display_l_max_when_glare": 204,
            },
            "thermal_extra": {},
            "e1_overrides": {
                "feature_e1_z_thresh": 1.1,
                "feature_e1_min_area": 8,
            },
            "display_grade": {"brightness": 0.05, "contrast": 1.06, "saturation": 1.0, "warmth": 0.0},
            "thermal_3dnr_alpha": 0.68,
        },
        "glare_heavy": {
            "label": "glare_heavy",
            "opt_overrides": {
                "display_l_max": 232,
                "display_l_max_when_glare": 194,
                "temporal_prev_weight": 0.48,
                "nir_high_pct": 93.0,
                "nir_saturate_at": 228.0,
            },
            "thermal_extra": {"thermal_detect_use_anti_glare": True},
            "e1_overrides": {
                "feature_e1_z_thresh": 1.55,
            },
            "display_grade": {"brightness": -0.03, "contrast": 0.94, "saturation": 0.92, "warmth": 0.0},
            "thermal_3dnr_alpha": None,
        },
        "backlight": {
            "label": "backlight",
            "opt_overrides": {
                "nir_high_pct": 92.0,
                "nir_saturate_at": 226.0,
                "display_l_max": 234,
                "display_l_max_when_glare": 196,
            },
            "thermal_extra": {},
            "e1_overrides": {"feature_e1_z_thresh": 1.45},
            "display_grade": {"brightness": 0.02, "contrast": 1.08, "saturation": 0.95, "warmth": 0.02},
            "thermal_3dnr_alpha": None,
        },
        "fog": {
            "label": "fog",
            "opt_overrides": {
                "fusion_alpha_boost": 0.12,
                "thermal_floor": 4.0,
                "thermal_fg_threshold": 20.0,
            },
            "thermal_extra": {},
            "e1_overrides": {
                "feature_e1_z_thresh": 1.08,
                "feature_e1_min_area": 10,
            },
            "display_grade": {"brightness": 0.0, "contrast": 1.12, "saturation": 0.9, "warmth": 0.0},
            "thermal_3dnr_alpha": 0.58,
        },
        "haze": {
            "label": "haze",
            "opt_overrides": {
                "fusion_alpha_boost": 0.06,
                "thermal_floor": 3.5,
                "thermal_fg_threshold": 19.0,
            },
            "thermal_extra": {},
            "e1_overrides": {"feature_e1_z_thresh": 1.15},
            "display_grade": {"brightness": 0.0, "contrast": 1.1, "saturation": 0.93, "warmth": 0.0},
            "thermal_3dnr_alpha": 0.60,
        },
        "high_contrast": {
            "label": "high_contrast",
            "opt_overrides": {
                "display_l_max": 238,
                "display_l_max_when_glare": 206,
            },
            "thermal_extra": {},
            "e1_overrides": {"feature_e1_z_thresh": 1.35},
            "display_grade": {"brightness": 0.0, "contrast": 1.0, "saturation": 1.05, "warmth": 0.0},
            "thermal_3dnr_alpha": None,
        },
        "cluttered_bg": {
            "label": "cluttered_bg",
            "opt_overrides": {},
            "thermal_extra": {},
            "e1_overrides": {
                "feature_e1_z_thresh": 1.65,
                "feature_e1_min_area": 16,
            },
            "display_grade": {},
            "thermal_3dnr_alpha": None,
        },
        "clear_day": {
            "label": "clear_day",
            "opt_overrides": {
                "temporal_prev_weight": 0.34,
                "display_l_max": 242,
                "display_l_max_when_glare": 210,
                "nir_high_pct": 96.0,
                "nir_saturate_at": 236.0,
            },
            "thermal_extra": {},
            "e1_overrides": {},
            "display_grade": {"brightness": 0.0, "contrast": 1.0, "saturation": 1.08, "warmth": 0.0},
            "thermal_3dnr_alpha": 0.62,
        },
        "indoor": {
            "label": "indoor",
            "opt_overrides": {
                "temporal_prev_weight": 0.40,
                "display_l_max": 240,
            },
            "thermal_extra": {},
            "e1_overrides": {"feature_e1_z_thresh": 1.2},
            "display_grade": {"brightness": 0.0, "contrast": 1.03, "saturation": 1.05, "warmth": 0.015},
            "thermal_3dnr_alpha": None,
        },
        "night_fog": {
            "label": "night_fog",
            "opt_overrides": {
                "temporal_prev_weight": 0.48,
                "display_l_max": 234,
                "display_l_max_when_glare": 198,
                "fusion_alpha_boost": 0.10,
                "thermal_floor": 4.0,
                "thermal_fg_threshold": 19.0,
                "nir_gamma": 0.70,
            },
            "thermal_extra": {},
            "e1_overrides": {
                "feature_e1_z_thresh": 1.0,
                "feature_e1_min_area": 8,
            },
            "display_grade": {"brightness": 0.04, "contrast": 1.1, "saturation": 0.95, "warmth": 0.0},
            "thermal_3dnr_alpha": 0.65,
        },
        "day_glare": {
            "label": "day_glare",
            "opt_overrides": {
                "display_l_max": 230,
                "display_l_max_when_glare": 192,
                "temporal_prev_weight": 0.46,
                "nir_high_pct": 92.0,
                "nir_saturate_at": 227.0,
            },
            "thermal_extra": {},
            "e1_overrides": {"feature_e1_z_thresh": 1.5},
            "display_grade": {"brightness": -0.04, "contrast": 0.93, "saturation": 0.9, "warmth": 0.0},
            "thermal_3dnr_alpha": None,
        },
        "low_light_cluttered": {
            "label": "low_light_cluttered",
            "opt_overrides": {
                "temporal_prev_weight": 0.44,
                "display_l_max": 236,
            },
            "thermal_extra": {},
            "e1_overrides": {
                "feature_e1_z_thresh": 1.45,
                "feature_e1_min_area": 14,
            },
            "display_grade": {"brightness": 0.04, "contrast": 1.05, "saturation": 1.0, "warmth": 0.0},
            "thermal_3dnr_alpha": 0.66,
        },
        "backlight_high_contrast": {
            "label": "backlight_high_contrast",
            "opt_overrides": {
                "nir_high_pct": 91.0,
                "nir_saturate_at": 225.0,
                "display_l_max": 232,
                "display_l_max_when_glare": 190,
            },
            "thermal_extra": {},
            "e1_overrides": {"feature_e1_z_thresh": 1.5},
            "display_grade": {"brightness": 0.0, "contrast": 1.06, "saturation": 0.94, "warmth": 0.02},
            "thermal_3dnr_alpha": None,
        },
    }


ENV_PRESETS: Dict[str, Any] = build_env_presets()


def merge_opt_cfg_with_preset(opt_cfg_base: Dict[str, Any], preset_name: str) -> Dict[str, Any]:
    """Gộp opt_cfg_base (đã có profile + haze) với opt_overrides của preset."""
    merged = dict(opt_cfg_base)
    p = ENV_PRESETS.get(preset_name) or ENV_PRESETS["default"]
    merged.update(p.get("opt_overrides", {}))
    return merged


# [ENV] Priority đơn nhãn (khi không khớp compound)
_ENV_SINGLE_PRIORITY: Tuple[str, ...] = (
    "glare_heavy",
    "backlight_high_contrast",
    "backlight",
    "fog",
    "haze",
    "night",
    "low_light",
    "cluttered_bg",
    "high_contrast",
    "clear_day",
    "indoor",
)


def select_env_preset_from_tags(tags: Set[str]) -> str:
    """Chọn một preset từ tập tag; ưu tiên cặp ghép rồi priority đơn."""
    if not tags:
        return "default"
    if "night" in tags and "fog" in tags:
        return "night_fog"
    if "clear_day" in tags and "glare_heavy" in tags:
        return "day_glare"
    if "low_light" in tags and "cluttered_bg" in tags:
        return "low_light_cluttered"
    if "backlight" in tags and "high_contrast" in tags:
        return "backlight_high_contrast"
    for key in _ENV_SINGLE_PRIORITY:
        if key in tags:
            return key
    return "default"


def infer_env_tags_auto_rule(
    *,
    nir_b_ema: Optional[float],
    is_night_mode: bool,
    nir_gray_std: Optional[float],
    glare_nir: bool,
    haze_config_on: bool,
    std_low: float,
    std_high: float,
) -> Set[str]:
    """Rule-based nhẹ: không ML; std tính trên gray downscale."""
    tags: Set[str] = set()
    if haze_config_on:
        tags.add("fog")
    if is_night_mode:
        tags.add("night")
    if nir_b_ema is not None:
        if not is_night_mode and nir_b_ema < 38.0:
            tags.add("low_light")
        if nir_b_ema >= 95.0:
            tags.add("clear_day")
    if glare_nir:
        tags.add("glare_heavy")
    if nir_gray_std is not None:
        if nir_gray_std < std_low:
            tags.add("haze")
        elif nir_gray_std > std_high:
            tags.add("cluttered_bg")
        if 35.0 <= nir_gray_std <= std_high and nir_b_ema is not None and 40.0 <= nir_b_ema <= 90.0:
            tags.add("high_contrast")
    # Heuristic backlight: sáng đuôi nhưng trung bình không cực cao (đã có glare tag)
    if glare_nir and nir_b_ema is not None and 45.0 <= nir_b_ema < 85.0:
        tags.add("backlight")
    if nir_b_ema is not None and 55.0 <= nir_b_ema < 88.0 and not is_night_mode and not glare_nir:
        tags.add("indoor")
    return tags


class EnvPresetController:
    """Hysteresis / debounce: giữ preset ổn định, tránh nhảy từng frame."""

    def __init__(
        self,
        *,
        fallback: str,
        hysteresis_frames: int,
    ):
        self.fallback = str(fallback) if fallback else "default"
        self.hysteresis_frames = int(max(1, hysteresis_frames))
        self.stable_name: str = self.fallback
        self._candidate: Optional[str] = None
        self._streak: int = 0

    def reset(self) -> None:
        self.stable_name = self.fallback
        self._candidate = None
        self._streak = 0

    def update(self, desired: str) -> str:
        d = desired if desired in ENV_PRESETS else self.fallback
        if d == self.stable_name:
            self._candidate = None
            self._streak = 0
            return self.stable_name
        if self._candidate != d:
            self._candidate = d
            self._streak = 1
        else:
            self._streak += 1
        if self._streak >= self.hysteresis_frames:
            self.stable_name = d
            self._candidate = None
            self._streak = 0
        return self.stable_name


def _nir_gray_std_quick(nir_bgr: np.ndarray, max_side: int = 128) -> float:
    g = cv.cvtColor(nir_bgr, cv.COLOR_BGR2GRAY)
    h, w = g.shape[:2]
    m = max(h, w)
    if m > max_side:
        s = max_side / float(m)
        g = cv.resize(g, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv.INTER_AREA)
    return float(np.std(g.astype(np.float32)))


def apply_e1_overrides(det: "ThermalAnomalyDetectorLite", overrides: Dict[str, Any]) -> None:
    if not overrides:
        return
    if "feature_e1_z_thresh" in overrides:
        det.z_thresh = float(overrides["feature_e1_z_thresh"])
    if "feature_e1_heat_thresh" in overrides:
        det.heat_thresh = float(overrides["feature_e1_heat_thresh"])
    if "feature_e1_min_area" in overrides:
        det.min_area = int(overrides["feature_e1_min_area"])
    if "feature_e1_raw_mix" in overrides:
        det.raw_mix = float(np.clip(float(overrides["feature_e1_raw_mix"]), 0.0, 1.0))
    if "feature_e1_local_kernel" in overrides:
        k = int(max(3, int(overrides["feature_e1_local_kernel"])))
        if k % 2 == 0:
            k += 1
        det.local_kernel = k


def snapshot_e1_defaults(det: "ThermalAnomalyDetectorLite") -> Dict[str, Any]:
    return {
        "feature_e1_z_thresh": float(det.z_thresh),
        "feature_e1_heat_thresh": float(det.heat_thresh),
        "feature_e1_min_area": int(det.min_area),
        "feature_e1_raw_mix": float(det.raw_mix),
        "feature_e1_local_kernel": int(det.local_kernel),
    }


def restore_e1_from_snapshot(det: "ThermalAnomalyDetectorLite", snap: Dict[str, Any]) -> None:
    apply_e1_overrides(det, snap)


# [OPTIMIZATION] S6 - Stream skew quality gate
# Purpose:
#   Chuyển S6 từ số đo thô sang trạng thái chất lượng đồng bộ (GOOD/DEGRADED/BAD) có hysteresis.
# Change:
#   EMA + bộ đếm chuyển trạng thái để tránh cảnh báo giả do spike ngắn.
# Expected impact:
#   Cảnh báo sync đáng tin cậy hơn trên handheld realtime.
class StreamSkewQualityGate:
    def __init__(
        self,
        *,
        ema_alpha: float = 0.25,
        degraded_on_ms: float = 20.0,
        bad_on_ms: float = 50.0,
        degraded_back_ms: float = 14.0,
        bad_back_ms: float = 32.0,
        hold_frames: int = 6,
    ):
        self.ema_alpha = float(ema_alpha)
        self.degraded_on_ms = float(degraded_on_ms)
        self.bad_on_ms = float(bad_on_ms)
        self.degraded_back_ms = float(degraded_back_ms)
        self.bad_back_ms = float(bad_back_ms)
        self.hold_frames = int(max(1, hold_frames))
        self.ema_ms: Optional[float] = None
        self.state: str = "GOOD"
        self._strike = 0

    def update(self, skew_ms: Optional[float]) -> tuple[str, Optional[float], bool]:
        if skew_ms is None:
            return self.state, self.ema_ms, False
        s = float(skew_ms)
        if self.ema_ms is None:
            self.ema_ms = s
        else:
            self.ema_ms = (1.0 - self.ema_alpha) * self.ema_ms + self.ema_alpha * s
        prev = self.state
        e = float(self.ema_ms)
        if self.state == "GOOD":
            if e >= self.degraded_on_ms:
                self._strike += 1
                if self._strike >= self.hold_frames:
                    self.state = "DEGRADED"
                    self._strike = 0
            else:
                self._strike = 0
        elif self.state == "DEGRADED":
            if e >= self.bad_on_ms:
                self._strike += 1
                if self._strike >= self.hold_frames:
                    self.state = "BAD"
                    self._strike = 0
            elif e <= self.degraded_back_ms:
                self._strike += 1
                if self._strike >= self.hold_frames:
                    self.state = "GOOD"
                    self._strike = 0
            else:
                self._strike = 0
        else:  # BAD
            if e <= self.bad_back_ms:
                self._strike += 1
                if self._strike >= self.hold_frames:
                    self.state = "DEGRADED"
                    self._strike = 0
            else:
                self._strike = 0
        return self.state, self.ema_ms, (self.state != prev)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — Thuật toán TOÀN CỤC (§1b) + chỗ thêm sau (IMU, GNSS, …)
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


def _percentile_u8_from_hist(hist: np.ndarray, pct: float) -> float:
    """Ước lượng percentile cho ảnh uint8 bằng histogram (nhanh hơn np.percentile cho frame nhỏ)."""
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
) -> tuple[bool, bool, float, float, float]:
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


# [PERF] Optimize glare computation
# Change:
#   Gộp đánh giá glare (need_compress + HUD) trong 1 pass trên cùng gray downscale;
#   tùy chọn bản lite dùng histogram percentile tránh np.percentile nhiều lần.
# Impact:
#   Giảm chi phí CPU ở mode IMX/Fusion, giữ hành vi gần tương đương ngưỡng cũ.
def nir_glare_eval(
    frame_bgr: np.ndarray,
    *,
    high_pct: float = 95.0,
    saturate_at: float = 233.0,
    use_fast: bool = True,
) -> tuple[bool, bool]:
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


# [PERF] Optimize jerk estimation
# Change:
#   Dùng percentile nhanh theo histogram trên ảnh chênh lệch uint8 (lite mode), tránh sort nặng mỗi frame.
#   Có fallback np.percentile khi tắt lite_mode.
# Impact:
#   Giảm CPU ở nhánh D2, vẫn giữ khả năng phân biệt steady/motion theo ngưỡng score cũ.
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
        near_active_ratio: float = 0.62,
        lite_mode: bool = True,
        perf_benchmark_samples: int = 24,
    ):
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
        self._prev_gray = None
        self._hold = 0
        self._strike = 0
        self.active = False
        self.near_active = False
        self.last_score = 0.0

    @staticmethod
    def _ema_update(cur: float, val: float, alpha: float = 0.12) -> float:
        if cur <= 0.0:
            return float(val)
        return float((1.0 - alpha) * cur + alpha * float(val))

    def get_saved_ms_estimate(self) -> float:
        if self._perf_n <= 0:
            return 0.0
        return max(0.0, self._perf_old_ms_ema - self._perf_new_ms_ema)

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

    def _percentile_fast_u8(self, arr_u8: np.ndarray, pct: float) -> float:
        hist = np.bincount(arr_u8.ravel(), minlength=256).astype(np.int32)
        return _percentile_u8_from_hist(hist, pct)

    def update(self, nir_bgr):
        if nir_bgr is None:
            if self._hold > 0:
                self._hold -= 1
            self.active = self._hold > 0
            self.near_active = False
            self.last_score = 0.0
            return
        gray = self._to_small_gray(nir_bgr)
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

    def process(
        self,
        frame_bgr: np.ndarray,
        jerk_active: bool = False,
        soft_motion_active: bool = False,
    ) -> np.ndarray:
        if self.mode == "off" or frame_bgr is None:
            return frame_bgr
        if jerk_active or soft_motion_active:
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

    def get_foreground_mask(
        self,
        frame,
        threshold=18.0,
        min_area=12,
        max_fg_ratio=0.5,
        morph_kernel_size: int = 5,
        open_iterations: int = 2,
        close_iterations: int = 1,
    ):
        """Mask nhị phân đã lọc nhiễu mạnh. Nếu foreground > max_fg_ratio thì coi như nhiễu → trả zeros."""
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
    def __init__(
        self,
        warmup=40,
        anti_glare=True,
        *,
        thermal_high_pct: float = 99.0,
        thermal_saturate_at: float = 240.0,
        thermal_gamma: float = 0.74,
        thermal_floor: float = 3.0,
        thermal_fg_threshold: float = 18.0,
        thermal_edge_strength: float = 0.2,
        thermal_agc_low_pct: float = 2.0,
        thermal_agc_high_pct: float = 98.0,
        detail_preserve_detect: bool = True,
        detect_raw_mix: float = 0.55,
        detail_grad_threshold: float = 9.0,
        detail_threshold_relax: float = 3.0,
        detect_use_anti_glare: bool = False,
        thermal_3dnr_alpha: float = 0.65,
    ):
        self.temporal = ThermalTemporalFilter(alpha=float(thermal_3dnr_alpha))
        self.bg = ThermalBackgroundModel(warmup_frames=warmup, adaptive_rate=0.005)
        self.anti_glare_enabled = anti_glare
        self.thermal_high_pct = float(thermal_high_pct)
        self.thermal_saturate_at = float(thermal_saturate_at)
        self.thermal_gamma = float(thermal_gamma)
        self.thermal_floor = float(thermal_floor)
        self.thermal_fg_threshold = float(thermal_fg_threshold)
        self.thermal_edge_strength = float(thermal_edge_strength)
        self.thermal_agc_low_pct = float(thermal_agc_low_pct)
        self.thermal_agc_high_pct = float(thermal_agc_high_pct)
        self.detail_preserve_detect = bool(detail_preserve_detect)
        self.detect_raw_mix = float(np.clip(detect_raw_mix, 0.0, 1.0))
        self.detail_grad_threshold = float(max(0.0, detail_grad_threshold))
        self.detail_threshold_relax = float(max(0.0, detail_threshold_relax))
        self.detect_use_anti_glare = bool(detect_use_anti_glare)
        self.last_detail_mode_active = False
        self.last_detail_grad_mean = 0.0

    def apply_env_runtime(
        self,
        merged_opt_cfg: Dict[str, Any],
        thermal_extra: Optional[Dict[str, Any]] = None,
        thermal_3dnr_alpha: Optional[float] = None,
    ) -> None:
        """[ENV] Cập nhật tham số thermal từ opt_cfg đã merge + extra; không reset BG model."""
        m = merged_opt_cfg
        self.thermal_high_pct = float(m["thermal_high_pct"])
        self.thermal_saturate_at = float(m["thermal_saturate_at"])
        self.thermal_gamma = float(m["thermal_gamma"])
        self.thermal_floor = float(m["thermal_floor"])
        self.thermal_fg_threshold = float(m["thermal_fg_threshold"])
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
        # Mỗi lần apply ENV preset: mặc định False; chỉ glare_heavy truyền True trong thermal_extra.
        self.detect_use_anti_glare = bool(te.get("thermal_detect_use_anti_glare", False))
        if thermal_3dnr_alpha is not None:
            self.temporal.alpha = float(np.clip(float(thermal_3dnr_alpha), 0.35, 0.85))

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
        fg_mask: mask nhị phân cho Surveillance.

        Gọi **một lần mỗi mẫu nhiệt mới** khi ``skip_thermal_reprocess_when_same_timestamp`` bật
        (main loop); không lặp trên cùng timestamp MI48 khi vòng lặp nhanh hơn FPS thermal."""
        denoised = self.temporal.process(raw_frame)
        self.bg.update(denoised)

        # [FIX] Thermal blur issue
        # Based on: fusion_advanced.py mode 7
        # Cause:
        #   Chuỗi làm mượt detect quá mạnh (temporal + morphology lớn + nén highlight) làm mềm biên nhiệt.
        # Fix:
        #   Tách detect path gần style mode 7: dùng raw+3DNR nhẹ, giảm morphology, mặc định không anti-glare cho detect.
        # Behavior:
        #   Giữ hiển thị thermal mượt nhưng detect giữ biên tốt hơn cho vật thể gần/nhỏ (vd bàn tay).
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
            grad_mag = cv.magnitude(gx, gy)
            grad_mean = float(np.mean(grad_mag))
            detail_mode = grad_mean >= self.detail_grad_threshold
            self.last_detail_grad_mean = grad_mean
            self.last_detail_mode_active = detail_mode
            morph_kernel = 3
            open_iters = 0 if detail_mode else 1
            close_iters = 1
            fg_threshold = max(8.0, self.thermal_fg_threshold - (self.detail_threshold_relax if detail_mode else 0.0))
        else:
            detect_src = denoised
            self.last_detail_grad_mean = 0.0
            self.last_detail_mode_active = False
            morph_kernel = 3
            open_iters = 1
            close_iters = 1
            fg_threshold = self.thermal_fg_threshold

        d_for_detect = (
            thermal_anti_glare(
                detect_src,
                high_pct=self.thermal_high_pct,
                saturate_at=self.thermal_saturate_at,
                gamma=self.thermal_gamma,
            )
            if self.anti_glare_enabled and self.detect_use_anti_glare
            else detect_src
        )
        agc = thermal_agc(denoised, low_pct=self.thermal_agc_low_pct, high_pct=self.thermal_agc_high_pct)
        agc = (
            thermal_anti_glare(
                agc,
                high_pct=self.thermal_high_pct,
                saturate_at=self.thermal_saturate_at,
                gamma=self.thermal_gamma,
            )
            if self.anti_glare_enabled
            else agc
        )
        enhanced = thermal_edge_enhance(agc, strength=self.thermal_edge_strength)
        heat_map = self.bg.get_heat_map(d_for_detect, floor=self.thermal_floor)
        fg_mask = self.bg.get_foreground_mask(
            d_for_detect,
            threshold=fg_threshold,
            min_area=12,
            morph_kernel_size=morph_kernel,
            open_iterations=open_iters,
            close_iterations=close_iters,
        )
        return denoised, enhanced, heat_map, fg_mask


# [FEATURE] E1-lite thermal detection
# Input:
#   thermal_raw, thermal_denoised, heat_map, fg_mask, jerk_active
# Output:
#   candidate_mask (uint8), blobs [{cx, cy, area, score}]
# Notes:
#   Lightweight local-stats + heat gating; ưu tiên detect path (ít smoothing) để giữ chi tiết gần.
class ThermalAnomalyDetectorLite:
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
    ) -> tuple[np.ndarray, List[Dict[str, float]]]:
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


def sector_from_bearing_deg(bearing_h_deg: float, center_half_width_deg: float = 6.0) -> str:
    if bearing_h_deg < -abs(center_half_width_deg):
        return "LEFT"
    if bearing_h_deg > abs(center_half_width_deg):
        return "RIGHT"
    return "CENTER"


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

    def __init__(
        self,
        proc_w=320,
        proc_h=240,
        patch_size=5,
        update_rate=8,
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
        self.A_buffer = deque(maxlen=10)
        self.A_buffer.append(self.last_A)
        self.brightness_buffer = deque(maxlen=15)
        self.is_night_mode = False
        self._night_hyst = False
        self._night_hyst_ready = False

        _cs = float(np.clip(float(clahe_clip_scale), 0.25, 2.0))
        self.clahe_levels = {
            "very_dark": cv.createCLAHE(clipLimit=3.0 * _cs, tileGridSize=(4, 4)),
            "dark": cv.createCLAHE(clipLimit=2.0 * _cs, tileGridSize=(6, 6)),
            "medium": cv.createCLAHE(clipLimit=1.5 * _cs, tileGridSize=(8, 8)),
        }
        self.detail_strength = float(max(0.0, detail_strength))
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


def display_color_grade_bgr(
    frame_bgr: np.ndarray,
    *,
    brightness: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    warmth: float = 0.0,
    shadows: float = 0.0,
    highlights: float = 0.0,
) -> np.ndarray:
    """[ENV] Display path only — một pass LAB nhẹ (Pi4-safe).

    - brightness: dịch L theo [-0.12, 0.12] * 255
    - contrast: co giãn L quanh 128
    - saturation: scale a,b quanh 128
    - warmth: dịch a (LAB) — ấm hơn khi warmth > 0
    - shadows / highlights: nhẹ (lift shadow / nén highlight trên L), tránh chồng glare-lite

    Không dùng cho: exposure (capture), detect path, thermal raw.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr
    if (
        abs(brightness) < 1e-6
        and abs(contrast - 1.0) < 1e-6
        and abs(saturation - 1.0) < 1e-6
        and abs(warmth) < 1e-6
        and abs(shadows) < 1e-6
        and abs(highlights) < 1e-6
    ):
        return frame_bgr
    lab = cv.cvtColor(frame_bgr, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    lf = l.astype(np.float32)
    if abs(shadows) > 1e-6:
        mask = np.clip((128.0 - lf) / 128.0, 0.0, 1.0)
        lf = lf + shadows * 40.0 * mask
    if abs(highlights) < 1e-6:
        pass
    else:
        hm = np.clip((lf - 200.0) / 55.0, 0.0, 1.0)
        lf = lf - highlights * 35.0 * hm
    lf = lf + brightness * 255.0
    lf = (lf - 128.0) * float(contrast) + 128.0
    lf = np.clip(lf, 0.0, 255.0)
    af = a.astype(np.float32)
    bf = b.astype(np.float32)
    af = (af - 128.0) * float(saturation) + 128.0 + warmth * 40.0
    bf = (bf - 128.0) * float(saturation) + 128.0
    out = cv.cvtColor(
        cv.merge([lf.astype(np.uint8), np.clip(af, 0, 255).astype(np.uint8), np.clip(bf, 0, 255).astype(np.uint8)]),
        cv.COLOR_LAB2BGR,
    )
    return out


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
#
# Mục 4 (song song I/O): NIR và thermal chạy **hai thread** riêng; luồng chính chỉ
# ``get_latest()`` + fusion. Thermal ~9 FPS (MI48), NIR có thể ~60 (Picamera2) —
# xử lý thermal không nên lặp trên cùng mẫu khi timestamp không đổi (xem main loop).

RPI_GPIO_I2C_CHANNEL = 1
RPI_GPIO_SPI_BUS = 0
RPI_GPIO_SPI_CE_MI48 = 0
MI48_I2C_ADDRESS = 0x40
MI48_SPI_MAX_SPEED_HZ = 7800000
MI48_SPI_CS_DELAY = 0.0001
SPI_XFER_SIZE_BYTES = 160
# [IMPROVEMENT] Proper heatmap mapping
# Change: dùng full colormap (ưu tiên TURBO, fallback JET) thay biểu diễn đỏ/xanh đơn giản.
THERMAL_COLORMAP = getattr(cv, "COLORMAP_TURBO", cv.COLORMAP_JET)


class ThermalCapture(threading.Thread):
    """MI48 qua SPI/I2C — không qua V4L2; ``last_mono`` đổi mỗi khi có mẫu nhiệt mới.

    ``set_idle(True)``: không gọi ``wait_for_active``/read — chỉ sleep; dùng khi mode 1 (IMX) không cần nhiệt.
    """

    def __init__(self, flip_h=False, blur_ksize: int = 3):
        super().__init__(daemon=True)
        self.latest = None
        self.last_mono: float | None = None
        self.lock = threading.Lock()
        self.running = True
        self._idle = False
        self.mi48 = None
        self.mi48_spi_cs_n = None
        self.flip_h = flip_h
        k = int(max(1, blur_ksize))
        if k % 2 == 0:
            k += 1
        self.blur_ksize = k

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
            if self._idle:
                time.sleep(0.05)
                continue
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
                if self.blur_ksize >= 3:
                    img8u = cv.GaussianBlur(img8u, (self.blur_ksize, self.blur_ksize), 0)
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

    def set_idle(self, idle: bool) -> None:
        if self._idle == idle:
            return
        self._idle = idle

    def stop(self):
        self.running = False


class NIRCapture(threading.Thread):
    """CSI camera qua **Picamera2** → stack **libcamera** + ISP (không phải V4L2 capture thủ công).

    Mục 4.F: ``nir_capture_fps`` điều khiển ``controls["FrameRate"]`` — giảm FPS nếu cần
    bớt tải copy/ISP khi pipeline chính không theo kịp.

    ``set_idle(True)``: không gọi ``capture_array`` — chỉ sleep; dùng khi chỉ mode thermal (2) và cho phép CONFIG.
    """

    def __init__(self, no_rgb2bgr=True, nir_fps: float = 60.0):
        super().__init__(daemon=True)
        self.latest = None
        self.last_mono: float | None = None
        self.lock = threading.Lock()
        self.running = True
        self._idle = False
        self.camera = None
        self.no_rgb2bgr = no_rgb2bgr
        self.nir_fps = float(np.clip(nir_fps, 5.0, 120.0))

    def run(self):
        if not _HAS_HARDWARE:
            return
        self.camera = Picamera2()
        _fps = int(round(self.nir_fps))
        config = self.camera.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            controls={
                "FrameRate": _fps,
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
            if self._idle:
                time.sleep(0.05)
                continue
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

    def set_idle(self, idle: bool) -> None:
        if self._idle == idle:
            return
        self._idle = idle

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


def _capture_hardware_idle_flags(
    mode: str,
    mode_switch_pending: Optional[str],
    cfg: Dict[str, Any],
) -> tuple[bool, bool]:
    """Trả về ``(thermal_idle, nir_idle)`` — ``True`` = thread capture **không** đọc MI48/Picamera2 (chỉ sleep ngắn).

    - Mode **1 (imx)**: chỉ cần NIR → thermal có thể idle (SPI/MI48 không chạy vòng read).
    - Mode **2 (thermal)**: chỉ cần thermal → NIR có thể idle nếu ``pause_nir_capture_when_thermal_only`` và không cần NIR cho ENV.
    - Mode **3 (fusion)**: cả hai luôn bật.
    - Trong **transition** (1/2/3 đang countdown): luôn bật sensor cần cho mode đích.

    ``env_mode == auto_rule`` bắt buộc lấy NIR → **không** idle NIR ở mode thermal (ghi đè ``pause_nir_capture_when_thermal_only``).
    """
    pause_th = bool(cfg.get("pause_thermal_capture_when_imx_only", True))
    pause_nir = bool(cfg.get("pause_nir_capture_when_thermal_only", False))
    _em = str(cfg.get("env_mode", "off")).strip().lower()
    if _em == "auto_rule":
        pause_nir = False

    if mode_switch_pending == "imx":
        return pause_th, False
    if mode_switch_pending == "thermal":
        return False, pause_nir
    if mode_switch_pending == "fusion":
        return False, False

    if mode_switch_pending is None:
        if mode == "imx":
            return pause_th, False
        if mode == "thermal":
            return False, pause_nir
        if mode == "fusion":
            return False, False
    return False, False


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN (3 MODE: IMX, THERMAL, FUSION)
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    cfg = CONFIG
    _configure_opencv_runtime(cfg)

    if not _HAS_HARDWARE:
        print("Thiếu phần cứng. Chỉ chạy trên RPi + senxor + picamera2.")
        sys.exit(1)

    homography_path = str(cfg.get("homography_path", "homography.json"))
    if not os.path.isabs(homography_path):
        cands = [
            os.path.join(os.getcwd(), homography_path),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), homography_path),
        ]
        found = None
        for pth in cands:
            if os.path.exists(pth):
                found = pth
                break
        if found is None:
            print(f"Không tìm thấy {homography_path}. Chạy calibrate trước.")
            sys.exit(1)
        homography_path = found
    elif not os.path.exists(homography_path):
        print(f"Không tìm thấy {homography_path}. Chạy calibrate trước.")
        sys.exit(1)

    H, thermal_size, nir_size, fov_h_meta, fov_v_meta = load_homography(homography_path)
    nir_w, nir_h = nir_size
    display_size = (int(cfg.get("display_width", 800)), int(cfg.get("display_height", 480)))
    alpha = float(cfg.get("fusion_alpha", 0.55))
    exp_context = build_experiment_context(cfg)
    opt_cfg_base = resolve_optimization_profile(
        str(cfg.get("opt_profile", "baseline")),
        bool(cfg.get("opt_haze_preset", False)),
        not bool(cfg.get("opt_disable_profiles", False)),
    )
    opt_runtime_config = {
        "profile": opt_cfg_base["name"],
        "profiles_enabled": bool(not cfg.get("opt_disable_profiles", False)),
        "haze_preset": bool(cfg.get("opt_haze_preset", False)),
        "debug": bool(cfg.get("debug", False)),
        "s6_quality_gate_enabled": bool(not cfg.get("opt_disable_s6_quality_gate", False)),
        "fix_nir_motion_gating": bool(cfg.get("fix_nir_motion_gating", True)),
        "fix_thermal_detail_preserve": bool(cfg.get("fix_thermal_detail_preserve", True)),
        "env_mode": str(cfg.get("env_mode", "off")),
        "opencv_num_threads": int(cfg.get("opencv_num_threads", 4)),
        "opencv_disable_opencl": bool(cfg.get("opencv_disable_opencl", True)),
        "nir_capture_fps": float(cfg.get("nir_capture_fps", 60.0)),
        "nir_brightness_subsample": int(cfg.get("nir_brightness_subsample", 4)),
        "skip_thermal_reprocess_when_same_timestamp": bool(
            cfg.get("skip_thermal_reprocess_when_same_timestamp", True)
        ),
        "pause_thermal_capture_when_imx_only": bool(cfg.get("pause_thermal_capture_when_imx_only", True)),
        "pause_nir_capture_when_thermal_only": bool(cfg.get("pause_nir_capture_when_thermal_only", False)),
    }

    fov_h = float(fov_h_meta) if fov_h_meta is not None else DEFAULT_DISPLAY_FOV_H_DEG
    fov_v = float(fov_v_meta) if fov_v_meta is not None else DEFAULT_DISPLAY_FOV_V_DEG
    display_l_max = int(opt_cfg_base["display_l_max"])
    temporal_glare = DisplayTemporalGlareBlend(prev_weight=float(opt_cfg_base["temporal_prev_weight"]))
    skew_gate = StreamSkewQualityGate()

    thermal_proc = ThermalProcessor(
        warmup=40,
        anti_glare=True,
        thermal_high_pct=float(opt_cfg_base["thermal_high_pct"]),
        thermal_saturate_at=float(opt_cfg_base["thermal_saturate_at"]),
        thermal_gamma=float(opt_cfg_base["thermal_gamma"]),
        thermal_floor=float(opt_cfg_base["thermal_floor"]),
        thermal_fg_threshold=float(opt_cfg_base["thermal_fg_threshold"]),
        thermal_edge_strength=float(opt_cfg_base["thermal_edge_strength"]),
        thermal_agc_low_pct=float(opt_cfg_base["thermal_agc_low_pct"]),
        thermal_agc_high_pct=float(opt_cfg_base["thermal_agc_high_pct"]),
        detail_preserve_detect=bool(cfg.get("fix_thermal_detail_preserve", True)),
        detect_raw_mix=float(cfg.get("thermal_detect_raw_mix", 0.55)),
        detail_grad_threshold=float(cfg.get("thermal_detail_grad_threshold", 9.0)),
        detail_threshold_relax=float(cfg.get("thermal_detail_threshold_relax", 3.0)),
        detect_use_anti_glare=bool(cfg.get("thermal_detect_use_anti_glare", False)),
        thermal_3dnr_alpha=float(cfg.get("thermal_3dnr_alpha", 0.65)),
    )
    nir_enhancer = HybridNIREnhancer(
        proc_w=320,
        proc_h=240,
        patch_size=5,
        update_rate=8,
        detail_strength=float(cfg.get("nir_enhancer_detail_strength", 0.25)),
        clahe_clip_scale=float(cfg.get("nir_enhancer_clahe_clip_scale", 1.0)),
    )
    fusion = GradientThermalFusion(base_alpha=0.6, colormap=THERMAL_COLORMAP)
    fps_counter = FPSCounter()
    jerk_gate = JerkGate(
        diff_threshold=8.5,
        hold_frames=12,
        consecutive_frames=2,
        percentile=94.0,
        max_side=128,
        near_active_ratio=float(np.clip(float(cfg.get("nir_soft_motion_ratio", 0.62)), 0.2, 1.0)),
        lite_mode=bool(cfg.get("perf_jerk_lite", True)),
        perf_benchmark_samples=int(cfg.get("perf_benchmark_samples", 24)),
    )
    opsec = OpsecLocalOnly(enabled=False)
    shake = DisplayShakeReducerLite(mode="blend", blend_current_weight=0.50)

    phase1_e1_enabled = bool(cfg.get("feature_e1_enable", True))
    phase1_alert_enabled = bool(cfg.get("feature_alert_enable", False))
    phase1_detect_interval = int(max(1, int(cfg.get("feature_e1_detect_interval", 2))))
    e1_detector = ThermalAnomalyDetectorLite(
        local_kernel=int(cfg.get("feature_e1_local_kernel", 5)),
        z_thresh=float(cfg.get("feature_e1_z_thresh", 1.25)),
        heat_thresh=float(cfg.get("feature_e1_heat_thresh", 46.0)),
        min_area=int(cfg.get("feature_e1_min_area", 10)),
        raw_mix=float(cfg.get("feature_e1_raw_mix", 0.45)),
    )
    e1_defaults_snapshot = snapshot_e1_defaults(e1_detector)
    env_fallback = str(cfg.get("env_fallback_preset", "default"))
    env_controller = EnvPresetController(
        fallback=env_fallback if env_fallback in ENV_PRESETS else "default",
        hysteresis_frames=int(cfg.get("env_hysteresis_frames", 18)),
    )
    prev_env_stable: Optional[str] = None

    def _map_thermal_raw_to_display_xy(x_raw: float, y_raw: float, raw_w: int, raw_h: int) -> tuple[int, int]:
        if raw_w <= 1 or raw_h <= 1:
            return 0, 0
        # Thermal render path: rotate 180 rồi resize ra display.
        xr = (raw_w - 1) - float(x_raw)
        yr = (raw_h - 1) - float(y_raw)
        xd = int(round(np.clip(xr * (display_size[0] / float(raw_w - 1)), 0, display_size[0] - 1)))
        yd = int(round(np.clip(yr * (display_size[1] / float(raw_h - 1)), 0, display_size[1] - 1)))
        return xd, yd

    def _map_thermal_raw_to_display_xy_fusion(x_raw: float, y_raw: float, raw_w: int, raw_h: int) -> tuple[int, int]:
        if raw_w <= 1 or raw_h <= 1:
            return 0, 0
        sx = thermal_size[0] / float(raw_w)
        sy = thermal_size[1] / float(raw_h)
        xt = float(x_raw) * sx
        yt = float(y_raw) * sy
        xt = (thermal_size[0] - 1) - xt
        yt = (thermal_size[1] - 1) - yt
        pt = np.array([[[xt, yt]]], dtype=np.float32)
        try:
            pnir = cv.perspectiveTransform(pt, H)[0, 0]
            xn = float(np.clip(pnir[0], 0, nir_w - 1))
            yn = float(np.clip(pnir[1], 0, nir_h - 1))
        except cv.error:
            xn, yn = 0.0, 0.0
        xd = int(round(np.clip(xn * (display_size[0] / float(nir_w)), 0, display_size[0] - 1)))
        yd = int(round(np.clip(yn * (display_size[1] / float(nir_h)), 0, display_size[1] - 1)))
        return xd, yd

    # Camera threads
    thermal_cap = ThermalCapture(
        flip_h=bool(cfg.get("thermal_flip", False)),
        blur_ksize=int(cfg.get("thermal_capture_blur_ksize", 1)),
    )
    nir_cap = NIRCapture(
        no_rgb2bgr=not bool(cfg.get("nir_rgb2bgr", False)),
        nir_fps=float(cfg.get("nir_capture_fps", 60.0)),
    )

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
        homography_path=os.path.abspath(homography_path),
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
        extra={
            "experiment_context": exp_context,
            "optimization_config": opt_runtime_config,
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
    NIR_BRIGHT_RAW_ON = float(cfg.get("nir_schmitt_raw_on", 30.0))  # mean mượt >= này → raw / tắt enhance
    NIR_DIM_ENHANCE_ON = float(cfg.get("nir_schmitt_dim_on", 18.0))  # mean mượt <= này → enhance + grayscale
    NIR_B_EMA = 0.18           # làm mượt đo sáng trước khi so Schmitt
    SWITCH_FRAMES = 28

    thesis_metrics = ThesisRunMetrics(
        session_id=session_id,
        homography_path=os.path.abspath(homography_path),
        display_size=(display_size[0], display_size[1]),
        fusion_alpha=alpha,
        nir_schmitt_raw_on=NIR_BRIGHT_RAW_ON,
        nir_schmitt_dim_on=NIR_DIM_ENHANCE_ON,
        nir_b_ema_coef=NIR_B_EMA,
        experiment_context=exp_context,
        optimization_config=opt_runtime_config,
    )
    _opt_log(
        bool(cfg.get("debug", False)),
        (
            f"startup profile={opt_cfg_base['name']} haze={bool(cfg.get('opt_haze_preset', False))} "
            f"display_l_max={opt_cfg_base['display_l_max']} temporal_prev={opt_cfg_base['temporal_prev_weight']:.2f} "
            f"nir_fix={bool(cfg.get('fix_nir_motion_gating', True))} "
            f"thermal_fix={bool(cfg.get('fix_thermal_detail_preserve', True))} "
            f"th_cap_blur_k={int(cfg.get('thermal_capture_blur_ksize', 1))}"
        ),
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
    _prev_soft_motion = False
    _prev_th_detail_mode = False
    phase1_frame_idx = 0
    phase1_blobs: List[Dict[str, float]] = []
    sector_alert_text = ""
    sector_alert_hold = 0
    _perf_glare_old_ms_ema = 0.0
    _perf_glare_new_ms_ema = 0.0
    _perf_glare_samples = 0
    _perf_frame_idx = 0
    _perf_report_interval = int(max(60, int(cfg.get("perf_report_interval", 180))))
    env_stable_for_hud = env_controller.stable_name
    env_display_grade: Dict[str, float] = {}
    # Mục 4.C/E: mean độ sáng Schmitt — subsample>1 giảm chi phí (xem _nir_mean_brightness_bgr).
    nir_brightness_subsample = int(max(1, int(cfg.get("nir_brightness_subsample", 4))))
    # Mục 4: bỏ qua thermal_proc khi chưa có mẫu MI48 mới (so timestamp với lần xử lý trước).
    _last_thermal_mono_processed: Optional[float] = None

    print("=" * 60)
    print("  1 = IMX (NIR)   2 = Thermal (3DNR)   3 = Fusion (NIR+Thermal)")
    print("  NIR: raw khi sáng (Schmitt), grayscale+enhance khi tối")
    print("  R = raw/processed   S = Save   A = Auto   +/- = trọng số pha trộn (Fusion)   Q = Thoát")
    print("  (C4 + A6b blend + D2 tự động; HUD [JERK]/[GLARE] khi có)")
    print(f"  Metrics → JSON cạnh ảnh; báo cáo phiên: {metrics_dir}/session_*.json (khi Q)")
    print(
        f"  Tier S+: A1 bearing @tâm (FOV≈{fov_h:.1f}°×{fov_v:.1f}°) | temporal loá(profile={opt_cfg_base['name']})"
        f" | trần L≤{display_l_max} | S6 quality gate"
    )
    print(f"  Manifest: {metrics_dir}/manifest_{session_id}.json")
    _em0 = str(cfg.get("env_mode", "off")).strip().lower()
    print(f"  [ENV] mode={_em0} fallback={env_fallback} (hysteresis={int(cfg.get('env_hysteresis_frames', 18))} frames)")
    _pth = bool(cfg.get("pause_thermal_capture_when_imx_only", True))
    _pnir = bool(cfg.get("pause_nir_capture_when_thermal_only", False))
    print(
        f"  [CAP] idle thermal khi chỉ IMX={_pth} | idle NIR khi chỉ thermal={_pnir} "
        f"(auto_rule luôn cần NIR)"
    )
    print("=" * 60)

    # Áp idle ngay (mặc định mode=imx → MI48 có thể idle sau warmup 2s).
    _th0, _nir0 = _capture_hardware_idle_flags(mode, None, cfg)
    thermal_cap.set_idle(_th0)
    nir_cap.set_idle(_nir0)

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

        # Mode 1/2/3: tắt đọc sensor không dùng (SPI/ISP) để giảm tranh CPU với luồng chính — xem _capture_hardware_idle_flags.
        _th_idle, _nir_idle = _capture_hardware_idle_flags(mode, mode_switch_pending, cfg)
        thermal_cap.set_idle(_th_idle)
        nir_cap.set_idle(_nir_idle)

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
                # Chuyển mode: luôn chạy thermal một lần (transition); không dùng nhánh skip timestamp.
                thermal_denoised, thermal_enhanced, heat_map, fg_mask = thermal_proc.process(thermal_raw)
                _tm_sw = thermal_cap.get_last_mono()
                if _tm_sw is not None:
                    _last_thermal_mono_processed = _tm_sw
            elif t == "fusion" and nir_raw is not None and thermal_raw is not None:
                thermal_denoised, thermal_enhanced, heat_map, fg_mask = thermal_proc.process(thermal_raw)
                _tm_sw = thermal_cap.get_last_mono()
                if _tm_sw is not None:
                    _last_thermal_mono_processed = _tm_sw
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
            nir_brightness = _nir_mean_brightness_bgr(nir_raw, nir_brightness_subsample)
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
        elif mode == "thermal" and nir_raw is not None:
            # [ENV] EMA độ sáng NIR cho auto_rule khi chỉ xem thermal (không đụng detect thermal)
            _nbm = _nir_mean_brightness_bgr(nir_raw, nir_brightness_subsample)
            if nir_b_ema is None:
                nir_b_ema = _nbm
            else:
                nir_b_ema = (1.0 - NIR_B_EMA) * nir_b_ema + NIR_B_EMA * _nbm

        # ThermalProcessor: mỗi mẫu nhiệt mới (MI48) mới chạy 3DNR+BG một lần — tránh lặp trên cùng
        # timestamp khi vòng lặp chính nhanh hơn tốc độ thermal (~9 FPS vs NIR ~60 FPS).
        _thermal_mono = thermal_cap.get_last_mono()
        _thermal_reuse_outputs = False
        if mode in ("thermal", "fusion") and thermal_raw is not None:
            if (
                bool(cfg.get("skip_thermal_reprocess_when_same_timestamp", True))
                and _thermal_mono is not None
                and _last_thermal_mono_processed is not None
                and _thermal_mono == _last_thermal_mono_processed
                and thermal_denoised is not None
            ):
                _thermal_reuse_outputs = True
            else:
                thermal_denoised, thermal_enhanced, heat_map, fg_mask = thermal_proc.process(thermal_raw)
                if _thermal_mono is not None:
                    _last_thermal_mono_processed = _thermal_mono

        jerk_gate.update(nir_raw)
        soft_motion_active = bool(bool(cfg.get("fix_nir_motion_gating", True)) and jerk_gate.near_active and not jerk_gate.active)
        if soft_motion_active != _prev_soft_motion:
            _opt_log(
                bool(cfg.get("debug", False)),
                (
                    f"NIR soft-motion gate={'ON' if soft_motion_active else 'OFF'} "
                    f"(score={jerk_gate.last_score:.2f}, thr={jerk_gate.diff_threshold:.2f})"
                ),
            )
            _prev_soft_motion = soft_motion_active
        if bool(getattr(thermal_proc, "last_detail_mode_active", False)) != _prev_th_detail_mode:
            _opt_log(
                bool(cfg.get("debug", False)),
                (
                    f"Thermal detail-detect mode={'ON' if thermal_proc.last_detail_mode_active else 'OFF'} "
                    f"(grad_mean={thermal_proc.last_detail_grad_mean:.2f})"
                ),
            )
            _prev_th_detail_mode = bool(getattr(thermal_proc, "last_detail_mode_active", False))

        # [ENV] Preset apply logic
        # Purpose: map environment labels to safe runtime parameters (reuse opt_cfg_base + overrides).
        # Notes: E1/thermal cập nhật khi preset đổi; opt_cfg_runtime dùng cho NIR glare + S4 mỗi frame.
        opt_cfg_runtime = dict(opt_cfg_base)
        env_display_grade = {}
        _em = str(cfg.get("env_mode", "off")).strip().lower()
        if _em == "off":
            env_stable_for_hud = "default"
        else:
            env_fb = str(cfg.get("env_fallback_preset", "default"))
            if env_fb not in ENV_PRESETS:
                env_fb = "default"
            desired = env_fb
            if _em == "manual":
                mp = str(cfg.get("env_manual_preset", "default"))
                desired = mp if mp in ENV_PRESETS else env_fb
            elif _em == "auto_rule":
                if nir_raw is not None:
                    _is_night = bool(nir_enhancer.is_night_mode) if mode in ("imx", "fusion") else (
                        nir_b_ema is not None and nir_b_ema < 22.0
                    )
                    _std = _nir_gray_std_quick(nir_raw)
                    _need_g, _gl_early = nir_glare_eval(
                        nir_raw,
                        high_pct=float(opt_cfg_base["nir_high_pct"]),
                        saturate_at=float(opt_cfg_base["nir_saturate_at"]),
                        use_fast=True,
                    )
                    tags = infer_env_tags_auto_rule(
                        nir_b_ema=nir_b_ema,
                        is_night_mode=_is_night,
                        nir_gray_std=_std,
                        glare_nir=_gl_early,
                        haze_config_on=bool(cfg.get("opt_haze_preset", False)),
                        std_low=float(cfg.get("env_auto_nir_gray_std_low", 20.0)),
                        std_high=float(cfg.get("env_auto_nir_gray_std_high", 52.0)),
                    )
                    desired = select_env_preset_from_tags(tags)
                else:
                    desired = env_fb
            stable = env_controller.update(desired if desired in ENV_PRESETS else env_fb)
            opt_cfg_runtime = merge_opt_cfg_with_preset(opt_cfg_base, stable)
            env_stable_for_hud = stable
            _pobj = ENV_PRESETS.get(stable) or ENV_PRESETS["default"]
            env_display_grade = dict(_pobj.get("display_grade") or {})
            if stable != prev_env_stable:
                print(f"[ENV] stable_preset = {stable}")
                print(_format_env_apply_log(stable, _pobj))
                thermal_proc.apply_env_runtime(
                    opt_cfg_runtime,
                    _pobj.get("thermal_extra"),
                    _pobj.get("thermal_3dnr_alpha"),
                )
                restore_e1_from_snapshot(e1_detector, e1_defaults_snapshot)
                apply_e1_overrides(e1_detector, _pobj.get("e1_overrides") or {})
                temporal_glare.prev_weight = float(opt_cfg_runtime["temporal_prev_weight"])
                prev_env_stable = stable

        if mode in ("thermal", "fusion") and thermal_raw is not None and phase1_e1_enabled:
            # E1 gắn với mẫu thermal: không tăng idx / không chạy detect khi tái dùng output (cùng timestamp).
            if not _thermal_reuse_outputs:
                phase1_frame_idx += 1
                run_slow_path = (phase1_frame_idx % phase1_detect_interval) == 0
                if run_slow_path:
                    _, phase1_blobs = e1_detector.process(
                        thermal_raw,
                        thermal_denoised,
                        heat_map,
                        fg_mask,
                        jerk_active=jerk_gate.active,
                    )
            if phase1_alert_enabled:
                if phase1_blobs:
                    b_best = phase1_blobs[0]
                    th_h, th_w = thermal_raw.shape[:2]
                    if mode == "fusion":
                        tx, ty = _map_thermal_raw_to_display_xy_fusion(
                            float(b_best["cx"]),
                            float(b_best["cy"]),
                            th_w,
                            th_h,
                        )
                    else:
                        tx, ty = _map_thermal_raw_to_display_xy(
                            float(b_best["cx"]),
                            float(b_best["cy"]),
                            th_w,
                            th_h,
                        )
                    bh_t, _bv_t = bearing_hv_deg_from_uv(
                        float(tx),
                        float(ty),
                        float(display_size[0]),
                        float(display_size[1]),
                        float(fov_h),
                        float(fov_v),
                    )
                    sector_alert_text = sector_from_bearing_deg(
                        bh_t,
                        center_half_width_deg=float(cfg.get("feature_sector_center_deg", 6.0)),
                    )
                    sector_alert_hold = int(max(0, int(cfg.get("feature_sector_hold_frames", 8))))
                elif sector_alert_hold > 0:
                    sector_alert_hold -= 1
                else:
                    sector_alert_text = ""
        else:
            if mode == "imx":
                phase1_blobs = []
                sector_alert_text = ""
                sector_alert_hold = 0

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
            glare_nir = False
            if thermal_proc.anti_glare_enabled and nir_ok_glare:
                t_ng0 = time.perf_counter()
                nir_need_compress, glare_nir = nir_glare_eval(
                    src_in,
                    high_pct=float(opt_cfg_runtime["nir_high_pct"]),
                    saturate_at=float(opt_cfg_runtime["nir_saturate_at"]),
                    use_fast=bool(cfg.get("perf_glare_lite", True)),
                )
                t_ng1 = time.perf_counter()
                _perf_glare_new_ms_ema = (t_ng1 - t_ng0) * 1000.0 if _perf_glare_new_ms_ema <= 0 else (
                    0.88 * _perf_glare_new_ms_ema + 0.12 * (t_ng1 - t_ng0) * 1000.0
                )
                if bool(cfg.get("perf_glare_lite", True)) and _perf_glare_samples < int(cfg.get("perf_benchmark_samples", 24)):
                    g_bench = _nir_gray_for_stats(src_in)
                    t_og0 = time.perf_counter()
                    _ = _nir_glare_metrics_from_gray(
                        g_bench,
                        high_pct=float(opt_cfg_runtime["nir_high_pct"]),
                        saturate_at=float(opt_cfg_runtime["nir_saturate_at"]),
                        use_fast=False,
                    )
                    t_og1 = time.perf_counter()
                    _old = (t_og1 - t_og0) * 1000.0
                    _perf_glare_old_ms_ema = _old if _perf_glare_old_ms_ema <= 0 else (0.88 * _perf_glare_old_ms_ema + 0.12 * _old)
                    _perf_glare_samples += 1
                if nir_need_compress:
                    src = nir_anti_glare_bgr(
                        src_in,
                        high_pct=float(opt_cfg_runtime["nir_high_pct"]),
                        saturate_at=float(opt_cfg_runtime["nir_saturate_at"]),
                        gamma=float(opt_cfg_runtime["nir_gamma"]),
                    )
            else:
                nir_need_compress = False
            out = cv.resize(src, display_size)
        elif mode == "thermal":
            # [FIX] Thermal blur issue
            # Based on: fusion_advanced.py mode 7
            # Change: removed over-smoothing, simplified pipeline
            # Behavior: giữ thermal rõ nét hơn bằng cách pha nhẹ raw vào 3DNR + AGC nhẹ trước khi tô màu.
            if show_raw.get("thermal"):
                thermal_vis = thermal_raw
            else:
                if thermal_denoised is not None:
                    thermal_vis = thermal_denoised
                    if thermal_raw is not None:
                        mix_raw = float(np.clip(float(cfg.get("thermal_display_raw_mix", 0.28)), 0.0, 1.0))
                        if mix_raw > 1e-6:
                            mixed = cv.addWeighted(
                                thermal_raw.astype(np.float32),
                                mix_raw,
                                thermal_denoised.astype(np.float32),
                                1.0 - mix_raw,
                                0,
                            )
                            thermal_vis = np.clip(mixed, 0, 255).astype(np.uint8)
                else:
                    thermal_vis = thermal_enhanced if thermal_enhanced is not None else thermal_raw
                lo_t = float(cfg.get("thermal_display_agc_low_pct", 1.5))
                hi_t = float(cfg.get("thermal_display_agc_high_pct", 99.2))
                if hi_t > lo_t:
                    thermal_vis = thermal_agc(thermal_vis, low_pct=lo_t, high_pct=hi_t)
            # [IMPROVEMENT] Proper heatmap mapping
            # Change: replace red/blue mapping with full colormap
            cm = cv.applyColorMap(thermal_vis, THERMAL_COLORMAP)
            cm = cv.resize(cm, thermal_size)
            cm = cv.rotate(cm, cv.ROTATE_180)
            out = cv.resize(cm, display_size)
            if not thermal_proc.is_ready:
                cv.putText(out, f"Warming up... {thermal_proc.warmup_pct}%",
                           (10, 48), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            if thermal_proc.anti_glare_enabled and thermal_denoised is not None:
                glare_th = thermal_glare_hud_should_apply(
                    thermal_denoised,
                    saturate_at=min(254.0, float(opt_cfg_runtime["thermal_saturate_at"]) + 12.0),
                )

        else:  # fusion (NIR bg + heat map overlay chỉ ở vùng có vật nóng)
            use_raw = show_raw.get("imx") or nir_use_raw_auto
            nir_src = nir_raw if use_raw else (nir_enhanced if nir_enhanced is not None else nir_raw)
            nir_ok_glare = nir_glare_allowed(
                nir_brightness, NIR_BRIGHT_RAW_ON, nir_enhancer.is_night_mode, display_raw=use_raw
            )
            nir_in = nir_src
            glare_nir = False
            if thermal_proc.anti_glare_enabled and nir_ok_glare:
                t_ng0 = time.perf_counter()
                nir_need_compress, glare_nir = nir_glare_eval(
                    nir_in,
                    high_pct=float(opt_cfg_runtime["nir_high_pct"]),
                    saturate_at=float(opt_cfg_runtime["nir_saturate_at"]),
                    use_fast=bool(cfg.get("perf_glare_lite", True)),
                )
                t_ng1 = time.perf_counter()
                _perf_glare_new_ms_ema = (t_ng1 - t_ng0) * 1000.0 if _perf_glare_new_ms_ema <= 0 else (
                    0.88 * _perf_glare_new_ms_ema + 0.12 * (t_ng1 - t_ng0) * 1000.0
                )
                if bool(cfg.get("perf_glare_lite", True)) and _perf_glare_samples < int(cfg.get("perf_benchmark_samples", 24)):
                    g_bench = _nir_gray_for_stats(nir_in)
                    t_og0 = time.perf_counter()
                    _ = _nir_glare_metrics_from_gray(
                        g_bench,
                        high_pct=float(opt_cfg_runtime["nir_high_pct"]),
                        saturate_at=float(opt_cfg_runtime["nir_saturate_at"]),
                        use_fast=False,
                    )
                    t_og1 = time.perf_counter()
                    _old = (t_og1 - t_og0) * 1000.0
                    _perf_glare_old_ms_ema = _old if _perf_glare_old_ms_ema <= 0 else (0.88 * _perf_glare_old_ms_ema + 0.12 * _old)
                    _perf_glare_samples += 1
                if nir_need_compress:
                    nir_src = nir_anti_glare_bgr(
                        nir_in,
                        high_pct=float(opt_cfg_runtime["nir_high_pct"]),
                        saturate_at=float(opt_cfg_runtime["nir_saturate_at"]),
                        gamma=float(opt_cfg_runtime["nir_gamma"]),
                    )
            else:
                nir_need_compress = False
            if thermal_proc.anti_glare_enabled and thermal_denoised is not None:
                glare_th = thermal_glare_hud_should_apply(
                    thermal_denoised,
                    saturate_at=min(254.0, float(opt_cfg_runtime["thermal_saturate_at"]) + 12.0),
                )
            nir_resized = cv.resize(nir_src, (nir_w, nir_h))
            alpha_runtime = min(1.0, alpha + float(opt_cfg_runtime["fusion_alpha_boost"]))
            if heat_map is not None and fg_mask is not None and np.any(fg_mask > 0):
                hm_r = cv.resize(heat_map, thermal_size, interpolation=cv.INTER_LINEAR)
                hm_r = cv.rotate(hm_r, cv.ROTATE_180)
                # [IMPROVEMENT] Proper heatmap mapping
                # Change: replace red/blue mapping with full colormap
                lo_hm = float(cfg.get("thermal_heatmap_agc_low_pct", 1.0))
                hi_hm = float(cfg.get("thermal_heatmap_agc_high_pct", 99.5))
                hm_vis = thermal_agc(hm_r, low_pct=lo_hm, high_pct=hi_hm) if hi_hm > lo_hm else hm_r
                hm_color = cv.applyColorMap(hm_vis, THERMAL_COLORMAP)
                hm_warped = cv.warpPerspective(hm_color, H, (nir_w, nir_h))
                fg_r = cv.resize(fg_mask, thermal_size)
                fg_r = cv.rotate(fg_r, cv.ROTATE_180)
                fg_w = cv.warpPerspective(fg_r, H, (nir_w, nir_h), flags=cv.INTER_NEAREST)
                mask_f = fg_w.astype(np.float32) / 255.0
                mask_f = cv.GaussianBlur(mask_f, (7, 7), 2.0)
                m3 = np.stack([mask_f] * 3, axis=-1) * alpha_runtime
                blended = nir_resized.astype(np.float32) * (1 - m3) + hm_warped.astype(np.float32) * m3
                out = cv.resize(np.clip(blended, 0, 255).astype(np.uint8), display_size)
            else:
                out = cv.resize(nir_resized, display_size)
            if not thermal_proc.is_ready:
                cv.putText(out, f"BG warming... {thermal_proc.warmup_pct}%",
                           (10, 48), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # [FIX] NIR ghosting under motion
        # Cause:
        #   Temporal glare blend + shake blend dùng frame trước; gate chỉ dựa jerk mạnh nên jerk nhẹ vẫn bị bóng ma.
        # Fix:
        #   Thêm soft-motion gate từ JerkGate.score để tắt/reset temporal sớm khi motion nhẹ nhưng liên tục.
        # Behavior:
        #   Đứng yên vẫn giữ anti-glare temporal; đang rung/pan nhẹ thì chuyển về spatial-only ngay, giảm ghost.
        if mode in ("imx", "fusion"):
            _apply_temporal = bool(nir_need_compress and not jerk_gate.active)
            _reset_temporal = bool(jerk_gate.active)
            if bool(cfg.get("fix_nir_motion_gating", True)) and soft_motion_active:
                _apply_temporal = False
                _reset_temporal = True
            out = temporal_glare.process(out, _apply_temporal, reset=_reset_temporal)
        out = shake.process(out, jerk_active=jerk_gate.active, soft_motion_active=soft_motion_active)
        # [ENV] display path only — sau shake, trước S4 cap (không ảnh hưởng detect)
        if env_display_grade:
            out = display_color_grade_bgr(out, **env_display_grade)
        _l_cap = int(opt_cfg_runtime["display_l_max"])
        if glare_nir or glare_th:
            _l_cap = min(_l_cap, int(opt_cfg_runtime["display_l_max_when_glare"]))
        out = display_luminance_cap_bgr(out, l_max=_l_cap)

        _perf_frame_idx += 1
        if _perf_frame_idx % _perf_report_interval == 0:
            glare_saved_ms = 0.0
            if _perf_glare_samples > 0:
                glare_saved_ms = max(0.0, _perf_glare_old_ms_ema - _perf_glare_new_ms_ema)
            jerk_saved_ms = float(jerk_gate.get_saved_ms_estimate())
            saved_ms_total = glare_saved_ms + jerk_saved_ms
            fps_after = max(0.1, float(fps))
            fps_before_est = 1.0 / ((1.0 / fps_after) + (saved_ms_total / 1000.0))
            print(
                "[PERF] FPS est before="
                f"{fps_before_est:.2f} after={fps_after:.2f} "
                f"(saved~{saved_ms_total:.3f}ms/frame; glare~{glare_saved_ms:.3f}ms jerk~{jerk_saved_ms:.3f}ms)"
            )

        skew_ms = None
        _tm = thermal_cap.get_last_mono()
        _nm = nir_cap.get_last_mono()
        # S6: chênh lệch thời gian lấy mẫu NIR vs thermal (ms) — đánh giá đồng bộ hai luồng (fusion/độ trễ cảm nhận).
        if _tm is not None and _nm is not None:
            skew_ms = abs(_tm - _nm) * 1000.0
        if bool(cfg.get("opt_disable_s6_quality_gate", False)):
            skew_quality, skew_ema_ms, skew_state_changed = "GOOD", None, False
        else:
            skew_quality, skew_ema_ms, skew_state_changed = skew_gate.update(skew_ms)
            if skew_state_changed:
                _opt_log(
                    bool(cfg.get("debug", False)),
                    f"S6 quality -> {skew_quality} (ema={skew_ema_ms:.2f}ms, instant={0.0 if skew_ms is None else skew_ms:.2f}ms)",
                )
        _wd, _hd = int(display_size[0]), int(display_size[1])
        _u = float(np.clip(a1_probe_xy[0], 0, max(0, _wd - 1)))
        _v = float(np.clip(a1_probe_xy[1], 0, max(0, _hd - 1)))
        bear_h, bear_v = bearing_hv_deg_from_uv(_u, _v, _wd, _hd, fov_h, fov_v)

        if mode in ("thermal", "fusion") and thermal_raw is not None:
            th_h, th_w = thermal_raw.shape[:2]
            for db in phase1_blobs:
                if mode == "fusion":
                    tx, ty = _map_thermal_raw_to_display_xy_fusion(
                        float(db["cx"]),
                        float(db["cy"]),
                        th_w,
                        th_h,
                    )
                else:
                    tx, ty = _map_thermal_raw_to_display_xy(
                        float(db["cx"]),
                        float(db["cy"]),
                        th_w,
                        th_h,
                    )
                col = (0, 220, 255)
                cv.circle(out, (tx, ty), 4, col, 1)
                lbl = f"D {float(db.get('score', 0.0)):.2f}"
                cv.putText(out, lbl, (tx + 6, ty - 4), cv.FONT_HERSHEY_SIMPLEX, 0.34, col, 1)

        # [FEATURE] Sector / wedge alert (HUD)
        # Input:
        #   blob detect hiện tại (frame-level) + pixel->bearing
        # Output:
        #   cảnh báo LEFT/CENTER/RIGHT
        # Notes:
        #   không dùng tracking; giữ nhịp cảnh báo bằng hold frame để tránh flicker.
        if mode in ("thermal", "fusion") and phase1_alert_enabled and sector_alert_text:
            cv.putText(
                out,
                f"[ALERT {sector_alert_text}]",
                (8, display_size[1] - 14),
                cv.FONT_HERSHEY_SIMPLEX,
                0.50,
                (60, 220, 255),
                1,
            )

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
            skew_quality=skew_quality,
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
            if soft_motion_active:
                tag_list.append("SOFT_MOTION")
            if bool(cfg.get("opt_haze_preset", False)):
                tag_list.append("HAZE")
            if str(cfg.get("env_mode", "off")).strip().lower() != "off":
                tag_list.append(f"ENV_{env_stable_for_hud}")
            if skew_quality != "GOOD":
                tag_list.append(f"SYNC_{skew_quality}")
            if phase1_alert_enabled and sector_alert_text and mode in ("thermal", "fusion"):
                tag_list.append(f"SECTOR_{sector_alert_text}")
            if mode in ("thermal", "fusion") and phase1_e1_enabled:
                tag_list.append(f"DET{len(phase1_blobs)}")
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
                skew_quality=skew_quality,
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
        if soft_motion_active:
            tag_parts.append("MOTION")
        if bool(cfg.get("opt_haze_preset", False)):
            tag_parts.append("HAZE")
        if str(cfg.get("env_mode", "off")).strip().lower() != "off":
            tag_parts.append(f"ENV:{env_stable_for_hud}")
        if tag_parts:
            cv.putText(out, "[" + " ".join(tag_parts) + "]", (8, hud_y), cv.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1)
        if skew_quality != "GOOD":
            cv.putText(out, f"[SYNC {skew_quality}]", (8, hud_y + 20), cv.FONT_HERSHEY_SIMPLEX, 0.42, (120, 210, 255), 1)

        _hh, _ww = out.shape[:2]
        _cx, _cy = _ww // 2, _hh // 2
        cv.line(out, (_cx - 18, _cy), (_cx + 18, _cy), (0, 220, 255), 1)
        cv.line(out, (_cx, _cy - 18), (_cx, _cy + 18), (0, 220, 255), 1)
        _pu, _pv = int(np.clip(a1_probe_xy[0], 0, _ww - 1)), int(np.clip(a1_probe_xy[1], 0, _hh - 1))
        cv.circle(out, (_pu, _pv), 5, (0, 255, 255), 1)
        _bh = f"{bear_h:+.2f}"
        _bv = f"{bear_v:+.2f}"
        if skew_ms is not None:
            _sk_disp = f"{skew_ms:.2f} ms/{skew_quality}"
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
