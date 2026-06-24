"""
Central configuration for the fusion runtime.

``CONFIG`` is merged with CLI flags and environment presets in ``main.py``.
Paths such as ``HOMOGRAPHY_PATH`` tie into ``hardware.load_homography``.
Optimization profiles from :func:`resolve_optimization_profile` feed display
and sensor tuning used by NIR, thermal, and fusion stages.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any, Dict

# ── Asset paths ───────────────────────────────────────────────────────────────
HOMOGRAPHY_PATH: Path = Path(__file__).parent / "assets" / "homography.json"


# ── Display / Tier-S constants ────────────────────────────────────────────────
DEFAULT_DISPLAY_FOV_H_DEG: float = 52.0
DEFAULT_DISPLAY_FOV_V_DEG: float = 40.0
TIER_S_DISPLAY_L_MAX: int = 240
TIER_S_DISPLAY_L_MAX_WHEN_GLARE: int = 208
TIER_S_GLARE_TEMPORAL_PREV_WEIGHT: float = 0.42


# ── Valid opt_overrides keys (used by env_presets.py for validation) ──────────
_VALID_OPT_OVERRIDES_KEYS: frozenset = frozenset({
    "nir_high_pct",
    "nir_saturate_at",
    "nir_gamma",
    "temporal_prev_weight",
    "thermal_floor",
    "thermal_fg_threshold",
    "thermal_fg_max_ratio",
    "fusion_alpha_boost",
    "display_l_max",
    "display_l_max_when_glare",
    # NIR bucket-dispatch overrides
    "nir_enhancer_clahe_clip_scale",
    "nir_enhancer_detail_strength",
    "fog_dehaze_omega",
    "rain_median_frames",
    # Thermal bilateral filter — wired via ThermalProcessor.update_runtime_params()
    "thermal_bilateral_d",
    "thermal_bilateral_sigma_color",
    "thermal_bilateral_sigma_space",
    # profile keys that may come through
    "thermal_high_pct",
    "thermal_saturate_at",
    "thermal_gamma",
    "thermal_edge_strength",
    "thermal_agc_low_pct",
    "thermal_agc_high_pct",
})


# ===== DEFAULT OPTIMIZED CONFIG =====
CONFIG: Dict[str, Any] = {
    "homography_path": str(HOMOGRAPHY_PATH),
    "display_width": 800,
    "display_height": 480,
    "fusion_alpha": 0.55,
    "thermal_flip": False,
    "nir_rgb2bgr": False,
    "debug": False,
    "opt_profile": "handheld_pan",   # baseline | static_scan | handheld_pan | high_glare
    "opt_disable_profiles": False,
    "opt_haze_preset": False,
    # When True, merge :data:`RPI_THROUGHPUT_MAX_DEFAULTS` in ``main`` (after CLI): full RPi throughput profile.
    "rpi_throughput_max": True,
    # Jerk / ML gray resize max side (64–128; lower = faster, coarser).
    "framecache_small_max_side": 128,
    # HybridNIR working size (must match precomputed 320×240 from FrameCache when using defaults).
    "nir_enhancer_proc_w": 320,
    "nir_enhancer_proc_h": 240,
    # Odd min/max filter half-width in HybridNIR (3 is cheaper than 5).
    "nir_enhancer_patch_size": 5,
    # A1: skip bearing, crosshair, and right HUD bearing text when False.
    "hud_bearing_enabled": True,
    # UTC clock on HUD top bar. Set False to disable for A/B parity checks vs pre-refactor builds.
    "hud_utc_clock_enabled": True,
    # Processing profile: "quality" (full pipeline) | "throughput" | "raw" (minimal sensor passthrough).
    # "throughput" maps to RPI_THROUGHPUT_MAX_DEFAULTS (apply_profile enforces regression lock).
    # "raw" = no CLAHE/HybridNIR/fusion blend; in fusion mode forces single-sensor view + toast.
    # Note: profile "raw" != show_raw (r/R keyboard per-mode preview flag) — do not conflate.
    "display_profile": "quality",
    # Per-profile verified flag: False → HUD shows [UNVERIFIED] badge until owner profiles on RPi.
    "display_profile_verified": {"quality": True, "throughput": True, "raw": False},
    # Splash frames when switching OPT/THM/FUS (streams settle + feedback). Clamped 8..120 in main.
    "mode_switch_prep_frames": 32,
    # Network-derived coarse location (P2.5, default OFF for OPSEC).
    "display_network_location": False,
    "log_network_location": False,
    "netloc_max_age_s": 600.0,
    "netloc_provider": "ip",   # "ip" | "wifi" | "off"
    # Video capture settings (P1).
    "capture_video_codec": "mp4v",
    "capture_video_max_mb": 2048,
    "capture_video_on_disk_full": "stop",   # "stop" | (future: "rotate")
    # Touch / GPIO controls (P1).
    "controls_touch_enabled": True,
    "controls_gpio_enabled": False,
    "controls_chrome_autofade_s": 5.0,
    # ``full`` = LAB grade + L cap; ``luma_only`` = single L cap (Tier S4) without color grade.
    "display_grade_mode": "full",  # full | luma_only
    # When True, apply per-pixel anti-glare on thermal AGC *display* path. False = 3DNR + AGC + edge only.
    "thermal_display_glare_suppression": True,
    # NIR: replace C/D/F with cheaper buckets (B/A/E with shorter median). Night classes unchanged.
    "nir_optical_lite": False,
    # When ``nir_optical_lite`` and bucket E, temporal median uses this many frames (2 is cheaper than 3).
    "rain_median_frames_lite": 2,
    "fix_nir_motion_gating": True,
    "nir_soft_motion_ratio": 0.62,
    # False: skip per-frame Sobel detail branch in ``ThermalProcessor.process`` (faster on RPi).
    "fix_thermal_detail_preserve": False,
    "thermal_detect_raw_mix": 0.55,
    "thermal_detail_grad_threshold": 9.0,
    "thermal_detail_threshold_relax": 3.0,
    "thermal_capture_blur_ksize": 1,  # 1 disables Gaussian blur on thermal capture
    # Thermal quality (0.28 → 0.10: less raw noise reinjected into 3DNR display path)
    "thermal_display_raw_mix": 0.10,
    # Thermal display edge enhancement (after AGC, before colormap in thermal/fusion)
    # 0.0 = off. 0.35 over-sharpened sensor noise on the 80×62 source frame.
    # Reduced to 0.25: cleaner edge pop without amplifying noise.
    "thermal_display_edge_strength": 0.18,
    "thermal_detect_use_anti_glare": False,  # default off; glare_heavy preset may enable
    "nir_enhancer_detail_strength": 0.16,
    "nir_enhancer_clahe_clip_scale": 1.0,  # scales CLAHE clipLimit (very_dark/dark/medium)
    # Guided filter: off by default — adds ~3–5 ms/frame in fusion when enabled.
    # Enable only when opencv-contrib (ximgproc) is installed and cross-sensor denoising
    # is needed. When False, main.py passes thermal_guide=None to nir_enhancer.process().
    "nir_guided_filter_enable": False,
    "thermal_display_agc_low_pct": 1.5,  # boosts thermal display contrast
    "thermal_display_agc_high_pct": 99.2,
    "perf_glare_lite": True,
    "perf_jerk_lite": True,
    "perf_report_interval": 180,
    "perf_benchmark_samples": 24,
    # --- Optimizations safe to enable immediately ---
    "opencv_num_threads": 4,
    "opencv_disable_opencl": True,
    "nir_capture_fps": 60.0,
    "nir_brightness_subsample": 4,
    "skip_thermal_reprocess_when_same_timestamp": True,
    "pause_thermal_capture_when_imx_only": True,
    "pause_nir_capture_when_thermal_only": False,
    # Thermal blob / sector alert features (detection-only HUD)
    "feature_e1_enable": True,
    "feature_e1_detect_interval": 2,
    "feature_e1_local_kernel": 5,
    "feature_e1_z_thresh": 1.25,
    "feature_e1_heat_thresh": 46.0,
    "feature_e1_min_area": 10,
    "feature_e1_raw_mix": 0.45,
    "feature_alert_enable": False,
    "feature_sector_center_deg": 6.0,
    "feature_sector_hold_frames": 8,
    # ── Lean pipeline flags (CLI merges here; defaults preserve full pipeline) ──
    # Overridden by --pipeline lean / --no-lk / --no-e1 / --no-mad.
    "lean_disable_lk": False,   # disables LK optical flow (~2 ms/frame; ML features only)
    "lean_disable_e1": False,   # disables E1 anomaly detector (Laplacian z-score blobs)
    "lean_disable_mad": False,  # disables MAD anomaly detector (temporal integrator)
    # [ENV] Adaptive environment presets
    # auto_rule drives bucket dispatch and logs ENV class into session JSON; requires ML_INFERENCE_ENABLED or rule layer.
    "env_mode": "auto_rule",  # off | manual | auto_rule
    "env_manual_preset": "default",
    "env_fallback_preset": "default",
    "env_hysteresis_frames": 18,
    "env_auto_nir_gray_std_low": 20.0,
    "env_auto_nir_gray_std_high": 52.0,
    "nir_schmitt_raw_on": 30.0,
    "nir_schmitt_dim_on": 18.0,
    # Thermal temporal smoothing. 0.78 was too slow to respond at 9 fps (time-constant ~500 ms).
    # ``legacy/py/final_fusion.py`` uses 0.65; slightly higher values trade smoothness vs lag.
    "thermal_3dnr_alpha": 0.65,
    # After 3DNR, only the enhanced branch (AGC/EE) — not used for detect_src / heat_map / fg_mask.
    # Off on RPi: ``bilateralFilter`` on 80×62 is a measurable chunk of ``thermal_proc``.
    "thermal_bilateral_display_enable": False,
    "thermal_bilateral_d": 5,
    "thermal_bilateral_sigma_color": 15.0,
    "thermal_bilateral_sigma_space": 5.0,
    # Foreground mask: if morph-cleaned FG covers more than this fraction of 80×62, mask is zeroed.
    # ``legacy/py/final_fusion.py`` uses 0.5.
    "thermal_fg_max_ratio": 0.5,
    # False-color LUT for thermal + fusion heatmap: numpy index into a precomputed table (no applyColorMap).
    # 32|64|128|256 — lower reduces work slightly and can add mild banding.
    "thermal_colormap_levels": 256,
    # Heatmap upscale: linear is cheaper than cubic (fusion overlay).
    "fusion_heatmap_interpolation": "linear",  # "linear" | "cubic" | "nearest"
    # ``legacy/py/final_fusion.py``: GaussianBlur(7, 7, 2.0) on warped fg mask.
    "fusion_mask_blur_ksize": 7,
    "fusion_mask_blur_sigma": 2.0,
    # Skip a second edge-enhance on the heatmap before colormap (saves a Laplacian on small map).
    "fusion_skip_heatmap_display_edge_enhance": True,
    # "fg_mask" matches ``legacy/py/final_fusion.py`` overlay. "gradient" heat-weights alpha without binary gating.
    "fusion_overlay_mode": "fg_mask",
    # In gradient mode: scales warped heat norm before multiplying user alpha (legacy default 0.6).
    "fusion_thermal_base_alpha": 0.6,
    # Warp/blend NIR space at (scale × nir_w/h) then upscale to display — fewer warp pixels, lower variance vs full res.
    # 1.0 = legacy full-res warp; 0.5–0.85 typical on RPi when set via throughput defaults.
    "fusion_warp_work_scale": 1.0,
    # HybridNIREnhancer: recompute dark/bright/weight only every N frames.
    "nir_hybrid_update_rate": 10,
    # Stabilization before Tier-S grading. ``off`` = passthrough (fastest; no handheld smoothing).
    "display_shake_mode": "off",  # off | blend | shift
    # Tier S3 NIR highlight temporal IIR. When False, skips one full-frame addWeighted when active.
    "display_temporal_glare_enable": True,
    # ENV auto_rule: run less often to reduce load alongside glare/jerk (legacy default was 10).
    "env_classification_interval": 15,
    # ── ML / ENV classifier (wired from main.py after offline schema validation) ──
    # Heavy: every-N-frame feature JSONL on SD; leave off unless collecting training features on-device.
    "ML_LOG_ENABLED": False,
    # BRISQUE no-reference score per optical bucket logged to session_*.json (1/30 frames).
    # Requires models/brisque/brisque_model_live.yml + brisque_range_live.yml (rsync or OpenCV extras).
    "iqa_logging_enabled": False,
    "ML_LOG_DIR": "logs/ml",
    "ML_LOG_INTERVAL": 5,    # log one feature record every N frames
    # Default model path; override with env ML_MODEL_PATH. Trained with TimeSeriesSplit CV + isotonic calibration.
    "ML_MODEL_PATH": "models/production/env_classifier.joblib",
    # ── ML runtime inference (drives auto_rule with rule fallback when conf < threshold) ──
    # Set ML_MODEL_PATH to a .joblib on the Pi (file is gitignored — rsync separately).
    # Env override: ML_INFERENCE_ENABLED=1|0, ML_MODEL_PATH=...
    "ML_INFERENCE_ENABLED": True,
    "ML_INFERENCE_INTERVAL": 15,  # run inference every N frames (decoupled from ML_LOG_INTERVAL)
    # ── ML routing (NIR optical bucket dispatch) ─────────────────────────────
    # Top-1: docs/tables/ml/ML_GATE_RATIONALE.md — threshold_sweep on from_logs_test: τ=0.62 → macro F1 (night) 0.9835; abstain 12.8%. Optional 0.65: F1 0.9866, abstain 14.9%.
    "ml_confidence_threshold": 0.62,
    # Top-2: secondary_threshold_sweep + compose; p90(p2 | p1≥0.62)≈0.20 on same hold-out → natural gate. τ2=0.22 → ~3.4% frames with hint vs ~4.7% at 0.20.
    "ml_secondary_confidence_threshold": 0.20,
    # ── ML posterior EMA (smooths per-class probabilities before MLTop2) ─────
    # Smooths the full per-class probability vector before deriving MLTop2.
    # alpha=1.0 disables EMA (passthrough). Resolved via ENV_CLASS_TO_INT in MLInferenceThread.
    "ml_posterior_ema_alpha": 0.55,
    "ml_posterior_ema_asym": {"glare": [0.85, 0.45]},  # {class_name: [alpha_up, alpha_down]}
    # ── NIR bucket-dispatch params ────────────────────────────────────────────
    "fog_dehaze_omega": 0.85,    # Bucket D: DCP transmission factor (0.5–0.95)
    "rain_median_frames": 3,     # Bucket E: temporal median window size
    "experiment": {
        "scenario_type": "field_session",
        "environment": "rpi4",
        "lighting_level": "unspecified",
        "operator_id": "",
        "build_label": "rpi_p3b",
        "notes_short": "rsync deploy; set operator_id/notes before run",
    },
    # ── Edge Impulse person-in-dark experimental integration ──────────────────
    # Default off. Enable with EI_PERSON_IN_DARK_ENABLED=1.
    # Does not affect env selection or any pipeline output when disabled.
    "ei_person": {
        "enabled": False,
        # Copy from Edge Impulse Linux export → models/ei/ (see models/ei/README.md).
        # Override: EI_PERSON_TFLITE_PATH. Path is relative to process CWD (repo root).
        "tflite_path": "models/ei/person_in_dark_fomo_int8.tflite",
        "infer_interval": 10,   # submit every N optical frames (~3 Hz at 30 FPS)
        "fit_mode": "crop",     # center-crop to square before 128×128 resize
        "threshold": 0.8,       # matches EI_CLASSIFIER_OBJECT_DETECTION_THRESHOLD
        "draw_bbox": False,    # amber box HUD overlay (debug only; default off per EDGE_IMPULSE_FUTURE_WORK.md)
        "num_threads": 2,       # TFLite thread count; leaves 2 cores for main pipeline
    },
}

# Merged over ``CONFIG`` (shallow) in ``main`` when ``rpi_throughput_max`` is True.
# Single profile: lean HUD/display + lean features + aggressive fusion / thermal / NIR tuning.
RPI_THROUGHPUT_MAX_DEFAULTS: Dict[str, Any] = {
    "hud_bearing_enabled": False,
    "display_grade_mode": "luma_only",
    "thermal_display_glare_suppression": False,
    "nir_optical_lite": True,
    "display_temporal_glare_enable": False,
    "lean_disable_lk": True,
    "lean_disable_e1": True,
    "lean_disable_mad": True,
    "framecache_small_max_side": 96,
    "fusion_mask_blur_ksize": 7,
    # Offline sweep: omega=0.92 maximises fog contrast (log_rms=0.7419) while the D.dark guard
    # prevents crush on night classes (pct_crush=0.006 on fog; night frames returned unchanged).
    "fog_dehaze_omega": 0.92,
    "fusion_mask_blur_sigma": 2.0,
    "fusion_heatmap_interpolation": "nearest",
    "nir_hybrid_update_rate": 20,
    "nir_enhancer_detail_strength": 0.0,
    "nir_enhancer_patch_size": 3,
    "thermal_display_edge_strength": 0.0,
    "env_classification_interval": 30,
    "ML_INFERENCE_INTERVAL": 10,
    "thermal_colormap_levels": 64,
    "fusion_warp_work_scale": 0.5,
    # Skip the BGR↔LAB luma cap when no glare detected (~63% of dark-night frames).
    "display_luma_cap_glare_gate": True,
    # Optional NIR proc size cut: 320×240 → 240×180 (mentor §C.3; saves ~25% HybridNIR pixels).
    # Optional NIR proc size cut (disabled by default; saves ~25% HybridNIR pixels at mild quality cost).
    # "nir_enhancer_proc_w": 240,
    # "nir_enhancer_proc_h": 180,
}


def _opt_log(enabled: bool, msg: str) -> None:
    """Print a debug line when ``cfg['debug']`` is true (called from ``main``)."""
    if enabled:
        print(f"[opt] {msg}")


def _configure_opencv_runtime(cfg: Dict[str, Any]) -> None:
    """Apply OpenCV threading and OpenCL settings from ``CONFIG``.

    When ``opencv_num_threads > 0``, calls ``cv.setNumThreads(n)`` so TBB/OpenMP
    uses a fixed core count on Raspberry Pi 4.

    When ``opencv_disable_opencl`` is True (default), disables OpenCL — VideoCore
    lacks standard OpenCL for OpenCV; avoids per-call device probe overhead.
    """
    import cv2 as cv
    n = int(cfg.get("opencv_num_threads", 4))
    if n > 0:
        cv.setNumThreads(n)
    if bool(cfg.get("opencv_disable_opencl", True)):
        try:
            cv.ocl.setUseOpenCL(False)
        except Exception:
            pass


def resolve_optimization_profile(profile: str, haze_preset: bool, profiles_enabled: bool) -> Dict[str, Any]:
    """Return display and sensor tuning dict for the named profile.

    Used by ``main`` to seed ``opt_cfg_base`` before ENV presets merge overrides.
    When ``profiles_enabled`` is False, the baseline profile is always used.
    """
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

    if haze_preset:
        baseline.update(
            {
                "fusion_alpha_boost": 0.12,
                "thermal_floor": 4.0,
                "thermal_fg_threshold": 18.0,
                "thermal_edge_strength": 0.14,
                "thermal_agc_low_pct": 1.0,
                "thermal_agc_high_pct": 96.0,
            }
        )
    return baseline


# ── Processing profiles (P2) ─────────────────────────────────────────────────
#
# Three named profiles, each a partial CONFIG override.
# apply_profile() returns a NEW dict (immutable pattern) — never mutates input.
#
# Regression lock: PROFILES["throughput"] MUST equal RPI_THROUGHPUT_MAX_DEFAULTS.
# test_profile_apply.py enforces this. Do not diverge them.
#
# "raw" profile disables optional enhancement only — see PROFILE_HOTSWAP_BLACKLIST
# for startup-only keys that must never appear in any profile dict.

PROFILES: Dict[str, Dict[str, Any]] = {
    # Full pipeline — no override. Semantically a no-op merge.
    "quality": {},
    # Throughput: must equal RPI_THROUGHPUT_MAX_DEFAULTS (regression-locked by tests).
    "throughput": RPI_THROUGHPUT_MAX_DEFAULTS,
    # Raw: closest to per-sensor output — no CLAHE, no HybridNIR, no fusion blend,
    # no display grading beyond unavoidable visualization steps.
    # Unavoidable: uint16→uint8 normalization (thermal), BGR layout for imshow.
    # Optional (stripped): CLAHE, detail boost, temporal denoising, LAB grading.
    "raw": {
        "display_profile_raw_mode": True,      # sentinel read by nir_pipeline + display_pipeline
        "nir_enhancer_detail_strength": 0.0,
        "nir_enhancer_clahe_clip_scale": 0.0,  # 0 = skip CLAHE (interpreted by nir_pipeline)
        "nir_optical_lite": True,              # skips C/D/F enhancement paths
        "display_grade_mode": "luma_only",     # strip LAB color grading
        "display_temporal_glare_enable": False,
        "thermal_display_edge_strength": 0.0,
        "thermal_bilateral_display_enable": False,
    },
}

# Keys consumed only at startup (camera resolution, SPI clock, model path).
# apply_profile() raises ValueError if any profile dict contains a blacklisted key.
PROFILE_HOTSWAP_BLACKLIST: frozenset = frozenset({
    "display_width",
    "display_height",
    "nir_capture_fps",
    "nir_enhancer_proc_w",
    "nir_enhancer_proc_h",
    "homography_path",
})

# Runtime rate-limit for profile hot-swap (seconds between swaps).
PROFILE_HOTSWAP_MIN_INTERVAL_S: float = 1.5

# Assert at import time that no profile contains a blacklisted key.
for _pname, _pdict in PROFILES.items():
    _conflicts = PROFILE_HOTSWAP_BLACKLIST & _pdict.keys()
    if _conflicts:
        raise AssertionError(
            f"Profile '{_pname}' contains blacklisted startup-only keys: {_conflicts}"
        )
del _pname, _pdict, _conflicts


def apply_profile(cfg: Dict[str, Any], profile_name: str) -> Dict[str, Any]:
    """Return a new cfg dict with the named profile merged in (immutable — never mutates input).

    Raises ValueError if profile_name is unknown or if the profile dict (via PROFILES)
    contains a key in PROFILE_HOTSWAP_BLACKLIST (belt-and-suspenders runtime guard).
    """
    if profile_name not in PROFILES:
        raise ValueError(f"Unknown profile {profile_name!r}; valid: {list(PROFILES)}")
    overrides = PROFILES[profile_name]
    blacklisted = PROFILE_HOTSWAP_BLACKLIST & overrides.keys()
    if blacklisted:
        raise ValueError(
            f"Profile '{profile_name}' contains startup-only keys: {blacklisted}"
        )
    return {**cfg, **overrides, "display_profile": profile_name}
