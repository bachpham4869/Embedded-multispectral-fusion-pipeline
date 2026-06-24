#!/usr/bin/env python3
"""
Interactive fusion runtime: NIR, thermal, and fused display loop.

Entry: ``python -m smartbinocular`` or ``main()`` from tests. Wires capture threads
(:mod:`smartbinocular.hardware`), thermal/NIR processors, ENV presets, optional ML
inference/logging, HUD, saves, and session metrics.

Notable integration fixes carried from the legacy fusion prototype:
  1. Avoid double IIR: when temporal glare blend runs, pass ``soft_motion_active`` or the
     temporal-applied flag into shake so it does not apply a second IIR blend.
  2. Fusion default: fg heat overlay (``final_fusion``-style); gradient overlay optional via config.
  3. Single LAB pass: ``display_grade_and_cap_bgr`` replaces separate grade + luminance cap.
  4. One shared small gray per frame for statistics via ``nir_compute_gray_cached`` / ``FrameCache``.
  5. NIR glare metrics cached from the ENV block when possible for reuse on the display path.
  6. Fusion mask uses ``mask_f[:, :, np.newaxis]`` instead of ``np.stack`` for broadcasting.
  7. Capture metadata writer is module-level ``write_capture_meta`` (not redefined per frame).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import queue
import signal as sig_module
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# #region agent log
def _agent_debug_log(
    *,
    hypothesis_id: str,
    location: str,
    message: str,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    """Append NDJSON for debug sessions: ``.cursor/`` and ``logs/debug/`` (pulled by rsync); stderr fallback."""
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    payload = {
        "sessionId": "04f9e8",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data or {},
        "timestamp": int(time.time() * 1000),
    }
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    _paths = (
        os.path.join(_root, ".cursor", "debug-04f9e8.log"),
        os.path.join(_root, "logs", "debug", "debug-04f9e8.ndjson"),
    )
    _ok = False
    for _path in _paths:
        try:
            os.makedirs(os.path.dirname(_path), exist_ok=True)
            with open(_path, "a", encoding="utf-8") as fh:
                fh.write(line)
            _ok = True
        except OSError:
            continue
    if not _ok:
        print(f"[agent-debug] {line.strip()}", file=sys.stderr)


# #endregion

import numpy as np
import cv2 as cv

from smartbinocular.config import (
    CONFIG,
    DEFAULT_DISPLAY_FOV_H_DEG,
    DEFAULT_DISPLAY_FOV_V_DEG,
    HOMOGRAPHY_PATH,
    PROFILE_HOTSWAP_MIN_INTERVAL_S,
    PROFILES,
    RPI_THROUGHPUT_MAX_DEFAULTS,
    TIER_S_DISPLAY_L_MAX,
    _configure_opencv_runtime,
    _opt_log,
    apply_profile,
    resolve_optimization_profile,
)
from smartbinocular.hud import ChromeState, HudState, draw_control_chrome, draw_debug_overlays, draw_hud
from smartbinocular.controls import BAR_HEIGHT as _CHROME_BAR_HEIGHT, MouseRouter
from smartbinocular.recording import VideoRecorder
from smartbinocular.metrics import (
    ThesisRunMetrics,
    _metrics_write_json,
    build_experiment_context,
    metrics_write_run_manifest,
    redact_capture_paths,
)
from smartbinocular.env_presets import (
    ENV_PRESETS,
    EnvPresetController,
    _format_env_apply_log,
    apply_e1_overrides,
    apply_secondary_hint,
    auto_rule_preset_to_env_class,
    compose_env_from_ml_top2,
    infer_env_tags_auto_rule,
    merge_opt_cfg_with_preset,
    restore_e1_from_snapshot,
    select_env_preset_from_tags,
    snapshot_e1_defaults,
)
from smartbinocular.thermal_pipeline import (
    ThermalAnomalyDetectorLite,
    ThermalMADAnomalyDetector,
    ThermalProcessor,
    thermal_agc,
    thermal_edge_enhance,
    thermal_glare_hud_should_apply,
)
from smartbinocular.nir_pipeline import (
    HybridNIREnhancer,
    OPTICAL_BUCKET_DISPATCH,
    RainTemporalMedian,
    resolve_optical_bucket,
    _nir_glare_metrics_from_gray,
    _nir_mean_brightness_bgr,
    _nir_gray_std_quick,
    nir_anti_glare_bgr,
    nir_compute_gray_cached,
    nir_dehaze_lite,
    nir_glare_eval,
    nir_skip_second_anti_glare_bgr,
    nir_nir_night_clahe,
    nir_transition_blend,
    compute_brisque_score,
)
from smartbinocular.display_pipeline import (
    DisplayTemporalGlareBlend,
    display_grade_and_cap_bgr,
    display_luminance_cap_bgr,
)

# NIR: night vs non-night (``legacy/py/final_fusion.py`` NIR_RAW_THRESHOLD=25; hysteresis in code only).
NIR_NIGHT_ENV_CLASSES = frozenset({"night_clear", "normal_night", "nir_night"})
NIR_NON_NIGHT_RAW_CENTER = 25.0
NIR_NON_NIGHT_HYST = 3.0
NIR_NON_NIGHT_RAW_ENTER = NIR_NON_NIGHT_RAW_CENTER + NIR_NON_NIGHT_HYST
NIR_NON_NIGHT_RAW_EXIT = NIR_NON_NIGHT_RAW_CENTER - NIR_NON_NIGHT_HYST
from smartbinocular.feature_extractor import FeatureExtractor
from smartbinocular.feature_schema import (
    ENV_CLASS_TO_INT,
    ENV_INT_TO_CLASS,
    ENV_INT_UNKNOWN,
    FEATURE_SET_OPTICAL_ONLY,
)
from smartbinocular.motion import (
    DisplayShakeReducerLite,
    JerkGate,
    OpsecLocalOnly,
    SparseOpticalFlowMotion,
)
from smartbinocular.hardware import (
    gray_to_thermal_bgr,
    NIRCapture,
    ThermalCapture,
    _capture_hardware_idle_flags,
    bearing_hv_deg_from_uv,
    ensure_fusion_capture_dirs,
    load_homography,
    sector_from_bearing_deg,
    _HAS_HARDWARE,
)
from smartbinocular.utils import (
    FPSCounter,
    FrameCache,
    build_frame_cache,
    CaptureIntegrityChain,
    dhash,
    MLLogger,
    StageProfiler,
    compute_homography_quality,
)
from smartbinocular.ml_inference import EnvClassifier, MLInferenceThread, MLSharedResult, MLTop2


def _homography_scaled_to_work_size(
    H: np.ndarray, nir_w: int, nir_h: int, work_scale: float
) -> tuple[np.ndarray, int, int]:
    """Apply output-side scale S @ H so warpPerspective runs on a smaller (nw×nh) canvas (fewer pixels / lower variance)."""
    ws = float(np.clip(work_scale, 0.25, 1.0))
    if ws >= 0.999:
        return H, nir_w, nir_h
    nw = max(2, int(round(nir_w * ws)))
    nh = max(2, int(round(nir_h * ws)))
    sx = nw / float(nir_w)
    sy = nh / float(nir_h)
    S = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return (S @ H), nw, nh


# Module-level capture integrity chain (initialised in main() after save_dir exists).
_capture_chain: Optional[CaptureIntegrityChain] = None

# ── Module-level capture meta writer (defined once; not redefined each frame) ──

def write_capture_meta(
    path_png: str,
    trigger: str,
    *,
    mode: str,
    jerk_gate: JerkGate,
    glare_nir: bool,
    glare_th: bool,
    soft_motion_active: bool,
    cfg: Dict[str, Any],
    env_stable_for_hud: str,
    phase1_alert_enabled: bool,
    sector_alert_text: str,
    phase1_e1_enabled: bool,
    phase1_blobs: List[Dict[str, Any]],
    thesis_metrics: ThesisRunMetrics,
    fps: float,
    alpha: float,
    nir_brightness: Optional[float],
    nir_b_ema: Optional[float],
    nir_use_raw_auto: bool,
    bear_h: float,
    bear_v: float,
    frame: Optional[np.ndarray] = None,
    profile_label: str = "",
) -> None:
    """Write sidecar JSON next to a saved PNG and optionally sign it with the integrity chain.

    Called from the main loop on manual save, auto-timer save, and shutdown paths.
    When ``frame`` is set and ``_capture_chain`` is initialised, appends dHash and HMAC fields.
    """
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
        show_raw_toggle=False,
        jerk_active=jerk_gate.active,
        glare_nir=glare_nir,
        glare_th=glare_th,
        tags_hud=tag_list,
        bearing_h_deg=bear_h,
        bearing_v_deg=bear_v,
    )
    # Active processing profile (additive field — no schema version bump)
    meta["display_profile"] = profile_label or str(cfg.get("display_profile", "quality"))
    # Optional perceptual hash + chained HMAC over capture metadata
    if frame is not None and _capture_chain is not None:
        meta["frame_dhash"] = dhash(frame)
        _capture_chain.sign(meta)
    _metrics_write_json(path_png.replace(".png", ".json"), redact_capture_paths(meta))


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Start cameras, run the OpenCV event/display loop, then write session metrics."""
    import subprocess
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
    except Exception:
        rev = "unknown"
    print(f"[debug] smartbinocular loaded from: {__file__}")
    print(f"[debug] git: {rev}")

    # ── CLI argument parsing ───────────────────────────────────────────────────
    # Default: no flags preserves full pipeline behaviour from CONFIG.
    import argparse
    _parser = argparse.ArgumentParser(
        prog="smartbinocular",
        description="SmartBinocular — NIR + Thermal fusion pipeline",
    )
    _parser.add_argument(
        "--pipeline",
        choices=["default", "lean"],
        default=None,
        help="Pipeline profile. 'lean' disables LK, E1, and MAD to reduce CPU load.",
    )
    _parser.add_argument(
        "--no-lk",
        dest="no_lk",
        action="store_true",
        default=False,
        help="Disable LK sparse optical flow (ML features only; does not affect display).",
    )
    _parser.add_argument(
        "--no-e1",
        dest="no_e1",
        action="store_true",
        default=False,
        help="Disable E1 anomaly detector (hard off; cannot toggle at runtime).",
    )
    _parser.add_argument(
        "--no-mad",
        dest="no_mad",
        action="store_true",
        default=False,
        help="Disable MAD anomaly detector (independent of --no-e1).",
    )
    _args = _parser.parse_args()

    global _capture_chain
    cfg: Dict[str, Any] = dict(CONFIG)
    if bool(cfg.get("rpi_throughput_max", False)):
        cfg.update(RPI_THROUGHPUT_MAX_DEFAULTS)
    _configure_opencv_runtime(cfg)

    # ── Lean pipeline flags ────────────────────────────────────────────────────
    # Precedence: CLI > CONFIG > default False. ``--pipeline lean`` sets all lean flags.
    _lean_lk: bool = (
        bool(cfg.get("lean_disable_lk", False))
        or (_args.pipeline == "lean")
        or _args.no_lk
    )
    _lean_e1: bool = (
        bool(cfg.get("lean_disable_e1", False))
        or (_args.pipeline == "lean")
        or _args.no_e1
    )
    _lean_mad: bool = (
        bool(cfg.get("lean_disable_mad", False))
        or (_args.pipeline == "lean")
        or _args.no_mad
    )

    if not _HAS_HARDWARE:
        logger.error("Hardware not available. Run on Raspberry Pi with Picamera2 and spidev/smbus installed.")
        print("Hardware not available. Run on Raspberry Pi with Picamera2 and spidev/smbus installed.")
        sys.exit(1)

    _hpath_raw = cfg.get("homography_path", str(HOMOGRAPHY_PATH))
    _hpath = os.path.abspath(_hpath_raw) if not os.path.isabs(_hpath_raw) else _hpath_raw
    if not os.path.exists(_hpath):
        # Fall back to the package-bundled asset
        _hpath = str(HOMOGRAPHY_PATH)
    if not os.path.exists(_hpath):
        logger.error(
            "Homography file not found. "
            "Please run calibration or provide homography.json in the assets directory."
        )
        print(
            "Homography file not found. "
            "Please run calibration or provide homography.json in the assets directory."
        )
        sys.exit(1)
    homography_path = _hpath

    H, thermal_size, nir_size, fov_h_meta, fov_v_meta = load_homography(homography_path)
    # Homography quality: load_homography returns (w,h) tuples; reverse to (h,w) for shape API.
    _hq = compute_homography_quality(
        H,
        thermal_shape=(thermal_size[1], thermal_size[0]),
        nir_shape=(nir_size[1], nir_size[0]),
    )
    nir_w, nir_h = nir_size
    display_size = (int(cfg.get("display_width", 800)), int(cfg.get("display_height", 480)))
    alpha = float(cfg.get("fusion_alpha", 0.55))
    exp_context = build_experiment_context(cfg)
    opt_cfg_base = resolve_optimization_profile(
        str(cfg.get("opt_profile", "baseline")),
        bool(cfg.get("opt_haze_preset", False)),
        not bool(cfg.get("opt_disable_profiles", False)),
    )
    opt_cfg_base["thermal_fg_max_ratio"] = float(
        np.clip(float(cfg.get("thermal_fg_max_ratio", 0.8)), 0.05, 0.95)
    )
    opt_runtime_config = {
        "profile": opt_cfg_base["name"],
        "profiles_enabled": bool(not cfg.get("opt_disable_profiles", False)),
        "haze_preset": bool(cfg.get("opt_haze_preset", False)),
        "debug": bool(cfg.get("debug", False)),
        "hud_bearing_enabled": bool(cfg.get("hud_bearing_enabled", True)),
        "display_grade_mode": str(cfg.get("display_grade_mode", "full")),
        "thermal_display_glare_suppression": bool(cfg.get("thermal_display_glare_suppression", True)),
        "nir_optical_lite": bool(cfg.get("nir_optical_lite", False)),
        "rpi_throughput_max": bool(cfg.get("rpi_throughput_max", False)),
        "fix_nir_motion_gating": bool(cfg.get("fix_nir_motion_gating", True)),
        "fix_thermal_detail_preserve": bool(cfg.get("fix_thermal_detail_preserve", False)),
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

    thermal_proc = ThermalProcessor(
        warmup=40,
        anti_glare=True,
        display_anti_glare=bool(cfg.get("thermal_display_glare_suppression", True)),
        thermal_high_pct=float(opt_cfg_base["thermal_high_pct"]),
        thermal_saturate_at=float(opt_cfg_base["thermal_saturate_at"]),
        thermal_gamma=float(opt_cfg_base["thermal_gamma"]),
        thermal_floor=float(opt_cfg_base["thermal_floor"]),
        thermal_fg_threshold=float(opt_cfg_base["thermal_fg_threshold"]),
        thermal_fg_max_ratio=float(opt_cfg_base["thermal_fg_max_ratio"]),
        thermal_edge_strength=float(opt_cfg_base["thermal_edge_strength"]),
        thermal_agc_low_pct=float(opt_cfg_base["thermal_agc_low_pct"]),
        thermal_agc_high_pct=float(opt_cfg_base["thermal_agc_high_pct"]),
        detail_preserve_detect=bool(cfg.get("fix_thermal_detail_preserve", False)),
        detect_raw_mix=float(cfg.get("thermal_detect_raw_mix", 0.55)),
        detail_grad_threshold=float(cfg.get("thermal_detail_grad_threshold", 9.0)),
        detail_threshold_relax=float(cfg.get("thermal_detail_threshold_relax", 3.0)),
        detect_use_anti_glare=bool(cfg.get("thermal_detect_use_anti_glare", False)),
        thermal_3dnr_alpha=float(cfg.get("thermal_3dnr_alpha", 0.65)),
        bilateral_display_enable=bool(cfg.get("thermal_bilateral_display_enable", True)),
        bilateral_d=int(cfg.get("thermal_bilateral_d", 5)),
        bilateral_sigma_color=float(cfg.get("thermal_bilateral_sigma_color", 15.0)),
        bilateral_sigma_space=float(cfg.get("thermal_bilateral_sigma_space", 5.0)),
    )
    _hm_mode = str(cfg.get("fusion_heatmap_interpolation", "cubic")).lower()
    if _hm_mode == "cubic":
        _fusion_hm_inter = cv.INTER_CUBIC
    elif _hm_mode == "nearest":
        _fusion_hm_inter = cv.INTER_NEAREST
    else:
        _fusion_hm_inter = cv.INTER_LINEAR
    _tcmap_lv = int(cfg.get("thermal_colormap_levels", 256))
    if _tcmap_lv not in (32, 64, 128, 256):
        _tcmap_lv = 256
    _nir_hybrid_ur = int(max(1, int(cfg.get("nir_hybrid_update_rate", 8))))
    _nir_pw = int(max(160, int(cfg.get("nir_enhancer_proc_w", 320))))
    _nir_ph = int(max(120, int(cfg.get("nir_enhancer_proc_h", 240))))
    _nir_psz = int(max(3, int(cfg.get("nir_enhancer_patch_size", 5))))
    if _nir_psz % 2 == 0:
        _nir_psz += 1
    nir_enhancer = HybridNIREnhancer(
        proc_w=_nir_pw,
        proc_h=_nir_ph,
        patch_size=_nir_psz,
        update_rate=_nir_hybrid_ur,
        detail_strength=float(cfg.get("nir_enhancer_detail_strength", 0.25)),
        clahe_clip_scale=float(cfg.get("nir_enhancer_clahe_clip_scale", 1.0)),
    )
    rain_processor = RainTemporalMedian(n_frames=int(cfg.get("rain_median_frames", 3)))
    rain_processor_lite = RainTemporalMedian(
        n_frames=int(max(2, int(cfg.get("rain_median_frames_lite", 2))))
    )

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
    _shake_mode = str(cfg.get("display_shake_mode", "blend")).strip().lower()
    if _shake_mode not in ("off", "blend", "shift"):
        _shake_mode = "blend"
    shake = DisplayShakeReducerLite(mode=_shake_mode, blend_current_weight=0.50)

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

    # ── E1 runtime toggle state ────────────────────────────────────────────────
    # _e1_runtime_enabled: toggled with the 'e' key unless _lean_e1 (--no-e1) forces E1 off.
    # A future GPIO hook can set _e1_runtime_enabled from a hardware callback.
    _e1_runtime_enabled: bool = True  # default on; 'e' toggles inside the loop

    # Sparse LK optical flow (alongside JerkGate; feeds ML motion features when not lean).
    lk_flow = SparseOpticalFlowMotion(max_points=20, refresh_interval=15)
    # MAD anomaly detector (optional upgrade path for thermal blobs).
    mad_detector = ThermalMADAnomalyDetector(
        mad_z_thresh=float(cfg.get("feature_e1_mad_z_thresh", 3.5)),
        temporal_window=int(cfg.get("feature_e1_mad_temporal_window", 3)),
        min_area=int(cfg.get("feature_e1_min_area", 10)),
    )
    env_fallback = str(cfg.get("env_fallback_preset", "default"))
    env_controller = EnvPresetController(
        fallback=env_fallback if env_fallback in ENV_PRESETS else "default",
        hysteresis_frames=int(cfg.get("env_hysteresis_frames", 18)),
    )
    prev_env_stable: Optional[str] = None

    def _map_thermal_raw_to_display_xy(x_raw: float, y_raw: float, raw_w: int, raw_h: int):
        """Map thermal blob centre from sensor pixels to display pixels (thermal mode, linear scale)."""
        if raw_w <= 1 or raw_h <= 1:
            return 0, 0
        xr = (raw_w - 1) - float(x_raw)
        yr = (raw_h - 1) - float(y_raw)
        xd = int(round(np.clip(xr * (display_size[0] / float(raw_w - 1)), 0, display_size[0] - 1)))
        yd = int(round(np.clip(yr * (display_size[1] / float(raw_h - 1)), 0, display_size[1] - 1)))
        return xd, yd

    def _map_thermal_raw_to_display_xy_fusion(x_raw: float, y_raw: float, raw_w: int, raw_h: int):
        """Project thermal blob centre through ``H`` into display space (fusion mode)."""
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

    # Teardown registry — each stoppable resource appends a callable here at creation time.
    # _cleanup_all() iterates in reverse (newest-first) and is idempotent: each handler
    # must be a no-op when called on an already-stopped resource.
    _cleanup_registry: list = []

    def _cleanup_all() -> None:
        for fn in reversed(_cleanup_registry):
            try:
                fn()
            except Exception:
                pass
        cv.destroyAllWindows()  # safe even if cv.namedWindow was never called

    def on_signal(sig, frame):
        """SIGINT/SIGTERM: run centralized teardown then exit."""
        _cleanup_all()
        sys.exit(0)
    sig_module.signal(sig_module.SIGINT, on_signal)
    sig_module.signal(sig_module.SIGTERM, on_signal)

    print("Starting cameras...")
    thermal_cap.start()
    nir_cap.start()
    _cleanup_registry.append(lambda: thermal_cap.stop())
    _cleanup_registry.append(lambda: nir_cap.stop())
    time.sleep(2)

    # Resolve ML model path once (CONFIG + env) for manifest hash + EnvClassifier below.
    _ml_model_path: Optional[str] = None
    _mp_cfg = cfg.get("ML_MODEL_PATH") or None
    if os.environ.get("ML_MODEL_PATH", "").strip():
        _mp_cfg = os.environ["ML_MODEL_PATH"].strip()
    if _mp_cfg:
        _ml_model_path = _mp_cfg if os.path.isabs(_mp_cfg) else os.path.abspath(_mp_cfg)

    save_dir, metrics_dir = ensure_fusion_capture_dirs()
    # Capture integrity chain uses a device-local key file under save_dir.
    _capture_chain = CaptureIntegrityChain(key_path=os.path.join(save_dir, ".session_key"))
    session_id = time.strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:10]
    _manifest_model_path = (
        _ml_model_path
        if (_ml_model_path and os.path.isfile(_ml_model_path))
        else None
    )
    _tier_s1 = bool(cfg.get("hud_bearing_enabled", True))
    _tier_s3 = bool(cfg.get("display_temporal_glare_enable", True))
    metrics_write_run_manifest(
        os.path.join(metrics_dir, f"manifest_{session_id}.json"),
        session_id=session_id,
        homography_path=os.path.abspath(homography_path),
        display_size=(display_size[0], display_size[1]),
        fov_deg=(fov_h, fov_v),
        tier_s_flags={
            "S1_bearing_hud": _tier_s1,
            "S2_run_manifest": True,
            "S3_temporal_glare": _tier_s3,
            "S4_display_luminance_cap": True,
            "S6_stream_timestamp_skew": False,
            "S5_battery": False,
        },
        model_path=_manifest_model_path,
        extra={
            "experiment_context": exp_context,
            "optimization_config": opt_runtime_config,
            "homography_quality": _hq,
        },
    )
    _ml_log_enabled = bool(cfg.get("ML_LOG_ENABLED", False))
    ml_logger: Optional[MLLogger] = None
    ml_feature_extractor: Optional[FeatureExtractor] = None
    if _ml_log_enabled:
        _mld_raw = str(cfg.get("ML_LOG_DIR", "logs/ml"))
        _ml_log_dir = _mld_raw if os.path.isabs(_mld_raw) else os.path.abspath(_mld_raw)
        os.makedirs(_ml_log_dir, exist_ok=True)
        _ml_log_path = os.path.join(_ml_log_dir, f"live_{session_id}.jsonl")
        ml_logger = MLLogger(_ml_log_path, buffer_size=100, flush_interval_s=30.0)
        ml_feature_extractor = FeatureExtractor()
        _cleanup_registry.append(lambda: ml_logger.close())
        print(f"  [ML] logging features → {_ml_log_path} (interval={int(cfg.get('ML_LOG_INTERVAL', 5))} frames)")

    # ── ML runtime inference init (observe-only) ─────────────────────────────
    # Env vars override CONFIG: ML_INFERENCE_ENABLED=1 and ML_MODEL_PATH=<path>
    # (_ml_model_path already resolved above for run manifest.)
    _ml_inference_enabled: bool = bool(cfg.get("ML_INFERENCE_ENABLED", False))
    if os.environ.get("ML_INFERENCE_ENABLED", "").strip().lower() in ("1", "true"):
        _ml_inference_enabled = True

    _ml_classifier: Optional[EnvClassifier] = None
    _ml_shared_result: Optional[MLSharedResult] = None
    _ml_infer_thread: Optional[MLInferenceThread] = None
    _ml_infer_queue: Optional[queue.Queue] = None  # type: ignore[type-arg]
    _ml_infer_extractor: Optional[FeatureExtractor] = None
    _ml_inference_active: bool = False

    if _ml_inference_enabled and _ml_model_path:
        _ml_classifier = EnvClassifier(str(_ml_model_path))
        if _ml_classifier.available:
            _ml_infer_queue = queue.Queue(maxsize=2)
            _ml_shared_result = MLSharedResult()
            _ml_infer_thread = MLInferenceThread(
                _ml_classifier, _ml_shared_result, _ml_infer_queue,
                ema_alpha=float(cfg.get("ml_posterior_ema_alpha", 0.55)),
                ema_asym=dict(cfg.get("ml_posterior_ema_asym") or {}),
            )
            _ml_infer_thread.start()
            _cleanup_registry.append(
                lambda: (_ml_infer_thread.stop(), _ml_infer_thread.join(timeout=1.0))
            )
            _ml_infer_extractor = FeatureExtractor()
            _ml_inference_active = True
            print(
                f"  [ML] inference active → {_ml_model_path} "
                f"(interval={int(cfg.get('ML_INFERENCE_INTERVAL', 15))} frames)"
            )
        else:
            print(
                f"  [ML] WARNING: model load failed ({_ml_model_path}) "
                "— inference disabled, running rule-based only"
            )
    elif _ml_inference_enabled:
        print("  [ML] WARNING: ML_INFERENCE_ENABLED=True but ML_MODEL_PATH not set — inference disabled")

    # ── Edge Impulse person-in-dark (experimental, default off) ──────────────
    # Enable with EI_PERSON_IN_DARK_ENABLED=1. No effect on pipeline when disabled.
    _ei_cfg = cfg.get("ei_person", {})
    _ei_enabled: bool = bool(_ei_cfg.get("enabled", False))
    if os.environ.get("EI_PERSON_IN_DARK_ENABLED", "").strip() in ("1", "true"):
        _ei_enabled = True

    _ei_worker = None
    _ei_shared = None
    _ei_infer_interval: int = int(_ei_cfg.get("infer_interval", 10))
    _ei_draw_bbox: bool = bool(_ei_cfg.get("draw_bbox", False))
    _ei_frame_idx: int = 0
    _ei_metrics: dict = {
        "frames_submitted": 0,
        "frames_dropped_overflow": 0,
        "inferences_ok": 0,
        "inferences_err": 0,
        "inference_ms_samples": [],
        "detections_per_frame": [],
    }

    if _ei_enabled:
        try:
            from smartbinocular.experimental.ei_person_in_dark import EIWorker as _EIWorker
            _ei_tflite_default = "models/ei/person_in_dark_fomo_int8.tflite"
            _ei_tflite = (
                os.environ.get("EI_PERSON_TFLITE_PATH", "").strip()
                or str(_ei_cfg.get("tflite_path", _ei_tflite_default))
            )
            _ei_worker = _EIWorker(
                tflite_path=_ei_tflite,
                num_threads=int(_ei_cfg.get("num_threads", 2)),
                threshold=float(_ei_cfg.get("threshold", 0.8)),
                fit_mode=str(_ei_cfg.get("fit_mode", "crop")),
            )
            _ei_worker.start()
            _cleanup_registry.append(lambda: _ei_worker.stop(timeout=1.0))
            _ei_shared = _ei_worker.get_shared()
            print(f"  [EI] person-in-dark worker started (interval={_ei_infer_interval} frames)")
        except Exception as _ei_exc:
            logger.warning("[EI] Worker init failed: %s — continuing without it", _ei_exc)
            _ei_worker = None
            _ei_shared = None

    cv.namedWindow("SmartBinocular", cv.WINDOW_NORMAL | cv.WINDOW_GUI_NORMAL)
    cv.setWindowProperty("SmartBinocular", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    cv.imshow("SmartBinocular", np.zeros((display_size[1], display_size[0], 3), dtype=np.uint8))
    cv.waitKey(200)
    # Let Openbox handle fullscreen natively via lxde-pi-rc.xml

    # OS-level brute force fallback moved to end of initialization

    # S1: default bearing reference is display centre; left click moves the probe pixel.
    a1_probe_xy = [display_size[0] // 2, display_size[1] // 2]

    _mouse_router = MouseRouter(display_size=display_size)
    # FIFO delivery up to maxsize; Full → silent drop (matches _ml_infer_queue pattern).
    # UX note: old list[None] had last-write-wins semantics; queue delivers in arrival order.
    # With MouseRouter's 250 ms debounce this difference is invisible in normal use.
    _button_action_q: queue.Queue = queue.Queue(maxsize=8)
    _CHROME_ACTIVE_ALPHA: float = 0.55
    _CHROME_FADE_ALPHA: float = 0.15
    _last_chrome_touch_t: float = time.monotonic()
    _chrome_alpha: float = _CHROME_ACTIVE_ALPHA
    _recorder: Optional[VideoRecorder] = None
    # _recorder is created dynamically (during recording); guard on None at teardown time.
    _cleanup_registry.append(lambda: _recorder.stop(timeout=2.0) if _recorder is not None else None)
    _toast_text: Optional[str] = None
    _toast_until_t: float = 0.0

    # P2 — profile hot-swap state
    _profile_cycle: list = list(PROFILES.keys())   # ["quality", "throughput", "raw"]
    _current_profile: str = str(cfg.get("display_profile", "quality"))
    _last_profile_swap_t: float = 0.0
    _base_cfg: Dict[str, Any] = dict(cfg)           # startup snapshot; apply_profile merges onto this

    def _on_mouse(event, x, y, _flags, _param):
        """Route clicks: bottom bar → button action; scene area → A1 bearing probe."""
        nonlocal _last_chrome_touch_t
        action = _mouse_router.on_event(event, x, y)
        if action is not None:
            try:
                _button_action_q.put_nowait(action)
            except queue.Full:
                pass
            _last_chrome_touch_t = time.monotonic()
        elif event == cv.EVENT_LBUTTONDOWN and y < display_size[1] - _CHROME_BAR_HEIGHT:
            a1_probe_xy[0] = x
            a1_probe_xy[1] = y

    cv.setMouseCallback("SmartBinocular", _on_mouse)

    NIR_B_EMA = 0.18
    SWITCH_FRAMES = int(max(8, min(120, int(cfg.get("mode_switch_prep_frames", 28)))))

    thesis_metrics = ThesisRunMetrics(
        session_id=session_id,
        homography_path=os.path.abspath(homography_path),
        display_size=(display_size[0], display_size[1]),
        fusion_alpha=alpha,
        nir_schmitt_raw_on=float(cfg.get("nir_schmitt_raw_on", 30.0)),
        nir_schmitt_dim_on=float(cfg.get("nir_schmitt_dim_on", 18.0)),
        nir_b_ema_coef=NIR_B_EMA,
        experiment_context=exp_context,
        optimization_config=opt_runtime_config,
        pipeline_config=cfg,
    )
    _opt_log(
        bool(cfg.get("debug", False)),
        (
            f"startup profile={opt_cfg_base['name']} haze={bool(cfg.get('opt_haze_preset', False))} "
            f"display_l_max={opt_cfg_base['display_l_max']} temporal_prev={opt_cfg_base['temporal_prev_weight']:.2f} "
            f"nir_fix={bool(cfg.get('fix_nir_motion_gating', True))} "
            f"thermal_fix={bool(cfg.get('fix_thermal_detail_preserve', False))} "
            f"th_cap_blur_k={int(cfg.get('thermal_capture_blur_ksize', 1))}"
        ),
    )

    mode = "imx"
    nir_b_ema = None
    nir_use_raw_auto = False   # Schmitt trigger removed; bucket dispatch handles routing
    _stable_env_class: str = "normal_night"  # tracks latest stabilized ENV_CLASS for bucket dispatch
    _nir_non_night_raw_latched: bool = False  # Schmitt: near-raw when bright; avoids flicker at ~25
    mode_switch_pending = None
    switch_frames_left = 0
    auto_start = None
    AUTO_DELAY = 5
    thermal_denoised = thermal_enhanced = heat_map = fg_mask = nir_enhanced = None
    _prev_soft_motion = False
    _prev_th_detail_mode = False
    phase1_frame_idx = 0
    _bucket: str = "A"          # last dispatched NIR optical bucket; init before frame loop
    # Per-stage latency profiler — attached to thesis_metrics for session JSON.
    _stage_profiler = StageProfiler()
    thesis_metrics.stage_profiler = _stage_profiler
    # Inner profiler for fusion_composite sub-stages — separate instance (StageProfiler is not re-entrant).
    _fuse_sp = StageProfiler()
    thesis_metrics.fuse_stage_profiler = _fuse_sp
    # Inner profiler for ThermalProcessor sub-stages — separate instance.
    _thermal_sp = StageProfiler()
    thesis_metrics.thermal_stage_profiler = _thermal_sp
    thermal_proc.stage_profiler = _thermal_sp
    # BRISQUE subsampling counter (one sample every 30 frames when enabled).
    _iqa_frame_idx: int = 0
    _iqa_enabled: bool = bool(cfg.get("iqa_logging_enabled", False))
    # ENV auto_rule classification rate limiter (expensive path every N frames).
    _env_frame_idx: int = 0
    _last_desired_env: str = "normal_night"  # default ENV_CLASS; overridden by ML or auto_rule
    ENV_CLASSIFICATION_INTERVAL: int = int(max(1, cfg.get("env_classification_interval", 10)))
    phase1_blobs: List[Dict[str, Any]] = []
    sector_alert_text = ""
    sector_alert_hold = 0
    _perf_frame_idx = 0
    _perf_report_interval = int(max(60, int(cfg.get("perf_report_interval", 180))))
    env_stable_for_hud = env_controller.stable_name
    env_display_grade: Dict[str, float] = {}
    nir_brightness_subsample = int(max(1, int(cfg.get("nir_brightness_subsample", 4))))
    _last_thermal_mono_processed: Optional[float] = None
    _ml_log_counter: int = 0
    _ml_infer_counter: int = 0
    _prev_ml_env_int: int = ENV_INT_UNKNOWN

    print("=" * 60)
    print("  1 = IMX (NIR)   2 = Thermal (3DNR)   3 = Fusion (NIR+Thermal)")
    print("  NIR: night → buckets A–F; else → near-RAW (bright) or CLAHE B (dim), hysteresis ~25")
    print("  R = raw/processed preview (OPT/THM; FUS toggles NIR raw vs enhanced)")
    print("  E = toggle E1/MAD detector at runtime")
    print("  (C4 + A6b blend + D2 automatic; HUD shows [JERK]/[GLARE] when active)")
    # Lean pipeline status
    _lean_active = _lean_lk or _lean_e1 or _lean_mad
    if _lean_active:
        _lean_off = ", ".join(
            x for x, f in [("LK", _lean_lk), ("E1", _lean_e1), ("MAD", _lean_mad)] if f
        )
        print(f"  [LEAN] disabled: {_lean_off}")
    print(f"  Metrics: JSON next to each image; session summary: {metrics_dir}/session_*.json (on quit)")
    print(
        f"  Tier S+: A1 bearing at centre (FOV≈{fov_h:.1f}°×{fov_v:.1f}°) | temporal glare (profile={opt_cfg_base['name']})"
        f" | L cap≤{display_l_max}"
    )
    print(f"  Manifest: {metrics_dir}/manifest_{session_id}.json")
    _em0 = str(cfg.get("env_mode", "off")).strip().lower()
    print(f"  [ENV] mode={_em0} fallback={env_fallback} (hysteresis={int(cfg.get('env_hysteresis_frames', 18))} frames)")
    _pth = bool(cfg.get("pause_thermal_capture_when_imx_only", True))
    _pnir = bool(cfg.get("pause_nir_capture_when_thermal_only", False))
    print(
        f"  [CAP] idle thermal in IMX-only mode={_pth} | idle NIR in thermal-only mode={_pnir} "
        f"(auto_rule always needs NIR)"
    )
    print("=" * 60)

    _th0, _nir0 = _capture_hardware_idle_flags(mode, None, cfg)
    thermal_cap.set_idle(_th0)
    nir_cap.set_idle(_nir0)

    # Fusion constants — hoisted; never change after startup.
    _fws = float(np.clip(float(cfg.get("fusion_warp_work_scale", 1.0)), 0.25, 1.0))
    _H_fuse, _nw_fuse, _nh_fuse = _homography_scaled_to_work_size(H, nir_w, nir_h, _fws)
    _nir_inter_fuse = cv.INTER_AREA if _fws < 0.999 else cv.INTER_LINEAR
    _fusion_mode_pre = str(cfg.get("fusion_overlay_mode", "fg_mask")).strip().lower()
    if _fusion_mode_pre not in ("gradient", "fg_mask"):
        _fusion_mode_pre = "fg_mask"
    _heat_base_pre = float(np.clip(float(cfg.get("fusion_thermal_base_alpha", 0.6)), 0.05, 1.0))
    _mbk_pre = int(cfg.get("fusion_mask_blur_ksize", 15))
    if _mbk_pre >= 3 and _mbk_pre % 2 == 0:
        _mbk_pre += 1
    _mbs_pre = float(cfg.get("fusion_mask_blur_sigma", 4.0))
    # Pre-allocated float32 blend buffers — eliminates per-frame heap allocs in fusion hot-path.
    _fuse_nir_f   = np.empty((_nh_fuse, _nw_fuse, 3), dtype=np.float32)
    _fuse_hm_f    = np.empty((_nh_fuse, _nw_fuse, 3), dtype=np.float32)
    _fuse_out_f   = np.empty((_nh_fuse, _nw_fuse, 3), dtype=np.float32)
    _fuse_alpha_f = np.empty((_nh_fuse, _nw_fuse),    dtype=np.float32)  # fg_mask scalar alpha

    # Native OpenCV fullscreen property applied during initialization.

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
        elif key == ord("a"):
            auto_start = time.time()
        elif key == ord("+") or key == ord("="):
            alpha = min(alpha + 0.05, 1.0)
            print(f"Fusion blend α={alpha:.2f} (thermal layer weight on NIR)")
        elif key == ord("-"):
            alpha = max(alpha - 0.05, 0.05)
            print(f"Fusion blend α={alpha:.2f} (thermal layer weight on NIR)")
        elif key == ord("e") or key == ord("E"):
            # Runtime E1+MAD toggle; ignored when --no-e1 forced lean E1 off.
            if _lean_e1:
                print("E1 is hard-disabled by --no-e1; cannot toggle.")
            else:
                _e1_runtime_enabled = not _e1_runtime_enabled
                _state_str = "ON" if _e1_runtime_enabled else "OFF"
                print(f"E1/MAD detector: {_state_str} (press E to toggle)")

        # ── Button dispatch (from mouse [OPT]/[THM]/[FUS]/[CAP]/[REC]/[PROF]) ──
        try:
            _action = _button_action_q.get_nowait()
        except queue.Empty:
            _action = None
        if _action == "OPT" and mode != "imx":
            mode_switch_pending = "imx"
            switch_frames_left = SWITCH_FRAMES
        elif _action == "THM" and mode != "thermal":
            mode_switch_pending = "thermal"
            switch_frames_left = SWITCH_FRAMES
        elif _action == "FUS" and mode != "fusion":
            mode_switch_pending = "fusion"
            switch_frames_left = SWITCH_FRAMES
        elif _action == "CAP":
            key = ord("s")   # reuse existing `s` capture path below
        elif _action == "REC":
            key = ord("v")   # reuse video toggle path below
        elif _action == "PROF":
            key = ord("p")   # reuse profile cycle path below

        # ── Profile cycle (p key or [PROF] button) ───────────────────────────
        if key == ord("p"):
            _now = time.monotonic()
            if _now - _last_profile_swap_t >= PROFILE_HOTSWAP_MIN_INTERVAL_S:
                _idx = _profile_cycle.index(_current_profile)
                _next_profile = _profile_cycle[(_idx + 1) % len(_profile_cycle)]
                cfg = apply_profile(_base_cfg, _next_profile)
                _current_profile = _next_profile
                _last_profile_swap_t = _now
                thesis_metrics.pipeline_config = cfg
                _toast_text = f"PROFILE: {_current_profile.upper()}"
                _toast_until_t = _now + 2.0
                switch_frames_left = SWITCH_FRAMES
                # #region agent log
                _agent_debug_log(
                    hypothesis_id="H2",
                    location="main.py:profile_cycle",
                    message="profile_applied",
                    data={
                        "profile": _current_profile,
                        "mode": mode,
                        "mode_switch_pending": mode_switch_pending,
                        "switch_frames_left": switch_frames_left,
                    },
                )
                # #endregion
                print(f"[profile] → {_current_profile}")
            else:
                print(f"[profile] rate-limited; wait {PROFILE_HOTSWAP_MIN_INTERVAL_S:.1f}s")

        # ── Video record toggle (v key or [REC] button) ───────────────────────
        if key == ord("v"):
            if _recorder is not None and _recorder.is_active:
                _recorder.stop()
                _recorder = None
                _toast_text = "REC STOPPED"
                _toast_until_t = time.monotonic() + 2.0
            else:
                _ts_rec = time.strftime("%Y%m%d-%H%M%S")
                _rec_dir = os.path.join(save_dir, "video", session_id)
                _rec_path = os.path.join(_rec_dir, f"{mode}_{_ts_rec}.mp4")
                _recorder = VideoRecorder(
                    max_mb=int(cfg.get("capture_video_max_mb", 2048)),
                    codec=str(cfg.get("capture_video_codec", "mp4v")),
                )
                _recorder.start(
                    _rec_path,
                    fps=max(1.0, fps),
                    frame_size=(int(display_size[0]), int(display_size[1])),
                )
                _toast_text = "REC START"
                _toast_until_t = time.monotonic() + 2.0

        _th_idle, _nir_idle = _capture_hardware_idle_flags(mode, mode_switch_pending, cfg)
        thermal_cap.set_idle(_th_idle)
        nir_cap.set_idle(_nir_idle)

        if mode_switch_pending is not None:
            nir_raw = nir_cap.get_latest()
            thermal_raw = thermal_cap.get_latest()
            if (
                (mode_switch_pending == "imx" and nir_raw is None)
                or (mode_switch_pending == "thermal" and thermal_raw is None)
                or (mode_switch_pending == "fusion" and (nir_raw is None or thermal_raw is None))
            ):
                time.sleep(0.02)
                continue
            t = mode_switch_pending
            if t == "imx" and nir_raw is not None:
                nir_enhanced = nir_enhancer.process(nir_raw)
            elif t == "thermal" and thermal_raw is not None:
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
                    rain_processor.reset()
                mode = entering
                mode_switch_pending = None
                shake.reset()
                jerk_gate.reset()
                # #region agent log
                _agent_debug_log(
                    hypothesis_id="H3",
                    location="main.py:mode_switch_complete",
                    message="entered_mode",
                    data={
                        "mode": mode,
                        "from_pending": entering,
                        "display_profile": str(cfg.get("display_profile", "")),
                        "show_raw_imx": bool(show_raw.get("imx")),
                    },
                )
                # #endregion
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

        _frame_cache: Optional[FrameCache] = None
        if nir_raw is not None:
            with _stage_profiler("framecache"):
                _fc_skip_160 = _lean_lk
                _fc_mss = int(np.clip(int(cfg.get("framecache_small_max_side", 128)), 64, 128))
                _frame_cache = build_frame_cache(
                    nir_raw,
                    thermal_raw if thermal_raw is not None else np.zeros((62, 80), dtype=np.uint8),
                    time.monotonic(),
                    skip_nir_160=_fc_skip_160,
                    small_max_side=_fc_mss,
                )

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
            # Guided filter: pass thermal guide only when enabled (adds ~3–5 ms/frame).
            _guided_enabled = bool(cfg.get("nir_guided_filter_enable", False))
            _th_guide = (
                thermal_raw
                if (_guided_enabled and mode == "fusion" and thermal_raw is not None)
                else None
            )
            _fc_small = _frame_cache.nir_320 if _frame_cache is not None else None
            # Bucket dispatch: route to purpose-built optical function per ENV_CLASS.
            _nir_lite = bool(cfg.get("nir_optical_lite", False))
            _table_bucket = resolve_optical_bucket(_stable_env_class, lite=_nir_lite)
            _is_night_env = _stable_env_class in NIR_NIGHT_ENV_CLASSES
            # Profile RAW: force NIR passthrough regardless of bucket/env. This matches config.py contract:
            # "raw" = no CLAHE / no HybridNIR / no fusion blend; unavoidable BGR layout only.
            if str(cfg.get("display_profile", "quality")) == "raw":
                nir_enhanced = nir_raw
                _bucket = "RAW_PROFILE"
                # #region agent log
                _agent_debug_log(
                    hypothesis_id="H6",
                    location="main.py:nir_bucket",
                    message="raw_profile_forces_passthrough",
                    data={
                        "mode": mode,
                        "env_class": _stable_env_class,
                        "table_bucket": _table_bucket,
                        "nir_brightness": float(nir_brightness),
                        "nir_non_night_raw_latched": bool(_nir_non_night_raw_latched),
                    },
                )
                # #endregion
            else:
                if not _is_night_env:
                    if _nir_non_night_raw_latched:
                        if float(nir_brightness) < NIR_NON_NIGHT_RAW_EXIT:
                            _nir_non_night_raw_latched = False
                    else:
                        if float(nir_brightness) >= NIR_NON_NIGHT_RAW_ENTER:
                            _nir_non_night_raw_latched = True
                with _stage_profiler("nir_bucket"):
                    if _is_night_env:
                        _bucket = _table_bucket
                        if _bucket == "A":
                            nir_enhanced = nir_enhancer.process(
                                nir_raw, thermal_guide=_th_guide, precomputed_small=_fc_small,
                            )
                        elif _bucket == "B":
                            nir_enhanced = nir_nir_night_clahe(
                                nir_raw,
                                clahe_clip_scale=float(cfg.get("nir_enhancer_clahe_clip_scale", 1.0)),
                            )
                        elif _bucket == "C":
                            nir_enhanced = nir_anti_glare_bgr(
                                nir_raw,
                                high_pct=float(opt_cfg_base["nir_high_pct"]),
                                saturate_at=float(opt_cfg_base["nir_saturate_at"]),
                            )
                        elif _bucket == "D":
                            nir_enhanced = nir_dehaze_lite(
                                nir_raw, omega=float(cfg.get("fog_dehaze_omega", 0.85)),
                            )
                        elif _bucket == "E":
                            _rp = rain_processor_lite if _nir_lite else rain_processor
                            nir_enhanced = _rp.process(nir_raw)
                        elif _bucket == "F":
                            _b_norm = float(nir_b_ema) / 255.0 if nir_b_ema is not None else 0.3
                            nir_enhanced = nir_transition_blend(
                                nir_raw, nir_enhancer, _b_norm, precomputed_small=_fc_small,
                            )
                        else:
                            nir_enhanced = nir_enhancer.process(
                                nir_raw, thermal_guide=_th_guide, precomputed_small=_fc_small,
                            )
                    else:
                        if _nir_non_night_raw_latched:
                            nir_enhanced = nir_raw
                            _bucket = "RAW"
                        else:
                            _bucket = "B"
                            nir_enhanced = nir_nir_night_clahe(
                                nir_raw,
                                clahe_clip_scale=float(cfg.get("nir_enhancer_clahe_clip_scale", 1.0)),
                            )
            # BRISQUE: subsampled 1/30 on night-relevant buckets A/B/D/E (and B for non-night dim path)
            if _iqa_enabled and nir_enhanced is not None and _bucket in ("A", "B", "D", "E"):
                _iqa_frame_idx += 1
                if _iqa_frame_idx % 30 == 0:
                    _bq = compute_brisque_score(nir_enhanced)
                    if _bq is not None:
                        thesis_metrics.record_brisque(_bucket, _bq)
        elif mode == "thermal" and nir_raw is not None:
            # [ENV] NIR brightness EMA for auto_rule while in thermal-only display mode
            _nbm = _nir_mean_brightness_bgr(nir_raw, nir_brightness_subsample)
            if nir_b_ema is None:
                nir_b_ema = _nbm
            else:
                nir_b_ema = (1.0 - NIR_B_EMA) * nir_b_ema + NIR_B_EMA * _nbm

        # ThermalProcessor: run 3DNR+background only when a new thermal sample arrives
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
                with _stage_profiler("thermal_proc"):
                    thermal_denoised, thermal_enhanced, heat_map, fg_mask = thermal_proc.process(
                        thermal_raw, compute_enhanced=(mode != "fusion")
                    )
                if _thermal_mono is not None:
                    _last_thermal_mono_processed = _thermal_mono

        # Pass FrameCache gray into JerkGate to skip internal resize/cvtColor.
        with _stage_profiler("jerk"):
            jerk_gate.update(
                nir_raw,
                precomputed_gray=_frame_cache.nir_gray if _frame_cache is not None else None,
            )
        # Sparse LK flow update for ML motion features (skipped in lean LK-off mode).
        if nir_raw is not None and _frame_cache is not None and not _lean_lk:
            if _frame_cache.nir_160 is not None:
                _nir_160_gray = _frame_cache.nir_160[:, :, 1]
            else:
                _nir_160_gray = cv.pyrDown(_frame_cache.nir_320)[:, :, 1]
            lk_flow.update(None, nir_gray_small=_nir_160_gray)
        soft_motion_active = bool(
            bool(cfg.get("fix_nir_motion_gating", True)) and jerk_gate.near_active and not jerk_gate.active
        )
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

        # Cache NIR luminance ONCE per frame before the ENV block.
        # Reuses the green-channel proxy from FrameCache; avoids redundant resizes in
        # brightness/std helpers and the NIR glare metric path.
        # Caches glare tuple from the ENV block for reuse on the display path when possible.
        _nir_gray_cache: Optional[np.ndarray] = None
        _nir_brightness_cache: Optional[float] = None
        # _nir_glare_result_cache: (need_compress, hud) or None if not yet computed this frame
        _nir_glare_result_cache: Optional[tuple] = None

        if mode in ("imx", "fusion") and nir_raw is not None:
            if _frame_cache is not None:
                # Reuse FrameCache gray for ENV statistics (no extra resize).
                _nir_gray_cache = _frame_cache.nir_gray
                _nir_brightness_cache = float(np.mean(_frame_cache.nir_gray))
            else:
                _nir_gray_cache, _nir_brightness_cache = nir_compute_gray_cached(
                    nir_raw, subsample=nir_brightness_subsample
                )

        # [ENV] Preset apply logic
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
            _ml_secondary_hint: Optional[str] = None  # only set by auto_rule + compositor
            if _em == "manual":
                mp = str(cfg.get("env_manual_preset", "default"))
                desired = mp if mp in ENV_PRESETS else env_fb
            elif _em == "auto_rule":
                # Rate-limit expensive ENV classification to every N frames.
                _env_frame_idx += 1
                _run_env_classification = (_env_frame_idx % ENV_CLASSIFICATION_INTERVAL == 0)

                # ML-primary routing: compositor maps top-1/top-2 to (env_class, hint).
                _ml_desired_class: Optional[str] = None
                if _ml_inference_active and _ml_shared_result is not None:
                    _ml_top2 = _ml_shared_result.get()
                    _thr_primary = float(cfg.get("ml_confidence_threshold", 0.62))
                    _thr_secondary = float(cfg.get("ml_secondary_confidence_threshold", 0.20))
                    _c1 = ENV_INT_TO_CLASS.get(_ml_top2.label_int_1)
                    _c2 = ENV_INT_TO_CLASS.get(_ml_top2.label_int_2)
                    _eff, _ml_secondary_hint = compose_env_from_ml_top2(
                        class_1=_c1,
                        proba_1=_ml_top2.proba_1,
                        class_2=_c2,
                        proba_2=_ml_top2.proba_2,
                        primary_threshold=_thr_primary,
                        secondary_threshold=_thr_secondary,
                    )
                    if _eff is not None:
                        _ml_desired_class = _eff

                if _ml_desired_class is not None:
                    # ML wins: desired is already an ENV_CLASS name
                    _last_desired_env = _ml_desired_class
                    desired = _ml_desired_class
                elif nir_raw is not None and _run_env_classification:
                    # Fallback: rule-based heuristic -> map to ENV_CLASS
                    if _nir_gray_cache is not None:
                        _std = float(np.std(_nir_gray_cache.astype(np.float32)))
                        _need_g, _nir_gl, _p_hi, _p99, _mx = _nir_glare_metrics_from_gray(
                            _nir_gray_cache,
                            high_pct=float(opt_cfg_base["nir_high_pct"]),
                            saturate_at=float(opt_cfg_base["nir_saturate_at"]),
                            use_fast=True,
                        )
                        _nir_glare_result_cache = (_need_g, _nir_gl)
                    else:
                        _std = _nir_gray_std_quick(nir_raw)
                        _need_g, _nir_gl = nir_glare_eval(
                            nir_raw,
                            high_pct=float(opt_cfg_base["nir_high_pct"]),
                            saturate_at=float(opt_cfg_base["nir_saturate_at"]),
                            use_fast=True,
                        )
                        _nir_glare_result_cache = (_need_g, _nir_gl)
                    tags = infer_env_tags_auto_rule(
                        nir_b_ema=nir_b_ema,
                        nir_gray_std=_std,
                        glare_nir=_nir_gl,
                        haze_config_on=bool(cfg.get("opt_haze_preset", False)),
                        std_low=float(cfg.get("env_auto_nir_gray_std_low", 20.0)),
                        std_high=float(cfg.get("env_auto_nir_gray_std_high", 52.0)),
                    )
                    _rule_preset = select_env_preset_from_tags(tags)
                    _last_desired_env = auto_rule_preset_to_env_class(_rule_preset)
                    desired = _last_desired_env
                else:
                    # Skip expensive feature extraction: reuse last decision
                    desired = _last_desired_env
            stable = env_controller.update(desired if desired in ENV_PRESETS else env_fb)
            # Normalize to ENV_CLASS for bucket dispatch: handles manual-mode old preset names.
            _stable_env_class = (
                stable if stable in OPTICAL_BUCKET_DISPATCH
                else auto_rule_preset_to_env_class(stable)
            )
            opt_cfg_runtime = merge_opt_cfg_with_preset(opt_cfg_base, stable)
            opt_cfg_runtime = apply_secondary_hint(opt_cfg_runtime, _ml_secondary_hint)
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
                thermal_proc.update_runtime_params(opt_cfg_runtime)
                nir_enhancer.update_runtime_params(opt_cfg_runtime)
                restore_e1_from_snapshot(e1_detector, e1_defaults_snapshot)
                apply_e1_overrides(e1_detector, _pobj.get("e1_overrides") or {})
                temporal_glare.prev_weight = float(opt_cfg_runtime["temporal_prev_weight"])
                prev_env_stable = stable

        # ── ML feature logging — after ENV + FrameCache; logger never blocks the loop ──
        if (
            _ml_log_enabled
            and ml_logger is not None
            and ml_feature_extractor is not None
            and _frame_cache is not None
            and nir_raw is not None
        ):
            _ml_log_counter += 1
            _ml_iv = int(max(1, int(cfg.get("ML_LOG_INTERVAL", 5))))
            if _ml_log_counter % _ml_iv == 0:
                _th_ch = (
                    "lwir"
                    if (
                        thermal_raw is not None
                        and thermal_raw.size > 0
                        and float(thermal_raw.max()) > 0.0
                    )
                    else "none"
                )
                _rec = ml_feature_extractor.extract(
                    _frame_cache,
                    nir_channel="nir",
                    thermal_channel=_th_ch,
                    ts=time.time(),
                    prev_env_class=_prev_ml_env_int,
                    source="rpi",
                    frame_idx=_ml_log_counter,
                    has_motion=False,
                    has_temporal=True,
                )
                if not _lean_lk:
                    _rec = dataclasses.replace(
                        _rec,
                        motion_magnitude=float(lk_flow.motion_magnitude),
                        motion_jerk=float(lk_flow.motion_jerk),
                        motion_available=True,
                    )
                _rec = dataclasses.replace(
                    _rec,
                    fusion_alpha=float(alpha),
                )
                _log_dict = _rec.to_dict()
                # Add latest ML prediction fields to the JSONL record (additive schema).
                if _ml_inference_active and _ml_shared_result is not None:
                    _ml_top2_log = _ml_shared_result.get()
                    _log_dict["ml_env_label"] = _ml_top2_log.label_int_1
                    _log_dict["ml_confidence"] = round(_ml_top2_log.proba_1, 4)
                    _log_dict["ml_env_label_2"] = _ml_top2_log.label_int_2
                    _log_dict["ml_confidence_2"] = round(_ml_top2_log.proba_2, 4)
                ml_logger.log(_log_dict)

        # ── ML inference dispatch (observe-only; non-blocking queue drop on overload) ──
        if (
            _ml_inference_active
            and _ml_infer_queue is not None
            and _ml_infer_extractor is not None
            and _frame_cache is not None
            and nir_raw is not None
        ):
            _ml_infer_counter += 1
            _ml_iv2 = int(max(1, int(cfg.get("ML_INFERENCE_INTERVAL", 15))))
            if _ml_infer_counter % _ml_iv2 == 0:
                try:
                    with _stage_profiler("ml_infer"):
                        _th_ch2 = (
                            "lwir"
                            if (
                                thermal_raw is not None
                                and thermal_raw.size > 0
                                and float(thermal_raw.max()) > 0.0
                            )
                            else "none"
                        )
                        _infer_rec = _ml_infer_extractor.extract(
                            _frame_cache,
                            nir_channel="nir",
                            thermal_channel=_th_ch2,
                            ts=time.time(),
                            prev_env_class=_prev_ml_env_int,
                            source="rpi",
                            frame_idx=_ml_infer_counter,
                            has_motion=False,
                            has_temporal=True,
                        )
                        _infer_vec = _infer_rec.to_feature_array(FEATURE_SET_OPTICAL_ONLY)
                        _ml_infer_queue.put_nowait(_infer_vec)
                except queue.Full:
                    pass  # queue full — drop stale dispatch, thread will catch up
                except Exception:  # noqa: BLE001
                    logger.debug("ml feature extract failed", exc_info=True)

        # E1 effective state: config flag AND CLI lean AND runtime toggle ('e' key).
        _e1_active: bool = phase1_e1_enabled and not _lean_e1 and _e1_runtime_enabled
        if mode in ("thermal", "fusion") and thermal_raw is not None and _e1_active:
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
                    # MAD detector may replace E1 blobs when active (skipped if lean MAD off).
                    if not _lean_mad:
                        _mad_score, _mad_active, _mad_blobs = mad_detector.process(
                            thermal_denoised if thermal_denoised is not None else thermal_raw,
                            fg_mask,
                            heat_map,
                            jerk_active=jerk_gate.active,
                        )
                        if _mad_active and _mad_blobs:
                            phase1_blobs = _mad_blobs
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
            use_raw = nir_use_raw_auto
            if use_raw:
                src = nir_raw
            else:
                src = nir_enhanced if nir_enhanced is not None else nir_raw
            src_in = src
            glare_nir = False
            if thermal_proc.anti_glare_enabled:
                # Reuse cached glare result from ENV when available.
                if _nir_glare_result_cache is not None:
                    nir_need_compress, glare_nir = _nir_glare_result_cache
                else:
                    nir_need_compress, glare_nir = nir_glare_eval(
                        src_in,
                        high_pct=float(opt_cfg_runtime["nir_high_pct"]),
                        saturate_at=float(opt_cfg_runtime["nir_saturate_at"]),
                        use_fast=bool(cfg.get("perf_glare_lite", True)),
                    )
                _skip_dag = nir_skip_second_anti_glare_bgr(
                    bucket=_bucket,
                    use_raw_nir=use_raw,
                    nir_optical_lite=bool(cfg.get("nir_optical_lite", False)),
                )
                if nir_need_compress and not _skip_dag:
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
            if False:
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
                # Display edge enhancement after AGC, before colormap (hot objects pop in grayscale).
                # 0.0 disables; strength comes from config (see thermal_display_edge_strength).
                _disp_ee = float(cfg.get("thermal_display_edge_strength", 0.0))
                if _disp_ee > 1e-6:
                    thermal_vis = thermal_edge_enhance(thermal_vis, strength=_disp_ee)
            cm = gray_to_thermal_bgr(thermal_vis, levels=_tcmap_lv)
            cm = cv.resize(cm, thermal_size)
            cm = cv.rotate(cm, cv.ROTATE_180)
            out = cv.resize(cm, display_size, interpolation=cv.INTER_LINEAR)
            if not thermal_proc.is_ready:
                cv.putText(out, f"Warming up... {thermal_proc.warmup_pct}%",
                           (10, 48), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            if thermal_proc.anti_glare_enabled and thermal_denoised is not None:
                glare_th = thermal_glare_hud_should_apply(
                    thermal_denoised,
                    saturate_at=min(254.0, float(opt_cfg_runtime["thermal_saturate_at"]) + 12.0),
                )

        else:  # fusion
            with _stage_profiler("fusion_composite"):
                use_raw = nir_use_raw_auto
                nir_src = nir_raw if use_raw else (nir_enhanced if nir_enhanced is not None else nir_raw)
                nir_in = nir_src
                glare_nir = False
                # Fusion: no second nir_anti_glare_bgr (optical bucket already processed NIR; only metrics for HUD/L cap).
                if thermal_proc.anti_glare_enabled:
                    if _nir_glare_result_cache is not None:
                        nir_need_compress, glare_nir = _nir_glare_result_cache
                    else:
                        nir_need_compress, glare_nir = nir_glare_eval(
                            nir_in,
                            high_pct=float(opt_cfg_runtime["nir_high_pct"]),
                            saturate_at=float(opt_cfg_runtime["nir_saturate_at"]),
                            use_fast=bool(cfg.get("perf_glare_lite", True)),
                        )
                else:
                    nir_need_compress = False
                if thermal_proc.anti_glare_enabled and thermal_denoised is not None:
                    glare_th = thermal_glare_hud_should_apply(
                        thermal_denoised,
                        saturate_at=min(254.0, float(opt_cfg_runtime["thermal_saturate_at"]) + 12.0),
                    )
                with _fuse_sp("fuse_nir_resize"):
                    nir_work = cv.resize(nir_src, (_nw_fuse, _nh_fuse), interpolation=_nir_inter_fuse)
                alpha_runtime = min(1.0, alpha + float(opt_cfg_runtime["fusion_alpha_boost"]))
                _did_fusion = False
                if heat_map is not None and thermal_proc.is_ready:
                    with _fuse_sp("fuse_thermal_prep"):
                        # Resize + rotate grayscale; preserved for heat_norm in gradient mode.
                        hm_r = cv.resize(heat_map, thermal_size, interpolation=_fusion_hm_inter)
                        hm_r = cv.rotate(hm_r, cv.ROTATE_180)
                        _hm_ee = float(cfg.get("thermal_display_edge_strength", 0.0))
                        _skip_hm_ee = bool(cfg.get("fusion_skip_heatmap_display_edge_enhance", False))
                        # Edge-enhance on native 80×62 (before resize) — LUT then runs on small grid.
                        hm_vis_native = thermal_edge_enhance(heat_map, strength=_hm_ee) if (_hm_ee > 1e-6 and not _skip_hm_ee) else heat_map
                    with _fuse_sp("fuse_colormap"):
                        # LUT on native 80×62, then resize+rotate BGR — saves ~15× LUT work vs thermal_size.
                        hm_color = cv.rotate(
                            cv.resize(
                                gray_to_thermal_bgr(hm_vis_native, levels=_tcmap_lv),
                                thermal_size,
                                interpolation=_fusion_hm_inter,
                            ),
                            cv.ROTATE_180,
                        )
                    if _fusion_mode_pre == "gradient" and int(np.max(hm_r)) > 0:
                        # Alpha ∝ local heat; no fg_mask gate.
                        heat_norm = hm_r.astype(np.float32) * (1.0 / 255.0)
                        bgra = np.dstack([hm_color, (heat_norm * 255).astype(np.uint8)])
                        bgra_w = cv.warpPerspective(bgra, _H_fuse, (_nw_fuse, _nh_fuse))
                        hm_warped = bgra_w[:, :, :3]
                        heat_warped = bgra_w[:, :, 3].astype(np.float32) * (1.0 / 255.0)
                        alpha_map = np.clip(heat_warped * _heat_base_pre * float(alpha_runtime), 0.0, 1.0)
                        m3 = alpha_map[:, :, np.newaxis]
                        np.multiply(nir_work, np.float32(1.0), out=_fuse_nir_f, casting="unsafe")
                        # hm_warped is uint8; copy into float buffer then subtract in-place to avoid uint8 wrap-on-underflow.
                        np.copyto(_fuse_hm_f, hm_warped, casting="unsafe")
                        np.subtract(_fuse_hm_f, _fuse_nir_f, out=_fuse_hm_f)
                        np.multiply(_fuse_hm_f, m3, out=_fuse_out_f)
                        np.add(_fuse_nir_f, _fuse_out_f, out=_fuse_out_f)
                        _fuse_u8 = cv.convertScaleAbs(_fuse_out_f)
                        out = cv.resize(_fuse_u8, display_size, interpolation=cv.INTER_LINEAR) if display_size != (_nw_fuse, _nh_fuse) else _fuse_u8
                        _did_fusion = True
                    elif _fusion_mode_pre == "fg_mask" and fg_mask is not None and np.any(fg_mask > 0):
                        with _fuse_sp("fuse_warp_prep"):
                            fg_r = cv.resize(fg_mask, thermal_size, interpolation=cv.INTER_NEAREST)
                            fg_r = cv.rotate(fg_r, cv.ROTATE_180)
                            # Single 4-channel warp: merge hm_color (BGR) + fg_r (A) to halve perspective divide cost.
                            bgra = np.dstack([hm_color, fg_r])
                        with _fuse_sp("fuse_warp_perspective"):
                            bgra_w = cv.warpPerspective(bgra, _H_fuse, (_nw_fuse, _nh_fuse), flags=cv.INTER_LINEAR)
                        hm_warped = bgra_w[:, :, :3]
                        fg_w = bgra_w[:, :, 3]
                        with _fuse_sp("fuse_blur_fg"):
                            if _mbk_pre >= 3:
                                fg_w = cv.GaussianBlur(fg_w, (_mbk_pre, _mbk_pre), _mbs_pre)
                        with _fuse_sp("fuse_blend_math"):
                            np.multiply(fg_w, np.float32(float(alpha_runtime) / 255.0), out=_fuse_alpha_f, casting="unsafe")
                            m3 = _fuse_alpha_f[:, :, np.newaxis]
                            np.multiply(nir_work, np.float32(1.0), out=_fuse_nir_f, casting="unsafe")
                            # hm_warped is uint8; copy into float buffer then subtract in-place to avoid uint8 wrap-on-underflow.
                            np.copyto(_fuse_hm_f, hm_warped, casting="unsafe")
                            np.subtract(_fuse_hm_f, _fuse_nir_f, out=_fuse_hm_f)
                            np.multiply(_fuse_hm_f, m3, out=_fuse_out_f)
                            np.add(_fuse_nir_f, _fuse_out_f, out=_fuse_out_f)
                            _fuse_u8 = cv.convertScaleAbs(_fuse_out_f)
                            out = cv.resize(_fuse_u8, display_size, interpolation=cv.INTER_LINEAR) if display_size != (_nw_fuse, _nh_fuse) else _fuse_u8
                        _did_fusion = True
                if not _did_fusion:
                    out = cv.resize(nir_src, display_size, interpolation=cv.INTER_LINEAR)
                if not thermal_proc.is_ready:
                    cv.putText(out, f"BG warming... {thermal_proc.warmup_pct}%",
                               (10, 48), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Guard against double IIR: temporal_glare.process() applies IIR when
        # nir_need_compress and not jerk; shake.process(mode="blend") ALSO applies IIR.
        # Fix: when temporal blend was applied, pass soft_motion_active=True to shake
        # so shake skips its own blend (both guards lead to pass-through).
        with _stage_profiler("blend"):
            _temporal_glare_on = bool(cfg.get("display_temporal_glare_enable", True))
            # No temporal IIR and no shake: grade only (imx/fusion).
            _blend_direct = (
                not _temporal_glare_on
                and _shake_mode == "off"
                and mode in ("imx", "fusion")
            )
            if _blend_direct:
                _temporal_blend_applied = False
            else:
                if mode in ("imx", "fusion") and _temporal_glare_on:
                    _apply_temporal = bool(nir_need_compress and not jerk_gate.active)
                    _reset_temporal = bool(jerk_gate.active)
                    if bool(cfg.get("fix_nir_motion_gating", True)) and soft_motion_active:
                        _apply_temporal = False
                        _reset_temporal = True
                    out = temporal_glare.process(out, _apply_temporal, reset=_reset_temporal)
                    _temporal_blend_applied = _apply_temporal and not _reset_temporal
                else:
                    _temporal_blend_applied = False
                out = shake.process(
                    out,
                    jerk_active=jerk_gate.active,
                    soft_motion_active=(soft_motion_active or _temporal_blend_applied),
                )

            # Single BGR↔LAB round trip: grade and luminance cap combined.
            _dgm = str(cfg.get("display_grade_mode", "full")).strip().lower()
            _l_cap = int(opt_cfg_runtime["display_l_max"])
            if glare_nir or glare_th:
                _l_cap = min(_l_cap, int(opt_cfg_runtime["display_l_max_when_glare"]))
            if _dgm == "luma_only":
                _skip_luma_cap = bool(cfg.get("display_luma_cap_glare_gate", False)) and not (glare_nir or glare_th)
                if not _skip_luma_cap:
                    out = display_luminance_cap_bgr(out, l_max=_l_cap)
            else:
                grade_params = dict(env_display_grade) if env_display_grade else {}
                grade_params["l_max"] = _l_cap
                out = display_grade_and_cap_bgr(out, **grade_params)

        _perf_frame_idx += 1
        if _perf_frame_idx % _perf_report_interval == 0:
            jerk_saved_ms = float(jerk_gate.get_saved_ms_estimate())
            fps_after = max(0.1, float(fps))
            fps_before_est = 1.0 / ((1.0 / fps_after) + (jerk_saved_ms / 1000.0))
            print(
                "[PERF] FPS est before="
                f"{fps_before_est:.2f} after={fps_after:.2f} "
                f"(jerk_gate~{jerk_saved_ms:.3f}ms/frame est.)"
            )

        _hud_bear = bool(cfg.get("hud_bearing_enabled", True))
        if _hud_bear:
            _wd, _hd = int(display_size[0]), int(display_size[1])
            _u = float(np.clip(a1_probe_xy[0], 0, max(0, _wd - 1)))
            _v = float(np.clip(a1_probe_xy[1], 0, max(0, _hd - 1)))
            bear_h, bear_v = bearing_hv_deg_from_uv(_u, _v, _wd, _hd, fov_h, fov_v)
        else:
            bear_h, bear_v = 0.0, 0.0

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
            jerk_active=jerk_gate.active,
            glare_nir=glare_nir,
            glare_th=glare_th,
            thermal_is_ready=thermal_proc.is_ready if mode in ("thermal", "fusion") else True,
            env_class=_stable_env_class,
            optical_bucket=_bucket,
        )

        # EI person-in-dark: same BGR as the on-screen frame for the current mode/profile
        # (imx / thermal / fusion), after grade/cap and light overlays; before L1 HUD text.
        # Preprocess (crop/resize/quantize) runs here; invoke() is off-thread.
        if _ei_worker is not None:
            _ei_frame_idx += 1
            if (
                out is not None
                and int(getattr(out, "size", 0) or 0) > 0
                and _ei_frame_idx % max(1, _ei_infer_interval) == 0
            ):
                if _ei_worker.submit_frame(_ei_frame_idx, out):
                    _ei_metrics["frames_submitted"] += 1
                else:
                    _ei_metrics["frames_dropped_overflow"] += 1

        # ── HUD Layer 1 (L1 — data — burned into all saved frames) ──────────
        _profile_name = str(cfg.get("display_profile", "quality"))
        _profile_verified_map = cfg.get(
            "display_profile_verified", {"quality": True, "throughput": True, "raw": False}
        )
        _profile_verified = bool(
            _profile_verified_map.get(_profile_name, False)
            if isinstance(_profile_verified_map, dict)
            else _profile_verified_map
        )
        _now_mono = time.monotonic()
        _active_toast: Optional[str] = _toast_text if _now_mono < _toast_until_t else None
        _hud_state = HudState(
            mode=mode,
            fps=fps,
            alpha=alpha,
            display_size=(int(display_size[0]), int(display_size[1])),
            jerk_active=bool(jerk_gate.active),
            glare_nir=bool(glare_nir),
            glare_th=bool(glare_th),
            soft_motion_active=bool(soft_motion_active),
            haze_active=bool(cfg.get("opt_haze_preset", False)),
            env_mode_active=str(cfg.get("env_mode", "off")).strip().lower() != "off",
            env_stable=env_stable_for_hud,
            lean_active=bool(_lean_lk or _lean_e1 or _lean_mad),
            e1_off=bool(phase1_e1_enabled and not _lean_e1 and not _e1_runtime_enabled),
            bear_h=bear_h,
            bear_v=bear_v,
            hud_bear_enabled=_hud_bear,
            a1_probe_xy=(int(a1_probe_xy[0]), int(a1_probe_xy[1])),
            utc_clock_enabled=bool(cfg.get("hud_utc_clock_enabled", True)),
            utc_time=time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
            profile_label=_profile_name.upper(),
            profile_verified=_profile_verified,
            net_location=None,  # P2.5: populated by NetLocReader when implemented
            capture_indicator=_active_toast,
        )
        with _stage_profiler("hud"):
            out = draw_hud(out, _hud_state)

        # L1 raster — saved to disk (PNG stills and video). Never contains L2 chrome.
        saveable_out = out

        # ── Video recording ───────────────────────────────────────────────────
        if _recorder is not None and _recorder.is_active:
            _recorder.write_nowait(saveable_out)
            if _recorder.disk_limit_hit:
                _recorder.stop()
                _recorder = None
                _toast_text = "REC STOPPED: DISK"
                _toast_until_t = time.monotonic() + 3.0

        # ── Auto capture ──────────────────────────────────────────────────────
        _auto_countdown_text: Optional[str] = None
        if auto_start is not None:
            elapsed = time.time() - auto_start
            countdown = max(0, AUTO_DELAY - int(elapsed))
            if countdown > 0:
                _auto_countdown_text = f"Capture in {countdown}s..."
            else:
                ts = time.strftime("%Y%m%d-%H%M%S")
                path = os.path.join(save_dir, f"auto_{mode}_{ts}.png")
                cv.imwrite(path, saveable_out)
                write_capture_meta(
                    path, "timer_auto",
                    mode=mode, jerk_gate=jerk_gate,
                    glare_nir=glare_nir, glare_th=glare_th,
                    soft_motion_active=soft_motion_active, cfg=cfg,
                    env_stable_for_hud=env_stable_for_hud,
                    phase1_alert_enabled=phase1_alert_enabled,
                    sector_alert_text=sector_alert_text,
                    phase1_e1_enabled=phase1_e1_enabled,
                    phase1_blobs=phase1_blobs,
                    thesis_metrics=thesis_metrics,
                    fps=fps, alpha=alpha, nir_brightness=nir_brightness,
                    nir_b_ema=nir_b_ema, nir_use_raw_auto=nir_use_raw_auto,
                    bear_h=bear_h, bear_v=bear_v,
                    frame=saveable_out, profile_label=_profile_name,
                )
                print("Saved:", path, "+ metadata JSON")
                auto_start = None

        # ── Manual still capture (s key or [CAP] button) ─────────────────────
        if key == ord("s"):
            ts = time.strftime("%Y%m%d-%H%M%S")
            path = os.path.join(save_dir, f"{mode}_{ts}.png")
            cv.imwrite(path, saveable_out)
            write_capture_meta(
                path, "key_s",
                mode=mode, jerk_gate=jerk_gate,
                glare_nir=glare_nir, glare_th=glare_th,
                soft_motion_active=soft_motion_active, cfg=cfg,
                env_stable_for_hud=env_stable_for_hud,
                phase1_alert_enabled=phase1_alert_enabled,
                sector_alert_text=sector_alert_text,
                phase1_e1_enabled=phase1_e1_enabled,
                phase1_blobs=phase1_blobs,
                thesis_metrics=thesis_metrics,
                fps=fps, alpha=alpha, nir_brightness=nir_brightness,
                nir_b_ema=nir_b_ema, nir_use_raw_auto=nir_use_raw_auto,
                bear_h=bear_h, bear_v=bear_v,
                frame=saveable_out, profile_label=_profile_name,
            )
            print("Saved:", path, "+ metadata JSON")

        # ── L2 control chrome (on-screen only; excluded from saved rasters) ───
        _debug_on = bool(cfg.get("debug", False))
        _now_chrome = time.monotonic()
        _chrome_alpha = (
            _CHROME_ACTIVE_ALPHA
            if _now_chrome - _last_chrome_touch_t
               <= float(cfg.get("controls_chrome_autofade_s", 5.0))
            else _CHROME_FADE_ALPHA
        )
        _chrome_state = ChromeState(
            active_mode=mode,
            is_recording=_recorder is not None and _recorder.is_active,
            display_profile=_profile_name,
            chrome_alpha=_chrome_alpha,
            display_size=(int(display_size[0]), int(display_size[1])),
            rec_dropped=_recorder.dropped_count if _recorder is not None else 0,
        )
        with _stage_profiler("chrome"):
            display_out = draw_control_chrome(saveable_out.copy(), _chrome_state)

        # Countdown overlay on display only (never burned into saved frames)
        if _auto_countdown_text is not None:
            cv.putText(
                display_out, _auto_countdown_text, (display_size[0] // 2 - 80, 48),
                cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1,
            )

        # ── Debug overlays (ML + EI — display only; excluded from saved rasters) ─
        # A1: EI metrics always accumulate (not gated by debug); only the overlay is.
        # See docs/eval/ei_person_find_in_dark/DECISIONS_AND_RISKS.md Q3.
        _ml_name1 = _ml_name2 = ""
        _ml_p1 = _ml_p2 = 0.0
        if _ml_inference_active and _ml_shared_result is not None and _debug_on:
            _ml_top2_hud = _ml_shared_result.get()
            _ml_name1 = (
                _ml_classifier.label_name(_ml_top2_hud.label_int_1)
                if _ml_classifier is not None
                else f"cls{_ml_top2_hud.label_int_1}"
            )
            _ml_name2 = (
                _ml_classifier.label_name(_ml_top2_hud.label_int_2)
                if _ml_classifier is not None
                else f"cls{_ml_top2_hud.label_int_2}"
            )
            _ml_p1 = _ml_top2_hud.proba_1
            _ml_p2 = _ml_top2_hud.proba_2

        _ei_n = 0
        _ei_bboxes_for_hud = None
        _ei_best_str = "--"
        _ei_age_ms_hud = 0.0
        _ei_inf_ms_hud = 0.0
        _ei_stale_hud = False
        if _ei_enabled and _ei_shared is not None:
            _ei_r = _ei_shared.get()
            _ei_n = len(_ei_r.detections)
            _ei_metrics["inferences_ok"] += (0 if _ei_r.stale else 1)
            _ei_metrics["inferences_err"] += (1 if _ei_r.stale else 0)
            if not _ei_r.stale and _ei_r.inference_ms > 0:
                _ei_metrics["inference_ms_samples"].append(_ei_r.inference_ms)
            _ei_metrics["detections_per_frame"].append(_ei_n)
            if _debug_on:
                _ei_best_str = f"{_ei_r.detections[0].score:.2f}" if _ei_n > 0 else "--"
                _ei_age_ms_hud = (
                    (time.perf_counter() - _ei_r.timestamp_monotonic) * 1000.0
                    if _ei_r.timestamp_monotonic > 0 else 0.0
                )
                _ei_inf_ms_hud = _ei_r.inference_ms
                _ei_stale_hud = _ei_r.stale
                _ei_bboxes_for_hud = _ei_r.detections if _ei_draw_bbox else None

        if _debug_on:
            draw_debug_overlays(
                display_out,
                ml_active=bool(_ml_name1),
                ml_name1=_ml_name1,
                ml_proba1=_ml_p1,
                ml_name2=_ml_name2,
                ml_proba2=_ml_p2,
                ei_enabled=_ei_enabled,
                ei_n=_ei_n,
                ei_best=_ei_best_str,
                ei_inference_ms=_ei_inf_ms_hud,
                ei_age_ms=_ei_age_ms_hud,
                ei_stale=_ei_stale_hud,
                ei_bboxes=_ei_bboxes_for_hud,
            )

        with _stage_profiler("display"):
            cv.imshow("SmartBinocular", display_out)

        # Update regardless of log guard: inference also reads prev_env_class and must not see a stale 0.
        if _ml_log_enabled or _ml_inference_active:
            _prev_ml_env_int = int(ENV_CLASS_TO_INT.get(env_stable_for_hud, ENV_INT_UNKNOWN))

    _cleanup_all()
    thesis_metrics.fusion_alpha = alpha
    thesis_metrics.record_ei_metrics(_ei_metrics if _ei_enabled else None)
    rep = thesis_metrics.finalize()
    _ts = time.strftime("%Y%m%d-%H%M%S")
    _session_path = os.path.join(metrics_dir, f"session_{_ts}.json")
    _metrics_write_json(_session_path, rep)
    print(thesis_metrics.summary_line(rep))
    print("Session metrics report:", _session_path)
    print("Exiting.")


if __name__ == "__main__":
    main()
