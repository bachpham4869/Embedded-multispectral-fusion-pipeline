"""
Session-level metrics, capture-sidecar JSON schema, and run manifests.

``main`` constructs :class:`ThesisRunMetrics`, calls :meth:`~ThesisRunMetrics.record_frame`
each loop iteration, and writes ``session_*.json`` on exit. Run manifests are written
once at startup beside capture metadata.
"""

from __future__ import annotations

import glob as _glob
import hashlib
import json
import math
import numpy as np
import os
import platform
import re as _re
import shutil
import subprocess
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


_METRICS_SCHEMA_VERSION = "1.5-opt"

_HOME_PATH_RE = _re.compile(r"(/(?:Users|home)/[^/\s]+)")


def _redact_paths(obj: Any) -> Any:
    """Recursively sanitize a capture-sidecar dict before writing to disk.

    Replaces /Users/<name>/... and /home/<name>/... with <HOME>/... in string
    values, and drops any dict key whose name starts with ``_`` (internal state).
    """
    if isinstance(obj, str):
        return _HOME_PATH_RE.sub("<HOME>", obj)
    if isinstance(obj, dict):
        return {k: _redact_paths(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_redact_paths(x) for x in obj]
    return obj


# Public alias for use from main.py
redact_capture_paths = _redact_paths

# Allowlist of CONFIG keys included in the pipeline fingerprint SHA-256.
# Seeded from config._VALID_OPT_OVERRIDES_KEYS; extended with global pipeline knobs.
# Adding a key here is safe; removing one changes the hash for existing sessions.
_PIPELINE_FINGERPRINT_KEYS: frozenset = frozenset({
    # Per-preset runtime overrides (mirrors _VALID_OPT_OVERRIDES_KEYS in config.py)
    "nir_high_pct", "nir_saturate_at", "nir_gamma", "temporal_prev_weight",
    "thermal_floor", "thermal_fg_threshold", "thermal_fg_max_ratio",
    "fusion_alpha_boost", "display_l_max", "display_l_max_when_glare",
    "nir_enhancer_clahe_clip_scale", "nir_enhancer_detail_strength",
    "fog_dehaze_omega", "rain_median_frames",
    "thermal_bilateral_d", "thermal_bilateral_sigma_color", "thermal_bilateral_sigma_space",
    "thermal_high_pct", "thermal_saturate_at", "thermal_gamma",
    "thermal_edge_strength", "thermal_agc_low_pct", "thermal_agc_high_pct",
    # Global pipeline knobs that affect timing and quality
    "fusion_warp_work_scale",
    "nir_enhancer_proc_w",
    "nir_enhancer_proc_h",
    "nir_hybrid_update_rate",
    "nir_optical_lite",
    "thermal_colormap_levels",
    "display_grade_mode",
    "rpi_throughput_max",
    "ml_confidence_threshold",
    "ml_posterior_ema_alpha",
    "fusion_alpha",
})

# Public aliases
SCHEMA_VERSION = _METRICS_SCHEMA_VERSION


def _metrics_write_json(path: str, data: Dict[str, Any]) -> None:
    """Write ``data`` as UTF-8 JSON to ``path`` (internal helper)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# Public alias
write_metrics_json = _metrics_write_json


def _capture_throttle_snapshot() -> Dict[str, Any]:
    """Return CPU throttle state dict. All values are None on non-Pi hosts (soft-fail)."""
    vcgencmd = shutil.which("vcgencmd") or "/usr/bin/vcgencmd"

    throttled: Optional[str] = None
    try:
        throttled = subprocess.check_output(
            [vcgencmd, "get_throttled"], timeout=2, stderr=subprocess.DEVNULL
        ).decode().strip()
    except (FileNotFoundError, subprocess.CalledProcessError, PermissionError, OSError):
        pass

    temp: Optional[str] = None
    try:
        temp = subprocess.check_output(
            [vcgencmd, "measure_temp"], timeout=2, stderr=subprocess.DEVNULL
        ).decode().strip()
    except (FileNotFoundError, subprocess.CalledProcessError, PermissionError, OSError):
        pass

    cpu_freqs: Optional[List[int]] = None
    try:
        freq_paths = sorted(_glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq"))
        if freq_paths:
            cpu_freqs = []
            for p in freq_paths:
                with open(p) as _f:
                    cpu_freqs.append(int(_f.read().strip()))
    except (FileNotFoundError, PermissionError, OSError, ValueError):
        pass

    return {
        "vcgencmd_throttled": throttled,
        "vcgencmd_temp": temp,
        "cpu_freq_khz": cpu_freqs,
        "platform": platform.machine(),
    }


def compute_pipeline_config_sha256(cfg: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Return (sha256_hex, sorted_allowlist_keys) for the pipeline-relevant CONFIG subset."""
    filtered = {k: cfg[k] for k in sorted(_PIPELINE_FINGERPRINT_KEYS) if k in cfg}
    blob = json.dumps(filtered, sort_keys=True, separators=(",", ":"), default=repr)
    sha = hashlib.sha256(blob.encode()).hexdigest()
    return sha, sorted(filtered.keys())


def _metrics_mean(xs: List[float]) -> float:
    """Arithmetic mean; empty input yields 0.0."""
    return float(sum(xs) / len(xs)) if xs else 0.0


def _metrics_std_sample(xs: List[float]) -> float:
    """Sample standard deviation (Bessel correction); n<2 yields 0.0."""
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
    model_path: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a one-time run manifest (git rev, display FOV, Tier-S flags, optional model hash)."""
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

    if model_path:
        try:
            import hashlib
            with open(model_path, "rb") as _mf:
                sha = hashlib.sha256(_mf.read()).hexdigest()[:16]
            data["ml_model_sha256_short"] = sha
        except (OSError, IOError):
            data["ml_model_sha256_short"] = None

    if extra:
        data.update(extra)
    _metrics_write_json(path, data)


@dataclass
class ThesisRunMetrics:
    """Accumulate per-frame counters and samples; :meth:`finalize` builds the session JSON."""

    session_id: str = ""
    homography_path: str = ""
    display_size: tuple = (0, 0)
    fusion_alpha: float = 0.55
    nir_schmitt_raw_on: float = 30.0
    nir_schmitt_dim_on: float = 18.0
    nir_b_ema_coef: float = 0.18
    experiment_context: Dict[str, Any] = field(default_factory=dict)
    optimization_config: Dict[str, Any] = field(default_factory=dict)
    # Full CONFIG dict for pipeline fingerprinting (set from main after CONFIG is built).
    pipeline_config: Optional[Dict[str, Any]] = field(default=None, repr=False)

    t0: float = field(default_factory=time.perf_counter)
    frame_count: int = 0
    frames_by_mode: Dict[str, int] = field(default_factory=dict)
    frames_by_env: Dict[str, int] = field(default_factory=dict)
    frames_by_bucket: Dict[str, int] = field(default_factory=dict)
    fps_samples: deque = field(default_factory=lambda: deque(maxlen=900))
    nir_brightness_samples: deque = field(default_factory=lambda: deque(maxlen=2000))
    jerk_frames: int = 0
    glare_nir_frames: int = 0
    glare_th_frames: int = 0
    thermal_not_ready_frames: int = 0
    # Optional :class:`~smartbinocular.utils.StageProfiler` wired from ``main`` after construction.
    stage_profiler: Optional[Any] = field(default=None, repr=False)
    # Inner profiler for fusion_composite sub-stages (separate instance — StageProfiler is not re-entrant).
    fuse_stage_profiler: Optional[Any] = field(default=None, repr=False)
    # Inner profiler for ThermalProcessor sub-stages (separate instance).
    thermal_stage_profiler: Optional[Any] = field(default=None, repr=False)
    # BRISQUE per-bucket accumulators (sum, sum_sq, n) — O(1) memory
    _brisque_sums: Dict[str, float] = field(default_factory=dict, repr=False)
    _brisque_sum_sqs: Dict[str, float] = field(default_factory=dict, repr=False)
    _brisque_ns: Dict[str, int] = field(default_factory=dict, repr=False)
    # EI person-in-dark metrics (None when EI_PERSON_IN_DARK_ENABLED is False; key absent from JSON)
    _ei_metrics: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def record_frame(
        self,
        *,
        mode: str,
        fps: float,
        nir_brightness: Optional[float] = None,
        nir_use_raw_auto: Optional[bool] = None,
        jerk_active: bool = False,
        glare_nir: bool = False,
        glare_th: bool = False,
        thermal_is_ready: bool = True,
        env_class: Optional[str] = None,
        optical_bucket: Optional[str] = None,
    ) -> None:
        """Update running totals for HUD conditions, FPS samples, and ENV/bucket histograms."""
        self.frame_count += 1
        self.frames_by_mode[mode] = self.frames_by_mode.get(mode, 0) + 1
        if env_class is not None:
            self.frames_by_env[env_class] = self.frames_by_env.get(env_class, 0) + 1
        if optical_bucket is not None:
            self.frames_by_bucket[optical_bucket] = self.frames_by_bucket.get(optical_bucket, 0) + 1
        self.fps_samples.append(float(fps))
        if nir_brightness is not None:
            self.nir_brightness_samples.append(float(nir_brightness))
        if jerk_active:
            self.jerk_frames += 1
        if glare_nir:
            self.glare_nir_frames += 1
        if glare_th:
            self.glare_th_frames += 1
        if not thermal_is_ready and mode in ("thermal", "fusion"):
            self.thermal_not_ready_frames += 1

    def record_ei_metrics(self, ei_metrics: Optional[Dict[str, Any]]) -> None:
        """Store EI person-in-dark counters collected during the session.

        Called at shutdown with the raw accumulator dict from main, or None when
        EI_PERSON_IN_DARK_ENABLED is False. When None, the key is absent from the session JSON.
        """
        self._ei_metrics = ei_metrics

    def record_brisque(self, bucket: str, score: float) -> None:
        """Accumulate a BRISQUE no-reference quality score for ``bucket`` (subsampled in ``main``)."""
        self._brisque_sums[bucket] = self._brisque_sums.get(bucket, 0.0) + score
        self._brisque_sum_sqs[bucket] = self._brisque_sum_sqs.get(bucket, 0.0) + score * score
        self._brisque_ns[bucket] = self._brisque_ns.get(bucket, 0) + 1

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
        jerk_active: bool = False,
        glare_nir: bool = False,
        glare_th: bool = False,
        tags_hud: Optional[List[str]] = None,
        bearing_h_deg: Optional[float] = None,
        bearing_v_deg: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Build the dict written next to a saved PNG (also signed by ``CaptureIntegrityChain``)."""
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
        return d

    def finalize(self) -> Dict[str, Any]:
        """Aggregate samples into the session summary dict (written on exit from ``main``)."""
        t1 = time.perf_counter()
        dur = t1 - self.t0
        fps_list = list(self.fps_samples)
        nir_list = list(self.nir_brightness_samples)
        n = max(1, self.frame_count)
        result: Dict[str, Any] = {
            "schema_version": _METRICS_SCHEMA_VERSION,
            "kind": "session_summary",
            "software": "smartbinocular",
            "session_id": self.session_id,
            "session_start_perf_s": self.t0,
            "duration_wall_s": round(dur, 3),
            "frames_total": self.frame_count,
            "frames_by_mode": dict(self.frames_by_mode),
            "frames_by_env": dict(self.frames_by_env),
            "frames_by_bucket": dict(self.frames_by_bucket),
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
            "thermal_warmup_frames_observed": self.thermal_not_ready_frames,
            "experiment_context": {
                **self.experiment_context,
                "throttle_snapshot": _capture_throttle_snapshot(),
            },
            "optimization_config": self.optimization_config,
            "platform": platform.platform(),
        }
        # CONFIG fingerprint — present only when pipeline_config was wired from main.
        if self.pipeline_config is not None:
            sha, keys = compute_pipeline_config_sha256(self.pipeline_config)
            result["pipeline_config_sha256"] = sha
            result["pipeline_config_keys"] = keys
        # Attach per-stage timing only when a profiler was wired from ``main``.
        if self.stage_profiler is not None:
            try:
                result["stage_timing_ms"] = self.stage_profiler.stats()
            except Exception:
                pass
        # Fusion sub-stage breakdown — present only when _fuse_sp is wired; consumers must tolerate absence.
        if self.fuse_stage_profiler is not None:
            try:
                result["fuse_stage_timing_ms"] = self.fuse_stage_profiler.stats()
            except Exception:
                pass
        # Thermal sub-stage breakdown — present only when thermal_stage_profiler is wired.
        if self.thermal_stage_profiler is not None:
            try:
                result["thermal_stage_timing_ms"] = self.thermal_stage_profiler.stats()
            except Exception:
                pass
        # BRISQUE per-bucket summary when any scores were recorded this session.
        if self._brisque_ns:
            brisque_out: Dict[str, Any] = {}
            for b, bn in self._brisque_ns.items():
                bmean = self._brisque_sums[b] / bn
                bvar = max(0.0, self._brisque_sum_sqs[b] / bn - bmean * bmean)
                brisque_out[b] = {
                    "mean": round(bmean, 3),
                    "std": round(bvar ** 0.5, 3),
                    "n": bn,
                }
            result["brisque_by_bucket"] = brisque_out
        # EI person-in-dark block — absent when EI_PERSON_IN_DARK_ENABLED is False.
        if self._ei_metrics is not None:
            _inf_ms = self._ei_metrics.get("inference_ms_samples") or []
            _dpf = self._ei_metrics.get("detections_per_frame") or []
            result["ei_person"] = {
                "frames_submitted": int(self._ei_metrics.get("frames_submitted", 0)),
                "frames_dropped_overflow": int(self._ei_metrics.get("frames_dropped_overflow", 0)),
                "inferences_ok": int(self._ei_metrics.get("inferences_ok", 0)),
                "inferences_err": int(self._ei_metrics.get("inferences_err", 0)),
                "mean_inference_ms": round(_metrics_mean(_inf_ms), 2) if _inf_ms else None,
                "p95_inference_ms": round(float(np.percentile(_inf_ms, 95, method="linear")), 2) if len(_inf_ms) >= 20 else None,
                "mean_detections_per_frame": round(_metrics_mean([float(x) for x in _dpf]), 4) if _dpf else None,
            }
        return result

    def summary_line(self, report: Optional[Dict[str, Any]] = None) -> str:
        """One-line human-readable summary for console shutdown (uses :meth:`finalize` if needed)."""
        r = report if report is not None else self.finalize()
        return (
            f"[metrics] frames={r['frames_total']} duration_s={r['duration_wall_s']:.1f} "
            f"fps_mean={r['fps_mean']:.1f}±{r['fps_std_sample']:.2f} "
            f"jerk_rate={r['rate_jerk_frames']:.3f} glare_nir_rate={r['rate_glare_nir_frames']:.3f}"
        )


def build_experiment_context(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize ``cfg['experiment']`` into manifest metadata (scenario, operator, notes)."""
    exp = cfg.get("experiment", {}) if isinstance(cfg, dict) else {}
    return {
        "scenario_type": str(exp.get("scenario_type", "unspecified")),
        "environment": str(exp.get("environment", "unspecified")),
        "lighting_level": str(exp.get("lighting_level", "unspecified")),
        "operator_id": str(exp.get("operator_id", "")),
        "build_label": str(exp.get("build_label", "fusion_live_optimized")),
        "notes_short": str(exp.get("notes_short", "")),
    }
