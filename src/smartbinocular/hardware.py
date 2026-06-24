"""
Raspberry Pi sensor I/O and calibration assets for the live pipeline.

:class:`ThermalCapture` and :class:`NIRCapture` run as daemon threads feeding
``main``. :func:`load_homography` reads the warp from packaged JSON; when the
vendor ``senxor`` package is missing, :mod:`smartbinocular.mi48_driver` supplies
a compatible MI48 shim.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2 as cv

try:
    # Production Pi: Meridian senxor (same as legacy/py/fusion_live_optimized.py)
    from senxor.mi48 import MI48
    from senxor.utils import data_to_frame
    from senxor.interfaces import SPI_Interface, I2C_Interface

    _SENXOR_BACKEND = "senxor"
except ImportError:
    from smartbinocular.mi48_driver import (
        MI48,
        SPI_Interface,
        I2C_Interface,
        data_to_frame,
    )

    _SENXOR_BACKEND = "mi48_driver"

log = logging.getLogger(__name__)
log.info("MI48 driver backend: %s", _SENXOR_BACKEND)

# ── Hardware constants ────────────────────────────────────────────────────────
RPI_GPIO_I2C_CHANNEL = 1
RPI_GPIO_SPI_BUS = 0
RPI_GPIO_SPI_CE_MI48 = 0
MI48_I2C_ADDRESS = 0x40
MI48_SPI_MAX_SPEED_HZ = 7800000
MI48_SPI_CS_DELAY = 0.0001
SPI_XFER_SIZE_BYTES = 160
THERMAL_COLORMAP = getattr(cv, "COLORMAP_TURBO", cv.COLORMAP_JET)

# Precomputed BGR tables for thermal / fusion heatmap false-color (avoids per-frame applyColorMap).
# Built once per distinct ``levels`` (32|64|128|256); fewer levels = slightly cheaper + mild banding.
_THERMAL_CMAP_LUT_BGR: Dict[int, np.ndarray] = {}


def _build_thermal_cmap_lut_bgr(levels: int) -> np.ndarray:
    levels = int(levels)
    if levels not in (32, 64, 128, 256):
        levels = 256
    idx = np.linspace(0, 255, levels, dtype=np.uint8)
    strip = cv.applyColorMap(idx.reshape(levels, 1), THERMAL_COLORMAP)
    return np.ascontiguousarray(strip.reshape(levels, 3))


def get_thermal_colormap_lut_bgr(levels: int = 256) -> np.ndarray:
    """Return a ``(L, 3)`` uint8 BGR table for :data:`THERMAL_COLORMAP` (cached)."""
    levels = int(levels)
    if levels not in (32, 64, 128, 256):
        levels = 256
    if levels not in _THERMAL_CMAP_LUT_BGR:
        _THERMAL_CMAP_LUT_BGR[levels] = _build_thermal_cmap_lut_bgr(levels)
    return _THERMAL_CMAP_LUT_BGR[levels]


def gray_to_thermal_bgr(gray_u8: np.ndarray, levels: int = 256) -> np.ndarray:
    """Map a single-channel uint8 thermal or heatmap image to BGR via LUT (numpy index; no applyColorMap).

    ``levels`` must be 32, 64, 128, or 256; invalid values default to 256.
    """
    if gray_u8.dtype != np.uint8:
        gray_u8 = np.clip(gray_u8, 0, 255).astype(np.uint8, copy=False)
    L = int(levels)
    if L not in (32, 64, 128, 256):
        L = 256
    lut = get_thermal_colormap_lut_bgr(L)
    if L == 256:
        return lut[gray_u8]
    # bins 0..L-1
    q = (gray_u8.astype(np.uint16) * (L - 1) // 255).astype(np.uint8, copy=False)
    return lut[q]


# ── Hardware imports guard ────────────────────────────────────────────────────
try:
    from smbus import SMBus
    from spidev import SpiDev
    from gpiozero import DigitalInputDevice, DigitalOutputDevice
    from picamera2 import Picamera2
    _HAS_HARDWARE = True
    log.info("Hardware imports OK (SMBus, SpiDev, gpiozero, Picamera2)")
except (ImportError, AttributeError) as e:
    err = str(e).lower()
    if "numpy" in err or "_array_api" in err or "multiarray" in err or "numpy.core" in err:
        print("NumPy 2.x is incompatible with this stack. Fix: pip3 install 'numpy<2'")
        sys.exit(1)
    _HAS_HARDWARE = False
    SMBus = SpiDev = DigitalInputDevice = DigitalOutputDevice = Picamera2 = None
    log.warning("Hardware imports unavailable (%s) — running without RPi hardware.", e)


def bearing_hv_deg_from_uv(
    u: float,
    v: float,
    w: int,
    h: int,
    fov_h_deg: float,
    fov_v_deg: float,
) -> tuple:
    """Horizontal and vertical bearing (degrees) from optical center to pixel (u,v); Tier S1 / A1 HUD."""
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    fh = math.radians(float(fov_h_deg)) * 0.5
    fv = math.radians(float(fov_v_deg)) * 0.5
    fx = cx / math.tan(fh) if fh > 1e-6 else float(w)
    fy = cy / math.tan(fv) if fv > 1e-6 else float(h)
    th = math.degrees(math.atan((u - cx) / fx))
    tv = math.degrees(math.atan((v - cy) / fy))
    return (float(th), float(tv))


def sector_from_bearing_deg(bearing_h_deg: float, center_half_width_deg: float = 6.0) -> str:
    """Map horizontal bearing to LEFT / CENTER / RIGHT for sector alerts."""
    if bearing_h_deg < -abs(center_half_width_deg):
        return "LEFT"
    if bearing_h_deg > abs(center_half_width_deg):
        return "RIGHT"
    return "CENTER"


class ThermalCapture(threading.Thread):
    """Background MI48 reader over SPI/I2C (not V4L2).

    ``last_mono`` updates on each fresh thermal sample. :meth:`set_idle` skips
    blocking reads so ``main`` can pause the sensor in IMX-only mode.
    """

    def __init__(self, flip_h: bool = False, blur_ksize: int = 3):
        """Optionally mirror horizontally and apply odd Gaussian blur kernel on raw frames."""
        super().__init__(daemon=True)
        self.latest = None
        self.last_mono: Optional[float] = None
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
        """Bring up MI48, stream frames into ``latest`` until :meth:`stop`."""
        if not _HAS_HARDWARE:
            log.warning("ThermalCapture: no hardware — thread exiting immediately.")
            return

        # ── SPI / I2C init ────────────────────────────────────────────────────
        log.info("ThermalCapture: initialising SPI (bus=%d, ce=%d, %.1f MHz)",
                 RPI_GPIO_SPI_BUS, RPI_GPIO_SPI_CE_MI48,
                 MI48_SPI_MAX_SPEED_HZ / 1e6)
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

        # ── Sensor bring-up ───────────────────────────────────────────────────
        self.mi48 = MI48([i2c, spi], data_ready=mi48_data_ready, reset_handler=Reset(mi48_reset_n))
        fw = self.mi48.fw_version
        log.info("ThermalCapture: MI48 fw_version=%s", fw)
        if int(fw[0]) >= 2:
            self.mi48.enable_filter(f1=True, f2=True, f3=False)
        self.mi48.set_fps(9)
        self.mi48.start(stream=True, with_header=True)
        log.info("ThermalCapture: MI48 stream started (fpa=%s)", self.mi48.fpa_shape)

        # ── Frame loop ────────────────────────────────────────────────────────
        _frame_count = 0
        _fps_t0 = time.monotonic()
        _consecutive_errors = 0

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
                    _consecutive_errors += 1
                    if _consecutive_errors % 10 == 1:
                        log.warning("ThermalCapture: %d consecutive None reads", _consecutive_errors)
                    continue

                img = data_to_frame(data, self.mi48.fpa_shape)
                if img is None or img.size == 0:
                    continue
                # Match legacy/py/fusion_live_optimized.py: uint8 before NORM_MINMAX (avoids minMaxIdx depth/assert
                # on float frames during MI48 boot). Constant frames: skip normalize (max==min).
                img_u8 = np.asarray(img, dtype=np.uint8)
                _mn = int(np.min(img_u8))
                _mx = int(np.max(img_u8))
                if _mx <= _mn:
                    img8u = np.full_like(img_u8, _mn, dtype=np.uint8)
                else:
                    img8u = cv.normalize(img_u8, None, 255, 0, cv.NORM_MINMAX, dtype=cv.CV_8U)
                if self.blur_ksize >= 3:
                    img8u = cv.GaussianBlur(img8u, (self.blur_ksize, self.blur_ksize), 0)
                if self.flip_h:
                    img8u = cv.flip(img8u, 1)

                with self.lock:
                    self.latest = img8u.copy()
                    self.last_mono = time.monotonic()

                _consecutive_errors = 0
                _frame_count += 1
                if _frame_count % 90 == 0:  # log FPS every ~10 s at 9 FPS
                    elapsed = time.monotonic() - _fps_t0
                    log.debug("ThermalCapture: %.1f FPS (last 90 frames)", 90.0 / max(elapsed, 1e-3))
                    _fps_t0 = time.monotonic()

            except Exception as e:
                if self.running:
                    log.error("ThermalCapture frame error: %s", e)
                    _consecutive_errors += 1

        try:
            self.mi48.stop(stop_timeout=0.5)
        except Exception:
            pass
        log.info("ThermalCapture: thread exited after %d frames", _frame_count)

    def get_latest(self) -> Optional[np.ndarray]:
        """Thread-safe copy of the latest thermal frame, or None if not ready."""
        with self.lock:
            return self.latest.copy() if self.latest is not None else None

    def get_last_mono(self) -> Optional[float]:
        """Monotonic timestamp of the last successful thermal read (skew / dedup)."""
        with self.lock:
            return self.last_mono

    def set_idle(self, idle: bool) -> None:
        """When True, the capture loop sleeps instead of reading the sensor."""
        if self._idle == idle:
            return
        self._idle = idle

    def stop(self):
        """Request thread exit; the run loop closes MI48 on the way out."""
        self.running = False


class NIRCapture(threading.Thread):
    """Picamera2 CSI capture (libcamera + ISP) at 640×480 RGB888.

    ``nir_fps`` maps to ``controls["FrameRate"]``. :meth:`set_idle` skips
    ``capture_array`` so thermal-only mode can pause NIR when configured.
    """

    def __init__(self, no_rgb2bgr: bool = True, nir_fps: float = 60.0):
        """``no_rgb2bgr`` keeps RGB order when True; False converts to BGR for OpenCV."""
        super().__init__(daemon=True)
        self.latest = None
        self.last_mono: Optional[float] = None
        self.lock = threading.Lock()
        self.running = True
        self._idle = False
        self.camera = None
        self.no_rgb2bgr = no_rgb2bgr
        self.nir_fps = float(np.clip(nir_fps, 5.0, 120.0))

    def run(self):
        """Configure Picamera2 preview stream and copy frames into ``latest``."""
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
                    print(f"NIR capture error: {e}")
        try:
            self.camera.stop()
        except Exception:
            pass

    def get_latest(self) -> Optional[np.ndarray]:
        """Thread-safe copy of the latest NIR BGR frame."""
        with self.lock:
            return self.latest.copy() if self.latest is not None else None

    def get_last_mono(self) -> Optional[float]:
        """Monotonic timestamp of the last NIR frame (stream skew vs thermal)."""
        with self.lock:
            return self.last_mono

    def set_idle(self, idle: bool) -> None:
        """When True, skip ``capture_array`` and sleep in the loop."""
        if self._idle == idle:
            return
        self._idle = idle

    def stop(self):
        """Stop the thread; attempts to stop Picamera2 cleanly."""
        self.running = False


def load_homography(path: str) -> tuple:
    """Load ``H``, thermal/NIR sizes, and optional FOV metadata from calibration JSON."""
    with open(path) as f:
        data = json.load(f)
    H = np.array(data["homography"], dtype=np.float32)
    meta = data.get("meta", {})
    thermal_size = tuple(meta.get("thermal_size", [320, 248]))
    nir_size = tuple(meta.get("nir_size", [640, 480]))
    fov = meta.get("fov_deg")
    if fov is not None and len(fov) >= 2:
        fov_h, fov_v = float(fov[0]), float(fov[1])
    else:
        fov_h, fov_v = None, None
    return H, thermal_size, nir_size, fov_h, fov_v


def ensure_fusion_capture_dirs() -> Tuple[str, str]:
    """Return ``(save_dir, metrics_dir)`` that are writable for PNG/JSON output.

    Tries ``./fusion_captures`` first, then ``~/fusion_captures`` if cwd is not writable.
    """
    metrics_name = "metrics"
    candidates: List[str] = [os.path.join(os.getcwd(), "fusion_captures")]
    home = os.path.expanduser("~")
    if home and os.path.abspath(home) != os.path.abspath(os.getcwd()):
        candidates.append(os.path.join(home, "fusion_captures"))

    last_err: Optional[OSError] = None
    for base in candidates:
        md = os.path.join(base, metrics_name)
        try:
            os.makedirs(md, exist_ok=True)
            probe = os.path.join(md, ".probe_write")
            with open(probe, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(probe)
            if base != candidates[0]:
                print(f"[fusion] Using fusion_captures under home (current directory not writable): {base}")
            return base, md
        except OSError as e:
            last_err = e
            continue

    raise PermissionError(
        "Cannot create or write fusion_captures (tried cwd and home). "
        f"Check directory permissions or run from a writable location. Last error: {last_err}"
    ) from last_err


def _capture_hardware_idle_flags(
    mode: str,
    mode_switch_pending: Optional[str],
    cfg: Dict[str, Any],
) -> Tuple[bool, bool]:
    """Return ``(thermal_idle, nir_idle)`` flags for power/CPU saving.

    - IMX mode: thermal may idle if ``pause_thermal_capture_when_imx_only``.
    - Thermal mode: NIR may idle if ``pause_nir_capture_when_thermal_only`` unless ENV auto_rule needs NIR.
    - Fusion: both streams stay active.
    - During pending mode switch: keep sensors needed for the target mode awake.

    When ``env_mode == auto_rule``, NIR never idles in thermal-only mode (features require NIR).
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
