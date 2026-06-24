#!/usr/bin/env python3
import argparse
import csv
import logging
import os
import threading
import time
from pathlib import Path

import cv2 as cv
import numpy as np

_HARDWARE_IMPORT_ERRORS = {}

try:
    from smbus import SMBus
except ImportError as e:
    SMBus = None
    _HARDWARE_IMPORT_ERRORS["smbus"] = e

try:
    from spidev import SpiDev
except ImportError as e:
    SpiDev = None
    _HARDWARE_IMPORT_ERRORS["spidev"] = e

try:
    from gpiozero import DigitalInputDevice, DigitalOutputDevice
except ImportError as e:
    DigitalInputDevice = None
    DigitalOutputDevice = None
    _HARDWARE_IMPORT_ERRORS["gpiozero"] = e

try:
    from picamera2 import Picamera2
except ImportError as e:
    Picamera2 = None
    _HARDWARE_IMPORT_ERRORS["picamera2"] = e

try:
    from senxor.mi48 import MI48
    from senxor.interfaces import SPI_Interface, I2C_Interface
    from senxor.utils import data_to_frame
except ImportError as e:
    MI48 = None
    SPI_Interface = None
    I2C_Interface = None
    data_to_frame = None
    _SENXOR_IMPORT_ERROR = e
else:
    _SENXOR_IMPORT_ERROR = None


# ============================================================
# Capture constants
# ============================================================

RPI_GPIO_I2C_CHANNEL = 1
RPI_GPIO_SPI_BUS = 0
RPI_GPIO_SPI_CE_MI48 = 0

MI48_I2C_ADDRESS = 0x40
MI48_SPI_MAX_SPEED_HZ_DEFAULT = 7_800_000
MI48_SPI_CS_DELAY = 0.0001
SPI_XFER_SIZE_BYTES = 160
MI48_FPA_ROWS = 62
MI48_FPA_COLS = 80
MI48_FRAME_BYTES = MI48_FPA_ROWS * MI48_FPA_COLS * 2
MI48_HEADER_BYTES = 8

PIN_MI48_CS_N = "BCM7"
PIN_MI48_DATA_READY = "BCM24"
PIN_MI48_RESET_N = "BCM23"


def _require_runtime_dependencies(names):
    missing = []
    for name in names:
        if globals().get(name) is None:
            missing.append(name)

    if not missing:
        return

    details = ", ".join(missing)
    raise RuntimeError(
        "Missing Raspberry Pi capture dependencies: "
        f"{details}. This file can be imported on a dev machine, but actual "
        "capture must run on the Pi with the camera/GPIO packages installed."
    )


if MI48 is None:
    class SPI_Interface:
        def __init__(self, device, xfer_size=160):
            self.device = device
            self.xfer_size = int(xfer_size)

        def read(self, nbytes):
            result = []
            remaining = int(nbytes)
            while remaining > 0:
                chunk = min(remaining, self.xfer_size)
                result.extend(self.device.xfer2([0x00] * chunk))
                remaining -= chunk
            return result[:nbytes]

    class I2C_Interface:
        def __init__(self, device, address):
            self.device = device
            self.address = int(address)

        def read_reg(self, reg, nbytes=1):
            return list(self.device.read_i2c_block_data(self.address, int(reg), int(nbytes)))

        def write_reg(self, reg, value):
            self.device.write_byte_data(self.address, int(reg), int(value) & 0xFF)

    class MI48:
        def __init__(self, interfaces, data_ready=None, reset_handler=None):
            self.i2c = interfaces[0]
            self.spi = interfaces[1]
            self.data_ready = data_ready
            self.fpa_shape = (MI48_FPA_ROWS, MI48_FPA_COLS)
            self._with_header = False
            if reset_handler is not None:
                reset_handler()

        @property
        def fw_version(self):
            try:
                return list(self.i2c.read_reg(0x00, 4))
            except Exception:
                return [2, 0, 0, 0]

        def enable_filter(self, f1=True, f2=True, f3=False):
            self.i2c.write_reg(0x10, 0x01 if f1 else 0x00)
            self.i2c.write_reg(0x11, 0x01 if f2 else 0x00)
            self.i2c.write_reg(0x12, 0x01 if f3 else 0x00)

        def set_fps(self, fps):
            self.i2c.write_reg(0x05, max(1, min(9, int(fps))))

        def start(self, stream=True, with_header=False):
            self._with_header = bool(with_header)
            self.i2c.write_reg(0x01, 0x03 if stream else 0x00)

        def read(self):
            header_len = MI48_HEADER_BYTES if self._with_header else 0
            total = MI48_FRAME_BYTES + header_len
            raw = self.spi.read(total)
            if len(raw) < total:
                return None, {}
            header = {}
            if self._with_header:
                header = {"raw_header": bytes(raw[:header_len])}
                raw = raw[header_len:]
            return raw, header

        def stop(self, stop_timeout=0.5):
            try:
                self.i2c.write_reg(0x01, 0x00)
            finally:
                time.sleep(min(float(stop_timeout), 0.5))

    def data_to_frame(data, shape):
        rows, cols = int(shape[0]), int(shape[1])
        n_pixels = rows * cols
        arr = np.frombuffer(bytes(data[: n_pixels * 2]), dtype=np.dtype("<u2"))
        if arr.size < n_pixels:
            raise ValueError(f"short MI48 frame: got {arr.size} pixels, expected {n_pixels}")
        return arr[:n_pixels].reshape(rows, cols).astype(np.float32) * 0.01


class Reset:
    def __init__(self, pin):
        self.pin = pin

    def __call__(self):
        self.pin.on()
        time.sleep(0.000035)
        self.pin.off()
        time.sleep(0.05)


# ============================================================
# Thermal capture thread
# ============================================================

class ThermalFrameCapture(threading.Thread):
    """
    MI48 capture thread.

    MI48 capture logic:
    - I2C/SPI/GPIO constants for the Raspberry Pi wiring
    - MI48([i2c, spi], data_ready, reset_handler)
    - enable_filter nếu fw >= 2
    - set_fps()
    - start(stream=True, with_header=True)
    - loop: wait_for_active -> CS on -> mi48.read() -> CS off
    - dùng data_to_frame(data, fpa_shape), không tự parse byte.
    """

    def __init__(
        self,
        fps=9,
        spi_hz=MI48_SPI_MAX_SPEED_HZ_DEFAULT,
        no_filter=False,
        edge_ready=False,
    ):
        super().__init__(daemon=True)

        self.fps = int(fps)
        self.spi_hz = int(spi_hz)
        self.no_filter = bool(no_filter)
        self.edge_ready = bool(edge_ready)

        self.running = False
        self.lock = threading.Lock()

        self.mi48 = None
        self.spi_cs_n = None
        self.data_ready = None
        self.reset_n = None

        self.fpa_shape = None

        # latest thermal frame from data_to_frame()
        self.latest_frame = None
        self.latest_ts_ns = None
        self.latest_id = 0

        self.frames = 0
        self.none_reads = 0
        self.bad_frames = 0
        self.errors = 0
        self.last_error = ""

    def _open(self):
        _require_runtime_dependencies([
            "SMBus",
            "SpiDev",
            "DigitalInputDevice",
            "DigitalOutputDevice",
            "MI48",
            "SPI_Interface",
            "I2C_Interface",
        ])

        i2c = I2C_Interface(
            SMBus(RPI_GPIO_I2C_CHANNEL),
            MI48_I2C_ADDRESS,
        )

        spi = SPI_Interface(
            SpiDev(RPI_GPIO_SPI_BUS, RPI_GPIO_SPI_CE_MI48),
            xfer_size=SPI_XFER_SIZE_BYTES,
        )

        spi.device.mode = 0
        spi.device.max_speed_hz = self.spi_hz
        spi.device.bits_per_word = 8
        spi.device.lsbfirst = False
        spi.device.cshigh = True
        spi.device.no_cs = True

        self.spi_cs_n = DigitalOutputDevice(
            PIN_MI48_CS_N,
            active_high=False,
            initial_value=False,
        )

        self.data_ready = DigitalInputDevice(
            PIN_MI48_DATA_READY,
            pull_up=False,
        )

        self.reset_n = DigitalOutputDevice(
            PIN_MI48_RESET_N,
            active_high=False,
            initial_value=True,
        )

        self.mi48 = MI48(
            [i2c, spi],
            data_ready=self.data_ready,
            reset_handler=Reset(self.reset_n),
        )

        fw = self.mi48.fw_version
        print(f"[INFO] MI48 fw_version={fw}")

        if not self.no_filter:
            try:
                if int(fw[0]) >= 2:
                    self.mi48.enable_filter(f1=True, f2=True, f3=False)
            except Exception as e:
                print(f"[WARN] MI48 enable_filter failed: {e}")

        self.mi48.set_fps(self.fps)
        self.mi48.start(stream=True, with_header=True)

        self.fpa_shape = tuple(self.mi48.fpa_shape)
        print(
            f"[INFO] MI48 stream started: "
            f"fpa_shape={self.fpa_shape}, fps={self.fps}, spi_hz={self.spi_hz}"
        )

    def _read_one(self):
        """
        Read one MI48 frame using the standard DATA_READY -> SPI read sequence.

        edge_ready=False uses level-triggered DATA_READY.
        edge_ready=True can help if DATA_READY remains stale-high.
        """
        if self.edge_ready:
            try:
                if self.data_ready.value:
                    self.data_ready.wait_for_inactive(timeout=0.2)
            except Exception:
                pass

        self.data_ready.wait_for_active()

        self.spi_cs_n.on()
        time.sleep(MI48_SPI_CS_DELAY)

        data, _ = self.mi48.read()
        ts_ns = time.monotonic_ns()

        time.sleep(MI48_SPI_CS_DELAY)
        self.spi_cs_n.off()

        return data, ts_ns

    def run(self):
        self.running = True

        try:
            self._open()

            while self.running:
                try:
                    data, ts_ns = self._read_one()

                    if data is None:
                        self.none_reads += 1
                        time.sleep(0.002)
                        continue

                    frame = data_to_frame(data, self.mi48.fpa_shape)

                    if frame is None:
                        self.bad_frames += 1
                        continue

                    frame = np.asarray(frame)

                    if frame.size == 0:
                        self.bad_frames += 1
                        continue

                    frame = frame.astype(np.float32, copy=False)

                    if not np.isfinite(frame).any():
                        self.bad_frames += 1
                        continue

                    with self.lock:
                        self.latest_frame = frame.copy()
                        self.latest_ts_ns = ts_ns
                        self.latest_id += 1

                    self.frames += 1

                except Exception as e:
                    self.errors += 1
                    self.last_error = str(e)
                    if self.running:
                        print(f"\n[WARN] Thermal capture error: {e}")
                    time.sleep(0.01)

        finally:
            self._close()

    def get_latest(self):
        with self.lock:
            if self.latest_frame is None:
                return None, None, None
            return self.latest_frame.copy(), self.latest_ts_ns, self.latest_id

    def _close(self):
        try:
            if self.mi48 is not None:
                self.mi48.stop(stop_timeout=0.5)
        except Exception:
            pass

        for dev in (self.spi_cs_n, self.data_ready, self.reset_n):
            try:
                if dev is not None:
                    dev.close()
            except Exception:
                pass

    def stop(self):
        self.running = False


# ============================================================
# IMX/NIR capture thread
# ============================================================

class IMXCapture(threading.Thread):
    """
    IMX/NIR capture thread.

    Bám NIRCapture:
    - Picamera2
    - main 640x480 RGB888
    - controls giống repo
    - capture_array()
    - cv.flip(frame, 1)
    """

    def __init__(
        self,
        width=640,
        height=480,
        fps=30,
        flip_h=True,
    ):
        super().__init__(daemon=True)

        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.flip_h = bool(flip_h)

        self.running = False
        self.lock = threading.Lock()

        self.camera = None

        self.latest_bgr = None
        self.latest_ts_ns = None
        self.latest_id = 0

        self.frames = 0
        self.errors = 0
        self.last_error = ""

    def _open(self):
        _require_runtime_dependencies(["Picamera2"])

        self.camera = Picamera2()

        config = self.camera.create_preview_configuration(
            main={
                "size": (self.width, self.height),
                "format": "RGB888",
            },
            controls={
                "FrameRate": self.fps,
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

        print(f"[INFO] IMX camera started: {self.width}x{self.height}@{self.fps}")

    def run(self):
        self.running = True

        try:
            self._open()

            while self.running:
                try:
                    frame_rgb = self.camera.capture_array()
                    ts_ns = time.monotonic_ns()

                    if self.flip_h:
                        frame_rgb = cv.flip(frame_rgb, 1)

                    frame_bgr = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)

                    with self.lock:
                        self.latest_bgr = frame_bgr.copy()
                        self.latest_ts_ns = ts_ns
                        self.latest_id += 1

                    self.frames += 1

                except Exception as e:
                    self.errors += 1
                    self.last_error = str(e)
                    if self.running:
                        print(f"\n[WARN] IMX capture error: {e}")
                    time.sleep(0.01)

        finally:
            self._close()

    def get_latest(self):
        with self.lock:
            if self.latest_bgr is None:
                return None, None, None
            return self.latest_bgr.copy(), self.latest_ts_ns, self.latest_id

    def _close(self):
        try:
            if self.camera is not None:
                self.camera.stop()
        except Exception:
            pass

    def stop(self):
        self.running = False


# ============================================================
# Thermal visualization - không dùng min/max toàn frame
# ============================================================

class RobustThermalNormalizer:
    """
    Convert thermal frame -> uint8 for MP4.

    Không dùng min/max toàn frame.
    Dùng percentile trên valid pixels để tránh bad pixels phá scale.

    Hỗ trợ auto 3 kiểu:
    - frame từ data_to_frame dạng Kelvin, khoảng 200..450
    - frame từ vendor driver dạng Celsius, khoảng -80..180
    - frame dạng raw count, khoảng 20000..45000
    """

    def __init__(
        self,
        p_low=2.0,
        p_high=98.0,
        valid_k_low=200.0,
        valid_k_high=450.0,
        valid_c_low=-80.0,
        valid_c_high=180.0,
        valid_raw_low=20000.0,
        valid_raw_high=45000.0,
        min_valid_pixels=200,
        ema_alpha=0.25,
    ):
        self.p_low = float(p_low)
        self.p_high = float(p_high)

        self.valid_k_low = float(valid_k_low)
        self.valid_k_high = float(valid_k_high)

        self.valid_c_low = float(valid_c_low)
        self.valid_c_high = float(valid_c_high)

        self.valid_raw_low = float(valid_raw_low)
        self.valid_raw_high = float(valid_raw_high)

        self.min_valid_pixels = int(min_valid_pixels)
        self.ema_alpha = float(ema_alpha)

        self.prev_lo = None
        self.prev_hi = None
        self.last_unit = "unknown"

    def _valid_values(self, frame):
        x = frame.astype(np.float32, copy=False)
        finite = np.isfinite(x)

        if not finite.any():
            return None

        finite_values = x[finite]
        med = float(np.median(finite_values))

        # Auto detect unit.
        if med > 1000.0:
            self.last_unit = "raw_count_or_0.01K"
            lo = self.valid_raw_low
            hi = self.valid_raw_high
        elif med >= self.valid_k_low:
            self.last_unit = "kelvin_or_temperature"
            lo = self.valid_k_low
            hi = self.valid_k_high
        else:
            self.last_unit = "celsius_temperature"
            lo = self.valid_c_low
            hi = self.valid_c_high

        mask = (
            finite
            & (x != 0)
            & (x != 65535)
            & (x >= lo)
            & (x <= hi)
        )

        valid = x[mask]

        if valid.size < self.min_valid_pixels:
            return None

        return valid

    def normalize(self, frame):
        valid = self._valid_values(frame)

        if valid is not None:
            lo, hi = np.percentile(valid, [self.p_low, self.p_high])

            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                if self.prev_lo is None or self.prev_hi is None:
                    self.prev_lo = float(lo)
                    self.prev_hi = float(hi)
                else:
                    a = self.ema_alpha
                    self.prev_lo = (1.0 - a) * self.prev_lo + a * float(lo)
                    self.prev_hi = (1.0 - a) * self.prev_hi + a * float(hi)

        if self.prev_lo is None or self.prev_hi is None or self.prev_hi <= self.prev_lo:
            return np.zeros_like(frame, dtype=np.uint8)

        x = frame.astype(np.float32, copy=False)
        img = (x - self.prev_lo) * 255.0 / (self.prev_hi - self.prev_lo)
        return np.clip(img, 0, 255).astype(np.uint8)

    def scale_text(self):
        if self.prev_lo is None or self.prev_hi is None:
            return "scale=unknown"

        if self.last_unit == "raw_count_or_0.01K":
            lo_c = self.prev_lo * 0.01 - 273.15
            hi_c = self.prev_hi * 0.01 - 273.15
            return f"scale={lo_c:.1f}..{hi_c:.1f}C(raw)"

        if self.last_unit == "kelvin_or_temperature":
            lo_c = self.prev_lo - 273.15
            hi_c = self.prev_hi - 273.15
            return f"scale={lo_c:.1f}..{hi_c:.1f}C(K)"

        if self.last_unit == "celsius_temperature":
            return f"scale={self.prev_lo:.1f}..{self.prev_hi:.1f}C(C)"

        return f"scale={self.prev_lo:.1f}..{self.prev_hi:.1f}"


def thermal_to_video_frame(frame, normalizer, scale=8, gray=False):
    img8 = normalizer.normalize(frame)

    if gray:
        out = cv.cvtColor(img8, cv.COLOR_GRAY2BGR)
    else:
        out = cv.applyColorMap(img8, cv.COLORMAP_TURBO)

    h, w = img8.shape[:2]
    out_size = (w * int(scale), h * int(scale))

    return cv.resize(
        out,
        out_size,
        interpolation=cv.INTER_NEAREST,
    )


def thermal_frame_to_u16_for_npz(frame):
    """
    Save-friendly approximate uint16 thermal.

    Nếu frame là Kelvin từ data_to_frame: raw_u16 ~= Kelvin * 100.
    Nếu frame đã là raw count: cast trực tiếp.
    """
    x = frame.astype(np.float32, copy=False)
    finite = np.isfinite(x)

    if not finite.any():
        return np.zeros_like(frame, dtype=np.uint16), "invalid"

    med = float(np.median(x[finite]))

    if med > 1000.0:
        raw = np.clip(x, 0, 65535).astype(np.uint16)
        return raw, "direct_raw_like"

    raw = np.clip(np.rint(x * 100.0), 0, 65535).astype(np.uint16)
    return raw, "derived_from_kelvin_x100"


# ============================================================
# Wait helpers
# ============================================================

def wait_until_ready(cap, name, timeout_sec=10.0):
    t0 = time.monotonic()

    while time.monotonic() - t0 < timeout_sec:
        frame, ts, fid = cap.get_latest()
        if frame is not None:
            print(f"[INFO] {name} ready: first_frame_id={fid}")
            return True
        time.sleep(0.01)

    return False


def wait_more_frames(cap, start_id, count=10, timeout_sec=5.0):
    target = int(start_id) + int(count)
    t0 = time.monotonic()

    while time.monotonic() - t0 < timeout_sec:
        _, _, fid = cap.get_latest()
        if fid is not None and fid >= target:
            return True
        time.sleep(0.01)

    return False


def safe_stop_join(cap, name):
    try:
        cap.stop()
    except Exception:
        pass

    try:
        if cap.is_alive():
            cap.join(timeout=2.0)
    except RuntimeError:
        # Thread was never started.
        pass
    except Exception as e:
        print(f"[WARN] Failed to join {name}: {e}")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--seconds", type=float, default=30.0, help="0 = record until Ctrl+C")
    ap.add_argument("--outdir", default="paired_capture")

    ap.add_argument("--thermal-fps", type=int, default=9)
    ap.add_argument("--imx-fps", type=int, default=30)
    ap.add_argument("--imx-width", type=int, default=640)
    ap.add_argument("--imx-height", type=int, default=480)

    ap.add_argument("--spi-hz", type=int, default=MI48_SPI_MAX_SPEED_HZ_DEFAULT)
    ap.add_argument("--edge-ready", action="store_true")

    ap.add_argument("--thermal-scale", type=int, default=8)
    ap.add_argument("--thermal-gray", action="store_true")

    ap.add_argument("--p-low", type=float, default=2.0)
    ap.add_argument("--p-high", type=float, default=98.0)

    ap.add_argument("--valid-k-low", type=float, default=200.0)
    ap.add_argument("--valid-k-high", type=float, default=450.0)
    ap.add_argument("--valid-raw-low", type=float, default=20000.0)
    ap.add_argument("--valid-raw-high", type=float, default=45000.0)

    ap.add_argument("--min-valid-pixels", type=int, default=200)
    ap.add_argument("--ema-alpha", type=float, default=0.25)

    ap.add_argument("--thermal-stable-frames", type=int, default=10)
    ap.add_argument("--post-imx-check-sec", type=float, default=8.0)
    ap.add_argument("--stall-timeout-sec", type=float, default=5.0)

    ap.add_argument("--save-raw-npz", action="store_true")
    ap.add_argument("--no-filter", action="store_true")
    ap.add_argument("--quiet-libcamera", action="store_true")

    args = ap.parse_args()

    if args.quiet_libcamera:
        os.environ["LIBCAMERA_LOG_LEVELS"] = "*:ERROR"

    logging.basicConfig(level=logging.WARNING)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    imx_mp4_path = outdir / "imx_paired.mp4"
    thermal_mp4_path = outdir / "thermal_paired.mp4"
    csv_path = outdir / "timestamps.csv"

    thermal_cap = ThermalFrameCapture(
        fps=args.thermal_fps,
        spi_hz=args.spi_hz,
        no_filter=args.no_filter,
        edge_ready=args.edge_ready,
    )

    imx_cap = IMXCapture(
        width=args.imx_width,
        height=args.imx_height,
        fps=args.imx_fps,
        flip_h=True,
    )

    normalizer = RobustThermalNormalizer(
        p_low=args.p_low,
        p_high=args.p_high,
        valid_k_low=args.valid_k_low,
        valid_k_high=args.valid_k_high,
        valid_raw_low=args.valid_raw_low,
        valid_raw_high=args.valid_raw_high,
        min_valid_pixels=args.min_valid_pixels,
        ema_alpha=args.ema_alpha,
    )

    imx_writer = None
    thermal_writer = None

    thermal_frame_list = []
    thermal_u16_list = []
    imx_bgr_list = []
    pair_ts_list = []
    thermal_ts_list = []
    imx_ts_list = []

    pair_idx = 0
    last_thermal_id = -1
    last_new_thermal_time = time.monotonic()
    t0_ns = time.monotonic_ns()

    try:
        # 1) Start thermal first
        print("[INFO] Starting thermal thread first...")
        thermal_cap.start()

        if not wait_until_ready(thermal_cap, "Thermal", timeout_sec=20.0):
            raise RuntimeError(
                "Thermal did not produce frames before IMX starts. "
                f"stats: frames={thermal_cap.frames}, "
                f"none_reads={thermal_cap.none_reads}, "
                f"bad_frames={thermal_cap.bad_frames}, "
                f"errors={thermal_cap.errors}, "
                f"last_error={thermal_cap.last_error}. "
                "Try thermal-only, --spi-hz 4000000, or --edge-ready."
            )

        _, _, th_id0 = thermal_cap.get_latest()

        if not wait_more_frames(
            thermal_cap,
            start_id=th_id0,
            count=args.thermal_stable_frames,
            timeout_sec=10.0,
        ):
            raise RuntimeError(
                "Thermal did not stay stable before IMX starts. "
                f"stats: frames={thermal_cap.frames}, "
                f"none_reads={thermal_cap.none_reads}, "
                f"bad_frames={thermal_cap.bad_frames}, "
                f"errors={thermal_cap.errors}, "
                f"last_error={thermal_cap.last_error}. "
                "Try --spi-hz 4000000 or --no-filter."
            )

        _, _, thermal_id_before_imx = thermal_cap.get_latest()
        none_before_imx = thermal_cap.none_reads
        bad_before_imx = thermal_cap.bad_frames

        print(
            f"[INFO] Thermal stable before IMX: "
            f"frames={thermal_cap.frames}, "
            f"latest_id={thermal_id_before_imx}, "
            f"none_reads={thermal_cap.none_reads}, "
            f"bad_frames={thermal_cap.bad_frames}, "
            f"errors={thermal_cap.errors}"
        )

        # 2) Start IMX after thermal is stable
        print("[INFO] Starting IMX thread...")
        imx_cap.start()

        if not wait_until_ready(imx_cap, "IMX", timeout_sec=10.0):
            raise RuntimeError("IMX camera did not produce frames.")

        # 3) Check thermal still alive after IMX active
        print("[INFO] Checking thermal after IMX starts...")

        t_check0 = time.monotonic()
        thermal_alive_after_imx = False

        while time.monotonic() - t_check0 < args.post_imx_check_sec:
            _, _, th_id_now = thermal_cap.get_latest()
            if th_id_now is not None and th_id_now > thermal_id_before_imx:
                thermal_alive_after_imx = True
                print(
                    f"[INFO] Thermal alive after IMX: "
                    f"old_id={thermal_id_before_imx}, "
                    f"new_id={th_id_now}, "
                    f"none_delta={thermal_cap.none_reads - none_before_imx}, "
                    f"bad_delta={thermal_cap.bad_frames - bad_before_imx}"
                )
                break
            time.sleep(0.01)

        if not thermal_alive_after_imx:
            raise RuntimeError(
                "Thermal stopped producing valid frames after IMX started. "
                f"stats: frames={thermal_cap.frames}, "
                f"none_reads={thermal_cap.none_reads}, "
                f"bad_frames={thermal_cap.bad_frames}, "
                f"errors={thermal_cap.errors}, "
                f"last_error={thermal_cap.last_error}. "
                "Likely power/SPI noise/resource conflict. "
                "Try --spi-hz 4000000, then 2000000, lower --imx-fps 15, "
                "or check MI48 power/GND/SPI wiring."
            )

        # 4) Open writers after both are confirmed OK
        thermal0, _, _ = thermal_cap.get_latest()
        imx0, _, _ = imx_cap.get_latest()

        if thermal0 is None or imx0 is None:
            raise RuntimeError("Cannot get initial paired frames.")

        th_rows, th_cols = thermal0.shape[:2]
        thermal_size = (
            th_cols * int(args.thermal_scale),
            th_rows * int(args.thermal_scale),
        )
        imx_size = (int(args.imx_width), int(args.imx_height))

        fourcc = cv.VideoWriter_fourcc(*"mp4v")

        imx_writer = cv.VideoWriter(
            str(imx_mp4_path),
            fourcc,
            float(args.thermal_fps),
            imx_size,
            True,
        )

        thermal_writer = cv.VideoWriter(
            str(thermal_mp4_path),
            fourcc,
            float(args.thermal_fps),
            thermal_size,
            True,
        )

        if not imx_writer.isOpened():
            raise RuntimeError(f"Cannot open VideoWriter: {imx_mp4_path}")

        if not thermal_writer.isOpened():
            raise RuntimeError(f"Cannot open VideoWriter: {thermal_mp4_path}")

        print("[INFO] Recording...")
        print("[INFO] Pair rule: new thermal frame k <-> latest IMX frame at that time.")

        with open(csv_path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow([
                "pair_idx",
                "pair_t_ns",
                "pair_t_sec",
                "thermal_ts_ns",
                "thermal_frame_id",
                "imx_ts_ns",
                "imx_frame_id",
                "skew_ms_imx_minus_thermal",
                "thermal_unit_for_display",
                "thermal_scale",
                "thermal_none_reads",
                "thermal_bad_frames",
                "thermal_errors",
                "imx_errors",
                "imx_video",
                "thermal_video",
            ])

            while True:
                now = time.monotonic()
                now_ns = time.monotonic_ns()

                if args.seconds > 0 and (now_ns - t0_ns) / 1e9 >= args.seconds:
                    break

                thermal_frame, thermal_ts_ns, thermal_id = thermal_cap.get_latest()

                if thermal_frame is None:
                    time.sleep(0.001)
                    continue

                if thermal_id == last_thermal_id:
                    if now - last_new_thermal_time > args.stall_timeout_sec:
                        raise RuntimeError(
                            f"Thermal stalled for > {args.stall_timeout_sec}s. "
                            f"frames={thermal_cap.frames}, "
                            f"none_reads={thermal_cap.none_reads}, "
                            f"bad_frames={thermal_cap.bad_frames}, "
                            f"errors={thermal_cap.errors}, "
                            f"last_error={thermal_cap.last_error}"
                        )
                    time.sleep(0.001)
                    continue

                last_thermal_id = thermal_id
                last_new_thermal_time = now

                imx_frame, imx_ts_ns, imx_id = imx_cap.get_latest()
                if imx_frame is None:
                    continue

                if imx_frame.shape[1] != args.imx_width or imx_frame.shape[0] != args.imx_height:
                    imx_frame = cv.resize(
                        imx_frame,
                        imx_size,
                        interpolation=cv.INTER_AREA,
                    )

                thermal_video_frame = thermal_to_video_frame(
                    thermal_frame,
                    normalizer=normalizer,
                    scale=args.thermal_scale,
                    gray=args.thermal_gray,
                )

                imx_writer.write(imx_frame)
                thermal_writer.write(thermal_video_frame)

                pair_t_ns = thermal_ts_ns - t0_ns
                pair_t_sec = pair_t_ns / 1e9
                skew_ms = (imx_ts_ns - thermal_ts_ns) / 1e6

                scale_text = normalizer.scale_text()

                wr.writerow([
                    pair_idx,
                    pair_t_ns,
                    f"{pair_t_sec:.9f}",
                    thermal_ts_ns,
                    thermal_id,
                    imx_ts_ns,
                    imx_id,
                    f"{skew_ms:.3f}",
                    normalizer.last_unit,
                    scale_text,
                    thermal_cap.none_reads,
                    thermal_cap.bad_frames,
                    thermal_cap.errors,
                    imx_cap.errors,
                    imx_mp4_path.name,
                    thermal_mp4_path.name,
                ])

                if args.save_raw_npz:
                    thermal_u16, _ = thermal_frame_to_u16_for_npz(thermal_frame)
                    thermal_frame_list.append(thermal_frame.copy())
                    thermal_u16_list.append(thermal_u16.copy())
                    imx_bgr_list.append(imx_frame.copy())
                    pair_ts_list.append(pair_t_ns)
                    thermal_ts_list.append(thermal_ts_ns)
                    imx_ts_list.append(imx_ts_ns)

                pair_idx += 1

                print(
                    f"\rpaired={pair_idx} "
                    f"| th_id={thermal_id} "
                    f"| imx_id={imx_id} "
                    f"| skew={skew_ms:+.2f}ms "
                    f"| th_none={thermal_cap.none_reads} "
                    f"| th_bad={thermal_cap.bad_frames} "
                    f"| th_err={thermal_cap.errors} "
                    f"| imx_err={imx_cap.errors} "
                    f"| {scale_text}",
                    end="",
                    flush=True,
                )

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        print("\n[INFO] Stopping...")

        if imx_writer is not None:
            imx_writer.release()

        if thermal_writer is not None:
            thermal_writer.release()

        safe_stop_join(thermal_cap, "thermal")
        safe_stop_join(imx_cap, "imx")

    if args.save_raw_npz and pair_idx > 0:
        npz_path = outdir / "paired_raw_frames.npz"

        np.savez_compressed(
            npz_path,
            thermal_frame=np.stack(thermal_frame_list).astype(np.float32),
            thermal_u16=np.stack(thermal_u16_list).astype(np.uint16),
            imx_bgr=np.stack(imx_bgr_list).astype(np.uint8),
            pair_t_ns=np.asarray(pair_ts_list, dtype=np.int64),
            thermal_ts_ns=np.asarray(thermal_ts_list, dtype=np.int64),
            imx_ts_ns=np.asarray(imx_ts_list, dtype=np.int64),
            thermal_frame_note="Output of data_to_frame(data, mi48.fpa_shape); usually Kelvin for fallback driver.",
            thermal_u16_note="If thermal_frame is Kelvin, thermal_u16 ~= Kelvin*100; if raw-like, direct cast.",
            imx_format="BGR888_from_Picamera2_RGB888",
        )

        print(f"[OK] saved raw npz       : {npz_path}")

    print(f"[OK] saved IMX video     : {imx_mp4_path}")
    print(f"[OK] saved thermal video : {thermal_mp4_path}")
    print(f"[OK] saved timestamps    : {csv_path}")
    print(f"[OK] paired frames       : {pair_idx}")
    print(f"[OK] thermal frames      : {thermal_cap.frames}")
    print(f"[OK] thermal none reads  : {thermal_cap.none_reads}")
    print(f"[OK] thermal bad frames  : {thermal_cap.bad_frames}")
    print(f"[OK] thermal errors      : {thermal_cap.errors}")
    print(f"[OK] imx frames          : {imx_cap.frames}")
    print(f"[OK] imx errors          : {imx_cap.errors}")


if __name__ == "__main__":
    main()
