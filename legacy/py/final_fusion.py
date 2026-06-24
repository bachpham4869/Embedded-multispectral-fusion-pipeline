#!/usr/bin/env python3
"""
LIVE Thermal-NIR Fusion v2 - MỘT FILE DUY NHẤT (3 MODE)
=========================================================
Chạy trên RPi4: thermal MI48 (SPI) + NIR IMX290 (Picamera2).

Phím:  1 = IMX (NIR)  2 = Thermal (3DNR)  3 = Fusion (NIR+Thermal heat)
       R = raw/processed  S = Save  A = Auto capture
       +/- = alpha  Q = Thoát

Pipeline:
  Thermal: 3DNR (EMA) → Background model (cold frame) → Continuous heat map
           (gradient nhiều mức, không chỉ 1 màu đỏ) → AGC → Edge enhance
  NIR:     Adaptive grayscale (SNR boost khi tối) → Dark/Bright Channel Prior
           → Fast Atmosphere Light → Weight map → Adaptive CLAHE + Detail
  Fusion:  Enhanced NIR + gradient thermal overlay (confidence-weighted)
"""

import sys
import os
import json
import time
import threading
import signal as sig_module
from collections import deque

import numpy as np
import cv2 as cv

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
    def __init__(self, warmup=40):
        self.temporal = ThermalTemporalFilter(alpha=0.65)
        self.bg = ThermalBackgroundModel(warmup_frames=warmup, adaptive_rate=0.005)

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
        agc = thermal_agc(denoised)
        enhanced = thermal_edge_enhance(agc, strength=0.2)
        heat_map = self.bg.get_heat_map(denoised, floor=3.0)
        fg_mask = self.bg.get_foreground_mask(denoised, threshold=18.0, min_area=12)
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
        self.is_night_mode = cur_bright < 0.45

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

    def stop(self):
        self.running = False


class NIRCapture(threading.Thread):
    def __init__(self, no_rgb2bgr=True):
        super().__init__(daemon=True)
        self.latest = None
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

    def stop(self):
        self.running = False


# ═══════════════════════════════════════════════════════════════════════════════
# HOMOGRAPHY
# ═══════════════════════════════════════════════════════════════════════════════

def load_homography(path):
    with open(path) as f:
        data = json.load(f)
    H = np.array(data["homography"], dtype=np.float32)
    meta = data.get("meta", {})
    thermal_size = tuple(meta.get("thermal_size", [320, 248]))
    nir_size = tuple(meta.get("nir_size", [640, 480]))
    return H, thermal_size, nir_size


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN (3 MODE: IMX, THERMAL, FUSION)
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    p = argparse.ArgumentParser(description="LIVE Thermal-NIR Fusion v2")
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

    H, thermal_size, nir_size = load_homography(args.homography)
    nir_w, nir_h = nir_size
    display_size = (args.width, args.height)
    alpha = args.alpha

    # Khởi tạo pipeline
    thermal_proc = ThermalProcessor(warmup=40)
    nir_enhancer = HybridNIREnhancer(proc_w=320, proc_h=240, patch_size=5, update_rate=8)
    fusion = GradientThermalFusion(base_alpha=0.6, colormap=THERMAL_COLORMAP)
    fps_counter = FPSCounter()

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

    save_dir = "fusion_captures"
    os.makedirs(save_dir, exist_ok=True)
    cv.namedWindow("SmartBinocular", cv.WINDOW_NORMAL)
    cv.resizeWindow("SmartBinocular", *display_size)

    NIR_RAW_THRESHOLD = 25   # Chỉ process khi thật sự tối; còn sáng/vừa thì hiển thị raw
    SWITCH_FRAMES = 28

    mode = "imx"
    show_raw = {"imx": False, "thermal": False}
    mode_switch_pending = None
    switch_frames_left = 0
    auto_start = None
    AUTO_DELAY = 5
    thermal_denoised = thermal_enhanced = heat_map = fg_mask = nir_enhanced = None

    print("=" * 60)
    print("  1 = IMX (NIR)   2 = Thermal (3DNR)   3 = Fusion (NIR+Thermal)")
    print("  NIR: raw khi sáng, grayscale+enhance khi tối")
    print("  R = raw/processed   S = Save   A = Auto   +/- = alpha   Q = Thoát")
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
            print(f"Alpha: {alpha:.2f}")
        elif key == ord("-"):
            alpha = max(alpha - 0.05, 0.05)
            print(f"Alpha: {alpha:.2f}")

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
                mode = mode_switch_pending
                mode_switch_pending = None
            fps_counter.tick()
            continue

        nir_raw = nir_cap.get_latest()
        thermal_raw = thermal_cap.get_latest()
        if mode == "imx" and nir_raw is None:
            time.sleep(0.02)
            continue
        if mode == "thermal" and thermal_raw is None:
            time.sleep(0.02)
            continue
        if mode == "fusion" and (nir_raw is None or thermal_raw is None):
            time.sleep(0.02)
            continue

        if mode == "imx":
            thermal_denoised = thermal_enhanced = heat_map = fg_mask = None
            nir_brightness = float(np.mean(cv.cvtColor(nir_raw, cv.COLOR_BGR2GRAY)))
            nir_enhanced = None if nir_brightness >= NIR_RAW_THRESHOLD else nir_enhancer.process(nir_raw)
        elif mode == "thermal":
            nir_enhanced = None
            thermal_denoised, thermal_enhanced, heat_map, fg_mask = thermal_proc.process(thermal_raw)
        else:  # fusion (NIR + thermal overlay)
            nir_brightness = float(np.mean(cv.cvtColor(nir_raw, cv.COLOR_BGR2GRAY)))
            nir_enhanced = None if nir_brightness >= NIR_RAW_THRESHOLD else nir_enhancer.process(nir_raw)
            thermal_denoised, thermal_enhanced, heat_map, fg_mask = thermal_proc.process(thermal_raw)

        fps = fps_counter.tick()

        if mode == "imx":
            nir_brightness = float(np.mean(cv.cvtColor(nir_raw, cv.COLOR_BGR2GRAY)))
            use_raw = show_raw.get("imx") or nir_brightness >= NIR_RAW_THRESHOLD
            if use_raw:
                src = nir_raw
            else:
                src = nir_enhanced if nir_enhanced is not None else nir_raw
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

        else:  # fusion (NIR bg + heat map overlay chỉ ở vùng có vật nóng)
            use_raw = show_raw.get("imx") or nir_brightness >= NIR_RAW_THRESHOLD
            nir_src = nir_raw if use_raw else (nir_enhanced if nir_enhanced is not None else nir_raw)
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
                print("Đã lưu:", path)
                auto_start = None

        if key == ord("s"):
            ts = time.strftime("%Y%m%d-%H%M%S")
            path = os.path.join(save_dir, f"{mode}_{ts}.png")
            cv.imwrite(path, out)
            print("Đã lưu:", path)

        # ── HUD (800x480 5-inch) ──
        labels = {"imx": "IMX (NIR)", "thermal": "Thermal (3DNR)", "fusion": "Fusion"}
        raw_tag = " [RAW]" if show_raw.get(mode) else ""
        cv.putText(out, f"{labels[mode]}{raw_tag}", (8, 22), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        cv.putText(out, f"FPS:{fps:.0f} a:{alpha:.2f}", (display_size[0] - 155, 22),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        if mode == "imx" and not show_raw.get("imx") and nir_enhancer.is_night_mode:
            cv.putText(out, "[NIGHT]", (8, 44), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

        cv.imshow("SmartBinocular", out)

    thermal_cap.stop()
    nir_cap.stop()
    cv.destroyAllWindows()
    print("Thoát.")


if __name__ == "__main__":
    main()
