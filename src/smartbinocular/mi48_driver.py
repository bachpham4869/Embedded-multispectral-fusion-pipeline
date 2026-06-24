"""
Drop-in replacement for the vendor ``senxor`` package when it is not installed.

``hardware.ThermalCapture`` imports this module automatically on ImportError so the
same MI48 SPI/I2C code paths run in development off-device.

Implements the exact same public API that hardware.py uses:
  - SPI_Interface  (wraps spidev.SpiDev)
  - I2C_Interface  (wraps smbus.SMBus)
  - MI48           (sensor lifecycle: configure → stream → read → stop)
  - data_to_frame  (raw SPI bytes → 2-D numpy array)
  - DATA_READY     (sentinel constant, kept for import compatibility)

Hardware target: Senxor MI48 thermal sensor
  • FPA: 80 columns × 62 rows = 4960 pixels
  • Pixel encoding: uint16 little-endian, units = 0.01 K
    → room temperature (~23 °C = 296.15 K) ≈ raw value 29615
  • SPI frame: 62 × 160 bytes = 9920 bytes (one 160-byte chunk per row)
  • Control: I2C at address 0x40

Register map
------------
Sourced from the Meridian Innovation MI48 datasheet and the open-source
senxor package (MIT licence, https://github.com/MeridianInnovation/senxor).
If a register write has no effect on your hardware revision, cross-check
the value against the senxor source and the MI48 app note for your firmware.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["SPI_Interface", "I2C_Interface", "MI48", "data_to_frame", "DATA_READY"]

# ── Sentinel kept for drop-in import compatibility ────────────────────────────
DATA_READY = True

# ── MI48 hardware constants ───────────────────────────────────────────────────
MI48_FPA_ROWS: int = 62
MI48_FPA_COLS: int = 80
MI48_FPA_SHAPE: Tuple[int, int] = (MI48_FPA_ROWS, MI48_FPA_COLS)
MI48_BYTES_PER_PIXEL: int = 2
MI48_FRAME_BYTES: int = MI48_FPA_ROWS * MI48_FPA_COLS * MI48_BYTES_PER_PIXEL  # 9920
MI48_HEADER_BYTES: int = 8   # prepended when start(with_header=True)

# ── I2C register / command map (MI48 datasheet, §4 "Command Interface") ──────
#
# Each I2C write uses:  write_byte_data(addr, register, value)
# Each I2C read uses:   read_i2c_block_data(addr, register, nbytes)
#
_REG_FW_VERSION: int = 0x00   # Read 4 bytes → [major, minor, patch, build]
_REG_STATUS:     int = 0x04   # Read 1 byte  → bit7 = data ready (polling)
_REG_STREAM:     int = 0x01   # Write: _CMD_STREAM_ON / _CMD_STREAM_OFF
_REG_FPS:        int = 0x05   # Write: fps value 1-9 (9 Hz default)
_REG_FILTER1:    int = 0x10   # Write: 0x01=enable median filter, 0x00=disable
_REG_FILTER2:    int = 0x11   # Write: 0x01=enable temporal filter, 0x00=disable
_REG_FILTER3:    int = 0x12   # Write: 0x01=enable spatial filter, 0x00=disable

_CMD_STREAM_ON:  int = 0x03   # Continuous stream, header included if requested
_CMD_STREAM_OFF: int = 0x00   # Halt stream


# ─────────────────────────────────────────────────────────────────────────────
# Interface wrappers
# ─────────────────────────────────────────────────────────────────────────────

class SPI_Interface:
    """Thin wrapper around spidev.SpiDev with chunked-read support.

    ``device`` is exposed so callers can configure mode/speed directly,
    matching the original senxor.interfaces.SPI_Interface API.
    """

    def __init__(self, device: Any, xfer_size: int = 160) -> None:
        self.device = device        # spidev.SpiDev instance
        self.xfer_size = xfer_size  # bytes per SPI transfer

    def read(self, nbytes: int) -> List[int]:
        """Read exactly ``nbytes`` from SPI, in ``xfer_size``-byte chunks."""
        result: List[int] = []
        remaining = nbytes
        while remaining > 0:
            chunk = min(remaining, self.xfer_size)
            result.extend(self.device.xfer2([0x00] * chunk))
            remaining -= chunk
        return result[:nbytes]


class I2C_Interface:
    """Thin wrapper around smbus.SMBus for MI48 register access."""

    def __init__(self, device: Any, address: int) -> None:
        self.device = device    # smbus.SMBus instance
        self.address = address  # I2C device address (0x40 for MI48)

    def read_reg(self, reg: int, nbytes: int = 1) -> List[int]:
        """Read ``nbytes`` starting at ``reg``."""
        return list(self.device.read_i2c_block_data(self.address, reg, nbytes))

    def write_reg(self, reg: int, value: int) -> None:
        """Write a single byte ``value`` to ``reg``."""
        self.device.write_byte_data(self.address, reg, value & 0xFF)


# ─────────────────────────────────────────────────────────────────────────────
# MI48 sensor driver
# ─────────────────────────────────────────────────────────────────────────────

class MI48:
    """Minimal MI48 driver using spidev + smbus directly (no senxor dependency).

    Public API matches senxor.mi48.MI48 exactly so hardware.py needs no changes
    beyond the import line.

    Lifecycle::

        mi48 = MI48([i2c_iface, spi_iface], data_ready=pin, reset_handler=fn)
        mi48.enable_filter(f1=True, f2=True, f3=False)
        mi48.set_fps(9)
        mi48.start(stream=True, with_header=True)
        # … in loop …
        data, _ = mi48.read()
        mi48.stop()
    """

    def __init__(
        self,
        interfaces: List[Any],
        data_ready: Optional[Any] = None,
        reset_handler: Optional[Any] = None,
    ) -> None:
        self.i2c: I2C_Interface = interfaces[0]
        self.spi: SPI_Interface = interfaces[1]
        self._data_ready_pin = data_ready
        self._with_header: bool = False
        self.fpa_shape: Tuple[int, int] = MI48_FPA_SHAPE

        if reset_handler is not None:
            log.debug("MI48: performing hardware reset")
            try:
                reset_handler()
            except Exception as exc:
                log.warning("MI48: reset_handler raised %s", exc)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def fw_version(self) -> List[int]:
        """Read firmware version from I2C.  Returns [major, minor, patch, build]."""
        try:
            raw = self.i2c.read_reg(_REG_FW_VERSION, 4)
            log.debug("MI48: fw_version raw=%s", raw)
            return list(raw)
        except Exception as exc:
            log.warning("MI48: fw_version read failed (%s) — defaulting to [2,0,0,0]", exc)
            return [2, 0, 0, 0]  # assume fw >= 2 so filters are enabled

    # ── Configuration ─────────────────────────────────────────────────────────

    def enable_filter(self, f1: bool = True, f2: bool = True, f3: bool = False) -> None:
        """Enable/disable MI48 on-chip filters.

        f1 — median filter (spatial noise)
        f2 — temporal IIR filter (temporal noise)
        f3 — additional spatial filter (disabled by default)
        """
        try:
            self.i2c.write_reg(_REG_FILTER1, 0x01 if f1 else 0x00)
            self.i2c.write_reg(_REG_FILTER2, 0x01 if f2 else 0x00)
            self.i2c.write_reg(_REG_FILTER3, 0x01 if f3 else 0x00)
            log.debug("MI48: filters set f1=%s f2=%s f3=%s", f1, f2, f3)
        except Exception as exc:
            log.error("MI48: enable_filter failed: %s", exc)

    def set_fps(self, fps: int) -> None:
        """Set capture frame rate (1–9 Hz)."""
        fps = int(max(1, min(9, fps)))
        try:
            self.i2c.write_reg(_REG_FPS, fps)
            log.debug("MI48: FPS set to %d", fps)
        except Exception as exc:
            log.error("MI48: set_fps(%d) failed: %s", fps, exc)

    # ── Streaming ─────────────────────────────────────────────────────────────

    def start(self, stream: bool = True, with_header: bool = False) -> None:
        """Send start-stream command over I2C."""
        self._with_header = with_header
        cmd = _CMD_STREAM_ON if stream else _CMD_STREAM_OFF
        try:
            self.i2c.write_reg(_REG_STREAM, cmd)
            log.info("MI48: stream started (with_header=%s)", with_header)
        except Exception as exc:
            log.error("MI48: start() failed: %s", exc)
            raise

    def read(self) -> Tuple[Optional[List[int]], Dict[str, Any]]:
        """Read one complete thermal frame via SPI.

        Returns ``(pixel_bytes, header_dict)``.
        ``pixel_bytes`` is a list of ints, ready for ``data_to_frame()``.
        Returns ``(None, {})`` if the transfer is short or fails.
        """
        header_len = MI48_HEADER_BYTES if self._with_header else 0
        total = MI48_FRAME_BYTES + header_len
        try:
            raw = self.spi.read(total)
        except Exception as exc:
            log.error("MI48: SPI read failed: %s", exc)
            return None, {}

        if len(raw) < total:
            log.warning("MI48: short SPI read — got %d, expected %d", len(raw), total)
            return None, {}

        header: Dict[str, Any] = {}
        if self._with_header:
            header = {"raw_header": bytes(raw[:header_len])}
            raw = raw[header_len:]

        return raw, header

    def stop(self, stop_timeout: float = 0.5) -> None:
        """Send stop-stream command and wait briefly for the sensor to drain."""
        try:
            self.i2c.write_reg(_REG_STREAM, _CMD_STREAM_OFF)
            time.sleep(min(stop_timeout, 0.5))
            log.info("MI48: stream stopped")
        except Exception as exc:
            log.warning("MI48: stop() raised %s (ignored)", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Frame conversion
# ─────────────────────────────────────────────────────────────────────────────

def data_to_frame(data: List[int], shape: Tuple[int, int]) -> np.ndarray:
    """Convert raw SPI byte list to a 2-D float32 temperature array.

    Args:
        data:  List of ints (bytes) returned by ``MI48.read()``.
        shape: ``(rows, cols)`` — typically ``(62, 80)`` for MI48.

    Returns:
        float32 ndarray of shape ``(rows, cols)``, values in Kelvin.
        (Subtract 273.15 for Celsius; the downstream pipeline normalises
        to uint8 via cv.NORM_MINMAX so absolute units don't matter.)
    """
    rows, cols = shape
    n_pixels = rows * cols
    arr = np.frombuffer(bytes(data[: n_pixels * 2]), dtype=np.dtype("<u2"))  # uint16 LE
    return arr.reshape(rows, cols).astype(np.float32) * 0.01  # → Kelvin
