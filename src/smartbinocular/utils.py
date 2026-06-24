"""
Cross-cutting helpers for the live fusion loop in ``main``.

Includes per-frame downsampling (:class:`FrameCache`), stream sync quality
(:class:`StreamSkewQualityGate`), capture signing (:class:`CaptureIntegrityChain`),
async ML JSONL logging (:class:`MLLogger`), and optional stage timing
(:class:`StageProfiler`).
"""

from __future__ import annotations

import dataclasses
import hashlib
import hmac as _hmac_module
import json
import os
import queue
import threading
import time
from collections import deque
from typing import Any, Dict, Optional, Tuple

import cv2 as cv
import numpy as np


# ─── FrameCache: single resize pyramid per captured frame ────────────────────

@dataclasses.dataclass
class FrameCache:
    """Precomputed frame pyramid built ONCE per frame to eliminate redundant resizes.

    nir_gray is max_side=128 green channel ([:,:,1]) of nir_small_bgr — NIR luminance proxy.
    nir_small_bgr is the same resize, BGR (HSV-S / saturation features).
    nir_320 / nir_160 use pyrDown which applies Gaussian anti-aliasing before downscale.
    When ``skip_nir_160`` is used, ``nir_160`` is None (LK path must pyrDown from ``nir_320``).
    """
    nir_full: np.ndarray    # 640×480 BGR  (native capture)
    nir_320: np.ndarray     # 320×240 BGR  (pyrDown once)
    nir_160: Optional[np.ndarray]  # 160×120 BGR, or None if second pyrDown skipped
    nir_small_bgr: np.ndarray  # max_side=`small_max_side` BGR (same geometry as nir_gray)
    nir_gray: np.ndarray    # same max_side as nir_small; green channel — NIR luminance proxy
    thermal_80: np.ndarray  # 80×62 raw uint8 or float32 from sensor
    ts: float               # monotonic timestamp of frame construction


def build_frame_cache(
    nir_bgr: np.ndarray,
    thermal_raw: np.ndarray,
    ts: float,
    *,
    skip_nir_160: bool = False,
    small_max_side: int = 128,
) -> FrameCache:
    """Build FrameCache from raw capture frames. ~0.4 ms on RPi4 at 640×480.

    Called once per frame immediately after capture. All downstream consumers
    read from the cache — no further resizes needed.

    Set ``skip_nir_160=True`` when LK optical flow is off to avoid a second pyrDown
    when ``nir_160`` is unused. Set ``small_max_side`` (64–128) to trade accuracy
    for less resize work in JerkGate / feature stats.
    """
    nir_320 = cv.pyrDown(nir_bgr)       # 640×480 → 320×240
    nir_160: Optional[np.ndarray] = None
    if not skip_nir_160:
        nir_160 = cv.pyrDown(nir_320)   # 320×240 → 160×120

    mss = int(np.clip(small_max_side, 64, 128))
    h, w = nir_bgr.shape[:2]
    m = max(h, w)
    if m > mss:
        scale = float(mss) / float(m)
        tw = max(1, int(w * scale))
        th = max(1, int(h * scale))
        small = cv.resize(nir_bgr, (tw, th), interpolation=cv.INTER_AREA)
    else:
        small = nir_bgr
    nir_gray = small[:, :, 1]  # green channel as NIR luminance proxy; view, no copy

    return FrameCache(
        nir_full=nir_bgr,
        nir_320=nir_320,
        nir_160=nir_160,
        nir_small_bgr=small,
        nir_gray=nir_gray,
        thermal_80=thermal_raw,
        ts=ts,
    )


# ── Homography quality metric (corner drift under ``H``) ────────────────────

def compute_homography_quality(
    H: np.ndarray,
    thermal_shape: Tuple[int, int],
    nir_shape: Tuple[int, int],
) -> Dict[str, Any]:
    """Project thermal corners through H; return drift and containment metrics.

    Args:
        H:              3×3 homography matrix (thermal → NIR coordinate space).
        thermal_shape:  (height, width) of the thermal frame, e.g. (62, 80).
        nir_shape:      (height, width) of the NIR frame, e.g. (480, 640).

    Returns:
        {
            "max_corner_drift_px": float   — max L2 distance from each projected
                                             corner to the NIR frame centre.
                                             Use as a drift proxy; not a true
                                             reprojection error.
            "all_corners_within_nir": bool — True if all four projected corners
                                             lie within the NIR frame boundaries.
        }
    """
    th, tw = thermal_shape
    nh, nw = nir_shape
    corners = np.array([
        [0,        0,        1.0],
        [tw - 1,   0,        1.0],
        [tw - 1,   th - 1,   1.0],
        [0,        th - 1,   1.0],
    ], dtype=np.float64).T  # shape (3, 4)

    proj = H.astype(np.float64) @ corners   # (3, 4)
    proj_xy = (proj[:2] / proj[2:3]).T      # (4, 2) — (x, y) in NIR px

    nir_centre = np.array([[nw / 2.0, nh / 2.0]])
    max_drift = float(np.linalg.norm(proj_xy - nir_centre, axis=1).max())
    within = bool(
        np.all(proj_xy[:, 0] >= 0) and np.all(proj_xy[:, 0] < nw)
        and np.all(proj_xy[:, 1] >= 0) and np.all(proj_xy[:, 1] < nh)
    )
    return {
        "max_corner_drift_px": round(max_drift, 3),
        "all_corners_within_nir": within,
    }


# ─── Per-capture integrity: dHash + HMAC chain ───────────────────────────────

def dhash(img: np.ndarray, size: int = 8) -> int:
    """64-bit difference perceptual hash for tamper detection. ~0.05 ms."""
    gray = cv.resize(cv.cvtColor(img, cv.COLOR_BGR2GRAY), (size + 1, size))
    diff = gray[:, 1:] > gray[:, :-1]
    return int(sum(bool(b) << i for i, b in enumerate(diff.flatten())))


def sign_capture(meta: dict, key: bytes) -> str:
    """HMAC-SHA256 of JSON-serialised capture metadata. ~0.01 ms."""
    payload = json.dumps(meta, sort_keys=True).encode()
    return _hmac_module.new(key, payload, hashlib.sha256).hexdigest()


class CaptureIntegrityChain:
    """Merkle-style capture chain: each JSON carries the HMAC of its own content
    plus the previous capture's HMAC, binding all captures in order.

    Key is loaded from a device-local 32-byte binary file (created on first run).
    """

    def __init__(self, key_path: str = ".session_key"):
        self._key = self._load_or_create_key(key_path)
        self._prev_hash: Optional[str] = None

    @staticmethod
    def _load_or_create_key(path: str) -> bytes:
        if os.path.isfile(path):
            with open(path, "rb") as f:
                data = f.read(32)
                if len(data) == 32:
                    return data
        key = os.urandom(32)
        with open(path, "wb") as f:
            f.write(key)
        return key

    def sign(self, meta: dict) -> dict:
        """In-place: add frame_dhash chain link and hmac_sha256 signature."""
        if self._prev_hash is not None:
            meta["prev_capture_hash"] = self._prev_hash
        sig = sign_capture(meta, self._key)
        meta["hmac_sha256"] = sig
        self._prev_hash = sig
        return meta


class FPSCounter:
    """Sliding-window FPS estimate from ``perf_counter`` deltas (HUD / metrics)."""

    def __init__(self, window: int = 30):
        """Keep the last ``window`` timestamps for smoothing."""
        self.times: deque = deque(maxlen=window)
        self.fps: float = 0.0

    def tick(self) -> float:
        """Record the current time and return the estimated FPS over the window."""
        now = time.perf_counter()
        self.times.append(now)
        if len(self.times) >= 2:
            dt = self.times[-1] - self.times[0]
            self.fps = (len(self.times) - 1) / dt if dt > 0 else 0
        return self.fps


class StreamSkewQualityGate:
    """S6: classify NIR–thermal timestamp skew into GOOD / DEGRADED / BAD with hysteresis.

    Smooths raw skew milliseconds with an EMA and requires sustained threshold crossings
    before changing state, avoiding HUD flicker on single-frame spikes.
    """

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

    def update(self, skew_ms: Optional[float]) -> Tuple[str, Optional[float], bool]:
        """Update internal EMA and state machine; return ``(state, ema_ms, state_changed)``."""
        # No sample: hold previous classification; do not flag a transition.
        if skew_ms is None:
            return self.state, self.ema_ms, False
        s = float(skew_ms)
        if self.ema_ms is None:
            self.ema_ms = s
        else:
            self.ema_ms = (1.0 - self.ema_alpha) * self.ema_ms + self.ema_alpha * s
        prev = self.state
        e = float(self.ema_ms)
        # Hysteresis: require ``hold_frames`` consecutive threshold hits to leave a state.
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
        else:
            # BAD: recover toward DEGRADED only after skew EMA drops materially.
            if e <= self.bad_back_ms:
                self._strike += 1
                if self._strike >= self.hold_frames:
                    self.state = "DEGRADED"
                    self._strike = 0
            else:
                self._strike = 0
        return self.state, self.ema_ms, (self.state != prev)


# ── ML: background JSONL writer (non-blocking feature log) ──────────────────

_ML_LOGGER_STOP = object()


class MLLogger:
    """Append-only JSONL writer on a daemon thread: buffer + time-based flush.

    Hot path calls ``log(dict)`` with ``put_nowait`` — never blocks the frame loop.
    Flushes when the buffer reaches ``buffer_size`` records or ``flush_interval_s``
    elapses, whichever comes first.
    """

    def __init__(
        self,
        output_path: str,
        *,
        buffer_size: int = 100,
        flush_interval_s: float = 30.0,
        queue_maxsize: int = 1000,
    ) -> None:
        self._path = os.path.abspath(output_path)
        _dir = os.path.dirname(self._path)
        if _dir:
            os.makedirs(_dir, exist_ok=True)
        self._buffer_size = max(1, int(buffer_size))
        self._flush_interval_s = float(flush_interval_s)
        self._q: queue.Queue = queue.Queue(maxsize=max(16, int(queue_maxsize)))
        self._thread = threading.Thread(target=self._worker, name="MLLogger", daemon=True)
        self._thread.start()

    def _flush(self, items: list[Dict[str, Any]]) -> None:
        if not items:
            return
        with open(self._path, "a", encoding="utf-8") as f:
            for d in items:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def _worker(self) -> None:
        buf: list[Dict[str, Any]] = []
        last_flush = time.monotonic()
        while True:
            remaining = self._flush_interval_s - (time.monotonic() - last_flush)
            timeout = min(0.5, max(0.05, remaining)) if remaining > 0 else 0.05
            try:
                item = self._q.get(timeout=timeout)
            except queue.Empty:
                item = None
            if item is _ML_LOGGER_STOP:
                break
            if item is not None:
                buf.append(item)
                if len(buf) >= self._buffer_size:
                    self._flush(buf)
                    buf.clear()
                    last_flush = time.monotonic()
            elif buf and (time.monotonic() - last_flush) >= self._flush_interval_s:
                self._flush(buf)
                buf.clear()
                last_flush = time.monotonic()
        if buf:
            self._flush(buf)

    def log(self, record: Dict[str, Any]) -> None:
        try:
            self._q.put_nowait(record)
        except queue.Full:
            pass

    def close(self, timeout: float = 8.0) -> None:
        self._q.put(_ML_LOGGER_STOP)
        self._thread.join(timeout=timeout)


# ── Per-stage latency profiler (optional session JSON attachment) ───────────

class StageProfiler:
    """Online per-stage latency accumulator for the main frame loop.

    Usage:
        profiler = StageProfiler()
        with profiler("nir_bucket"):
            nir_enhanced = bucket_fn(frame)
        stats = profiler.stats()  # {stage: {mean_ms, std_ms, n}}

    Stores only (sum, sum_sq, n) per stage — O(1) memory regardless of session
    length.  Thread-unsafe by design (intended for single-threaded frame loop).
    """

    def __init__(self) -> None:
        """Create empty accumulators for named stages."""
        self._sums: Dict[str, float] = {}
        self._sum_sqs: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}
        self._t0: Optional[float] = None
        self._label: Optional[str] = None

    def __call__(self, stage: str) -> "StageProfiler":
        self._label = stage
        return self

    def __enter__(self) -> "StageProfiler":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        if self._t0 is not None and self._label is not None:
            ms = (time.perf_counter() - self._t0) * 1000.0
            s = self._label
            self._sums[s] = self._sums.get(s, 0.0) + ms
            self._sum_sqs[s] = self._sum_sqs.get(s, 0.0) + ms * ms
            self._counts[s] = self._counts.get(s, 0) + 1
        self._t0 = None
        self._label = None

    def stats(self) -> Dict[str, Dict[str, float]]:
        """Return {stage: {mean_ms, std_ms, n}} for all recorded stages."""
        result: Dict[str, Dict[str, float]] = {}
        for s, n in self._counts.items():
            mean = self._sums[s] / n
            var = max(0.0, self._sum_sqs[s] / n - mean * mean)
            result[s] = {
                "mean_ms": round(mean, 3),
                "std_ms": round(var ** 0.5, 3),
                "n": n,
            }
        return result

    def reset(self) -> None:
        self._sums.clear()
        self._sum_sqs.clear()
        self._counts.clear()

