"""P1 — VideoRecorder: daemon thread + drop-on-full backpressure."""
import time
import threading
from typing import Optional

import numpy as np
import pytest

from smartbinocular.recording import VideoRecorder


class _SlowWriter:
    """Mock cv.VideoWriter that sleeps write_delay_s per frame."""

    def __init__(self, write_delay_s: float = 0.050) -> None:
        self._delay = write_delay_s
        self.frames_written = 0
        self._lock = threading.Lock()

    def write(self, frame: np.ndarray) -> None:
        time.sleep(self._delay)
        with self._lock:
            self.frames_written += 1

    def release(self) -> None:
        pass


class _NullWriter:
    """Mock writer that writes instantly (no-op)."""

    def write(self, frame: np.ndarray) -> None:
        pass

    def release(self) -> None:
        pass


def _dummy_frame(w: int = 800, h: int = 480) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ── write_nowait never blocks ─────────────────────────────────────────────────

def test_write_nowait_does_not_block_under_load():
    """200 rapid write_nowait calls must each return in < 2 ms."""
    slow = _SlowWriter(write_delay_s=0.050)
    rec = VideoRecorder(max_mb=2048, queue_maxsize=60)
    rec.start("dummy.mp4", fps=30.0, frame_size=(800, 480), _writer_factory=lambda: slow)

    frame = _dummy_frame()
    max_call_s = 0.0
    for _ in range(200):
        t0 = time.perf_counter()
        rec.write_nowait(frame)
        elapsed = time.perf_counter() - t0
        max_call_s = max(max_call_s, elapsed)

    rec.stop(timeout=2.0)

    assert max_call_s < 0.002, (
        f"write_nowait blocked for {max_call_s*1000:.1f} ms — must stay < 2 ms"
    )


# ── Frames are dropped when queue is full ────────────────────────────────────

def test_dropped_count_increases_when_queue_saturates():
    """With a slow writer and queue_maxsize=60, 200 rapid submits must produce drops."""
    slow = _SlowWriter(write_delay_s=0.050)
    rec = VideoRecorder(max_mb=2048, queue_maxsize=60)
    rec.start("dummy.mp4", fps=30.0, frame_size=(800, 480), _writer_factory=lambda: slow)

    frame = _dummy_frame()
    for _ in range(200):
        rec.write_nowait(frame)

    rec.stop(timeout=2.0)

    assert rec.dropped_count > 0, (
        "Expected dropped_count > 0 when 200 frames are submitted faster than the slow writer "
        "can consume them through a queue of size 60"
    )


# ── Total frames = written + dropped = submitted ──────────────────────────────

def test_written_plus_dropped_equals_submitted():
    """Every submitted frame is either written or dropped — none lost silently."""
    slow = _SlowWriter(write_delay_s=0.050)
    rec = VideoRecorder(max_mb=2048, queue_maxsize=60)
    rec.start("dummy.mp4", fps=30.0, frame_size=(800, 480), _writer_factory=lambda: slow)

    submitted = 200
    frame = _dummy_frame()
    for _ in range(submitted):
        rec.write_nowait(frame)

    rec.stop(timeout=5.0)  # wait for queue to drain

    total = slow.frames_written + rec.dropped_count
    assert total == submitted, (
        f"frames_written({slow.frames_written}) + dropped({rec.dropped_count}) = {total} "
        f"!= submitted({submitted})"
    )


# ── is_active flag ────────────────────────────────────────────────────────────

def test_is_active_reflects_state():
    null = _NullWriter()
    rec = VideoRecorder(queue_maxsize=60)
    assert not rec.is_active

    rec.start("dummy.mp4", fps=30.0, frame_size=(800, 480), _writer_factory=lambda: null)
    assert rec.is_active

    rec.stop(timeout=1.0)
    assert not rec.is_active


# ── write_nowait is a no-op when not active ───────────────────────────────────

def test_write_nowait_noop_when_not_active():
    rec = VideoRecorder()
    frame = _dummy_frame()
    rec.write_nowait(frame)  # should not raise
    assert rec.dropped_count == 0


# ── stop is idempotent ────────────────────────────────────────────────────────

def test_stop_is_idempotent():
    null = _NullWriter()
    rec = VideoRecorder(queue_maxsize=60)
    rec.start("dummy.mp4", fps=30.0, frame_size=(800, 480), _writer_factory=lambda: null)
    rec.stop(timeout=1.0)
    rec.stop(timeout=1.0)  # must not raise
