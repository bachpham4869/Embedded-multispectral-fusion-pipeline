"""Non-blocking video recorder — daemon thread + bounded queue.

Follows the MLLogger backpressure pattern (utils.py:284-350):
  - Hot path calls ``write_nowait`` which uses ``put_nowait`` and drops on Full.
  - Writer thread is daemon so it never blocks clean process exit.
  - Caller checks ``disk_limit_hit`` to show a HUD toast and stop recording.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import Any, Callable, Optional, Tuple

logger = logging.getLogger(__name__)

import cv2 as cv
import numpy as np

_SENTINEL = object()


class VideoRecorder:
    """Write video frames on a daemon thread; never block the frame loop.

    Usage::

        rec = VideoRecorder(max_mb=2048, codec="mp4v")
        rec.start(path, fps=30.0, frame_size=(800, 480))
        rec.write_nowait(frame)   # hot path — never blocks
        rec.stop()

    When the on-disk file exceeds ``max_mb``, the writer self-stops and sets
    ``disk_limit_hit``. The frame loop should check this and show a HUD toast.
    """

    DISK_CHECK_INTERVAL = 150  # frames between os.path.getsize checks (~5 s at 30 fps)

    def __init__(
        self,
        max_mb: int = 2048,
        codec: str = "mp4v",
        queue_maxsize: int = 60,
    ) -> None:
        self._max_bytes: int = max_mb * 1024 * 1024
        self._codec_str: str = codec
        self._queue_maxsize: int = queue_maxsize
        self._q: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._thread: Optional[threading.Thread] = None
        self._path: str = ""
        self.dropped_count: int = 0
        self._active: bool = False
        self._disk_limit_hit: bool = False
        # Injected in tests to avoid real cv.VideoWriter
        self._writer_factory: Optional[Callable[[], Any]] = None

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def disk_limit_hit(self) -> bool:
        return self._disk_limit_hit

    def start(
        self,
        path: str,
        fps: float,
        frame_size: Tuple[int, int],
        *,
        _writer_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        """Open VideoWriter and start the daemon writer thread."""
        self._path = path
        self._disk_limit_hit = False
        self.dropped_count = 0
        self._active = True

        _dir = os.path.dirname(path)
        if _dir:
            os.makedirs(_dir, exist_ok=True)

        self._writer_factory = _writer_factory

        self._thread = threading.Thread(
            target=self._worker,
            args=(path, max(1.0, float(fps)), frame_size),
            name="VideoRecorder",
            daemon=True,
        )
        self._thread.start()

    def write_nowait(self, frame: np.ndarray) -> None:
        """Submit a frame without blocking. Increments dropped_count on queue full."""
        if not self._active:
            return
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            self.dropped_count += 1

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the writer to finish and wait for the thread to join."""
        if not self._active:
            return
        self._active = False
        try:
            self._q.put_nowait(_SENTINEL)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _worker(self, path: str, fps: float, frame_size: Tuple[int, int]) -> None:
        if self._writer_factory is not None:
            writer = self._writer_factory()
        else:
            fourcc = cv.VideoWriter_fourcc(*self._codec_str)
            writer = cv.VideoWriter(path, fourcc, fps, frame_size)

        frame_count = 0
        try:
            while True:
                try:
                    item = self._q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if item is _SENTINEL:
                    break
                if writer is not None:
                    writer.write(item)
                frame_count += 1
                if frame_count % self.DISK_CHECK_INTERVAL == 0 and self._path:
                    try:
                        if os.path.getsize(self._path) >= self._max_bytes:
                            self._disk_limit_hit = True
                            self._active = False
                            break
                    except OSError:
                        logger.warning("disk-size check failed — disk limit enforcement disabled", exc_info=True)
        finally:
            if writer is not None:
                writer.release()
