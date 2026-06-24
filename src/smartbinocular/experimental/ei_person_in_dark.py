"""Experimental Edge Impulse FOMO person-detection worker (observe-only).

Loads the non-EON Linux bundle's INT8 FOMO ``.tflite`` (default path:
``models/ei/person_in_dark_fomo_int8.tflite``; copy from Edge Impulse export —
see ``models/ei/README.md``) and runs FOMO inference in a daemon thread. The
frame loop submits **display-matched** BGR frames via ``submit_frame()`` (same
resolution and processing as the on-screen preview for the active mode, before
L1 HUD); results are read non-blocking from the shared ``EIDetectionResult``.

Controlled by ``EI_PERSON_IN_DARK_ENABLED`` (default False). Does not affect env
class selection or any pipeline behavior when disabled. Removal:
``docs/EDGE_IMPULSE_FUTURE_WORK.md``.

``tflite_runtime`` is imported lazily so this module loads cleanly on systems
without the wheel.
"""

from __future__ import annotations

import dataclasses
import logging
import queue
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Public types ──────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class EIDetection:
    """One FOMO detection for a single grid cell above threshold."""

    label: str    # "person"
    score: float  # [0, 1]
    cx: float     # box centre x, normalized [0,1] in 128-px input frame
    cy: float     # box centre y, normalized [0,1]
    w: float      # normalized cell width  (= 1/grid_w; FOMO is single-cell boxes)
    h: float      # normalized cell height (= 1/grid_h)


@dataclasses.dataclass(frozen=True)
class EIDetectionResult:
    """Immutable snapshot from one inference pass. Safe for lock-free read after get()."""

    frame_id: int
    detections: Tuple[EIDetection, ...]
    inference_ms: float          # invoke() wall time (worker thread)
    submit_ms: float             # preprocessing wall time (frame-loop thread)
    stale: bool                  # True when worker errored or no result yet
    timestamp_monotonic: float


_EMPTY_RESULT = EIDetectionResult(
    frame_id=-1,
    detections=(),
    inference_ms=0.0,
    submit_ms=0.0,
    stale=True,
    timestamp_monotonic=0.0,
)


# ── Thread-safe result holder ─────────────────────────────────────────────────

class EISharedResult:
    """Lock-protected holder for the latest EIDetectionResult.

    get() never blocks: returns a stale empty result before the first inference.
    Mirrors the shape of MLSharedResult.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._result: EIDetectionResult = _EMPTY_RESULT

    def set(self, result: EIDetectionResult) -> None:
        with self._lock:
            self._result = result

    def get(self) -> EIDetectionResult:
        with self._lock:
            return self._result


# ── Worker ────────────────────────────────────────────────────────────────────

class EIWorker(threading.Thread):
    """Daemon thread running FOMO inference on the person_in_dark .tflite.

    Concurrency model mirrors MLInferenceThread:
      - Queue maxsize=2; producer uses put_nowait, drops silently on Full.
      - submit_frame() does center-crop + resize + cvtColor + INT8 quantize on
        the calling (frame-loop) thread so the worker only runs set_tensor→invoke→postproc.
      - Three consecutive invoke() failures stop the worker for the session.

    tflite_runtime is imported inside __init__; if the import fails, the worker
    logs a single WARNING and submit_frame() returns False without raising.
    """

    _MAX_CONSECUTIVE_ERRORS = 3

    def __init__(
        self,
        tflite_path: str,
        *,
        num_threads: int = 2,
        threshold: float = 0.8,
        fit_mode: str = "crop",
    ) -> None:
        super().__init__(name="EIWorker", daemon=True)
        self._tflite_path = tflite_path
        self._num_threads = int(num_threads)
        self._threshold = float(threshold)
        self._fit_mode = fit_mode

        self._queue: queue.Queue[Tuple] = queue.Queue(maxsize=2)
        self._shared = EISharedResult()
        self._stop_event = threading.Event()
        self._available = False

        # Quantization params updated in run() after allocate_tensors(); defaults match
        # typical EI image-impulse values (scale=1/255, zero_point=-128).
        self._in_scale: float = 0.003922
        self._in_zp: int = -128
        self._out_scale: float = 1.0
        self._out_zp: int = 0
        self._inp_idx: int = 0
        self._out_idx: int = 0

        # Interpreter is constructed in run() to keep it on a single thread.
        self._interp = None
        self._Interp = None
        self._init_error: Optional[str] = None

        try:
            from tflite_runtime.interpreter import Interpreter as _Interp  # type: ignore
            self._Interp = _Interp
        except ImportError as exc:
            self._init_error = str(exc)
            logger.warning(
                "[EI] tflite_runtime not available (%s) — EIWorker disabled. "
                "Install with: pip install tflite-runtime",
                exc,
            )

    @property
    def available(self) -> bool:
        return self._available

    def get_shared(self) -> EISharedResult:
        return self._shared

    def submit_frame(self, frame_id: int, bgr: np.ndarray) -> bool:
        """Preprocess bgr and post the int8 tensor to the inference queue.

        Preprocessing runs on the calling thread to keep the worker hot path minimal.
        Returns True if the item was enqueued, False if dropped (queue full,
        worker unavailable, or tflite_runtime missing).
        """
        if self._init_error is not None or not self._available:
            return False
        t0 = time.monotonic()
        try:
            tensor = _prepare_ei_input(bgr, self._in_scale, self._in_zp, self._fit_mode)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[EI] preprocessing error frame %d: %s", frame_id, exc)
            return False
        submit_ms = (time.monotonic() - t0) * 1000.0
        try:
            self._queue.put_nowait((frame_id, tensor, submit_ms))
            return True
        except queue.Full:
            return False

    def stop(self, timeout: float = 1.0) -> None:
        self._stop_event.set()
        self.join(timeout=timeout)

    def run(self) -> None:
        if self._init_error is not None or self._Interp is None:
            return

        import os as _os
        try:
            if not _os.path.isfile(self._tflite_path):
                raise FileNotFoundError(f"Not found: {self._tflite_path}")
            self._interp = self._Interp(model_path=self._tflite_path, num_threads=self._num_threads)
            self._interp.allocate_tensors()
            inp_d = self._interp.get_input_details()[0]
            out_d = self._interp.get_output_details()[0]
            self._in_scale = float(inp_d["quantization"][0])
            self._in_zp = int(inp_d["quantization"][1])
            self._out_scale = float(out_d["quantization"][0])
            self._out_zp = int(out_d["quantization"][1])
            self._inp_idx = inp_d["index"]
            self._out_idx = out_d["index"]
            self._available = True
            logger.info(
                "[EI] Worker ready — %s  output=%s  in_quant=(%.5f,%d)  out_quant=(%.5f,%d)",
                self._tflite_path,
                out_d["shape"].tolist(),
                self._in_scale, self._in_zp,
                self._out_scale, self._out_zp,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[EI] Interpreter init failed: %s — worker disabled", exc)
            return

        consecutive_errors = 0
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            frame_id, tensor, submit_ms = item
            t_invoke = time.monotonic()
            try:
                self._interp.set_tensor(self._inp_idx, tensor)
                self._interp.invoke()
                raw_out = self._interp.get_tensor(self._out_idx)
                detections = _fomo_postprocess(
                    raw_out,
                    out_scale=self._out_scale,
                    out_zp=self._out_zp,
                    threshold=self._threshold,
                )
                inference_ms = (time.monotonic() - t_invoke) * 1000.0
                self._shared.set(EIDetectionResult(
                    frame_id=frame_id,
                    detections=tuple(detections),
                    inference_ms=inference_ms,
                    submit_ms=submit_ms,
                    stale=False,
                    timestamp_monotonic=time.monotonic(),
                ))
                consecutive_errors = 0
            except Exception as exc:  # noqa: BLE001
                consecutive_errors += 1
                logger.warning(
                    "[EI] invoke error (frame %d, consecutive=%d): %s",
                    frame_id, consecutive_errors, exc,
                )
                prev = self._shared.get()
                self._shared.set(dataclasses.replace(prev, stale=True))
                if consecutive_errors >= self._MAX_CONSECUTIVE_ERRORS:
                    logger.warning(
                        "[EI] %d consecutive invoke errors — worker disabled for this session",
                        consecutive_errors,
                    )
                    self._available = False
                    return

        logger.debug("[EI] Worker thread exiting")


# ── Preprocessing (runs on frame-loop thread inside submit_frame) ─────────────

def _prepare_ei_input(
    bgr: np.ndarray,
    scale: float,
    zero_point: int,
    fit_mode: str = "crop",
) -> np.ndarray:
    """Return a (1, 128, 128, 3) int8 tensor from a BGR frame.

    Center-crop → resize to 128×128 → BGR→RGB → INT8 quantize.
    The crop default avoids letterbox black bands while keeping a square input.
    """
    import cv2

    h, w = bgr.shape[:2]
    s = min(h, w)
    if fit_mode == "crop":
        y0, x0 = (h - s) // 2, (w - s) // 2
        square = bgr[y0 : y0 + s, x0 : x0 + s]
    else:
        square = bgr
    resized = cv2.resize(square, (128, 128), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    if scale > 0.0:
        q = np.round(rgb.astype(np.float32) / 255.0 / scale + zero_point)
    else:
        # Degenerate case: identity-scale quantization (uncommon for EI image impulses)
        q = rgb.astype(np.float32) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)[np.newaxis]  # (1,128,128,3)


# ── FOMO postprocessing (runs on worker thread) ───────────────────────────────

def _fomo_postprocess(
    raw_out: np.ndarray,
    *,
    out_scale: float,
    out_zp: int,
    threshold: float,
) -> List[EIDetection]:
    """Decode a FOMO int8 heatmap (1, gh, gw, C) into EIDetection objects.

    C=2: softmax over background+person channels, person is channel 1.
    C=1: sigmoid on the single channel.
    One detection per qualifying grid cell; no NMS needed (single class, one box per cell).
    Detections are sorted descending by score for deterministic output.
    """
    probs = (raw_out.astype(np.float32) - out_zp) * out_scale
    _, gh, gw, C = probs.shape

    if C >= 2:
        shifted = probs - probs.max(axis=-1, keepdims=True)
        exp_ = np.exp(shifted)
        person_p: np.ndarray = (exp_ / exp_.sum(axis=-1, keepdims=True))[0, :, :, 1]
    else:
        person_p = 1.0 / (1.0 + np.exp(-probs[0, :, :, 0]))

    hits = np.argwhere(person_p >= threshold)
    detections: List[EIDetection] = []
    for row, col in hits:
        detections.append(EIDetection(
            label="person",
            score=float(person_p[row, col]),
            cx=(col + 0.5) / gw,
            cy=(row + 0.5) / gh,
            w=1.0 / gw,
            h=1.0 / gh,
        ))
    detections.sort(key=lambda d: d.score, reverse=True)
    return detections
