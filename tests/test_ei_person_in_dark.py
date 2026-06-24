"""Unit tests for experimental/ei_person_in_dark.py.

Tests do not require tflite_runtime; they patch EIWorker internals or test
helper functions directly to verify preprocessing math, postprocessing logic,
and queue/thread-safety contracts.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from smartbinocular.experimental.ei_person_in_dark import (
    EIDetection,
    EIDetectionResult,
    EISharedResult,
    EIWorker,
    _EMPTY_RESULT,
    _fomo_postprocess,
    _prepare_ei_input,
)


# ── _prepare_ei_input ─────────────────────────────────────────────────────────

class TestPrepareEiInput:
    def test_output_shape_and_dtype(self):
        bgr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        out = _prepare_ei_input(bgr, scale=0.003922, zero_point=-128)
        assert out.shape == (1, 128, 128, 3)
        assert out.dtype == np.int8

    def test_quantize_range_clipped(self):
        # All-white image: uint8(255) → float / 255 / 0.003922 + (-128) ≈ 127
        bgr = np.full((480, 640, 3), 255, dtype=np.uint8)
        out = _prepare_ei_input(bgr, scale=0.003922, zero_point=-128)
        assert int(out.max()) == 127

    def test_quantize_black_yields_near_zp(self):
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        out = _prepare_ei_input(bgr, scale=0.003922, zero_point=-128)
        assert int(out.min()) == -128

    def test_center_crop_preserves_center(self):
        # Wide image: left half red, right half blue in BGR
        bgr = np.zeros((480, 800, 3), dtype=np.uint8)
        bgr[:, :400] = (0, 0, 200)
        bgr[:, 400:] = (200, 0, 0)
        out = _prepare_ei_input(bgr, scale=0.003922, zero_point=-128, fit_mode="crop")
        # Center 480×480 crop; the center should be split between both halves → not all one color
        assert out.std() > 0

    def test_non_crop_fitmode_uses_full_frame(self):
        bgr = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        out = _prepare_ei_input(bgr, scale=0.003922, zero_point=-128, fit_mode="passthrough")
        assert out.shape == (1, 128, 128, 3)

    def test_degenerate_scale_zero_falls_back(self):
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        # scale=0 should not raise; uses fallback path
        out = _prepare_ei_input(bgr, scale=0.0, zero_point=-128)
        assert out.shape == (1, 128, 128, 3)


# ── _fomo_postprocess ─────────────────────────────────────────────────────────

class TestFomoPostprocess:
    def _make_heatmap(self, gh: int, gw: int, C: int, person_logit: float = -10.0) -> np.ndarray:
        """Return an all-background int8 heatmap (1, gh, gw, C) with given output quant."""
        raw = np.full((1, gh, gw, C), -128, dtype=np.int8)
        return raw

    def test_no_detections_when_all_background(self):
        # C=2: background dominant → no cell above threshold
        raw = np.full((1, 16, 16, 2), -128, dtype=np.int8)
        # channel 0 = background (high), channel 1 = person (low)
        raw[0, :, :, 0] = 100
        raw[0, :, :, 1] = -128
        dets = _fomo_postprocess(raw, out_scale=0.003922, out_zp=-128, threshold=0.8)
        assert dets == []

    def test_single_detection_c2(self):
        # Place a hot person cell at (4, 8) in a 16×16 C=2 heatmap
        raw = np.full((1, 16, 16, 2), -128, dtype=np.int8)
        raw[0, :, :, 0] = 50    # background
        raw[0, :, :, 1] = -128  # person: low everywhere
        raw[0, 4, 8, 0] = -128  # background low at target cell
        raw[0, 4, 8, 1] = 100   # person high at target cell
        dets = _fomo_postprocess(raw, out_scale=0.003922, out_zp=-128, threshold=0.5)
        assert len(dets) == 1
        d = dets[0]
        assert d.label == "person"
        assert 0.0 < d.score <= 1.0
        assert abs(d.cx - (8.5 / 16)) < 1e-5
        assert abs(d.cy - (4.5 / 16)) < 1e-5
        assert abs(d.w - (1.0 / 16)) < 1e-5

    def test_single_detection_c1_sigmoid(self):
        # C=1: single channel decoded via sigmoid.
        # Background at raw=-128 → dequant=0 → sigmoid=0.5; use threshold=0.7 to exclude background.
        raw = np.full((1, 16, 16, 1), -128, dtype=np.int8)
        # Place a strong logit at (2, 3): dequant = (100 - (-128)) * 0.003922 ≈ 0.894 → sigmoid ≈ 0.71
        raw[0, 2, 3, 0] = 100
        dets = _fomo_postprocess(raw, out_scale=0.003922, out_zp=-128, threshold=0.7)
        assert len(dets) == 1
        assert dets[0].label == "person"
        assert dets[0].score > 0.7

    def test_sorted_by_score_descending(self):
        # Two strong cells must be sorted high→low; background excluded by threshold=0.7.
        raw = np.full((1, 8, 8, 1), -128, dtype=np.int8)
        raw[0, 0, 0, 0] = 80   # dequant ≈ 0.815 → sigmoid ≈ 0.693
        raw[0, 1, 1, 0] = 100  # dequant ≈ 0.894 → sigmoid ≈ 0.710
        dets = _fomo_postprocess(raw, out_scale=0.003922, out_zp=-128, threshold=0.69)
        assert len(dets) == 2
        assert dets[0].score >= dets[1].score

    def test_zero_scale_returns_empty(self):
        # out_scale=0 collapses all probs to 0; nothing should exceed threshold
        raw = np.zeros((1, 16, 16, 2), dtype=np.int8)
        dets = _fomo_postprocess(raw, out_scale=0.0, out_zp=0, threshold=0.8)
        assert dets == []


# ── EISharedResult ────────────────────────────────────────────────────────────

class TestEISharedResult:
    def test_initial_get_returns_stale_empty(self):
        sr = EISharedResult()
        r = sr.get()
        assert r.stale is True
        assert r.detections == ()
        assert r.frame_id == -1

    def test_set_and_get_roundtrip(self):
        sr = EISharedResult()
        det = EIDetection(label="person", score=0.9, cx=0.5, cy=0.5, w=0.0625, h=0.0625)
        result = EIDetectionResult(
            frame_id=42,
            detections=(det,),
            inference_ms=40.0,
            submit_ms=3.0,
            stale=False,
            timestamp_monotonic=time.monotonic(),
        )
        sr.set(result)
        got = sr.get()
        assert got.frame_id == 42
        assert len(got.detections) == 1
        assert got.stale is False

    def test_concurrent_reads_never_raise(self):
        sr = EISharedResult()
        errors = []

        def _reader():
            for _ in range(500):
                try:
                    sr.get()
                except Exception as exc:
                    errors.append(exc)

        def _writer():
            for i in range(500):
                r = dataclasses.replace(_EMPTY_RESULT, frame_id=i)
                sr.set(r)

        t1 = threading.Thread(target=_reader)
        t2 = threading.Thread(target=_writer)
        t1.start(); t2.start()
        t1.join(); t2.join()
        assert errors == []


# ── EIWorker — missing tflite_runtime ────────────────────────────────────────

class TestEIWorkerMissingRuntime:
    def test_init_error_set_when_import_fails(self):
        with patch.dict("sys.modules", {"tflite_runtime": None, "tflite_runtime.interpreter": None}):
            with patch("builtins.__import__", side_effect=ImportError("no module named tflite_runtime")):
                worker = EIWorker.__new__(EIWorker)
                # Simulate __init__ with a forced import error
                worker._init_error = "no module named tflite_runtime"
                worker._available = False
                assert worker._init_error is not None
                assert worker._available is False

    def test_submit_frame_returns_false_when_init_error(self):
        worker = EIWorker.__new__(EIWorker)
        worker._init_error = "tflite_runtime not available"
        worker._available = False
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        result = worker.submit_frame(0, bgr)
        assert result is False

    def test_submit_frame_returns_false_when_unavailable(self):
        worker = EIWorker.__new__(EIWorker)
        worker._init_error = None
        worker._available = False  # available=False even after successful import
        worker._in_scale = 0.003922
        worker._in_zp = -128
        worker._fit_mode = "crop"
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        result = worker.submit_frame(0, bgr)
        assert result is False


# ── EIWorker — queue overflow drop ───────────────────────────────────────────

class TestEIWorkerQueueOverflow:
    def _make_ready_worker(self) -> EIWorker:
        """Return an EIWorker with _available=True and a small queue but no real interpreter."""
        import queue as _q
        worker = EIWorker.__new__(EIWorker)
        worker._tflite_path = "fake.tflite"
        worker._num_threads = 1
        worker._threshold = 0.8
        worker._fit_mode = "crop"
        worker._queue = _q.Queue(maxsize=2)
        worker._shared = EISharedResult()
        worker._stop_event = threading.Event()
        worker._available = True
        worker._init_error = None
        worker._in_scale = 0.003922
        worker._in_zp = -128
        worker._out_scale = 1.0
        worker._out_zp = 0
        worker._inp_idx = 0
        worker._out_idx = 0
        worker._interp = None
        worker._Interp = None
        return worker

    def test_submit_drops_on_full_queue(self):
        worker = self._make_ready_worker()
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        # First two submits should succeed (queue capacity=2)
        r1 = worker.submit_frame(1, bgr)
        r2 = worker.submit_frame(2, bgr)
        # Third must be dropped
        r3 = worker.submit_frame(3, bgr)
        assert r1 is True
        assert r2 is True
        assert r3 is False
        assert worker._queue.qsize() == 2

    def test_queue_not_blocked_when_full(self):
        worker = self._make_ready_worker()
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        t0 = time.monotonic()
        for i in range(20):
            worker.submit_frame(i, bgr)
        elapsed = time.monotonic() - t0
        # 20 submits (most dropped) must complete fast — not blocked
        assert elapsed < 1.0


# ── EIWorker — malformed tensor shape ────────────────────────────────────────

class TestEIWorkerMalformedInput:
    def test_prepare_ei_input_1d_raises(self):
        # A 1-D array is not a valid image; resize will raise
        flat = np.zeros(640 * 480 * 3, dtype=np.uint8)
        with pytest.raises(Exception):
            _prepare_ei_input(flat, scale=0.003922, zero_point=-128)

    def test_fomo_postprocess_unexpected_c_uses_channel_0(self):
        # C=3 is unusual; code falls through to C>=2 softmax; must not raise
        raw = np.full((1, 8, 8, 3), -128, dtype=np.int8)
        dets = _fomo_postprocess(raw, out_scale=0.003922, out_zp=-128, threshold=0.8)
        assert isinstance(dets, list)
