"""P1-G: Tests for night-relevant nir_pipeline paths only.

Covers buckets A (HybridNIREnhancer), D (nir_dehaze_lite), E (RainTemporalMedian).
Buckets B (nir_nir_night_clahe) and C (nir_a1_lite_tone_map_bgr) are out of scope
for the night-vision thesis evaluation surface and are intentionally excluded.
"""

from __future__ import annotations

import numpy as np
import pytest

from smartbinocular.nir_pipeline import (
    HybridNIREnhancer,
    RainTemporalMedian,
    nir_dehaze_lite,
)


def _bgr(h: int = 120, w: int = 160, value: int = 80) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


# ── Bucket D: nir_dehaze_lite ─────────────────────────────────────────────────

class TestNirDehazeLite:
    def test_output_shape_matches_input(self):
        frame = _bgr(480, 640)
        out = nir_dehaze_lite(frame)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    def test_small_input_shape_preserved(self):
        frame = _bgr(62, 80)
        out = nir_dehaze_lite(frame)
        assert out.shape == frame.shape

    def test_output_is_uint8(self):
        out = nir_dehaze_lite(_bgr(120, 160, value=40))
        assert out.dtype == np.uint8


# ── Bucket E: RainTemporalMedian ──────────────────────────────────────────────

class TestRainTemporalMedian:
    def test_passthrough_until_buffer_full(self):
        """Returns the raw frame while the buffer has fewer than n_frames entries."""
        rmed = RainTemporalMedian(n_frames=3)
        frame = _bgr(value=100)
        out1 = rmed.process(frame)
        out2 = rmed.process(frame)
        # Both should be the raw frame (not yet at full buffer)
        np.testing.assert_array_equal(out1, frame)
        np.testing.assert_array_equal(out2, frame)

    def test_median_of_known_triple(self):
        """Pixel-wise median of frames with values 100, 200, 150 should be 150."""
        rmed = RainTemporalMedian(n_frames=3)
        h, w = 4, 4
        f1 = np.full((h, w, 3), 100, dtype=np.uint8)
        f2 = np.full((h, w, 3), 200, dtype=np.uint8)
        f3 = np.full((h, w, 3), 150, dtype=np.uint8)
        rmed.process(f1)
        rmed.process(f2)
        out = rmed.process(f3)
        expected = np.full((h, w, 3), 150, dtype=np.uint8)
        np.testing.assert_array_equal(out, expected)

    def test_output_dtype_is_uint8(self):
        rmed = RainTemporalMedian(n_frames=2)
        frame = _bgr(value=80)
        rmed.process(frame)
        out = rmed.process(frame)
        assert out.dtype == np.uint8

    def test_reset_clears_buffer(self):
        rmed = RainTemporalMedian(n_frames=2)
        rmed.process(_bgr(value=50))
        rmed.reset()
        out = rmed.process(_bgr(value=200))
        # Buffer cleared, so still in passthrough
        assert out.dtype == np.uint8
        assert out.shape == (120, 160, 3)


# ── Bucket A: HybridNIREnhancer ───────────────────────────────────────────────

class TestHybridNIREnhancer:
    def test_output_shape_matches_input_dark(self):
        """Dark frame (value=20) exercises the cur_bright < 0.25 night boost branch."""
        enh = HybridNIREnhancer()
        frame = _bgr(480, 640, value=20)
        out = enh.process(frame)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    def test_output_shape_matches_input_medium(self):
        """Medium brightness (value=80) exercises the dark/medium CLAHE branch."""
        enh = HybridNIREnhancer()
        frame = _bgr(480, 640, value=80)
        out = enh.process(frame)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    def test_output_shape_with_precomputed_small(self):
        """When precomputed_small matches proc_w×proc_h it is used directly."""
        enh = HybridNIREnhancer(proc_w=320, proc_h=240)
        frame = _bgr(480, 640, value=50)
        small = _bgr(240, 320, value=50)
        out = enh.process(frame, precomputed_small=small)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    def test_multiple_frames_do_not_change_output_shape(self):
        """Shape contract must hold across frames (brightness_buffer grows)."""
        enh = HybridNIREnhancer()
        frame = _bgr(480, 640, value=30)
        for _ in range(5):
            out = enh.process(frame)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8
