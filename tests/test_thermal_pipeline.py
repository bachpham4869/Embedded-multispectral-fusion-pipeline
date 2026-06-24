"""P1-G: Tests for thermal_pipeline — Kalman BG, EMA warmup, ThermalProcessor shape contract."""

from __future__ import annotations

import numpy as np
import pytest

from smartbinocular.thermal_pipeline import KalmanThermalBackground, ThermalProcessor


def _frame(value: float, shape: tuple = (62, 80)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


# ── KalmanThermalBackground ───────────────────────────────────────────────────

class TestKalmanThermalBackground:
    def test_not_ready_before_first_frame(self):
        bg = KalmanThermalBackground()
        assert not bg.is_ready

    def test_ready_after_first_frame(self):
        bg = KalmanThermalBackground()
        bg.update(_frame(100))
        assert bg.is_ready

    def test_converges_to_constant_value_in_5_frames(self):
        """High initial P=100 means the first frame dominates; 5 identical frames
        should bring the estimate within 2 counts of the true value."""
        bg = KalmanThermalBackground()
        target = 80.0
        for _ in range(5):
            bg.update(_frame(target))
        assert bg._x is not None
        np.testing.assert_allclose(bg._x, target, atol=2.0)

    def test_heat_map_shape_after_ready(self):
        bg = KalmanThermalBackground()
        cold = _frame(80)
        bg.update(cold)
        heat = bg.get_heat_map(_frame(120))
        assert heat is not None
        assert heat.shape == cold.shape
        assert heat.dtype == np.uint8

    def test_fg_mask_shape_after_ready(self):
        bg = KalmanThermalBackground()
        cold = _frame(80)
        bg.update(cold)
        mask = bg.get_foreground_mask(_frame(130))
        assert mask is not None
        assert mask.shape == cold.shape
        assert mask.dtype == np.uint8

    def test_heat_map_none_before_ready(self):
        bg = KalmanThermalBackground()
        # _x is None before first update
        bg._x = None
        assert bg.get_heat_map(_frame(100)) is None

    def test_fg_mask_none_before_ready(self):
        bg = KalmanThermalBackground()
        bg._x = None
        assert bg.get_foreground_mask(_frame(100)) is None


# ── ThermalProcessor ──────────────────────────────────────────────────────────

class TestThermalProcessor:
    def test_not_ready_before_first_frame(self):
        proc = ThermalProcessor()
        assert not proc.is_ready

    def test_ready_after_ema_warmup(self):
        """Default uses EMA cold-frame: needs ``warmup`` identical frames before ``is_ready``."""
        w = 3
        proc = ThermalProcessor(warmup=w, use_kalman_background=False)
        for _ in range(w - 1):
            proc.process(_frame(100))
            assert not proc.is_ready
        proc.process(_frame(100))
        assert proc.is_ready

    def test_ready_after_first_process_kalman(self):
        proc = ThermalProcessor(use_kalman_background=True)
        proc.process(_frame(100))
        assert proc.is_ready

    def test_process_returns_4_tuple(self):
        proc = ThermalProcessor()
        result = proc.process(_frame(128))
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_denoised_shape_and_dtype(self):
        proc = ThermalProcessor()
        raw = _frame(128)
        denoised, *_ = proc.process(raw)
        assert denoised.shape == raw.shape
        assert denoised.dtype == np.uint8

    def test_enhanced_shape_and_dtype(self):
        proc = ThermalProcessor()
        raw = _frame(128)
        _, enhanced, *_ = proc.process(raw)
        assert enhanced.shape == raw.shape
        assert enhanced.dtype == np.uint8

    def test_heat_map_shape_after_ema_warmup(self):
        w = 3
        proc = ThermalProcessor(warmup=w, use_kalman_background=False)
        for _ in range(w):
            _, _, heat_map, _ = proc.process(_frame(128))
        assert heat_map is not None
        assert heat_map.shape == (62, 80)

    def test_fg_mask_shape_after_ema_warmup(self):
        w = 3
        proc = ThermalProcessor(warmup=w, use_kalman_background=False)
        for _ in range(w):
            _, _, _, fg_mask = proc.process(_frame(128))
        assert fg_mask is not None
        assert fg_mask.shape == (62, 80)
