"""LUT false-color for thermal / fusion heatmap (replaces per-frame applyColorMap)."""

from __future__ import annotations

import numpy as np

from smartbinocular.hardware import get_thermal_colormap_lut_bgr, gray_to_thermal_bgr


def test_lut_cache_shapes():
    for L in (32, 64, 128, 256):
        lut = get_thermal_colormap_lut_bgr(L)
        assert lut.shape == (L, 3)
        assert lut.dtype == np.uint8


def test_gray_to_thermal_bgr_matched_apply_color_map_at_256():
    import cv2 as cv
    from smartbinocular.hardware import THERMAL_COLORMAP

    g = (np.random.RandomState(0).randint(0, 256, size=(24, 32))).astype(np.uint8)
    ref = cv.applyColorMap(g, THERMAL_COLORMAP)
    got = gray_to_thermal_bgr(g, levels=256)
    np.testing.assert_array_equal(got, ref)


def test_quantized_levels_produces_valid_bgr():
    g = np.linspace(0, 255, 12 * 8, dtype=np.uint8).reshape(12, 8)
    c = gray_to_thermal_bgr(g, levels=32)
    assert c.shape == (12, 8, 3)
    assert c.dtype == np.uint8
    assert c.max() <= 255
