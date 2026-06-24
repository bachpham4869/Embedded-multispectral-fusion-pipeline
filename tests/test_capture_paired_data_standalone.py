from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np


def test_capture_paired_data_imports_without_src_fallback():
    module = importlib.import_module("tools.capture_paired_data")

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "smartbinocular.mi48_driver" not in source
    assert callable(module.data_to_frame)


def test_thermal_normalizer_accepts_celsius_frames():
    module = importlib.import_module("tools.capture_paired_data")

    normalizer = module.RobustThermalNormalizer(min_valid_pixels=4, ema_alpha=1.0)
    frame = np.array(
        [
            [21.0, 22.0, 23.0, 24.0],
            [25.0, 26.0, 27.0, 28.0],
        ],
        dtype=np.float32,
    )

    img = normalizer.normalize(frame)

    assert img.dtype == np.uint8
    assert int(img.max()) > int(img.min())
    assert normalizer.last_unit == "celsius_temperature"
    assert normalizer.scale_text().endswith("C(C)")
