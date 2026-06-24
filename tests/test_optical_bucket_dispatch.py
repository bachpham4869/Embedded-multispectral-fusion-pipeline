"""tests/test_optical_bucket_dispatch.py — Tests for v4 optical bucket dispatch.

Tests:
  T040 — OPTICAL_BUCKET_DISPATCH covers all 9 ENV_CLASSES + "default"
  T041 — Bucket A called for night_clear / normal_night; HybridNIREnhancer.process() invoked
  T042 — Bucket A NOT called for non-night classes (B/C/D/E/F)
  T043 — Bucket B (nir_nir_night_clahe) produces valid BGR output for nir_night
  T044 — Bucket C (nir_anti_glare_bgr) invoked for glare / backlight / normal_day
  T045 — Bucket D (nir_dehaze_lite) shape and dtype preserved
  T046 — RainTemporalMedian ring buffer returns frame immediately when < n_frames
  T047 — RainTemporalMedian produces median result after n_frames accumulate
  T048 — EnvPresetController asymmetric hysteresis: glare onset=2, decay=20
  T049 — auto_rule_preset_to_env_class covers all old preset names
  T050 — EnvPresetController accepts ENV_CLASS names (new presets in ENV_PRESETS)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from smartbinocular.env_presets import (
    ENV_PRESETS,
    EnvPresetController,
    auto_rule_preset_to_env_class,
)
from smartbinocular.feature_schema import ENV_CLASSES
from smartbinocular.nir_pipeline import (
    OPTICAL_BUCKET_DISPATCH,
    HybridNIREnhancer,
    RainTemporalMedian,
    nir_anti_glare_bgr,
    nir_dehaze_lite,
    nir_nir_night_clahe,
    nir_transition_blend,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def frame_640x480() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(30, 200, size=(480, 640, 3), dtype=np.uint8)


@pytest.fixture()
def frame_dark() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 50, size=(480, 640, 3), dtype=np.uint8)


@pytest.fixture()
def enhancer() -> HybridNIREnhancer:
    return HybridNIREnhancer(proc_w=160, proc_h=120, update_rate=1)


# ── T040: dispatch table completeness ────────────────────────────────────────

def test_dispatch_table_covers_all_env_classes():
    """T040: Every ENV_CLASS is present in OPTICAL_BUCKET_DISPATCH."""
    for cls in ENV_CLASSES:
        assert cls in OPTICAL_BUCKET_DISPATCH, f"missing: {cls}"


def test_dispatch_table_valid_bucket_keys():
    """T040b: All values are valid bucket keys A-F or 'A' fallback."""
    valid = {"A", "B", "C", "D", "E", "F"}
    for cls, bucket in OPTICAL_BUCKET_DISPATCH.items():
        assert bucket in valid, f"{cls} maps to unknown bucket '{bucket}'"


# ── T041: Bucket A called only for night classes ──────────────────────────────

def test_bucket_a_for_night_clear(frame_640x480, enhancer):
    """T041: night_clear → Bucket A → HybridNIREnhancer.process() called."""
    assert OPTICAL_BUCKET_DISPATCH["night_clear"] == "A"
    assert OPTICAL_BUCKET_DISPATCH["normal_night"] == "A"
    out = enhancer.process(frame_640x480)
    assert out.shape == frame_640x480.shape
    assert out.dtype == np.uint8


# ── T042: Bucket A NOT called for non-night classes ──────────────────────────

@pytest.mark.parametrize("env_class", [
    "nir_night", "glare", "backlight", "normal_day", "fog", "rain", "transition"
])
def test_bucket_a_not_for_day_classes(env_class):
    """T042: Non-night classes must NOT map to Bucket A (HybridNIREnhancer)."""
    bucket = OPTICAL_BUCKET_DISPATCH.get(env_class)
    assert bucket != "A", (
        f"{env_class} maps to Bucket A but should use a lighter processing path"
    )


# ── T043: Bucket B output ─────────────────────────────────────────────────────

def test_nir_nir_night_clahe_shape_dtype(frame_640x480):
    """T043: Bucket B preserves shape and dtype."""
    out = nir_nir_night_clahe(frame_640x480, clahe_clip_scale=1.0)
    assert out.shape == frame_640x480.shape
    assert out.dtype == np.uint8


def test_nir_nir_night_clahe_dark_frame_brightens(frame_dark):
    """T043b: Bucket B increases mean brightness on a very dark frame."""
    out = nir_nir_night_clahe(frame_dark, clahe_clip_scale=1.5)
    assert float(out.mean()) > float(frame_dark.mean())


# ── T044: Bucket C coverage ───────────────────────────────────────────────────

@pytest.mark.parametrize("env_class", ["glare", "backlight", "normal_day"])
def test_bucket_c_for_glare_classes(env_class):
    """T044: glare / backlight / normal_day route to Bucket C."""
    assert OPTICAL_BUCKET_DISPATCH[env_class] == "C"


def test_nir_anti_glare_passthrough_when_not_saturated():
    """T044b: Bucket C returns the same object when no highlights to compress."""
    frame = np.full((240, 320, 3), 100, dtype=np.uint8)  # mid-brightness
    out = nir_anti_glare_bgr(frame, high_pct=95.0, saturate_at=233.0)
    # Should return the identical frame (no copy) since no compression needed
    assert out is frame or np.array_equal(out, frame)


# ── T045: Bucket D shape/dtype ────────────────────────────────────────────────

def test_nir_dehaze_lite_shape_dtype(frame_640x480):
    """T045: Bucket D preserves original resolution and dtype."""
    out = nir_dehaze_lite(frame_640x480, omega=0.85)
    assert out.shape == frame_640x480.shape
    assert out.dtype == np.uint8


# ── T046-T047: Bucket E (RainTemporalMedian) ─────────────────────────────────

def test_rain_temporal_median_returns_frame_before_buffer_full():
    """T046: Before n_frames accumulate, returns the raw input frame unchanged."""
    processor = RainTemporalMedian(n_frames=3)
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    out = processor.process(frame)
    np.testing.assert_array_equal(out, frame)


def test_rain_temporal_median_after_n_frames():
    """T047: After n_frames, output is the pixel-wise median of the buffer."""
    processor = RainTemporalMedian(n_frames=3)
    h, w = 60, 80
    f1 = np.full((h, w, 3), 10, dtype=np.uint8)
    f2 = np.full((h, w, 3), 100, dtype=np.uint8)
    f3 = np.full((h, w, 3), 200, dtype=np.uint8)
    processor.process(f1)
    processor.process(f2)
    out = processor.process(f3)
    # Median of [10, 100, 200] = 100
    assert int(out[0, 0, 0]) == 100


def test_rain_temporal_median_reset():
    """T047b: reset() clears buffer so next frame is returned as-is again."""
    processor = RainTemporalMedian(n_frames=3)
    frame = np.full((60, 80, 3), 50, dtype=np.uint8)
    for _ in range(3):
        processor.process(frame)
    processor.reset()
    out = processor.process(frame)
    np.testing.assert_array_equal(out, frame)


# ── T048: Asymmetric hysteresis ───────────────────────────────────────────────

def test_glare_onset_hysteresis_is_2_frames():
    """T048a: glare preset transitions TO stable after exactly 2 frames."""
    ctrl = EnvPresetController(fallback="normal_night", hysteresis_frames=18)
    ctrl.stable_name = "normal_night"
    assert ctrl.update("glare") == "normal_night"  # streak=1
    assert ctrl.update("glare") == "glare"          # streak=2 → switches (onset=2)


def test_glare_decay_hysteresis_is_20_frames():
    """T048b: leaving glare requires 20 consecutive frames, not 18."""
    ctrl = EnvPresetController(fallback="normal_night", hysteresis_frames=18)
    ctrl.stable_name = "glare"
    # 19 frames of "normal_night" should NOT switch yet (decay=20)
    for _ in range(19):
        result = ctrl.update("normal_night")
    assert result == "glare", "should still be glare after 19 frames (decay=20)"
    # 20th frame switches
    assert ctrl.update("normal_night") == "normal_night"


def test_fog_hysteresis_onset_6_decay_18():
    """T048c: fog uses _TRANSITION_HYSTERESIS onset=6 / decay=18 (Task 2).

    Onset: 6 consecutive 'fog' updates from 'normal_night' stable → switches.
    Decay: 18 consecutive 'normal_night' updates from 'fog' stable → switches back.
    """
    # Onset: 5 frames stay normal_night; 6th switches to fog
    ctrl = EnvPresetController(fallback="normal_night", hysteresis_frames=99)  # high default; fog overrides
    for i in range(5):
        assert ctrl.update("fog") == "normal_night", f"fog must not switch before frame 6 (frame {i+1})"
    assert ctrl.update("fog") == "fog", "fog must switch on 6th frame"

    # Decay: 17 frames stay fog; 18th switches back
    for i in range(17):
        assert ctrl.update("normal_night") == "fog", f"fog must not decay before frame 18 (decay frame {i+1})"
    assert ctrl.update("normal_night") == "normal_night", "fog must decay on 18th frame"


# ── T049: auto_rule_preset_to_env_class mapping ───────────────────────────────

_OLD_PRESETS = [
    "night", "low_light", "glare_heavy", "backlight", "backlight_high_contrast",
    "fog", "haze", "night_fog", "clear_day", "day_glare", "high_contrast",
    "cluttered_bg", "low_light_cluttered", "default",
]


@pytest.mark.parametrize("preset", _OLD_PRESETS)
def test_auto_rule_preset_to_env_class_all_old_presets(preset):
    """T049: All old preset names map to a valid ENV_CLASS string."""
    result = auto_rule_preset_to_env_class(preset)
    from smartbinocular.feature_schema import ENV_CLASSES
    assert result in ENV_CLASSES, f"'{preset}' mapped to unknown ENV_CLASS '{result}'"


def test_auto_rule_preset_to_env_class_unknown_falls_back():
    """T049b: Unknown preset name returns 'normal_night' fallback."""
    assert auto_rule_preset_to_env_class("totally_unknown") == "normal_night"


# ── T050: EnvPresetController accepts ENV_CLASS names ────────────────────────

@pytest.mark.parametrize("env_class", [
    "night_clear", "normal_night", "normal_day", "glare",
    "nir_night", "rain", "transition", "fog",
])
def test_env_preset_controller_accepts_env_class_names(env_class):
    """T050: All new ENV_CLASS-aligned presets exist in ENV_PRESETS."""
    assert env_class in ENV_PRESETS, f"'{env_class}' missing from ENV_PRESETS"


def test_env_preset_controller_stabilizes_to_env_class():
    """T050b: Controller stabilizes to an ENV_CLASS name when given one.
    fog has _TRANSITION_HYSTERESIS onset=6 (Task 2), so uses 6 frames regardless
    of the controller's default hysteresis_frames.
    """
    ctrl = EnvPresetController(fallback="normal_night", hysteresis_frames=2)
    for _ in range(6):
        ctrl.update("fog")
    assert ctrl.stable_name == "fog"
