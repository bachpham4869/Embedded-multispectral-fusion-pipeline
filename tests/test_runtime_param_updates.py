"""test_runtime_param_updates.py — Unit tests for Task 1: runtime parameter update wiring.

Tests:
  - _VALID_OPT_OVERRIDES_KEYS includes new thermal bilateral and NIR detail keys.
  - ThermalProcessor.update_runtime_params() updates bilateral fields in-place.
  - HybridNIREnhancer.update_runtime_params() updates detail_strength and CLAHE objects.
  - Unknown keys are ignored (no KeyError).
"""

from __future__ import annotations

import pytest


# ── _VALID_OPT_OVERRIDES_KEYS ─────────────────────────────────────────────────

def test_valid_overrides_keys_includes_nir_detail_strength():
    from smartbinocular.config import _VALID_OPT_OVERRIDES_KEYS
    assert "nir_enhancer_detail_strength" in _VALID_OPT_OVERRIDES_KEYS


def test_valid_overrides_keys_includes_thermal_bilateral():
    from smartbinocular.config import _VALID_OPT_OVERRIDES_KEYS
    for key in ("thermal_bilateral_d", "thermal_bilateral_sigma_color", "thermal_bilateral_sigma_space"):
        assert key in _VALID_OPT_OVERRIDES_KEYS, f"missing key: {key}"


# ── ThermalProcessor.update_runtime_params ────────────────────────────────────

@pytest.fixture()
def thermal_proc():
    from smartbinocular.thermal_pipeline import ThermalProcessor
    return ThermalProcessor(
        bilateral_d=5,
        bilateral_sigma_color=15.0,
        bilateral_sigma_space=5.0,
    )


def test_thermal_update_runtime_sigma_color(thermal_proc):
    thermal_proc.update_runtime_params({"thermal_bilateral_sigma_color": 25.0})
    assert thermal_proc.bilateral_sigma_color == pytest.approx(25.0)


def test_thermal_update_runtime_sigma_space(thermal_proc):
    thermal_proc.update_runtime_params({"thermal_bilateral_sigma_space": 8.0})
    assert thermal_proc.bilateral_sigma_space == pytest.approx(8.0)


def test_thermal_update_runtime_d_odd(thermal_proc):
    thermal_proc.update_runtime_params({"thermal_bilateral_d": 7})
    assert thermal_proc.bilateral_d == 7


def test_thermal_update_runtime_d_even_bumped_to_odd(thermal_proc):
    # Even d values must be bumped to maintain OpenCV bilateral contract.
    thermal_proc.update_runtime_params({"thermal_bilateral_d": 4})
    assert thermal_proc.bilateral_d % 2 == 1


def test_thermal_update_runtime_unknown_keys_ignored(thermal_proc):
    # Should not raise even if unrecognised keys are present.
    thermal_proc.update_runtime_params({"unknown_key_xyz": 99, "another_key": "value"})
    assert thermal_proc.bilateral_d == 5  # unchanged


def test_thermal_update_runtime_empty_dict_no_change(thermal_proc):
    thermal_proc.update_runtime_params({})
    assert thermal_proc.bilateral_d == 5
    assert thermal_proc.bilateral_sigma_color == pytest.approx(15.0)
    assert thermal_proc.bilateral_sigma_space == pytest.approx(5.0)


# ── HybridNIREnhancer.update_runtime_params ───────────────────────────────────

@pytest.fixture()
def nir_enhancer():
    from smartbinocular.nir_pipeline import HybridNIREnhancer
    return HybridNIREnhancer(detail_strength=0.25, clahe_clip_scale=1.0)


def test_nir_update_runtime_detail_strength(nir_enhancer):
    nir_enhancer.update_runtime_params({"nir_enhancer_detail_strength": 0.40})
    assert nir_enhancer.detail_strength == pytest.approx(0.40)


def test_nir_update_runtime_detail_strength_clamped_to_zero(nir_enhancer):
    nir_enhancer.update_runtime_params({"nir_enhancer_detail_strength": -1.0})
    assert nir_enhancer.detail_strength == pytest.approx(0.0)


def test_nir_update_runtime_clahe_clip_scale_rebuilds_levels(nir_enhancer):
    old_clahe = nir_enhancer.clahe_levels["dark"]
    nir_enhancer.update_runtime_params({"nir_enhancer_clahe_clip_scale": 1.5})
    # CLAHE objects are rebuilt; same keys must be present.
    assert set(nir_enhancer.clahe_levels.keys()) == {"very_dark", "dark", "medium"}
    assert nir_enhancer.clahe_levels["dark"] is not old_clahe  # new object created


def test_nir_update_runtime_unknown_keys_ignored(nir_enhancer):
    nir_enhancer.update_runtime_params({"completely_unknown": 42})
    assert nir_enhancer.detail_strength == pytest.approx(0.25)  # unchanged


def test_nir_update_runtime_empty_dict_no_change(nir_enhancer):
    nir_enhancer.update_runtime_params({})
    assert nir_enhancer.detail_strength == pytest.approx(0.25)
