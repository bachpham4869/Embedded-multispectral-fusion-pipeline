"""Regression tests for infer_env_tags_auto_rule (rule-based ENV fallback)."""

from __future__ import annotations

from smartbinocular.env_presets import infer_env_tags_auto_rule, select_env_preset_from_tags


def test_night_band_low_spatial_std_does_not_tag_haze():
    """Pi logs: uniform dark NIR on max_side≈128 gray yields std≈2–7 < std_low; not fog."""
    tags = infer_env_tags_auto_rule(
        nir_b_ema=16.0,
        nir_gray_std=2.14,
        glare_nir=False,
        haze_config_on=False,
        std_low=20.0,
        std_high=52.0,
    )
    assert "night" in tags
    assert "haze" not in tags
    assert select_env_preset_from_tags(tags) == "night"


def test_low_light_low_std_still_tags_haze():
    """Above night brightness: diffuse flat scene may still be haze."""
    tags = infer_env_tags_auto_rule(
        nir_b_ema=30.0,
        nir_gray_std=5.0,
        glare_nir=False,
        haze_config_on=False,
        std_low=20.0,
        std_high=52.0,
    )
    assert "low_light" in tags
    assert "haze" in tags


def test_unknown_brightness_low_std_tags_haze_when_ema_none():
    tags = infer_env_tags_auto_rule(
        nir_b_ema=None,
        nir_gray_std=3.0,
        glare_nir=False,
        haze_config_on=False,
        std_low=20.0,
        std_high=52.0,
    )
    assert "haze" in tags


def test_haze_config_on_adds_fog_tag_independent_of_std():
    tags = infer_env_tags_auto_rule(
        nir_b_ema=16.0,
        nir_gray_std=40.0,
        glare_nir=False,
        haze_config_on=True,
        std_low=20.0,
        std_high=52.0,
    )
    assert "fog" in tags
