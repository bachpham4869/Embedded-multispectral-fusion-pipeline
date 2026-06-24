"""P2 — apply_profile() immutability, regression lock, and semantic correctness."""
import copy

import pytest

from smartbinocular.config import (
    CONFIG,
    PROFILES,
    PROFILE_HOTSWAP_BLACKLIST,
    RPI_THROUGHPUT_MAX_DEFAULTS,
    apply_profile,
)


# ── Immutability ─────────────────────────────────────────────────────────────

def test_apply_profile_does_not_mutate_input():
    cfg = copy.deepcopy(CONFIG)
    original = copy.deepcopy(cfg)
    apply_profile(cfg, "throughput")
    assert cfg == original, "apply_profile must not mutate the input dict"


def test_apply_profile_returns_new_dict():
    cfg = copy.deepcopy(CONFIG)
    result = apply_profile(cfg, "quality")
    assert result is not cfg


# ── Quality profile ───────────────────────────────────────────────────────────

def test_quality_profile_is_identity_except_display_profile_key():
    cfg = copy.deepcopy(CONFIG)
    result = apply_profile(cfg, "quality")
    expected = {**cfg, "display_profile": "quality"}
    assert result == expected


# ── Throughput profile — regression lock ─────────────────────────────────────

def test_throughput_profile_matches_rpi_throughput_max_defaults():
    """Regression lock: apply_profile("throughput") must be equal to merging
    RPI_THROUGHPUT_MAX_DEFAULTS into CONFIG. Prevents drift between the named
    profile and the legacy rpi_throughput_max flag."""
    cfg = copy.deepcopy(CONFIG)
    from_profile = apply_profile(cfg, "throughput")
    expected = {**cfg, **RPI_THROUGHPUT_MAX_DEFAULTS, "display_profile": "throughput"}
    assert from_profile == expected, (
        "throughput profile diverged from RPI_THROUGHPUT_MAX_DEFAULTS — update PROFILES['throughput']"
    )


def test_throughput_profile_sets_display_profile_key():
    cfg = copy.deepcopy(CONFIG)
    result = apply_profile(cfg, "throughput")
    assert result["display_profile"] == "throughput"


# ── Raw profile ───────────────────────────────────────────────────────────────

def test_raw_profile_disables_enhancer_detail():
    cfg = copy.deepcopy(CONFIG)
    result = apply_profile(cfg, "raw")
    assert result.get("nir_enhancer_detail_strength") == 0.0


def test_raw_profile_disables_clahe():
    cfg = copy.deepcopy(CONFIG)
    result = apply_profile(cfg, "raw")
    assert result.get("nir_enhancer_clahe_clip_scale") == 0.0


def test_raw_profile_disables_display_grading():
    cfg = copy.deepcopy(CONFIG)
    result = apply_profile(cfg, "raw")
    assert result.get("display_grade_mode") == "luma_only"


def test_raw_profile_sets_raw_sentinel():
    cfg = copy.deepcopy(CONFIG)
    result = apply_profile(cfg, "raw")
    assert result.get("display_profile_raw_mode") is True


def test_raw_profile_does_not_touch_blacklisted_keys():
    cfg = copy.deepcopy(CONFIG)
    result = apply_profile(cfg, "raw")
    for key in PROFILE_HOTSWAP_BLACKLIST:
        if key in cfg:
            assert result[key] == cfg[key], (
                f"Raw profile changed blacklisted startup-only key '{key}'"
            )


def test_raw_profile_sets_display_profile_key():
    cfg = copy.deepcopy(CONFIG)
    result = apply_profile(cfg, "raw")
    assert result["display_profile"] == "raw"


# ── Blacklist enforcement ─────────────────────────────────────────────────────

def test_no_profile_contains_blacklisted_keys():
    """Import-time assertion should have already caught this, but belt-and-suspenders."""
    for name, overrides in PROFILES.items():
        conflicts = PROFILE_HOTSWAP_BLACKLIST & overrides.keys()
        assert not conflicts, (
            f"Profile '{name}' contains blacklisted startup-only keys: {conflicts}"
        )


def test_apply_profile_raises_on_unknown_profile():
    cfg = copy.deepcopy(CONFIG)
    with pytest.raises(ValueError, match="Unknown profile"):
        apply_profile(cfg, "nonexistent_profile")


# ── Profile cycle completeness ────────────────────────────────────────────────

def test_all_three_profiles_defined():
    assert "quality" in PROFILES
    assert "throughput" in PROFILES
    assert "raw" in PROFILES
