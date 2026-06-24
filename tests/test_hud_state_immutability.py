"""P0 — HudState must be a frozen dataclass: mutation raises FrozenInstanceError."""
import dataclasses
import pytest

from smartbinocular.hud import HudState, make_default_hud_state


def test_hud_state_is_frozen():
    state = make_default_hud_state()
    with pytest.raises((dataclasses.FrozenInstanceError, TypeError)):
        state.fps = 99.0  # type: ignore[misc]


def test_hud_state_is_dataclass():
    assert dataclasses.is_dataclass(HudState)


def test_make_default_hud_state_returns_hud_state():
    state = make_default_hud_state()
    assert isinstance(state, HudState)


def test_hud_state_fields_exist():
    state = make_default_hud_state()
    # Core display fields
    assert hasattr(state, "mode")
    assert hasattr(state, "fps")
    assert hasattr(state, "alpha")
    # Tag flags
    assert hasattr(state, "jerk_active")
    assert hasattr(state, "glare_nir")
    assert hasattr(state, "glare_th")
    assert hasattr(state, "soft_motion_active")
    assert hasattr(state, "haze_active")
    assert hasattr(state, "env_mode_active")
    assert hasattr(state, "env_stable")
    assert hasattr(state, "lean_active")
    assert hasattr(state, "e1_off")
    # Bearing
    assert hasattr(state, "bear_h")
    assert hasattr(state, "bear_v")
    assert hasattr(state, "hud_bear_enabled")
    assert hasattr(state, "a1_probe_xy")
    # New P0 fields
    assert hasattr(state, "utc_time")
    assert hasattr(state, "utc_clock_enabled")
    assert hasattr(state, "profile_label")
    assert hasattr(state, "profile_verified")
    assert hasattr(state, "net_location")
    assert hasattr(state, "capture_indicator")
    assert hasattr(state, "display_size")


def test_replace_creates_new_instance():
    state = make_default_hud_state()
    updated = dataclasses.replace(state, fps=42.0)
    assert updated.fps == 42.0
    assert state.fps != 42.0  # original unchanged
    assert updated is not state
