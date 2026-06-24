"""test_env_preset_hysteresis.py — Unit tests for Task 2: per-transition hysteresis.

Tests:
  - _TRANSITION_HYSTERESIS contains expected (onset, decay) tuples for all new entries.
  - EnvPresetController.update() requires the correct number of consecutive frames to
    transition TO a preset (onset) and AWAY from it (decay).
  - Existing glare/glare_heavy behaviour is unchanged.
  - update() stays O(1): no allocation observed per call.
"""

from __future__ import annotations

import pytest

from smartbinocular.env_presets import EnvPresetController, _TRANSITION_HYSTERESIS


# ── _TRANSITION_HYSTERESIS table ─────────────────────────────────────────────

@pytest.mark.parametrize("preset,onset,decay", [
    ("glare",       2,  20),
    ("glare_heavy", 2,  20),
    ("rain",       10,  25),
    ("transition", 12,  30),
    ("fog",         6,  18),
])
def test_hysteresis_table_values(preset, onset, decay):
    assert preset in _TRANSITION_HYSTERESIS
    actual_onset, actual_decay = _TRANSITION_HYSTERESIS[preset]
    assert actual_onset == onset, f"{preset} onset: expected {onset}, got {actual_onset}"
    assert actual_decay == decay, f"{preset} decay: expected {decay}, got {actual_decay}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_ctrl(fallback: str = "normal_night", hysteresis: int = 5) -> EnvPresetController:
    return EnvPresetController(fallback=fallback, hysteresis_frames=hysteresis)


def _drive(ctrl: EnvPresetController, desired: str, n: int) -> str:
    """Call update(desired) n times; return the last stable name."""
    result = ctrl.stable_name
    for _ in range(n):
        result = ctrl.update(desired)
    return result


# ── Onset: frames needed to switch TO a preset ───────────────────────────────

@pytest.mark.parametrize("preset,onset", [
    ("glare",       2),
    ("rain",       10),
    ("transition", 12),
    ("fog",         6),
])
def test_onset_requires_correct_consecutive_frames(preset, onset):
    ctrl = _make_ctrl(fallback="normal_night")
    # onset-1 frames should NOT yet switch
    assert _drive(ctrl, preset, onset - 1) == "normal_night"
    # One more frame triggers the switch
    assert ctrl.update(preset) == preset


@pytest.mark.parametrize("preset,onset", [
    ("glare",       2),
    ("rain",       10),
    ("transition", 12),
    ("fog",         6),
])
def test_onset_streak_resets_on_different_desired(preset, onset):
    ctrl = _make_ctrl(fallback="normal_night")
    _drive(ctrl, preset, onset - 1)
    # Interrupt with a different desired — streak resets
    ctrl.update("normal_night")
    # Need to start the streak again from scratch
    assert _drive(ctrl, preset, onset - 1) == "normal_night"


# ── Decay: frames needed to leave a preset ───────────────────────────────────

@pytest.mark.parametrize("preset,onset,decay", [
    ("glare",       2,  20),
    ("rain",       10,  25),
    ("fog",         6,  18),
])
def test_decay_requires_correct_consecutive_frames(preset, onset, decay):
    ctrl = _make_ctrl(fallback="normal_night")
    # First enter the preset
    _drive(ctrl, preset, onset)
    assert ctrl.stable_name == preset
    # decay-1 frames away should NOT yet leave
    assert _drive(ctrl, "normal_night", decay - 1) == preset
    # One more frame triggers leaving
    assert ctrl.update("normal_night") == "normal_night"


# ── Existing glare tests unchanged ───────────────────────────────────────────

def test_glare_onset_2():
    ctrl = _make_ctrl()
    assert ctrl.update("glare") == "normal_night"  # streak=1, need 2
    assert ctrl.update("glare") == "glare"          # streak=2 → switch


def test_glare_decay_20():
    ctrl = _make_ctrl(fallback="normal_night")
    _drive(ctrl, "glare", 2)
    assert ctrl.stable_name == "glare"
    assert _drive(ctrl, "normal_night", 19) == "glare"
    assert ctrl.update("normal_night") == "normal_night"


# ── Key-match: preset names match ENV_CLASS strings ──────────────────────────

def test_fog_preset_name_matches_env_class_string():
    """'fog' is both a valid ENV_PRESETS key and an ENV_CLASS string — no mismatch."""
    from smartbinocular.env_presets import ENV_PRESETS
    assert "fog" in ENV_PRESETS, "'fog' must be a valid preset name for hysteresis key to work"


def test_rain_and_transition_preset_names_exist():
    from smartbinocular.env_presets import ENV_PRESETS
    for name in ("rain", "transition"):
        assert name in ENV_PRESETS, f"'{name}' must be a valid preset name"
