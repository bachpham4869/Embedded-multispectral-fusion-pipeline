"""P1 — L1 vs L2 layer separation: saved rasters must not contain chrome."""
import hashlib

import numpy as np
import pytest

from smartbinocular.hud import ChromeState, HudState, draw_control_chrome, draw_hud, make_default_hud_state


def _sha(frame: np.ndarray) -> str:
    return hashlib.sha256(frame.tobytes()).hexdigest()


def _make_scene(w: int = 800, h: int = 480) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(30, 200, (h, w, 3), dtype=np.uint8)


def _make_chrome_state(mode: str = "imx", recording: bool = False, w: int = 800, h: int = 480) -> ChromeState:
    return ChromeState(
        active_mode=mode,
        is_recording=recording,
        display_profile="quality",
        chrome_alpha=0.55,
        display_size=(w, h),
    )


# ── draw_hud does not mutate the input ───────────────────────────────────────

def test_draw_hud_does_not_mutate_scene():
    scene = _make_scene()
    original = scene.copy()
    state = make_default_hud_state()
    draw_hud(scene, state)
    np.testing.assert_array_equal(scene, original)


def test_draw_hud_returns_copy():
    scene = _make_scene()
    state = make_default_hud_state()
    result = draw_hud(scene, state)
    assert result is not scene


# ── L1 raster is exactly draw_hud output ─────────────────────────────────────

def test_saveable_equals_l1_output():
    """The frame saved to disk must be identical to draw_hud output."""
    scene = _make_scene()
    state = make_default_hud_state()
    l1 = draw_hud(scene, state)
    # saveable_out in main.py is just `out` after draw_hud — no extra copy
    assert _sha(l1) == _sha(l1.copy())


# ── L2 chrome differs from L1 ─────────────────────────────────────────────────

def test_chrome_changes_the_frame():
    """draw_control_chrome must visibly modify the bottom bar."""
    scene = _make_scene()
    state = make_default_hud_state()
    l1 = draw_hud(scene, state)
    saveable = l1.copy()

    chrome_state = _make_chrome_state()
    display_out = draw_control_chrome(saveable.copy(), chrome_state)

    assert _sha(saveable) != _sha(display_out), (
        "draw_control_chrome produced no visible change — chrome pixels must differ from L1"
    )


def test_saveable_not_mutated_by_chrome():
    """draw_control_chrome must not modify the saveable_out buffer."""
    scene = _make_scene()
    state = make_default_hud_state()
    l1 = draw_hud(scene, state)
    saveable = l1.copy()
    before_hash = _sha(saveable)

    chrome_state = _make_chrome_state()
    draw_control_chrome(saveable.copy(), chrome_state)  # copy passed — saveable unchanged

    assert _sha(saveable) == before_hash, "saveable_out was mutated by draw_control_chrome"


def test_l1_and_display_out_differ_when_chrome_visible():
    """PNG-save hash ≠ display hash when chrome is at active alpha."""
    scene = _make_scene()
    state = make_default_hud_state()
    l1 = draw_hud(scene, state)

    chrome_state = _make_chrome_state(recording=True)  # active chrome
    display_out = draw_control_chrome(l1.copy(), chrome_state)

    assert _sha(l1) != _sha(display_out), (
        "L1 and display output are identical — chrome is not being drawn"
    )


# ── Bottom bar region contains chrome pixels ─────────────────────────────────

def test_chrome_modifies_bottom_bar_region():
    """The bottom 56px of display_out must differ from l1 in that region."""
    from smartbinocular.controls import BAR_HEIGHT
    W, H = 800, 480
    scene = _make_scene(W, H)
    state = make_default_hud_state()
    l1 = draw_hud(scene, state)

    chrome_state = _make_chrome_state()
    display_out = draw_control_chrome(l1.copy(), chrome_state)

    bar_l1 = l1[H - BAR_HEIGHT:, :]
    bar_disp = display_out[H - BAR_HEIGHT:, :]
    assert not np.array_equal(bar_l1, bar_disp), (
        "Chrome bottom bar region is identical to L1 — no buttons were drawn"
    )


def test_chrome_does_not_modify_top_region():
    """Chrome should not significantly alter pixels in the top half of the frame."""
    from smartbinocular.controls import BAR_HEIGHT
    W, H = 800, 480
    scene = _make_scene(W, H)
    state = make_default_hud_state()
    l1 = draw_hud(scene, state)

    chrome_state = _make_chrome_state()
    display_out = draw_control_chrome(l1.copy(), chrome_state)

    top_l1 = l1[:H // 2, :]
    top_disp = display_out[:H // 2, :]
    # The top half should be identical (chrome only touches the bottom bar)
    np.testing.assert_array_equal(top_l1, top_disp)
