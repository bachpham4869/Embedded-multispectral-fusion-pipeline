"""P1 — MouseRouter: bottom-bar priority, A1 probe fallback, 250 ms debounce."""
import time

import pytest

try:
    import cv2 as cv
    _CV_LBUTTONDOWN = cv.EVENT_LBUTTONDOWN
    _CV_MOUSEMOVE = cv.EVENT_MOUSEMOVE
except ImportError:
    _CV_LBUTTONDOWN = 1
    _CV_MOUSEMOVE = 0

from smartbinocular.controls import (
    BAR_HEIGHT,
    BUTTON_IDS,
    MouseRouter,
    compute_hit_zones,
    hit_test,
)

_W, _H = 800, 480


# ── compute_hit_zones geometry ────────────────────────────────────────────────

def test_hit_zones_count():
    zones = compute_hit_zones((_W, _H))
    assert len(zones) == len(BUTTON_IDS)


def test_hit_zones_cover_all_button_ids():
    zones = compute_hit_zones((_W, _H))
    ids = [z[0] for z in zones]
    assert sorted(ids) == sorted(BUTTON_IDS)


def test_hit_zones_min_size():
    """Every button must be at least 48×48 px (WCAG 2.2 §2.5.5)."""
    for bid, x0, y0, x1, y1 in compute_hit_zones((_W, _H)):
        w = x1 - x0
        h = y1 - y0
        assert w >= 48, f"Button {bid} width {w} < 48 px"
        assert h >= 48, f"Button {bid} height {h} < 48 px"


def test_hit_zones_non_overlapping():
    zones = compute_hit_zones((_W, _H))
    for i, (bid_a, x0a, y0a, x1a, y1a) in enumerate(zones):
        for bid_b, x0b, y0b, x1b, y1b in zones[i + 1:]:
            overlap_x = x0a < x1b and x0b < x1a
            overlap_y = y0a < y1b and y0b < y1a
            assert not (overlap_x and overlap_y), f"Zones {bid_a} and {bid_b} overlap"


def test_hit_zones_within_frame():
    for bid, x0, y0, x1, y1 in compute_hit_zones((_W, _H)):
        assert x0 >= 0 and x1 <= _W, f"Button {bid} x-range [{x0},{x1}] out of frame"
        assert y0 >= 0 and y1 <= _H, f"Button {bid} y-range [{y0},{y1}] out of frame"


# ── hit_test ──────────────────────────────────────────────────────────────────

def test_hit_test_returns_none_in_scene_area():
    """Clicks well above the bar should not match any button."""
    result = hit_test(400, 100, (_W, _H))
    assert result is None


def test_hit_test_returns_button_id_in_bar():
    """Centre of each button zone should be detected."""
    for bid, x0, y0, x1, y1 in compute_hit_zones((_W, _H)):
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        assert hit_test(cx, cy, (_W, _H)) == bid, f"Centre of {bid} not detected"


def test_hit_test_returns_none_outside_all_buttons():
    """A click in the bar area but between buttons (gutter) returns None."""
    # Deliberately pick a x that falls in the left margin
    result = hit_test(2, _H - BAR_HEIGHT + 10, (_W, _H))
    assert result is None


# ── MouseRouter: bottom-bar priority over A1 probe ────────────────────────────

def test_bar_click_returns_button_not_none():
    router = MouseRouter(display_size=(_W, _H), debounce_s=0.0)
    # Click the centre of the first button
    _, x0, y0, x1, y1 = compute_hit_zones((_W, _H))[0]
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    result = router.on_event(_CV_LBUTTONDOWN, cx, cy)
    assert result is not None
    assert result in BUTTON_IDS


def test_scene_click_returns_none():
    """A click in the scene area (y < H - BAR_HEIGHT) must return None."""
    router = MouseRouter(display_size=(_W, _H), debounce_s=0.0)
    result = router.on_event(_CV_LBUTTONDOWN, 400, 100)
    assert result is None, "Scene click should return None — A1 probe area, not button bar"


def test_non_lbuttondown_returns_none():
    router = MouseRouter(display_size=(_W, _H), debounce_s=0.0)
    _, x0, y0, x1, y1 = compute_hit_zones((_W, _H))[0]
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    result = router.on_event(_CV_MOUSEMOVE, cx, cy)
    assert result is None


# ── Debounce ──────────────────────────────────────────────────────────────────

def test_debounce_suppresses_rapid_repeat():
    """Second click within debounce window must be ignored."""
    router = MouseRouter(display_size=(_W, _H), debounce_s=0.250)
    _, x0, y0, x1, y1 = compute_hit_zones((_W, _H))[0]
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2

    first = router.on_event(_CV_LBUTTONDOWN, cx, cy)
    second = router.on_event(_CV_LBUTTONDOWN, cx, cy)  # immediate repeat

    assert first is not None, "First click should be registered"
    assert second is None, "Second click within debounce window must be suppressed"


def test_debounce_allows_click_after_window():
    """Click after the debounce window must be registered."""
    router = MouseRouter(display_size=(_W, _H), debounce_s=0.050)
    _, x0, y0, x1, y1 = compute_hit_zones((_W, _H))[0]
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2

    first = router.on_event(_CV_LBUTTONDOWN, cx, cy)
    time.sleep(0.060)  # wait out the debounce window
    second = router.on_event(_CV_LBUTTONDOWN, cx, cy)

    assert first is not None
    assert second is not None, "Click after debounce window should be registered"


def test_different_buttons_independent_debounce():
    """Debounce per button — clicking a different button should not be blocked."""
    router = MouseRouter(display_size=(_W, _H), debounce_s=0.250)
    zones = compute_hit_zones((_W, _H))
    z0 = zones[0]  # first button
    z1 = zones[1]  # second button

    cx0, cy0 = (z0[1] + z0[3]) // 2, (z0[2] + z0[4]) // 2
    cx1, cy1 = (z1[1] + z1[3]) // 2, (z1[2] + z1[4]) // 2

    r0 = router.on_event(_CV_LBUTTONDOWN, cx0, cy0)
    r1 = router.on_event(_CV_LBUTTONDOWN, cx1, cy1)  # different button — no debounce

    assert r0 is not None
    assert r1 is not None, "Different button should not be debounced"


# ── is_in_bar helper ─────────────────────────────────────────────────────────

def test_is_in_bar_returns_true_for_bar_y():
    router = MouseRouter(display_size=(_W, _H))
    assert router.is_in_bar(_H - 1) is True
    assert router.is_in_bar(_H - BAR_HEIGHT) is True


def test_is_in_bar_returns_false_for_scene_y():
    router = MouseRouter(display_size=(_W, _H))
    assert router.is_in_bar(_H - BAR_HEIGHT - 1) is False
    assert router.is_in_bar(0) is False
