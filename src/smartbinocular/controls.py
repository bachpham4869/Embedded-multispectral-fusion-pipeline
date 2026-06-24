"""Input controls — soft-button bar hit-testing, mouse routing, GPIO buttons.

L2 chrome button layout (6 buttons, ASCII labels only):
    OPT | THM | FUS | CAP | REC | PROF

Hit zones are computed dynamically from display_size so the bar scales to any
resolution while keeping every target ≥ 48×48 px (WCAG 2.2 §2.5.5).

GPIO support is optional: import-guarded so Mac/CI never hard-fails.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

# ── Button IDs (canonical order; the only place this list is defined) ─────────

BUTTON_IDS: Tuple[str, ...] = ("OPT", "THM", "FUS", "CAP", "REC", "PROF")

# Buttons whose hit zone is inside the bottom bar (y ≥ H - BAR_HEIGHT).
BAR_HEIGHT: int = 56          # px — total height of the bottom chrome strip
_LEFT_MARGIN: int = 8
_RIGHT_MARGIN: int = 8
_GUTTER: int = 8

# ── Hit zone ──────────────────────────────────────────────────────────────────

def compute_hit_zones(
    display_size: Tuple[int, int],
) -> List[Tuple[str, int, int, int, int]]:
    """Return ``[(button_id, x0, y0, x1, y1), ...]`` for every button.

    Zones are non-overlapping and fill the bottom bar row. Each zone is at
    least 48×48 px at standard 800×480 resolution.
    """
    W, H = int(display_size[0]), int(display_size[1])
    n = len(BUTTON_IDS)
    total_w = W - _LEFT_MARGIN - _RIGHT_MARGIN
    btn_w = (total_w - (n - 1) * _GUTTER) // n
    y0 = H - BAR_HEIGHT
    y1 = H
    zones = []
    for i, bid in enumerate(BUTTON_IDS):
        x0 = _LEFT_MARGIN + i * (btn_w + _GUTTER)
        x1 = x0 + btn_w
        zones.append((bid, x0, y0, x1, y1))
    return zones


def hit_test(
    x: int,
    y: int,
    display_size: Tuple[int, int],
) -> Optional[str]:
    """Return button_id if (x, y) falls inside the bottom button bar, else None."""
    W, H = int(display_size[0]), int(display_size[1])
    if y < H - BAR_HEIGHT:
        return None
    for bid, x0, y0, x1, y1 in compute_hit_zones(display_size):
        if x0 <= x < x1 and y0 <= y < y1:
            return bid
    return None


# ── Mouse router ──────────────────────────────────────────────────────────────

class MouseRouter:
    """Route OpenCV mouse events with priority: button bar > A1 probe.

    Bottom bar (y ≥ H − BAR_HEIGHT) takes priority over the A1 bearing probe.
    Per-button debounce prevents rapid repeat-fire on gloved taps.

    Usage::

        router = MouseRouter(display_size=(800, 480))
        # In OpenCV mouse callback:
        result = router.on_event(event, x, y)
        if result is not None:
            # result is one of BUTTON_IDS — handle button action
        else:
            # A1 probe area — handle bearing update
    """

    def __init__(
        self,
        display_size: Tuple[int, int],
        debounce_s: float = 0.250,
    ) -> None:
        self._display_size = display_size
        self._debounce_s = debounce_s
        # Per-button last-click timestamps
        self._last_click: Dict[str, float] = {bid: 0.0 for bid in BUTTON_IDS}

    def on_event(self, event: int, x: int, y: int) -> Optional[str]:
        """Call from an OpenCV mouse callback.

        Returns the button_id if a button was clicked (and not debounced), or
        ``None`` if the click was in the A1 probe area (y < H − BAR_HEIGHT)
        or debounced.
        """
        import cv2 as cv  # local import — avoids forcing cv2 at module load time
        if event != cv.EVENT_LBUTTONDOWN:
            return None
        W, H = self._display_size
        if y >= H - BAR_HEIGHT:
            bid = hit_test(x, y, self._display_size)
            if bid is None:
                return None
            now = time.monotonic()
            if now - self._last_click[bid] < self._debounce_s:
                return None
            self._last_click[bid] = now
            return bid
        # y < H - BAR_HEIGHT → A1 probe area
        return None

    def is_in_bar(self, y: int) -> bool:
        """True if y coordinate is in the button bar zone."""
        return y >= self._display_size[1] - BAR_HEIGHT


# ── GPIO button wrapper ───────────────────────────────────────────────────────

try:
    from gpiozero import Button as _GpioButton  # type: ignore[import-untyped]
    _GPIOZERO_OK = True
except ImportError:
    _GpioButton = None
    _GPIOZERO_OK = False


class ButtonInput:
    """Wrap a physical GPIO button via gpiozero.

    Import-guarded so Mac/CI installations (no gpiozero) never hard-fail.
    Raises ``RuntimeError`` at construction time when gpiozero is absent,
    rather than silently propagating a None.

    Usage::

        btn = ButtonInput(pin=17, callback=lambda: print("pressed"))
    """

    def __init__(
        self,
        pin: int,
        callback: Callable[[], None],
        hold_time: float = 0.0,
    ) -> None:
        if not _GPIOZERO_OK or _GpioButton is None:
            raise RuntimeError(
                "gpiozero is not installed. "
                "Install with: pip install -e \".[gpio]\"  "
                "or: pip install gpiozero"
            )
        self._btn = _GpioButton(pin, hold_time=hold_time if hold_time > 0 else None)
        self._btn.when_pressed = callback

    def close(self) -> None:
        """Release the GPIO resource."""
        self._btn.close()
