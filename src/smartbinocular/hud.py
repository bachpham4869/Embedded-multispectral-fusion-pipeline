"""
HUD rendering for SmartBinocular — Layer 1 (data only, always burned into saved frames).

Inspired by public HMI legibility guidance (NASA HFACS, ISO 9241-303, MIL-STD-1472H as
reference); not a certified mil-spec implementation.

Layer model
-----------
L1  data_hud  — draw_hud()          — burned into all captured PNGs and video frames
L2  chrome    — draw_control_chrome() (P1) — on-screen only; excluded from saved rasters
"""
from __future__ import annotations

import dataclasses
import time
from typing import Optional, Tuple

import cv2 as cv
import numpy as np

# ── Typography tokens (single source of truth) ───────────────────────────────
_FONT = cv.FONT_HERSHEY_SIMPLEX

_FS_LABEL = 0.42       # labels, tags, secondary text
_FS_MODE = 0.55        # mode name in top bar
_FS_NUMERIC = 0.45     # live numerics (FPS, alpha)
_FS_ALERT = 0.55       # caution / critical alerts

_TH_NORMAL = 1
_TH_ALERT = 2

_BGR_LABEL = (180, 220, 255)    # cool-white: labels, tags
_BGR_NUMERIC = (0, 255, 0)      # NV-green: live numerics
_BGR_CAUTION = (0, 220, 255)    # amber: caution alerts
_BGR_CRITICAL = (0, 0, 255)     # red: critical / unverified badge
_BGR_SHADOW = (0, 0, 0)         # drop-shadow

# ── Safe zone constants (800×480 reference) ───────────────────────────────────
_TOP_BAR_Y = 22          # baseline of top-bar text
_TAG_STRIP_Y = 44        # baseline of tag strip / secondary row
_NET_LOC_Y = 62          # baseline of optional net-loc line
_RIGHT_MARGIN = 10       # px from right edge
_LEFT_MARGIN = 8         # px from left edge


# ── HudState frozen dataclass ────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class HudState:
    """Immutable snapshot of HUD data for one frame.

    All fields read here; none are mutated. Use ``dataclasses.replace()`` to derive
    updated states.
    """
    # Core display
    mode: str = "imx"                 # "imx" | "thermal" | "fusion"
    fps: float = 0.0
    alpha: float = 0.55               # fusion blend weight
    display_size: tuple = (800, 480)  # (W, H)

    # Alert / status tags
    jerk_active: bool = False
    glare_nir: bool = False
    glare_th: bool = False
    soft_motion_active: bool = False
    haze_active: bool = False
    env_mode_active: bool = False
    env_stable: str = "normal_night"
    lean_active: bool = False
    e1_off: bool = False

    # Bearing / A1 probe (crosshair + dH/dV)
    bear_h: float = 0.0
    bear_v: float = 0.0
    hud_bear_enabled: bool = True
    a1_probe_xy: tuple = (400, 240)   # (x, y) in display pixels

    # Clock (P0) — gated by flag for A/B parity checks against pre-refactor builds
    utc_clock_enabled: bool = True
    utc_time: str = ""                # pre-formatted "2026-05-04 14:32:18Z"

    # Processing profile badge
    profile_label: str = "QUALITY"   # "QUALITY" | "THROUGHPUT" | "RAW"
    profile_verified: bool = True     # False → show [UNVERIFIED] badge in red

    # Optional network-derived coarse location (P2.5)
    # None → line entirely omitted (no placeholder, no "NO FIX" text)
    net_location: Optional[str] = None

    # Transient indicators (toasts etc.)
    capture_indicator: Optional[str] = None   # ASCII-only: e.g. "REC" "CAP" "RAW: single sensor only"


def make_default_hud_state() -> HudState:
    """Return a zero-valued HudState suitable for tests and as a constructor baseline."""
    return HudState(
        utc_time=time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
    )


# ── Internal helpers ─────────────────────────────────────────────────────────

def _put(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    font_scale: float,
    bgr: tuple,
    thickness: int = _TH_NORMAL,
    shadow: bool = True,
) -> None:
    """Draw text with a 1 px black drop-shadow for contrast on bright imagery."""
    if shadow:
        cv.putText(img, text, (x + 1, y + 1), _FONT, font_scale, _BGR_SHADOW, thickness, cv.LINE_AA)
    cv.putText(img, text, (x, y), _FONT, font_scale, bgr, thickness, cv.LINE_AA)


def _put_right(
    img: np.ndarray,
    text: str,
    y: int,
    font_scale: float,
    bgr: tuple,
    thickness: int = _TH_NORMAL,
    right_margin: int = _RIGHT_MARGIN,
) -> None:
    """Right-align text within the frame width."""
    w = img.shape[1]
    (tw, _), _ = cv.getTextSize(text, _FONT, font_scale, thickness)
    x = max(_LEFT_MARGIN, w - tw - right_margin)
    _put(img, text, x, y, font_scale, bgr, thickness)


# ── Public API ───────────────────────────────────────────────────────────────

def draw_hud(scene_bgr: np.ndarray, state: HudState) -> np.ndarray:
    """Render Layer-1 HUD onto a copy of scene_bgr and return it.

    Does NOT mutate the input. The returned frame is safe to save to disk.
    Control chrome (L2) is NOT rendered here — see draw_control_chrome() in P1.
    """
    out = scene_bgr.copy()
    h, w = out.shape[:2]

    # ── Top bar ──────────────────────────────────────────────────────────────
    # Left: UTC clock (when enabled)
    if state.utc_clock_enabled and state.utc_time:
        _put(out, state.utc_time, _LEFT_MARGIN, _TOP_BAR_Y, _FS_LABEL, _BGR_LABEL)

    # Centre-left: mode name + raw preview indicator
    mode_labels = {"imx": "OPT", "thermal": "THM", "fusion": "FUS"}
    mode_text = mode_labels.get(state.mode, state.mode.upper())
    # Place mode label after clock; estimate clock width (≈12px/char at FS_LABEL 0.42)
    clock_offset = (len(state.utc_time) * 12 + 16) if (state.utc_clock_enabled and state.utc_time) else _LEFT_MARGIN
    _put(out, mode_text, clock_offset, _TOP_BAR_Y, _FS_MODE, _BGR_NUMERIC)

    # Right: FPS + alpha (fusion) + profile badge
    profile_text = f"PROFILE:{state.profile_label}"
    fps_text = f"FPS:{state.fps:.0f} a:{state.alpha:.2f}"
    _put_right(out, fps_text, _TOP_BAR_Y, _FS_NUMERIC, _BGR_NUMERIC)

    # Profile badge (right rail, second row)
    _put_right(out, profile_text, _TAG_STRIP_Y - 2, _FS_LABEL, _BGR_LABEL)
    if not state.profile_verified and state.profile_label != "QUALITY":
        (tw, _), _ = cv.getTextSize(profile_text, _FONT, _FS_LABEL, _TH_NORMAL)
        badge_x = max(_LEFT_MARGIN, w - tw - _RIGHT_MARGIN - 110)
        _put(out, "[UNVERIFIED]", badge_x, _TAG_STRIP_Y - 2, _FS_LABEL, _BGR_CRITICAL, shadow=True)

    # ── Tag strip ─────────────────────────────────────────────────────────────
    tag_parts: list[str] = []
    if state.jerk_active:
        tag_parts.append("JERK")
    if state.glare_nir or state.glare_th:
        tag_parts.append("GLARE")
    if state.soft_motion_active:
        tag_parts.append("MOTION")
    if state.haze_active:
        tag_parts.append("HAZE")
    if state.env_mode_active:
        tag_parts.append(f"ENV:{state.env_stable}")
    if state.lean_active:
        tag_parts.append("LEAN")
    if state.e1_off:
        tag_parts.append("E1:OFF")
    if tag_parts:
        _put(out, "[" + " ".join(tag_parts) + "]", _LEFT_MARGIN, _TAG_STRIP_Y, _FS_LABEL, _BGR_LABEL)

    # ── Optional net-location line ────────────────────────────────────────────
    # Rendered only when net_location is a non-None string. Absence is silent.
    if state.net_location is not None:
        _put(out, state.net_location, _LEFT_MARGIN, _NET_LOC_Y, _FS_LABEL, _BGR_LABEL)

    # ── Optional toast / capture indicator ────────────────────────────────────
    if state.capture_indicator is not None:
        _put(
            out,
            state.capture_indicator,
            _LEFT_MARGIN,
            h - 12,
            _FS_ALERT,
            _BGR_CAUTION,
            thickness=_TH_ALERT,
        )

    # ── Bearing: crosshair + A1 probe circle ─────────────────────────────────
    if state.hud_bear_enabled:
        cx, cy = w // 2, h // 2
        cv.line(out, (cx - 18, cy), (cx + 18, cy), _BGR_CAUTION, 1)
        cv.line(out, (cx, cy - 18), (cx, cy + 18), _BGR_CAUTION, 1)
        pu = int(np.clip(state.a1_probe_xy[0], 0, w - 1))
        pv = int(np.clip(state.a1_probe_xy[1], 0, h - 1))
        cv.circle(out, (pu, pv), 5, (0, 255, 255), 1)
        bh_text = f"A1 dH={state.bear_h:+.2f} dV={state.bear_v:+.2f}"
        _put_right(out, bh_text, _TAG_STRIP_Y, _FS_LABEL, (200, 230, 255))

    return out


# ── Layer 2: control chrome ───────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class ChromeState:
    """Immutable snapshot of L2 control chrome state for one frame.

    ``chrome_alpha`` drives the backing-strip opacity (0.15 faded → 0.55 active).
    ``is_recording`` makes the REC button red. ``display_profile`` is shown in PROF.
    """
    active_mode: str        # "imx" | "thermal" | "fusion"
    is_recording: bool
    display_profile: str    # "quality" | "throughput" | "raw"
    chrome_alpha: float     # 0.15..0.55
    display_size: Tuple[int, int]  # (W, H)
    rec_dropped: int = 0    # shown in REC button label when > 0


# ── Button rendering helpers ──────────────────────────────────────────────────

_BTN_LABEL_MAP = {
    "OPT": "OPT",
    "THM": "THM",
    "FUS": "FUS",
    "CAP": "CAP",
    "REC": "REC",
    "PROF": "PROF",
}

_BGR_BTN_ACTIVE   = (60, 160, 60)    # green tint — active mode button
_BGR_BTN_REC      = (30, 30, 200)    # red — recording active
_BGR_BTN_NORMAL   = (40, 40, 40)     # dark — idle buttons
_BGR_BTN_TEXT     = (230, 230, 230)  # near-white button labels
_BGR_BTN_TEXT_ACT = (255, 255, 255)  # bright-white active button label
_BGR_BAR_BG       = (0, 0, 0)        # backing strip base colour

_BAR_HEIGHT = 56   # must match controls.BAR_HEIGHT


def _mode_to_btn(mode: str) -> str:
    return {"imx": "OPT", "thermal": "THM", "fusion": "FUS"}.get(mode, "OPT")


def _prof_suffix(profile: str) -> str:
    return {"quality": "Q", "throughput": "T", "raw": "R"}.get(profile, "?")


def draw_control_chrome(out: np.ndarray, state: ChromeState) -> np.ndarray:
    """Render L2 control chrome onto *out* in-place and return it.

    The caller must pass a copy of the L1 saveable frame so the saved raster
    is not contaminated::

        display_out = draw_control_chrome(saveable_out.copy(), chrome_state)

    L2 chrome is NEVER written to disk — it is for on-screen display only.
    """
    from smartbinocular.controls import compute_hit_zones  # local to avoid circular import

    H, W = out.shape[:2]
    alpha = float(state.chrome_alpha)

    # ── Semi-transparent backing strip ────────────────────────────────────────
    bar_y0 = H - _BAR_HEIGHT
    overlay = out.copy()
    cv.rectangle(overlay, (0, bar_y0), (W, H), _BGR_BAR_BG, -1)
    cv.addWeighted(overlay, alpha, out, 1.0 - alpha, 0, out)

    # ── Buttons ───────────────────────────────────────────────────────────────
    active_btn = _mode_to_btn(state.active_mode)
    zones = compute_hit_zones(state.display_size)

    # Single composite pass: one allocation + one blend instead of 6 + 6.
    # Pixel-equivalent to the per-button loop when zones are non-overlapping (invariant
    # guaranteed by compute_hit_zones which lays out 6 equal-width non-overlapping cells).
    btn_overlay = out.copy()
    for bid, x0, y0, x1, y1 in zones:
        if bid == "REC" and state.is_recording:
            bg = _BGR_BTN_REC
        elif bid in ("OPT", "THM", "FUS") and bid == active_btn:
            bg = _BGR_BTN_ACTIVE
        else:
            bg = _BGR_BTN_NORMAL
        cv.rectangle(btn_overlay, (x0, y0), (x1, y1), bg, -1)
    cv.addWeighted(btn_overlay, 0.7, out, 0.3, 0, out)

    # Border + label pass (direct draw — no overlay needed)
    for bid, x0, y0, x1, y1 in zones:
        cv.rectangle(out, (x0, y0), (x1, y1), (80, 80, 80), 1)

        label = _BTN_LABEL_MAP[bid]
        if bid == "PROF":
            label = f"PRF:{_prof_suffix(state.display_profile)}"
        elif bid == "REC" and state.is_recording:
            label = "REC" if state.rec_dropped == 0 else f"REC-{state.rec_dropped}"

        txt_col = _BGR_BTN_TEXT_ACT if (
            (bid in ("OPT", "THM", "FUS") and bid == active_btn)
            or (bid == "REC" and state.is_recording)
        ) else _BGR_BTN_TEXT

        (tw, th), _ = cv.getTextSize(label, _FONT, _FS_LABEL, _TH_NORMAL)
        tx = x0 + (x1 - x0 - tw) // 2
        ty = y0 + (y1 - y0 + th) // 2
        _put(out, label, tx, ty, _FS_LABEL, txt_col, shadow=True)

    return out


def draw_debug_overlays(
    out: np.ndarray,
    *,
    ml_active: bool = False,
    ml_name1: str = "",
    ml_proba1: float = 0.0,
    ml_name2: str = "",
    ml_proba2: float = 0.0,
    ei_enabled: bool = False,
    ei_n: int = 0,
    ei_best: str = "--",
    ei_inference_ms: float = 0.0,
    ei_age_ms: float = 0.0,
    ei_stale: bool = False,
    ei_bboxes: Optional[list] = None,
) -> np.ndarray:
    """Render ML + EI debug overlays onto out in-place. Requires debug=True to be meaningful.

    Kept separate from draw_hud so the L1 raster (saved to disk) can be produced
    without debug overlays if desired.
    """
    if ml_active and ml_name1:
        ml_text = f"ML:{ml_name1} {ml_proba1:.2f} | {ml_name2} {ml_proba2:.2f}"
        _put_right(out, ml_text, 88, _FS_LABEL, _BGR_LABEL)

    if ei_enabled:
        stale_tag = " [stale]" if ei_stale else ""
        ei_text = (
            f"EI: person x{ei_n} best={ei_best} "
            f"inf={ei_inference_ms:.0f}ms age={ei_age_ms:.0f}ms{stale_tag}"
        )
        _put_right(out, ei_text, 110, _FS_LABEL, (0, 165, 255))

        if ei_bboxes and ei_n > 0:
            dw, dh = out.shape[1], out.shape[0]
            for det in ei_bboxes:
                bx = int((det.cx - det.w / 2) * dw)
                by = int((det.cy - det.h / 2) * dh)
                bw = max(1, int(det.w * dw))
                bh = max(1, int(det.h * dh))
                cv.rectangle(out, (bx, by), (bx + bw, by + bh), (0, 165, 255), 1)

    return out
