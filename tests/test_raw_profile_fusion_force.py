"""P2 — raw profile + fusion mode forces single-sensor display.

These tests verify the behavioural contract documented in docs/UI_FIELD_CHECKLIST.md:
  - apply_profile(cfg, "raw") sets display_profile_raw_mode=True (sentinel)
  - the raw profile does not set a display_mode override itself — mode forcing
    is handled by the frame loop dispatcher in main.py; tests here validate
    the config semantics and simulate the dispatcher logic in isolation.
  - switching away from raw does NOT auto-restore fusion (one-way switch).
"""
from smartbinocular.config import (
    CONFIG,
    PROFILE_HOTSWAP_MIN_INTERVAL_S,
    PROFILES,
    apply_profile,
)


def _base() -> dict:
    return dict(CONFIG)


# ── Sentinel flag ─────────────────────────────────────────────────────────────

def test_raw_profile_sets_raw_mode_sentinel():
    cfg = apply_profile(_base(), "raw")
    assert cfg.get("display_profile_raw_mode") is True, (
        "display_profile_raw_mode sentinel must be True when profile=raw"
    )


def test_non_raw_profiles_do_not_set_sentinel():
    for name in ("quality", "throughput"):
        cfg = apply_profile(_base(), name)
        assert not cfg.get("display_profile_raw_mode"), (
            f"Profile '{name}' must not set display_profile_raw_mode"
        )


# ── Dispatcher logic (simulated, no frame loop) ───────────────────────────────

class _MockDispatcher:
    """Minimal simulation of the profile-cycle dispatcher in main.py.

    Records whether a mode_switch_pending was triggered and what toast was queued.
    """
    def __init__(self, mode: str, current_profile: str):
        self.mode = mode
        self.current_profile = current_profile
        self.mode_switch_pending = None
        self.toast_text = None
        self._profile_cycle = list(PROFILES.keys())
        self._last_swap_t = -(PROFILE_HOTSWAP_MIN_INTERVAL_S + 1.0)

    def cycle_profile(self, base_cfg: dict, now: float = 1000.0) -> dict:
        if now - self._last_swap_t < PROFILE_HOTSWAP_MIN_INTERVAL_S:
            return apply_profile(base_cfg, self.current_profile)
        idx = self._profile_cycle.index(self.current_profile)
        next_profile = self._profile_cycle[(idx + 1) % len(self._profile_cycle)]
        cfg = apply_profile(base_cfg, next_profile)
        self.current_profile = next_profile
        self._last_swap_t = now
        self.toast_text = f"PROFILE: {self.current_profile.upper()}"
        if self.current_profile == "raw" and self.mode == "fusion":
            self.mode_switch_pending = "imx"
            self.toast_text = "RAW: single sensor only"
        return cfg


def test_raw_in_fusion_triggers_mode_switch():
    """When profile cycles to raw while mode=fusion, mode_switch_pending is set to imx."""
    d = _MockDispatcher(mode="fusion", current_profile="throughput")
    base = _base()
    # cycle: throughput → raw
    d.cycle_profile(base)
    assert d.current_profile == "raw"
    assert d.mode_switch_pending == "imx", (
        "mode_switch_pending must be 'imx' when raw profile is applied in fusion mode"
    )
    assert d.toast_text == "RAW: single sensor only"


def test_raw_in_non_fusion_does_not_trigger_mode_switch():
    """Raw profile while mode=imx or thermal must NOT force a mode switch."""
    for mode in ("imx", "thermal"):
        d = _MockDispatcher(mode=mode, current_profile="throughput")
        d.cycle_profile(_base())
        assert d.current_profile == "raw"
        assert d.mode_switch_pending is None, (
            f"mode_switch_pending must be None when raw applied in mode={mode}"
        )


def test_leaving_raw_does_not_auto_restore_fusion():
    """Cycling away from raw must NOT set mode_switch_pending back to fusion."""
    d = _MockDispatcher(mode="imx", current_profile="raw")
    # cycle: raw → quality
    d.cycle_profile(_base(), now=2000.0)
    assert d.current_profile == "quality"
    assert d.mode_switch_pending is None, (
        "Exiting raw profile must not auto-restore fusion — operator must press [FUS] or key 3"
    )


def test_rate_limit_blocks_rapid_swaps():
    """A second cycle within PROFILE_HOTSWAP_MIN_INTERVAL_S must be ignored."""
    d = _MockDispatcher(mode="imx", current_profile="quality")
    d.cycle_profile(_base(), now=100.0)    # quality → throughput
    assert d.current_profile == "throughput"
    d.cycle_profile(_base(), now=100.5)   # within rate limit window — ignored
    assert d.current_profile == "throughput", (
        "Second cycle within rate-limit window must not advance profile"
    )
    d.cycle_profile(_base(), now=100.0 + PROFILE_HOTSWAP_MIN_INTERVAL_S + 0.1)  # after window
    assert d.current_profile == "raw"


def test_cycle_wraps_around():
    """Profile cycle must wrap: quality → throughput → raw → quality."""
    d = _MockDispatcher(mode="imx", current_profile="quality")
    t = 0.0
    gap = PROFILE_HOTSWAP_MIN_INTERVAL_S + 0.1
    d.cycle_profile(_base(), now=t); t += gap
    assert d.current_profile == "throughput"
    d.cycle_profile(_base(), now=t); t += gap
    assert d.current_profile == "raw"
    d.cycle_profile(_base(), now=t); t += gap
    assert d.current_profile == "quality"


# ── Config contract when raw is applied ──────────────────────────────────────

def test_raw_profile_disables_nir_enhancement():
    cfg = apply_profile(_base(), "raw")
    assert cfg.get("nir_enhancer_detail_strength") == 0.0
    assert cfg.get("nir_enhancer_clahe_clip_scale") == 0.0


def test_raw_profile_sets_display_profile_key():
    cfg = apply_profile(_base(), "raw")
    assert cfg["display_profile"] == "raw"


def test_quality_after_raw_clears_raw_sentinel():
    """Cycling back to quality must not carry the raw sentinel forward."""
    base = _base()
    cfg_raw = apply_profile(base, "raw")
    assert cfg_raw.get("display_profile_raw_mode") is True
    # quality is applied onto base (not cfg_raw) — sentinel must be absent
    cfg_quality = apply_profile(base, "quality")
    assert not cfg_quality.get("display_profile_raw_mode"), (
        "quality profile must not inherit display_profile_raw_mode from raw"
    )
