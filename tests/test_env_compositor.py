"""tests/test_env_compositor.py — Unit tests for the ML top-2 compositor.

Tests:
  TC01 — top-1 below primary threshold → (None, None)
  TC02 — top-2 below secondary threshold → (class_1, None)
  TC03 — top-2 absent → (class_1, None)
  TC04 — redundant night-axis pair → collapse to (top-1, None)
  TC05 — day primary + night secondary → night preferred, day dropped
  TC06 — night primary + day secondary → night kept, day dropped
  TC07 — night primary + fog secondary → (night, "fog")
  TC08 — fog primary + night secondary → (fog, night_flavor)
  TC09 — primary=glare, secondary=normal_day → (glare, None) (not in hints)
  TC10 — apply_secondary_hint None → copy, not same object
  TC11 — apply_secondary_hint "fog" → fusion_alpha_boost applied
  TC12 — apply_secondary_hint unknown hint → no change
  TC13 — regression: fog top-1 + night top-2 → bucket D via OPTICAL_BUCKET_DISPATCH
"""

from __future__ import annotations

from typing import Optional

import pytest

from smartbinocular.env_presets import (
    apply_secondary_hint,
    compose_env_from_ml_top2,
)
from smartbinocular.nir_pipeline import OPTICAL_BUCKET_DISPATCH


# ── compose_env_from_ml_top2 parametrized cases ───────────────────────────────

@pytest.mark.parametrize("class_1,proba_1,class_2,proba_2,pri_thr,sec_thr,exp_env,exp_hint", [
    # TC01: top-1 below primary threshold → ML must not drive
    ("normal_night", 0.50, "fog", 0.25, 0.62, 0.20, None, None),
    # TC01b: class_1 is None
    (None, 0.80, "fog", 0.25, 0.62, 0.20, None, None),
    # TC02: top-2 below secondary threshold → single-label mode
    ("normal_night", 0.80, "fog", 0.10, 0.62, 0.20, "normal_night", None),
    # TC03: top-2 absent
    ("normal_night", 0.80, None, 0.0, 0.62, 0.20, "normal_night", None),
    # TC03b: same class_1 == class_2 → single-label
    ("normal_night", 0.80, "normal_night", 0.30, 0.62, 0.20, "normal_night", None),
    # TC04: redundant night pairs — collapse to top-1
    ("night_clear", 0.75, "normal_night", 0.22, 0.62, 0.20, "night_clear", None),
    ("normal_night", 0.70, "nir_night", 0.21, 0.62, 0.20, "normal_night", None),
    ("nir_night", 0.65, "night_clear", 0.25, 0.62, 0.20, "nir_night", None),
    # TC05: day primary + night secondary → prefer night (day is noise)
    ("normal_day", 0.72, "night_clear", 0.21, 0.62, 0.20, "night_clear", None),
    ("normal_day", 0.72, "normal_night", 0.21, 0.62, 0.20, "normal_night", None),
    # TC06: night primary + day secondary → drop day
    ("night_clear", 0.78, "normal_day", 0.21, 0.62, 0.20, "night_clear", None),
    ("normal_night", 0.68, "normal_day", 0.21, 0.62, 0.20, "normal_night", None),
    # TC07: night primary + fog secondary → (night, "fog")
    ("night_clear", 0.78, "fog", 0.22, 0.62, 0.20, "night_clear", "fog"),
    ("normal_night", 0.70, "fog", 0.21, 0.62, 0.20, "normal_night", "fog"),
    ("nir_night", 0.65, "fog", 0.25, 0.62, 0.20, "nir_night", "fog"),
    # TC08: fog primary + night secondary → (fog, night_flavor)
    ("fog", 0.75, "night_clear", 0.22, 0.62, 0.20, "fog", "night_clear"),
    ("fog", 0.75, "normal_night", 0.21, 0.62, 0.20, "fog", "normal_night"),
    # TC09: glare primary + normal_day secondary → hint not in valid set
    ("glare", 0.70, "normal_day", 0.22, 0.62, 0.20, "glare", None),
    # TC09b: rain secondary → in valid hints
    ("night_clear", 0.70, "rain", 0.22, 0.62, 0.20, "night_clear", "rain"),
    # TC09c: glare secondary → in valid hints
    ("fog", 0.70, "glare", 0.22, 0.62, 0.20, "fog", "glare"),
])
def test_compose_env_from_ml_top2(
    class_1: Optional[str],
    proba_1: float,
    class_2: Optional[str],
    proba_2: float,
    pri_thr: float,
    sec_thr: float,
    exp_env: Optional[str],
    exp_hint: Optional[str],
) -> None:
    eff, hint = compose_env_from_ml_top2(
        class_1=class_1,
        proba_1=proba_1,
        class_2=class_2,
        proba_2=proba_2,
        primary_threshold=pri_thr,
        secondary_threshold=sec_thr,
    )
    assert eff == exp_env, f"Expected env={exp_env!r}, got {eff!r}"
    assert hint == exp_hint, f"Expected hint={exp_hint!r}, got {hint!r}"


# ── apply_secondary_hint ──────────────────────────────────────────────────────

def test_apply_secondary_hint_none_returns_copy() -> None:
    """TC10: None hint → new dict with same contents; not the same object."""
    base = {"temporal_prev_weight": 0.46, "display_l_max": 238}
    result = apply_secondary_hint(base, None)
    assert result == base
    assert result is not base, "apply_secondary_hint must return a copy, not the same dict"


def test_apply_secondary_hint_fog_applies_delta() -> None:
    """TC11: hint='fog' → fusion_alpha_boost applied on top of base."""
    base = {"temporal_prev_weight": 0.50, "fusion_alpha_boost": 0.0}
    result = apply_secondary_hint(base, "fog")
    assert result["fusion_alpha_boost"] == 0.05
    assert result["temporal_prev_weight"] == 0.50, "Unrelated key must be unchanged"
    assert base["fusion_alpha_boost"] == 0.0, "Original must not be mutated"


def test_apply_secondary_hint_night_clear_applies_delta() -> None:
    """TC11b: hint='night_clear' (fog primary, night secondary) applies tone tweaks."""
    base = {"temporal_prev_weight": 0.40, "fusion_alpha_boost": 0.12}
    result = apply_secondary_hint(base, "night_clear")
    assert result["temporal_prev_weight"] == pytest.approx(0.48)
    assert result["nir_gamma"] == pytest.approx(0.72)
    assert result["fusion_alpha_boost"] == 0.12, "Unrelated key must be unchanged"


def test_apply_secondary_hint_unknown_hint_no_change() -> None:
    """TC12: unknown hint key → no change to opt_cfg."""
    base = {"temporal_prev_weight": 0.46}
    result = apply_secondary_hint(base, "nonexistent_hint")
    assert result == base
    assert result is not base


# ── TC13: regression — bucket dispatch integrity ──────────────────────────────

def test_fog_primary_routes_to_bucket_d() -> None:
    """TC13: fog primary + borderline night secondary → bucket D (not A)."""
    eff, hint = compose_env_from_ml_top2(
        class_1="fog",
        proba_1=0.75,
        class_2="night_clear",
        proba_2=0.22,
        primary_threshold=0.62,
        secondary_threshold=0.20,
    )
    assert eff == "fog"
    assert OPTICAL_BUCKET_DISPATCH.get(eff) == "D", (
        f"fog must route to bucket D, got {OPTICAL_BUCKET_DISPATCH.get(eff)!r}"
    )
    assert hint == "night_clear"  # light night hint; does not change bucket


def test_night_primary_routes_to_bucket_a() -> None:
    """night_clear → bucket A regardless of secondary hint."""
    eff, hint = compose_env_from_ml_top2(
        class_1="night_clear",
        proba_1=0.80,
        class_2="fog",
        proba_2=0.22,
        primary_threshold=0.62,
        secondary_threshold=0.20,
    )
    assert eff == "night_clear"
    assert OPTICAL_BUCKET_DISPATCH.get(eff) == "A"
    assert hint == "fog"


# ── TC14–TC16 + full rule table (Task 3 new rules) ───────────────────────────

@pytest.mark.parametrize("class_1,class_2,exp_env,exp_hint", [
    # TC14: Rule 5 — nir_night + fog → illumination-primary (nir_night is in _NIGHT_AXIS)
    ("nir_night",    "fog",          "nir_night",    "fog"),
    # TC15: Rule 6 (NEW) — night_clear + rain → illumination-primary
    ("night_clear",  "rain",         "night_clear",  "rain"),
    # TC16: Rule 6 (NEW) — normal_night + rain → illumination-primary
    ("normal_night", "rain",         "normal_night", "rain"),
    # Rule 6 — nir_night + rain
    ("nir_night",    "rain",         "nir_night",    "rain"),
    # Rule 7 (NEW) — day primary + fog secondary → day stays env_class
    ("normal_day",   "fog",          "normal_day",   "fog"),
    # Rule 7 (NEW) — day primary + rain secondary
    ("normal_day",   "rain",         "normal_day",   "rain"),
    # Rule 7 (NEW) — day primary + glare secondary
    ("normal_day",   "glare",        "normal_day",   "glare"),
    # Rule 9 (NEW) — rain primary + night secondary
    ("rain",         "normal_night", "rain",         "normal_night"),
    ("rain",         "nir_night",    "rain",         "nir_night"),
    ("rain",         "night_clear",  "rain",         "night_clear"),
    # Rule 10 fallthrough — glare + night: night NOT in _VALID_SECONDARY_HINTS → hint=None.
    # (night_clear IS in _SECONDARY_HINT_OPT_DELTA but the gate is _VALID_SECONDARY_HINTS.)
    ("glare",        "night_clear",  "glare",        None),
    # Rule 10 fallthrough — fog + day: day not in _VALID_SECONDARY_HINTS → hint=None
    ("fog",          "normal_day",   "fog",          None),
])
def test_compositor_rule_table(class_1: str, class_2: str, exp_env: str, exp_hint) -> None:
    """Full compound rule table — one parametrized row per rule branch."""
    eff, hint = compose_env_from_ml_top2(
        class_1=class_1, proba_1=0.80,
        class_2=class_2, proba_2=0.30,
        primary_threshold=0.62, secondary_threshold=0.20,
    )
    assert eff == exp_env, f"env: expected {exp_env!r}, got {eff!r} (class_1={class_1!r}, class_2={class_2!r})"
    assert hint == exp_hint, f"hint: expected {exp_hint!r}, got {hint!r} (class_1={class_1!r}, class_2={class_2!r})"


def test_night_top1_never_becomes_fog_or_rain_env_class() -> None:
    """Invariant guard: when class_1 is a night flavor, env_class must stay class_1.
    No rule may promote fog/rain/glare to env_class when illumination is top-1.
    """
    for night in ("night_clear", "normal_night", "nir_night"):
        for weather in ("fog", "rain", "glare"):
            eff, _ = compose_env_from_ml_top2(
                class_1=night, proba_1=0.80,
                class_2=weather, proba_2=0.30,
                primary_threshold=0.62, secondary_threshold=0.20,
            )
            assert eff == night, (
                f"Invariant broken: night={night!r}, weather={weather!r} → env_class={eff!r}"
            )


def test_day_top1_never_becomes_interference_env_class() -> None:
    """Symmetric day-invariant: when class_1=normal_day and class_2 is not night,
    env_class stays normal_day (Rule 7 applies, not any interference-primary rule).
    """
    for weather in ("fog", "rain", "glare"):
        eff, _ = compose_env_from_ml_top2(
            class_1="normal_day", proba_1=0.80,
            class_2=weather, proba_2=0.30,
            primary_threshold=0.62, secondary_threshold=0.20,
        )
        assert eff == "normal_day", (
            f"Day-invariant broken: class_2={weather!r} → env_class={eff!r}"
        )
