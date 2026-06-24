"""
Environment presets and hysteresis controller used from ``main``.

Maps ENV modes (manual / auto_rule) and ML hints into merged ``opt_cfg`` slices
that tune thermal, NIR, display, and E1 parameters without editing ``CONFIG`` by hand.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple  # noqa: F401


def _format_env_apply_log(preset_name: str, pobj: Dict[str, Any]) -> str:
    """Single-line summary of applied preset and key overrides (console-friendly)."""
    ov = pobj.get("opt_overrides") or {}
    te = pobj.get("thermal_extra") or {}
    e1 = pobj.get("e1_overrides") or {}
    a3 = pobj.get("thermal_3dnr_alpha")
    parts: List[str] = []
    for k in (
        "nir_high_pct",
        "nir_saturate_at",
        "nir_gamma",
        "temporal_prev_weight",
        "thermal_floor",
        "thermal_fg_threshold",
        "thermal_fg_max_ratio",
        "fusion_alpha_boost",
        "display_l_max",
        "display_l_max_when_glare",
    ):
        if k in ov:
            v = ov[k]
            parts.append(f"{k}={v}" if not isinstance(v, float) else f"{k}={v:.3g}")
    if a3 is not None:
        parts.append(f"thermal_3dnr_alpha={float(a3):.2f}")
    if te.get("thermal_detect_use_anti_glare"):
        parts.append("thermal_detect_use_anti_glare=True")
    if e1:
        parts.append("e1=" + ",".join(f"{kk}={e1[kk]}" for kk in sorted(e1.keys())))
    dg = pobj.get("display_grade") or {}
    if dg:
        parts.append("display_grade=on")
    return f"[ENV] apply preset={preset_name} " + (" ".join(parts) if parts else "(defaults)")


def build_env_presets() -> Dict[str, Any]:
    """Return the canonical preset table (only standardized override keys)."""
    return {
        "default": {
            "label": "default",
            "opt_overrides": {},
            "thermal_extra": {},
            "e1_overrides": {},
            "display_grade": {},
            "thermal_3dnr_alpha": None,
        },
        "night": {
            "label": "night",
            "opt_overrides": {
                "temporal_prev_weight": 0.50,
                "display_l_max": 236,
                "display_l_max_when_glare": 200,
                "nir_gamma": 0.70,
            },
            "thermal_extra": {},
            "e1_overrides": {
                "feature_e1_z_thresh": 1.05,
                "feature_e1_min_area": 8,
            },
            "display_grade": {"brightness": 0.04, "contrast": 1.05, "saturation": 1.0, "warmth": 0.0,
                               "shadows": 0.0, "highlights": 0.0},
            "thermal_3dnr_alpha": 0.72,
        },
        "low_light": {
            "label": "low_light",
            "opt_overrides": {
                "temporal_prev_weight": 0.46,
                "display_l_max": 238,
                "display_l_max_when_glare": 204,
            },
            "thermal_extra": {},
            "e1_overrides": {
                "feature_e1_z_thresh": 1.1,
                "feature_e1_min_area": 8,
            },
            "display_grade": {"brightness": 0.05, "contrast": 1.06, "saturation": 1.0, "warmth": 0.0,
                               "shadows": 0.0, "highlights": 0.0},
            "thermal_3dnr_alpha": 0.68,
        },
        "glare_heavy": {
            "label": "glare_heavy",
            "opt_overrides": {
                "display_l_max": 232,
                "display_l_max_when_glare": 194,
                "temporal_prev_weight": 0.48,
                "nir_high_pct": 93.0,
                "nir_saturate_at": 228.0,
                "thermal_fg_max_ratio": 0.72,
            },
            "thermal_extra": {"thermal_detect_use_anti_glare": True},
            "e1_overrides": {
                "feature_e1_z_thresh": 1.55,
            },
            "display_grade": {"brightness": -0.03, "contrast": 0.94, "saturation": 0.92, "warmth": 0.0,
                               "shadows": 0.0, "highlights": -0.05},
            "thermal_3dnr_alpha": None,
        },
        "backlight": {
            "label": "backlight",
            "opt_overrides": {
                # Offline sweep: high_pct=90, sat_at=220 suppresses backlight highlights
                # with minimal contrast loss (pct_sat_after=0.0, Δlog_rms=+0.0472).
                "nir_high_pct": 90.0,
                "nir_saturate_at": 220.0,
                "display_l_max": 234,
                "display_l_max_when_glare": 196,
                "thermal_fg_max_ratio": 0.74,
            },
            "thermal_extra": {},
            "e1_overrides": {"feature_e1_z_thresh": 1.45},
            "display_grade": {"brightness": 0.02, "contrast": 1.08, "saturation": 0.95, "warmth": 0.02,
                               "shadows": 0.02, "highlights": -0.03},
            "thermal_3dnr_alpha": None,
        },
        "fog": {
            "label": "fog",
            "opt_overrides": {
                "fusion_alpha_boost": 0.12,
                "thermal_floor": 4.0,
                "thermal_fg_threshold": 20.0,
                "thermal_fg_max_ratio": 0.72,
            },
            "thermal_extra": {},
            "e1_overrides": {
                "feature_e1_z_thresh": 1.08,
                "feature_e1_min_area": 10,
            },
            "display_grade": {"brightness": 0.0, "contrast": 1.12, "saturation": 0.9, "warmth": 0.0,
                               "shadows": 0.0, "highlights": 0.0},
            "thermal_3dnr_alpha": 0.58,
        },
        "haze": {
            "label": "haze",
            "opt_overrides": {
                "fusion_alpha_boost": 0.06,
                "thermal_floor": 3.5,
                "thermal_fg_threshold": 19.0,
                "thermal_fg_max_ratio": 0.72,
            },
            "thermal_extra": {},
            "e1_overrides": {"feature_e1_z_thresh": 1.15},
            "display_grade": {"brightness": 0.0, "contrast": 1.1, "saturation": 0.93, "warmth": 0.0,
                               "shadows": 0.0, "highlights": 0.0},
            "thermal_3dnr_alpha": 0.60,
        },
        "high_contrast": {
            "label": "high_contrast",
            "opt_overrides": {
                "display_l_max": 238,
                "display_l_max_when_glare": 206,
                "thermal_fg_max_ratio": 0.76,
            },
            "thermal_extra": {},
            "e1_overrides": {"feature_e1_z_thresh": 1.35},
            "display_grade": {"brightness": 0.0, "contrast": 1.0, "saturation": 1.05, "warmth": 0.0,
                               "shadows": 0.0, "highlights": 0.0},
            "thermal_3dnr_alpha": None,
        },
        "cluttered_bg": {
            "label": "cluttered_bg",
            "opt_overrides": {
                "thermal_fg_max_ratio": 0.62,
            },
            "thermal_extra": {},
            "e1_overrides": {
                "feature_e1_z_thresh": 1.65,
                "feature_e1_min_area": 16,
            },
            "display_grade": {},
            "thermal_3dnr_alpha": None,
        },
        "clear_day": {
            "label": "clear_day",
            "opt_overrides": {
                "temporal_prev_weight": 0.34,
                "display_l_max": 242,
                "display_l_max_when_glare": 210,
                "nir_high_pct": 96.0,
                "nir_saturate_at": 236.0,
                "thermal_fg_max_ratio": 0.88,
            },
            "thermal_extra": {},
            "e1_overrides": {},
            "display_grade": {"brightness": 0.0, "contrast": 1.0, "saturation": 1.08, "warmth": 0.0,
                               "shadows": 0.0, "highlights": 0.0},
            "thermal_3dnr_alpha": 0.62,
        },
        "night_fog": {
            "label": "night_fog",
            "opt_overrides": {
                "temporal_prev_weight": 0.48,
                "display_l_max": 234,
                "display_l_max_when_glare": 198,
                "fusion_alpha_boost": 0.10,
                "thermal_floor": 4.0,
                "thermal_fg_threshold": 19.0,
                "nir_gamma": 0.70,
                "thermal_fg_max_ratio": 0.70,
            },
            "thermal_extra": {},
            "e1_overrides": {
                "feature_e1_z_thresh": 1.0,
                "feature_e1_min_area": 8,
            },
            "display_grade": {"brightness": 0.04, "contrast": 1.1, "saturation": 0.95, "warmth": 0.0,
                               "shadows": 0.0, "highlights": 0.0},
            "thermal_3dnr_alpha": 0.65,
        },
        "day_glare": {
            "label": "day_glare",
            "opt_overrides": {
                "display_l_max": 230,
                "display_l_max_when_glare": 192,
                "temporal_prev_weight": 0.46,
                "nir_high_pct": 92.0,
                "nir_saturate_at": 227.0,
                "thermal_fg_max_ratio": 0.72,
            },
            "thermal_extra": {},
            "e1_overrides": {"feature_e1_z_thresh": 1.5},
            "display_grade": {"brightness": -0.04, "contrast": 0.93, "saturation": 0.9, "warmth": 0.0,
                               "shadows": 0.0, "highlights": 0.0},
            "thermal_3dnr_alpha": None,
        },
        "low_light_cluttered": {
            "label": "low_light_cluttered",
            "opt_overrides": {
                "temporal_prev_weight": 0.44,
                "display_l_max": 236,
                "thermal_fg_max_ratio": 0.62,
            },
            "thermal_extra": {},
            "e1_overrides": {
                "feature_e1_z_thresh": 1.45,
                "feature_e1_min_area": 14,
            },
            "display_grade": {"brightness": 0.04, "contrast": 1.05, "saturation": 1.0, "warmth": 0.0,
                               "shadows": 0.0, "highlights": 0.0},
            "thermal_3dnr_alpha": 0.66,
        },
        "backlight_high_contrast": {
            "label": "backlight_high_contrast",
            "opt_overrides": {
                "nir_high_pct": 91.0,
                "nir_saturate_at": 225.0,
                "display_l_max": 232,
                "display_l_max_when_glare": 190,
                "thermal_fg_max_ratio": 0.74,
            },
            "thermal_extra": {},
            "e1_overrides": {"feature_e1_z_thresh": 1.5},
            "display_grade": {"brightness": 0.0, "contrast": 1.06, "saturation": 0.94, "warmth": 0.02,
                               "shadows": 0.0, "highlights": 0.0},
            "thermal_3dnr_alpha": None,
        },
        # ── ENV_CLASS-aligned presets (v4 bucket dispatch) ────────────────────
        # These mirror the 9 ENV_CLASS names so env_controller can accept them directly.
        "night_clear": {
            "label": "night_clear",
            "opt_overrides": {
                "temporal_prev_weight": 0.50,
                "display_l_max": 236,
                "display_l_max_when_glare": 200,
                "nir_gamma": 0.70,
                # Offline sweep: detail_strength=0.35 with clip=0.5 and 320×240 proc size maximises
                # night-clear contrast (log_rms=0.9200) while keeping saturation within the 0.05 ceiling.
                "nir_enhancer_detail_strength": 0.35,
            },
            "thermal_extra": {},
            "e1_overrides": {"feature_e1_z_thresh": 1.05, "feature_e1_min_area": 8},
            "display_grade": {"brightness": 0.04, "contrast": 1.05, "saturation": 1.0,
                               "warmth": 0.0, "shadows": 0.0, "highlights": 0.0},
            "thermal_3dnr_alpha": 0.72,
        },
        "normal_night": {
            "label": "normal_night",
            "opt_overrides": {
                "temporal_prev_weight": 0.46,
                "display_l_max": 238,
                "display_l_max_when_glare": 204,
                "nir_gamma": 0.70,
            },
            "thermal_extra": {},
            "e1_overrides": {"feature_e1_z_thresh": 1.1, "feature_e1_min_area": 8},
            "display_grade": {"brightness": 0.05, "contrast": 1.06, "saturation": 1.0,
                               "warmth": 0.0, "shadows": 0.0, "highlights": 0.0},
            "thermal_3dnr_alpha": 0.68,
        },
        "normal_day": {
            "label": "normal_day",
            "opt_overrides": {
                "temporal_prev_weight": 0.34,
                "display_l_max": 242,
                "display_l_max_when_glare": 210,
                "nir_high_pct": 96.0,
                "nir_saturate_at": 236.0,
                "thermal_fg_max_ratio": 0.88,
            },
            "thermal_extra": {},
            "e1_overrides": {},
            "display_grade": {"brightness": 0.0, "contrast": 1.0, "saturation": 1.08,
                               "warmth": 0.0, "shadows": 0.0, "highlights": 0.0},
            "thermal_3dnr_alpha": 0.62,
        },
        "glare": {
            "label": "glare",
            "opt_overrides": {
                "display_l_max": 232,
                "display_l_max_when_glare": 194,
                "temporal_prev_weight": 0.48,
                # Offline sweep: high_pct=90, sat_at=220 gives best glare suppression
                # with near-zero residual saturation (pct_sat=0.0011, Δlog_rms=+0.0493).
                "nir_high_pct": 90.0,
                "nir_saturate_at": 220.0,
                "thermal_fg_max_ratio": 0.72,
            },
            "thermal_extra": {"thermal_detect_use_anti_glare": True},
            "e1_overrides": {"feature_e1_z_thresh": 1.55},
            "display_grade": {"brightness": -0.03, "contrast": 0.94, "saturation": 0.92,
                               "warmth": 0.0, "shadows": 0.0, "highlights": -0.05},
            "thermal_3dnr_alpha": None,
        },
        "nir_night": {
            "label": "nir_night",
            "opt_overrides": {
                "temporal_prev_weight": 0.44,
                "display_l_max": 238,
                "display_l_max_when_glare": 204,
                # Offline sweep: clip_scale=0.16 (effective clip=max(0.5, 3×0.16)=0.5) is the
                # night-optimal minimum — lower clip suppresses tile artefacts without contrast loss.
                "nir_enhancer_clahe_clip_scale": 0.16,
            },
            "thermal_extra": {},
            "e1_overrides": {"feature_e1_z_thresh": 1.15, "feature_e1_min_area": 8},
            "display_grade": {"brightness": 0.03, "contrast": 1.04, "saturation": 1.0,
                               "warmth": 0.0, "shadows": 0.0, "highlights": 0.0},
            "thermal_3dnr_alpha": 0.70,
        },
        "rain": {
            "label": "rain",
            "opt_overrides": {
                "temporal_prev_weight": 0.50,
                "display_l_max": 236,
                "display_l_max_when_glare": 202,
                "rain_median_frames": 3,
            },
            "thermal_extra": {},
            "e1_overrides": {"feature_e1_z_thresh": 1.35, "feature_e1_min_area": 12},
            "display_grade": {"brightness": 0.02, "contrast": 1.06, "saturation": 0.95,
                               "warmth": 0.0, "shadows": 0.0, "highlights": 0.0},
            "thermal_3dnr_alpha": 0.65,
        },
        "transition": {
            "label": "transition",
            "opt_overrides": {
                "temporal_prev_weight": 0.40,
                "display_l_max": 240,
                "display_l_max_when_glare": 206,
            },
            "thermal_extra": {},
            "e1_overrides": {},
            "display_grade": {},
            "thermal_3dnr_alpha": None,
        },
    }


ENV_PRESETS: Dict[str, Any] = build_env_presets()

# [ENV] Asymmetric hysteresis overrides per-preset: (onset_frames, decay_frames).
# onset = frames needed to switch TO this preset; decay = frames needed to leave it.
_TRANSITION_HYSTERESIS: Dict[str, Tuple[int, int]] = {
    "glare":       (2, 20),
    "glare_heavy": (2, 20),
    # Rain: slow onset (10) to debounce brief drops; slow decay (25) to keep smoothing.
    "rain":        (10, 25),
    # Transition: slowest onset (12) + very slow decay (30); transition is a catch-all.
    "transition":  (12, 30),
    # Fog: moderate onset (6), moderate decay (18); fog usually develops gradually.
    "fog":         (6, 18),
}

# [ENV] Single-tag priority when no compound tag pair matches
_ENV_SINGLE_PRIORITY: Tuple[str, ...] = (
    "glare_heavy",
    "backlight_high_contrast",
    "backlight",
    "fog",
    "haze",
    "night",
    "low_light",
    "cluttered_bg",
    "high_contrast",
    "clear_day",
)


def select_env_preset_from_tags(tags: Set[str]) -> str:
    """Map heuristic tags to a preset name (compound pairs first, then single priority).

    Used by the auto_rule rule-based fallback path only. When ML is active
    and top-1 confidence passes ml_confidence_threshold, compound decisions
    are handled by compose_env_from_ml_top2 instead.
    """
    if not tags:
        return "default"
    if "night" in tags and "fog" in tags:
        return "night_fog"
    if "clear_day" in tags and "glare_heavy" in tags:
        return "day_glare"
    if "low_light" in tags and "cluttered_bg" in tags:
        return "low_light_cluttered"
    if "backlight" in tags and "high_contrast" in tags:
        return "backlight_high_contrast"
    for key in _ENV_SINGLE_PRIORITY:
        if key in tags:
            return key
    return "default"


_PRESET_TO_ENV_CLASS: Dict[str, str] = {
    "night":                    "normal_night",
    "low_light":                "nir_night",
    "glare_heavy":              "glare",
    "backlight":                "backlight",
    "backlight_high_contrast":  "backlight",
    "fog":                      "fog",
    "haze":                     "fog",
    "night_fog":                "fog",
    "clear_day":                "normal_day",
    "day_glare":                "glare",
    "high_contrast":            "normal_day",
    "cluttered_bg":             "normal_night",
    "low_light_cluttered":      "nir_night",
    "default":                  "normal_night",
}


def auto_rule_preset_to_env_class(preset_name: str) -> str:
    """Map old auto_rule preset name → ENV_CLASS string for bucket dispatch.

    Used by auto_rule rule-based fallback only; ML-driven compound decisions
    go through compose_env_from_ml_top2.
    """
    return _PRESET_TO_ENV_CLASS.get(preset_name, "normal_night")


# ── ML top-2 compositor ──────────────────────────────────────────────────────

# Night-axis classes: collapse any pair to one label (no "two nights").
_NIGHT_AXIS: frozenset = frozenset({"night_clear", "normal_night", "nir_night"})
# Day-axis classes: treated as noise when paired with a night-axis class.
_DAY_AXIS: frozenset = frozenset({"normal_day"})
# Secondary hints we actually act on; all other class_2 values are dropped.
_VALID_SECONDARY_HINTS: frozenset = frozenset({"fog", "rain", "glare"})

# Light opt_cfg tweaks applied on top of the primary preset when a secondary
# hint is present. Primary preset owns bucket selection; keep deltas small.
_SECONDARY_HINT_OPT_DELTA: Dict[str, Dict[str, Any]] = {
    # Primary = night; secondary = "also fog" → minor dehaze / fusion nudge
    "fog":          {"fusion_alpha_boost": 0.05},
    # Primary = fog; secondary = "also night" → minor tone / temporal tweak
    "night_clear":  {"temporal_prev_weight": 0.48, "nir_gamma": 0.72},
    "normal_night": {"temporal_prev_weight": 0.46, "nir_gamma": 0.72},
    "nir_night":    {"temporal_prev_weight": 0.46},
    # Light glare suppression overlay (when primary is not already glare)
    "glare":        {"nir_high_pct": 93.0, "nir_saturate_at": 228.0},
    # Rain smoothing nudge (when primary is not rain)
    "rain":         {"temporal_prev_weight": 0.50},
}


def compose_env_from_ml_top2(
    *,
    class_1: Optional[str],
    proba_1: float,
    class_2: Optional[str],
    proba_2: float,
    primary_threshold: float,
    secondary_threshold: float,
) -> Tuple[Optional[str], Optional[str]]:
    """Map ML top-1/top-2 predictions to (effective_env_class, secondary_hint).

    Returns (None, None) when ML must not drive env (caller uses rule-based
    fallback). Otherwise returns the ENV_CLASS name for OPTICAL_BUCKET_DISPATCH
    and an optional secondary hint for light opt_cfg tweaks via apply_secondary_hint.

    Invariant: when class_1 ∈ (_NIGHT_AXIS ∪ _DAY_AXIS), env_class == class_1.
    Interference (fog/rain/glare) replaces env_class ONLY when it is top-1 (Rules 8/9).

    Rules (in priority order):
      1. top-1 below primary_threshold → (None, None)
      2. top-2 absent / == class_1 / below secondary_threshold → (class_1, None)
      3. Redundant night-axis pair → (class_1, None)
      4a. Night primary + day secondary → (class_1, None); day treated as noise
      4b. Day primary + night secondary → (class_2, None); day treated as noise
      5. Night primary + fog secondary → (class_1, "fog")   [illumination-primary + hint]
      6. Night primary + rain secondary → (class_1, "rain") [illumination-primary + hint]
      7. Day primary + {fog|rain|glare} secondary → (class_1, class_2)  [day-primary + hint]
      8. Fog primary + night secondary → (class_1, class_2) [interference-primary; existing]
      9. Rain primary + night secondary → (class_1, class_2) [interference-primary; symmetric]
      10. Any other pair → (class_1, class_2 if in _VALID_SECONDARY_HINTS else None)
    """
    if not class_1 or proba_1 < primary_threshold:
        return None, None

    if (
        class_2 is None
        or class_2 == class_1
        or proba_2 < secondary_threshold
    ):
        return class_1, None

    # Rule 3: redundant night-axis pair → collapse to top-1
    if class_1 in _NIGHT_AXIS and class_2 in _NIGHT_AXIS:
        return class_1, None

    # Rule 4a: night primary + day secondary → drop day as noise
    if class_1 in _NIGHT_AXIS and class_2 in _DAY_AXIS:
        return class_1, None

    # Rule 4b: day primary + night secondary → prefer night (day treated as noise)
    if class_1 in _DAY_AXIS and class_2 in _NIGHT_AXIS:
        return class_2, None

    # Rule 5: night primary + fog secondary → full night path, light fog hint
    if class_1 in _NIGHT_AXIS and class_2 == "fog":
        return class_1, "fog"

    # Rule 6: night primary + rain secondary → full night path, light rain hint
    if class_1 in _NIGHT_AXIS and class_2 == "rain":
        return class_1, "rain"

    # Rule 7: day primary + weather secondary → day stays env_class, weather as hint
    if class_1 in _DAY_AXIS and class_2 in _VALID_SECONDARY_HINTS:
        return class_1, class_2

    # Rule 8: fog primary + night secondary → full fog path, light night hint
    if class_1 == "fog" and class_2 in _NIGHT_AXIS:
        return class_1, class_2

    # Rule 9: rain primary + night secondary → full rain path, light night hint
    # night flavors ARE in _SECONDARY_HINT_OPT_DELTA → apply_secondary_hint applies nudge.
    if class_1 == "rain" and class_2 in _NIGHT_AXIS:
        return class_1, class_2

    # Rule 10: other pairs — keep secondary only if in the curated hint set
    hint = class_2 if class_2 in _VALID_SECONDARY_HINTS else None
    return class_1, hint


def apply_secondary_hint(
    opt_cfg: Dict[str, Any], hint: Optional[str]
) -> Dict[str, Any]:
    """Overlay minor opt_cfg tweaks for a secondary ML hint.

    Only keys present in _VALID_OPT_OVERRIDES_KEYS are applied.
    Returns a new dict; never mutates opt_cfg.
    """
    from smartbinocular.config import _VALID_OPT_OVERRIDES_KEYS

    if not hint:
        return dict(opt_cfg)
    delta = _SECONDARY_HINT_OPT_DELTA.get(hint) or {}
    merged = dict(opt_cfg)
    merged.update({k: v for k, v in delta.items() if k in _VALID_OPT_OVERRIDES_KEYS})
    return merged


def infer_env_tags_auto_rule(
    *,
    nir_b_ema: Optional[float],
    nir_gray_std: Optional[float],
    glare_nir: bool,
    haze_config_on: bool,
    std_low: float,
    std_high: float,
) -> Set[str]:
    """Lightweight rule tags without ML; ``nir_gray_std`` comes from downscaled gray.

    ``night`` / ``low_light`` derive from ``nir_b_ema`` only (not a separate night flag).

    Very low ``nir_gray_std`` on dark-night thumbnails (tiny spatial variance) does not add
    ``haze`` when ``nir_b_ema`` is already in the night band (<25); see branch below.
    """
    tags: Set[str] = set()
    if haze_config_on:
        tags.add("fog")
    if nir_b_ema is not None:
        if nir_b_ema < 25.0:
            tags.add("night")
        elif nir_b_ema < 38.0:
            tags.add("low_light")
        if nir_b_ema >= 95.0:
            tags.add("clear_day")
    if glare_nir:
        tags.add("glare_heavy")
    if nir_gray_std is not None:
        if nir_gray_std < std_low:
            # Downscaled NIR (FrameCache nir_gray, max_side≈128) has tiny spatial std on
            # uniform dark scenes — not meteorological haze. Only tag haze when brightness
            # is above the same "night" band (<25) so pitch-black nights stay normal_night.
            if nir_b_ema is None or nir_b_ema >= 25.0:
                tags.add("haze")
        elif nir_gray_std > std_high:
            tags.add("cluttered_bg")
        if 35.0 <= nir_gray_std <= std_high and nir_b_ema is not None and 40.0 <= nir_b_ema <= 90.0:
            tags.add("high_contrast")
    # Heuristic backlight: glare flag with mid-range mean brightness
    if glare_nir and nir_b_ema is not None and 45.0 <= nir_b_ema < 85.0:
        tags.add("backlight")
    return tags


class EnvPresetController:
    """Debounce desired preset changes so HUD and tuning do not flicker frame-to-frame."""

    def __init__(
        self,
        *,
        fallback: str,
        hysteresis_frames: int,
    ):
        """``fallback`` is used when an unknown preset is requested; ``hysteresis_frames`` is the default streak length."""
        self.fallback = str(fallback) if fallback else "default"
        self.hysteresis_frames = int(max(1, hysteresis_frames))
        self.stable_name: str = self.fallback
        self._candidate: Optional[str] = None
        self._streak: int = 0

    def reset(self) -> None:
        """Clear candidate streak and return ``stable_name`` to fallback."""
        self.stable_name = self.fallback
        self._candidate = None
        self._streak = 0

    def _hysteresis_threshold(self, desired: str) -> int:
        """Return per-transition frame threshold (onset or decay)."""
        if desired in _TRANSITION_HYSTERESIS:
            return _TRANSITION_HYSTERESIS[desired][0]
        if self.stable_name in _TRANSITION_HYSTERESIS:
            return _TRANSITION_HYSTERESIS[self.stable_name][1]
        return self.hysteresis_frames

    def update(self, desired: str) -> str:
        """Ingest ``desired`` preset; return the debounced ``stable_name``."""
        d = desired if desired in ENV_PRESETS else self.fallback
        if d == self.stable_name:
            self._candidate = None
            self._streak = 0
            return self.stable_name
        if self._candidate != d:
            self._candidate = d
            self._streak = 1
        else:
            self._streak += 1
        if self._streak >= self._hysteresis_threshold(d):
            self.stable_name = d
            self._candidate = None
            self._streak = 0
        return self.stable_name


def merge_opt_cfg_with_preset(opt_cfg_base: Dict[str, Any], preset_name: str) -> Dict[str, Any]:
    """Merge profile+haze ``opt_cfg_base`` with a preset's ``opt_overrides``.

    Validates keys against ``_VALID_OPT_OVERRIDES_KEYS``; unknown keys print a warning.
    """
    from smartbinocular.config import _VALID_OPT_OVERRIDES_KEYS
    merged = dict(opt_cfg_base)
    p = ENV_PRESETS.get(preset_name) or ENV_PRESETS["default"]
    overrides = p.get("opt_overrides", {})
    invalid = set(overrides) - _VALID_OPT_OVERRIDES_KEYS
    if invalid:
        print(f"[ENV] WARN: unknown opt_overrides keys in '{preset_name}': {invalid}")
    merged.update({k: v for k, v in overrides.items() if k in _VALID_OPT_OVERRIDES_KEYS})
    return merged


def apply_e1_overrides(det: Any, overrides: Dict[str, Any]) -> None:
    if not overrides:
        return
    if "feature_e1_z_thresh" in overrides:
        det.z_thresh = float(overrides["feature_e1_z_thresh"])
    if "feature_e1_heat_thresh" in overrides:
        det.heat_thresh = float(overrides["feature_e1_heat_thresh"])
    if "feature_e1_min_area" in overrides:
        det.min_area = int(overrides["feature_e1_min_area"])
    if "feature_e1_raw_mix" in overrides:
        import numpy as np
        det.raw_mix = float(np.clip(float(overrides["feature_e1_raw_mix"]), 0.0, 1.0))
    if "feature_e1_local_kernel" in overrides:
        k = int(max(3, int(overrides["feature_e1_local_kernel"])))
        if k % 2 == 0:
            k += 1
        det.local_kernel = k


def snapshot_e1_defaults(det: Any) -> Dict[str, Any]:
    return {
        "feature_e1_z_thresh": float(det.z_thresh),
        "feature_e1_heat_thresh": float(det.heat_thresh),
        "feature_e1_min_area": int(det.min_area),
        "feature_e1_raw_mix": float(det.raw_mix),
        "feature_e1_local_kernel": int(det.local_kernel),
    }


def restore_e1_from_snapshot(det: Any, snap: Dict[str, Any]) -> None:
    apply_e1_overrides(det, snap)
