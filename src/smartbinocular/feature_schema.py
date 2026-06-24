"""Feature sets, :class:`FeatureRecord`, and ENV class taxonomy for ML and logging.

Consumed by ``feature_extractor``, ``ml_inference.EnvClassifier``, and ``main`` when
building JSONL training or inference records. Feature names must stay in sync with
offline training code.

CONSTRAINTS enforced here (from legacy/md/OFFLINE_ML_PLAN.md):
  C7  — No temporal feature simulation from still images.
        Temporal features are None unless has_temporal=True is asserted by the caller.
  C8  — No zero-imputation for missing optional features.
        All optional groups default to None; to_feature_array() raises if any required
        feature for the chosen feature_set is None.
  C10 — No cross-nir_channel normalization.
        Normalization must be done per nir_channel group — enforced in train_classifier.py.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Optional

import numpy as np


# ── ENV class taxonomy ────────────────────────────────────────────────────────

ENV_CLASSES: List[str] = [
    "night_clear",   # Night, open sky, no ambient light (optical night-vision target env)
    "normal_night",  # Night with ambient light (urban, street-lit)
    "normal_day",    # Daytime, normal conditions
    "fog",           # Fog or smog; visibility degraded
    "rain",          # Rain; wet surfaces, drops on lens
    "glare",         # Direct light source (headlights, direct sun) — field-collect only
    "backlight",     # Subject against bright background — field-collect only
    "transition",    # Dawn or dusk
    "nir_night",     # Dark scene, IMX290 NIR-dominant mono (active NIR / extended spectral response)
]

ENV_CLASS_TO_INT: Dict[str, int] = {cls: idx + 1 for idx, cls in enumerate(ENV_CLASSES)}
ENV_INT_TO_CLASS: Dict[int, str] = {v: k for k, v in ENV_CLASS_TO_INT.items()}
# 0 = unknown / uninitialized
ENV_INT_UNKNOWN: int = 0


# ── Atomic feature sets ───────────────────────────────────────────────────────
# These are the canonical names used in FeatureRecord and to_feature_array().

FEATURE_SET_CORE: List[str] = [
    # Optical image statistics — computed from NIR or RGB; always available.
    "nir_mean_brightness",   # 1  mean of gray small image [0,255]
    "nir_std",               # 2  std of gray small image
    "nir_entropy",           # 3  Shannon entropy of gray histogram (bits)
    "nir_p95",               # 4  95th percentile of gray values
    "nir_glare_score",       # 5  fraction of pixels > 240 (saturation proxy)
    "nir_sharpness",         # 6  Laplacian variance (focus proxy)
    "nir_dark_fraction",     # 7  fraction of pixels < 30 (darkness proxy)
    "nir_saturation_mean",   # 8  mean HSV S-channel on small BGR [0,255]
    # Temporal context
    "hour_of_day_sin",       # 9  sin(2π·hour/24) — cyclic time encoding
    "hour_of_day_cos",       # 10 cos(2π·hour/24)
    "prev_env_class",        # 11 int 0=unknown; ENV_CLASS_TO_INT[cls] otherwise
    # Blue-channel mean EMA (DISTINCT from main.py's nir_b_ema brightness EMA)
    "nir_blue_mean_ema",     # 12 EMA of B-channel mean of nir_small_bgr — nir_night discriminator
]  # 12 features — mandatory; FeatureRecord raises if any is None

FEATURE_SET_THERMAL: List[str] = [
    # Thermal enhancement — only when thermal_channel = "lwir" (MI48 or KAIST).
    "thm_mean",              # 8  mean raw thermal value
    "thm_std",               # 9  std of thermal values
    "thm_max",               # 10 max value (hot-spot indicator)
    "thm_p95_p05_delta",     # 11 p95 − p05 (dynamic range proxy)
    "thm_fg_fraction",       # 12 fraction of pixels above mean+1.5·std
    "thm_anomaly_score",     # 13 (max−median)/(std+ε) — hot-spot z-score
]  # 6 features — None when thermal_channel = "none"

FEATURE_SET_MOTION: List[str] = [
    # Motion from optical flow — only when has_motion=True (video sequence).
    "motion_magnitude",      # 14 mean LK flow magnitude (px/frame)
    "motion_jerk",           # 15 change in magnitude vs previous frame
]  # 2 features — None for still images (C7)

FEATURE_SET_TEMPORAL: List[str] = [
    # Temporal deltas — only when has_temporal=True and ≥10 frame window.
    "nir_brightness_delta_10f",  # 18 nir_mean_brightness[now] − mean(window[-10:])
    "thm_mean_delta_10f",        # 19 thm_mean[now] − mean(thm_window[-10:])
    #    thm_mean_delta_10f = None when thermal is also None
]  # 2 features — None for stills or short sequences (C7)

FEATURE_SET_RUNTIME_ONLY: List[str] = [
    # RPi hardware state — produced only during live capture.
    # NEVER include in training; these are None in all offline records.
    "skew_ms",      # 16 stream skew between NIR and thermal (ms) — S6 inactive; always None in production
    "fusion_alpha", # 17 current fusion blend alpha
]  # 2 features — always None offline; EXCLUDE from all TRAINING_MODES


# ── Composite feature sets (used in TRAINING_MODES) ───────────────────────────

FEATURE_SET_OPTICAL_ONLY: List[str] = list(FEATURE_SET_CORE)
# 12 features. Source: any RGB/NIR image, with or without thermal.
# → PRIMARY PRODUCTION model. Works on ALL hardware configs.

FEATURE_SET_OPTICAL_9: List[str] = [
    f for f in FEATURE_SET_CORE
    if f not in {"hour_of_day_sin", "hour_of_day_cos", "prev_env_class"}
]
# 9 features. Drops the three zero-importance temporal/context features from FEATURE_SET_CORE.
# → ABLATION only: compare against FEATURE_SET_OPTICAL_ONLY to document feature selection.
#   DO NOT deploy: 9-feature bundles fail EnvClassifier.feature_set validation.

FEATURE_SET_OPTICAL_THERMAL: List[str] = list(FEATURE_SET_CORE + FEATURE_SET_THERMAL)
# 17 features. Source: NIR+LWIR (RPi field sessions) or rgb+lwir (KAIST ablation).
# → ENHANCEMENT: only when MI48 available; must gracefully degrade to optical_only.

FEATURE_SET_OPTICAL_MOTION: List[str] = list(
    FEATURE_SET_CORE + FEATURE_SET_MOTION + FEATURE_SET_TEMPORAL
)
# 15 features. Source: video sequences only.

FEATURE_SET_FULL_OFFLINE: List[str] = list(
    FEATURE_SET_CORE + FEATURE_SET_THERMAL + FEATURE_SET_MOTION + FEATURE_SET_TEMPORAL
)
# 21 features. Source: thermal video sequences (KAIST) or RPi field sessions.

FEATURE_SET_RUNTIME_FULL: List[str] = list(FEATURE_SET_FULL_OFFLINE + FEATURE_SET_RUNTIME_ONLY)
# 23 features. RPi live inference only — includes skew_ms and fusion_alpha.

# All known feature names for validation
_ALL_FEATURE_NAMES: frozenset = frozenset(FEATURE_SET_RUNTIME_FULL)


# ── FeatureRecord dataclass ────────────────────────────────────────────────────

@dataclasses.dataclass
class FeatureRecord:
    """One frame's feature vector plus metadata and label.

    CORE fields are mandatory (never None). Optional groups default to None and
    are excluded from training when not available. Zero-imputation is forbidden
    (Constraint C8): to_feature_array() raises ValueError if a required feature
    is None rather than silently substituting 0.
    """

    # ── CORE OPTICAL — mandatory, never None ─────────────────────────────────
    nir_mean_brightness: float
    nir_std: float
    nir_entropy: float
    nir_p95: float
    nir_glare_score: float
    nir_sharpness: float
    nir_dark_fraction: float
    nir_saturation_mean: float
    hour_of_day_sin: float
    hour_of_day_cos: float
    prev_env_class: int                       # 0 = unknown
    # EMA of B-channel mean of nir_small_bgr; DISTINCT from main.py's nir_b_ema brightness EMA
    nir_blue_mean_ema: float = 0.0            # seeded to 0.0 on first frame; never None

    # ── THERMAL OPTIONAL — None when thermal_channel = "none" ────────────────
    thm_mean: Optional[float] = None
    thm_std: Optional[float] = None
    thm_max: Optional[float] = None
    thm_p95_p05_delta: Optional[float] = None
    thm_fg_fraction: Optional[float] = None
    thm_anomaly_score: Optional[float] = None

    # ── MOTION — None for still images (C7: no simulation) ───────────────────
    motion_magnitude: Optional[float] = None
    motion_jerk: Optional[float] = None

    # ── TEMPORAL — None for stills or short sequences (C7) ───────────────────
    nir_brightness_delta_10f: Optional[float] = None
    thm_mean_delta_10f: Optional[float] = None

    # ── RUNTIME ONLY — None when offline; never train ────────────────────────
    skew_ms: Optional[float] = None
    fusion_alpha: Optional[float] = None

    # ── METADATA ──────────────────────────────────────────────────────────────
    ts: float = 0.0
    frame_idx: int = 0
    source: str = "rpi"                       # e.g. "rpi", "offline_image2weather"
    nir_channel: str = "nir"                  # "nir" | "rgb"
    thermal_channel: str = "none"             # "lwir" | "none"
    thermal_available: bool = False
    motion_available: bool = False
    temporal_available: bool = False

    # ── LABELS ────────────────────────────────────────────────────────────────
    label: Optional[str] = None
    # "dataset_original" | "manual" | "weak_heuristic" | None
    label_source: Optional[str] = None
    # Heuristic label — only when dataset lacks original ENV labels
    weak_label: Optional[str] = None
    label_confidence: Optional[float] = None

    # ── Derived helpers ───────────────────────────────────────────────────────

    def effective_label(self) -> Optional[str]:
        """Ground truth label if available; fallback to weak_label.

        Datasets with original labels (Image2Weather, Weather-Time) always have
        self.label set, so weak_label is never used for them (C9 from plan).
        """
        return self.label if self.label is not None else self.weak_label

    def is_compatible_with(self, feature_set: List[str]) -> bool:
        """Return True if every feature in feature_set is non-None on this record."""
        return all(getattr(self, name, None) is not None for name in feature_set)

    def to_feature_array(self, feature_set: List[str]) -> np.ndarray:
        """Build numpy array for the given feature_set.

        Raises ValueError if any feature is None — no zero-imputation (C8).
        Callers should use is_compatible_with() to pre-filter records.
        """
        values: List[float] = []
        missing: List[str] = []
        for name in feature_set:
            val = getattr(self, name, None)
            if val is None:
                missing.append(name)
            else:
                values.append(float(val))
        if missing:
            raise ValueError(
                f"FeatureRecord missing required features for feature_set: {missing}. "
                "Use is_compatible_with() to filter records before calling to_feature_array()."
            )
        return np.array(values, dtype=np.float32)

    def to_dict(self) -> dict:
        """Serialize to plain dict for JSONL writing."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureRecord":
        """Deserialize from plain dict (JSONL record). Unknown keys are ignored."""
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


# ── Utility ────────────────────────────────────────────────────────────────────

def encode_hour(ts: float) -> tuple[float, float]:
    """Return (sin, cos) cyclic encoding of hour-of-day from a Unix timestamp.

    Uses local time. If ts=0 (offline/unknown), returns (0.0, 1.0) which
    encodes hour=0 — acceptable for datasets with unknown capture time.
    """
    import datetime
    if ts <= 0:
        return 0.0, 1.0
    hour = datetime.datetime.fromtimestamp(ts).hour + datetime.datetime.fromtimestamp(ts).minute / 60.0
    angle = 2.0 * math.pi * hour / 24.0
    return math.sin(angle), math.cos(angle)
