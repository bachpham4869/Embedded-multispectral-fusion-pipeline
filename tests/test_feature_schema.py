"""tests/test_feature_schema.py — Tests for feature_schema.py (Task 5: nir_blue_mean_ema).

Tests:
  FS01 — FEATURE_SET_OPTICAL_ONLY has 12 features, last is 'nir_blue_mean_ema'
  FS02 — 'nir_blue_mean_ema' is in FEATURE_SET_OPTICAL_ONLY; 'nir_b_ema' is NOT
  FS03 — FeatureRecord.to_feature_array(FEATURE_SET_OPTICAL_ONLY) returns shape (12,)
  FS04 — Round-trip FeatureRecord with nir_blue_mean_ema via to_dict/from_dict
  FS05 — to_feature_array raises ValueError when nir_blue_mean_ema is None
  FS06 — from_dict on old 11-feature record (missing key) seeds default 0.0
  FS07 — test_feature_set_mismatch_disables_classifier still passes (regression)
"""

from __future__ import annotations

import pytest
import numpy as np

from smartbinocular.feature_schema import (
    FEATURE_SET_OPTICAL_ONLY,
    FeatureRecord,
    ENV_INT_UNKNOWN,
)


# ── Fixture: minimal valid 12-feature FeatureRecord ───────────────────────────

def _make_record(**overrides) -> FeatureRecord:
    base = dict(
        nir_mean_brightness=80.0,
        nir_std=20.0,
        nir_entropy=6.5,
        nir_p95=150.0,
        nir_glare_score=0.01,
        nir_sharpness=120.0,
        nir_dark_fraction=0.05,
        nir_saturation_mean=30.0,
        hour_of_day_sin=0.5,
        hour_of_day_cos=0.866,
        prev_env_class=ENV_INT_UNKNOWN,
        nir_blue_mean_ema=45.0,
    )
    base.update(overrides)
    return FeatureRecord(**base)


# ── FS01: Feature count and last feature name ─────────────────────────────────

def test_feature_set_optical_only_has_12_features() -> None:
    """FS01: FEATURE_SET_OPTICAL_ONLY must have exactly 12 features."""
    assert len(FEATURE_SET_OPTICAL_ONLY) == 12, (
        f"Expected 12 features, got {len(FEATURE_SET_OPTICAL_ONLY)}: {FEATURE_SET_OPTICAL_ONLY}"
    )


def test_feature_set_last_is_nir_blue_mean_ema() -> None:
    """FS01b: last feature must be 'nir_blue_mean_ema' (appended at end, scaler col order stable)."""
    assert FEATURE_SET_OPTICAL_ONLY[-1] == "nir_blue_mean_ema", (
        f"Expected last feature to be 'nir_blue_mean_ema', got '{FEATURE_SET_OPTICAL_ONLY[-1]}'"
    )


# ── FS02: Naming collision guard ──────────────────────────────────────────────

def test_nir_blue_mean_ema_in_feature_set() -> None:
    """FS02: 'nir_blue_mean_ema' is in FEATURE_SET_OPTICAL_ONLY."""
    assert "nir_blue_mean_ema" in FEATURE_SET_OPTICAL_ONLY


def test_nir_b_ema_not_in_feature_set() -> None:
    """FS02b: 'nir_b_ema' must NOT be in FEATURE_SET_OPTICAL_ONLY — guards against collision
    with main.py's existing brightness EMA (coefficient NIR_B_EMA, lines 748-798).
    """
    assert "nir_b_ema" not in FEATURE_SET_OPTICAL_ONLY, (
        "'nir_b_ema' must not appear in FEATURE_SET_OPTICAL_ONLY — "
        "it is main.py's brightness EMA, not the new blue-channel ML feature"
    )


# ── FS03: to_feature_array returns shape (12,) ────────────────────────────────

def test_to_feature_array_returns_12_elements() -> None:
    """FS03: to_feature_array(FEATURE_SET_OPTICAL_ONLY) returns a float32 array of shape (12,)."""
    rec = _make_record()
    arr = rec.to_feature_array(FEATURE_SET_OPTICAL_ONLY)
    assert arr.shape == (12,), f"Expected shape (12,), got {arr.shape}"
    assert arr.dtype == np.float32


def test_to_feature_array_nir_blue_mean_ema_value() -> None:
    """FS03b: nir_blue_mean_ema appears at position 11 (0-based) in the array."""
    rec = _make_record(nir_blue_mean_ema=99.0)
    arr = rec.to_feature_array(FEATURE_SET_OPTICAL_ONLY)
    assert abs(arr[11] - 99.0) < 1e-5, f"Expected arr[11]=99.0, got {arr[11]}"


# ── FS04: Round-trip via to_dict / from_dict ──────────────────────────────────

def test_feature_record_round_trip_with_nir_blue_mean_ema() -> None:
    """FS04: to_dict → from_dict preserves nir_blue_mean_ema."""
    rec = _make_record(nir_blue_mean_ema=37.5)
    d = rec.to_dict()
    assert "nir_blue_mean_ema" in d
    assert abs(d["nir_blue_mean_ema"] - 37.5) < 1e-5

    rec2 = FeatureRecord.from_dict(d)
    assert abs(rec2.nir_blue_mean_ema - 37.5) < 1e-5


# ── FS05: to_feature_array raises if nir_blue_mean_ema is None ────────────────

def test_to_feature_array_raises_when_nir_blue_mean_ema_is_none() -> None:
    """FS05: to_feature_array raises ValueError if nir_blue_mean_ema is None (C8)."""
    # Bypass the dataclass default by directly patching after construction
    rec = _make_record()
    object.__setattr__(rec, "nir_blue_mean_ema", None)
    with pytest.raises((ValueError, TypeError)):
        rec.to_feature_array(FEATURE_SET_OPTICAL_ONLY)


# ── FS06: from_dict on old 11-feature record uses default 0.0 ────────────────

def test_from_dict_old_record_without_nir_blue_mean_ema_defaults_to_zero() -> None:
    """FS06: from_dict on an old-style record (without nir_blue_mean_ema key) seeds 0.0."""
    old_dict = {
        "nir_mean_brightness": 80.0,
        "nir_std": 20.0,
        "nir_entropy": 6.5,
        "nir_p95": 150.0,
        "nir_glare_score": 0.01,
        "nir_sharpness": 120.0,
        "nir_dark_fraction": 0.05,
        "nir_saturation_mean": 30.0,
        "hour_of_day_sin": 0.5,
        "hour_of_day_cos": 0.866,
        "prev_env_class": 0,
        # no nir_blue_mean_ema key
    }
    rec = FeatureRecord.from_dict(old_dict)
    assert rec.nir_blue_mean_ema == 0.0, (
        f"Old records without nir_blue_mean_ema should default to 0.0, got {rec.nir_blue_mean_ema}"
    )


# ── FS07: Feature set mismatch disables EnvClassifier (regression guard) ──────

def test_feature_set_mismatch_disables_classifier(tmp_path) -> None:
    """FS07: 11-feature bundle is rejected (feature_set mismatch) → EnvClassifier.available=False."""
    import joblib
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from smartbinocular.feature_schema import ENV_INT_UNKNOWN
    from smartbinocular.ml_inference import EnvClassifier

    # Build an 11-feature bundle (old format, missing nir_blue_mean_ema)
    old_fs = list(FEATURE_SET_OPTICAL_ONLY[:-1])  # drop last (nir_blue_mean_ema)
    assert len(old_fs) == 11
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 11)).astype(np.float32)
    y = np.array([1, 2] * 10, dtype=np.int32)
    scaler = StandardScaler().fit(X)
    clf = RandomForestClassifier(n_estimators=3, random_state=0).fit(scaler.transform(X), y)
    bundle = {
        "rf": clf,
        "scalers": {"nir": scaler},
        "feature_set": old_fs,
        "normalize_by": "nir_channel",
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
    }
    model_path = tmp_path / "old_bundle.joblib"
    joblib.dump(bundle, model_path)

    clf_loaded = EnvClassifier(str(model_path))
    assert clf_loaded.available is False, (
        "11-feature bundle must be rejected (feature_set mismatch with 12-feature FEATURE_SET_OPTICAL_ONLY)"
    )
    result = clf_loaded.predict_top2_safe(np.zeros(11, dtype=np.float32))
    assert result.label_int_1 == ENV_INT_UNKNOWN
    assert result.proba_1 == 0.0
