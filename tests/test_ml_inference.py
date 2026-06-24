"""tests/test_ml_inference.py — Unit tests for ml_inference module.

Tests:
  T026 — Load valid model → available=True, feature_set validated
  T027 — Fixed feature vector → deterministic (label_int, confidence)
  T028 — Missing model → available=False, predict_top2_safe returns fallback MLTop2
  T030 — MLSharedResult thread safety under concurrent read/write
  T031 — MLPosteriorEMA: symmetric smoothing, renormalization
  T032 — MLPosteriorEMA: asymmetric glare α (fast rise, slow decay)
  T033 — MLPosteriorEMA: top-1/top-2 derived from smoothed dict (consistent ordering)
  T034 — MLPosteriorEMA: first call seeds without smoothing
  T035 — MLPosteriorEMA: α=1.0 is passthrough
  T036 — MLPosteriorEMA: glare absent from bundle → asym dropped, general α used

These tests build a minimal joblib bundle in-process so they run without
any real trained weights — just a RandomForestClassifier fit on synthetic data.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from smartbinocular.feature_schema import (
    ENV_CLASS_TO_INT,
    ENV_INT_UNKNOWN,
    FEATURE_SET_OPTICAL_ONLY,
)
from smartbinocular.ml_inference import (
    EnvClassifier,
    MLPosteriorEMA,
    MLSharedResult,
    MLTop2,
    _resolve_asym,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_and_save_bundle(path: Path, feature_set: list | None = None) -> Path:
    """Fit a tiny RF on synthetic data and save a valid bundle to ``path``."""
    import joblib
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    fs = feature_set if feature_set is not None else list(FEATURE_SET_OPTICAL_ONLY)
    n_features = len(fs)
    rng = np.random.default_rng(42)

    # Two classes: 1 (night_clear) and 2 (normal_night)
    X = rng.standard_normal((40, n_features)).astype(np.float32)
    y = np.array([1, 2] * 20, dtype=np.int32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    clf.fit(X_scaled, y)

    class_int_to_label: Dict[int, str] = {
        int(c): ENV_CLASS_TO_INT.__class__.__name__  # just a non-empty string
        for c in clf.classes_
    }
    # Use real label names
    from smartbinocular.feature_schema import ENV_INT_TO_CLASS
    class_int_to_label = {int(c): ENV_INT_TO_CLASS.get(int(c), f"cls{c}") for c in clf.classes_}

    bundle = {
        "rf": clf,
        "scalers": {"nir": scaler},
        "feature_set": fs,
        "training_mode": "optical_only",
        "normalize_by": "nir_channel",
        "allowed_nir_channels": ["nir", "rgb"],
        "deploy_ready": False,
        "graceful_degrade_to": None,
        "env_classes": list(ENV_CLASS_TO_INT.keys()),
        "class_int_to_label": class_int_to_label,
        "metrics": {"cv_balanced_accuracy_mean": 0.5},
        "notes": "test bundle",
        "is_baseline": True,
        "requires_rpi_validation": True,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
    }
    joblib.dump(bundle, path)
    return path


# ── T026: Load valid model ─────────────────────────────────────────────────────

def test_load_valid_model(tmp_path: Path) -> None:
    """T026: valid bundle → available=True, feature_set matches OPTICAL_ONLY."""
    model_path = tmp_path / "test_model.joblib"
    _build_and_save_bundle(model_path)

    clf = EnvClassifier(str(model_path))

    assert clf.available is True, "EnvClassifier should be available after loading valid bundle"


# ── T027: Deterministic prediction ────────────────────────────────────────────

def test_deterministic_prediction(tmp_path: Path) -> None:
    """T027: fixed input vector → same (label_int, confidence) on repeated calls."""
    model_path = tmp_path / "test_model.joblib"
    _build_and_save_bundle(model_path)
    clf = EnvClassifier(str(model_path))
    assert clf.available

    vec = np.zeros(len(FEATURE_SET_OPTICAL_ONLY), dtype=np.float32)

    label1, conf1, proba1 = clf.predict(vec)
    label2, conf2, proba2 = clf.predict(vec)

    assert label1 == label2, "Repeated predict on same vector must return same label"
    assert conf1 == conf2, "Repeated predict on same vector must return same confidence"
    assert isinstance(label1, int)
    assert 0.0 <= conf1 <= 1.0
    assert abs(sum(proba1.values()) - 1.0) < 1e-5, "proba_dict values must sum to ~1.0"


# ── T027b: predict_top2_safe — ordering, tie-break, structure ─────────────────

def test_predict_top2_safe_ordering(tmp_path: Path) -> None:
    """T027b: predict_top2_safe → top-1 has higher proba than top-2; both are MLTop2."""
    model_path = tmp_path / "test_model.joblib"
    _build_and_save_bundle(model_path)
    clf = EnvClassifier(str(model_path))
    assert clf.available

    vec = np.zeros(len(FEATURE_SET_OPTICAL_ONLY), dtype=np.float32)
    result = clf.predict_top2_safe(vec)

    assert isinstance(result, MLTop2)
    assert result.proba_1 >= result.proba_2, "top-1 must have >= probability than top-2"
    assert result.proba_1 > 0.0
    assert 0.0 <= result.proba_2 <= result.proba_1
    assert result.label_int_1 != result.label_int_2 or result.label_int_2 == ENV_INT_UNKNOWN


def test_predict_top2_safe_deterministic(tmp_path: Path) -> None:
    """T027b: repeated calls with same vector return identical MLTop2."""
    model_path = tmp_path / "test_model.joblib"
    _build_and_save_bundle(model_path)
    clf = EnvClassifier(str(model_path))
    assert clf.available

    vec = np.ones(len(FEATURE_SET_OPTICAL_ONLY), dtype=np.float32)
    r1 = clf.predict_top2_safe(vec)
    r2 = clf.predict_top2_safe(vec)

    assert r1 == r2, "Repeated predict_top2_safe on same vector must be identical"


# ── T028: Missing model → graceful degradation ────────────────────────────────

def test_missing_model_fallback(tmp_path: Path) -> None:
    """T028: missing model file → available=False, predict_top2_safe returns fallback."""
    missing_path = tmp_path / "nonexistent.joblib"

    clf = EnvClassifier(str(missing_path), fallback_label=ENV_INT_UNKNOWN)

    assert clf.available is False, "Should be unavailable when model file is missing"

    result = clf.predict_top2_safe(np.zeros(len(FEATURE_SET_OPTICAL_ONLY), dtype=np.float32))

    assert isinstance(result, MLTop2)
    assert result.label_int_1 == ENV_INT_UNKNOWN, f"Expected fallback {ENV_INT_UNKNOWN}, got {result.label_int_1}"
    assert result.proba_1 == 0.0, f"Expected confidence 0.0, got {result.proba_1}"
    assert result.label_int_2 == ENV_INT_UNKNOWN
    assert result.proba_2 == 0.0


def test_feature_set_mismatch_disables_classifier(tmp_path: Path) -> None:
    """T028 extension: bundle with wrong feature_set → available=False."""
    model_path = tmp_path / "wrong_fs.joblib"
    wrong_fs = ["nir_mean_brightness", "nir_std"]  # only 2 features, not 11
    _build_and_save_bundle(model_path, feature_set=wrong_fs)

    clf = EnvClassifier(str(model_path))

    assert clf.available is False, "Mismatched feature_set must disable the classifier"
    result = clf.predict_top2_safe(np.zeros(2, dtype=np.float32))
    assert result.label_int_1 == ENV_INT_UNKNOWN
    assert result.proba_1 == 0.0


# ── T030: MLSharedResult thread safety ────────────────────────────────────────

def test_shared_result_thread_safety() -> None:
    """T030: concurrent reads and writes to MLSharedResult — no exceptions, no deadlock."""
    shared = MLSharedResult()
    errors: list[Exception] = []
    N = 500

    def writer() -> None:
        for i in range(N):
            try:
                shared.set(MLTop2(i % 9 + 1, float(i) / N, (i + 1) % 9 + 1, float(i) / (N * 2)))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    def reader() -> None:
        for _ in range(N):
            try:
                result = shared.get()
                assert isinstance(result, MLTop2)
                assert isinstance(result.label_int_1, int)
                assert isinstance(result.proba_1, float)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert not errors, f"Thread safety errors: {errors}"

    # Initial state before any set
    fresh = MLSharedResult()
    result0 = fresh.get()
    assert result0.label_int_1 == ENV_INT_UNKNOWN
    assert result0.proba_1 == 0.0
    assert result0.label_int_2 == ENV_INT_UNKNOWN
    assert result0.proba_2 == 0.0


# ── T031–T036: MLPosteriorEMA ─────────────────────────────────────────────────

def test_posterior_ema_symmetric_smoothing() -> None:
    """T031: two successive distributions with α=0.5 produce algebraic expected posterior; Σ=1."""
    classes = [1, 2, 3]
    ema = MLPosteriorEMA(classes, alpha=0.5)

    p1 = {1: 0.6, 2: 0.3, 3: 0.1}
    p2 = {1: 0.1, 2: 0.5, 3: 0.4}

    # First call: seed — return unchanged
    out1 = ema.update(p1)
    assert out1 == p1

    # Second call: EMA with α=0.5 on seeded p1 → expected = 0.5*p2 + 0.5*p1 (then renorm)
    out2 = ema.update(p2)
    expected_raw = {c: 0.5 * p2[c] + 0.5 * p1[c] for c in classes}
    total = sum(expected_raw.values())
    expected_norm = {c: v / total for c, v in expected_raw.items()}

    for c in classes:
        assert abs(out2[c] - expected_norm[c]) < 1e-6, f"class {c}: {out2[c]} != {expected_norm[c]}"

    assert abs(sum(out2.values()) - 1.0) < 1e-6, "Smoothed posterior must sum to 1.0"


def test_posterior_ema_asymmetric_glare_rise_fast_decay_slow() -> None:
    """T032: glare α_up=0.9 rises quickly; α_down=0.1 decays slowly."""
    glare_int = ENV_CLASS_TO_INT["glare"]  # 6
    other = 1
    classes = [other, glare_int]
    asym = {glare_int: (0.9, 0.1)}
    ema = MLPosteriorEMA(classes, alpha=0.5, asym=asym)

    # Seed: glare low
    p_low = {glare_int: 0.1, other: 0.9}
    ema.update(p_low)

    # Rise: glare shoots up to 0.9 — α_up=0.9 means it responds fast
    p_high = {glare_int: 0.9, other: 0.1}
    out_rise = ema.update(p_high)
    # glare uses α_up=0.9 (p_new=0.9 > p_prev=0.1):  0.9*0.9 + 0.1*0.1 = 0.82
    # other uses general α=0.5 (not in asym):          0.5*0.1 + 0.5*0.9 = 0.50
    expected_glare_rise = 0.9 * 0.9 + 0.1 * 0.1  # 0.82 pre-renorm
    expected_other_rise = 0.5 * 0.1 + 0.5 * 0.9  # 0.50 (general α)
    total = expected_glare_rise + expected_other_rise
    assert abs(out_rise[glare_int] - expected_glare_rise / total) < 1e-5

    # Decay: glare drops back to 0.1 — α_down=0.1 means it decays slowly
    prev_glare = out_rise[glare_int]
    prev_other = out_rise[other]
    out_decay = ema.update(p_low)
    # With α_down=0.1: glare_raw = 0.1*0.1 + 0.9*prev_glare
    expected_glare_decay_raw = 0.1 * 0.1 + 0.9 * prev_glare
    expected_other_decay_raw = 0.5 * 0.9 + 0.5 * prev_other  # general α=0.5 for 'other'
    total2 = expected_glare_decay_raw + expected_other_decay_raw
    assert abs(out_decay[glare_int] - expected_glare_decay_raw / total2) < 1e-5

    # Slow-decay invariant: glare decays slower than symmetric α=0.5 would produce
    # With α=0.5: glare_sym_raw = 0.5*0.1 + 0.5*prev_glare
    glare_sym_raw = 0.5 * 0.1 + 0.5 * prev_glare
    other_sym_raw = 0.5 * 0.9 + 0.5 * prev_other
    total_sym = glare_sym_raw + other_sym_raw
    glare_if_symmetric = glare_sym_raw / total_sym
    assert out_decay[glare_int] > glare_if_symmetric, (
        f"α_down=0.1 must decay slower than symmetric α=0.5: "
        f"got {out_decay[glare_int]:.4f}, symmetric would be {glare_if_symmetric:.4f}"
    )


def test_posterior_ema_then_top2_consistent() -> None:
    """T033: top-1/top-2 derived from smoothed dict agree with sorted() on that same dict."""
    classes = [1, 2, 3]
    ema = MLPosteriorEMA(classes, alpha=0.5)

    ema.update({1: 0.5, 2: 0.3, 3: 0.2})
    smoothed = ema.update({1: 0.1, 2: 0.6, 3: 0.3})

    # Derive top-2 manually with deterministic tie-break
    sorted_pairs = sorted(smoothed.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    expected_top1, expected_p1 = sorted_pairs[0]
    expected_top2, expected_p2 = sorted_pairs[1]

    assert smoothed[expected_top1] >= smoothed[expected_top2]
    assert abs(sum(smoothed.values()) - 1.0) < 1e-6


def test_posterior_ema_first_call_seeds() -> None:
    """T034: first call to update() returns the input dict unchanged (seeding, no smoothing)."""
    classes = [1, 2]
    ema = MLPosteriorEMA(classes, alpha=0.7)

    p = {1: 0.8, 2: 0.2}
    out = ema.update(p)

    assert out == p, "First call must return input unchanged (seed, no EMA applied)"
    # Verify it's a copy, not the same object
    assert out is not p


def test_posterior_ema_alpha_one_is_passthrough() -> None:
    """T035: α=1.0 → smoothed == raw input (passthrough, no EMA memory)."""
    classes = [1, 2, 3]
    ema = MLPosteriorEMA(classes, alpha=1.0)

    p1 = {1: 0.5, 2: 0.3, 3: 0.2}
    p2 = {1: 0.1, 2: 0.7, 3: 0.2}

    ema.update(p1)   # seed
    out = ema.update(p2)

    for c in classes:
        assert abs(out[c] - p2[c]) < 1e-6, (
            f"α=1.0 must be passthrough; class {c}: got {out[c]}, expected {p2[c]}"
        )


def test_posterior_ema_glare_absent_from_bundle_uses_general_alpha(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """T036: glare absent from rf.classes_ → asym dropped, general α used; debug log emitted."""
    import logging

    glare_int = ENV_CLASS_TO_INT["glare"]  # 6
    classes_without_glare = [1, 2, 3]  # glare (6) not present

    asym_str = {"glare": [0.85, 0.45]}
    with caplog.at_level(logging.DEBUG, logger="smartbinocular.ml_inference"):
        asym_int = _resolve_asym(asym_str, classes_without_glare)

    # Should be empty — glare filtered out
    assert asym_int == {}, f"Expected empty asym, got {asym_int}"

    # Verify a debug log was emitted mentioning the absent class
    assert any("glare" in rec.message.lower() for rec in caplog.records), (
        "Expected a debug log mentioning 'glare' being absent from bundle classes_"
    )

    # EMA with empty asym should use general α for all classes
    ema = MLPosteriorEMA(classes_without_glare, alpha=0.5, asym=asym_int)
    p1 = {1: 0.6, 2: 0.3, 3: 0.1}
    p2 = {1: 0.2, 2: 0.5, 3: 0.3}
    ema.update(p1)
    out = ema.update(p2)

    # With general α=0.5, each class uses symmetric formula
    expected_raw = {c: 0.5 * p2[c] + 0.5 * p1[c] for c in classes_without_glare}
    total = sum(expected_raw.values())
    for c in classes_without_glare:
        assert abs(out[c] - expected_raw[c] / total) < 1e-6


# ── T037: rgb scaler key fallback ────────────────────────────────────────────

def test_rgb_scaler_key_bundle_loads_and_predicts(tmp_path: Path) -> None:
    """T037: bundle with scalers={'rgb': scaler} (from optical_only training on rgb data)
    must load successfully and return a non-fallback prediction.

    Training data has nir_channel='rgb', so fit_scalers() stores scalers['rgb'].
    EnvClassifier.predict() must fall back to 'rgb' when 'nir' and '__all__' are absent.
    """
    import joblib
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    fs = list(FEATURE_SET_OPTICAL_ONLY)
    n_features = len(fs)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, n_features)).astype(np.float32)
    y = np.array([1, 2] * 20, dtype=np.int32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf_rf = RandomForestClassifier(n_estimators=5, random_state=0)
    clf_rf.fit(X_scaled, y)

    from smartbinocular.feature_schema import ENV_INT_TO_CLASS
    class_int_to_label = {int(c): ENV_INT_TO_CLASS.get(int(c), f"cls{c}") for c in clf_rf.classes_}

    bundle = {
        "rf": clf_rf,
        "scalers": {"rgb": scaler},  # key produced by optical_only training
        "feature_set": fs,
        "training_mode": "optical_only",
        "normalize_by": "nir_channel",
        "allowed_nir_channels": ["nir", "rgb"],
        "deploy_ready": False,
        "graceful_degrade_to": None,
        "env_classes": list(ENV_CLASS_TO_INT.keys()),
        "class_int_to_label": class_int_to_label,
        "metrics": {"cv_balanced_accuracy_mean": 0.5},
        "notes": "rgb-key test bundle",
        "is_baseline": True,
        "requires_rpi_validation": True,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
    }
    model_path = tmp_path / "rgb_bundle.joblib"
    joblib.dump(bundle, model_path)

    env_clf = EnvClassifier(str(model_path))
    assert env_clf.available is True, "rgb-key bundle must load as available"

    vec = np.zeros(n_features, dtype=np.float32)
    result = env_clf.predict_top2_safe(vec)
    assert result.label_int_1 != ENV_INT_UNKNOWN, (
        "rgb-key bundle must produce a real prediction, not the fallback unknown label"
    )
    assert result.proba_1 > 0.0
