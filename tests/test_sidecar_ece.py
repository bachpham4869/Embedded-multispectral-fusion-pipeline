"""
Policy-bound test: ECE values in models/production/env_classifier.json.

Permitted reads: models/production/env_classifier.json (JSON only).
No GPU, no joblib.load, no network.

Policy bounds (conservative for future retrains and class-imbalance drift):
  - Global bound: all classes ECE ≤ 0.15
  - Night classes (night_clear, normal_night, nir_night): ECE ≤ 0.10

These bounds are intentionally wider than the current observed values (all < 0.05)
to accommodate class-imbalance drift if the training set is augmented or rebalanced.
Setting bounds at the current values would be a snapshot pin, not a policy.

Cross-reference: docs/PIPELINE_EVIDENCE_REGISTER.md §D.12
"""
import json
import pathlib
import pytest

_REPO = pathlib.Path(__file__).parent.parent
_SIDECAR = _REPO / "models/production/env_classifier.json"

_NIGHT_CLASSES = {"night_clear", "normal_night", "nir_night"}
_GLOBAL_ECE_BOUND = 0.15
_NIGHT_ECE_BOUND = 0.10


def _load_sidecar() -> dict:
    if not _SIDECAR.exists():
        pytest.skip(f"sidecar not found: {_SIDECAR}")
    return json.loads(_SIDECAR.read_text())


def test_ece_present_for_all_classes():
    """ece_by_class must contain an entry for every ENV_CLASS (9 classes)."""
    sidecar = _load_sidecar()
    ece = sidecar.get("ece_by_class", {})
    from smartbinocular.feature_schema import ENV_CLASSES
    missing = [c for c in ENV_CLASSES if c not in ece]
    assert not missing, f"ece_by_class missing classes: {missing}"


def test_global_ece_bound():
    """All classes must have ECE ≤ 0.15 (policy bound — not a snapshot)."""
    sidecar = _load_sidecar()
    ece = sidecar.get("ece_by_class", {})
    violations = {
        cls: val for cls, val in ece.items() if float(val) > _GLOBAL_ECE_BOUND
    }
    assert not violations, (
        f"Classes exceeding global ECE bound ({_GLOBAL_ECE_BOUND}): {violations}"
    )


def test_night_class_ece_bound():
    """Night classes (night_clear, normal_night, nir_night) must have ECE ≤ 0.10.

    Tighter bound: night-class miscalibration directly affects the critical
    surveillance use case. The policy bound is conservative for future retrains.
    """
    sidecar = _load_sidecar()
    ece = sidecar.get("ece_by_class", {})
    violations = {
        cls: ece[cls]
        for cls in _NIGHT_CLASSES
        if cls in ece and float(ece[cls]) > _NIGHT_ECE_BOUND
    }
    assert not violations, (
        f"Night classes exceeding tight ECE bound ({_NIGHT_ECE_BOUND}): {violations}"
    )
