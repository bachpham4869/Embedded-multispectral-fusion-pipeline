#!/usr/bin/env python3
"""train_classifier.py — Train ENV classifiers for SmartBinocular.

⚠️ BASELINE OUTPUT — Models saved to models/baseline/ are NOT production-ready.
   Deploy only after Phase 3C RPi validation (accuracy ≥ 0.75 on-device).
   See legacy/md/OFFLINE_ML_PLAN.md Constraint [C6] and Section 7.3.

Usage:
    # Train optical_only baseline (PRIMARY model)
    python models/train_classifier.py \\
        --mode optical_only \\
        --dataset data/training/optical_only.jsonl \\
        --output models/baseline/rf_optical_only.joblib

    # Evaluate only (no training)
    python models/train_classifier.py \\
        --mode optical_only \\
        --dataset data/training/optical_only.jsonl \\
        --eval-only

    # Evaluate a saved model on a random hold-out subset of a JSONL (quick sanity check)
    python models/train_classifier.py \\
        --mode optical_only \\
        --dataset data/training/optical_only.jsonl \\
        --evaluate-model models/baseline/rf_optical_only.joblib \\
        --eval-test-size 0.2 \\
        --eval-max-samples 2000

    # After training, emit reliability PNGs + ECE for all 9 ENV_CLASSES (OOF)
    python models/train_classifier.py --mode optical_only --dataset <jsonl> --output <path>.joblib \\
        --reliability-scope all

    # Train ablation (research only, NOT for deployment)
    python models/train_classifier.py \\
        --mode rgb_thermal_ablation \\
        --dataset data/training/optical_thermal_ablation.jsonl \\
        --output models/ablation/rf_rgb_thermal.joblib
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smartbinocular.feature_schema import (
    ENV_CLASSES,
    FEATURE_SET_OPTICAL_9,
    FEATURE_SET_OPTICAL_ONLY,
    FEATURE_SET_OPTICAL_THERMAL,
    FeatureRecord,
)


# ── Training mode definitions ─────────────────────────────────────────────────

TRAINING_MODES: Dict[str, Dict[str, Any]] = {
    "optical_only": {
        # ── PRIMARY PRODUCTION MODE ──────────────────────────────────────────
        "feature_set": FEATURE_SET_OPTICAL_ONLY,          # 12 features
        "allowed_nir_channels": ["nir", "rgb"],           # both OK
        "allowed_thermal_channels": ["none", "lwir"],     # thermal not required
        "requires_thermal": False,
        "requires_motion": False,
        "normalize_by": "nir_channel",                    # separate scaler per channel (C10)
        "preferred_label_sources": ["dataset_original", "manual"],
        "deploy_ready": True,
        "graceful_degrade_to": None,
        "output_dir": "models/baseline",
        "notes": (
            "Primary production model. Works on ALL hardware configs. "
            "Optical-only — no MI48 required."
        ),
    },
    "with_thermal": {
        # ── OPTIONAL ENHANCEMENT ─────────────────────────────────────────────
        "feature_set": FEATURE_SET_OPTICAL_THERMAL,       # 17 features
        "allowed_nir_channels": ["nir"],                  # RPi NIR only
        "allowed_thermal_channels": ["lwir"],
        "requires_thermal": True,
        "requires_motion": False,
        "normalize_by": "fixed",                          # single combo (nir+lwir)
        "preferred_label_sources": ["manual"],
        "deploy_ready": True,                             # only after RPi validation
        "graceful_degrade_to": "optical_only",            # MANDATORY fallback
        "output_dir": "models/baseline",
        "notes": (
            "Enhancement when MI48 present. MUST gracefully degrade to optical_only "
            "when thermal unavailable. NOT achievable from offline datasets alone "
            "(use RPi field sessions — Phase 3C)."
        ),
    },
    "rgb_thermal_ablation": {
        # ── RESEARCH / ABLATION — NEVER DEPLOY ───────────────────────────────
        "feature_set": FEATURE_SET_OPTICAL_THERMAL,
        "allowed_nir_channels": ["rgb"],
        "allowed_thermal_channels": ["lwir"],
        "requires_thermal": True,
        "requires_motion": False,
        "normalize_by": "fixed",
        "preferred_label_sources": ["dataset_original", "manual", "weak_heuristic"],
        "deploy_ready": False,                            # rgb ≠ NIR channel of RPi
        "graceful_degrade_to": None,
        "output_dir": "models/ablation",
        "notes": (
            "Ablation only. Validates thermal feature pipeline with KAIST data. "
            "DO NOT deploy: RGB channel ≠ NIR channel of RPi IMX290."
        ),
    },
    "optical_only_9": {
        # ── FEATURE-ABLATION — NEVER DEPLOY ──────────────────────────────────
        "feature_set": FEATURE_SET_OPTICAL_9,              # 9 features (drops temporal zeros)
        "allowed_nir_channels": ["nir", "rgb"],
        "allowed_thermal_channels": ["none", "lwir"],
        "requires_thermal": False,
        "requires_motion": False,
        "normalize_by": "nir_channel",
        "preferred_label_sources": ["dataset_original", "manual"],
        "deploy_ready": False,
        "graceful_degrade_to": None,
        "output_dir": "models/ablation",
        "notes": (
            "Feature ablation: 9-feature optical-only dropping hour_of_day_sin/cos "
            "and prev_env_class (zero-importance features). Compare CV accuracy and ECE "
            "against optical_only (12 features) to document the feature selection choice. "
            "DO NOT deploy: 9-feature bundle fails EnvClassifier.feature_set validation."
        ),
    },
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_dataset(
    path: Path,
    mode_cfg: dict,
    require_label_sources: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Load JSONL, filter by mode constraints, return (X, y, groups, scalers_data).

    Returns:
        X:            (N, n_features) float32 feature matrix
        y:            (N,) int label array (ENV_CLASS_TO_INT mapping)
        nir_groups:   (N,) string array of nir_channel values (for per-group scaling)
        raw_records:  List of raw dicts (for debugging)
    """
    from smartbinocular.feature_schema import ENV_CLASS_TO_INT

    feature_set: List[str] = mode_cfg["feature_set"]
    if mode_cfg is TRAINING_MODES.get("optical_only") or feature_set is FEATURE_SET_OPTICAL_ONLY:
        assert len(feature_set) == 12, (
            f"optical_only feature_set must have 12 features (got {len(feature_set)}). "
            "Regenerate training JSONL via tools/offline_pipeline.py after the "
            "nir_blue_mean_ema (feature #12) schema bump."
        )
    allowed_nir = set(mode_cfg["allowed_nir_channels"])
    allowed_thm = set(mode_cfg["allowed_thermal_channels"])

    rows: List[np.ndarray] = []
    labels: List[int] = []
    nir_groups: List[str] = []
    skipped = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Filter by allowed channels
            if d.get("nir_channel") not in allowed_nir:
                skipped += 1
                continue
            if d.get("thermal_channel") not in allowed_thm:
                skipped += 1
                continue

            # Effective label
            lbl = d.get("label") or d.get("weak_label")
            if not lbl or lbl not in ENV_CLASS_TO_INT:
                skipped += 1
                continue

            # Label source filter
            if require_label_sources:
                ls = d.get("label_source")
                if ls not in require_label_sources:
                    skipped += 1
                    continue

            # Feature compatibility
            try:
                record = FeatureRecord.from_dict(d)
                vec = record.to_feature_array(feature_set)
            except (ValueError, TypeError):
                skipped += 1
                continue

            rows.append(vec)
            labels.append(ENV_CLASS_TO_INT[lbl])
            nir_groups.append(d.get("nir_channel", "rgb"))

    if not rows:
        raise ValueError(f"No compatible records found in {path} for mode.")

    X = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    nir_arr = np.array(nir_groups)

    print(f"  Loaded {len(rows)} records, skipped {skipped}.")
    return X, y, nir_arr


# ── Normalization ─────────────────────────────────────────────────────────────

def fit_scalers(
    X: np.ndarray, nir_groups: np.ndarray, normalize_by: str
) -> Dict[str, Any]:
    """Fit and return per-group StandardScalers (C10 — no cross-channel mixing)."""
    from sklearn.preprocessing import StandardScaler  # type: ignore

    scalers: Dict[str, StandardScaler] = {}
    if normalize_by == "nir_channel":
        for group in np.unique(nir_groups):
            mask = nir_groups == group
            scaler = StandardScaler()
            scaler.fit(X[mask])
            scalers[group] = scaler
    else:
        # Single scaler for fixed combo
        scaler = StandardScaler()
        scaler.fit(X)
        scalers["__all__"] = scaler
    return scalers


def apply_scalers(
    X: np.ndarray, nir_groups: np.ndarray, scalers: Dict[str, Any], normalize_by: str
) -> np.ndarray:
    """Apply fitted scalers to X in-place (returns new array)."""
    X_out = X.copy()
    if normalize_by == "nir_channel":
        for group, scaler in scalers.items():
            mask = nir_groups == group
            if mask.any():
                X_out[mask] = scaler.transform(X[mask])
    else:
        X_out = scalers["__all__"].transform(X)
    return X_out


def subsample_stratified(
    X: np.ndarray,
    y: np.ndarray,
    nir_groups: np.ndarray,
    max_samples: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce to at most max_samples rows with stratified sampling."""
    if len(X) <= max_samples:
        return X, y, nir_groups
    from sklearn.model_selection import StratifiedShuffleSplit  # type: ignore

    sss = StratifiedShuffleSplit(
        n_splits=1, train_size=max_samples, random_state=random_state
    )
    for train_idx, _ in sss.split(X, y):
        return X[train_idx], y[train_idx], nir_groups[train_idx]
    return X, y, nir_groups


def evaluate_saved_model(
    bundle: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    nir_groups: np.ndarray,
    mode_cfg: Dict[str, Any],
    *,
    test_size: float,
    max_samples: Optional[int],
    random_state: int,
    show_confusion: bool,
    held_out_file_only: bool = False,
) -> None:
    """Run held-out metrics on a frozen joblib bundle (RF + scalers).

    If ``max_samples`` is set, stratified-subsample the loaded matrix first, then split.

    If ``held_out_file_only``, use all loaded rows as the test set (no internal split).
    """
    from sklearn.metrics import balanced_accuracy_score  # type: ignore
    from sklearn.metrics import classification_report, confusion_matrix  # type: ignore
    from sklearn.model_selection import train_test_split  # type: ignore

    rf = bundle["rf"]
    scalers: Dict[str, Any] = bundle["scalers"]
    normalize_by: str = bundle["normalize_by"]
    feat_saved: List[str] = list(bundle.get("feature_set") or [])
    if feat_saved != mode_cfg["feature_set"]:
        print(
            "⚠️  feature_set in bundle differs from current mode — using bundle's training assumptions."
        )

    n0 = len(X)
    if max_samples is not None and n0 > max_samples:
        X, y, nir_groups = subsample_stratified(
            X, y, nir_groups, max_samples, random_state
        )
        print(f"  Subsampled stratified: {n0} → {len(X)} rows (--eval-max-samples).")

    if len(X) < 1:
        raise ValueError("Not enough rows after subsampling for evaluation.")

    if held_out_file_only:
        X_te, y_te, g_te = X, y, nir_groups
    else:
        if len(X) < 4:
            raise ValueError("Not enough rows after subsampling for a train/test split.")
        X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
            X,
            y,
            nir_groups,
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )

    X_te_scaled = apply_scalers(X_te, g_te, scalers, normalize_by)
    y_pred = rf.predict(X_te_scaled)
    proba = rf.predict_proba(X_te_scaled)
    conf = float(np.max(proba, axis=1).mean())

    bal = float(balanced_accuracy_score(y_te, y_pred))
    acc = float(np.mean(y_pred == y_te))

    from smartbinocular.feature_schema import ENV_INT_TO_CLASS

    labels_sorted = sorted(np.unique(np.concatenate([y_te, y_pred])))

    print(f"\n{'='*60}")
    print("Held-out evaluation (frozen model)")
    print(f"{'='*60}")
    if held_out_file_only:
        print(f"  Test rows: {len(y_te)} (full --test-dataset file; no internal split)")
    else:
        print(f"  Test rows: {len(y_te)} (hold-out fraction={test_size:.3f} of eval pool)")
    print(f"  Accuracy:           {acc:.4f}")
    print(f"  Balanced accuracy:  {bal:.4f}")
    print(f"  Mean max(P(class)): {conf:.4f}  (proxy for avg. prediction confidence)")

    print("\n  Per-class (precision / recall / F1 on test):")
    print(
        classification_report(
            y_te,
            y_pred,
            labels=labels_sorted,
            target_names=[ENV_INT_TO_CLASS.get(int(i), str(i)) for i in labels_sorted],
            digits=3,
            zero_division=0,
        )
    )

    if show_confusion:
        cm = confusion_matrix(y_te, y_pred, labels=labels_sorted)
        print("  Confusion matrix (rows=true, cols=pred):")
        header = " " * 18 + "".join(f"{ENV_INT_TO_CLASS.get(int(c), '?')[:7]:>8}" for c in labels_sorted)
        print(header)
        for i, row in enumerate(cm):
            ti = ENV_INT_TO_CLASS.get(int(labels_sorted[i]), "?")[:16]
            print(f"  {ti:16s}" + "".join(f"{v:8d}" for v in row))
    print(f"{'='*60}\n")


# ── Training ───────────────────────────────────────────────────────────────────

def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Binary one-vs-rest Expected Calibration Error."""
    bin_edges = np.linspace(0.0, 1.0 + 1e-9, n_bins + 1)
    binids = np.clip(np.digitize(y_prob, bin_edges) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = binids == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        ece += (n_b / len(y_true)) * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return ece


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    nir_groups: np.ndarray,
    mode_cfg: dict,
    n_estimators: int = 200,
    cv_folds: int = 5,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """Train RF with TimeSeriesSplit CV + isotonic calibration.

    Returns (calibrated_model, scalers, metrics).

    CV strategy: TimeSeriesSplit — respects temporal ordering of field-log streams,
    preventing same-session frames from leaking across folds (P1-B).

    Calibration: CalibratedClassifierCV(method='isotonic', cv=5) wraps the final
    RF so predict_proba() values are well-calibrated (prerequisite for the 0.62
    confidence gate).  Feature importances are extracted from the uncalibrated
    final_clf before wrapping (P1-A).

    OOF probability matrix stored under '_oof_*' keys for reliability diagrams;
    stripped from the JSON sidecar in save_model() (P1-D).
    """
    from sklearn.calibration import CalibratedClassifierCV  # type: ignore
    from sklearn.ensemble import RandomForestClassifier  # type: ignore
    from sklearn.metrics import balanced_accuracy_score  # type: ignore
    from sklearn.model_selection import TimeSeriesSplit  # type: ignore

    normalize_by = mode_cfg["normalize_by"]

    cv = TimeSeriesSplit(n_splits=cv_folds)
    cv_scores: List[float] = []

    # OOF proba matrix aligned to global class order (P1-D)
    global_classes = np.sort(np.unique(y))
    n_global = len(global_classes)
    class_to_col = {int(c): i for i, c in enumerate(global_classes)}
    oof_proba_mat = np.zeros((len(y), n_global), dtype=np.float32)
    oof_label_arr = np.full(len(y), -1, dtype=np.int32)
    oof_covered = np.zeros(len(y), dtype=bool)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        g_tr = nir_groups[train_idx]
        g_val = nir_groups[val_idx]

        scalers_fold = fit_scalers(X_tr, g_tr, normalize_by)
        X_tr_scaled = apply_scalers(X_tr, g_tr, scalers_fold, normalize_by)
        X_val_scaled = apply_scalers(X_val, g_val, scalers_fold, normalize_by)

        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=20, min_samples_leaf=4,
            max_features="sqrt", random_state=42, n_jobs=-1, class_weight="balanced",
        )
        clf.fit(X_tr_scaled, y_tr)
        proba_val = clf.predict_proba(X_val_scaled)
        preds = clf.classes_[np.argmax(proba_val, axis=1)]
        # Map this fold's class columns to the global OOF matrix
        for fi, cls in enumerate(clf.classes_):
            gc = class_to_col.get(int(cls), -1)
            if gc >= 0:
                oof_proba_mat[val_idx, gc] = proba_val[:, fi]
        oof_label_arr[val_idx] = y_val
        oof_covered[val_idx] = True
        score = balanced_accuracy_score(y_val, preds)
        cv_scores.append(score)
        print(f"    Fold {fold+1}/{cv_folds}: balanced_accuracy = {score:.4f}")

    mean_cv = float(np.mean(cv_scores))
    std_cv = float(np.std(cv_scores))
    print(f"  CV balanced_accuracy: {mean_cv:.4f} ± {std_cv:.4f}")

    # Final model on full data with depth/leaf constraints to reduce overfitting (P1-A)
    scalers_final = fit_scalers(X, nir_groups, normalize_by)
    X_scaled = apply_scalers(X, nir_groups, scalers_final, normalize_by)
    final_clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=20, min_samples_leaf=4,
        max_features="sqrt", random_state=42, n_jobs=-1, class_weight="balanced",
    )
    final_clf.fit(X_scaled, y)
    train_pred = final_clf.predict(X_scaled)
    train_score = balanced_accuracy_score(y, train_pred)

    # Isotonic calibration — cv=5 learns the mapping from OOF fold predictions (P1-A)
    calibrated = CalibratedClassifierCV(final_clf, method="isotonic", cv=5)
    calibrated.fit(X_scaled, y)

    metrics: Dict[str, Any] = {
        "cv_balanced_accuracy_mean": mean_cv,
        "cv_balanced_accuracy_std": std_cv,
        "train_balanced_accuracy": train_score,
        "n_samples": len(y),
        "n_features": X.shape[1],
        "cv_folds": cv_folds,
        "cv_strategy": "TimeSeriesSplit",
        "calibration_method": "isotonic",
    }

    feature_set = mode_cfg["feature_set"]
    importances = dict(zip(feature_set, final_clf.feature_importances_.tolist()))
    metrics["feature_importances"] = importances

    # OOF arrays for reliability diagrams — stripped from JSON sidecar by save_model
    oof_mask = oof_covered
    metrics["_oof_proba"] = oof_proba_mat[oof_mask]
    metrics["_oof_labels"] = oof_label_arr[oof_mask]
    metrics["_oof_classes"] = global_classes

    return calibrated, scalers_final, metrics


# ── Save model bundle ──────────────────────────────────────────────────────────

def save_model(
    output_path: Path,
    clf: Any,
    scalers: Dict[str, Any],
    mode_name: str,
    mode_cfg: dict,
    metrics: Dict[str, Any],
) -> None:
    """Save model, scalers, and metadata as a joblib bundle."""
    import joblib  # type: ignore
    import sklearn  # type: ignore

    from smartbinocular.feature_schema import ENV_INT_TO_CLASS

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build inverse label mapping for only the classes the model was trained on
    class_int_to_label: Dict[int, str] = {
        int(c): ENV_INT_TO_CLASS.get(int(c), f"unknown_{c}")
        for c in clf.classes_
    }

    bundle = {
        "rf": clf,
        "scalers": scalers,
        "feature_set": mode_cfg["feature_set"],
        "training_mode": mode_name,
        "normalize_by": mode_cfg["normalize_by"],
        "allowed_nir_channels": mode_cfg["allowed_nir_channels"],
        "deploy_ready": mode_cfg["deploy_ready"],
        "graceful_degrade_to": mode_cfg["graceful_degrade_to"],
        "env_classes": ENV_CLASSES,
        "class_int_to_label": class_int_to_label,
        "metrics": metrics,
        "notes": mode_cfg["notes"],
        "is_baseline": True,
        "requires_rpi_validation": not mode_cfg["deploy_ready"] or True,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "calibration_method": metrics.get("calibration_method", "none"),
        "cv_strategy": metrics.get("cv_strategy", "unknown"),
    }
    joblib.dump(bundle, output_path)
    print(f"  Saved model bundle → {output_path}")

    # P2-E: write / update model_registry.json with full SHA-256 of this bundle
    try:
        import hashlib as _hl
        with open(output_path, "rb") as _mf:
            _sha256 = _hl.sha256(_mf.read()).hexdigest()
        _reg_path = output_path.parent.parent / "model_registry.json"
        _registry: Dict[str, Any] = {}
        if _reg_path.is_file():
            with open(_reg_path) as _rf:
                _registry = json.load(_rf)
        _registry[str(output_path.name)] = {
            "sha256": _sha256,
            "training_mode": mode_name,
            "sklearn_version": sklearn.__version__,
            "timestamp_iso": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(_reg_path, "w") as _rf:
            json.dump(_registry, _rf, indent=2)
        print(f"  Updated registry   → {_reg_path}")
    except Exception as _e:
        print(f"  [warn] Could not update model registry: {_e}")

    # JSON sidecar — strip numpy OOF arrays before serializing
    _json_skip = {"_oof_proba", "_oof_labels", "_oof_classes", "feature_importances"}
    safe_metrics = {k: v for k, v in metrics.items() if k not in _json_skip}
    safe_metrics["feature_importances"] = metrics.get("feature_importances", {})
    safe_metrics["training_mode"] = mode_name
    safe_metrics["feature_set"] = mode_cfg["feature_set"]
    safe_metrics["deploy_ready"] = mode_cfg["deploy_ready"]
    safe_metrics["notes"] = mode_cfg["notes"]
    metrics_path = output_path.with_suffix(".json")
    with open(metrics_path, "w") as f:
        json.dump(safe_metrics, f, indent=2)
    print(f"  Saved metrics      → {metrics_path}")


# Night classes (subset of ENV_CLASSES) for --reliability-scope night
_NIGHT_CLASS_NAMES = frozenset({"night_clear", "normal_night", "nir_night"})


def plot_reliability_diagrams(
    oof_proba: np.ndarray,
    oof_labels: np.ndarray,
    oof_classes: np.ndarray,
    output_base: Path,
    reliability_scope: str = "night",
) -> Dict[str, float]:
    """Write per-class reliability PNGs and return ECE for each plotted class.

    * ``reliability_scope=night`` — only night_clear, normal_night, nir_night
      (default, backward compatible).
    * ``reliability_scope=all`` — every class in :data:`ENV_CLASSES` present
      in ``oof_classes`` (9 ENV labels).

    Skips a class if matplotlib/sklearn are unavailable, or if it has fewer
    than 20 positive labels (one-vs-rest).

    Args:
        oof_proba:   (N, n_classes) OOF probability matrix from CV folds.
        oof_labels:  (N,) true integer labels aligned with oof_proba rows.
        oof_classes: (n_classes,) class-int array aligned with oof_proba columns.
        output_base: model bundle path; PNGs land beside it as
                     <stem>_reliability_<class_name>.png.
        reliability_scope: \"night\" or \"all\".

    Returns:
        ``{class_name: ECE}`` for each class that was plotted successfully.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve  # type: ignore
    except ImportError:
        print("  [reliability] matplotlib/sklearn unavailable — skipping plots.")
        return {}

    from smartbinocular.feature_schema import (
        ENV_CLASSES,
        ENV_CLASS_TO_INT,
        ENV_INT_TO_CLASS,
    )

    if reliability_scope not in ("night", "all"):
        raise ValueError(
            f"reliability_scope must be 'night' or 'all', got {reliability_scope!r}"
        )
    if reliability_scope == "night":
        allowed_ints = {
            int(ENV_CLASS_TO_INT[n]) for n in _NIGHT_CLASS_NAMES if n in ENV_CLASS_TO_INT
        }
    else:
        allowed_ints = {
            int(ENV_CLASS_TO_INT[n]) for n in ENV_CLASSES if n in ENV_CLASS_TO_INT
        }

    ece_by_class: Dict[str, float] = {}

    for ci, cls_int in enumerate(oof_classes):
        cls_int = int(cls_int)
        if cls_int not in allowed_ints:
            continue
        cls_name = ENV_INT_TO_CLASS.get(cls_int, str(cls_int))
        col_proba = oof_proba[:, ci]
        binary = (oof_labels == cls_int).astype(np.int32)
        n_pos = int(binary.sum())
        if n_pos < 20:
            print(f"  [reliability] {cls_name}: {n_pos} positives < 20 — skipping.")
            continue

        ece = _compute_ece(binary, col_proba, n_bins=10)
        ece_by_class[cls_name] = ece

        try:
            prob_true, prob_pred = calibration_curve(binary, col_proba, n_bins=10, strategy="uniform")
        except Exception as exc:
            print(f"  [reliability] {cls_name}: calibration_curve error: {exc}")
            continue

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
        ax.plot(prob_pred, prob_true, "s-", label=f"{cls_name}\nECE={ece:.4f}")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"Reliability — {cls_name} (OOF, n={n_pos})")
        ax.legend(loc="upper left", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        out_png = output_base.parent / f"{output_base.stem}_reliability_{cls_name}.png"
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        print(f"  [reliability] {cls_name}: ECE={ece:.4f} → {out_png}")

    if ece_by_class:
        tag = "night classes" if reliability_scope == "night" else "all scoped classes"
        print(f"  ECE ({tag}): {ece_by_class}")
    return ece_by_class


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train ENV classifiers from JSONL feature datasets."
    )
    parser.add_argument(
        "--mode", required=True, choices=list(TRAINING_MODES.keys()),
        help="Training mode (see TRAINING_MODES)"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "Input JSONL for training (or for --evaluate-model when --test-dataset is omitted). "
            "Not required if --evaluate-model and --test-dataset are both set."
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output model bundle path (default: mode's output_dir/rf_{mode}.joblib)"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Run cross-validation only; do not save model"
    )
    parser.add_argument(
        "--n-estimators", type=int, default=200,
        help="Number of RF trees (default: 200)"
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Cross-validation folds (default: 5)"
    )
    parser.add_argument(
        "--require-label-source", default=None,
        help="Comma-separated: dataset_original,manual"
    )
    parser.add_argument(
        "--evaluate-model", type=Path, default=None,
        help=(
            "Path to a saved .joblib bundle: load RF+scalers and run held-out metrics "
            "on --dataset (no training), or on --test-dataset if set."
        ),
    )
    parser.add_argument(
        "--test-dataset",
        type=Path,
        default=None,
        help=(
            "With --evaluate-model: load examples only from this JSONL and score all rows "
            "(true held-out test set; no random split). Omit to use --dataset with "
            "--eval-test-size hold-out."
        ),
    )
    parser.add_argument(
        "--eval-test-size",
        type=float,
        default=0.2,
        help="Fraction of the (optionally subsampled) pool held out for testing (default: 0.2).",
    )
    parser.add_argument(
        "--eval-max-samples",
        type=int,
        default=None,
        help="Stratified subsample cap before train/test split (default: use all loaded rows).",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=42,
        help="RNG seed for subsample and hold-out split (default: 42).",
    )
    parser.add_argument(
        "--reliability-scope",
        choices=("night", "all"),
        default="night",
        help=(
            "OOF per-class reliability diagrams: 'night' = three night classes only "
            "(default); 'all' = all ENV_CLASSES. See legacy/md/ML_PER_CLASS_CONFIDENCE_PLAN.md (archived plan)."
        ),
    )
    parser.add_argument(
        "--eval-no-confusion",
        action="store_true",
        help="Skip printing the confusion matrix.",
    )
    args = parser.parse_args()
    if args.dataset is None and not (args.evaluate_model is not None and args.test_dataset is not None):
        parser.error("--dataset is required unless both --evaluate-model and --test-dataset are set")

    mode_cfg = TRAINING_MODES[args.mode]
    print(f"\n{'='*60}")
    print(f"Training mode : {args.mode}")
    print(f"Notes         : {mode_cfg['notes']}")
    if not mode_cfg["deploy_ready"]:
        print(f"⚠️  deploy_ready=False — DO NOT deploy this model.")
    print(f"{'='*60}\n")

    require_sources = None
    if args.require_label_source:
        require_sources = [s.strip() for s in args.require_label_source.split(",")]
    elif mode_cfg.get("preferred_label_sources"):
        # Default: use preferred sources for the mode
        # (weak_heuristic allowed for ablation, not for production)
        preferred = mode_cfg["preferred_label_sources"]
        if "weak_heuristic" not in preferred:
            require_sources = preferred
            print(f"Auto-filtering to label_sources: {require_sources}")

    from smartbinocular.feature_schema import ENV_INT_TO_CLASS

    if args.evaluate_model is not None:
        import joblib  # type: ignore

        eval_path = args.test_dataset if args.test_dataset is not None else args.dataset
        print(f"Loading dataset for evaluation: {eval_path}")
        X, y, nir_groups = load_dataset(eval_path, mode_cfg, require_sources)

        by_class: Dict[int, int] = defaultdict(int)
        for yi in y:
            by_class[yi] += 1
        print("\nClass distribution (eval pool):")
        for cls_int, cnt in sorted(by_class.items()):
            cls_name = ENV_INT_TO_CLASS.get(cls_int, f"unknown({cls_int})")
            print(f"  {cls_name:20s}: {cnt}")

        print(f"\n--evaluate-model: {args.evaluate_model}")
        bundle = joblib.load(args.evaluate_model)
        tm = bundle.get("training_mode")
        if tm != args.mode:
            print(
                f"⚠️  Bundle training_mode={tm!r} differs from --mode={args.mode!r}. "
                "Using CLI mode only for load_dataset filters; ensure they match the trained model."
            )
        if args.test_dataset is None:
            print(
                "\nℹ️  Using --dataset with an internal hold-out split. If this JSONL overlaps "
                "the training file, metrics can be optimistic."
            )
        else:
            print("\nℹ️  Evaluating on --test-dataset only (disjoint hold-out file).")

        evaluate_saved_model(
            bundle,
            X,
            y,
            nir_groups,
            mode_cfg,
            test_size=float(np.clip(args.eval_test_size, 0.05, 0.95)),
            max_samples=args.eval_max_samples,
            random_state=int(args.eval_seed),
            show_confusion=not args.eval_no_confusion,
            held_out_file_only=args.test_dataset is not None,
        )
        print("Done (evaluate only).")
        return

    print(f"Loading dataset: {args.dataset}")
    X, y, nir_groups = load_dataset(args.dataset, mode_cfg, require_sources)

    by_class = defaultdict(int)
    for yi in y:
        by_class[yi] += 1
    print("\nClass distribution:")
    for cls_int, cnt in sorted(by_class.items()):
        cls_name = ENV_INT_TO_CLASS.get(cls_int, f"unknown({cls_int})")
        print(f"  {cls_name:20s}: {cnt}")

    print(f"\nTraining with {len(X)} records, {X.shape[1]} features, {args.cv_folds}-fold CV...\n")

    clf, scalers, metrics = train_and_evaluate(
        X, y, nir_groups, mode_cfg,
        n_estimators=args.n_estimators,
        cv_folds=args.cv_folds,
    )

    cv_score = metrics["cv_balanced_accuracy_mean"]
    if cv_score < 0.60:
        print(f"\n⚠️  WARNING: CV accuracy {cv_score:.3f} < 0.60 (minimum baseline threshold).")
        print("   More data or field collection may be needed before Phase 3C.")
    elif cv_score >= 0.75:
        print(f"\n✓ CV accuracy {cv_score:.3f} ≥ 0.75. Candidate for RPi validation (Phase 3C).")
    else:
        print(f"\n✓ CV accuracy {cv_score:.3f} ≥ 0.60. Baseline accepted.")
        print("  RPi validation needed before production deployment (Phase 3C).")

    if args.eval_only:
        print("\n--eval-only: model not saved.")
        return

    if args.output:
        out_path = args.output
    else:
        out_dir = Path(mode_cfg["output_dir"])
        out_path = out_dir / f"rf_{args.mode}.joblib"

    save_model(out_path, clf, scalers, args.mode, mode_cfg, metrics)

    # P1-D: reliability diagrams from OOF CV predictions
    oof_proba = metrics.get("_oof_proba")
    oof_labels = metrics.get("_oof_labels")
    oof_classes = metrics.get("_oof_classes")
    if oof_proba is not None and oof_labels is not None and oof_classes is not None:
        ece_dict = plot_reliability_diagrams(
            oof_proba,
            oof_labels,
            oof_classes,
            out_path,
            reliability_scope=args.reliability_scope,
        )
        if ece_dict:
            metrics_path = out_path.with_suffix(".json")
            with open(metrics_path) as _f:
                saved = json.load(_f)
            saved["ece_by_class"] = ece_dict
            saved["ece_by_night_class"] = {
                k: v for k, v in ece_dict.items() if k in _NIGHT_CLASS_NAMES
            }
            with open(metrics_path, "w") as _f:
                json.dump(saved, _f, indent=2)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
