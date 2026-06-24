"""Runtime environment-classifier inference (observe-only).

This module provides:
  - MLTop2: immutable dataclass holding top-1 and top-2 class predictions.
  - EnvClassifier: loads a joblib model bundle and runs predictions.
  - MLSharedResult: thread-safe holder for the latest MLTop2 result.
  - MLInferenceThread: daemon thread that reads feature vectors from a queue,
    calls EnvClassifier.predict_top2_safe(), and posts results to MLSharedResult.

INVARIANT: Nothing here mutates :class:`~smartbinocular.env_presets.EnvPresetController`
or display decisions. Predictions are for HUD debug text and JSONL only; rule-based
ENV selection in ``main`` remains authoritative unless explicitly wired otherwise.
"""

from __future__ import annotations

import dataclasses
import logging
import queue
import threading
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from smartbinocular.feature_schema import (
    ENV_CLASS_TO_INT,
    ENV_INT_UNKNOWN,
    FEATURE_SET_OPTICAL_ONLY,
)

logger = logging.getLogger(__name__)

# Type alias for a loaded joblib bundle dict
_Bundle = Dict


@dataclasses.dataclass(frozen=True)
class MLTop2:
    """Immutable snapshot of the top-1 and top-2 RF predictions.

    label_int_* uses ENV_CLASS_TO_INT encoding (1-indexed).
    label_int_2 / proba_2 are ENV_INT_UNKNOWN / 0.0 before the first result
    or when fewer than 2 classes exist in the model.
    Tie-break policy (within predict_top2_safe): sort descending by probability;
    ties broken by descending class-int for determinism.
    """

    label_int_1: int    # top-1 class int
    proba_1: float      # top-1 probability [0, 1]
    label_int_2: int    # top-2 class int (ENV_INT_UNKNOWN if unavailable)
    proba_2: float      # top-2 probability (0.0 if unavailable)


# ── EnvClassifier ─────────────────────────────────────────────────────────────

class EnvClassifier:
    """Wraps a joblib model bundle for single-vector inference.

    Loads on construction. If the model file is missing, unreadable, or its
    feature_set mismatches FEATURE_SET_OPTICAL_ONLY, ``available`` is set to
    False and every predict call returns the fallback label with confidence 0.
    """

    def __init__(self, model_path: str, fallback_label: int = ENV_INT_UNKNOWN) -> None:
        self.model_path = model_path
        self.fallback_label = fallback_label
        self.available: bool = False
        self._rf = None
        self._scalers: Dict = {}
        self._normalize_by: str = "nir_channel"
        self._class_int_to_label: Dict[int, str] = {}
        self._sklearn_version: Optional[str] = None
        self._numpy_version: Optional[str] = None

        self._load()

    def _load(self) -> None:
        try:
            import joblib  # type: ignore
            import sklearn  # type: ignore

            bundle: _Bundle = joblib.load(self.model_path)

            # Validate feature_set matches our expected OPTICAL_ONLY set
            bundle_fs = list(bundle.get("feature_set") or [])
            expected_fs = list(FEATURE_SET_OPTICAL_ONLY)
            if bundle_fs != expected_fs:
                logger.warning(
                    "[ML] feature_set mismatch: bundle has %d features %s, "
                    "expected %d features %s (last feature must be 'nir_blue_mean_ema') "
                    "— inference disabled; retrain with tools/offline_pipeline.py",
                    len(bundle_fs), bundle_fs,
                    len(expected_fs), expected_fs,
                )
                return

            self._rf = bundle["rf"]
            self._scalers = bundle.get("scalers") or {}
            self._normalize_by = str(bundle.get("normalize_by", "nir_channel"))
            self._class_int_to_label = bundle.get("class_int_to_label") or {}
            self._sklearn_version = bundle.get("sklearn_version")
            self._numpy_version = bundle.get("numpy_version")

            # Warn on sklearn version mismatch — still load; RF is generally tolerant
            if self._sklearn_version and self._sklearn_version != sklearn.__version__:
                warnings.warn(
                    f"[ML] sklearn version mismatch: model trained with "
                    f"{self._sklearn_version}, runtime is {sklearn.__version__}. "
                    "Predictions may differ slightly.",
                    UserWarning,
                    stacklevel=2,
                )

            self.available = True
            logger.info(
                "[ML] Loaded model from %s | sklearn=%s | features=%d",
                self.model_path,
                self._sklearn_version,
                len(expected_fs),
            )

            # Registry SHA-256 check: warn-only — never blocks load; mismatches flag stale deployments.
            try:
                import hashlib as _hl, json as _json, os as _os, pathlib as _pl
                _bundle_path = _pl.Path(self.model_path)
                _reg_path = _bundle_path.parent.parent / "model_registry.json"
                if _reg_path.is_file():
                    with open(_reg_path) as _rf:
                        _registry = _json.load(_rf)
                    _expected = (_registry.get(_bundle_path.name) or {}).get("sha256")
                    if _expected:
                        with open(_bundle_path, "rb") as _mf:
                            _actual = _hl.sha256(_mf.read()).hexdigest()
                        if _actual != _expected:
                            logger.warning(
                                "[ML] SHA-256 mismatch for %s — expected %s…, got %s… "
                                "(bundle may have been modified; inference continues)",
                                _bundle_path.name, _expected[:12], _actual[:12],
                            )
                        else:
                            logger.info("[ML] SHA-256 verified for %s", _bundle_path.name)
            except Exception as _e:
                logger.debug("[ML] SHA-256 registry check skipped: %s", _e)

        except FileNotFoundError:
            logger.warning("[ML] Model file not found: %s — inference disabled", self.model_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[ML] Failed to load model from %s: %s — inference disabled", self.model_path, exc)

    def predict(
        self, feature_vec: np.ndarray
    ) -> Tuple[int, float, Dict[int, float]]:
        """Run inference on a single feature vector.

        Args:
            feature_vec: 1-D float32 array of length len(FEATURE_SET_OPTICAL_ONLY).

        Returns:
            (label_int, confidence, proba_dict)
            - label_int: predicted ENV_CLASS_TO_INT value (1-indexed), or fallback
            - confidence: max class probability [0.0, 1.0]
            - proba_dict: {label_int: probability} for all RF classes

        Raises:
            RuntimeError: if not available (caller should use predict_top2_safe instead)
        """
        if not self.available or self._rf is None:
            raise RuntimeError("EnvClassifier not available — model not loaded")

        vec_2d = feature_vec.reshape(1, -1)

        # Apply scalers: try the "nir" group first (live RPi produces nir channel),
        # fall back to "__all__" for fixed-scaler bundles.
        if self._normalize_by == "nir_channel":
            scaler = self._scalers.get("nir") or self._scalers.get("rgb") or self._scalers.get("__all__")
        else:
            scaler = self._scalers.get("__all__")

        if scaler is not None:
            vec_2d = scaler.transform(vec_2d)

        label_arr = self._rf.predict(vec_2d)
        proba_arr = self._rf.predict_proba(vec_2d)  # shape (1, n_classes)

        label_int = int(label_arr[0])
        proba_flat = proba_arr[0]
        confidence = float(np.max(proba_flat))

        proba_dict: Dict[int, float] = {
            int(cls): float(p)
            for cls, p in zip(self._rf.classes_, proba_flat)
        }

        return label_int, confidence, proba_dict

    def predict_top2_safe(self, feature_vec: np.ndarray) -> "MLTop2":
        """Return an MLTop2 with top-1 and top-2 RF predictions. Never raises.

        Reuses the single predict() call (no extra predict_proba invocation).
        On any error or when not available, returns
        MLTop2(fallback_label, 0.0, ENV_INT_UNKNOWN, 0.0).
        """
        try:
            label_int, confidence, proba_dict = self.predict(feature_vec)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[ML] predict_top2_safe caught exception: %s", exc)
            return MLTop2(self.fallback_label, 0.0, ENV_INT_UNKNOWN, 0.0)

        # Sort by (proba_desc, class_int_desc) for deterministic tie-breaking.
        sorted_pairs = sorted(
            proba_dict.items(),
            key=lambda kv: (kv[1], kv[0]),
            reverse=True,
        )
        if len(sorted_pairs) >= 2:
            top2_cls, top2_p = sorted_pairs[1]
        else:
            top2_cls, top2_p = ENV_INT_UNKNOWN, 0.0

        return MLTop2(
            label_int_1=label_int,
            proba_1=confidence,
            label_int_2=int(top2_cls),
            proba_2=float(top2_p),
        )

    def label_name(self, label_int: int) -> str:
        """Return the string class name for a label int, or 'unknown'."""
        return self._class_int_to_label.get(label_int, f"unknown({label_int})")


# ── MLPosteriorEMA ────────────────────────────────────────────────────────────

def _resolve_asym(
    asym_str: Dict[str, Any],
    classes: Sequence[int],
) -> Dict[int, Tuple[float, float]]:
    """Resolve string class-name keys to ints; filter to those present in classes.

    Args:
        asym_str: {"glare": [0.85, 0.45], ...} from CONFIG (JSON-safe string keys).
        classes:  rf.classes_ (ints) from the loaded bundle.

    Returns a dict {class_int: (alpha_up, alpha_down)} limited to classes present
    in the bundle.  Unknown names log DEBUG and are dropped silently.
    """
    class_set = set(int(c) for c in classes)
    result: Dict[int, Tuple[float, float]] = {}
    for name, pair in asym_str.items():
        cls_int = ENV_CLASS_TO_INT.get(str(name))
        if cls_int is None:
            logger.debug("[ML] posterior-ema: unknown class name %r in asym config — skipped", name)
            continue
        if cls_int not in class_set:
            logger.debug(
                "[ML] posterior-ema: class %r (int=%d) absent from bundle classes_; "
                "falling back to general alpha",
                name, cls_int,
            )
            continue
        alpha_up = float(pair[0])
        alpha_down = float(pair[1])
        result[cls_int] = (alpha_up, alpha_down)
    return result


class MLPosteriorEMA:
    """Exponential moving average over the full RF posterior probability vector.

    Smooths the per-class probability distribution (not just top-1 confidence)
    so that MLTop2 top-1/top-2 ordering is derived from a temporally stable
    distribution rather than a single noisy frame.

    Args:
        classes:  Ordered sequence of class ints (from rf.classes_). Defines the
                  support of the distribution.
        alpha:    General smoothing factor (0 < α ≤ 1). α=1.0 → passthrough (no smoothing).
        asym:     Per-class asymmetric alphas {class_int: (alpha_up, alpha_down)}.
                  Keys MUST be in classes (use _resolve_asym to pre-filter).
                  alpha_up applied when p_new > p_prev; alpha_down otherwise.

    Thread-safety: NOT thread-safe. Owned exclusively by MLInferenceThread.
    """

    def __init__(
        self,
        classes: Sequence[int],
        alpha: float = 0.55,
        asym: Optional[Dict[int, Tuple[float, float]]] = None,
    ) -> None:
        self._classes: List[int] = [int(c) for c in classes]
        self._alpha = float(alpha)
        self._asym: Dict[int, Tuple[float, float]] = dict(asym) if asym else {}
        self._prev: Optional[Dict[int, float]] = None  # seeded on first call

    def update(self, proba: Dict[int, float]) -> Dict[int, float]:
        """Apply EMA to the full posterior vector; return a new renormalized dict.

        First call: seed _prev and return proba unchanged.
        Subsequent calls: per-class EMA with optional asymmetric alpha; renormalize.
        """
        if self._prev is None:
            # Seed: return input as-is (no smoothing on first frame).
            self._prev = dict(proba)
            return dict(proba)

        smoothed: Dict[int, float] = {}
        for c in self._classes:
            p_new = proba.get(c, 0.0)
            p_prev = self._prev.get(c, 0.0)
            if c in self._asym:
                a = self._asym[c][0] if p_new > p_prev else self._asym[c][1]
            else:
                a = self._alpha
            smoothed[c] = a * p_new + (1.0 - a) * p_prev

        # Renormalize to keep Σ=1 (numeric drift after many frames).
        total = sum(smoothed.values())
        if total > 0.0:
            smoothed = {c: v / total for c, v in smoothed.items()}

        self._prev = smoothed
        return smoothed


# ── MLSharedResult ─────────────────────────────────────────────────────────────

class MLSharedResult:
    """Thread-safe holder for the latest MLTop2 prediction.

    get() never blocks: returns a default MLTop2 (all unknowns) before the
    first set(). The stored value is an immutable MLTop2, so readers always
    see a consistent snapshot without partial updates.

    Also stores the latest epistemic variance (mean of per-class tree variance
    across the RF ensemble) in a separate lock slot so the hot set() / get()
    path for MLTop2 is not serialized with the slower epistemic update.
    """

    _DEFAULT: MLTop2 = MLTop2(ENV_INT_UNKNOWN, 0.0, ENV_INT_UNKNOWN, 0.0)

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._result: MLTop2 = self._DEFAULT
        self._evar_lock = threading.Lock()
        self._epistemic_var: Optional[float] = None

    def set(self, result: MLTop2) -> None:
        with self._lock:
            self._result = result

    def get(self) -> MLTop2:
        with self._lock:
            return self._result

    def set_epistemic_var(self, var: float) -> None:
        with self._evar_lock:
            self._epistemic_var = float(var)

    def get_epistemic_var(self) -> Optional[float]:
        with self._evar_lock:
            return self._epistemic_var


# ── MLInferenceThread ──────────────────────────────────────────────────────────

class MLInferenceThread(threading.Thread):
    """Daemon thread that drains an inference queue and posts results.

    Queue items are 1-D numpy float32 arrays (feature vectors).
    The thread calls EnvClassifier.predict_top2_safe() on each vector and posts
    the resulting MLTop2 to MLSharedResult.

    maxsize=2: if inference is slower than dispatch, old vectors are dropped
    by the dispatcher (put_nowait raises queue.Full, which the caller catches).
    This ensures the shared result always reflects recent data.
    """

    def __init__(
        self,
        classifier: EnvClassifier,
        shared_result: MLSharedResult,
        input_queue: "queue.Queue[np.ndarray]",
        *,
        ema_alpha: float = 0.55,
        ema_asym: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            ema_alpha:  General EMA smoothing factor (0 < α ≤ 1); 1.0 = passthrough.
            ema_asym:   Per-class asymmetric alphas as JSON-safe string dict,
                        e.g. {"glare": [0.85, 0.45]}.  Keys are resolved to ints
                        via ENV_CLASS_TO_INT and filtered to rf.classes_ on first run.
        """
        super().__init__(name="MLInferenceThread", daemon=True)
        self._classifier = classifier
        self._shared_result = shared_result
        self._queue = input_queue
        self._stop_event = threading.Event()
        self._ema_alpha = float(ema_alpha)
        self._ema_asym_str: Dict[str, Any] = dict(ema_asym) if ema_asym else {}

    def stop(self) -> None:
        """Signal the thread to exit after the current item."""
        self._stop_event.set()

    def run(self) -> None:
        logger.debug("[ML] Inference thread started (ema_alpha=%.2f)", self._ema_alpha)

        # Construct EMA using the loaded bundle's rf.classes_ so asym keys are filtered
        # to the actual trained classes.
        ema: Optional[MLPosteriorEMA] = None
        if self._classifier.available and self._classifier._rf is not None:
            classes = list(self._classifier._rf.classes_)
            asym_int = _resolve_asym(self._ema_asym_str, classes)
            ema = MLPosteriorEMA(classes, alpha=self._ema_alpha, asym=asym_int)

        while not self._stop_event.is_set():
            try:
                vec = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if ema is not None and self._classifier.available:
                try:
                    _, _, proba_dict = self._classifier.predict(vec)
                    smoothed = ema.update(proba_dict)
                    # Derive MLTop2 from smoothed distribution with deterministic tie-break.
                    sorted_pairs = sorted(
                        smoothed.items(),
                        key=lambda kv: (kv[1], kv[0]),
                        reverse=True,
                    )
                    top1_cls, top1_p = sorted_pairs[0]
                    if len(sorted_pairs) >= 2:
                        top2_cls, top2_p = sorted_pairs[1]
                    else:
                        top2_cls, top2_p = ENV_INT_UNKNOWN, 0.0
                    result = MLTop2(
                        label_int_1=int(top1_cls),
                        proba_1=float(top1_p),
                        label_int_2=int(top2_cls),
                        proba_2=float(top2_p),
                    )
                    # Epistemic variance: mean of per-class tree variance across the RF ensemble.
                    # CalibratedClassifierCV path: calibrated_classifiers_[0].estimator.estimators_
                    # Bare RF fallback: _rf.estimators_ directly.
                    try:
                        _rf = self._classifier._rf
                        if hasattr(_rf, "calibrated_classifiers_"):
                            _base_rf = _rf.calibrated_classifiers_[0].estimator
                        else:
                            _base_rf = _rf
                        if hasattr(_base_rf, "estimators_") and len(_base_rf.estimators_) > 1:
                            import numpy as _np
                            _tree_probas = _np.array([
                                t.predict_proba(vec.reshape(1, -1))[0]
                                for t in _base_rf.estimators_
                            ])  # shape (n_trees, n_classes)
                            _evar = float(_np.mean(_np.var(_tree_probas, axis=0)))
                            self._shared_result.set_epistemic_var(_evar)
                    except Exception:  # noqa: BLE001
                        pass  # epistemic variance is best-effort; never block inference
                except Exception as exc:  # noqa: BLE001
                    logger.debug("[ML] EMA inference error: %s", exc)
                    result = MLTop2(self._classifier.fallback_label, 0.0, ENV_INT_UNKNOWN, 0.0)
            else:
                result = self._classifier.predict_top2_safe(vec)

            self._shared_result.set(result)
        logger.debug("[ML] Inference thread exiting")
