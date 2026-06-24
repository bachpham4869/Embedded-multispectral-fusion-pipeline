# ML Hybrid System Plan — SmartBinocular Night-Vision Pipeline
**Version:** 1.1  
**Date:** 2026-04-08  
**Target hardware:** Raspberry Pi 4B (4GB, quad-core ARM Cortex-A72 @ 1.8GHz, no GPU/NPU)  
**Status:** Development guide — actionable, phase-by-phase

---

> **Phase 3 restructured** — Tasks 3.6/3.7/3.9 replaced by dataset-first offline pipeline.
> See **`OFFLINE_ML_PLAN.md`** (this directory) v3.0 for the full optical-first ML architecture,
> feature schema, dataset pipeline, training strategy, and implementation plan.

---

## Table of Contents

1. [Overview](#overview)
2. [System Constraints](#system-constraints)
3. [Research Findings & Adaptations](#research-findings--adaptations)
4. [Proposed Pipeline Improvements](#proposed-pipeline-improvements)
5. [Hybrid System Design](#hybrid-system-design)
6. [ML Development Plan](#ml-development-plan)
7. [Detailed Execution Roadmap](#detailed-execution-roadmap)
8. [Notes for Future Expansion](#notes-for-future-expansion)
9. [Execution Environment & Logging Strategy](#execution-environment--logging-strategy)

---

## 1. Overview

This document is a full development guide for elevating the SmartBinocular pipeline from a rule-based embedded system to a research-grade hybrid rule+ML system — while remaining real-time on Raspberry Pi 4 (CPU-only).

**Current state:**  
- Modular Python pipeline: NIR → Thermal → Fusion → Display  
- Rule-based environment classification (14 ENV presets)  
- Frame-level stabilization (IIR, Schmitt trigger, 3DNR)  
- No ML component  

**Target state:**  
- Same real-time performance (≥20 FPS at 640×480)  
- Hybrid rule+ML environment classification  
- Improved fusion quality (gradient/pyramid-domain)  
- Lightweight anomaly detection (E1 upgrade)  
- Integrity/anti-tampering layer  
- Full data logging infrastructure for continuous ML improvement  

**Design philosophy:**  
> Rules for speed and safety. ML for generalization and robustness. Never block the frame loop.

---

## 2. System Constraints

| Resource | Limit | Design implication |
| --- | --- | --- |
| CPU | 4× ARM A72 @ 1.8GHz, NEON SIMD | Use OpenCV NEON paths; avoid Python loops on arrays |
| RAM | 4GB | Keep working set < 512MB; avoid buffering full-res sequences |
| Frame rate | ≥20 FPS target | Budget: ≤50ms per frame total |
| NIR resolution | 640×480 | All processing at native or 320×240 downscale |
| Thermal resolution | 80×62 (MI48) | Trivially cheap to process |
| ML inference budget | ≤2ms on-frame, or background thread | Must not block frame loop |
| Storage (logging) | SD card, slow writes | Log async; batch flush; use binary format |
| Power | USB-C 5V/3A | No active cooling expected; thermal throttle risk |

**Hard constraints (never violate):**
- No dense CNN inference in the frame loop  
- No dynamic memory allocation in hot path  
- All ML inference either: (a) <2ms in-frame, or (b) background thread with shared state  

---

## 3. Research Findings & Adaptations

Each entry: original idea → adaptation for this system → where it fits → cost → benefit.

---

### R1 — Laplacian Pyramid Fusion (Multi-scale NIR + Thermal)

**Original idea:** Multi-scale image fusion using Gaussian/Laplacian pyramids (Burt & Adelson 1983). Used in HDR, medical imaging, and multi-modal fusion.

**Adaptation for this system:**  
Use a 3-level Laplacian pyramid with `cv2.pyrDown` / `cv2.pyrUp`. Build pyramids for NIR and warped thermal. At each level, fuse using a saliency-weighted average (thermal weight = thermal activity mask, NIR weight = NIR sharpness map). Collapse and clip. Replace the current hard fg_mask blend.

```
nir_pyr   = laplacian_pyramid(nir_enhanced, levels=3)
thm_pyr   = laplacian_pyramid(thermal_warped, levels=3)
w_nir     = sharpness_map(nir_enhanced)        # Laplacian magnitude, normalized
w_thm     = activity_map(thermal_warped)       # std in 5×5 window, normalized
fused_pyr = [blend(n, t, wn, wt) for n,t,wn,wt in zip(nir_pyr, thm_pyr, w_nir_pyr, w_thm_pyr)]
result    = collapse(fused_pyr)
```

**Where it fits:** `nir_pipeline.py` / `main.py` fusion path (mode 3), replacing `alpha * thermal_warped + (1-alpha) * nir_enhanced`.

**Cost:** Medium (~8ms additional at 640×480, can reduce to ~4ms at 320×240 pyramid base)  
**Benefit:** Significantly better edge preservation, no halo artifacts at fg_mask boundary, natural multi-scale detail blending.

---

### R2 — Guided Filter for NIR Enhancement (Replaces Bilateral)

**Original idea:** Guided image filter (He et al., 2013). Edge-preserving smoothing in O(N) time, faster than bilateral filter.

**Adaptation:**  
Use thermal (upsampled 80→640) as the *guide image* and NIR as the *input*. The guided filter will transfer thermal edge structure into the NIR smoothing — useful for denoising NIR while preserving thermally-relevant boundaries (e.g., warm body edges in cold background).

```python
# In nir_pipeline.py
guide = cv2.resize(thermal_gray, (640, 480))   # upsampled thermal
nir_smoothed = cv2.ximgproc.guidedFilter(
    guide=guide, src=nir_gray, radius=8, eps=0.01**2
)
```

**Where it fits:** `nir_pipeline.py` — inside `HybridNIREnhancer`, before CLAHE.  
**Cost:** Low (~2ms, O(N) complexity, uses OpenCV `ximgproc`)  
**Benefit:** Better noise suppression in dark NIR, thermal-edge-aligned smoothing, no bilateral filter tuning needed.

---

### R3 — Adaptive CLAHE Parameterization

**Original idea:** Histogram entropy as a measure of local contrast quality (standard signal processing).

**Adaptation:**  
Compute the entropy of the NIR histogram each frame (fast: `np.histogram` + log sum). Map entropy → CLAHE clip limit dynamically:
- Low entropy (flat histogram, foggy/uniform): high clip limit (3.0–4.0) — aggressive enhancement
- High entropy (rich histogram, good visibility): low clip limit (1.0–1.5) — gentle enhancement
- Use EMA smoothing on the clip limit to avoid rapid oscillation

```python
hist, _ = np.histogram(nir_gray_8u, bins=64, range=(0,256))
hist_norm = hist / hist.sum()
entropy = -np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0]))
clip_limit = np.clip(EMA(4.5 - 0.35 * entropy), 1.0, 4.0)
```

**Where it fits:** `nir_pipeline.py` — `HybridNIREnhancer._apply_clahe()`.  
**Cost:** Very low (<0.5ms)  
**Benefit:** Removes need to manually tune ENV presets for CLAHE; adapts per frame.

---

### R4 — Sparse Optical Flow for Motion Detection (Replaces Frame Difference)

**Original idea:** Lucas-Kanade sparse optical flow on feature points (Shi-Tomasi corners).

**Adaptation:**  
Track 20–30 Shi-Tomasi corners on downscaled NIR (160×120). LK flow gives velocity vectors per point. Derive:
- `motion_magnitude` = median flow vector length
- `motion_direction` = circular mean of flow angles (detects pan vs. shake vs. stationary)
- `jerk` = frame-to-frame change in motion_magnitude (replaces D2 JerkGate)

Use background thread to refresh corner set every 15 frames. Maintains point tracks in foreground between refreshes.

```python
# 160×120 downscale, ~1ms for LK on 20 points
pts_new, status, _ = cv2.calcOpticalFlowPyrLK(
    prev_small, curr_small, pts_prev, None,
    winSize=(11,11), maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS, 10, 0.03)
)
```

**Where it fits:** `motion.py` — replaces `D2JerkGate` frame difference.  
**Cost:** Low (~2ms on 160×120 with 20 points)  
**Benefit:** Direction-aware motion, distinguishes camera shake from scene motion, feeds better features to ML classifier.

---

### R5 — Kalman Filter for Thermal Background Model

**Original idea:** Kalman filter as principled alternative to EMA for signal tracking.

**Adaptation:**  
Replace the current EMA-based background model in `thermal_pipeline.py` with a per-pixel 1D Kalman filter (scalar state = background temperature, scalar observation = current pixel value). Use pixel-level `process_noise` and `measurement_noise` estimated from a warmup period.

Key advantage over EMA: Kalman propagates uncertainty. Pixels with high variance get updated faster; stable background pixels are protected. This naturally handles the warmup problem — Kalman initializes with high uncertainty and converges.

For RPi4 feasibility: operate on the 80×62 thermal image only (4960 pixels). Full Kalman update at 80×62 costs ~0.5ms in vectorized numpy.

```python
# Vectorized across all pixels
K = P / (P + R)            # Kalman gain
x = x + K * (z - x)       # state update (background estimate)
P = (1 - K) * P            # covariance update
```

**Where it fits:** `thermal_pipeline.py` — `ThermalProcessor._update_background()`.  
**Cost:** Very low (~0.5ms at 80×62)  
**Benefit:** Better fg/bg separation, principled uncertainty, faster convergence, no warmup counter needed.

---

### R6 — Perceptual Hash Integrity Layer (Anti-Tampering)

**Original idea:** dHash / pHash for perceptual image fingerprinting. Used in digital forensics.

**Adaptation:**  
For each captured frame (on `S` key), compute a 64-bit dHash of the output frame and embed it in the capture JSON alongside a HMAC-SHA256 signature derived from a session key (loaded from a device-local secret file at startup). This provides:
1. Tamper detection: verify dHash matches stored hash
2. Chain integrity: each JSON includes the hash of the previous capture JSON (Merkle-style chain)
3. Session authenticity: HMAC ties captures to the physical device

```python
import hashlib, hmac

def dhash(img: np.ndarray, size: int = 8) -> int:
    gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (size+1, size))
    diff = gray[:, 1:] > gray[:, :-1]
    return sum(b << i for i, b in enumerate(diff.flatten()))

def sign_capture(meta: dict, key: bytes) -> str:
    payload = json.dumps(meta, sort_keys=True).encode()
    return hmac.new(key, payload, hashlib.sha256).hexdigest()
```

**Where it fits:** `utils.py` + `main.py` capture path (`_write_capture_meta`).  
**Cost:** Negligible (<0.1ms)  
**Benefit:** Military-grade evidence chain; tamper-evident captures; session provenance.

---

### R7 — Statistical Anomaly Detection on Thermal (E1 Upgrade)

**Original idea:** Robust z-score anomaly detection using median absolute deviation (MAD), as used in robust statistics (Leys et al., 2013).

**Adaptation:**  
Replace the current local-stats E1 detector with MAD-based anomaly scoring on the thermal foreground pixels:

```python
fg_pixels = thermal_normalized[fg_mask > 0]
if len(fg_pixels) < 5:
    return 0.0
med = np.median(fg_pixels)
mad = np.median(np.abs(fg_pixels - med))
modified_z = 0.6745 * (fg_pixels - med) / (mad + 1e-6)
anomaly_score = float(np.mean(modified_z > 3.5))  # fraction of outlier pixels
```

Add a temporal integrator: anomaly event fires only if `anomaly_score > threshold` for 3 consecutive frames (prevents single-frame noise triggers).

**Where it fits:** `main.py` E1 block → can be promoted to `thermal_pipeline.py`.  
**Cost:** Very low (<0.3ms on 80×62)  
**Benefit:** More robust than local mean/std detector, scale-invariant, handles asymmetric thermal thermal distributions.

---

### R8 — Random Forest Environment Classifier (Core ML Component)

**Original idea:** Ensemble decision trees for tabular/feature-based classification.

**Adaptation:**  
Extract ~22 scalar features from each frame (see Section 5 for full feature list). Train a Random Forest (50–100 trees, max_depth=8) offline using logged data. Deploy via `scikit-learn` (or `joblib`-serialized model) with inference every 30 frames (~1 second at 30fps) in a background thread.

The classifier outputs one of 8 macro-ENV classes (night/fog/glare/backlight/normal_day/normal_night/rain_sim/transition). This replaces/supplements the rule-based ENV_PRESETS controller.

**Inference cost on RPi4:**  
`sklearn` RandomForestClassifier (100 trees, 23 features) → ~0.8ms per call. Entirely feasible in-thread or background.

**Where it fits:** New `ml_classifier.py` module, called from `env_presets.py` `EnvPresetController.tick()`.  
**Cost:** Low (inference), Medium (data collection + training)  
**Benefit:** Handles corner cases that rules miss, learns from real deployment data, explainable (feature importance).

---

### R9 — Temporal Consistency via Sequence Voting

**Original idea:** Majority voting over a sliding window to improve stability of discrete classifiers.

**Adaptation:**  
Maintain a deque of the last 5 ML environment predictions. The active ENV is the majority-vote winner. A new ENV is only adopted if it wins ≥3/5 consecutive votes AND the rule-based system does not veto it (rule veto = hard safety fallback).

This eliminates rapid ENV oscillation without requiring longer hysteresis timers.

**Where it fits:** `env_presets.py` — `EnvPresetController` state machine.  
**Cost:** Negligible  
**Benefit:** Stable ENV transitions, no single-frame mis-classification affects display.

---

### R10 — Frame Quality Score for Capture Filtering

**Original idea:** No-reference image quality metrics (Brenner sharpness, variance of Laplacian).

**Adaptation:**  
Before saving a captured frame, compute:
- `sharpness = variance_of_laplacian(gray_downscaled_to_160x120)`
- `exposure_ok = 20 < mean_brightness < 220`
- `thermal_ok = thermal_std > 1.5` (not isothermal/dead sensor)

Combine into a scalar quality score 0–1. Log score in capture JSON. Later: use as training data weight (low-quality frames → lower weight in ML training).

**Where it fits:** `utils.py` + capture path in `main.py`.  
**Cost:** Negligible  
**Benefit:** Better training data quality, automated capture quality feedback.

---

## 4. Proposed Pipeline Improvements

### 4.1 Identified Weak Points

| Location | Weak point | Root cause |
| --- | --- | --- |
| `main.py` fusion loop | Hard fg_mask boundary artifacts | Binary mask, no spatial blending transition |
| `motion.py` D2JerkGate | Frame difference sensitive to sensor noise | Pixel diff not motion-aware |
| `thermal_pipeline.py` background model | EMA has no uncertainty model | Can't distinguish fast convergence from instability |
| `env_presets.py` ENV classification | 14 rules hand-coded, brittle to edge cases | No learned generalization |
| `nir_pipeline.py` CLAHE | Fixed clip limit per ENV preset | Not per-frame adaptive |
| `main.py` glare | Dual glare eval paths (display + ENV block) | Cached in refactor but logic is duplicated |
| `metrics.py` | No per-frame quality scores | Can't filter training data by quality |

### 4.2 Concrete Fixes

**Fix 1 — Soft fusion mask (replaces hard fg_mask)**  
```python
# Current (hard):
fused = nir * (1 - fg_mask_f) + thermal_colored * fg_mask_f * alpha

# Improved (Gaussian-blurred soft mask):
soft_mask = cv2.GaussianBlur(fg_mask_f[:,:,np.newaxis], (15,15), 0)
fused = nir * (1 - soft_mask) + thermal_colored * soft_mask * alpha
```
Cost: 1ms. Eliminates hard edge artifact at body/background boundary.

**Fix 2 — Single glare path**  
Extract `nir_glare_score` once per frame in `nir_pipeline.py`, cache on the frame state object. All downstream consumers (display grade, ENV auto-rule, HUD) read the cache. Remove all secondary recomputations.

**Fix 3 — ENV classification rate limit**  
ENV auto-rule currently runs every frame. Change to: run full ENV feature extraction every 10 frames (300ms at 30fps), apply result with sequence voting (R9). Saves ~3ms/frame of repeated computation.

**Fix 4 — Thermal background: EMA → Kalman (R5)**  
Direct replacement in `ThermalProcessor`. No interface change. Removes warmup counter and frame-skip logic.

**Fix 5 — Merge downscale operations**  
Several modules independently downscale NIR to 320×240 or 160×120. Add a `FrameCache` dataclass to `utils.py` that holds `{nir_full, nir_320, nir_160, nir_gray, thermal_80}` and is built once per frame at the top of the main loop. Pass it down to all processors.

```python
@dataclasses.dataclass
class FrameCache:
    nir_full:    np.ndarray   # 640×480 BGR
    nir_320:     np.ndarray   # 320×240 BGR  (pyrDown once)
    nir_160:     np.ndarray   # 160×120 BGR  (pyrDown twice)
    nir_gray:    np.ndarray   # 640×480 GRAY (single BGR2GRAY)
    thermal_80:  np.ndarray   # 80×62 float32 raw
    ts:          float
```

**Fix 6 — Deterministic mode switching**  
Current: mode switch on keypress takes effect next frame. Proposed: add a `pending_mode` field; commit on frame boundary after completing current frame. Prevents partial-frame state mix.

---

## 5. Hybrid System Design

### 5.1 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     FRAME LOOP (≤50ms)                  │
│                                                         │
│  [Capture] → [FrameCache] → [NIR proc] → [Thermal proc]│
│                    ↓                                    │
│          [Rule-based fast path]   ←─ ENV rules          │
│          (every frame, <1ms)         14 presets         │
│                    ↓                                    │
│          [Fusion + Display]                             │
│                    ↓                                    │
│          [HUD + imshow]                                 │
└─────────────────────────────────────────────────────────┘
                         ↕ shared state (thread-safe)
┌─────────────────────────────────────────────────────────┐
│              ML SLOW PATH (background thread)            │
│                                                         │
│  Every 30 frames (~1s):                                 │
│    1. Read FrameCache snapshot                          │
│    2. Extract 23 features                               │
│    3. RandomForest.predict() → ENV class + confidence   │
│    4. Write to shared MLState (atomic)                  │
│    5. Log feature vector to JSONL                       │
└─────────────────────────────────────────────────────────┘
```

### 5.2 Feature Vector (23 features)

All features are scalar, computed in <1ms total from `FrameCache`.

| # | Feature | Source | Notes |
| --- | --- | --- | --- |
| 1 | `nir_mean_brightness` | NIR gray 160×120 | EMA-smoothed |
| 2 | `nir_std` | NIR gray 160×120 | Texture richness |
| 3 | `nir_entropy` | NIR histogram 64-bin | R3 signal |
| 4 | `nir_p95` | NIR gray | High-percentile brightness |
| 5 | `nir_glare_score` | NIR p95 > threshold | 0–1 normalized |
| 6 | `nir_sharpness` | Variance of Laplacian, 160×120 | R10 |
| 7 | `nir_dark_fraction` | Fraction pixels < 20 | Night indicator |
| 8 | `nir_saturation_mean` | HSV S on small BGR | Colorfulness vs near-gray |
| 9 | `thm_mean` | Thermal 80×62 | Mean scene temperature |
| 10 | `thm_std` | Thermal 80×62 | Thermal texture |
| 11 | `thm_max` | Thermal 80×62 | Hot spot |
| 12 | `thm_p95_p05_delta` | Thermal dynamic range | Scene contrast |
| 13 | `thm_fg_fraction` | Kalman fg_mask area | Object presence |
| 14 | `thm_anomaly_score` | R7 MAD score | E1 signal |
| 15 | `motion_magnitude` | LK flow or JerkGate | R4 |
| 16 | `motion_jerk` | Delta of motion_magnitude | Camera shake |
| 17 | `skew_ms` | NIR/thermal timestamp delta | S6 signal |
| 18 | `fusion_alpha` | Current blend alpha | User state |
| 19 | `nir_brightness_delta_10f` | Mean change over 10 frames | Temporal trend |
| 20 | `thm_mean_delta_10f` | Thermal mean change over 10 frames | Temporal trend |
| 21 | `hour_of_day_sin` | sin(2π × hour/24) | Cyclic time encoding |
| 22 | `hour_of_day_cos` | cos(2π × hour/24) | Cyclic time encoding |
| 23 | `prev_env_class` | Last committed ENV | Transition regularization |

### 5.3 ML Model Specification

```
Model:         RandomForestClassifier
n_estimators:  100
max_depth:     8
min_samples_leaf: 5
class_weight:  balanced
Features:      23 scalar floats
Output:        8-class ENV (night/fog/glare/backlight/day/night_clear/rain/transition)
               + class probability vector (for confidence threshold)
Inference:     ~0.8ms on RPi4 (sklearn, single sample)
Serialization: joblib (model.pkl, ~500KB)
```

### 5.4 Decision Fusion Strategy

```
rule_env    = current rule-based ENV  (always available)
ml_env      = ML prediction (available every 30 frames, via shared state)
ml_conf     = ML confidence (max class probability)
vote_window = deque of last 5 ML predictions

DECISION LOGIC:
if ml_conf < 0.65:
    # ML uncertain → trust rules
    active_env = rule_env

elif rule_env == ml_env:
    # Full agreement → fast commit
    active_env = ml_env

elif rule_env in SAFETY_OVERRIDE_SET:
    # Rule detected critical condition (glare_heavy, sensor_fault)
    # Rules always win on safety-critical ENVs
    active_env = rule_env

elif majority_vote(vote_window) == ml_env and vote_count >= 3:
    # ML consistently disagrees → trust ML
    active_env = ml_env
    log_rule_override(rule_env, ml_env, ml_conf)

else:
    # Unstable → hold previous ENV
    active_env = prev_active_env
```

### 5.5 Hysteresis & Stability

- ENV change only committed if stable for ≥3 consecutive ML votes (R9)
- ENV transition has a 2-second cooldown (no second transition within cooldown)
- Rule-based SAFETY_OVERRIDE_SET = `{glare_heavy, sensor_fault, backlight_extreme}` — rules always win here
- All overrides are logged to `ml_log/overrides_<date>.jsonl` for analysis

---

## 6. ML Development Plan

### Stage 1 — Data Collection Infrastructure

**Goal:** Build a logging system that captures feature vectors with timestamps during real operation.

**Implementation:**  
Add `MLLogger` class to `utils.py`:
```python
class MLLogger:
    def __init__(self, log_dir: str):
        self._file = open(os.path.join(log_dir, f"features_{ts()}.jsonl"), "a")

    def log(self, features: dict, label: str = ""):
        entry = {**features, "label": label, "ts": time.time()}
        self._file.write(json.dumps(entry) + "\n")

    def flush(self):
        self._file.flush()  # call every 60s
```

Log every 5 frames. Each entry = full 23-feature vector + optional label + timestamp.

**Output:** `logs/ml/features_YYYYMMDD_HHMMSS.jsonl`

---

### Stage 2 — Feature Extraction Validation

**Goal:** Verify that logged features are informative, not degenerate.

**Checks:**
- Plot feature distributions for each known-labeled session
- Check for zero-variance features (broken sensor or bug)
- Check temporal autocorrelation (are consecutive samples redundant? → downsample logging)
- Mutual information between each feature and ENV label

**Tools:** pandas + matplotlib (offline, on local Mac).

---

### Stage 3 — Dataset Design

**Label strategy:**  
Primary: expert manual labeling via a review script that replays sessions.

```
python3 tools/label_session.py logs/ml/features_20260401_2130.jsonl
```

The script shows a timestamp window and prompts: `[1] night  [2] fog  [3] glare ...`

**Classes (8):**

| Class | Description | Min samples target |
| --- | --- | --- |
| `night_clear` | Dark, low thermal activity, no glare | 500 |
| `night_fog` | Dark, low contrast, soft texture | 300 |
| `glare` | High p95, NIR overexposed | 400 |
| `backlight` | Bright background, dark foreground | 300 |
| `day_normal` | Balanced brightness, good contrast | 500 |
| `rain_fog` | Low std, soft edges, high skew | 200 |
| `thermal_dominant` | Low NIR visibility, thermal key | 300 |
| `transition` | Rapid brightness/thermal change | 200 |

**Total target:** ~2700 labeled samples  
**Avoid overfitting:**
- Balance classes (use `class_weight='balanced'` in RF)
- Hold out 20% by session (not by sample — prevents temporal leakage)
- Train on sessions 1–N, test on session N+1 (temporal holdout)

**Edge cases to capture explicitly:**
- Sensor startup/warmup (thermal not yet stable)
- Rapid NIR occlusion (object close to lens)
- Simultaneous glare + thermal activity (fire through fog)
- High motion + low light (running at night)

---

### Stage 4 — Training Pipeline

**Environment:** Local Mac (not RPi4 for training).

```bash
# Install
pip install scikit-learn pandas matplotlib joblib

# Train
python3 tools/train_classifier.py \
    --data logs/ml/ \
    --output models/env_classifier_v1.pkl \
    --test-session 20260415
```

**`tools/train_classifier.py` outline:**
1. Load all JSONL files
2. Extract feature matrix X (22 cols) and label vector y
3. Train/test split by session date
4. `GridSearchCV` over `n_estimators=[50,100,200]`, `max_depth=[6,8,10]`
5. Fit final model on full train set
6. Evaluate: accuracy, per-class F1, confusion matrix
7. Serialize: `joblib.dump(model, output_path)`
8. Report feature importances (top-10 printed)

**Evaluation metrics:**
- Per-class F1 score (macro average ≥0.80 required)
- Confusion matrix (look for systematic rule/ML disagreements)
- Out-of-distribution test: run model on session it never saw
- Latency benchmark on RPi4: `time model.predict(X_test[:1000])`

---

### Stage 5 — Deployment on Raspberry Pi 4

**Package requirements (already available or lightweight):**
```
scikit-learn >= 1.3   # ~15MB, no C++ compilation needed
joblib                # bundled with sklearn
numpy                 # already required
```

**Loading at startup (`main.py`):**
```python
import joblib

def _load_ml_classifier(model_path: str):
    if not os.path.exists(model_path):
        return None
    try:
        clf = joblib.load(model_path)
        # Warmup call (avoids cold-start latency on first real call)
        clf.predict([np.zeros(22)])
        return clf
    except Exception as e:
        print(f"[ML] Model load failed: {e} — running rules-only")
        return None
```

**Version management:**  
Model file includes version in name: `env_classifier_v1.pkl`. Symlink `model.pkl` → current version. Update by swapping symlink, no code change required.

---

### Stage 6 — Integration into Real-Time System

**New file:** `ml_classifier.py`

```python
"""ml_classifier.py — Background ML inference thread for ENV classification."""

import threading, time, collections
import numpy as np

class MLEnvClassifier:
    INFERENCE_INTERVAL_FRAMES = 30

    def __init__(self, model_path: str, feature_extractor):
        self._clf = _load_classifier(model_path)
        self._extractor = feature_extractor
        self._state = MLState(env_class=None, confidence=0.0, ts=0.0)
        self._vote_window = collections.deque(maxlen=5)
        self._lock = threading.Lock()
        self._frame_count = 0

    def tick(self, frame_cache) -> None:
        """Call every frame from main loop. Non-blocking."""
        self._frame_count += 1
        if self._frame_count % self.INFERENCE_INTERVAL_FRAMES != 0:
            return
        # Snapshot features (cheap, in main thread)
        features = self._extractor.extract(frame_cache)
        # Submit to background thread
        threading.Thread(
            target=self._infer, args=(features,), daemon=True
        ).start()

    def _infer(self, features: np.ndarray) -> None:
        if self._clf is None:
            return
        proba = self._clf.predict_proba([features])[0]
        cls_idx = np.argmax(proba)
        conf = proba[cls_idx]
        cls_name = self._clf.classes_[cls_idx]
        with self._lock:
            self._vote_window.append(cls_name)
            self._state = MLState(env_class=cls_name, confidence=conf, ts=time.time())

    def get_state(self) -> MLState:
        with self._lock:
            return self._state

    def voted_env(self) -> tuple[str | None, float]:
        """Return majority-vote ENV class and fraction of agreeing votes."""
        with self._lock:
            w = list(self._vote_window)
        if not w:
            return None, 0.0
        most_common = max(set(w), key=w.count)
        return most_common, w.count(most_common) / len(w)
```

---

## 7. Detailed Execution Roadmap

> **Environment key:**  
> **[LOCAL]** — develop and complete on Mac only  
> **[RPi]** — must execute/validate on Raspberry Pi 4 with sensors  
> **[HYBRID]** — develop on Mac, must validate timing + sensor behavior on RPi before marking complete

---

### Phase 1 — Apply Researched Improvements to Current Pipeline

**Goal:** Better image quality and robustness, no ML yet.  
**Dependencies:** Modular codebase (complete).  
**Status:** ✅ COMPLETE (2026-04-08)

```
[x] 1.1  utils.py               Add FrameCache dataclass + build_frame_cache()
         - @dataclass with nir_full/nir_320/nir_160/nir_gray/thermal_80/ts
         - build_frame_cache(): pyrDown x2 + max_side=128 gray in ~0.4ms
         - File: utils.py (new: FrameCache, build_frame_cache)
         - Env: [LOCAL] — pure data structure, no sensor dependency

[x] 1.2  main.py                Build FrameCache once per frame, pass to all consumers
         - Built after capture confirm, before any processing (~line 600)
         - Passed to jerk_gate, lk_flow, nir_enhancer, ENV block
         - File: main.py
         - Env: [HYBRID] — frame loop integration; verify no timing regression on RPi4
         - RPi Validation: Run 5 min, FPS ≥20, frame_time_ms does not increase

[x] 1.3  thermal_pipeline.py    Replace EMA background with Kalman filter (R5)
         - Added KalmanThermalBackground: vectorised per-pixel K/P/Q/R update
         - No warmup counter: converges naturally within ~5 frames (P₀=100)
         - ThermalProcessor now instantiates KalmanThermalBackground
         - File: thermal_pipeline.py (new class, ThermalProcessor.__init__ updated)
         - Env: [HYBRID] — thermal sensor timing; Kalman cost must be ≤0.5ms on RPi4
         - RPi Validation: Run 5 min, verify fg/bg mask quality, CPU ≤65%

[x] 1.4  nir_pipeline.py        Adaptive CLAHE clip limit from histogram entropy (R3)
         - _apply_clahe_boost: compute L-channel entropy, map → clip via 4.5-0.35*H
         - EMA smoothing (α=0.2) prevents rapid oscillation
         - Replaces fixed 3-level CLAHE selection with single adaptive CLAHE object
         - File: nir_pipeline.py (HybridNIREnhancer._apply_clahe_boost)
         - Env: [HYBRID] — NIR camera required; verify entropy responds to real scenes
         - RPi Validation: Visually inspect CLAHE adaptation across night/indoor/glare scenes

[x] 1.5  nir_pipeline.py        Guided filter pass with thermal guide (R2)
         - HybridNIREnhancer.process() accepts thermal_guide param
         - Upsamples thermal 80×62 → 320×240 as guide, applies cv.ximgproc.guidedFilter
         - Gracefully skipped if opencv-contrib not installed (_HAS_XIMGPROC flag)
         - File: nir_pipeline.py (new _HAS_XIMGPROC flag, process() signature)
         - Env: [HYBRID] — opencv-contrib availability on RPi must be verified; 2ms cost on RPi4
         - RPi Validation: Confirm ximgproc is available; guided filter adds ≤2ms/frame

[x] 1.6  main.py                Soft mask upgrade: (7,7) → (15,15) Gaussian kernel
         - fg_mask boundary blur upgraded from GaussianBlur((7,7), 2.0) to (15,15), 4.0
         - Eliminates hard thermal/NIR boundary artefacts
         - File: main.py (fusion blend path)
         - Env: [HYBRID] — visual quality change; Gaussian cost on RPi4 at 640×480
         - RPi Validation: Visual quality check + FPS impact ≤1ms

[x] 1.7  motion.py              Sparse LK optical flow alongside JerkGate (R4)
         - Added SparseOpticalFlowMotion: 20 Shi-Tomasi corners on 160×120
         - Refreshes corners every 15 frames; publishes motion_magnitude/direction/jerk
         - Called every frame from main.py with precomputed 160×120 gray
         - File: motion.py (new class SparseOpticalFlowMotion)
         - Env: [HYBRID] — motion vectors depend on real camera motion; timing on RPi4
         - RPi Validation: Verify LK corners track correctly on real NIR footage; cost ≤2ms

[x] 1.8  thermal_pipeline.py    MAD anomaly detector alongside E1 (R7)
         - Added ThermalMADAnomalyDetector: Leys et al. 2013 MAD z-score
         - Temporal integrator: fires only after 3 consecutive anomaly frames
         - Runs alongside ThermalAnomalyDetectorLite; overrides blobs when active
         - File: thermal_pipeline.py (new class), main.py (E1 block integration)
         - Env: [HYBRID] — thermal foreground pixels depend on real sensor data
         - RPi Validation: Test E1 firing with warm objects in scene; no false positives for 5 min

[x] 1.9  utils.py               dHash + HMAC capture signing (R6)
         - Added dhash(), sign_capture(), CaptureIntegrityChain
         - Merkle-style chain: each capture JSON includes hash of previous
         - write_capture_meta() accepts frame= param; signs when _capture_chain set
         - File: utils.py (new functions/class), main.py (module-level chain init)
         - Note: session key file created at save_dir/.session_key on first run
         - Env: [LOCAL] — pure cryptographic computation, no sensor dependency

[x] 1.10 main.py / motion.py / nir_pipeline.py   Merge downscale ops via FrameCache
         - JerkGate.update() accepts precomputed_gray= → skips internal resize+cvtColor
         - HybridNIREnhancer.process() accepts precomputed_small= → skips pyrDown
         - ENV block reads nir_gray from frame_cache directly (no extra cvtColor)
         - Estimated: ~1.5 ms/frame saved (3× resize+cvtColor eliminated)
         - Files: motion.py, nir_pipeline.py, main.py
         - Env: [HYBRID] — estimated savings must be confirmed on RPi4 NEON paths
         - RPi Validation: Profile frame_time_ms before/after; confirm ≥1ms improvement

[x] 1.11 main.py                Rate-limit ENV classification to every 10 frames
         - _env_frame_idx counter + ENV_CLASSIFICATION_INTERVAL (default=10)
         - Feature extraction (entropy, glare, std) skipped on non-classified frames
         - env_controller.update() still called every frame with cached desired
         - Estimated: ~3 ms/frame saved at 30 FPS (10× fewer full feature extractions)
         - File: main.py (auto_rule block)
         - Configurable via config key: env_classification_interval
         - Env: [HYBRID] — timing benefit must be measured on RPi4; verify ENV stability
         - RPi Validation: Log ENV transitions, confirm no rapid oscillation or missed changes
```

**Exit criteria:** All tests pass, FPS ≥20 at 640×480 on RPi4.

---

### Phase 2 — Finalize Stable Pipeline (Pre-ML)

**Goal:** Stable, deterministic pipeline with no regressions. Baseline for ML comparison.  
**Dependencies:** Phase 1 complete.

| Task | File | Description | Env |
| --- | --- | --- | --- |
| 2.1 | All | Run 30-minute soak test, log FPS and memory | **[RPi]** |
| 2.2 | `main.py` | Implement `FrameCache` sharing (verify no race conditions) | **[HYBRID]** |
| 2.3 | `env_presets.py` | Add sequence voting (R9) to `EnvPresetController` | **[HYBRID]** |
| 2.4 | `display_pipeline.py` | Verify single glare path (Fix 2) | **[LOCAL]** |
| 2.5 | `main.py` | Implement deterministic mode switching (Fix 6) | **[HYBRID]** |
| 2.6 | All | Profile hot path: identify top-3 time consumers, document | **[RPi]** |
| 2.7 | `utils.py` | Add `FrameQualityScore` (R10) to capture path | **[LOCAL]** |
| 2.8 | — | Tag git commit as `v2.0-stable-pre-ml` | **[LOCAL]** |

**RPi Validation Requirements:**

- **2.1** [RPi]: Full 30-min session with NIR + thermal sensors active.
  - FPS ≥20 sustained (no degradation over time)
  - RSS memory stable (< ±10MB drift over session)
  - CPU ≤75% average across all cores
  - Zero crashes or exceptions
  - Pull logs: `logs/performance/perf_*.jsonl`

- **2.2** [HYBRID]: Run 10-min session after FrameCache sharing changes.
  - No frame drops or visual artifacts from partial cache reads
  - FPS unchanged from Phase 1 baseline

- **2.3** [HYBRID]: Induce ENV transitions (glare, then dark, then normal).
  - Verify sequence voting stabilises transitions (no rapid oscillation)
  - ENV changes occur within ≤3 × 10 frames of actual condition change

- **2.5** [HYBRID]: Press mode keys rapidly during live session.
  - Verify no partial-frame state mix visible in output
  - No crash on rapid key input

- **2.6** [RPi]: Run `python -m cProfile -o profile.out -m smartbinocular` for 2 min.
  - Pull `profile.out` to Mac for analysis with `pstats`
  - Document top-3 functions by cumulative time
  - Target: no single function > 20ms/frame

**Exit criteria:** Stable 30-min session, no crashes, no visible artifacts, FPS documented.

---

### Phase 3 — Logging & Data Collection

**Goal:** Infrastructure for collecting labeled training data.  
**Dependencies:** Phase 2 stable.

| Task | File | Description | Env |
| --- | --- | --- | --- |
| 3.1 | `utils.py` | Implement `MLLogger` (async JSONL writer) | **[LOCAL]** |
| 3.2 | `main.py` | Wire `MLLogger.log()` every 5 frames with all 23 features | **[HYBRID]** |
| 3.3 | `tools/` | Create `tools/label_session.py` replay labeling script | **[LOCAL]** |
| 3.4 | `tools/` | Create `tools/check_features.py` (distribution plots, variance check) | **[LOCAL]** |
| 3.5 | `config.py` | Add `ML_LOG_ENABLED`, `ML_LOG_DIR`, `ML_LOG_INTERVAL` config keys | **[LOCAL]** |
| 3.6 | — | Run 5+ field sessions covering: night, fog sim, glare, normal, indoor | **[RPi]** |
| 3.7 | — | Label all sessions using `label_session.py` | **[LOCAL]** |
| 3.8 | — | Run `check_features.py` — verify no zero-variance features | **[LOCAL]** |
| 3.9 | — | Target: ≥2700 labeled samples across all 8 classes | **[RPi]** |

**RPi Validation Requirements:**

- **3.2** [HYBRID]: Wire MLLogger then run 10-min session.
  - FPS must NOT decrease (logger is buffered/async)
  - Confirm `logs/ml/features_*.jsonl` written correctly
  - Verify no file I/O errors on SD card (RPi SD write speed varies)

- **3.6** [RPi]: Each field session:
  - Minimum 15 minutes per session
  - Cover at least one ENV class per session
  - Confirm `logs/ml/` accumulates entries at expected rate (≈ FPS / 5 entries/sec)
  - After each session: `rsync -avz pi@<rpi_ip>:/home/pi/smartBinocular/logs/ml/ ./logs_rpi/ml/`

- **3.9** [RPi]: Ongoing data collection milestone.
  - Check class balance after each sync: run `check_features.py`
  - If any class < 100 samples, plan additional dedicated session for that ENV

**Exit criteria:** ≥2700 samples, ≥6 distinct sessions, all 8 classes represented.

---

### Phase 4 — Train ML Model

**Goal:** Trained, validated, deployable RandomForest classifier.  
**Dependencies:** Phase 3 data ≥2700 samples.

| Task | File | Description | Env |
| --- | --- | --- | --- |
| 4.1 | `tools/train_classifier.py` | Write full training script (see Stage 4 above) | **[LOCAL]** |
| 4.2 | — | Run GridSearchCV, select best hyperparameters | **[LOCAL]** |
| 4.3 | — | Evaluate: macro F1 ≥0.80, review confusion matrix | **[LOCAL]** |
| 4.4 | — | Benchmark inference time on RPi4: must be <2ms | **[RPi]** |
| 4.5 | — | Serialize to `models/env_classifier_v1.pkl` | **[LOCAL]** |
| 4.6 | `tools/` | Create `tools/explain_model.py` (feature importances, sample prediction audit) | **[LOCAL]** |
| 4.7 | — | Document top-5 important features, validate they make physical sense | **[LOCAL]** |

**RPi Validation Requirements:**

- **4.4** [RPi]: Copy `models/env_classifier_v1.pkl` to RPi via rsync.
  ```bash
  rsync -avz models/env_classifier_v1.pkl pi@<rpi_ip>:/home/pi/smartBinocular/models/
  ```
  Then run inference benchmark:
  ```bash
  python3 -c "
  import joblib, numpy as np, time
  clf = joblib.load('models/env_classifier_v1.pkl')
  X = np.random.rand(1000, 22)
  t0 = time.perf_counter()
  clf.predict(X)
  print(f'{(time.perf_counter()-t0)*1000/1000:.3f} ms/call')
  "
  ```
  - Required: ≤2ms per single-sample call
  - If >2ms: reduce `n_estimators` or `max_depth` and retrain

**Failure criteria (requires more data collection):**  
- Any class F1 < 0.65  
- Inference > 2ms on RPi4  
- Top feature is a time-related feature (would indicate label leakage)  

---

### Phase 5 — Integrate Hybrid System

**Goal:** ML running alongside rules, decision fusion active.  
**Dependencies:** Phase 4 model validated, Phase 2 stable pipeline.

| Task | File | Description | Env |
| --- | --- | --- | --- |
| 5.1 | `ml_classifier.py` | Implement `MLEnvClassifier` (see Stage 6) | **[LOCAL]** |
| 5.2 | `env_presets.py` | Integrate `MLEnvClassifier.tick()` into `EnvPresetController` | **[HYBRID]** |
| 5.3 | `env_presets.py` | Implement decision fusion logic (Section 5.4) | **[HYBRID]** |
| 5.4 | `config.py` | Add `ML_MODEL_PATH`, `ML_CONFIDENCE_THRESHOLD`, `ML_ENABLED` | **[LOCAL]** |
| 5.5 | `main.py` | Pass ML state to HUD (display `ML:night_clear 0.87` in corner) | **[HYBRID]** |
| 5.6 | — | Run side-by-side test: rules-only vs hybrid, log ENV decisions | **[RPi]** |
| 5.7 | — | Review `overrides_<date>.jsonl` — find cases where ML overrode rules | **[LOCAL]** |
| 5.8 | — | Validate: hybrid must not reduce FPS more than 1 FPS vs. rules-only | **[RPi]** |

**RPi Validation Requirements:**

- **5.2** [HYBRID]: Run 10-min session with `ML_ENABLED=True`.
  - ENV decisions appear in `logs/runtime/session_*.log`
  - Background inference thread does not cause frame jitter (check frame_time std)

- **5.3** [HYBRID]: Force transitions by covering lens (dark), pointing at light (glare).
  - Verify decision fusion selects correct ENV for each condition
  - Verify safety overrides fire correctly for `glare_heavy`

- **5.5** [HYBRID]: Verify HUD displays ML class + confidence without visual lag.

- **5.6** [RPi]: 30-min session with `ML_ENABLED=True`.
  - Pull `logs/runtime/session_*.log` for override analysis
  - Compare ENV time-series vs. rules-only baseline from Phase 2

- **5.8** [RPi]: Compare FPS counter from Phase 2 soak vs. Phase 5 hybrid.
  - Max acceptable regression: 1 FPS
  - If regression >1 FPS: profile background thread scheduling

**Exit criteria:** Hybrid running stably for 60 min, ML override rate logged, no FPS regression.

---

### Phase 6 — Optimize & Validate

**Goal:** Production-ready hybrid system, documented, benchmarked, versioned.  
**Dependencies:** Phase 5 integrated.

| Task | File | Description | Env |
| --- | --- | --- | --- |
| 6.1 | — | Full 2-hour soak test in each major ENV class | **[RPi]** |
| 6.2 | — | Memory profile: verify no leak over 2h session | **[RPi]** |
| 6.3 | `tools/` | Create `tools/eval_hybrid.py` — replay session, compare rule vs hybrid ENV decisions | **[LOCAL]** |
| 6.4 | — | Retrain model with Phase 5 data + new edge cases found in 6.1 | **[LOCAL]** |
| 6.5 | `config.py` | Tune `ML_CONFIDENCE_THRESHOLD` based on override analysis | **[HYBRID]** |
| 6.6 | All | Final code review, remove debug prints, verify all TODOs resolved | **[LOCAL]** |
| 6.7 | `README.md` | Update with hybrid system description, model version, training instructions | **[LOCAL]** |
| 6.8 | — | Tag git commit as `v3.0-hybrid` | **[LOCAL]** |

**RPi Validation Requirements:**

- **6.1** [RPi]: 2-hour continuous session per major ENV class (night_clear, glare, fog).
  - Pull `logs/performance/perf_*.jsonl` after each session
  - FPS ≥20 sustained throughout; no thermal throttling
  - Zero crashes, zero unhandled exceptions in `logs/debug/errors_*.log`

- **6.2** [RPi]: During 2-hour session, monitor RSS every 5 minutes.
  ```bash
  # On RPi, run alongside the pipeline:
  while true; do
    ps -o rss= -p $(pgrep -f smartbinocular) >> /tmp/rss_log.txt
    sleep 300
  done
  ```
  - Pull and plot `rss_log.txt` on Mac
  - Accept: RSS drift < 20MB over 2h

- **6.5** [HYBRID]: After reviewing override analysis from 6.3:
  - Adjust `ML_CONFIDENCE_THRESHOLD` in config
  - Run 30-min session on RPi to validate new threshold reduces false overrides

**Exit criteria:** 2-hour soak test passed, no memory leak, hybrid documented and tagged.

---

## 8. Notes for Future Expansion

### ML Integration Points (already designed for ML, per README)

| Location | Current state | Next ML step |
| --- | --- | --- |
| `thermal_pipeline.py` post-processing | Kalman + MAD anomaly | Tiny ONNX anomaly autoencoder (when NPU available) |
| `nir_pipeline.py` enhancement | Rule-based Schmitt | Small MLP for brightness mode prediction |
| `env_presets.py` classification | RF hybrid (Phase 5) | Online learning (Hoeffding Tree, incremental updates) |
| `motion.py` LK optical flow | Sparse tracking | EfficientDet-Nano for object detection (RPi5 / NPU) |
| Capture chain | HMAC + dHash | Blockchain-anchored session manifest (research prototype) |

### Online/Incremental Learning (Post-Phase 6)

Once the system accumulates 10,000+ labeled frames, consider switching from RandomForest to an online learner (River library: `HoeffdingTreeClassifier`). Benefits:
- Adapts to new sensor drift without full retrain
- No batch retraining cycle
- Minimal RAM (~50KB model state)

### Hardware Upgrade Path

| Upgrade | Unlocks |
| --- | --- |
| Raspberry Pi 5 (2× CPU speed) | Dense optical flow, larger RF, ONNX CNN |
| Hailo-8 NPU (26 TOPS) | Full YOLOv8n object detection, 30fps |
| Coral USB TPU (4 TOPS) | MobileNetV2 inference, <10ms |

Design current system to be NPU-drop-in-ready: all ML calls go through `ml_classifier.py` interface. Replace the sklearn backend without touching `main.py`.

### Security Expansion

- Replace session HMAC key with a TPM-backed key (RPi4 does not have TPM natively, but `tpm2-tools` + hardware TPM module is feasible)
- Add GPS timestamp anchoring to capture metadata for forensic provenance
- Encrypt `logs/ml/` JSONL files at rest (AES-256 via `cryptography` package)

---

## 9. Execution Environment & Logging Strategy

### 9.1 Environment Model

| Environment | Responsibilities |
|-------------|-----------------|
| **Local Mac** | Development, refactoring, debugging, ML training, dataset preparation, log analysis, offline tools (`tools/`) |
| **Raspberry Pi 4** | Real-time pipeline execution, hardware integration (NIR + Thermal sensors), soak tests, final performance validation |

**Classification key:**

| Tag | Meaning |
|-----|---------|
| **[LOCAL]** | Fully executable on Mac. No hardware needed. |
| **[RPi]** | Must execute on RPi4 with both sensors connected. Cannot be validated on Mac. |
| **[HYBRID]** | Develop and unit-test on Mac. Must validate timing, memory, and sensor behavior on RPi before marking complete. |

**Summary of Phase 1–6 task classification:**

| Phase | LOCAL | HYBRID | RPi |
|-------|-------|--------|-----|
| Phase 1 (11 tasks) | 2 | 9 | 0 |
| Phase 2 (8 tasks) | 3 | 3 | 2 |
| Phase 3 (9 tasks) | 5 | 1 | 3 |
| Phase 4 (7 tasks) | 6 | 0 | 1 |
| Phase 5 (8 tasks) | 2 | 3 | 3 |
| Phase 6 (8 tasks) | 5 | 1 | 2 |
| **Total** | **23** | **17** | **11** |

**When to develop LOCAL only:**
- Pure algorithmic changes (new math, data structures, helper functions)
- Offline tools that don't touch sensor hardware (`tools/`, `models/`)
- Code that runs without a camera/thermal sensor attached
- ML training, GridSearchCV, dataset analysis, confusion matrix review

**When HYBRID validation is required:**
- Any change to the frame loop's timing budget
- New OpenCV operations with unknown RPi4 NEON performance profile
- Changes to memory allocation patterns in the hot path
- New sensor-dependent data paths (thermal background model, NIR enhancement)
- UI/HUD changes visible on the physical display

**When RPi-only:**
- Soak tests (sensor behavior over time, SD card I/O health, thermal throttling)
- Hardware profiling (actual CPU/memory measurements on the target device)
- Sensor correctness validation (MI48 SPI timing, Picamera2 frame timing)
- Final performance sign-off before phase completion

---

### 9.2 Logging Architecture

**Directory structure:**
```
logs/
├── runtime/          # High-level session events (plaintext)
├── performance/      # Per-frame sampled metrics (JSONL)
├── ml/               # Feature vectors for ML training (JSONL, Phase 3+)
└── debug/            # Stack traces and exceptions (plaintext)
```

**Log file patterns:**

| File pattern | Content | Format | Written by |
|---|---|---|---|
| `logs/runtime/session_<ts>.log` | Start/stop, mode changes, ENV transitions | Plaintext | `main.py` |
| `logs/performance/perf_<ts>.jsonl` | fps, frame_time_ms, cpu_percent, memory_mb | JSONL | `metrics.py` |
| `logs/debug/errors_<ts>.log` | Exceptions, stack traces | Plaintext | global exception hook |
| `logs/ml/features_<ts>.jsonl` | 23-feature vector + optional label + timestamp | JSONL | `MLLogger` (Phase 3+) |

**Performance log sample record:**
```json
{"ts": 1744100000.123, "frame": 1500, "fps": 28.4, "frame_time_ms": 35.2, "cpu_percent": 61.0, "memory_mb": 187.3}
```

**Session log sample events:**
```
2026-04-08 21:00:00 [START] version=1.1 config=night_clear fps_target=30
2026-04-08 21:00:45 [ENV_CHANGE] old=night_clear new=glare frame=1350
2026-04-08 21:05:00 [MODE_SWITCH] mode=3 frame=9000
2026-04-08 21:30:00 [END] frames=54000 duration=1800s avg_fps=30.1 peak_cpu=68%
```

---

### 9.3 Instrumentation Requirements

**Config flags to add to `config.py`:**
```python
ENABLE_LOGGING       = True      # master on/off switch
LOG_DIR              = "logs"    # relative to repo root
PERF_LOG_INTERVAL    = 10        # sample every N frames
ML_LOG_ENABLED       = False     # enable feature logging (Phase 3+)
ML_LOG_DIR           = "logs/ml"
ML_LOG_INTERVAL      = 5         # log features every N frames
```

**Frame loop instrumentation rules:**
- Sample `FPSCounter.fps` and `frame_time_ms` every `PERF_LOG_INTERVAL` frames
- Write to an in-memory `collections.deque` buffer (max 100 entries)
- Flush buffer to disk in a **background thread** every 60 seconds — **never flush in the hot path**
- The background thread acquires a lock only for deque drain, not for per-frame writes

**Session log events to capture:**
- `session_start`: timestamp, config snapshot, software version, sensor IDs
- `env_change`: old ENV, new ENV, frame number, trigger (rule/ml/manual)
- `mode_switch`: mode name, frame number
- `ml_override`: rule_env, ml_env, confidence, frame number (Phase 5+)
- `session_end`: total frames, duration, avg FPS, peak CPU, total captures

**Non-blocking guarantee:** No log write should add >0.1ms to frame_time. Use buffered I/O (`io.BufferedWriter` with `buffer_size=65536`) and OS-level async flush.

---

### 9.4 Sync Workflow: RPi ↔ Local Mac

**Pull all logs from RPi to Mac:**
```bash
rsync -avz pi@<rpi_ip>:/home/pi/smartBinocular/logs/ ./logs_rpi/
```

**Pull ML feature logs only (Phase 3+ dataset sync):**
```bash
rsync -avz pi@<rpi_ip>:/home/pi/smartBinocular/logs/ml/ ./logs_rpi/ml/
```

**Push updated code to RPi:**
```bash
rsync -avz --exclude='__pycache__' --exclude='logs/' --exclude='models/' \
    ./ pi@<rpi_ip>:/home/pi/smartBinocular/
```

**Push a trained model to RPi (Phase 5+):**
```bash
rsync -avz models/env_classifier_v1.pkl pi@<rpi_ip>:/home/pi/smartBinocular/models/
```

**Which logs are required per phase:**

| Phase | Required log types | Purpose |
|-------|-------------------|---------|
| Phase 2 validation | `performance/*.jsonl`, `runtime/*.log` | Confirm FPS ≥20, no crashes over 30 min |
| Phase 3 dataset building | `ml/features_*.jsonl` | Source data for ML training |
| Phase 4 inference benchmark | Manual benchmark output | Validate ≤2ms on RPi4 |
| Phase 5 evaluation | `runtime/*.log` (ENV decisions), `ml/features_*.jsonl` | Override analysis, hybrid comparison |
| Phase 6 final validation | All log types | 2-hour soak test evidence, memory profile |

---

### 9.5 Raspberry Pi Setup Checklist

Before any RPi validation session:

- [ ] Repository synced: `rsync` push completed
- [ ] Package installed: `pip install -e .` on RPi
- [ ] Sensors connected: NIR camera (Picamera2) + thermal (MI48 SPI)
- [ ] Log directories exist: `logs/{runtime,performance,ml,debug}/`
- [ ] Sufficient SD card space: ≥2GB free (`df -h /`)
- [ ] Power supply: 5V/3A USB-C (not USB 2.0 hub)
- [ ] Note RPi CPU temp before starting: `vcgencmd measure_temp` (warn if >60°C)
- [ ] After session: pull logs via rsync, check `logs/debug/errors_*.log` for exceptions

---

*End of ML_HYBRID_SYSTEM_PLAN.md*
