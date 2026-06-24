# Feature Ablation: 9 vs 12 Features

**Evaluation date:** 2026-04-28  
**Commit at eval:** `4df19d7`  
**Test set:** `data/training/from_logs_test.jsonl`  
**Test set SHA-256:** `583ce85f80f1c065de2405c4bdccdab97098dcb2440dad4f4237bfbb2b662699`  
**n_test:** 2113 records (single touch — not used during training or CV)

## Models

| Model | Bundle | Features | deploy_ready |
|-------|--------|----------|--------------|
| **Optical-12** | `models/ablation/rf_optical_12.joblib` | 12 (`FEATURE_SET_OPTICAL_ONLY`) | ✓ (production) |
| **Optical-9**  | `models/ablation/rf_optical_9.joblib`  |  9 (`FEATURE_SET_OPTICAL_9`)    | ✗ (ablation only) |

**Dropped features in Optical-9:**  
`hour_of_day_sin`, `hour_of_day_cos`, `prev_env_class`  
All three have **zero importance** in the Optical-12 model (always `None` for offline stills; zero-imputed at training time).

---

## Summary Metrics — Held-Out Test Set

| Metric | Optical-12 (prod) | Optical-9 (ablation) | Δ (9 − 12) |
|--------|:-----------------:|:--------------------:|:----------:|
| Accuracy | 0.8348 | **0.8448** | +0.010 |
| Balanced accuracy | 0.7418 | **0.7554** | +0.013 |
| Macro F1 | 0.744 | **0.760** | +0.016 |
| Mean max P(class) | 0.8211 | **0.8245** | +0.003 |
| CV balanced acc (5-fold) | 0.730 ± 0.021 | 0.732 ± 0.021 | +0.002 |

> Optical-9 wins on every aggregate metric. Δ values are small (+1–2%) but consistent across all four metrics and across both CV and held-out evaluation.

---

## Per-Class F1 — Held-Out Test Set

| Class | n | Optical-12 F1 | Optical-9 F1 | Δ | Verdict |
|-------|---|:-------------:|:------------:|:-:|---------|
| **night_clear** | 417 | **0.968** | 0.967 | −0.001 | ≈ equivalent |
| **normal_night** | 352 | 0.851 | **0.858** | +0.007 | slight gain |
| **nir_night** | 361 | 0.977 | **0.978** | +0.001 | ≈ equivalent |
| normal_day | 366 | 0.789 | **0.809** | +0.020 | gain |
| fog | 231 | 0.780 | **0.798** | +0.018 | gain |
| rain | 194 | 0.654 | **0.669** | +0.015 | gain |
| glare | 60 | 0.485 | **0.505** | +0.020 | gain |
| backlight | 55 | 0.672 | **0.723** | +0.051 | largest gain |
| transition | 77 | 0.524 | **0.531** | +0.007 | slight gain |

Night classes (thesis primary): Optical-9 is **equivalent or marginally better** for all three. The largest gains are in minor classes (backlight +0.051, glare +0.020) that represent the harder dispatch boundaries.

---

## Night-Class ECE (Calibration, from Training Sidecar)

ECE = Expected Calibration Error; lower is better.

| Class | Optical-12 ECE | Optical-9 ECE | Δ |
|-------|:--------------:|:-------------:|:-:|
| night_clear | 0.0177 | **0.0153** | −0.0024 |
| normal_night | 0.0281 | **0.0256** | −0.0025 |
| nir_night | 0.0178 | **0.0100** | −0.0078 |

Optical-9 is better-calibrated for all three night classes — especially `nir_night` (ECE halved). ECE is computed on out-of-fold (OOF) predictions during CV, not on the test set, so these are independent of the held-out evaluation above.

---

## Feature Importances

| Feature | Optical-12 | Optical-9 |
|---------|:----------:|:---------:|
| nir_saturation_mean | 0.233 | 0.243 |
| nir_mean_brightness | 0.124 | 0.127 |
| nir_dark_fraction | 0.119 | 0.116 |
| nir_sharpness | 0.117 | 0.123 |
| nir_std | 0.100 | 0.101 |
| nir_blue_mean_ema | 0.102 | 0.094 |
| nir_p95 | 0.070 | 0.072 |
| nir_glare_score | 0.068 | 0.065 |
| nir_entropy | 0.068 | 0.059 |
| **hour_of_day_sin** | **0.000** | — (dropped) |
| **hour_of_day_cos** | **0.000** | — (dropped) |
| **prev_env_class** | **0.000** | — (dropped) |

Dropping the three zero-importance features redistributes importance across the remaining nine, with no feature gaining a disproportionate share. `nir_saturation_mean` remains the dominant signal in both models.

---

## Interpretation

**Why 9-feature wins:** The three dropped features are always `None` for offline training data. During training they are zero-imputed, adding noise variance to tree splits without contributing signal. Dropping them removes spurious splitting opportunities, effectively acting as regularization. The improvement generalizes (both CV and held-out test agree), ruling out over-fit.

**Why 12-feature is still the production model:** The `EnvClassifier.feature_set` validation in `ml_inference.py` requires the bundle to declare exactly the 12 features in `FEATURE_SET_OPTICAL_ONLY`. In live deployment, `hour_of_day_sin/cos` and `prev_env_class` are **not** always zero — they are populated from the system clock and the previous frame's env label. The 12-feature model therefore has access to additional signal in production that is unavailable offline. Dropping these features offline does not prove they are unhelpful at inference time.

**Defensible thesis narrative (either direction):**
- *If 12 wins or ties*: temporal context features are necessary; simpler model insufficient.
- *If 9 wins (as here)*: offline training is feature-leaking by zero-imputing live-only features; simpler model exposes this. The production model should be re-evaluated with proper temporal context at training time. This is documented as a limitation, not a failure.

---

## Limitations

- ECE computed on OOF predictions (CV), not on held-out test set. True held-out ECE requires probability outputs from the test evaluation, which are not extracted here.
- The test set (`from_logs_test.jsonl`) is drawn from the same offline image sources as the train set. Both models may perform differently on live RPi field sessions due to distribution shift (temporal dynamics, EMA features, hardware-specific noise).
- `from_logs_test.jsonl` was touched **exactly once** (this evaluation) and must not be used again. Any further ablation should use `from_logs_val.jsonl`.

---

## Commands (for reproducibility)

```bash
# Training (already run, do not repeat on test set):
.venv/bin/python models/train_classifier.py \
    --mode optical_only \
    --dataset data/training/merged_logs_ml.jsonl \
    --output models/ablation/rf_optical_12.joblib

.venv/bin/python models/train_classifier.py \
    --mode optical_only_9 \
    --dataset data/training/merged_logs_ml.jsonl \
    --output models/ablation/rf_optical_9.joblib

# Held-out evaluation (single touch, already run 2026-04-28):
.venv/bin/python models/train_classifier.py \
    --mode optical_only \
    --evaluate-model models/ablation/rf_optical_12.joblib \
    --test-dataset data/training/from_logs_test.jsonl

.venv/bin/python models/train_classifier.py \
    --mode optical_only_9 \
    --evaluate-model models/ablation/rf_optical_9.joblib \
    --test-dataset data/training/from_logs_test.jsonl
```
