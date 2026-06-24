# Feature Ablation: 9-Feature vs 12-Feature RF Classifier

**Generated:** 2026-04-27  
**Train data:** `data/training/from_logs_train_a.jsonl` (10 182 samples, seed=42, 85/15 split)  
**Test data:** `data/training/from_logs_test.jsonl` (2 113 samples — touched once per model)  
**Models:** `models/ablation/rf_optical_12.joblib`, `models/ablation/rf_optical_9.joblib`  
**Full sweep CSVs:** `docs/tables/ml/ablation_12feat_threshold_sweep.csv`, `docs/tables/ml/ablation_9feat_threshold_sweep.csv`  
**Reliability diagrams (night classes):** `docs/figures/ml/ablation/rf_optical_*_reliability_*.png` (joblib artifacts remain under `models/ablation/`)

---

## Feature sets compared

| Feature | 12-feat | 9-feat |
|---------|---------|--------|
| nir\_mean\_brightness | ✓ | ✓ |
| nir\_std | ✓ | ✓ |
| nir\_entropy | ✓ | ✓ |
| nir\_p95 | ✓ | ✓ |
| nir\_glare\_score | ✓ | ✓ |
| nir\_sharpness | ✓ | ✓ |
| nir\_dark\_fraction | ✓ | ✓ |
| nir\_saturation\_mean | ✓ | ✓ |
| nir\_blue\_mean\_ema | ✓ | ✓ |
| hour\_of\_day\_sin | ✓ | — dropped |
| hour\_of\_day\_cos | ✓ | — dropped |
| prev\_env\_class | ✓ | — dropped |

The 9-feature set removes three temporal-context features that require session metadata (`hour_of_day_*`) or runtime state (`prev_env_class`). These are not available from a single still image and require the pipeline to be running to accumulate state.

---

## Cross-validation (5-fold, balanced accuracy)

| Model | CV balanced\_acc | σ |
|-------|----------------|---|
| 12-feat | 0.7301 | ±0.0212 |
| **9-feat** | **0.7321** | ±0.0212 |

The 9-feature model has a marginally higher cross-validated balanced accuracy (+0.002). The difference is within one standard deviation and not statistically significant — the two models are functionally equivalent on this metric.

---

## One-shot test evaluation at τ = 0.62 (production threshold)

| Metric | 12-feat | 9-feat | Δ (9−12) |
|--------|---------|--------|-----------|
| F1 night\_clear | 0.9709 | 0.9711 | +0.0002 |
| F1 normal\_night | 0.9079 | 0.9097 | **+0.0018** |
| F1 nir\_night | **0.9819** | 0.9806 | −0.0013 |
| Macro F1 (night) | 0.9536 | **0.9538** | +0.0002 |
| Abstention rate | 0.1827 | **0.1798** | −0.0029 |
| n accepted (of 2113) | 1727 | 1733 | +6 |

At the production threshold the 9-feature model:
- **Equal or better** on F1 night\_clear and normal\_night
- **Marginally lower** on F1 nir\_night (−0.0013, within rounding noise)
- **Lower abstention** by 0.003 — accepts 6 more samples, reducing abstained cases
- **Net macro F1 identical** within the precision of the sweep

---

## Calibration quality (ECE on night classes, from training CV)

Lower ECE = better-calibrated probability estimates.

| Night class | ECE (12-feat) | ECE (9-feat) | Δ |
|-------------|---------------|--------------|---|
| night\_clear | 0.0177 | **0.0153** | −0.0024 |
| normal\_night | 0.0281 | **0.0256** | −0.0025 |
| nir\_night | 0.0178 | **0.0100** | −0.0078 |

The 9-feature model is **better calibrated** across all three night classes. This is the most defensible signal: the temporal features (`hour_of_day_*`, `prev_env_class`) appear to add calibration noise without precision benefit, possibly because they correlate with label sequences in training data in ways that do not generalize.

---

## Threshold sweep — 9-feature model (full)

| τ | abstain | n\_acc | F1\_nc | F1\_nn | F1\_nir | macro |
|---|---------|--------|--------|--------|---------|-------|
| 0.50 | 0.0795 | 1945 | 0.9688 | 0.8823 | 0.9780 | 0.9430 |
| 0.55 | 0.1226 | 1854 | 0.9688 | 0.8916 | 0.9793 | 0.9466 |
| 0.60 | 0.1614 | 1772 | 0.9699 | 0.9027 | 0.9806 | 0.9511 |
| **0.62** | **0.1798** | **1733** | **0.9711** | **0.9097** | **0.9806** | **0.9538** |
| 0.65 | 0.2101 | 1669 | 0.9757 | 0.9163 | 0.9806 | 0.9575 |
| 0.70 | 0.2556 | 1573 | 0.9777 | 0.9231 | 0.9845 | 0.9618 |
| 0.75 | 0.3076 | 1463 | 0.9786 | 0.9314 | 0.9858 | 0.9653 |

---

## Threshold sweep — 12-feature model (full)

| τ | abstain | n\_acc | F1\_nc | F1\_nn | F1\_nir | macro |
|---|---------|--------|--------|--------|---------|-------|
| 0.50 | 0.0814 | 1941 | 0.9688 | 0.8770 | 0.9766 | 0.9408 |
| 0.55 | 0.1202 | 1859 | 0.9687 | 0.8889 | 0.9778 | 0.9452 |
| 0.60 | 0.1656 | 1763 | 0.9710 | 0.9035 | 0.9805 | 0.9517 |
| **0.62** | **0.1827** | **1727** | **0.9709** | **0.9079** | **0.9819** | **0.9536** |
| 0.65 | 0.2101 | 1669 | 0.9732 | 0.9145 | 0.9819 | 0.9565 |
| 0.70 | 0.2641 | 1555 | 0.9750 | 0.9204 | 0.9844 | 0.9600 |
| 0.75 | 0.3124 | 1453 | 0.9796 | 0.9257 | 0.9844 | 0.9632 |

---

## Thesis narrative

**The 9-feature model is the recommended production candidate.** Removing `hour_of_day_*` and `prev_env_class`:

1. **Does not degrade performance** — macro F1 at τ=0.62 is within 0.0002 of the 12-feature model
2. **Improves calibration** — ECE drops by 0.0024–0.0078 across all night classes, meaning confidence scores are more trustworthy for the `ml_confidence_threshold=0.62` gate
3. **Simplifies deployment** — the model no longer requires session-level temporal state at inference time; each frame is classified independently

This validates the offline optical features (`nir_mean_brightness`, `nir_std`, `nir_entropy`, `nir_p95`, `nir_glare_score`, `nir_sharpness`, `nir_dark_fraction`, `nir_saturation_mean`, `nir_blue_mean_ema`) as sufficient discriminators for the night/NIR taxonomy without temporal context.

**Non-claim:** This result does not imply that temporal features are universally unhelpful — they may improve generalization on longer field sessions with strong time-of-day signals. The finding is specific to this dataset (dominated by still images from weather datasets) and this task (9-class ENV taxonomy).
