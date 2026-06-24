# SmartBinocular ML Phase 1 Refinement Plan

Status: Phase 1 implementation-facing audit and plan. No production retrain,
schema rename, class drop/merge, dataset mixing, or `models/production/env_classifier.joblib`
overwrite is authorized by this document.

## Locked Constraints

- Current model wording in report/thesis: `optical RGB-proxy baseline` or
  `optical-only proxy baseline`. Do not call it an NIR-trained classifier until
  live IMX290/NIR logs, manual labels, and a domain-shift report exist.
- TFLite/image-input candidates are allowed only as separate baselines, not as
  direct same-feature competitors to the 12-feature RF.
- Deployment target for benchmark language: Raspberry Pi 4 CPU, 20 FPS system
  goal. Non-RPi latency is a proxy benchmark.
- Dataset candidate metadata may be prepared, but no large dataset download and
  no candidate dataset mixing into train/test without user confirmation.
- No production model retrain and no production model overwrite without user
  confirmation.
- No schema-breaking ENV rename in Phase 1. Code keeps current class names;
  docs define report-facing aliases.
- GitNexus is preferred for future edits to indexed symbols, but optional. If it
  is unavailable, use `rg`, `git grep`, targeted tests, manual import/call
  search, and static inspection.

## Required Evidence Standard

Every thesis/report metric must include:

- executed command
- git commit hash
- model path and SHA256, or explicit in-memory benchmark hash
- dataset path and hash/manifest
- train/test row counts
- feature-set version
- split method
- random seed
- Python/sklearn/numpy/scipy versions
- per-class support
- raw and normalized confusion matrix for classification metrics

If any field is missing, mark the metric `preliminary / not thesis-ready`.
Current frozen held-out values (`accuracy=0.9389`, `balanced_accuracy=0.9281`,
`macro-F1=0.916`) are not thesis evidence until
`docs/ml/METRIC_DISCREPANCY_INVESTIGATION.md` is closed.

## Current ML State

Current ENV classes in `src/smartbinocular/feature_schema.py`:
`night_clear`, `normal_night`, `normal_day`, `fog`, `rain`, `glare`,
`backlight`, `transition`, `nir_night`.

Current feature set: `FEATURE_SET_OPTICAL_ONLY` has 12 optical proxy features:
`nir_mean_brightness`, `nir_std`, `nir_entropy`, `nir_p95`,
`nir_glare_score`, `nir_sharpness`, `nir_dark_fraction`,
`nir_saturation_mean`, `hour_of_day_sin`, `hour_of_day_cos`,
`prev_env_class`, `nir_blue_mean_ema`.

Current training modes in `models/train_classifier.py`: `optical_only`,
`with_thermal`, `rgb_thermal_ablation`, and `optical_only_9`.

Current RF hyperparameters in `train_and_evaluate`: `n_estimators=200` default,
`max_depth=20`, `min_samples_leaf=4`, `max_features="sqrt"`,
`random_state=42`, `n_jobs=-1`, `class_weight="balanced"`, TimeSeriesSplit CV,
and isotonic calibration.

Current production sidecar: `models/production/env_classifier.json`, promoted
from `models/baseline/rf_phase1_retrain_optical12.joblib`, training mode
`optical_only`, 14,094 samples, 12 features, CV balanced accuracy mean
0.7408, train balanced accuracy 0.9509. These sidecar metrics need full
provenance before thesis use.

Current runtime flow: `FeatureExtractor` emits 12 optical proxy features;
`MLInferenceThread` runs a background classifier every configured interval;
`EnvClassifier` validates feature schema and returns top-2 probabilities;
`compose_env_from_ml_top2` gates by `tau1=0.62` and `tau2=0.20`; `main.py`
routes the selected ENV class into `OPTICAL_BUCKET_DISPATCH` in
`nir_pipeline.py`.

Current dataset sources in generated JSONL: `offline_darkface`,
`offline_exdark_street`, `offline_weather_time`, `offline_weather11`,
`offline_mwd`, `offline_gray_nir`, `offline_glare_street`, and
`offline_backlight`. Generated train/test rows are all `nir_channel=rgb` and
`thermal_channel=none`, so this is an optical RGB-proxy baseline.

## Problems Mapped to GV Feedback

| GV feedback | Phase 1 response | Evidence/output |
| --- | --- | --- |
| Class/bucket names are ambiguous | Add report-facing ENV taxonomy and bucket policy docs; no code rename yet | `docs/ml/ENV_TAXONOMY.md`, `docs/ml/PROCESSING_BUCKETS.md` |
| Missing data distribution | Add reproducible distribution script and generated tables/figures | `tools/analyze_dataset_distribution.py`, `docs/tables/ml/dataset_distribution_*.md` |
| Decide keep/drop class from data | Add risk table and transition decision record | `docs/tables/ml/class_support_risk_summary.md`, `docs/ml/DECISION_transition_class.md` |
| 12 features are basic | Add feature engineering guardrails and v2 candidate plan | This file, future `FEATURE_SET_RATIONALE.md` |
| RF hyperparameters unclear | Document exact RF settings and compare configs in benchmark script | This file, `tools/compare_classifiers.py` |
| RF not proven lightest/best | Add same-feature classical comparison script with CI/latency | `tools/compare_classifiers.py`, `docs/tables/ml/model_comparison_12features.md` |
| Need specialized model comparison | Add research survey separating direct vs image-input/literature comparison | `docs/ml/MODEL_RESEARCH_SURVEY.md` |
| Domain shift not handled | Add domain shift plan and wording constraints | `docs/ml/DOMAIN_SHIFT_EVALUATION.md` |
| Fusion simple | Record ML-to-bucket/fusion policy dependency; do not expand fusion in Phase 1 | `docs/ml/PROCESSING_BUCKETS.md` |
| Need metrics | Add reproducibility standard, leakage protocol, CI benchmark scaffold | This file, `docs/ml/DATA_SPLIT_AND_LEAKAGE_PROTOCOL.md` |

## Contradictions and Gaps Found

- README says `ENV_PRESETS` has 14 named presets; current
  `build_env_presets()` defines 21 entries including ENV-class-aligned presets.
- `FEATURE_SET_OPTICAL_THERMAL` comment says 17 features, but current code is
  12 core + 6 thermal = 18.
- `models/README.md` retrain step mentions `--input`; current
  `models/train_classifier.py` uses `--dataset`.
- README includes strong ML routing numbers and "macro night F1=0.98"; these
  need the evidence standard above before thesis use.
- Current train/test JSONL has only `nir_channel=rgb`, so any NIR deployment
  claim is a domain-shift claim not supported by current distribution tables.

## Proposed Taxonomy

Phase 1 keeps code names and introduces report-facing aliases:

| Current class | Report-facing alias | Phase 1 status |
| --- | --- | --- |
| `night_clear` | `dark_clear_optical` | Keep |
| `normal_night` | `urban_ambient_night` | Keep |
| `normal_day` | `clear_daylight` | Keep |
| `fog` | `reduced_visibility_fog_haze` | Keep |
| `rain` | `rain_wet_scene` | Keep |
| `glare` | `direct_glare_highlight` | Provisional |
| `backlight` | `backlit_high_dynamic_range` | Provisional |
| `transition` | `dawn_dusk_transition` | Weak ENV class; consider runtime state |
| `nir_night` | `nir_assisted_mono_night` | Keep as proxy until live validation |

Processing buckets should be algorithm/policy names, not environment names:
`A=night_hybrid_enhance`, `B=nir_mono_clahe`, `C=highlight_tone_map`,
`D=fog_dehaze_lite`, `E=rain_temporal_median`, `F=dawn_dusk_blend`.

## Data Distribution Plan

Implemented script:

```bash
.venv/bin/python tools/analyze_dataset_distribution.py \
  --train data/training/from_logs_train.jsonl \
  --test data/training/from_logs_test.jsonl \
  --reference data/training/merged_logs_ml.jsonl \
  --out-dir docs/tables/ml \
  --fig-dir docs/figures/ml
```

Generated outputs include per-class train/test/reference counts, percentages,
source contribution, label source distribution, label confidence summary,
channel distribution, imbalance ratio, low-support flags, and class risk
recommendations.

## Dataset Candidate Search Plan

No dataset is downloaded in Phase 1. Candidate review should cover:

- weather/adverse condition classification: fog/haze, rain, glare, backlight,
  dawn/dusk
- low-light/night/NIR: low-light outdoor, night driving, visible-infrared
  paired datasets
- RGB-T/thermal: visible-thermal aligned datasets for optional ablation

For each candidate, create metadata under
`data/_candidate_datasets/<dataset_name>/SOURCE.md` and
`mapping_proposal.yaml`, then ask for confirmation before download or mixing.

## Model Comparison Plan

Direct benchmark: same train/test split, same 12 handcrafted features, fixed
seed, no test tuning. Models: Logistic Regression, Linear SVM/SGD, GaussianNB,
Decision Tree, RF variants, ExtraTrees, GradientBoosting/HistGradientBoosting,
KNN, and small MLP. The implemented script supports bootstrap 95% CI,
per-class support/F1 CI, ECE/Brier when `predict_proba` exists, model size,
training time, and mean/median/p95 latency.

Image-input/TFLite baselines (MobileNetV3, EfficientNet-Lite, ShuffleNet,
MobileOne, ResNet18-lite, Places-style models) must be reported separately
because the input is image pixels, not the 12-feature vector.

## Feature Engineering Plan

Feature candidates require a feature-set ID, runtime cost estimate, live NIR
compatibility assessment, leakage review, and benchmark evidence before schema
changes. Candidate families: edge/texture, haze/fog, glare/backlight,
rain-streak or temporal residuals, night/NIR color ratios, temporal stability,
and thermal statistics when real thermal logs exist.

No production feature schema change is authorized until a backward-compatible
loader, migration note, and benchmark improvement justify the added complexity.

## Domain Shift Plan

Current JSONL train/test distribution is RGB proxy only. The domain-shift plan
requires live IMX290/NIR logs, manual labels, feature distribution comparison,
confidence shift, calibration shift, and live confusion matrix where labels
exist. Until then, the model is not "NIR sensor validated".

## Files Modified in Phase 1

- New scripts: `tools/analyze_dataset_distribution.py`,
  `tools/check_dataset_leakage.py`, `tools/compare_classifiers.py`
- New tests: `tests/test_ml_phase1_tools.py`
- New docs under `docs/ml/`
- Generated tables under `docs/tables/ml/`
- Generated figures under `docs/figures/ml/`
- Benchmark confusion CSVs under `artifacts/ml/model_comparison_12features/`

## Risks

- Source overlap is expected in the current stratified split but can inflate
  frozen held-out metrics if near-duplicate frames/images cross the split.
- JSONL lacks path/session metadata, so source-image duplicate and pHash checks
  cannot be completed from the current files alone.
- `glare`, `backlight`, and `transition` have low support; confidence intervals
  and source-diversity review are required before strong claims.
- Non-RPi latency is a proxy benchmark only.
- Report-facing aliases may confuse readers if code names are not mapped
  consistently in thesis tables.

## Questions for User Confirmation

- Can Phase 2 collect or access live IMX290/NIR logs with manual labels?
- What maximum dataset download size is acceptable for candidate review?
- Are research-only dataset licenses acceptable for thesis experiments, or must
  all datasets allow redistribution/commercial-like usage?
- Should deep learning baselines be benchmarked only as TFLite image-input
  references, or should one small model be trained for ablation after Phase 1?
- Can `transition` be migrated to a runtime state if Phase 2 evidence remains
  weak?

## Milestone Checklist

- [x] Create isolated branch for ML Phase 1.
- [x] Add distribution analysis script with `--help`.
- [x] Add leakage check script with `--help`.
- [x] Add same-feature comparison script with `--help`.
- [x] Generate train/test/reference distribution tables and figures.
- [x] Generate metadata-level leakage summary.
- [x] Run a preliminary smoke benchmark without saving production models.
- [x] Create taxonomy, bucket, discrepancy, domain, report, and review docs.
- [ ] Run full same-feature benchmark on all models.
- [ ] Resolve metric discrepancy before thesis claims.
- [ ] Collect live NIR/sensor logs and run domain-shift report.
