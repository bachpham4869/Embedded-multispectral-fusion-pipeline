# Metric Discrepancy Investigation

Status: open after Phase 2. Frozen held-out metrics remain
`preliminary / not thesis-ready` because source overlap remains, image-level
duplicates cannot be checked, and removing feature-vector-overlap test records
does not explain the gap versus TimeSeriesSplit CV.

## Observed Discrepancy

The current sidecar reports TimeSeriesSplit CV balanced accuracy around 0.7408,
while frozen held-out evaluation on `data/training/from_logs_test.jsonl`
reports much higher metrics:

```bash
.venv/bin/python models/train_classifier.py \
  --mode optical_only \
  --evaluate-model models/production/env_classifier.joblib \
  --test-dataset data/training/from_logs_test.jsonl
```

Run provenance:

| Field | Value |
| --- | --- |
| git_commit | `15fbe64d5c70873cf200eac688f37d03e8f103da` |
| model_path | `models/production/env_classifier.joblib` |
| model_sha256 | `00c4f0eecb6e7cc56b77cfea911c447e1f0e9448d0ec984ab9bc799c637d6b48` |
| model_sidecar_sha256 | `791012caed008c4ac573aef513ef247e73033880254e9474de0671dd78d1b5ba` |
| test_dataset | `data/training/from_logs_test.jsonl` |
| test_dataset_sha256 | `583ce85f80f1c065de2405c4bdccdab97098dcb2440dad4f4237bfbb2b662699` |
| test_rows | 2113 |
| feature_set_version | `optical_12_baseline` |
| split_method | Existing frozen JSONL file; no internal split |
| random_seed | Not used for full-file eval |
| python/sklearn/numpy/scipy | 3.12.12 / 1.8.0 / 1.26.4 / 1.17.1 |
| metric_status | `preliminary / not thesis-ready` |

## Current Frozen Evaluation Snapshot

These numbers are recorded only to investigate the discrepancy, not as final
thesis evidence.

| Metric | Value | Status |
| --- | --- | --- |
| Accuracy | 0.9389 | preliminary / not thesis-ready |
| Balanced accuracy | 0.9281 | preliminary / not thesis-ready |
| Macro-F1 | 0.916 | preliminary / not thesis-ready |
| Mean max probability | 0.8588 | proxy confidence only |

Per-class support and F1:

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| night_clear | 0.988 | 0.976 | 0.982 | 417 |
| normal_night | 0.927 | 0.935 | 0.931 | 352 |
| normal_day | 0.936 | 0.913 | 0.924 | 366 |
| fog | 0.922 | 0.918 | 0.920 | 231 |
| rain | 0.920 | 0.887 | 0.903 | 194 |
| glare | 0.844 | 0.900 | 0.871 | 60 |
| backlight | 0.783 | 0.982 | 0.871 | 55 |
| transition | 0.868 | 0.857 | 0.863 | 77 |
| nir_night | 0.981 | 0.986 | 0.983 | 361 |

Raw confusion matrix, rows=true, columns=predicted:

| True \ Pred | night_clear | normal_night | normal_day | fog | rain | glare | backlight | transition | nir_night |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| night_clear | 407 | 9 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| normal_night | 5 | 329 | 3 | 0 | 2 | 6 | 5 | 2 | 0 |
| normal_day | 0 | 3 | 334 | 11 | 7 | 1 | 2 | 4 | 4 |
| fog | 0 | 0 | 8 | 212 | 5 | 0 | 0 | 4 | 2 |
| rain | 0 | 1 | 7 | 3 | 172 | 2 | 8 | 0 | 1 |
| glare | 0 | 6 | 0 | 0 | 0 | 54 | 0 | 0 | 0 |
| backlight | 0 | 1 | 0 | 0 | 0 | 0 | 54 | 0 | 0 |
| transition | 0 | 4 | 5 | 1 | 0 | 1 | 0 | 66 | 0 |
| nir_night | 0 | 2 | 0 | 3 | 0 | 0 | 0 | 0 | 356 |

Normalized confusion matrix is not emitted by `models/train_classifier.py`; use
`tools/compare_classifiers.py` for raw and normalized confusion CSVs in future
benchmark runs.

## Train/Test Leakage Check

Command:

```bash
.venv/bin/python tools/check_dataset_leakage.py \
  --train data/training/from_logs_train.jsonl \
  --test data/training/from_logs_test.jsonl \
  --out docs/tables/ml/leakage_check_summary.md \
  --feature-overlap-details-md docs/tables/ml/feature_vector_overlap_details.md \
  --feature-overlap-details-csv artifacts/ml/leakage/feature_vector_overlaps.csv
```

Current result:

- exact row hash overlap: 0
- path overlap: 0, but JSONL has no path-like metadata, so image/path overlap
  cannot actually be checked
- filename overlap: 0, with the same metadata limitation
- session/frame overlap: 0, but JSONL has no session metadata
- source overlap: 8 sources overlap across train/test
- optical_12 feature-vector hash overlap: 21 unique test records, 22 train/test
  row pairs because one test feature vector matches two train rows
- perceptual hash near-duplicate check: not run because source image paths are
  not present in JSONL

Interpretation: there is no exact JSON-row duplicate in the current JSONL, but
the 21 feature-vector duplicate hashes are a warning signal. They may be
legitimate repeated feature values or duplicate/near-duplicate images. This
must be resolved before using frozen held-out metrics as thesis evidence.

Detailed output:

- `docs/tables/ml/feature_vector_overlap_details.md`
- `artifacts/ml/leakage/feature_vector_overlaps.csv`

All 22 overlap pairs are exact `optical_12_baseline` feature-vector matches.
They are not rounding artifacts. Every pair lacks original image path/hash
metadata and session metadata, so image-level duplicate status remains
unverified.

## Phase 2 Sensitivity Evaluation

Command:

```bash
.venv/bin/python tools/evaluate_frozen_classifier.py \
  --model models/production/env_classifier.joblib \
  --train data/training/from_logs_train.jsonl \
  --test data/training/from_logs_test.jsonl \
  --out-dir artifacts/ml/eval \
  --summary docs/tables/ml/frozen_eval_overlap_sensitivity.md
```

Guardrail: this command loads `models/production/env_classifier.joblib`
read-only. The train JSONL is used only to identify feature-vector overlaps and
metadata limitations. No refit, retrain, or recalibration is performed.

| Variant | Rows | Accuracy | Balanced accuracy | Macro-F1 | Weighted-F1 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| original | 2113 | 0.9389 | 0.9281 | 0.9164 | 0.9392 | preliminary / not thesis-ready |
| exclude_test_feature_vector_overlaps | 2092 | 0.9383 | 0.9265 | 0.9149 | 0.9385 | preliminary / not thesis-ready |

Interpretation: excluding the 21 overlap-related test records changes balanced
accuracy by about -0.0016 and macro-F1 by about -0.0015. Therefore the 0.7408
CV versus 0.9281 frozen held-out discrepancy is not explained by these
feature-vector overlaps alone.

## Source Overlap

All current source families appear in both train and test:
`offline_backlight`, `offline_darkface`, `offline_exdark_street`,
`offline_glare_street`, `offline_gray_nir`, `offline_mwd`,
`offline_weather11`, and `offline_weather_time`.

Source overlap is not automatically leakage because the existing split is
stratified by rows, but it can make the held-out set easier than a source-held-
out or session-held-out evaluation. A thesis-ready split should include either
source-grouped CV or a source/session-held-out audit.

## Is the Test Set Easier Than CV?

Likely possible, not proven yet. Phase 2 strengthens this hypothesis because
removing feature-vector-overlap test rows does not materially lower the frozen
held-out score.

Reasons:

- Sidecar CV uses `TimeSeriesSplit` over the merged dataset order. If source
  order is clustered, CV folds may train on earlier sources and validate on
  later source distributions, making CV harder.
- Frozen held-out `from_logs_test.jsonl` appears to preserve the same source
  proportions as train/reference. It may be easier because each source appears
  in both train and test.
- Low-support classes (`glare`, `backlight`, `transition`) have 55-77 test rows,
  so class-level F1 can move materially with a small number of samples.
- The model sidecar was trained on `merged_logs_ml.jsonl` (14,094 rows). The
  frozen test file is a split from the same data lineage, so independence must
  be demonstrated at image/session/source level.

## CV Strategy vs Frozen Evaluation

| Aspect | Sidecar CV | Frozen held-out eval |
| --- | --- | --- |
| Rows | 14,094 merged rows | 2,113 test rows |
| Split | TimeSeriesSplit, 5 folds | Existing full test JSONL |
| Scaler | Fit per fold on train fold | Production scaler from full training bundle |
| Model | RF per fold | Frozen calibrated production bundle |
| Calibration | Sidecar says isotonic | Frozen bundle loaded with calibration |
| Leakage resistance | Better for temporal/order leakage if rows are ordered by session/source | Depends on split construction and metadata; source overlap exists |
| Current status | Sidecar metric lacks full provenance | Eval metric preliminary due discrepancy |

Most likely current explanation: the sidecar TimeSeriesSplit is harder because
row order may be clustered by source/time, while the frozen train/test split
shares all major source families and may preserve similar source distributions.
This is an inference from current artifacts, not a proven conclusion, because
the JSONL lacks original image path/hash/session fields.

## Low-Support Inflation Risk

`glare` support is 60, `backlight` support is 55, and `transition` support is
77 in the frozen test set. Their high F1 values should not be used as strong
class claims without confidence intervals and source-diversity review.

The Phase 2 quick benchmark in
`docs/tables/ml/model_comparison_12features.md` shows much lower F1 for
`glare`, `backlight`, and `transition` than the frozen production model. This
reinforces that those class claims should remain provisional until leakage,
source diversity, and confidence intervals are resolved.

## Same-Feature Benchmark Context

Phase 2 ran a quick direct benchmark with the same current train/test split and
`optical_12_baseline` features:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python tools/compare_classifiers.py \
  --train data/training/from_logs_train.jsonl \
  --test data/training/from_logs_test.jsonl \
  --models logistic_regression linear_svm sgd_linear_svm gaussian_nb decision_tree random_forest_50 random_forest_100 random_forest_200_current_config random_forest_depth_8 random_forest_depth_12 random_forest_depth_none extra_trees gradient_boosting hist_gradient_boosting knn mlp_small_32 mlp_small_64_32 \
  --bootstrap 100 \
  --latency-repeats 100 \
  --hardware-label "macOS proxy benchmark (not Raspberry Pi 4 CPU)" \
  --persist-models \
  --quick
```

Best quick same-feature family: Random Forest. RF100 and RF200 are close on
balanced accuracy. This benchmark is useful engineering evidence but not final
thesis evidence because it uses the unresolved current split and a macOS proxy
latency environment.

## Which Numbers Can Be Used in Report?

Allowed now:

- data distribution counts and percentages with provenance
- statement that current train/test JSONL is RGB proxy only
- statement that leakage check is metadata-limited and found 21 feature-vector
  duplicate hashes
- sidecar RF hyperparameter/configuration description
- Phase 2 quick benchmark as preliminary engineering evidence, with the
  `quick / preliminary / not thesis-ready` label

Not allowed as primary thesis evidence yet:

- frozen held-out accuracy 0.9389
- frozen held-out balanced accuracy 0.9281
- frozen held-out macro-F1 0.916
- strong per-class claims for `glare`, `backlight`, or `transition`
- final deployment latency from macOS benchmark
- NIR sensor validation claims

Needed to close this investigation:

- source-image path manifest or image SHA256 manifest for train/test
- pHash near-duplicate check using original images
- source-held-out or group-aware split comparison
- bootstrap CI for the production RF frozen eval
- source-aware or leakage-clean same-feature benchmark under identical split
- live NIR domain-shift evaluation

## Phase 3 Update

Phase 3 recovered verified raw-image metadata for 5,791 of 14,094 merged rows.
The remaining 8,303 rows are `inferred_low_confidence` because their source
families were shuffled/subsampled and the original JSONL has no path/hash
identity.

Current enriched train/test split:

- exact file SHA256 overlap: 22 pairs within verified coverage
- dHash near-duplicate screening pairs at Hamming threshold <= 4: 1,419
- source-name overlap: 8 sources

The group-aware split uses `split_group_id`, with verified rows grouped by
`file_sha256::<sha>`.

- exact file SHA256 overlap: 0
- split-group overlap: 0
- feature-vector weak-hint overlap: 0
- dHash near-duplicate screening pairs at Hamming threshold <= 4: 1,313
- source-name overlap: 8 sources

Conclusion: Phase 3 reduces exact duplicate leakage in verified coverage but
does not close visual near-duplicate risk or source overlap. The frozen held-out
metric discrepancy remains unresolved for thesis purposes. Group-aware benchmark
results are stronger engineering evidence than Phase 2, but still
`preliminary / not thesis-ready`.

## Phase 4 Update

Phase 4 rebuilt duplicate clusters over the full
`data/training_v2/merged_logs_ml_metadata.jsonl` file, not only the Phase 3
cross-split dHash pair CSV.

Two cluster modes were generated:

| Mode | dHash threshold | Clusters | Edges | Max cluster size | Feasible split |
| --- | ---: | ---: | ---: | ---: | --- |
| strict | <= 2 | 13,673 | 1,380 | 100 | yes |
| conservative | <= 4 | 13,375 | 4,447 | 333 | yes |

The benchmark uses the conservative split because it preserves every ENV class
in train and test and keeps the test ratio at 0.1499.

Conservative cluster-aware leakage check:

- exact JSON-row overlap: 0
- exact file SHA256 overlap: 0 within file-hash coverage
- split-group overlap: 0
- duplicate-cluster overlap: 0
- feature-vector weak-hint overlap: 0
- dHash cross-split screening pairs at Hamming threshold <= 4: 0
- source-name overlap: 8 sources

This closes the measured dHash blocker within records that have dHash coverage.
It does not prove absence of all visual duplicates because 8,303 rows remain
`inferred_low_confidence` and do not have verified image hash coverage.

## Current Thesis Evidence Decision

The Phase 4 same-feature benchmark can be used as offline optical RGB-proxy
evidence with caveats. It should be introduced as a duplicate-cluster-aware
offline evaluation, not as a live NIR/LWIR validation and not as a
source-held-out evaluation.

Allowed with caveat:

- Phase 4 conservative cluster-aware same-feature benchmark in
  `docs/tables/ml/model_comparison_cluster_aware.md`
- Phase 4 leakage summary in
  `docs/tables/ml/cluster_aware_leakage_summary.md`
- split progression table in `docs/tables/ml/split_benchmark_progression.md`

Still not allowed as primary thesis evidence:

- frozen held-out values from the original production split
- macOS proxy latency as final RPi4 CPU latency
- strong claims for `glare`, `backlight`, or `transition`
- NIR/LWIR deployment validation claims
