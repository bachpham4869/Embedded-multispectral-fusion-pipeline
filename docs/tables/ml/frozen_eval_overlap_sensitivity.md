# Frozen Eval Overlap Sensitivity

Status: preliminary / not thesis-ready. Production model is loaded read-only; no refit, retrain, or recalibration is performed.

## Reproducibility

| Field | Value |
| --- | --- |
| command | /Users/phongpham/Downloads/smartBinocular/.venv/bin/python tools/evaluate_frozen_classifier.py --model models/production/env_classifier.joblib --train data/training/from_logs_train.jsonl --test data/training/from_logs_test.jsonl --out-dir artifacts/ml/eval --summary docs/tables/ml/frozen_eval_overlap_sensitivity.md |
| git_commit | 15fbe64d5c70873cf200eac688f37d03e8f103da |
| branch | refactor/ml-taxonomy-eval-plan |
| model_path | models/production/env_classifier.joblib |
| model_sha256 | 00c4f0eecb6e7cc56b77cfea911c447e1f0e9448d0ec984ab9bc799c637d6b48 |
| dataset_manifest_hash | 416c8ce44c6882625e1a8dd6c7bcd492f78a1177cb2baf0e01bf358ea91dc2ee |
| train_rows | 11981 |
| test_rows | 2113 |
| feature_set_version | optical_12_baseline (12 features) |
| split_method | existing frozen JSONL test file; sensitivity excludes feature-vector-overlap test rows |
| random_seed | not used |
| python | 3.12.12 |
| sklearn | 1.8.0 |
| numpy | 1.26.4 |
| scipy | 1.17.1 |
| metric_status | preliminary / not thesis-ready |

## Dataset Manifest

| Path | SHA256 | Bytes |
| --- | --- | --- |
| data/training/from_logs_train.jsonl | 0a03e757a434956df1af4e5595d4bfc029242957b1cad15cb0053dfc0a744410 | 11501860 |
| data/training/from_logs_test.jsonl | 583ce85f80f1c065de2405c4bdccdab97098dcb2440dad4f4237bfbb2b662699 | 2027542 |

## Aggregate Metrics

| Variant | Rows | Accuracy | Balanced accuracy | Macro-F1 | Weighted-F1 | Status |
| --- | --- | --- | --- | --- | --- | --- |
| original | 2113 | 0.9389 | 0.9281 | 0.9164 | 0.9392 | preliminary / not thesis-ready |
| exclude_test_feature_vector_overlaps | 2092 | 0.9383 | 0.9265 | 0.9149 | 0.9385 | preliminary / not thesis-ready |

## Interpretation

- Original held-out rows evaluated: 2113.
- No-feature-overlap rows evaluated: 2092 after excluding 21 test records.
- If the metrics remain materially higher than sidecar CV, source overlap / split difficulty / missing image-level duplicate checks remain likely explanations.