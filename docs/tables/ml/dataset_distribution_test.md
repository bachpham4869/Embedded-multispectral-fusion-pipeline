# Dataset Distribution: test

Status: distribution evidence. Classification metrics are not included here.

## Reproducibility

| Field | Value |
| --- | --- |
| command | /Users/phongpham/Downloads/smartBinocular/.venv/bin/python tools/analyze_dataset_distribution.py --train data/training/from_logs_train.jsonl --test data/training/from_logs_test.jsonl --reference data/training/merged_logs_ml.jsonl --out-dir docs/tables/ml --fig-dir docs/figures/ml |
| git_commit | 15fbe64d5c70873cf200eac688f37d03e8f103da |
| dataset_manifest_hash | 4e52f18848fe9a11b120d9cc73b3af28c655a3c3be687c3edac45a314e5d68bf |
| row_count | 2113 |
| feature_set_version | optical_12_baseline (12 features) |
| python | 3.12.12 |
| sklearn | 1.8.0 |
| numpy | 1.26.4 |
| scipy | 1.17.1 |

Dataset files:

| Path | SHA256 | Bytes |
| --- | --- | --- |
| data/training/from_logs_test.jsonl | 583ce85f80f1c065de2405c4bdccdab97098dcb2440dad4f4237bfbb2b662699 | 2027542 |

## Per-Class Distribution

| Class | Count | Percent | Mean Confidence | Top Sources | Recommendation |
| --- | --- | --- | --- | --- | --- |
| night_clear | 417 | 19.73 | 0.800 | offline_darkface:417 | keep for current audit; still requires leakage and domain-shift checks |
| normal_night | 352 | 16.66 | 0.760 | offline_exdark_street:352 | keep for current audit; still requires leakage and domain-shift checks |
| normal_day | 366 | 17.32 | 0.793 | offline_weather_time:282, offline_mwd:84 | keep for current audit; still requires leakage and domain-shift checks |
| fog | 231 | 10.93 | 0.730 | offline_weather11:231 | keep for current audit; still requires leakage and domain-shift checks |
| rain | 194 | 9.18 | 0.900 | offline_weather_time:88, offline_weather11:69, offline_mwd:37 | keep for current audit; still requires leakage and domain-shift checks |
| glare | 60 | 2.84 | 0.780 | offline_glare_street:60 | provisional: below support threshold; need more data or merge/drop review |
| backlight | 55 | 2.60 | 0.780 | offline_backlight:55 | provisional: below support threshold; need more data or merge/drop review |
| transition | 77 | 3.64 | 0.786 | offline_mwd:56, offline_weather_time:21 | provisional: below support threshold; need more data or merge/drop review |
| nir_night | 361 | 17.08 | 0.800 | offline_gray_nir:361 | keep for current audit; still requires leakage and domain-shift checks |

## Label and Channel Summary

| Category | Distribution |
| --- | --- |
| label_source | dataset_original:2113 |
| nir_channel | rgb:2113 |
| thermal_channel | none:2113 |
| imbalance_ratio_max_min | 7.582 |
| low_support_classes | backlight, glare, transition |

## Label Confidence

| Count | Min | P25 | Median | Mean | P75 | Max |
| --- | --- | --- | --- | --- | --- | --- |
| 2113 | 0.600 | 0.760 | 0.800 | 0.792 | 0.800 | 0.900 |
