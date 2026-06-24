# Dataset Distribution: reference

Status: distribution evidence. Classification metrics are not included here.

## Reproducibility

| Field | Value |
| --- | --- |
| command | /Users/phongpham/Downloads/smartBinocular/.venv/bin/python tools/analyze_dataset_distribution.py --train data/training/from_logs_train.jsonl --test data/training/from_logs_test.jsonl --reference data/training/merged_logs_ml.jsonl --out-dir docs/tables/ml --fig-dir docs/figures/ml |
| git_commit | 15fbe64d5c70873cf200eac688f37d03e8f103da |
| dataset_manifest_hash | f299a629df4ab61f26da3df3c3b3975b57e7da1a9a8c8e470f715bf593c10e79 |
| row_count | 14094 |
| feature_set_version | optical_12_baseline (12 features) |
| python | 3.12.12 |
| sklearn | 1.8.0 |
| numpy | 1.26.4 |
| scipy | 1.17.1 |

Dataset files:

| Path | SHA256 | Bytes |
| --- | --- | --- |
| data/training/merged_logs_ml.jsonl | 09b6ca5ee4015a3a3f078c6a485d4e0e4cdd4904073e918fe603c1de57668760 | 13529402 |

## Per-Class Distribution

| Class | Count | Percent | Mean Confidence | Top Sources | Recommendation |
| --- | --- | --- | --- | --- | --- |
| night_clear | 2781 | 19.73 | 0.800 | offline_darkface:2781 | keep for current audit; still requires leakage and domain-shift checks |
| normal_night | 2347 | 16.65 | 0.760 | offline_exdark_street:2347 | keep for current audit; still requires leakage and domain-shift checks |
| normal_day | 2443 | 17.33 | 0.796 | offline_weather_time:1891, offline_mwd:552 | keep for current audit; still requires leakage and domain-shift checks |
| fog | 1542 | 10.94 | 0.738 | offline_weather11:1542 | keep for current audit; still requires leakage and domain-shift checks |
| rain | 1291 | 9.16 | 0.900 | offline_weather_time:551, offline_weather11:526, offline_mwd:214 | keep for current audit; still requires leakage and domain-shift checks |
| glare | 400 | 2.84 | 0.780 | offline_glare_street:400 | provisional: below support threshold; need more data or merge/drop review |
| backlight | 368 | 2.61 | 0.780 | offline_backlight:368 | provisional: below support threshold; need more data or merge/drop review |
| transition | 515 | 3.65 | 0.785 | offline_mwd:357, offline_weather_time:158 | provisional: verify as ENV class vs dawn_dusk_blend runtime state |
| nir_night | 2407 | 17.08 | 0.800 | offline_gray_nir:2407 | keep for current audit; still requires leakage and domain-shift checks |

## Label and Channel Summary

| Category | Distribution |
| --- | --- |
| label_source | dataset_original:14094 |
| nir_channel | rgb:14094 |
| thermal_channel | none:14094 |
| imbalance_ratio_max_min | 7.557 |
| low_support_classes | backlight, glare |

## Label Confidence

| Count | Min | P25 | Median | Mean | P75 | Max |
| --- | --- | --- | --- | --- | --- | --- |
| 14094 | 0.600 | 0.760 | 0.800 | 0.793 | 0.800 | 0.900 |
