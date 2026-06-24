# Dataset Distribution: train

Status: distribution evidence. Classification metrics are not included here.

## Reproducibility

| Field | Value |
| --- | --- |
| command | /Users/phongpham/Downloads/smartBinocular/.venv/bin/python tools/analyze_dataset_distribution.py --train data/training/from_logs_train.jsonl --test data/training/from_logs_test.jsonl --reference data/training/merged_logs_ml.jsonl --out-dir docs/tables/ml --fig-dir docs/figures/ml |
| git_commit | 15fbe64d5c70873cf200eac688f37d03e8f103da |
| dataset_manifest_hash | 464f2179bb78e0ca6834e4062707b77ddbe6d060e5da28d3f9d22b9dc0fa6dd8 |
| row_count | 11981 |
| feature_set_version | optical_12_baseline (12 features) |
| python | 3.12.12 |
| sklearn | 1.8.0 |
| numpy | 1.26.4 |
| scipy | 1.17.1 |

Dataset files:

| Path | SHA256 | Bytes |
| --- | --- | --- |
| data/training/from_logs_train.jsonl | 0a03e757a434956df1af4e5595d4bfc029242957b1cad15cb0053dfc0a744410 | 11501860 |

## Per-Class Distribution

| Class | Count | Percent | Mean Confidence | Top Sources | Recommendation |
| --- | --- | --- | --- | --- | --- |
| night_clear | 2364 | 19.73 | 0.800 | offline_darkface:2364 | keep for current audit; still requires leakage and domain-shift checks |
| normal_night | 1995 | 16.65 | 0.760 | offline_exdark_street:1995 | keep for current audit; still requires leakage and domain-shift checks |
| normal_day | 2077 | 17.34 | 0.796 | offline_weather_time:1609, offline_mwd:468 | keep for current audit; still requires leakage and domain-shift checks |
| fog | 1311 | 10.94 | 0.739 | offline_weather11:1311 | keep for current audit; still requires leakage and domain-shift checks |
| rain | 1097 | 9.16 | 0.900 | offline_weather_time:463, offline_weather11:457, offline_mwd:177 | keep for current audit; still requires leakage and domain-shift checks |
| glare | 340 | 2.84 | 0.780 | offline_glare_street:340 | provisional: below support threshold; need more data or merge/drop review |
| backlight | 313 | 2.61 | 0.780 | offline_backlight:313 | provisional: below support threshold; need more data or merge/drop review |
| transition | 438 | 3.66 | 0.784 | offline_mwd:301, offline_weather_time:137 | provisional: below support threshold; need more data or merge/drop review |
| nir_night | 2046 | 17.08 | 0.800 | offline_gray_nir:2046 | keep for current audit; still requires leakage and domain-shift checks |

## Label and Channel Summary

| Category | Distribution |
| --- | --- |
| label_source | dataset_original:11981 |
| nir_channel | rgb:11981 |
| thermal_channel | none:11981 |
| imbalance_ratio_max_min | 7.553 |
| low_support_classes | backlight, glare, transition |

## Label Confidence

| Count | Min | P25 | Median | Mean | P75 | Max |
| --- | --- | --- | --- | --- | --- | --- |
| 11981 | 0.600 | 0.760 | 0.800 | 0.794 | 0.800 | 0.900 |
