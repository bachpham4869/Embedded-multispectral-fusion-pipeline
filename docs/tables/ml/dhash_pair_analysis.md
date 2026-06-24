# dHash Pair Analysis

dHash pairs are screening evidence only. This table does not classify all dHash pairs as confirmed leakage.

## Summary

| Metric | Value |
| --- | --- |
| total_pairs | 1313 |
| low_support_label_pairs | 703 |
| exact_file_duplicate | 0 |
| likely_near_duplicate | 454 |
| dhash_false_positive_candidate | 859 |
| unresolved | 0 |

## Category Counts

| Category | Count |
| --- | --- |
| dhash_false_positive_candidate | 859 |
| likely_near_duplicate | 454 |

## Distance Counts

| Hamming distance | Count |
| --- | --- |
| 4 | 545 |
| 3 | 365 |
| 2 | 226 |
| 1 | 105 |
| 0 | 72 |

## Top Label Pairs

| Label pair | Count |
| --- | --- |
| transition -> fog | 297 |
| fog -> fog | 207 |
| transition -> transition | 113 |
| transition -> rain | 100 |
| transition -> normal_day | 95 |
| normal_day -> normal_day | 89 |
| rain -> fog | 66 |
| normal_day -> fog | 57 |
| fog -> normal_day | 56 |
| fog -> transition | 51 |
| fog -> rain | 51 |
| rain -> rain | 45 |
| rain -> transition | 28 |
| normal_day -> rain | 20 |
| rain -> normal_day | 19 |
| normal_day -> transition | 19 |

## Top Source Pairs

| Source pair | Count |
| --- | --- |
| offline_mwd -> offline_weather11 | 343 |
| offline_mwd -> offline_mwd | 266 |
| offline_weather11 -> offline_weather11 | 243 |
| offline_weather11 -> offline_mwd | 122 |
| offline_mwd -> offline_weather_time | 116 |
| offline_weather11 -> offline_weather_time | 73 |
| offline_weather_time -> offline_weather_time | 72 |
| offline_weather_time -> offline_weather11 | 43 |
| offline_weather_time -> offline_mwd | 35 |

## Metadata Status Pairs

| Metadata status pair | Count |
| --- | --- |
| verified -> verified | 1313 |
