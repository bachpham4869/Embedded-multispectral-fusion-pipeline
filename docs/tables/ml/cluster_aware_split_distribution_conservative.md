# Cluster-Aware Split Distribution

This is a duplicate-cluster-aware group split, not a source-held-out split unless source overlap is zero.

## Summary

| Metric | Value |
| --- | --- |
| train_rows | 11981 |
| test_rows | 2113 |
| test_ratio | 0.1499 |
| super_group_count | 13375 |
| duplicate_cluster_overlap_count | 0 |
| split_group_overlap_count | 0 |
| source_overlap_count | 8 |
| selected_for_benchmark_feasible | True |

## Per-Class Support

| Class | Train | Test |
| --- | --- | --- |
| backlight | 313 | 55 |
| fog | 1339 | 203 |
| glare | 340 | 60 |
| night_clear | 2364 | 417 |
| nir_night | 2046 | 361 |
| normal_day | 2077 | 366 |
| normal_night | 1995 | 352 |
| rain | 1104 | 187 |
| transition | 403 | 112 |

## Source Distribution

| Source | Train | Test |
| --- | --- | --- |
| offline_backlight | 313 | 55 |
| offline_darkface | 2364 | 417 |
| offline_exdark_street | 1995 | 352 |
| offline_glare_street | 340 | 60 |
| offline_gray_nir | 2046 | 361 |
| offline_mwd | 916 | 207 |
| offline_weather11 | 1797 | 271 |
| offline_weather_time | 2210 | 390 |

## Low-Support Warnings

No zero-support or test<30 class warning.

## Remaining Source Overlap

offline_backlight, offline_darkface, offline_exdark_street, offline_glare_street, offline_gray_nir, offline_mwd, offline_weather11, offline_weather_time
