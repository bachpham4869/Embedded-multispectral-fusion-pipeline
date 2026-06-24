# Cluster-Aware Split Distribution

This is a duplicate-cluster-aware group split, not a source-held-out split unless source overlap is zero.

## Summary

| Metric | Value |
| --- | --- |
| train_rows | 12028 |
| test_rows | 2066 |
| test_ratio | 0.1466 |
| super_group_count | 13673 |
| duplicate_cluster_overlap_count | 0 |
| split_group_overlap_count | 0 |
| source_overlap_count | 8 |
| selected_for_benchmark_feasible | True |

## Per-Class Support

| Class | Train | Test |
| --- | --- | --- |
| backlight | 313 | 55 |
| fog | 1315 | 227 |
| glare | 340 | 60 |
| night_clear | 2364 | 417 |
| nir_night | 2046 | 361 |
| normal_day | 2079 | 364 |
| normal_night | 1995 | 352 |
| rain | 1101 | 190 |
| transition | 475 | 40 |

## Source Distribution

| Source | Train | Test |
| --- | --- | --- |
| offline_backlight | 313 | 55 |
| offline_darkface | 2364 | 417 |
| offline_exdark_street | 1995 | 352 |
| offline_glare_street | 340 | 60 |
| offline_gray_nir | 2046 | 361 |
| offline_mwd | 987 | 136 |
| offline_weather11 | 1764 | 304 |
| offline_weather_time | 2219 | 381 |

## Low-Support Warnings

No zero-support or test<30 class warning.

## Remaining Source Overlap

offline_backlight, offline_darkface, offline_exdark_street, offline_glare_street, offline_gray_nir, offline_mwd, offline_weather11, offline_weather_time
