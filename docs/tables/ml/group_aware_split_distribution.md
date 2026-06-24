# Group-Aware Split Distribution

This is a group-aware split, not a source-held-out split unless source overlap is zero.

## Summary

| Metric | Value |
| --- | --- |
| train_rows | 11981 |
| test_rows | 2113 |
| group_field | split_group_id |
| group_count | 14027 |
| group_overlap_count | 0 |
| source_overlap_count | 8 |
| unresolved_train_rows | 0 |
| unresolved_test_rows | 0 |

## Per-Class Support

| Class | Train | Test |
| --- | --- | --- |
| backlight | 313 | 55 |
| fog | 1311 | 231 |
| glare | 340 | 60 |
| night_clear | 2364 | 417 |
| nir_night | 2046 | 361 |
| normal_day | 2077 | 366 |
| normal_night | 1995 | 352 |
| rain | 1097 | 194 |
| transition | 438 | 77 |

## Source Distribution

| Source | Train | Test |
| --- | --- | --- |
| offline_backlight | 313 | 55 |
| offline_darkface | 2364 | 417 |
| offline_exdark_street | 1995 | 352 |
| offline_glare_street | 340 | 60 |
| offline_gray_nir | 2046 | 361 |
| offline_mwd | 964 | 159 |
| offline_weather11 | 1770 | 298 |
| offline_weather_time | 2189 | 411 |

## Remaining Source Overlap

offline_backlight, offline_darkface, offline_exdark_street, offline_glare_street, offline_gray_nir, offline_mwd, offline_weather11, offline_weather_time
