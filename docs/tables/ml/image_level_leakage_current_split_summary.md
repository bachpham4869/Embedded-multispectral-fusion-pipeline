# Image-Level Leakage Summary

Status: preliminary until reviewed with raw-image coverage and split policy.

No absolute visual-duplicate claim is made. If dHash near-duplicate pairs are zero, that means only that no dHash pairs were found within records with dHash coverage at the configured threshold.

Feature-vector/hash matching is a weak hint only and is not used as original-image identity evidence.

## Coverage and Pair Counts

| Metric | Value |
| --- | --- |
| train_rows | 11981 |
| test_rows | 2113 |
| train_file_sha256_rows | 4923 |
| test_file_sha256_rows | 868 |
| train_dhash_rows | 4923 |
| test_dhash_rows | 868 |
| dhash_threshold | 4 |
| source_overlap_count | 8 |
| exact_json_row_pairs | 0 |
| file_sha256_overlap_pairs | 22 |
| split_group_id_overlap_pairs | 22 |
| session_overlap_pairs | 0 |
| feature_vector_weak_hint_pairs | 22 |
| dhash_near_duplicate_pairs | 1419 |

## Source Overlap

offline_backlight, offline_darkface, offline_exdark_street, offline_glare_street, offline_gray_nir, offline_mwd, offline_weather11, offline_weather_time
