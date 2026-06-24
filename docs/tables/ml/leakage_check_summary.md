# Leakage Check Summary

Status: metadata-level leakage audit. This does not prove absence of leakage.

Interpretation: no exact JSON-row overlap was found. Feature-vector overlaps are investigated separately. Image-level and near-duplicate leakage remain unverified unless original image path/hash metadata is present.

## Reproducibility

| Field | Value |
| --- | --- |
| command | /Users/phongpham/Downloads/smartBinocular/.venv/bin/python tools/check_dataset_leakage.py --train data/training/from_logs_train.jsonl --test data/training/from_logs_test.jsonl --out docs/tables/ml/leakage_check_summary.md --feature-overlap-details-md docs/tables/ml/feature_vector_overlap_details.md --feature-overlap-details-csv artifacts/ml/leakage/feature_vector_overlaps.csv |
| git_commit | 15fbe64d5c70873cf200eac688f37d03e8f103da |
| dataset_manifest_hash | 416c8ce44c6882625e1a8dd6c7bcd492f78a1177cb2baf0e01bf358ea91dc2ee |
| train_rows | 11981 |
| test_rows | 2113 |
| split_method | existing JSONL train/test files; no split performed by this script |
| random_seed | not used |
| python | 3.12.12 |
| sklearn | 1.8.0 |
| numpy | 1.26.4 |
| scipy | 1.17.1 |

## Dataset Files

| Split | Path | SHA256 | Bytes |
| --- | --- | --- | --- |
| train | data/training/from_logs_train.jsonl | 0a03e757a434956df1af4e5595d4bfc029242957b1cad15cb0053dfc0a744410 | 11501860 |
| test | data/training/from_logs_test.jsonl | 583ce85f80f1c065de2405c4bdccdab97098dcb2440dad4f4237bfbb2b662699 | 2027542 |

## Overlap Checks

| Check | Overlap Count | Examples |
| --- | --- | --- |
| source overlap | 8 | offline_backlight, offline_darkface, offline_exdark_street, offline_glare_street, offline_gray_nir, offline_mwd, offline_weather11, offline_weather_time |
| path overlap | 0 | none |
| filename overlap | 0 | none |
| session/frame overlap | 0 | none |
| exact row hash overlap | 0 | none |
| feature-vector hash overlap | 21 | 044c22eca660325c2cfe3039d10eceac9047ff58a85c8eacce92de28eaef1fc3, 11d237832a09a381b517ba237e4504e74d85154c1bfdf89906e655b642d3a2bb, 226aca795c0e288060f8a94a992f3bf8d63d0f03619c21dbd0651128099510b3, 2add1f249a6a4554c7bbfa9b412b06b7a9f14e2aaa9a8e9ef050809ed4e17cd7, 368efa6d96f9b76590f8a66a4d39c7ca24f0c39c42451ed6bdfee329eb419184, 47b71c62880ee9f7d42defc6a9eebf9a675dc8900bd5742cd6653d4d1bb38e3b, 517beeee70d306b6c5f0d4329473b5a7f3204dfa1c35660af09b78a41657d5a3, 6d93871ad233066490bfdbe7348fb69a25a614054ea61cd5a3e0cb999ecd73a4, 6dbf61bcc78168ba13c43991c9d0e1363236cb7388ab104c98062990a9faba0e, 711115b096f08fe1fc57b0984b7f6eadbb2f09ead5d19386eb0a5b53a5db0f83, 7c827cfde2feef7059e497074d6ed8aa3b371e8cf916dd74f68d3da9c5130cbe, 8efbf9198ef6e12517bd8c952792704a200d921fefa4ad4709718d74bb40bb5d, a9985123de32fabf48ed93fa7d39470fbd51c02bc0a18a6fdf0704daf5ade18d, cbd54c881287f9faf8e8dc2bcb6644b94e1a8be38196cec4c369f6260fe724be, d725c81d1e5ca62ae70696843cd3f5e0f5ff3eebf38e9d41944e45f2cba07a44, d82338c8987671c8c10ef5c4ac9b306c859cc5a5e7453f31a7c3a62cca7879f7, e7a284103ed8bb82caaaf7a2a0ae6766e2c9419941b79030be0c8d9d12f900e2, eab19213143c19bcd0df8fea7e28f105ebf0a469465bf3fa710efec6cd1e85a7, ef82b80fba91fb2bb15b22f2887a099bf2224317861773fb1fafcfd7445e099c, f0901c26fad1f5d16a374015a67d94be448eb5fe72255a3c6c8b3ae5f518172f |
| perceptual hash near-duplicate | not run | requires resolvable image paths and --phash |

## Feature-Vector Overlap Details

Detailed overlap rows are written to `docs/tables/ml/feature_vector_overlap_details.md`.
Detailed train/test overlap pairs: 22.

## Source Overlap By Class

| Split | Class | Rows From Overlapped Sources |
| --- | --- | --- |
| train | backlight | 313 |
| train | fog | 1311 |
| train | glare | 340 |
| train | night_clear | 2364 |
| train | nir_night | 2046 |
| train | normal_day | 2077 |
| train | normal_night | 1995 |
| train | rain | 1097 |
| train | transition | 438 |
| test | backlight | 55 |
| test | fog | 231 |
| test | glare | 60 |
| test | night_clear | 417 |
| test | nir_night | 361 |
| test | normal_day | 366 |
| test | normal_night | 352 |
| test | rain | 194 |
| test | transition | 77 |

## Limitations

- No path-like metadata fields found; image/path overlap cannot be checked.
- No session metadata fields found; session/frame overlap cannot be checked.
- Image-level/near-duplicate leakage remains unverified because JSONL lacks original image path/hash metadata or --phash was not run with resolvable image paths.