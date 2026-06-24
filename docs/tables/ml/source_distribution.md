# Source Distribution

Status: source-level distribution evidence. Source overlap is not leakage by itself, but it must be reviewed with the leakage protocol.

Command: `/Users/phongpham/Downloads/smartBinocular/.venv/bin/python tools/analyze_dataset_distribution.py --train data/training/from_logs_train.jsonl --test data/training/from_logs_test.jsonl --reference data/training/merged_logs_ml.jsonl --out-dir docs/tables/ml --fig-dir docs/figures/ml`
Git commit: `15fbe64d5c70873cf200eac688f37d03e8f103da`

| Dataset | Source | Count | Percent |
| --- | --- | --- | --- |
| train | offline_backlight | 313 | 2.61 |
| train | offline_darkface | 2364 | 19.73 |
| train | offline_exdark_street | 1995 | 16.65 |
| train | offline_glare_street | 340 | 2.84 |
| train | offline_gray_nir | 2046 | 17.08 |
| train | offline_mwd | 946 | 7.90 |
| train | offline_weather11 | 1768 | 14.76 |
| train | offline_weather_time | 2209 | 18.44 |
| test | offline_backlight | 55 | 2.60 |
| test | offline_darkface | 417 | 19.73 |
| test | offline_exdark_street | 352 | 16.66 |
| test | offline_glare_street | 60 | 2.84 |
| test | offline_gray_nir | 361 | 17.08 |
| test | offline_mwd | 177 | 8.38 |
| test | offline_weather11 | 300 | 14.20 |
| test | offline_weather_time | 391 | 18.50 |
| reference | offline_backlight | 368 | 2.61 |
| reference | offline_darkface | 2781 | 19.73 |
| reference | offline_exdark_street | 2347 | 16.65 |
| reference | offline_glare_street | 400 | 2.84 |
| reference | offline_gray_nir | 2407 | 17.08 |
| reference | offline_mwd | 1123 | 7.97 |
| reference | offline_weather11 | 2068 | 14.67 |
| reference | offline_weather_time | 2600 | 18.45 |
