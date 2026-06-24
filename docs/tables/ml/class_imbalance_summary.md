# Class Imbalance Summary

Command: `/Users/phongpham/Downloads/smartBinocular/.venv/bin/python tools/analyze_dataset_distribution.py --train data/training/from_logs_train.jsonl --test data/training/from_logs_test.jsonl --reference data/training/merged_logs_ml.jsonl --out-dir docs/tables/ml --fig-dir docs/figures/ml`
Git commit: `15fbe64d5c70873cf200eac688f37d03e8f103da`

| Dataset | Rows | Max/Min Class Ratio | Below Threshold |
| --- | --- | --- | --- |
| train | 11981 | 7.553 | backlight, glare, transition |
| test | 2113 | 7.582 | backlight, glare, transition |
| reference | 14094 | 7.557 | backlight, glare |
