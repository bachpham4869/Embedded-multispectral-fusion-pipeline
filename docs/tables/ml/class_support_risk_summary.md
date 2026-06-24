# Class Support Risk Summary

Status: Phase 1 risk table. It does not prove classifier quality.

Command: `/Users/phongpham/Downloads/smartBinocular/.venv/bin/python tools/analyze_dataset_distribution.py --train data/training/from_logs_train.jsonl --test data/training/from_logs_test.jsonl --reference data/training/merged_logs_ml.jsonl --out-dir docs/tables/ml --fig-dir docs/figures/ml`
Git commit: `15fbe64d5c70873cf200eac688f37d03e8f103da`

| Class | train Count | train Percent | train Mean Conf | test Count | test Percent | test Mean Conf | reference Count | reference Percent | reference Mean Conf | Phase 1 Risk | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| night_clear | 2364 | 19.73 | 0.800 | 417 | 19.73 | 0.800 | 2781 | 19.73 | 0.800 | standard | Usable for current audit subject to leakage/domain-shift checks. |
| normal_night | 1995 | 16.65 | 0.760 | 352 | 16.66 | 0.760 | 2347 | 16.65 | 0.760 | standard | Usable for current audit subject to leakage/domain-shift checks. |
| normal_day | 2077 | 17.34 | 0.796 | 366 | 17.32 | 0.793 | 2443 | 17.33 | 0.796 | standard | Usable for current audit subject to leakage/domain-shift checks. |
| fog | 1311 | 10.94 | 0.739 | 231 | 10.93 | 0.730 | 1542 | 10.94 | 0.738 | standard | Usable for current audit subject to leakage/domain-shift checks. |
| rain | 1097 | 9.16 | 0.900 | 194 | 9.18 | 0.900 | 1291 | 9.16 | 0.900 | standard | Usable for current audit subject to leakage/domain-shift checks. |
| glare | 340 | 2.84 | 0.780 | 60 | 2.84 | 0.780 | 400 | 2.84 | 0.780 | high support/source-diversity risk | Keep provisional; require CI, source diversity, and live validation. |
| backlight | 313 | 2.61 | 0.780 | 55 | 2.60 | 0.780 | 368 | 2.61 | 0.780 | high support/source-diversity risk | Keep provisional; require CI, source diversity, and live validation. |
| transition | 438 | 3.66 | 0.784 | 77 | 3.64 | 0.786 | 515 | 3.65 | 0.785 | high taxonomy risk | Do not claim strong ENV class until decision record evidence is complete. |
| nir_night | 2046 | 17.08 | 0.800 | 361 | 17.08 | 0.800 | 2407 | 17.08 | 0.800 | standard | Usable for current audit subject to leakage/domain-shift checks. |
