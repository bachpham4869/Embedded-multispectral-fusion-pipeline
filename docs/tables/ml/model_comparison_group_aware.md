# Model Comparison: optical_12_baseline

Scope: same-feature classical ML baselines only. These rows are directly comparable because they use the same train/test JSONL files and the same 12 handcrafted optical features.

Latency status: proxy benchmark unless hardware is Raspberry Pi 4 CPU. Do not claim final deployment latency from non-RPi runs.

## Reproducibility

| Field | Value |
| --- | --- |
| command | /Users/phongpham/Downloads/smartBinocular/.venv/bin/python tools/compare_classifiers.py --train data/training_v2/group_aware_train.jsonl --test data/training_v2/group_aware_test.jsonl --models random_forest_100 random_forest_200_current_config extra_trees logistic_regression linear_svm hist_gradient_boosting mlp_small_32 --bootstrap 500 --latency-repeats 300 --hardware-label macOS proxy benchmark (not Raspberry Pi 4 CPU) --out-csv docs/tables/ml/model_comparison_group_aware.csv --out-md docs/tables/ml/model_comparison_group_aware.md --out-ci-md docs/tables/ml/model_comparison_group_aware_ci.md --per-class-csv docs/tables/ml/model_comparison_group_aware_per_class.csv --artifact-dir artifacts/ml/model_comparison_group_aware --fig-dir docs/figures/ml/group_aware --split-method-label Phase 3 group-aware split by split_group_id from metadata-enriched JSONL; source overlap reported separately --metric-status preliminary / not thesis-ready: dHash near-duplicate screening still has cross-split pairs; model artifacts not persisted due disk guardrail |
| git_commit | 15fbe64d5c70873cf200eac688f37d03e8f103da |
| branch | refactor/ml-taxonomy-eval-plan |
| feature_set_version | optical_12_baseline (12 features) |
| split_method | Phase 3 group-aware split by split_group_id from metadata-enriched JSONL; source overlap reported separately |
| random_seed | 42 |
| train_rows | 11981 |
| test_rows | 2113 |
| dataset_manifest_hash | 802955cce0fb07264d61b13e04cfc03e57ccbc41a507bcee812c349c0ee9cfe7 |
| python | 3.12.12 |
| sklearn | 1.8.0 |
| numpy | 1.26.4 |
| scipy | 1.17.1 |
| hardware | macOS proxy benchmark (not Raspberry Pi 4 CPU) |
| latency_repeats | 300 |
| run_status | preliminary / not thesis-ready: dHash near-duplicate screening still has cross-split pairs; model artifacts not persisted due disk guardrail |

## Dataset Files

| Split | Path | SHA256 | Bytes |
| --- | --- | --- | --- |
| train | data/training_v2/group_aware_train.jsonl | f4ed7eccc89e87987f1ea5d03827000bbb7afef6c4cdee2529cbe34852fc5890 | 17944860 |
| test | data/training_v2/group_aware_test.jsonl | 913afd9b439596a3469b49889505dd04293cccec0e679c91f2e2dd91ae8391a6 | 3166352 |

## Aggregate Metrics

| Model | Accuracy | Acc 95% CI | Balanced Acc | Bal Acc 95% CI | Macro-F1 | Weighted-F1 | ECE | Brier | Model Bytes | Train s | Load s | Mean ms | Median ms | P95 ms | Warnings | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_forest_100 | 0.8363 | 0.8201-0.8523 | 0.7618 | 0.7386-0.7846 | 0.7533 | 0.8360 | 0.0969 | 0.2571 | 22907198 | 0.339 | 0.0146 | 14.966 | 15.940 | 16.282 | 0 | preliminary / not thesis-ready: dHash near-duplicate screening still has cross-split pairs; model artifacts not persisted due disk guardrail |
| random_forest_200_current_config | 0.8358 | 0.8206-0.8519 | 0.7564 | 0.7311-0.7806 | 0.7502 | 0.8353 | 0.0959 | 0.2549 | 45796662 | 0.718 | 0.0244 | 14.913 | 15.674 | 16.267 | 0 | preliminary / not thesis-ready: dHash near-duplicate screening still has cross-split pairs; model artifacts not persisted due disk guardrail |
| extra_trees | 0.8017 | 0.7847-0.8183 | 0.7285 | 0.7058-0.7534 | 0.7039 | 0.8028 | 0.2552 | 0.3704 | 38683662 | 0.273 | 0.0197 | 15.131 | 15.652 | 16.291 | 0 | preliminary / not thesis-ready: dHash near-duplicate screening still has cross-split pairs; model artifacts not persisted due disk guardrail |
| logistic_regression | 0.6763 | 0.6559-0.6952 | 0.6419 | 0.6172-0.6649 | 0.6010 | 0.6786 | 0.0439 | 0.4223 | 1632 | 0.717 | 0.0000 | 0.028 | 0.027 | 0.035 | 0 | preliminary / not thesis-ready: dHash near-duplicate screening still has cross-split pairs; model artifacts not persisted due disk guardrail |
| linear_svm | 0.7194 | 0.6997-0.7390 | 0.6043 | 0.5828-0.6260 | 0.5786 | 0.6953 | n/a | n/a | 1561 | 0.087 | 0.0000 | 0.030 | 0.030 | 0.037 | 0 | preliminary / not thesis-ready: dHash near-duplicate screening still has cross-split pairs; model artifacts not persisted due disk guardrail |
| hist_gradient_boosting | 0.8490 | 0.8341-0.8632 | 0.7404 | 0.7160-0.7651 | 0.7573 | 0.8442 | 0.0190 | 0.2286 | 1548828 | 4.658 | 0.0020 | 25.379 | 25.375 | 38.710 | 0 | preliminary / not thesis-ready: dHash near-duplicate screening still has cross-split pairs; model artifacts not persisted due disk guardrail |
| mlp_small_32 | 0.7903 | 0.7728-0.8069 | 0.6550 | 0.6308-0.6819 | 0.6744 | 0.7774 | 0.0349 | 0.3150 | 15104 | 1.062 | 0.0002 | 0.060 | 0.058 | 0.062 | 0 | preliminary / not thesis-ready: dHash near-duplicate screening still has cross-split pairs; model artifacts not persisted due disk guardrail |

## Model Provenance

| Model | Model Path | Model SHA256 | Serialized Bytes |
| --- | --- | --- | --- |
| random_forest_100 | in-memory benchmark model; not persisted | 751d5818c62b3dbaefd21192ed910805d6bd2148507c90a643c8e1b8c1664d54 | 22907198 |
| random_forest_200_current_config | in-memory benchmark model; not persisted | 787c3608c374ef534ba08d89df872e43806dc1c02f277e5dd1823c38c46c9cff | 45796662 |
| extra_trees | in-memory benchmark model; not persisted | 4ceef7518dd18490f02787a18a53db2f45838f4e8f10fc0a47e0cd181061cb6c | 38683662 |
| logistic_regression | in-memory benchmark model; not persisted | fac5518447bf42a029d7d95dee223af2c7375e27013ca883cc779e16e11abd8f | 1632 |
| linear_svm | in-memory benchmark model; not persisted | 914f17094fe581a22a44dbbd97c152fa412e1ad815ba8b35822fab8b177120c0 | 1561 |
| hist_gradient_boosting | in-memory benchmark model; not persisted | 631fe8349c4eb562a103b3fba1e9f358c3161ff8e3fe269a264fa61b7d1bf4ac | 1548828 |
| mlp_small_32 | in-memory benchmark model; not persisted | 84d619457a15cb59f10d2b9b4d824f3421d78401a1fd78398c180c9d5c87b86e | 15104 |

## Per-Class Metrics

| Model | Class | Support | Precision | Recall | F1 | F1 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| random_forest_100 | night_clear | 417 | 0.9780 | 0.9592 | 0.9685 | 0.9556-0.9802 |
| random_forest_100 | normal_night | 352 | 0.8433 | 0.8409 | 0.8421 | 0.8129-0.8668 |
| random_forest_100 | normal_day | 366 | 0.7889 | 0.8169 | 0.8027 | 0.7696-0.8343 |
| random_forest_100 | fog | 231 | 0.8133 | 0.7922 | 0.8026 | 0.7655-0.8422 |
| random_forest_100 | rain | 194 | 0.6901 | 0.6082 | 0.6466 | 0.5895-0.7035 |
| random_forest_100 | glare | 60 | 0.5455 | 0.5000 | 0.5217 | 0.4178-0.6222 |
| random_forest_100 | backlight | 55 | 0.5946 | 0.8000 | 0.6822 | 0.5763-0.7641 |
| random_forest_100 | transition | 77 | 0.5244 | 0.5584 | 0.5409 | 0.4415-0.6211 |
| random_forest_100 | nir_night | 361 | 0.9646 | 0.9806 | 0.9725 | 0.9613-0.9848 |
| random_forest_200_current_config | night_clear | 417 | 0.9804 | 0.9592 | 0.9697 | 0.9576-0.9812 |
| random_forest_200_current_config | normal_night | 352 | 0.8446 | 0.8494 | 0.8470 | 0.8174-0.8736 |
| random_forest_200_current_config | normal_day | 366 | 0.7804 | 0.8060 | 0.7930 | 0.7595-0.8232 |
| random_forest_200_current_config | fog | 231 | 0.8087 | 0.8052 | 0.8069 | 0.7685-0.8432 |
| random_forest_200_current_config | rain | 194 | 0.6977 | 0.6186 | 0.6557 | 0.5960-0.7103 |
| random_forest_200_current_config | glare | 60 | 0.5556 | 0.5000 | 0.5263 | 0.4219-0.6270 |
| random_forest_200_current_config | backlight | 55 | 0.5972 | 0.7818 | 0.6772 | 0.5687-0.7674 |
| random_forest_200_current_config | transition | 77 | 0.5000 | 0.5065 | 0.5032 | 0.4044-0.5876 |
| random_forest_200_current_config | nir_night | 361 | 0.9646 | 0.9806 | 0.9725 | 0.9609-0.9841 |
| extra_trees | night_clear | 417 | 0.9470 | 0.9424 | 0.9447 | 0.9296-0.9618 |
| extra_trees | normal_night | 352 | 0.8426 | 0.7756 | 0.8077 | 0.7748-0.8363 |
| extra_trees | normal_day | 366 | 0.7724 | 0.7787 | 0.7755 | 0.7412-0.8077 |
| extra_trees | fog | 231 | 0.7520 | 0.8009 | 0.7757 | 0.7340-0.8147 |
| extra_trees | rain | 194 | 0.7029 | 0.5000 | 0.5843 | 0.5184-0.6440 |
| extra_trees | glare | 60 | 0.4429 | 0.5167 | 0.4769 | 0.3589-0.5768 |
| extra_trees | backlight | 55 | 0.4327 | 0.8182 | 0.5660 | 0.4660-0.6565 |
| extra_trees | transition | 77 | 0.4268 | 0.4545 | 0.4403 | 0.3390-0.5337 |
| extra_trees | nir_night | 361 | 0.9589 | 0.9695 | 0.9642 | 0.9526-0.9771 |
| logistic_regression | night_clear | 417 | 0.8374 | 0.8273 | 0.8323 | 0.8060-0.8570 |
| logistic_regression | normal_night | 352 | 0.6949 | 0.5369 | 0.6058 | 0.5580-0.6498 |
| logistic_regression | normal_day | 366 | 0.6298 | 0.4973 | 0.5557 | 0.5095-0.5990 |
| logistic_regression | fog | 231 | 0.6875 | 0.7619 | 0.7228 | 0.6745-0.7608 |
| logistic_regression | rain | 194 | 0.3436 | 0.3454 | 0.3445 | 0.2884-0.4030 |
| logistic_regression | glare | 60 | 0.2759 | 0.5333 | 0.3636 | 0.2696-0.4444 |
| logistic_regression | backlight | 55 | 0.5114 | 0.8182 | 0.6294 | 0.5343-0.7224 |
| logistic_regression | transition | 77 | 0.3273 | 0.4675 | 0.3850 | 0.2942-0.4697 |
| logistic_regression | nir_night | 361 | 0.9520 | 0.9889 | 0.9701 | 0.9584-0.9823 |
| linear_svm | night_clear | 417 | 0.8035 | 0.8729 | 0.8368 | 0.8108-0.8607 |
| linear_svm | normal_night | 352 | 0.6966 | 0.6392 | 0.6667 | 0.6254-0.7037 |
| linear_svm | normal_day | 366 | 0.6418 | 0.7978 | 0.7113 | 0.6747-0.7420 |
| linear_svm | fog | 231 | 0.6493 | 0.8095 | 0.7206 | 0.6708-0.7631 |
| linear_svm | rain | 194 | 0.6154 | 0.1237 | 0.2060 | 0.1360-0.2655 |
| linear_svm | glare | 60 | 0.2692 | 0.1167 | 0.1628 | 0.0690-0.2674 |
| linear_svm | backlight | 55 | 0.4828 | 0.7636 | 0.5915 | 0.4926-0.6764 |
| linear_svm | transition | 77 | 0.3467 | 0.3377 | 0.3421 | 0.2402-0.4410 |
| linear_svm | nir_night | 361 | 0.9619 | 0.9778 | 0.9698 | 0.9570-0.9818 |
| hist_gradient_boosting | night_clear | 417 | 0.9664 | 0.9664 | 0.9664 | 0.9543-0.9782 |
| hist_gradient_boosting | normal_night | 352 | 0.8221 | 0.8665 | 0.8437 | 0.8142-0.8723 |
| hist_gradient_boosting | normal_day | 366 | 0.7694 | 0.8934 | 0.8268 | 0.7984-0.8545 |
| hist_gradient_boosting | fog | 231 | 0.8400 | 0.8182 | 0.8289 | 0.7915-0.8669 |
| hist_gradient_boosting | rain | 194 | 0.7301 | 0.6134 | 0.6667 | 0.6061-0.7200 |
| hist_gradient_boosting | glare | 60 | 0.6667 | 0.4000 | 0.5000 | 0.3735-0.6105 |
| hist_gradient_boosting | backlight | 55 | 0.6792 | 0.6545 | 0.6667 | 0.5609-0.7542 |
| hist_gradient_boosting | transition | 77 | 0.6429 | 0.4675 | 0.5414 | 0.4417-0.6450 |
| hist_gradient_boosting | nir_night | 361 | 0.9673 | 0.9834 | 0.9753 | 0.9634-0.9867 |
| mlp_small_32 | night_clear | 417 | 0.8929 | 0.9400 | 0.9159 | 0.8969-0.9352 |
| mlp_small_32 | normal_night | 352 | 0.7971 | 0.7926 | 0.7949 | 0.7613-0.8252 |
| mlp_small_32 | normal_day | 366 | 0.6776 | 0.8497 | 0.7539 | 0.7199-0.7834 |
| mlp_small_32 | fog | 231 | 0.7143 | 0.8225 | 0.7646 | 0.7257-0.8062 |
| mlp_small_32 | rain | 194 | 0.6636 | 0.3763 | 0.4803 | 0.4081-0.5505 |
| mlp_small_32 | glare | 60 | 0.5000 | 0.2500 | 0.3333 | 0.2171-0.4690 |
| mlp_small_32 | backlight | 55 | 0.7692 | 0.5455 | 0.6383 | 0.5270-0.7339 |
| mlp_small_32 | transition | 77 | 0.5778 | 0.3377 | 0.4262 | 0.3043-0.5329 |
| mlp_small_32 | nir_night | 361 | 0.9440 | 0.9806 | 0.9620 | 0.9479-0.9757 |

## Confusion Matrices

Raw and normalized confusion matrices are written as CSV/PNG artifacts per model when this script runs.