# Model Comparison: optical_12_baseline

Scope: same-feature classical ML baselines only. These rows are directly comparable because they use the same train/test JSONL files and the same 12 handcrafted optical features.

Latency status: proxy benchmark unless hardware is Raspberry Pi 4 CPU. Do not claim final deployment latency from non-RPi runs.

## Reproducibility

| Field | Value |
| --- | --- |
| command | /Users/phongpham/Downloads/smartBinocular/.venv/bin/python tools/compare_classifiers.py --train data/training_v3/cluster_aware_conservative_train.jsonl --test data/training_v3/cluster_aware_conservative_test.jsonl --models random_forest_100 random_forest_200_current_config extra_trees logistic_regression linear_svm hist_gradient_boosting mlp_small_32 --bootstrap 500 --latency-repeats 300 --hardware-label macOS proxy benchmark (not Raspberry Pi 4 CPU) --out-csv docs/tables/ml/model_comparison_cluster_aware.csv --out-md docs/tables/ml/model_comparison_cluster_aware.md --out-ci-md docs/tables/ml/model_comparison_cluster_aware_ci.md --per-class-csv docs/tables/ml/model_comparison_cluster_aware_per_class.csv --artifact-dir artifacts/ml/model_comparison_cluster_aware --fig-dir docs/figures/ml/cluster_aware --split-method-label Phase 4 conservative duplicate-cluster-aware group split (dHash<=4 + split_group_id); source overlap reported separately --metric-status offline optical RGB-proxy baseline; not live NIR/LWIR validated; macOS latency proxy |
| git_commit | 15fbe64d5c70873cf200eac688f37d03e8f103da |
| branch | refactor/ml-taxonomy-eval-plan |
| feature_set_version | optical_12_baseline (12 features) |
| split_method | Phase 4 conservative duplicate-cluster-aware group split (dHash<=4 + split_group_id); source overlap reported separately |
| random_seed | 42 |
| train_rows | 11981 |
| test_rows | 2113 |
| dataset_manifest_hash | 5dfff8530202dc8192371231ad77019c7d09b9157e00a7fd16bf00dc00abb3c9 |
| python | 3.12.12 |
| sklearn | 1.8.0 |
| numpy | 1.26.4 |
| scipy | 1.17.1 |
| hardware | macOS proxy benchmark (not Raspberry Pi 4 CPU) |
| latency_repeats | 300 |
| run_status | offline optical RGB-proxy baseline; not live NIR/LWIR validated; macOS latency proxy |

## Dataset Files

| Split | Path | SHA256 | Bytes |
| --- | --- | --- | --- |
| train | data/training_v3/cluster_aware_conservative_train.jsonl | 53ce44bd58f62a62b49e04eb3f0ab616b7b5c562e080e3127ba0c86b7aa2f78f | 20594250 |
| test | data/training_v3/cluster_aware_conservative_test.jsonl | bb7aea99accd63082c9d100c87dc243aeac29ab058044e8e6db6c2e70f463642 | 3631736 |

## Aggregate Metrics

| Model | Accuracy | Acc 95% CI | Balanced Acc | Bal Acc 95% CI | Macro-F1 | Weighted-F1 | ECE | Brier | Model Bytes | Train s | Load s | Mean ms | Median ms | P95 ms | Warnings | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_forest_100 | 0.8230 | 0.8062-0.8408 | 0.7415 | 0.7152-0.7633 | 0.7325 | 0.8201 | 0.0852 | 0.2601 | 22994782 | 0.401 | 0.0155 | 14.824 | 15.875 | 16.522 | 0 | offline optical RGB-proxy baseline; not live NIR/LWIR validated; macOS latency proxy |
| random_forest_200_current_config | 0.8263 | 0.8090-0.8429 | 0.7463 | 0.7202-0.7690 | 0.7362 | 0.8234 | 0.0885 | 0.2592 | 45873094 | 0.777 | 0.0178 | 15.816 | 15.886 | 17.017 | 0 | offline optical RGB-proxy baseline; not live NIR/LWIR validated; macOS latency proxy |
| extra_trees | 0.8008 | 0.7834-0.8202 | 0.7386 | 0.7133-0.7611 | 0.7149 | 0.8002 | 0.2582 | 0.3754 | 39202330 | 0.328 | 0.0171 | 15.211 | 15.817 | 16.285 | 0 | offline optical RGB-proxy baseline; not live NIR/LWIR validated; macOS latency proxy |
| logistic_regression | 0.6673 | 0.6486-0.6872 | 0.6335 | 0.6086-0.6557 | 0.5927 | 0.6688 | 0.0425 | 0.4295 | 1632 | 0.703 | 0.0001 | 0.027 | 0.024 | 0.033 | 0 | offline optical RGB-proxy baseline; not live NIR/LWIR validated; macOS latency proxy |
| linear_svm | 0.7170 | 0.6969-0.7366 | 0.6108 | 0.5892-0.6313 | 0.5845 | 0.6912 | n/a | n/a | 1561 | 0.079 | 0.0000 | 0.026 | 0.024 | 0.029 | 0 | offline optical RGB-proxy baseline; not live NIR/LWIR validated; macOS latency proxy |
| hist_gradient_boosting | 0.8282 | 0.8114-0.8445 | 0.7096 | 0.6861-0.7334 | 0.7238 | 0.8191 | 0.0401 | 0.2434 | 1898821 | 5.433 | 0.0024 | 25.496 | 27.438 | 31.402 | 0 | offline optical RGB-proxy baseline; not live NIR/LWIR validated; macOS latency proxy |
| mlp_small_32 | 0.7941 | 0.7780-0.8121 | 0.6614 | 0.6370-0.6845 | 0.6758 | 0.7797 | 0.0236 | 0.2994 | 16309 | 1.925 | 0.0003 | 0.085 | 0.075 | 0.112 | 0 | offline optical RGB-proxy baseline; not live NIR/LWIR validated; macOS latency proxy |

## Model Provenance

| Model | Model Path | Model SHA256 | Serialized Bytes |
| --- | --- | --- | --- |
| random_forest_100 | in-memory benchmark model; not persisted | 219fae47bb67a8dc58ccb5d4e7dcf549cb83f4c4b01913436e7d72fc64e0e87a | 22994782 |
| random_forest_200_current_config | in-memory benchmark model; not persisted | 4653b87b88777cf047689d6a0dd0311356001d13c214099784f7003919ca31da | 45873094 |
| extra_trees | in-memory benchmark model; not persisted | 2c8a454ea8c001822085db278d4ceecab167520dd44f1531d7af637cd0f690dd | 39202330 |
| logistic_regression | in-memory benchmark model; not persisted | b6315b13fd3c559c9d625006e23666f41dd0014c758f9f8c97cb3815f507d5d9 | 1632 |
| linear_svm | in-memory benchmark model; not persisted | a730d29d195eba2d345c8f2d1988d3097dc148c553cf429bb46b5c386e4e844f | 1561 |
| hist_gradient_boosting | in-memory benchmark model; not persisted | 0ce801702c2ea41493be11ae255d1ebf0ef87b049b7513edc830142b2b2c38d0 | 1898821 |
| mlp_small_32 | in-memory benchmark model; not persisted | fbda13d857a8973a18b6ab9c701bc2a3c4792bde4e431be23644f7f4a725e9ca | 16309 |

## Per-Class Metrics

| Model | Class | Support | Precision | Recall | F1 | F1 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| random_forest_100 | night_clear | 417 | 0.9828 | 0.9616 | 0.9721 | 0.9594-0.9813 |
| random_forest_100 | normal_night | 352 | 0.8209 | 0.8466 | 0.8336 | 0.8014-0.8625 |
| random_forest_100 | normal_day | 366 | 0.7708 | 0.8087 | 0.7893 | 0.7569-0.8201 |
| random_forest_100 | fog | 203 | 0.7897 | 0.8325 | 0.8106 | 0.7681-0.8518 |
| random_forest_100 | rain | 187 | 0.6386 | 0.5668 | 0.6006 | 0.5321-0.6615 |
| random_forest_100 | glare | 60 | 0.5254 | 0.5167 | 0.5210 | 0.4162-0.6285 |
| random_forest_100 | backlight | 55 | 0.5309 | 0.7818 | 0.6324 | 0.5280-0.7212 |
| random_forest_100 | transition | 112 | 0.5811 | 0.3839 | 0.4624 | 0.3730-0.5475 |
| random_forest_100 | nir_night | 361 | 0.9670 | 0.9751 | 0.9710 | 0.9578-0.9837 |
| random_forest_200_current_config | night_clear | 417 | 0.9829 | 0.9664 | 0.9746 | 0.9629-0.9839 |
| random_forest_200_current_config | normal_night | 352 | 0.8301 | 0.8466 | 0.8383 | 0.8041-0.8682 |
| random_forest_200_current_config | normal_day | 366 | 0.7738 | 0.8224 | 0.7974 | 0.7637-0.8285 |
| random_forest_200_current_config | fog | 203 | 0.7925 | 0.8276 | 0.8096 | 0.7606-0.8468 |
| random_forest_200_current_config | rain | 187 | 0.6442 | 0.5615 | 0.6000 | 0.5290-0.6594 |
| random_forest_200_current_config | glare | 60 | 0.5424 | 0.5333 | 0.5378 | 0.4324-0.6444 |
| random_forest_200_current_config | backlight | 55 | 0.5238 | 0.8000 | 0.6331 | 0.5381-0.7211 |
| random_forest_200_current_config | transition | 112 | 0.5811 | 0.3839 | 0.4624 | 0.3766-0.5456 |
| random_forest_200_current_config | nir_night | 361 | 0.9697 | 0.9751 | 0.9724 | 0.9596-0.9842 |
| extra_trees | night_clear | 417 | 0.9495 | 0.9472 | 0.9484 | 0.9330-0.9626 |
| extra_trees | normal_night | 352 | 0.8262 | 0.7699 | 0.7971 | 0.7640-0.8321 |
| extra_trees | normal_day | 366 | 0.7772 | 0.7814 | 0.7793 | 0.7455-0.8115 |
| extra_trees | fog | 203 | 0.7489 | 0.8227 | 0.7840 | 0.7396-0.8237 |
| extra_trees | rain | 187 | 0.6225 | 0.5027 | 0.5562 | 0.4831-0.6170 |
| extra_trees | glare | 60 | 0.4648 | 0.5500 | 0.5038 | 0.4036-0.5994 |
| extra_trees | backlight | 55 | 0.4700 | 0.8545 | 0.6065 | 0.5089-0.6857 |
| extra_trees | transition | 112 | 0.5484 | 0.4554 | 0.4976 | 0.4057-0.5800 |
| extra_trees | nir_night | 361 | 0.9587 | 0.9640 | 0.9613 | 0.9463-0.9757 |
| logistic_regression | night_clear | 417 | 0.8313 | 0.8153 | 0.8232 | 0.7954-0.8502 |
| logistic_regression | normal_night | 352 | 0.6606 | 0.5142 | 0.5783 | 0.5329-0.6257 |
| logistic_regression | normal_day | 366 | 0.6318 | 0.5109 | 0.5650 | 0.5142-0.6108 |
| logistic_regression | fog | 203 | 0.6736 | 0.7931 | 0.7285 | 0.6800-0.7701 |
| logistic_regression | rain | 187 | 0.3297 | 0.3262 | 0.3280 | 0.2670-0.3871 |
| logistic_regression | glare | 60 | 0.2500 | 0.5000 | 0.3333 | 0.2480-0.4047 |
| logistic_regression | backlight | 55 | 0.4500 | 0.8182 | 0.5806 | 0.4798-0.6625 |
| logistic_regression | transition | 112 | 0.4153 | 0.4375 | 0.4261 | 0.3417-0.5022 |
| logistic_regression | nir_night | 361 | 0.9570 | 0.9861 | 0.9714 | 0.9593-0.9824 |
| linear_svm | night_clear | 417 | 0.8004 | 0.8657 | 0.8318 | 0.8061-0.8584 |
| linear_svm | normal_night | 352 | 0.6800 | 0.6278 | 0.6529 | 0.6117-0.6952 |
| linear_svm | normal_day | 366 | 0.6492 | 0.8142 | 0.7224 | 0.6866-0.7568 |
| linear_svm | fog | 203 | 0.6320 | 0.8374 | 0.7203 | 0.6644-0.7621 |
| linear_svm | rain | 187 | 0.5385 | 0.1123 | 0.1858 | 0.1179-0.2571 |
| linear_svm | glare | 60 | 0.3667 | 0.1833 | 0.2444 | 0.1324-0.3628 |
| linear_svm | backlight | 55 | 0.4468 | 0.7636 | 0.5638 | 0.4595-0.6483 |
| linear_svm | transition | 112 | 0.4533 | 0.3036 | 0.3636 | 0.2690-0.4463 |
| linear_svm | nir_night | 361 | 0.9623 | 0.9889 | 0.9754 | 0.9636-0.9856 |
| hist_gradient_boosting | night_clear | 417 | 0.9759 | 0.9712 | 0.9736 | 0.9625-0.9836 |
| hist_gradient_boosting | normal_night | 352 | 0.7912 | 0.8722 | 0.8297 | 0.8002-0.8595 |
| hist_gradient_boosting | normal_day | 366 | 0.7506 | 0.8634 | 0.8030 | 0.7701-0.8317 |
| hist_gradient_boosting | fog | 203 | 0.7746 | 0.8128 | 0.7933 | 0.7490-0.8333 |
| hist_gradient_boosting | rain | 187 | 0.6457 | 0.6043 | 0.6243 | 0.5603-0.6811 |
| hist_gradient_boosting | glare | 60 | 0.6154 | 0.4000 | 0.4848 | 0.3728-0.5907 |
| hist_gradient_boosting | backlight | 55 | 0.6735 | 0.6000 | 0.6346 | 0.5189-0.7291 |
| hist_gradient_boosting | transition | 112 | 0.6739 | 0.2768 | 0.3924 | 0.2946-0.4909 |
| hist_gradient_boosting | nir_night | 361 | 0.9700 | 0.9861 | 0.9780 | 0.9658-0.9874 |
| mlp_small_32 | night_clear | 417 | 0.9013 | 0.9640 | 0.9316 | 0.9130-0.9494 |
| mlp_small_32 | normal_night | 352 | 0.8118 | 0.7841 | 0.7977 | 0.7660-0.8325 |
| mlp_small_32 | normal_day | 366 | 0.6821 | 0.8852 | 0.7705 | 0.7374-0.8007 |
| mlp_small_32 | fog | 203 | 0.7210 | 0.8276 | 0.7706 | 0.7239-0.8115 |
| mlp_small_32 | rain | 187 | 0.6218 | 0.3957 | 0.4837 | 0.4103-0.5530 |
| mlp_small_32 | glare | 60 | 0.6154 | 0.2667 | 0.3721 | 0.2377-0.4860 |
| mlp_small_32 | backlight | 55 | 0.6250 | 0.5455 | 0.5825 | 0.4689-0.6818 |
| mlp_small_32 | transition | 112 | 0.6071 | 0.3036 | 0.4048 | 0.3121-0.4917 |
| mlp_small_32 | nir_night | 361 | 0.9568 | 0.9806 | 0.9685 | 0.9557-0.9807 |

## Confusion Matrices

Raw and normalized confusion matrices are written as CSV/PNG artifacts per model when this script runs.