# Model Comparison: optical_12_baseline

Scope: same-feature classical ML baselines only. These rows are directly comparable because they use the same train/test JSONL files and the same 12 handcrafted optical features.

Latency status: proxy benchmark unless hardware is Raspberry Pi 4 CPU. Do not claim final deployment latency from non-RPi runs.

## Reproducibility

| Field | Value |
| --- | --- |
| command | /Users/phongpham/Downloads/smartBinocular/.venv/bin/python tools/compare_classifiers.py --train data/training/from_logs_train.jsonl --test data/training/from_logs_test.jsonl --models logistic_regression linear_svm sgd_linear_svm gaussian_nb decision_tree random_forest_50 random_forest_100 random_forest_200_current_config random_forest_depth_8 random_forest_depth_12 random_forest_depth_none extra_trees gradient_boosting hist_gradient_boosting knn mlp_small_32 mlp_small_64_32 --bootstrap 100 --latency-repeats 100 --hardware-label macOS proxy benchmark (not Raspberry Pi 4 CPU) --persist-models --quick |
| git_commit | 15fbe64d5c70873cf200eac688f37d03e8f103da |
| branch | refactor/ml-taxonomy-eval-plan |
| feature_set_version | optical_12_baseline (12 features) |
| split_method | existing frozen JSONL train/test files; no tuning on test |
| random_seed | 42 |
| train_rows | 11981 |
| test_rows | 2113 |
| dataset_manifest_hash | 416c8ce44c6882625e1a8dd6c7bcd492f78a1177cb2baf0e01bf358ea91dc2ee |
| python | 3.12.12 |
| sklearn | 1.8.0 |
| numpy | 1.26.4 |
| scipy | 1.17.1 |
| hardware | macOS proxy benchmark (not Raspberry Pi 4 CPU) |
| latency_repeats | 100 |
| run_status | quick / preliminary / not thesis-ready |

## Dataset Files

| Split | Path | SHA256 | Bytes |
| --- | --- | --- | --- |
| train | data/training/from_logs_train.jsonl | 0a03e757a434956df1af4e5595d4bfc029242957b1cad15cb0053dfc0a744410 | 11501860 |
| test | data/training/from_logs_test.jsonl | 583ce85f80f1c065de2405c4bdccdab97098dcb2440dad4f4237bfbb2b662699 | 2027542 |

## Aggregate Metrics

| Model | Accuracy | Acc 95% CI | Balanced Acc | Bal Acc 95% CI | Macro-F1 | Weighted-F1 | ECE | Brier | Model Bytes | Train s | Load s | Mean ms | Median ms | P95 ms | Warnings | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.6872 | 0.6663-0.7047 | 0.6439 | 0.6201-0.6736 | 0.6073 | 0.6896 | 0.0460 | 0.4229 | 1775 | 21.625 | 0.0009 | 0.049 | 0.046 | 0.063 | 0 | quick / preliminary / not thesis-ready |
| linear_svm | 0.7250 | 0.7071-0.7454 | 0.6126 | 0.5902-0.6370 | 0.5863 | 0.7005 | n/a | n/a | 1691 | 0.202 | 0.0004 | 0.046 | 0.039 | 0.065 | 0 | quick / preliminary / not thesis-ready |
| sgd_linear_svm | 0.6687 | 0.6483-0.6879 | 0.5799 | 0.5563-0.6020 | 0.5501 | 0.6626 | 0.0944 | 0.4647 | 1647 | 0.482 | 0.0003 | 0.041 | 0.039 | 0.044 | 0 | quick / preliminary / not thesis-ready |
| gaussian_nb | 0.6559 | 0.6357-0.6806 | 0.5549 | 0.5274-0.5825 | 0.5487 | 0.6417 | 0.2138 | 0.5424 | 1755 | 0.007 | 0.0003 | 0.198 | 0.189 | 0.238 | 0 | quick / preliminary / not thesis-ready |
| decision_tree | 0.7582 | 0.7401-0.7722 | 0.6849 | 0.6574-0.7158 | 0.6572 | 0.7649 | 0.1567 | 0.3921 | 285489 | 0.174 | 0.0006 | 0.045 | 0.042 | 0.057 | 0 | quick / preliminary / not thesis-ready |
| random_forest_50 | 0.8476 | 0.8306-0.8602 | 0.7638 | 0.7407-0.7859 | 0.7609 | 0.8463 | 0.1049 | 0.2451 | 11504625 | 0.435 | 0.0198 | 13.996 | 14.049 | 14.494 | 0 | quick / preliminary / not thesis-ready |
| random_forest_100 | 0.8476 | 0.8291-0.8604 | 0.7677 | 0.7445-0.7912 | 0.7661 | 0.8464 | 0.1044 | 0.2444 | 22938913 | 1.143 | 0.0399 | 14.654 | 13.964 | 14.620 | 0 | quick / preliminary / not thesis-ready |
| random_forest_200_current_config | 0.8471 | 0.8289-0.8594 | 0.7665 | 0.7448-0.7931 | 0.7632 | 0.8459 | 0.1048 | 0.2436 | 46069425 | 2.223 | 0.0752 | 30.460 | 26.772 | 58.415 | 0 | quick / preliminary / not thesis-ready |
| random_forest_depth_8 | 0.7681 | 0.7474-0.7840 | 0.7037 | 0.6789-0.7287 | 0.6757 | 0.7698 | 0.1387 | 0.3504 | 8168129 | 1.658 | 0.0360 | 51.955 | 27.392 | 129.158 | 0 | quick / preliminary / not thesis-ready |
| random_forest_depth_12 | 0.8202 | 0.8026-0.8334 | 0.7475 | 0.7263-0.7731 | 0.7367 | 0.8202 | 0.1143 | 0.2780 | 26472913 | 2.844 | 0.1701 | 25.721 | 26.510 | 29.531 | 0 | quick / preliminary / not thesis-ready |
| random_forest_depth_none | 0.8457 | 0.8277-0.8592 | 0.7639 | 0.7417-0.7872 | 0.7615 | 0.8444 | 0.1029 | 0.2436 | 47148721 | 1.636 | 0.0954 | 43.812 | 28.421 | 129.621 | 0 | quick / preliminary / not thesis-ready |
| extra_trees | 0.8088 | 0.7917-0.8238 | 0.7487 | 0.7243-0.7706 | 0.7262 | 0.8093 | 0.2566 | 0.3652 | 39645857 | 1.742 | 0.0613 | 39.805 | 28.114 | 82.460 | 0 | quick / preliminary / not thesis-ready |
| gradient_boosting | 0.8183 | 0.8029-0.8354 | 0.6874 | 0.6615-0.7167 | 0.7109 | 0.8096 | 0.0187 | 0.2591 | 1147981 | 34.129 | 0.0181 | 0.705 | 0.658 | 0.946 | 0 | quick / preliminary / not thesis-ready |
| hist_gradient_boosting | 0.8443 | 0.8286-0.8578 | 0.7300 | 0.7057-0.7546 | 0.7471 | 0.8386 | 0.0325 | 0.2190 | 2025800 | 16.814 | 0.1215 | 88.459 | 72.728 | 204.187 | 0 | quick / preliminary / not thesis-ready |
| knn | 0.7998 | 0.7820-0.8159 | 0.6906 | 0.6614-0.7156 | 0.7092 | 0.7941 | 0.0451 | 0.2888 | 2032966 | 0.104 | 0.0014 | 0.598 | 0.570 | 0.754 | 0 | quick / preliminary / not thesis-ready |
| mlp_small_32 | 0.7875 | 0.7709-0.8084 | 0.6497 | 0.6286-0.6744 | 0.6687 | 0.7744 | 0.0343 | 0.3069 | 15892 | 4.923 | 0.0014 | 0.107 | 0.096 | 0.123 | 0 | quick / preliminary / not thesis-ready |
| mlp_small_64_32 | 0.8060 | 0.7898-0.8235 | 0.6669 | 0.6427-0.6893 | 0.6916 | 0.7950 | 0.0256 | 0.2867 | 45076 | 36.942 | 0.0026 | 0.114 | 0.101 | 0.175 | 0 | quick / preliminary / not thesis-ready |

## Model Provenance

| Model | Model Path | Model SHA256 | Serialized Bytes |
| --- | --- | --- | --- |
| logistic_regression | artifacts/ml/model_comparison_12features/logistic_regression/model.joblib | 25cb4981e634a94529956f79a2117f0ca2da7f73c19fb938c8cbe30739dd4c81 | 1775 |
| linear_svm | artifacts/ml/model_comparison_12features/linear_svm/model.joblib | eb9e34b0d8a5e8998135e50f57b5071bede0383bfc859831ed319dd106f4204f | 1691 |
| sgd_linear_svm | artifacts/ml/model_comparison_12features/sgd_linear_svm/model.joblib | e588e041d7d34a3475950289da20f14592e0350e4d546351078859fc5f0aecc5 | 1647 |
| gaussian_nb | artifacts/ml/model_comparison_12features/gaussian_nb/model.joblib | 3081ea0f211adc8670bb45639d7a08564f0d287e09f1e8ed6e43b53b9c7f99c7 | 1755 |
| decision_tree | artifacts/ml/model_comparison_12features/decision_tree/model.joblib | 25be284a8e04b82b38296191554d54781bb8cc3508c9fcbe0cbcde910a38aac3 | 285489 |
| random_forest_50 | artifacts/ml/model_comparison_12features/random_forest_50/model.joblib | 38056458efb3839911f10224488825af84572099e4d6e511a70129e06e059658 | 11504625 |
| random_forest_100 | artifacts/ml/model_comparison_12features/random_forest_100/model.joblib | b6ad25f2d6d64f2ab0a0572eca6ab2bb2d09f63dfefd2490a3d7f2b3250f83dd | 22938913 |
| random_forest_200_current_config | artifacts/ml/model_comparison_12features/random_forest_200_current_config/model.joblib | 148ec60827d02c9af199d6e167e774afa54e28658570340b06fcdb0b5be962e8 | 46069425 |
| random_forest_depth_8 | artifacts/ml/model_comparison_12features/random_forest_depth_8/model.joblib | 9ce42158375ea38c9e45ae6ef8981ae30bcbb9ca22dd2ed400c60586d3f80ae7 | 8168129 |
| random_forest_depth_12 | artifacts/ml/model_comparison_12features/random_forest_depth_12/model.joblib | 3cdbb196954121635bdfea82f0cf832a03b03b5e1e9b18bf1227252f2f56a7ae | 26472913 |
| random_forest_depth_none | artifacts/ml/model_comparison_12features/random_forest_depth_none/model.joblib | 71c2650c686a974714e416024921c9f60d37ba5593c5646fb17153ff3ed876ba | 47148721 |
| extra_trees | artifacts/ml/model_comparison_12features/extra_trees/model.joblib | c7ef80463a2eb8ddadc866b11aeaa4b1632221215c38bd13271c515690f25061 | 39645857 |
| gradient_boosting | artifacts/ml/model_comparison_12features/gradient_boosting/model.joblib | 1dcec647b78ac2db2eefa9bbeb4d647f1d52381a489e7a2879753162d770a384 | 1147981 |
| hist_gradient_boosting | artifacts/ml/model_comparison_12features/hist_gradient_boosting/model.joblib | 1e8c8bd2bb0e4c5413fcd7d5b33e99376135862daa9861e82617fcce26e620f9 | 2025800 |
| knn | artifacts/ml/model_comparison_12features/knn/model.joblib | 07f17af440ffc7ca8267b23d96122ecb6490f6727a03a1c071a37793061b6c32 | 2032966 |
| mlp_small_32 | artifacts/ml/model_comparison_12features/mlp_small_32/model.joblib | 96e0613b9cb85c41ab41df3ea471a25c8619b98ff53380e2f607363943cea84a | 15892 |
| mlp_small_64_32 | artifacts/ml/model_comparison_12features/mlp_small_64_32/model.joblib | d76c2425dab8e34b0b9e2cf89f03172eefa8c7a9166c3160ca5ceed6065dbb0b | 45076 |

## Per-Class Metrics

| Model | Class | Support | Precision | Recall | F1 | F1 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | night_clear | 417 | 0.8212 | 0.8369 | 0.8290 | 0.7981-0.8531 |
| logistic_regression | normal_night | 352 | 0.6985 | 0.5398 | 0.6090 | 0.5612-0.6463 |
| logistic_regression | normal_day | 366 | 0.6844 | 0.5273 | 0.5957 | 0.5426-0.6357 |
| logistic_regression | fog | 231 | 0.6742 | 0.7706 | 0.7192 | 0.6762-0.7604 |
| logistic_regression | rain | 194 | 0.4084 | 0.4021 | 0.4052 | 0.3401-0.4478 |
| logistic_regression | glare | 60 | 0.2626 | 0.4333 | 0.3270 | 0.2445-0.4132 |
| logistic_regression | backlight | 55 | 0.5233 | 0.8182 | 0.6383 | 0.5385-0.7052 |
| logistic_regression | transition | 77 | 0.3008 | 0.4805 | 0.3700 | 0.2886-0.4403 |
| logistic_regression | nir_night | 361 | 0.9596 | 0.9861 | 0.9727 | 0.9583-0.9832 |
| linear_svm | night_clear | 417 | 0.7944 | 0.8897 | 0.8394 | 0.8126-0.8626 |
| linear_svm | normal_night | 352 | 0.7197 | 0.6420 | 0.6787 | 0.6404-0.7188 |
| linear_svm | normal_day | 366 | 0.6502 | 0.7923 | 0.7143 | 0.6839-0.7539 |
| linear_svm | fog | 231 | 0.6385 | 0.8182 | 0.7173 | 0.6773-0.7645 |
| linear_svm | rain | 194 | 0.8333 | 0.1289 | 0.2232 | 0.1595-0.2896 |
| linear_svm | glare | 60 | 0.4286 | 0.1500 | 0.2222 | 0.1245-0.3210 |
| linear_svm | backlight | 55 | 0.4421 | 0.7636 | 0.5600 | 0.4602-0.6480 |
| linear_svm | transition | 77 | 0.3553 | 0.3506 | 0.3529 | 0.2527-0.4190 |
| linear_svm | nir_night | 361 | 0.9592 | 0.9778 | 0.9684 | 0.9529-0.9806 |
| sgd_linear_svm | night_clear | 417 | 0.7995 | 0.8321 | 0.8155 | 0.7806-0.8400 |
| sgd_linear_svm | normal_night | 352 | 0.6889 | 0.6165 | 0.6507 | 0.6107-0.6853 |
| sgd_linear_svm | normal_day | 366 | 0.6862 | 0.5437 | 0.6067 | 0.5617-0.6494 |
| sgd_linear_svm | fog | 231 | 0.6067 | 0.8615 | 0.7120 | 0.6758-0.7526 |
| sgd_linear_svm | rain | 194 | 0.4032 | 0.1289 | 0.1953 | 0.1457-0.2594 |
| sgd_linear_svm | glare | 60 | 0.1386 | 0.4667 | 0.2137 | 0.1512-0.2657 |
| sgd_linear_svm | backlight | 55 | 0.6552 | 0.6909 | 0.6726 | 0.5683-0.7731 |
| sgd_linear_svm | transition | 77 | 0.1509 | 0.1039 | 0.1231 | 0.0538-0.1993 |
| sgd_linear_svm | nir_night | 361 | 0.9488 | 0.9751 | 0.9617 | 0.9426-0.9748 |
| gaussian_nb | night_clear | 417 | 0.7511 | 0.8537 | 0.7991 | 0.7676-0.8273 |
| gaussian_nb | normal_night | 352 | 0.6237 | 0.5085 | 0.5603 | 0.5220-0.6016 |
| gaussian_nb | normal_day | 366 | 0.5266 | 0.7295 | 0.6117 | 0.5766-0.6534 |
| gaussian_nb | fog | 231 | 0.6009 | 0.5541 | 0.5766 | 0.5144-0.6278 |
| gaussian_nb | rain | 194 | 0.4028 | 0.1495 | 0.2180 | 0.1684-0.2804 |
| gaussian_nb | glare | 60 | 0.2740 | 0.3333 | 0.3008 | 0.2194-0.3842 |
| gaussian_nb | backlight | 55 | 0.5172 | 0.5455 | 0.5310 | 0.4231-0.6416 |
| gaussian_nb | transition | 77 | 0.3913 | 0.3506 | 0.3699 | 0.2760-0.4326 |
| gaussian_nb | nir_night | 361 | 0.9722 | 0.9695 | 0.9709 | 0.9569-0.9817 |
| decision_tree | night_clear | 417 | 0.9369 | 0.9616 | 0.9491 | 0.9325-0.9620 |
| decision_tree | normal_night | 352 | 0.8328 | 0.6790 | 0.7480 | 0.7079-0.7884 |
| decision_tree | normal_day | 366 | 0.7312 | 0.6393 | 0.6822 | 0.6401-0.7241 |
| decision_tree | fog | 231 | 0.7325 | 0.7229 | 0.7277 | 0.6798-0.7721 |
| decision_tree | rain | 194 | 0.5622 | 0.5361 | 0.5488 | 0.4980-0.5994 |
| decision_tree | glare | 60 | 0.2755 | 0.4500 | 0.3418 | 0.2537-0.4211 |
| decision_tree | backlight | 55 | 0.4286 | 0.6545 | 0.5180 | 0.4143-0.6181 |
| decision_tree | transition | 77 | 0.3500 | 0.5455 | 0.4264 | 0.3390-0.5000 |
| decision_tree | nir_night | 361 | 0.9697 | 0.9751 | 0.9724 | 0.9591-0.9837 |
| random_forest_50 | night_clear | 417 | 0.9689 | 0.9712 | 0.9701 | 0.9592-0.9806 |
| random_forest_50 | normal_night | 352 | 0.8629 | 0.8580 | 0.8604 | 0.8306-0.8864 |
| random_forest_50 | normal_day | 366 | 0.7916 | 0.8197 | 0.8054 | 0.7688-0.8316 |
| random_forest_50 | fog | 231 | 0.7983 | 0.8225 | 0.8102 | 0.7663-0.8478 |
| random_forest_50 | rain | 194 | 0.7529 | 0.6598 | 0.7033 | 0.6444-0.7594 |
| random_forest_50 | glare | 60 | 0.5882 | 0.5000 | 0.5405 | 0.4231-0.6333 |
| random_forest_50 | backlight | 55 | 0.5676 | 0.7636 | 0.6512 | 0.5535-0.7465 |
| random_forest_50 | transition | 77 | 0.5672 | 0.4935 | 0.5278 | 0.4257-0.6089 |
| random_forest_50 | nir_night | 361 | 0.9727 | 0.9861 | 0.9794 | 0.9682-0.9887 |
| random_forest_100 | night_clear | 417 | 0.9666 | 0.9712 | 0.9689 | 0.9590-0.9796 |
| random_forest_100 | normal_night | 352 | 0.8642 | 0.8494 | 0.8567 | 0.8262-0.8837 |
| random_forest_100 | normal_day | 366 | 0.7973 | 0.8169 | 0.8070 | 0.7721-0.8303 |
| random_forest_100 | fog | 231 | 0.8008 | 0.8182 | 0.8094 | 0.7716-0.8456 |
| random_forest_100 | rain | 194 | 0.7182 | 0.6701 | 0.6933 | 0.6389-0.7483 |
| random_forest_100 | glare | 60 | 0.6444 | 0.4833 | 0.5524 | 0.4471-0.6522 |
| random_forest_100 | backlight | 55 | 0.6143 | 0.7818 | 0.6880 | 0.5929-0.7705 |
| random_forest_100 | transition | 77 | 0.5467 | 0.5325 | 0.5395 | 0.4374-0.6069 |
| random_forest_100 | nir_night | 361 | 0.9727 | 0.9861 | 0.9794 | 0.9682-0.9887 |
| random_forest_200_current_config | night_clear | 417 | 0.9689 | 0.9712 | 0.9701 | 0.9608-0.9804 |
| random_forest_200_current_config | normal_night | 352 | 0.8678 | 0.8580 | 0.8629 | 0.8334-0.8876 |
| random_forest_200_current_config | normal_day | 366 | 0.7989 | 0.8142 | 0.8065 | 0.7693-0.8344 |
| random_forest_200_current_config | fog | 231 | 0.8069 | 0.8139 | 0.8103 | 0.7773-0.8486 |
| random_forest_200_current_config | rain | 194 | 0.7011 | 0.6649 | 0.6825 | 0.6282-0.7313 |
| random_forest_200_current_config | glare | 60 | 0.6042 | 0.4833 | 0.5370 | 0.4290-0.6313 |
| random_forest_200_current_config | backlight | 55 | 0.6111 | 0.8000 | 0.6929 | 0.5973-0.7784 |
| random_forest_200_current_config | transition | 77 | 0.5493 | 0.5065 | 0.5270 | 0.4224-0.6095 |
| random_forest_200_current_config | nir_night | 361 | 0.9727 | 0.9861 | 0.9794 | 0.9682-0.9887 |
| random_forest_depth_8 | night_clear | 417 | 0.9476 | 0.9544 | 0.9510 | 0.9349-0.9653 |
| random_forest_depth_8 | normal_night | 352 | 0.8511 | 0.7472 | 0.7958 | 0.7640-0.8273 |
| random_forest_depth_8 | normal_day | 366 | 0.7030 | 0.6339 | 0.6667 | 0.6264-0.7007 |
| random_forest_depth_8 | fog | 231 | 0.6860 | 0.7662 | 0.7239 | 0.6781-0.7625 |
| random_forest_depth_8 | rain | 194 | 0.5153 | 0.4330 | 0.4706 | 0.4131-0.5393 |
| random_forest_depth_8 | glare | 60 | 0.3929 | 0.5500 | 0.4583 | 0.3622-0.5479 |
| random_forest_depth_8 | backlight | 55 | 0.5000 | 0.7818 | 0.6099 | 0.5201-0.6974 |
| random_forest_depth_8 | transition | 77 | 0.3814 | 0.4805 | 0.4253 | 0.3306-0.5029 |
| random_forest_depth_8 | nir_night | 361 | 0.9727 | 0.9861 | 0.9794 | 0.9682-0.9887 |
| random_forest_depth_12 | night_clear | 417 | 0.9665 | 0.9688 | 0.9677 | 0.9566-0.9780 |
| random_forest_depth_12 | normal_night | 352 | 0.8720 | 0.8324 | 0.8517 | 0.8179-0.8779 |
| random_forest_depth_12 | normal_day | 366 | 0.7521 | 0.7377 | 0.7448 | 0.7114-0.7801 |
| random_forest_depth_12 | fog | 231 | 0.7647 | 0.7879 | 0.7761 | 0.7374-0.8163 |
| random_forest_depth_12 | rain | 194 | 0.6162 | 0.5876 | 0.6016 | 0.5379-0.6623 |
| random_forest_depth_12 | glare | 60 | 0.5333 | 0.5333 | 0.5333 | 0.4266-0.6438 |
| random_forest_depth_12 | backlight | 55 | 0.5946 | 0.8000 | 0.6822 | 0.5895-0.7653 |
| random_forest_depth_12 | transition | 77 | 0.4935 | 0.4935 | 0.4935 | 0.4040-0.5618 |
| random_forest_depth_12 | nir_night | 361 | 0.9727 | 0.9861 | 0.9794 | 0.9682-0.9887 |
| random_forest_depth_none | night_clear | 417 | 0.9665 | 0.9688 | 0.9677 | 0.9563-0.9778 |
| random_forest_depth_none | normal_night | 352 | 0.8625 | 0.8551 | 0.8588 | 0.8313-0.8827 |
| random_forest_depth_none | normal_day | 366 | 0.7957 | 0.8087 | 0.8022 | 0.7662-0.8277 |
| random_forest_depth_none | fog | 231 | 0.8008 | 0.8182 | 0.8094 | 0.7684-0.8435 |
| random_forest_depth_none | rain | 194 | 0.7065 | 0.6701 | 0.6878 | 0.6418-0.7460 |
| random_forest_depth_none | glare | 60 | 0.6087 | 0.4667 | 0.5283 | 0.4097-0.6302 |
| random_forest_depth_none | backlight | 55 | 0.5890 | 0.7818 | 0.6719 | 0.5763-0.7546 |
| random_forest_depth_none | transition | 77 | 0.5797 | 0.5195 | 0.5479 | 0.4435-0.6230 |
| random_forest_depth_none | nir_night | 361 | 0.9727 | 0.9861 | 0.9794 | 0.9682-0.9887 |
| extra_trees | night_clear | 417 | 0.9327 | 0.9640 | 0.9481 | 0.9353-0.9628 |
| extra_trees | normal_night | 352 | 0.8626 | 0.7670 | 0.8120 | 0.7831-0.8429 |
| extra_trees | normal_day | 366 | 0.7809 | 0.7596 | 0.7701 | 0.7323-0.7978 |
| extra_trees | fog | 231 | 0.7233 | 0.7922 | 0.7562 | 0.7165-0.7989 |
| extra_trees | rain | 194 | 0.7123 | 0.5361 | 0.6118 | 0.5579-0.6745 |
| extra_trees | glare | 60 | 0.5238 | 0.5500 | 0.5366 | 0.4240-0.6227 |
| extra_trees | backlight | 55 | 0.5000 | 0.8182 | 0.6207 | 0.5304-0.7101 |
| extra_trees | transition | 77 | 0.4639 | 0.5844 | 0.5172 | 0.4316-0.5767 |
| extra_trees | nir_night | 361 | 0.9588 | 0.9668 | 0.9628 | 0.9483-0.9747 |
| gradient_boosting | night_clear | 417 | 0.9482 | 0.9664 | 0.9572 | 0.9466-0.9708 |
| gradient_boosting | normal_night | 352 | 0.8047 | 0.8778 | 0.8397 | 0.8095-0.8635 |
| gradient_boosting | normal_day | 366 | 0.6968 | 0.8415 | 0.7624 | 0.7295-0.7922 |
| gradient_boosting | fog | 231 | 0.7692 | 0.7792 | 0.7742 | 0.7333-0.8159 |
| gradient_boosting | rain | 194 | 0.6463 | 0.4897 | 0.5572 | 0.4934-0.6333 |
| gradient_boosting | glare | 60 | 0.7241 | 0.3500 | 0.4719 | 0.3656-0.5970 |
| gradient_boosting | backlight | 55 | 0.6383 | 0.5455 | 0.5882 | 0.4805-0.6945 |
| gradient_boosting | transition | 77 | 0.7105 | 0.3506 | 0.4696 | 0.3455-0.5607 |
| gradient_boosting | nir_night | 361 | 0.9700 | 0.9861 | 0.9780 | 0.9655-0.9879 |
| hist_gradient_boosting | night_clear | 417 | 0.9739 | 0.9832 | 0.9785 | 0.9670-0.9870 |
| hist_gradient_boosting | normal_night | 352 | 0.8373 | 0.8920 | 0.8638 | 0.8383-0.8892 |
| hist_gradient_boosting | normal_day | 366 | 0.7441 | 0.8579 | 0.7970 | 0.7645-0.8258 |
| hist_gradient_boosting | fog | 231 | 0.7931 | 0.7965 | 0.7948 | 0.7589-0.8356 |
| hist_gradient_boosting | rain | 194 | 0.7117 | 0.5979 | 0.6499 | 0.5806-0.7075 |
| hist_gradient_boosting | glare | 60 | 0.6053 | 0.3833 | 0.4694 | 0.3602-0.5993 |
| hist_gradient_boosting | backlight | 55 | 0.6852 | 0.6727 | 0.6789 | 0.5891-0.7752 |
| hist_gradient_boosting | transition | 77 | 0.7045 | 0.4026 | 0.5124 | 0.4163-0.6228 |
| hist_gradient_boosting | nir_night | 361 | 0.9753 | 0.9834 | 0.9793 | 0.9692-0.9891 |
| knn | night_clear | 417 | 0.9117 | 0.9904 | 0.9494 | 0.9350-0.9631 |
| knn | normal_night | 352 | 0.8440 | 0.7841 | 0.8130 | 0.7767-0.8497 |
| knn | normal_day | 366 | 0.6959 | 0.8251 | 0.7550 | 0.7186-0.7861 |
| knn | fog | 231 | 0.7303 | 0.7619 | 0.7458 | 0.6944-0.7933 |
| knn | rain | 194 | 0.6065 | 0.4845 | 0.5387 | 0.4822-0.6014 |
| knn | glare | 60 | 0.5909 | 0.4333 | 0.5000 | 0.3883-0.6076 |
| knn | backlight | 55 | 0.7317 | 0.5455 | 0.6250 | 0.4978-0.7162 |
| knn | transition | 77 | 0.6034 | 0.4545 | 0.5185 | 0.4024-0.6101 |
| knn | nir_night | 361 | 0.9389 | 0.9363 | 0.9376 | 0.9162-0.9535 |
| mlp_small_32 | night_clear | 417 | 0.9138 | 0.9400 | 0.9267 | 0.9075-0.9399 |
| mlp_small_32 | normal_night | 352 | 0.7898 | 0.8324 | 0.8105 | 0.7803-0.8383 |
| mlp_small_32 | normal_day | 366 | 0.6711 | 0.8251 | 0.7402 | 0.7061-0.7746 |
| mlp_small_32 | fog | 231 | 0.6803 | 0.7922 | 0.7320 | 0.6867-0.7748 |
| mlp_small_32 | rain | 194 | 0.6542 | 0.3608 | 0.4651 | 0.3909-0.5514 |
| mlp_small_32 | glare | 60 | 0.5600 | 0.2333 | 0.3294 | 0.2322-0.4612 |
| mlp_small_32 | backlight | 55 | 0.7692 | 0.5455 | 0.6383 | 0.5360-0.7231 |
| mlp_small_32 | transition | 77 | 0.5306 | 0.3377 | 0.4127 | 0.3030-0.4869 |
| mlp_small_32 | nir_night | 361 | 0.9465 | 0.9806 | 0.9633 | 0.9478-0.9751 |
| mlp_small_64_32 | night_clear | 417 | 0.9134 | 0.9616 | 0.9369 | 0.9221-0.9503 |
| mlp_small_64_32 | normal_night | 352 | 0.8049 | 0.8324 | 0.8184 | 0.7890-0.8460 |
| mlp_small_64_32 | normal_day | 366 | 0.6786 | 0.8825 | 0.7672 | 0.7330-0.7980 |
| mlp_small_64_32 | fog | 231 | 0.7685 | 0.7186 | 0.7427 | 0.7006-0.7837 |
| mlp_small_64_32 | rain | 194 | 0.6739 | 0.4794 | 0.5602 | 0.5046-0.6235 |
| mlp_small_64_32 | glare | 60 | 0.6190 | 0.2167 | 0.3210 | 0.2173-0.4559 |
| mlp_small_64_32 | backlight | 55 | 0.8158 | 0.5636 | 0.6667 | 0.5620-0.7449 |
| mlp_small_64_32 | transition | 77 | 0.5600 | 0.3636 | 0.4409 | 0.3342-0.5200 |
| mlp_small_64_32 | nir_night | 361 | 0.9569 | 0.9834 | 0.9699 | 0.9508-0.9823 |

## Confusion Matrices

Raw and normalized confusion matrices are written as CSV/PNG artifacts per model when this script runs.
