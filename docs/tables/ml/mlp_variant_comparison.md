# MLP Variant Comparison

Status: offline optical RGB-proxy baseline; not live NIR/LWIR validated. Calibrated rows use an internal train/validation split only, not the test set.

| Model | Calibration | Seeds | Balanced acc mean | Macro-F1 mean | ECE mean | Brier mean | Abstain mean | Latency ms | Bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calibrated_mlp_32 | calibrated_internal_train_validation_split | 42,43,44 | 0.6283 | 0.6406 | 0.0397 | 0.3395 | 0.1972 | 1.3482 | 15673 |
| calibrated_mlp_64 | calibrated_internal_train_validation_split | 42,43,44 | 0.6382 | 0.6525 | 0.0412 | 0.3262 | 0.1861 | 1.0549 | 23906 |
| extra_trees | uncalibrated | 42,43,44 | 0.7370 | 0.7139 | 0.2514 | 0.3710 | 0.6365 | 24.4796 | 39772267 |
| hist_gradient_boosting | uncalibrated | 42,43,44 | 0.7131 | 0.7244 | 0.0396 | 0.2455 | 0.1295 | 57.3592 | 1877884 |
| mlp_128 | uncalibrated | 42,43,44 | 0.6658 | 0.6780 | 0.0245 | 0.2927 | 0.2535 | 0.2291 | 40374 |
| mlp_128_64 | uncalibrated | 42,43,44 | 0.6854 | 0.6926 | 0.0177 | 0.2799 | 0.2190 | 0.2090 | 132645 |
| mlp_32 | uncalibrated | 42,43,44 | 0.6523 | 0.6645 | 0.0233 | 0.3021 | 0.2593 | 0.2608 | 16001 |
| mlp_64 | uncalibrated | 42,43,44 | 0.6457 | 0.6588 | 0.0245 | 0.3086 | 0.2847 | 0.2128 | 23198 |
| mlp_64_32 | uncalibrated | 42,43,44 | 0.6882 | 0.6973 | 0.0123 | 0.2821 | 0.2196 | 0.2686 | 45701 |
| random_forest_100 | uncalibrated | 42,43,44 | 0.7477 | 0.7384 | 0.0897 | 0.2597 | 0.3269 | 13.9686 | 23023886 |
| random_forest_200_current_config | uncalibrated | 42,43,44 | 0.7480 | 0.7380 | 0.0895 | 0.2587 | 0.3294 | 24.0612 | 45986155 |

## Low-Support Class F1

| Model | Class | Seed | F1 |
| --- | --- | --- | --- |
| random_forest_100 | glare | 42 | 0.5210 |
| random_forest_100 | backlight | 42 | 0.6324 |
| random_forest_100 | transition | 42 | 0.4624 |
| random_forest_200_current_config | glare | 42 | 0.5378 |
| random_forest_200_current_config | backlight | 42 | 0.6331 |
| random_forest_200_current_config | transition | 42 | 0.4624 |
| extra_trees | glare | 42 | 0.5038 |
| extra_trees | backlight | 42 | 0.6065 |
| extra_trees | transition | 42 | 0.4976 |
| hist_gradient_boosting | glare | 42 | 0.4848 |
| hist_gradient_boosting | backlight | 42 | 0.6346 |
| hist_gradient_boosting | transition | 42 | 0.3924 |
| mlp_32 | glare | 42 | 0.3721 |
| mlp_32 | backlight | 42 | 0.5825 |
| mlp_32 | transition | 42 | 0.4048 |
| calibrated_mlp_32 | glare | 42 | 0.3614 |
| calibrated_mlp_32 | backlight | 42 | 0.5631 |
| calibrated_mlp_32 | transition | 42 | 0.3333 |
| mlp_64 | glare | 42 | 0.3529 |
| mlp_64 | backlight | 42 | 0.5872 |
| mlp_64 | transition | 42 | 0.3827 |
| calibrated_mlp_64 | glare | 42 | 0.3614 |
| calibrated_mlp_64 | backlight | 42 | 0.5825 |
| calibrated_mlp_64 | transition | 42 | 0.3669 |
| mlp_128 | glare | 42 | 0.3721 |
| mlp_128 | backlight | 42 | 0.5849 |
| mlp_128 | transition | 42 | 0.3590 |
| mlp_64_32 | glare | 42 | 0.4130 |
| mlp_64_32 | backlight | 42 | 0.6422 |
| mlp_64_32 | transition | 42 | 0.3879 |
| mlp_128_64 | glare | 42 | 0.3333 |
| mlp_128_64 | backlight | 42 | 0.6129 |
| mlp_128_64 | transition | 42 | 0.3614 |
| random_forest_100 | glare | 43 | 0.5167 |
| random_forest_100 | backlight | 43 | 0.6765 |
| random_forest_100 | transition | 43 | 0.4706 |
| random_forest_200_current_config | glare | 43 | 0.5210 |
| random_forest_200_current_config | backlight | 43 | 0.6667 |
| random_forest_200_current_config | transition | 43 | 0.4432 |
| extra_trees | glare | 43 | 0.4882 |
| extra_trees | backlight | 43 | 0.5860 |
| extra_trees | transition | 43 | 0.4752 |
| hist_gradient_boosting | glare | 43 | 0.4400 |
| hist_gradient_boosting | backlight | 43 | 0.6549 |
| hist_gradient_boosting | transition | 43 | 0.3522 |
| mlp_32 | glare | 43 | 0.3614 |
| mlp_32 | backlight | 43 | 0.6346 |
| mlp_32 | transition | 43 | 0.3659 |
| calibrated_mlp_32 | glare | 43 | 0.3500 |
| calibrated_mlp_32 | backlight | 43 | 0.5773 |
| calibrated_mlp_32 | transition | 43 | 0.3333 |
| mlp_64 | glare | 43 | 0.4000 |
| mlp_64 | backlight | 43 | 0.5882 |
| mlp_64 | transition | 43 | 0.3291 |
| calibrated_mlp_64 | glare | 43 | 0.4096 |
| calibrated_mlp_64 | backlight | 43 | 0.6122 |
| calibrated_mlp_64 | transition | 43 | 0.3210 |
| mlp_128 | glare | 43 | 0.3956 |
| mlp_128 | backlight | 43 | 0.6111 |
| mlp_128 | transition | 43 | 0.4024 |
| mlp_64_32 | glare | 43 | 0.4356 |
| mlp_64_32 | backlight | 43 | 0.6500 |
| mlp_64_32 | transition | 43 | 0.4294 |
| mlp_128_64 | glare | 43 | 0.4000 |
| mlp_128_64 | backlight | 43 | 0.6355 |
| mlp_128_64 | transition | 43 | 0.4070 |
| random_forest_100 | glare | 44 | 0.5470 |
| random_forest_100 | backlight | 44 | 0.6667 |
| random_forest_100 | transition | 44 | 0.4541 |
| random_forest_200_current_config | glare | 44 | 0.5565 |
| random_forest_200_current_config | backlight | 44 | 0.6423 |
| random_forest_200_current_config | transition | 44 | 0.4656 |
| extra_trees | glare | 44 | 0.5000 |
| extra_trees | backlight | 44 | 0.5987 |
| extra_trees | transition | 44 | 0.4975 |
| hist_gradient_boosting | glare | 44 | 0.5000 |
| hist_gradient_boosting | backlight | 44 | 0.6972 |
| hist_gradient_boosting | transition | 44 | 0.3625 |
| mlp_32 | glare | 44 | 0.3373 |
| mlp_32 | backlight | 44 | 0.5769 |
| mlp_32 | transition | 44 | 0.3165 |
| calibrated_mlp_32 | glare | 44 | 0.3636 |
| calibrated_mlp_32 | backlight | 44 | 0.5714 |
| calibrated_mlp_32 | transition | 44 | 0.3558 |
| mlp_64 | glare | 44 | 0.3704 |
| mlp_64 | backlight | 44 | 0.5743 |
| mlp_64 | transition | 44 | 0.3625 |
| calibrated_mlp_64 | glare | 44 | 0.3333 |
| calibrated_mlp_64 | backlight | 44 | 0.5474 |
| calibrated_mlp_64 | transition | 44 | 0.3580 |
| mlp_128 | glare | 44 | 0.4091 |
| mlp_128 | backlight | 44 | 0.5714 |
| mlp_128 | transition | 44 | 0.3855 |
| mlp_64_32 | glare | 44 | 0.4348 |
| mlp_64_32 | backlight | 44 | 0.5981 |
| mlp_64_32 | transition | 44 | 0.4186 |
| mlp_128_64 | glare | 44 | 0.4242 |
| mlp_128_64 | backlight | 44 | 0.6607 |
| mlp_128_64 | transition | 44 | 0.4205 |
