# Feature Set Comparison: 12 vs v2 vs 21

Status: non-production ablation. Subset feature sets are not compared directly against full 9-class baseline unless class coverage matches.

| Feature set | Model | Status | Train | Test | Classes | Balanced acc | Macro-F1 | ECE | Brier | Mean ms | Limitation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| optical_12_baseline | random_forest_100 | subset/preliminary | 4923 | 868 | fog,normal_day,rain,transition | 0.7452 | 0.7559 | 0.1209 | 0.3427 | 15.8075 | subset comparison only; do not compare directly with full 9-class baseline |
| optical_12_baseline | random_forest_200_current_config | subset/preliminary | 4923 | 868 | fog,normal_day,rain,transition | 0.7482 | 0.7590 | 0.1267 | 0.3409 | 20.5134 | subset comparison only; do not compare directly with full 9-class baseline |
| optical_12_baseline | extra_trees | subset/preliminary | 4923 | 868 | fog,normal_day,rain,transition | 0.7284 | 0.7281 | 0.2200 | 0.4352 | 18.8488 | subset comparison only; do not compare directly with full 9-class baseline |
| optical_v2_candidate | random_forest_100 | subset/preliminary | 4923 | 868 | fog,normal_day,rain,transition | 0.7560 | 0.7695 | 0.1044 | 0.3054 | 13.6107 | subset comparison only; do not compare directly with full 9-class baseline |
| optical_v2_candidate | random_forest_200_current_config | subset/preliminary | 4923 | 868 | fog,normal_day,rain,transition | 0.7584 | 0.7719 | 0.1108 | 0.3058 | 22.9910 | subset comparison only; do not compare directly with full 9-class baseline |
| optical_v2_candidate | extra_trees | subset/preliminary | 4923 | 868 | fog,normal_day,rain,transition | 0.7616 | 0.7645 | 0.1683 | 0.3533 | 20.3821 | subset comparison only; do not compare directly with full 9-class baseline |
| optical_21_candidate | random_forest_100 | not_run_insufficient_compatible_labeled_rows | 0 | 0 |  | n/a | n/a | n/a | n/a | n/a | subset comparison only; do not compare directly with full 9-class baseline |
| optical_21_candidate | random_forest_200_current_config | not_run_insufficient_compatible_labeled_rows | 0 | 0 |  | n/a | n/a | n/a | n/a | n/a | subset comparison only; do not compare directly with full 9-class baseline |
| optical_21_candidate | extra_trees | not_run_insufficient_compatible_labeled_rows | 0 | 0 |  | n/a | n/a | n/a | n/a | n/a | subset comparison only; do not compare directly with full 9-class baseline |
