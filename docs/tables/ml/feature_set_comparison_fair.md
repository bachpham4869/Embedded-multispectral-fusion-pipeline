# Feature Set Comparison: 12 vs v2 vs 21

Status: non-production ablation. Subset feature sets are not compared directly against full 9-class baseline unless class coverage matches.

`optical_21_candidate_still` excludes temporal-only fields; its actual feature count is reported in the table and it must not be described as the full temporal 21-feature set.

| Feature set | Model | Status | Train | Test | Classes | Coverage | Features | Balanced acc | Macro-F1 | ECE | Brier | Mean ms | Limitation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| optical_12_baseline | random_forest_100 | subset/preliminary | 4923 | 868 | fog,normal_day,rain,transition | 4/9 | 12 | 0.7452 | 0.7559 | 0.1209 | 0.3427 | 35.1970 | subset comparison only; do not compare directly with full 9-class baseline |
| optical_12_baseline | random_forest_200_current_config | subset/preliminary | 4923 | 868 | fog,normal_day,rain,transition | 4/9 | 12 | 0.7482 | 0.7590 | 0.1267 | 0.3409 | 45.3736 | subset comparison only; do not compare directly with full 9-class baseline |
| optical_12_baseline | extra_trees | subset/preliminary | 4923 | 868 | fog,normal_day,rain,transition | 4/9 | 12 | 0.7284 | 0.7281 | 0.2200 | 0.4352 | 50.7813 | subset comparison only; do not compare directly with full 9-class baseline |
| optical_v2_candidate | random_forest_100 | subset/preliminary | 4923 | 868 | fog,normal_day,rain,transition | 4/9 | 20 | 0.7560 | 0.7695 | 0.1044 | 0.3054 | 21.5134 | subset comparison only; do not compare directly with full 9-class baseline |
| optical_v2_candidate | random_forest_200_current_config | subset/preliminary | 4923 | 868 | fog,normal_day,rain,transition | 4/9 | 20 | 0.7584 | 0.7719 | 0.1108 | 0.3058 | 57.5308 | subset comparison only; do not compare directly with full 9-class baseline |
| optical_v2_candidate | extra_trees | subset/preliminary | 4923 | 868 | fog,normal_day,rain,transition | 4/9 | 20 | 0.7616 | 0.7645 | 0.1683 | 0.3533 | 39.6885 | subset comparison only; do not compare directly with full 9-class baseline |
| optical_21_candidate_still | random_forest_100 | subset/preliminary | 4923 | 868 | fog,normal_day,rain,transition | 4/9 | 20 | 0.7560 | 0.7695 | 0.1044 | 0.3054 | 28.0438 | subset comparison only; do not compare directly with full 9-class baseline |
| optical_21_candidate_still | random_forest_200_current_config | subset/preliminary | 4923 | 868 | fog,normal_day,rain,transition | 4/9 | 20 | 0.7584 | 0.7719 | 0.1108 | 0.3058 | 59.6023 | subset comparison only; do not compare directly with full 9-class baseline |
| optical_21_candidate_still | extra_trees | subset/preliminary | 4923 | 868 | fog,normal_day,rain,transition | 4/9 | 20 | 0.7616 | 0.7645 | 0.1683 | 0.3533 | 43.3325 | subset comparison only; do not compare directly with full 9-class baseline |

HistGradientBoosting was attempted in this Phase M10 subset run and stopped for
runtime. No HGB feature-set number is reported here.
