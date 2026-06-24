# Feature 21 Ablation

`optical_21_candidate` includes `temporal_brightness_std`, which is not zero-imputed for still-image rows. If rows are insufficient, this is a data coverage limitation, not a model result.
`optical_21_candidate_still` is the still-compatible 21-derived candidate and reports its actual feature count separately.

| Feature set | Model | Status | Train | Test | Classes | Coverage | Features | Balanced acc | Macro-F1 | ECE | Brier | Mean ms | Limitation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| optical_21_candidate_still | random_forest_100 | subset/preliminary | 4923 | 868 | fog,normal_day,rain,transition | 4/9 | 20 | 0.7560 | 0.7695 | 0.1044 | 0.3054 | 28.0438 | subset comparison only; do not compare directly with full 9-class baseline |
| optical_21_candidate_still | random_forest_200_current_config | subset/preliminary | 4923 | 868 | fog,normal_day,rain,transition | 4/9 | 20 | 0.7584 | 0.7719 | 0.1108 | 0.3058 | 59.6023 | subset comparison only; do not compare directly with full 9-class baseline |
| optical_21_candidate_still | extra_trees | subset/preliminary | 4923 | 868 | fog,normal_day,rain,transition | 4/9 | 20 | 0.7616 | 0.7645 | 0.1683 | 0.3533 | 43.3325 | subset comparison only; do not compare directly with full 9-class baseline |
