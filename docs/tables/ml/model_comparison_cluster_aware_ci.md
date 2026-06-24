# Model Comparison CI Summary

Status: preliminary / not thesis-ready until leakage and metric discrepancy are resolved.

## Aggregate 95% Bootstrap CI

| Model | Accuracy | Balanced accuracy | Macro-F1 | Weighted-F1 |
| --- | --- | --- | --- | --- |
| random_forest_100 | 0.8062-0.8408 | 0.7152-0.7633 | 0.7080-0.7552 | 0.8034-0.8365 |
| random_forest_200_current_config | 0.8090-0.8429 | 0.7202-0.7690 | 0.7117-0.7588 | 0.8072-0.8403 |
| extra_trees | 0.7834-0.8202 | 0.7133-0.7611 | 0.6917-0.7392 | 0.7814-0.8170 |
| logistic_regression | 0.6486-0.6872 | 0.6086-0.6557 | 0.5718-0.6148 | 0.6499-0.6901 |
| linear_svm | 0.6969-0.7366 | 0.5892-0.6313 | 0.5603-0.6080 | 0.6694-0.7137 |
| hist_gradient_boosting | 0.8114-0.8445 | 0.6861-0.7334 | 0.6998-0.7479 | 0.8016-0.8349 |
| mlp_small_32 | 0.7780-0.8121 | 0.6370-0.6845 | 0.6501-0.7018 | 0.7605-0.7971 |

## Low-Support Class F1 95% Bootstrap CI

| Model | Class | Support | F1 CI |
| --- | --- | --- | --- |
| random_forest_100 | glare | 60 | 0.4162-0.6285 |
| random_forest_100 | backlight | 55 | 0.5280-0.7212 |
| random_forest_100 | transition | 112 | 0.3730-0.5475 |
| random_forest_200_current_config | glare | 60 | 0.4324-0.6444 |
| random_forest_200_current_config | backlight | 55 | 0.5381-0.7211 |
| random_forest_200_current_config | transition | 112 | 0.3766-0.5456 |
| extra_trees | glare | 60 | 0.4036-0.5994 |
| extra_trees | backlight | 55 | 0.5089-0.6857 |
| extra_trees | transition | 112 | 0.4057-0.5800 |
| logistic_regression | glare | 60 | 0.2480-0.4047 |
| logistic_regression | backlight | 55 | 0.4798-0.6625 |
| logistic_regression | transition | 112 | 0.3417-0.5022 |
| linear_svm | glare | 60 | 0.1324-0.3628 |
| linear_svm | backlight | 55 | 0.4595-0.6483 |
| linear_svm | transition | 112 | 0.2690-0.4463 |
| hist_gradient_boosting | glare | 60 | 0.3728-0.5907 |
| hist_gradient_boosting | backlight | 55 | 0.5189-0.7291 |
| hist_gradient_boosting | transition | 112 | 0.2946-0.4909 |
| mlp_small_32 | glare | 60 | 0.2377-0.4860 |
| mlp_small_32 | backlight | 55 | 0.4689-0.6818 |
| mlp_small_32 | transition | 112 | 0.3121-0.4917 |
