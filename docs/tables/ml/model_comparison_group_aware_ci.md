# Model Comparison CI Summary

Status: preliminary / not thesis-ready until leakage and metric discrepancy are resolved.

## Aggregate 95% Bootstrap CI

| Model | Accuracy | Balanced accuracy | Macro-F1 | Weighted-F1 |
| --- | --- | --- | --- | --- |
| random_forest_100 | 0.8201-0.8523 | 0.7386-0.7846 | 0.7282-0.7727 | 0.8207-0.8515 |
| random_forest_200_current_config | 0.8206-0.8519 | 0.7311-0.7806 | 0.7242-0.7693 | 0.8209-0.8506 |
| extra_trees | 0.7847-0.8183 | 0.7058-0.7534 | 0.6768-0.7237 | 0.7860-0.8198 |
| logistic_regression | 0.6559-0.6952 | 0.6172-0.6649 | 0.5775-0.6241 | 0.6592-0.6989 |
| linear_svm | 0.6997-0.7390 | 0.5828-0.6260 | 0.5534-0.6013 | 0.6737-0.7151 |
| hist_gradient_boosting | 0.8341-0.8632 | 0.7160-0.7651 | 0.7293-0.7800 | 0.8286-0.8596 |
| mlp_small_32 | 0.7728-0.8069 | 0.6308-0.6819 | 0.6483-0.7004 | 0.7603-0.7947 |

## Low-Support Class F1 95% Bootstrap CI

| Model | Class | Support | F1 CI |
| --- | --- | --- | --- |
| random_forest_100 | glare | 60 | 0.4178-0.6222 |
| random_forest_100 | backlight | 55 | 0.5763-0.7641 |
| random_forest_100 | transition | 77 | 0.4415-0.6211 |
| random_forest_200_current_config | glare | 60 | 0.4219-0.6270 |
| random_forest_200_current_config | backlight | 55 | 0.5687-0.7674 |
| random_forest_200_current_config | transition | 77 | 0.4044-0.5876 |
| extra_trees | glare | 60 | 0.3589-0.5768 |
| extra_trees | backlight | 55 | 0.4660-0.6565 |
| extra_trees | transition | 77 | 0.3390-0.5337 |
| logistic_regression | glare | 60 | 0.2696-0.4444 |
| logistic_regression | backlight | 55 | 0.5343-0.7224 |
| logistic_regression | transition | 77 | 0.2942-0.4697 |
| linear_svm | glare | 60 | 0.0690-0.2674 |
| linear_svm | backlight | 55 | 0.4926-0.6764 |
| linear_svm | transition | 77 | 0.2402-0.4410 |
| hist_gradient_boosting | glare | 60 | 0.3735-0.6105 |
| hist_gradient_boosting | backlight | 55 | 0.5609-0.7542 |
| hist_gradient_boosting | transition | 77 | 0.4417-0.6450 |
| mlp_small_32 | glare | 60 | 0.2171-0.4690 |
| mlp_small_32 | backlight | 55 | 0.5270-0.7339 |
| mlp_small_32 | transition | 77 | 0.3043-0.5329 |
