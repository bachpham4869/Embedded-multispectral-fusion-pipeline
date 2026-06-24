# Model Comparison CI Summary

Status: preliminary / not thesis-ready until leakage and metric discrepancy are resolved.

## Aggregate 95% Bootstrap CI

| Model | Accuracy | Balanced accuracy | Macro-F1 | Weighted-F1 |
| --- | --- | --- | --- | --- |
| logistic_regression | 0.6663-0.7047 | 0.6201-0.6736 | 0.5798-0.6288 | 0.6722-0.7079 |
| linear_svm | 0.7071-0.7454 | 0.5902-0.6370 | 0.5613-0.6131 | 0.6824-0.7215 |
| sgd_linear_svm | 0.6483-0.6879 | 0.5563-0.6020 | 0.5269-0.5687 | 0.6473-0.6851 |
| gaussian_nb | 0.6357-0.6806 | 0.5274-0.5825 | 0.5234-0.5751 | 0.6198-0.6597 |
| decision_tree | 0.7401-0.7722 | 0.6574-0.7158 | 0.6299-0.6805 | 0.7472-0.7830 |
| random_forest_50 | 0.8306-0.8602 | 0.7407-0.7859 | 0.7373-0.7855 | 0.8306-0.8581 |
| random_forest_100 | 0.8291-0.8604 | 0.7445-0.7912 | 0.7431-0.7895 | 0.8320-0.8599 |
| random_forest_200_current_config | 0.8289-0.8594 | 0.7448-0.7931 | 0.7390-0.7855 | 0.8293-0.8597 |
| random_forest_depth_8 | 0.7474-0.7840 | 0.6789-0.7287 | 0.6532-0.6973 | 0.7535-0.7904 |
| random_forest_depth_12 | 0.8026-0.8334 | 0.7263-0.7731 | 0.7151-0.7608 | 0.8048-0.8359 |
| random_forest_depth_none | 0.8277-0.8592 | 0.7417-0.7872 | 0.7365-0.7853 | 0.8300-0.8581 |
| extra_trees | 0.7917-0.8238 | 0.7243-0.7706 | 0.7035-0.7488 | 0.7899-0.8256 |
| gradient_boosting | 0.8029-0.8354 | 0.6615-0.7167 | 0.6763-0.7371 | 0.7915-0.8263 |
| hist_gradient_boosting | 0.8286-0.8578 | 0.7057-0.7546 | 0.7240-0.7735 | 0.8253-0.8537 |
| knn | 0.7820-0.8159 | 0.6614-0.7156 | 0.6816-0.7335 | 0.7764-0.8095 |
| mlp_small_32 | 0.7709-0.8084 | 0.6286-0.6744 | 0.6394-0.6935 | 0.7562-0.7929 |
| mlp_small_64_32 | 0.7898-0.8235 | 0.6427-0.6893 | 0.6648-0.7197 | 0.7769-0.8112 |

## Low-Support Class F1 95% Bootstrap CI

| Model | Class | Support | F1 CI |
| --- | --- | --- | --- |
| logistic_regression | glare | 60 | 0.2445-0.4132 |
| logistic_regression | backlight | 55 | 0.5385-0.7052 |
| logistic_regression | transition | 77 | 0.2886-0.4403 |
| linear_svm | glare | 60 | 0.1245-0.3210 |
| linear_svm | backlight | 55 | 0.4602-0.6480 |
| linear_svm | transition | 77 | 0.2527-0.4190 |
| sgd_linear_svm | glare | 60 | 0.1512-0.2657 |
| sgd_linear_svm | backlight | 55 | 0.5683-0.7731 |
| sgd_linear_svm | transition | 77 | 0.0538-0.1993 |
| gaussian_nb | glare | 60 | 0.2194-0.3842 |
| gaussian_nb | backlight | 55 | 0.4231-0.6416 |
| gaussian_nb | transition | 77 | 0.2760-0.4326 |
| decision_tree | glare | 60 | 0.2537-0.4211 |
| decision_tree | backlight | 55 | 0.4143-0.6181 |
| decision_tree | transition | 77 | 0.3390-0.5000 |
| random_forest_50 | glare | 60 | 0.4231-0.6333 |
| random_forest_50 | backlight | 55 | 0.5535-0.7465 |
| random_forest_50 | transition | 77 | 0.4257-0.6089 |
| random_forest_100 | glare | 60 | 0.4471-0.6522 |
| random_forest_100 | backlight | 55 | 0.5929-0.7705 |
| random_forest_100 | transition | 77 | 0.4374-0.6069 |
| random_forest_200_current_config | glare | 60 | 0.4290-0.6313 |
| random_forest_200_current_config | backlight | 55 | 0.5973-0.7784 |
| random_forest_200_current_config | transition | 77 | 0.4224-0.6095 |
| random_forest_depth_8 | glare | 60 | 0.3622-0.5479 |
| random_forest_depth_8 | backlight | 55 | 0.5201-0.6974 |
| random_forest_depth_8 | transition | 77 | 0.3306-0.5029 |
| random_forest_depth_12 | glare | 60 | 0.4266-0.6438 |
| random_forest_depth_12 | backlight | 55 | 0.5895-0.7653 |
| random_forest_depth_12 | transition | 77 | 0.4040-0.5618 |
| random_forest_depth_none | glare | 60 | 0.4097-0.6302 |
| random_forest_depth_none | backlight | 55 | 0.5763-0.7546 |
| random_forest_depth_none | transition | 77 | 0.4435-0.6230 |
| extra_trees | glare | 60 | 0.4240-0.6227 |
| extra_trees | backlight | 55 | 0.5304-0.7101 |
| extra_trees | transition | 77 | 0.4316-0.5767 |
| gradient_boosting | glare | 60 | 0.3656-0.5970 |
| gradient_boosting | backlight | 55 | 0.4805-0.6945 |
| gradient_boosting | transition | 77 | 0.3455-0.5607 |
| hist_gradient_boosting | glare | 60 | 0.3602-0.5993 |
| hist_gradient_boosting | backlight | 55 | 0.5891-0.7752 |
| hist_gradient_boosting | transition | 77 | 0.4163-0.6228 |
| knn | glare | 60 | 0.3883-0.6076 |
| knn | backlight | 55 | 0.4978-0.7162 |
| knn | transition | 77 | 0.4024-0.6101 |
| mlp_small_32 | glare | 60 | 0.2322-0.4612 |
| mlp_small_32 | backlight | 55 | 0.5360-0.7231 |
| mlp_small_32 | transition | 77 | 0.3030-0.4869 |
| mlp_small_64_32 | glare | 60 | 0.2173-0.4559 |
| mlp_small_64_32 | backlight | 55 | 0.5620-0.7449 |
| mlp_small_64_32 | transition | 77 | 0.3342-0.5200 |
