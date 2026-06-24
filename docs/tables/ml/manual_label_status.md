# Manual Label Status

Runtime validation selects the newest completed CSV with valid taxonomy labels. If no completed labels exist, raw sensor accuracy is not measured.

| Field | Value |
| --- | --- |
| status | no_completed_labels |
| selected_label_path | none |
| candidate_count | 3 |
| row_count | 0 |
| filled_labels | 0 |
| manual_label_count | 0 |
| accepted_suggested_labels | 0 |
| missing_labels | 0 |
| low_confidence_count | 0 |
| metric_status | not measured |

## Class Distribution

| Class | Count |
| --- | --- |
|  |  |

## Candidate Files

| Path | Status | Rows | Filled | Missing | Invalid labels |
| --- | --- | --- | --- | --- | --- |
| artifacts/ml/sensor_domain_shift/manual_label_template_autofilled.csv | template_no_completed_labels | 120 | 0 | 120 |  |
| artifacts/ml/sensor_domain_shift/suggested_labels.csv | invalid | 590 | 0 | 590 |  |
| artifacts/ml/sensor_domain_shift/manual_label_template.csv | template_no_completed_labels | 120 | 0 | 120 |  |
