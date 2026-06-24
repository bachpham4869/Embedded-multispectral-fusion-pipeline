# Paired NIR Prediction Summary

Status: unlabeled / preliminary. Inference scope: `RGB-scaler proxy inference, not validated NIR classifier accuracy`. No sensor accuracy is claimed.

| Metric | Value |
| --- | --- |
| row_count | 584 |
| tau1 | 0.62 |
| accepted_count | 579 |
| abstention_rate | 0.0086 |
| mean_confidence | 0.9479 |
| mean_entropy | 0.3084 |
| inference_scope | RGB-scaler proxy inference, not validated NIR classifier accuracy |

## Top-1 Distribution

| Label | Count |
| --- | --- |
| normal_night | 579 |
| transition | 5 |

## By pairing_tier

| pairing_tier | Rows | Accepted | Abstention | Top-1 distribution |
| --- | --- | --- | --- | --- |
| frame_strict | 584 | 579 | 0.0086 | {'normal_night': 579, 'transition': 5} |

## By session_id

| session_id | Rows | Accepted | Abstention | Top-1 distribution |
| --- | --- | --- | --- | --- |
| paired_data | 584 | 579 | 0.0086 | {'normal_night': 579, 'transition': 5} |

## By nir_modality

| nir_modality | Rows | Accepted | Abstention | Top-1 distribution |
| --- | --- | --- | --- | --- |
| unknown_optical | 584 | 579 | 0.0086 | {'normal_night': 579, 'transition': 5} |

## By thermal_modality

| thermal_modality | Rows | Accepted | Abstention | Top-1 distribution |
| --- | --- | --- | --- | --- |
| display_heatmap_like | 584 | 579 | 0.0086 | {'normal_night': 579, 'transition': 5} |

## By label_source

| label_source | Rows | Accepted | Abstention | Top-1 distribution |
| --- | --- | --- | --- | --- |
| none | 584 | 579 | 0.0086 | {'normal_night': 579, 'transition': 5} |
