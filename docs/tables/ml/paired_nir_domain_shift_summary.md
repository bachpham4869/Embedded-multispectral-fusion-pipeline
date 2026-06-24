# Paired NIR Domain Shift Summary

Status: unlabeled / preliminary. Inference scope: `RGB-scaler proxy inference, not validated NIR classifier accuracy`. No sensor-real accuracy is claimed.

| Metric | Value |
| --- | --- |
| sensor_rows | 584 |
| abstention_rate_tau1_0.62 | 0.0086 |
| mean_confidence | 0.9479 |
| mean_entropy | 0.3084 |
| inference_scope | RGB-scaler proxy inference, not validated NIR classifier accuracy |

## Highest Drift Features

| Feature | KS | Wasserstein | PSI | Out-of-range |
| --- | --- | --- | --- | --- |
| nir_sharpness | 0.6804 | 1313.5113 | 8.3783 | 0.6832 |
| nir_entropy | 0.6606 | 1.5915 | 4.7640 | 0.0137 |
| nir_p95 | 0.6527 | 95.5178 | 4.6280 | 0.0000 |
| nir_blue_mean_ema | 0.6508 | 61.4443 | 8.4711 | 0.0000 |
| nir_mean_brightness | 0.6462 | 62.0975 | 8.3864 | 0.0000 |
| nir_dark_fraction | 0.6404 | 0.4216 | 6.3991 | 0.0942 |
| nir_std | 0.5993 | 24.7162 | 2.9508 | 0.1592 |
| nir_glare_score | 0.4220 | 0.0216 | 2.2487 | 0.0000 |
| nir_saturation_mean | 0.3565 | 36.9921 | 7.0082 | 0.0000 |
| hour_of_day_sin | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hour_of_day_cos | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| prev_env_class | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

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
