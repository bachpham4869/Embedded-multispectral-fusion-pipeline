# Raw Sensor Domain Shift Summary

Status: unlabeled / preliminary. Raw sensor inference is `RGB-scaler proxy inference`; no sensor-real accuracy is claimed.

| Metric | Value |
| --- | --- |
| sensor_rows | 590 |
| abstention_rate_tau1_0.62 | 0.5593 |
| mean_confidence | 0.6065 |
| mean_entropy | 1.4628 |
| inference_scope | RGB-scaler proxy inference |

## Highest Drift Features

| Feature | KS | Wasserstein | PSI | Out-of-range |
| --- | --- | --- | --- | --- |
| nir_blue_mean_ema | 0.3821 | 38.2849 | 5.5085 | 0.0000 |
| nir_saturation_mean | 0.3703 | 38.1263 | 3.3377 | 0.0000 |
| nir_mean_brightness | 0.3641 | 36.8668 | 4.5890 | 0.0017 |
| nir_dark_fraction | 0.3397 | 0.2492 | 3.7287 | 0.0000 |
| nir_glare_score | 0.3145 | 0.0143 | 0.7174 | 0.0102 |
| nir_p95 | 0.3065 | 44.3433 | 2.8572 | 0.0407 |
| nir_std | 0.3006 | 9.3183 | 0.5249 | 0.0559 |
| nir_entropy | 0.2896 | 0.8037 | 1.7099 | 0.0119 |
| nir_sharpness | 0.2403 | 494.2834 | 0.4382 | 0.0763 |
| hour_of_day_sin | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hour_of_day_cos | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| prev_env_class | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
