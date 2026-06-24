# Raw Sensor Feature Drift

Compared against Phase 4 duplicate-cluster-aware train rows. Sensor rows are unlabeled.

| Feature | Train mean | Sensor mean | KS | Wasserstein | PSI | Out-of-range |
| --- | --- | --- | --- | --- | --- | --- |
| nir_mean_brightness | 85.9964 | 98.9869 | 0.3641 | 36.8668 | 4.5890 | 0.0017 |
| nir_std | 44.3904 | 53.2908 | 0.3006 | 9.3183 | 0.5249 | 0.0559 |
| nir_entropy | 6.3521 | 7.1335 | 0.2896 | 0.8037 | 1.7099 | 0.0119 |
| nir_p95 | 161.5165 | 204.8157 | 0.3065 | 44.3433 | 2.8572 | 0.0407 |
| nir_glare_score | 0.0243 | 0.0231 | 0.3145 | 0.0143 | 0.7174 | 0.0102 |
| nir_sharpness | 1494.5415 | 1652.7335 | 0.2403 | 494.2834 | 0.4382 | 0.0763 |
| nir_dark_fraction | 0.3193 | 0.0703 | 0.3397 | 0.2492 | 3.7287 | 0.0000 |
| nir_saturation_mean | 72.9108 | 46.7596 | 0.3703 | 38.1263 | 3.3377 | 0.0000 |
| hour_of_day_sin | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hour_of_day_cos | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| prev_env_class | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| nir_blue_mean_ema | 81.6195 | 93.8140 | 0.3821 | 38.2849 | 5.5085 | 0.0000 |
