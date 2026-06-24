# Paired NIR Feature Drift

Compared against Phase 4 duplicate-cluster-aware train rows. Sensor rows are unlabeled unless trusted labels are explicitly documented.

| Feature | Train mean | Sensor mean | KS | Wasserstein | PSI | Out-of-range |
| --- | --- | --- | --- | --- | --- | --- |
| nir_mean_brightness | 85.9964 | 27.5120 | 0.6462 | 62.0975 | 8.3864 | 0.0000 |
| nir_std | 44.3904 | 19.6813 | 0.5993 | 24.7162 | 2.9508 | 0.1592 |
| nir_entropy | 6.3521 | 4.8590 | 0.6606 | 1.5915 | 4.7640 | 0.0137 |
| nir_p95 | 161.5165 | 67.5805 | 0.6527 | 95.5178 | 4.6280 | 0.0000 |
| nir_glare_score | 0.0243 | 0.0028 | 0.4220 | 0.0216 | 2.2487 | 0.0000 |
| nir_sharpness | 1494.5415 | 181.0312 | 0.6804 | 1313.5113 | 8.3783 | 0.6832 |
| nir_dark_fraction | 0.3193 | 0.7408 | 0.6404 | 0.4216 | 6.3991 | 0.0942 |
| nir_saturation_mean | 72.9108 | 57.3077 | 0.3565 | 36.9921 | 7.0082 | 0.0000 |
| hour_of_day_sin | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hour_of_day_cos | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| prev_env_class | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| nir_blue_mean_ema | 81.6195 | 23.0503 | 0.6508 | 61.4443 | 8.4711 | 0.0000 |
