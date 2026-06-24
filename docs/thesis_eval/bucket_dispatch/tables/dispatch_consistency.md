# Dispatch Consistency — Rule vs Manifest

**Source:** `data/eval/nir_val/manifest_v2.csv`  
**n:** 270 images, 0 skipped  
**Overall agreement:** 34/270 = 12.6%  

Rule signals: `nir_b_ema` (`_nir_mean_brightness_bgr`), `nir_gray_std` (`_nir_gray_std_quick`), `glare_nir` (`nir_highlight_need_compress`).
Same code path as runtime — `haze_config_on=False` for offline eval.

**Important:** Disagreements are evidence of rule-layer fragility, not a bug to fix.
The ML compositor (`compose_env_from_ml_top2`) exists precisely to cover this gap.

## Confusion Matrix (rows=manifest label, cols=rule-inferred label)

| Manifest \ Rule | backlight | fog | glare | nir_night | normal_day | normal_night | agree% |
|----------------|-----------|-----|-------|-----------|------------|--------------|--------|
| backlight | 0 | 0 | 23 | 0 | 0 | 2 | 0% |
| fog | 0 | 2 | 13 | 0 | 13 | 2 | 7% |
| glare | 6 | 1 | 23 | 0 | 0 | 0 | 77% |
| mixed_edge | 0 | 2 | 19 | 2 | 0 | 2 | 0% |
| night_clear | 0 | 22 | 7 | 0 | 0 | 11 | 0% |
| nir_night | 2 | 0 | 20 | 0 | 8 | 10 | 0% |
| normal_day | 0 | 0 | 14 | 0 | 4 | 2 | 20% |
| normal_night | 4 | 1 | 20 | 8 | 2 | 5 | 12% |
| rain | 1 | 0 | 13 | 0 | 3 | 3 | 0% |

## Dispatch-fragile classes (agreement < 80%)

- **backlight**: 0/25 = 0.0% agreement
- **fog**: 2/30 = 6.7% agreement
- **glare**: 23/30 = 76.7% agreement
- **mixed_edge**: 0/25 = 0.0% agreement
- **night_clear**: 0/40 = 0.0% agreement
- **nir_night**: 0/40 = 0.0% agreement
- **normal_day**: 4/20 = 20.0% agreement
- **normal_night**: 5/40 = 12.5% agreement
- **rain**: 0/20 = 0.0% agreement

> These classes are dispatch-fragile. Thesis §8: production dispatch quality depends on the ML compositor, not the rule layer alone.
