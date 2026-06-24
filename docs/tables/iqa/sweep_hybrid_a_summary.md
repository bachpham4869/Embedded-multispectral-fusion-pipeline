# Bucket A Parameter Sweep Summary

**Stopping rule:** max `mean_log_rms_after` s.t. `mean_pct_sat_after < 0.05`

| env_class | detail_strength | clahe_clip_scale | proc | log_rms_after | pct_sat_after | stopping_rule_pass |
|-----------|----------------|-----------------|------|--------------|--------------|--------------------|
| night_clear | 0.35 | 0.5 | 320×240 | 0.9200 | 0.0448 | YES |
| nir_night | — | — | — | — | — | NO (all fail sat limit) |
| normal_night | — | — | — | — | — | NO (all fail sat limit) |
