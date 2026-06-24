# Feature Candidate Table Build Summary

Status: non-production feature tables. Production schema and loader are unchanged.

| Split | Input rows | Output rows | Skipped missing image | Labels |
| --- | --- | --- | --- | --- |
| train | 11981 | 4923 | 7058 | {"fog": 1339, "normal_day": 2077, "rain": 1104, "transition": 403} |
| test | 2113 | 868 | 1245 | {"fog": 203, "normal_day": 366, "rain": 187, "transition": 112} |

| Sensor candidate rows | Value |
| --- | --- |
| raw_sensor_features_v2 | 590 |

`temporal_brightness_std` is emitted only when sequential frame history exists. Still-image training rows do not receive zero-imputed temporal values.
`optical_21_candidate_still` excludes temporal-only fields and is emitted for verified still-image rows as a 21-derived still-compatible feature set.
