# Data Recoverability Audit

Status: Phase 3 audit for the optical RGB-proxy baseline. This document does not claim live NIR/LWIR validation.

## JSONL Field Coverage

| JSONL | Rows | Trace fields present | Field count |
| --- | --- | --- | --- |
| data/training/from_logs_train.jsonl | 11981 | source, frame_idx, ts, label, label_source, label_confidence | 36 |
| data/training/from_logs_test.jsonl | 2113 | source, frame_idx, ts, label, label_source, label_confidence | 36 |
| data/training/merged_logs_ml.jsonl | 14094 | source, frame_idx, ts, label, label_source, label_confidence | 36 |
| logs/ml/offline_backlight.jsonl | 368 | source, frame_idx, ts, label, label_source, label_confidence | 36 |
| logs/ml/offline_darkface.jsonl | 2781 | source, frame_idx, ts, label, label_source, label_confidence | 36 |
| logs/ml/offline_exdark.jsonl | 2347 | source, frame_idx, ts, label, label_source, label_confidence | 36 |
| logs/ml/offline_glare.jsonl | 400 | source, frame_idx, ts, label, label_source, label_confidence | 36 |
| logs/ml/offline_gray_nir.jsonl | 2407 | source, frame_idx, ts, label, label_source, label_confidence | 36 |
| logs/ml/offline_mwd.jsonl | 1123 | source, frame_idx, ts, label, label_source, label_confidence | 36 |
| logs/ml/offline_weather11.jsonl | 2068 | source, frame_idx, ts, label, label_source, label_confidence | 36 |
| logs/ml/offline_weather_time.jsonl | 2600 | source, frame_idx, ts, label, label_source, label_confidence | 36 |

## Source-Level Recoverability

| Source | Rows | Status | Reason |
| --- | --- | --- | --- |
| offline_backlight | 368 | inferred_low_confidence | raw images exist but original JSONL lacks path/hash; source used shuffled/subsampled order |
| offline_darkface | 2781 | inferred_low_confidence | raw images exist but original JSONL lacks path/hash; source used shuffled/subsampled order |
| offline_exdark_street | 2347 | inferred_low_confidence | raw images exist but original JSONL lacks path/hash; source used shuffled/subsampled order |
| offline_glare_street | 400 | inferred_low_confidence | raw images exist but original JSONL lacks path/hash; source used shuffled/subsampled order |
| offline_gray_nir | 2407 | inferred_low_confidence | raw images exist but original JSONL lacks path/hash; source used shuffled/subsampled order |
| offline_mwd | 1123 | verified | source plus frame_idx can replay deterministic raw image order |
| offline_weather11 | 2068 | verified | source plus frame_idx can replay deterministic raw image order |
| offline_weather_time | 2600 | verified | source plus frame_idx can replay deterministic raw image order |

## Conclusion

- `verified` is limited to sources where the raw path identity can be replayed from source plus frame index.
- `inferred_low_confidence` sources keep candidate metadata separate from verified `file_sha256`/`dhash` fields.
- Feature-vector/hash matching remains only a weak hint and is never treated as original-image identity.
- Unresolved source count: 0.
- Inferred-low-confidence source count: 5.
