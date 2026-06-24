# Raw Data Recoverability

Feature-vector/hash matches are not identity evidence. `verified` means path/filename/frame identity is replayable or file SHA is available.

| Source | Rows | Dataset | Raw root exists | Candidate images | Metadata status | Reason |
| --- | --- | --- | --- | --- | --- | --- |
| offline_backlight | 368 | backlight | True | 368 | inferred_low_confidence | raw images exist but original JSONL lacks path/hash; source used shuffled/subsampled order |
| offline_darkface | 2781 | darkface | True | 6000 | inferred_low_confidence | raw images exist but original JSONL lacks path/hash; source used shuffled/subsampled order |
| offline_exdark_street | 2347 | exdark_street | True | 2347 | inferred_low_confidence | raw images exist but original JSONL lacks path/hash; source used shuffled/subsampled order |
| offline_glare_street | 400 | glare_street | True | 400 | inferred_low_confidence | raw images exist but original JSONL lacks path/hash; source used shuffled/subsampled order |
| offline_gray_nir | 2407 | gray_nir | True | 7129 | inferred_low_confidence | raw images exist but original JSONL lacks path/hash; source used shuffled/subsampled order |
| offline_mwd | 1123 | mwd | True | 1123 | verified | source plus frame_idx can replay deterministic raw image order |
| offline_weather11 | 2068 | weather11 | True | 2068 | verified | source plus frame_idx can replay deterministic raw image order |
| offline_weather_time | 2600 | weather_time | True | 2600 | verified | source plus frame_idx can replay deterministic raw image order |
