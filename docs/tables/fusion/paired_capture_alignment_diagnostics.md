# Paired Capture Alignment Diagnostics

| diagnostic | status | n | mean | measured_p95 | estimated_p95 | p95_source | result | pairing_tier | captured_runtime_fusion_available | alignment_metadata | caveat | thesis_wording |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frame_skew_ms | measured | 584 | 32.597942 | 48.473850 |  | measured_from_timestamps_csv | mean=32.597942; p95=48.473850 | frame_strict:584 | false | not_measured | Frame skew supports capture pairing only; it does not prove spatial alignment quality. | strong |
| resolution_mismatch | measured | 2 |  |  |  |  | 640x480 vs 640x496 | frame_strict:584 | false | not_measured | Resolution mismatch requires explicit resize/warp policy; do not infer spatial alignment quality. | caveated |
| homography_alignment_metadata | not_measured | 0 |  |  |  |  | not measured | frame_strict:584 | false | not_measured | Do not claim alignment quality without homography/alignment metadata and visual verification. | not measured |
