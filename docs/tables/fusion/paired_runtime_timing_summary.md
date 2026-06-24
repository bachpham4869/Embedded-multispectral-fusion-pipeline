# Paired Runtime Timing Summary

| source_or_session | metric | n | mean | measured_p95 | estimated_p95 | p95_source | nir_processing_latency_ms | thermal_processing_latency_ms | fusion_composite_latency_ms | caveat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paired_data | pair_interval_ms | 583 | 124.948452 | 127.453198 |  | measured_from_timestamps_csv | not_measured | not_measured | not_measured | Capture cadence evidence only; stage profiler fields are not present in timestamps.csv. |
| paired_data | fps | 583 | 8.006552 | 8.167573 |  | measured_from_timestamps_csv | not_measured | not_measured | not_measured | FPS is measured from paired capture timestamps; processing stage latency is not measured. |
| paired_data | abs_skew_ms | 584 | 32.597942 | 48.473850 |  | measured_from_timestamps_csv | not_measured | not_measured | not_measured | Pairing skew evidence; not per-stage processing latency. |
