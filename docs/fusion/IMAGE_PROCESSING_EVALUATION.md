# Image Processing Evaluation

The paired-data image-processing evaluation uses `584` frame-strict NIR/thermal rows.

- NIR processing metrics are Tier 2 no-reference IQA unless a task-specific paired target is available.
- Forced offline processing buckets are reported as `bucket_evidence_status=forced_offline_algorithm`, not runtime bucket performance.
- Thermal metrics apply to `thermal_modality=display_heatmap_like`; raw radiometric thermal quality is not measured.
- `rain_temporal_median` remains not measured because no explicit rain/wet sequence label is present.
- `dawn_dusk_blend` remains not measured unless future metadata or confident human labels support dawn/dusk.
