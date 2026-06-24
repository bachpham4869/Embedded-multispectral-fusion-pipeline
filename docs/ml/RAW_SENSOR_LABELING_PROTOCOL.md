# Raw Sensor Labeling Protocol

Status: manual review package. No manual labels are inferred by the tool.

Fill the `manual_label` column in the CSV. If you agree with a pre-filled `suggested_label`, set `accept_suggested_label=yes`; otherwise leave it blank or write the corrected `manual_label`. Suggestions are not ground truth until a human confirms them.

Allowed labels: `night_clear`, `normal_night`, `normal_day`, `fog`, `rain`, `glare`, `backlight`, `nir_night`. Use `transition` only for true dawn/dusk transient scenes.

| Item | Value |
| --- | --- |
| candidate_frame_count | 120 |
| template | artifacts/ml/sensor_domain_shift/manual_label_template_autofilled.csv |
| review_dir | artifacts/ml/sensor_domain_shift/manual_label_candidates |
| metric_status | manual labels pending; sensor accuracy not measured |
