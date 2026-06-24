# Paired Data Labeling Protocol

Use the paired NIR/thermal contact sheet and CSV template for manual review. Model outputs are suggestions only and are not ground truth.

Allowed labels: `night_clear`, `normal_night`, `normal_day`, `fog`, `rain`, `glare`, `backlight`, `nir_night`; use `transition` only for true dawn/dusk transient scenes.

| Item | Value |
| --- | --- |
| template | artifacts/paired_eval/manual_label_template_paired.csv |
| contact_sheet | docs/figures/ml/paired_manual_label_contact_sheet.png |
| metric_status | paired labeled eval not measured until trusted labels exist |
