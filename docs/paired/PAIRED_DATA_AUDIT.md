# Paired Data Audit

- Input directory: `data/paired_data`
- Timestamp CSV: `data/paired_data/timestamps.csv`
- Total manifest rows: `584`
- Strict paired rows: `584`
- Time strict <=100 ms rows: `584`
- Thermal modality: `display_heatmap_like` unless raw numeric thermal arrays are listed in the inventory.
- Caveat: `thermal_paired.mp4` is not raw radiometric thermal when `thermal_modality=display_heatmap_like`.
- CSV manifest: `artifacts/paired_eval/strict_paired_manifest.csv`
- JSONL manifest: `artifacts/paired_eval/paired_data_manifest.jsonl`
