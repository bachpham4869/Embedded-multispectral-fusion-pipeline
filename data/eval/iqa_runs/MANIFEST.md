# Offline NIR IQA — eval run index

Index of dated batch-IQA CSVs. Each row includes the git commit hash and manifest SHA-256 at run time.
Large raw per-image CSVs may be gitignored; aggregated summaries stay under `docs/tables/iqa/`.

| CSV | git | manifest_sha | note |
|-----|-----|-------------|------|
| docs/tables/iqa/raw/mis_dispatch_matrix.csv | f71e50f | manifest_v2 | Round 0: baseline 6×9 mis-dispatch matrix, pre-guard, manifest_v2 |
| round_2026-04-28.csv | f1a3c48 | 5f9aecb5e1ea | Round 1: preset tuning — A detail night_clear, B clip nir_night, C high_pct/sat glare+backlight, D omega fog |
