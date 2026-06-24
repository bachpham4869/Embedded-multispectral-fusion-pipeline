# Current vs Legacy Algorithms

| taxonomy_name | legacy_or_current | input | output | latency_risk |
| --- | --- | --- | --- | --- |
| night_hybrid_enhance | current | NIR BGR | enhanced NIR | high |
| nir_mono_clahe | current | NIR BGR | CLAHE NIR | low |
| highlight_tone_map | current | glare/highlight NIR | compressed highlights | low |
| fog_dehaze_lite | current | fog NIR/RGB proxy | dehazed-lite frame | medium |
| rain_temporal_median | current | short sequence | temporal median | medium |
| dawn_dusk_blend | current | transition NIR | A/C blend | medium |
| legacy_gradient_overlay | legacy/pre-optimization | NIR + heat map | gradient thermal overlay | medium |
