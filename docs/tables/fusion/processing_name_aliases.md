# Processing Name Aliases

| report_name | category | code_or_artifact_alias | rename_policy |
| --- | --- | --- | --- |
| night_hybrid_enhance | nir_processing_bucket | HybridNIREnhancer / bucket A / dark-scene hybrid enhancement | report alias only; no production symbol rename |
| nir_mono_clahe | nir_processing_bucket | NIR grayscale CLAHE / bucket B | report alias only; no production symbol rename |
| highlight_tone_map | nir_processing_bucket | anti-glare tone map / bucket C | report alias only; no production symbol rename |
| fog_dehaze_lite | nir_processing_bucket | dehaze-lite / bucket D | report alias only; no production symbol rename |
| rain_temporal_median | nir_processing_bucket | temporal median rain denoise / bucket E | report alias only; no production symbol rename |
| dawn_dusk_blend | nir_processing_bucket | dawn/dusk blend / bucket F | report alias only; no production symbol rename |
| nir_only_baseline | fusion_mode | NIR-only comparison frame | report alias only; no production symbol rename |
| thermal_heatmap_only | fusion_mode | thermal heatmap-only comparison frame | report alias only; no production symbol rename |
| alpha_blend_baseline | fusion_mode | simple weighted alpha blend | report alias only; no production symbol rename |
| foreground_mask_overlay | fusion_mode | current foreground-mask thermal overlay | report alias only; no production symbol rename |
| mask_weighted_blend | fusion_mode | offline mask-weighted blend baseline | report alias only; no production symbol rename |
| legacy_gradient_overlay | fusion_mode | legacy/pre-optimization gradient overlay | report alias only; no production symbol rename |
| laplacian_pyramid_fusion | fusion_mode | offline Laplacian pyramid fusion baseline | report alias only; no production symbol rename |
| thermal_raw | thermal_stage | MI48/raw thermal frame | report alias only; no production symbol rename |
| thermal_denoised_3dnr | thermal_stage | thermal temporal 3DNR output | report alias only; no production symbol rename |
| thermal_agc | thermal_stage | thermal automatic gain control output | report alias only; no production symbol rename |
| thermal_heatmap | thermal_stage | false-color thermal heatmap | report alias only; no production symbol rename |
| thermal_foreground_mask | thermal_stage | thermal foreground mask | report alias only; no production symbol rename |
