# Fusion Failure And Limitation Summary

| failure_type | algorithm | n | pairing_tier | fusion_source | captured_runtime_fusion_available | evidence_tier | severity_trigger | report_caveat | thesis_wording |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clipping_increase | foreground_mask_overlay | 3 | frame_strict | paired_generated_fusion | false | Tier 2 | clipping_increase | Generated/proxy failures are diagnostic only, not proof of runtime fusion failure. | preliminary |
| clipping_increase | legacy_gradient_overlay | 3 | frame_strict | paired_generated_fusion | false | Tier 2 | clipping_increase | Generated/proxy failures are diagnostic only, not proof of runtime fusion failure. | preliminary |
| clipping_increase | nir_only_baseline | 6 | frame_strict | paired_generated_fusion | false | Tier 2 | clipping_increase | Generated/proxy failures are diagnostic only, not proof of runtime fusion failure. | preliminary |
| contrast_drop | mask_weighted_blend | 3 | frame_strict | paired_generated_fusion | false | Tier 2 | contrast_drop | Generated/proxy failures are diagnostic only, not proof of runtime fusion failure. | preliminary |
| contrast_drop | nir_only_baseline | 393 | frame_strict | paired_generated_fusion | false | Tier 2 | contrast_drop | Generated/proxy failures are diagnostic only, not proof of runtime fusion failure. | preliminary |
| fusion_target_faded | alpha_blend_baseline | 584 | frame_strict | paired_generated_fusion | false | Tier 3 | fusion_target_faded | Generated/proxy failures are diagnostic only, not proof of runtime fusion failure. | preliminary |
| fusion_target_faded | laplacian_pyramid_fusion | 584 | frame_strict | paired_generated_fusion | false | Tier 3 | fusion_target_faded | Generated/proxy failures are diagnostic only, not proof of runtime fusion failure. | preliminary |
| fusion_target_faded | legacy_gradient_overlay | 346 | frame_strict | paired_generated_fusion | false | Tier 3 | fusion_target_faded | Generated/proxy failures are diagnostic only, not proof of runtime fusion failure. | preliminary |
| fusion_target_faded | mask_weighted_blend | 584 | frame_strict | paired_generated_fusion | false | Tier 3 | fusion_target_faded | Generated/proxy failures are diagnostic only, not proof of runtime fusion failure. | preliminary |
| fusion_target_faded | thermal_heatmap_only | 584 | frame_strict | paired_generated_fusion | false | Tier 3 | fusion_target_faded | Generated/proxy failures are diagnostic only, not proof of runtime fusion failure. | preliminary |
| noise_increase | foreground_mask_overlay | 573 | frame_strict | paired_generated_fusion | false | Tier 2 | noise_increase | Generated/proxy failures are diagnostic only, not proof of runtime fusion failure. | preliminary |
| noise_increase | legacy_gradient_overlay | 584 | frame_strict | paired_generated_fusion | false | Tier 2 | noise_increase | Generated/proxy failures are diagnostic only, not proof of runtime fusion failure. | preliminary |
| noise_increase | mask_weighted_blend | 584 | frame_strict | paired_generated_fusion | false | Tier 2 | noise_increase | Generated/proxy failures are diagnostic only, not proof of runtime fusion failure. | preliminary |
| noise_increase | thermal_heatmap_only | 584 | frame_strict | paired_generated_fusion | false | Tier 2 | noise_increase | Generated/proxy failures are diagnostic only, not proof of runtime fusion failure. | preliminary |
| sharpness_up_contrast_down | laplacian_pyramid_fusion | 1 | frame_strict | paired_generated_fusion | false | Tier 2 | sharpness_up_contrast_down | Generated/proxy failures are diagnostic only, not proof of runtime fusion failure. | preliminary |
| sharpness_up_contrast_down | mask_weighted_blend | 11 | frame_strict | paired_generated_fusion | false | Tier 2 | sharpness_up_contrast_down | Generated/proxy failures are diagnostic only, not proof of runtime fusion failure. | preliminary |
| sharpness_up_contrast_down | thermal_heatmap_only | 1 | frame_strict | paired_generated_fusion | false | Tier 2 | sharpness_up_contrast_down | Generated/proxy failures are diagnostic only, not proof of runtime fusion failure. | preliminary |
