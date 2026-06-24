# Strict Paired Failure Case Summary

| failure_type | algorithm | pairing_tier | fusion_source | evidence_tier | n | caveat |
| --- | --- | --- | --- | --- | --- | --- |
| clipping_increase | foreground_mask_overlay | frame_strict | paired_generated_fusion | Tier 2 | 3 | Generated/proxy failures are diagnostic only, not proof of runtime fusion failure. |
| clipping_increase | legacy_gradient_overlay | frame_strict | paired_generated_fusion | Tier 2 | 3 | Generated/proxy failures are diagnostic only, not proof of runtime fusion failure. |
| clipping_increase | nir_only_baseline | frame_strict | paired_generated_fusion | Tier 2 | 6 | Generated/proxy failures are diagnostic only, not proof of runtime fusion failure. |
| contrast_drop | mask_weighted_blend | frame_strict | paired_generated_fusion | Tier 2 | 3 | Generated/proxy failures are diagnostic only, not proof of runtime fusion failure. |
| contrast_drop | nir_only_baseline | frame_strict | paired_generated_fusion | Tier 2 | 393 | Generated/proxy failures are diagnostic only, not proof of runtime fusion failure. |
| fusion_target_faded | alpha_blend_baseline | frame_strict | paired_generated_fusion | Tier 3 | 584 | Generated/proxy failures are diagnostic only, not proof of runtime fusion failure. |
| fusion_target_faded | laplacian_pyramid_fusion | frame_strict | paired_generated_fusion | Tier 3 | 584 | Generated/proxy failures are diagnostic only, not proof of runtime fusion failure. |
| fusion_target_faded | legacy_gradient_overlay | frame_strict | paired_generated_fusion | Tier 3 | 346 | Generated/proxy failures are diagnostic only, not proof of runtime fusion failure. |
| fusion_target_faded | mask_weighted_blend | frame_strict | paired_generated_fusion | Tier 3 | 584 | Generated/proxy failures are diagnostic only, not proof of runtime fusion failure. |
| fusion_target_faded | thermal_heatmap_only | frame_strict | paired_generated_fusion | Tier 3 | 584 | Generated/proxy failures are diagnostic only, not proof of runtime fusion failure. |
| noise_increase | foreground_mask_overlay | frame_strict | paired_generated_fusion | Tier 2 | 573 | Generated/proxy failures are diagnostic only, not proof of runtime fusion failure. |
| noise_increase | legacy_gradient_overlay | frame_strict | paired_generated_fusion | Tier 2 | 584 | Generated/proxy failures are diagnostic only, not proof of runtime fusion failure. |
| noise_increase | mask_weighted_blend | frame_strict | paired_generated_fusion | Tier 2 | 584 | Generated/proxy failures are diagnostic only, not proof of runtime fusion failure. |
| noise_increase | thermal_heatmap_only | frame_strict | paired_generated_fusion | Tier 2 | 584 | Generated/proxy failures are diagnostic only, not proof of runtime fusion failure. |
| sharpness_up_contrast_down | laplacian_pyramid_fusion | frame_strict | paired_generated_fusion | Tier 2 | 1 | Generated/proxy failures are diagnostic only, not proof of runtime fusion failure. |
| sharpness_up_contrast_down | mask_weighted_blend | frame_strict | paired_generated_fusion | Tier 2 | 11 | Generated/proxy failures are diagnostic only, not proof of runtime fusion failure. |
| sharpness_up_contrast_down | thermal_heatmap_only | frame_strict | paired_generated_fusion | Tier 2 | 1 | Generated/proxy failures are diagnostic only, not proof of runtime fusion failure. |
