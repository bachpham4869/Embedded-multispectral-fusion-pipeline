# Fusion Algorithm Survey

| mode | implemented | dependency | rpi_suitability | evaluated |
| --- | --- | --- | --- | --- |
| nir_only_baseline | yes | OpenCV/NumPy | high | yes |
| thermal_heatmap_only | yes | OpenCV | high | yes |
| alpha_blend_baseline | yes | OpenCV | high | yes |
| foreground_mask_overlay | current | OpenCV | high | yes |
| mask_weighted_blend | offline baseline | OpenCV/NumPy | high | yes |
| legacy_gradient_overlay | legacy comparable | OpenCV | medium | yes |
| laplacian_pyramid_fusion | offline baseline | OpenCV/NumPy | medium | yes |
| DWT/guided/deep fusion | literature only | extra/heavy | low-medium | no |
