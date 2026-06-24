# Required Capture Metadata

| field | required | description | strict_pair_requirement |
| --- | --- | --- | --- |
| session_id | yes | Stable capture session identifier shared by all modalities. | same session and <= 1s gap |
| frame_idx | yes | Monotonic frame index or processing-cycle index. | same session and <= 1s gap |
| timestamp_iso | yes | ISO timestamp from the same clock for NIR, thermal, and fusion outputs. | same session and <= 1s gap |
| mode | yes | Capture mode such as nir, thermal, or fusion. | must reference the same frame/cycle when present |
| env_class | yes | Environment class or scenario label if known. | must reference the same frame/cycle when present |
| processing_bucket | yes | Report-facing NIR bucket name. | must reference the same frame/cycle when present |
| fusion_mode | yes | Report-facing fusion mode name. | must reference the same frame/cycle when present |
| homography_path | yes | Homography/calibration file used for thermal-to-NIR alignment. | must reference the same frame/cycle when present |
| nir_raw_path | yes | Raw NIR frame path. | must reference the same frame/cycle when present |
| nir_enhanced_path | yes | Enhanced NIR output path. | must reference the same frame/cycle when present |
| thermal_raw_path | yes | Raw thermal frame or numeric array path. | must reference the same frame/cycle when present |
| thermal_heatmap_path | yes | Thermal heatmap path. | must reference the same frame/cycle when present |
| thermal_mask_path | yes | Foreground mask path. | must reference the same frame/cycle when present |
| fusion_output_path | yes | Final fused frame path. | must reference the same frame/cycle when present |
| stage_timings_ms | recommended | Per-stage timing dict for NIR, thermal, fusion, display, and logging when available. | must reference the same frame/cycle when present |
