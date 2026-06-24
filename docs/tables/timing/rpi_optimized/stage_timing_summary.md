# Stage Timing Summary

Generated from 10 session file(s) in `fusion_captures/metrics_rpi_optimized`.

| stage | n_frames | mean_ms | p50_ms | p95_ms | p99_ms |
|-------|----------|---------|--------|--------|--------|
| blend | 10 | 0.607 | 0.058 | 2.219 | 2.331 |
| chrome | 10 | 9.155 | 9.268 | 9.858 | 9.94 |
| display | 10 | 1.298 | 1.285 | 1.391 | 1.408 |
| framecache | 10 | 6.623 | 6.46 | 7.478 | 7.562 |
| fuse_blend_math | 6 | 12.452 | 12.514 | 12.989 | 13.019 |
| fuse_blur_fg | 6 | 1.035 | 1.097 | 1.249 | 1.263 |
| fuse_colormap | 7 | 1.525 | 1.483 | 1.682 | 1.727 |
| fuse_nir_resize | 7 | 0.645 | 0.648 | 0.763 | 0.781 |
| fuse_thermal_prep | 7 | 0.509 | 0.499 | 0.675 | 0.707 |
| fuse_warp_perspective | 6 | 4.507 | 4.472 | 4.924 | 5.004 |
| fuse_warp_prep | 6 | 2.361 | 2.381 | 2.61 | 2.625 |
| fusion_composite | 7 | 17.882 | 17.752 | 24.996 | 25.617 |
| hud | 10 | 1.987 | 2.019 | 2.179 | 2.194 |
| jerk | 10 | 0.906 | 0.887 | 1.006 | 1.012 |
| nir_bucket | 10 | 15.866 | 14.33 | 20.936 | 21.01 |
| thermal_proc | 7 | 1.485 | 1.442 | 1.67 | 1.704 |
