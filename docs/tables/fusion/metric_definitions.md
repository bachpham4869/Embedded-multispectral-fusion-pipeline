# Fusion Metric Definitions

Strong thesis claims are allowed only for Tier 1. Tier 2/3 must carry limitations.

| metric | tier | meaning | limitation |
| --- | --- | --- | --- |
| foreground_contrast_gain | 1 | Task-specific target contrast gain on strict paired captures | Only strong if strict paired masks are valid |
| fps_mean/fps_p95/stage_latency | 1 | Real session runtime evidence | Session duration and thermal state still matter |
| entropy/rms/tenengrad/laplacian/clipping/noise | 2 | No-reference IQA statistics | Not absolute perceptual quality |
| MI/NMI/SSIM/Qabf-style edge proxy | 3 | Proxy fusion/preservation statistics | Not proof of real fusion quality |
