# Image Fusion Metrics

Metric validity tiers govern thesis wording. Tier 1 direct paired/task-specific metrics can support strong claims; Tier 2 no-reference IQA and Tier 3 proxy/synthetic/unpaired metrics must be reported with limitations.

| metric | tier | meaning | limitation |
| --- | --- | --- | --- |
| foreground_contrast_gain | 1 | Task-specific target contrast gain on strict paired captures | Only strong if strict paired masks are valid |
| fps_mean/fps_p95/stage_latency | 1 | Real session runtime evidence | Session duration and thermal state still matter |
| entropy/rms/tenengrad/laplacian/clipping/noise | 2 | No-reference IQA statistics | Not absolute perceptual quality |
| MI/NMI/SSIM/Qabf-style edge proxy | 3 | Proxy fusion/preservation statistics | Not proof of real fusion quality |
