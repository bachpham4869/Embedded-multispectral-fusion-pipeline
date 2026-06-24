# Processing Buckets

Status: report-facing processing policy documentation. Phase 1 does not rename
code symbols.

Current dispatch table: `src/smartbinocular/nir_pipeline.py::OPTICAL_BUCKET_DISPATCH`.

| Bucket | Report-facing policy name | Current ENV routes | Algorithm/policy | Why separate from ENV taxonomy | Evidence status |
| --- | --- | --- | --- | --- | --- |
| A | `night_hybrid_enhance` | `night_clear`, `normal_night`, `default` | HybridNIREnhancer: dark/bright channel weighting, adaptive CLAHE, detail sharpening, color correction | Algorithm for low-light enhancement, not an environment label | Existing bucket/IQA docs exist; still needs per-class report integration |
| B | `nir_mono_clahe` | `nir_night` | Lower-power single CLAHE for NIR-like mono night | Algorithm tuned for mono/NIR-like input | Proxy only until live NIR validation |
| C | `highlight_tone_map` | `glare`, `backlight`, `normal_day` | anti-glare tone map or passthrough when compression not needed | Highlight/HDR handling policy, shared by multiple ENV classes | Needs source-diversity review for glare/backlight |
| D | `fog_dehaze_lite` | `fog` | Dark-channel-prior-lite dehaze at reduced resolution | Visibility restoration policy | Needs feature/fusion ablation evidence |
| E | `rain_temporal_median` | `rain` | 3-frame temporal median for rain/noisy streak suppression | Temporal denoising policy | Requires video/live validation for real rain |
| F | `dawn_dusk_blend` | `transition` | Blend Bucket A and C by brightness/EMA | Runtime transition policy, not necessarily a stable ENV class | Weak evidence; keep as runtime state candidate |

Lite dispatch maps several classes to cheaper A/B substitutes when
`nir_optical_lite` is enabled. That is a deployment policy, not a class rename.

## Report Rule

Use ENV class names for scene condition, and bucket names for algorithm choice:

- Correct: "dawn/dusk samples route to `dawn_dusk_blend` Bucket F."
- Avoid: "Bucket F is a transition class."

## ML/Fusion Link

ML matters to fusion because ENV class selects the processing bucket, and bucket
output affects the NIR background before thermal foreground blending. Phase 1
does not change fusion, but future evaluation should record bucket share,
per-bucket IQA, and fusion policy effects by ENV class.
