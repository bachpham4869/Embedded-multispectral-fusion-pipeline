# Strict Paired Fusion Capture Protocol

This protocol remains required to unlock stronger captured runtime fusion-quality evidence.

- Current paired input evidence: `584` `frame_strict` NIR/thermal rows.
- captured_runtime_fusion_available=false.
- Current fusion comparisons are `paired_generated_fusion`, not runtime-captured fusion validation.
- Current thermal modality is `display_heatmap_like`, not raw radiometric thermal.

## Required Captures

- NIR raw and NIR enhanced.
- Thermal raw numeric/radiometric arrays where available.
- Thermal display/heatmap and foreground mask.
- Runtime fusion output captured from the production display path.
- Homography/alignment metadata and stage timing fields.

## Strict Pair Rule

- Prefer same-frame or same-cycle capture.
- Otherwise require same session/source and timestamp skew `<= 1s`.
- Thesis should prioritize `frame_strict` and `time_strict_100ms` when enough samples exist.

## Minimum Recommendation

- Capture at least 20 strict paired frames per major condition and at least 5 sessions where feasible.

See `docs/tables/fusion/required_capture_metadata.md` for sidecar fields.
