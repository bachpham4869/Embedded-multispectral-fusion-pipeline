# Fusion Evaluation

Paired input evidence now includes `584` `frame_strict` NIR/thermal rows. Fusion candidates are generated offline, so `captured_runtime_fusion_available=false` and `fusion_source=paired_generated_fusion` for fusion-mode comparisons.

- Foreground-mask overlay result: `n=584; mean=2.306456; delta=3.017641; win_rate=1.000000`.
- This supports a caveated algorithmic comparison against `alpha_blend_baseline` on paired inputs.
- It does not prove runtime-captured fusion behavior because no captured fusion output path is present.
- Entropy row: `n=584; mean=5.200249; delta=-0.822760; win_rate=0.006849`.
- Entropy decrease is ambiguous and is not interpreted as good or bad without visual or task-specific evidence.
- Thermal input is `display_heatmap_like`, not raw radiometric thermal.

See `docs/tables/fusion/fusion_result_summary_for_report.md` for thesis wording per claim.
