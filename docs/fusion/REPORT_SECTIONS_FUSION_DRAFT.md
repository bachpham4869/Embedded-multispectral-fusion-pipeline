# Report Sections Fusion Draft

## 1. Image Processing And Fusion Evaluation Protocol
Use the paired manifest from `timestamps.csv`; classify metric tiers and evidence source for every row.

## 2. Paired Data And Pairing Quality
Report `584` frame-strict pairs and frame-skew p95 from the capture diagnostics table.

## 3. NIR Enhancement Metrics
Use no-reference Tier 2 IQA metrics with caveats; do not call forced offline buckets runtime selections.

## 4. Thermal Display/Heatmap Evaluation
State that thermal is display/heatmap-like and not raw radiometric.

## 5. Fusion Algorithm Comparison
Discuss foreground-mask overlay versus alpha blend using caveated generated-fusion evidence.

## 6. Per-Bucket Evidence
Use `per_bucket_report_summary.md`; mark rain/dawn as not measured unless metadata is added.

## 7. Failure Cases
Use failure mining as diagnostic limitation evidence only.

## 8. Timing/Cadence Evidence
Separate frame skew/capture cadence from stage-processing latency.

## 9. Limitations And Future Validation
Require captured runtime fusion output, raw numeric thermal arrays, homography/alignment metadata, and human/user labels for stronger claims.
