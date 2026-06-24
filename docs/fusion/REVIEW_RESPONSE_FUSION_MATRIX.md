# Review Response Fusion Matrix

| reviewer_concern | response | evidence | status |
| --- | --- | --- | --- |
| Fusion method is too simple | Compared `foreground_mask_overlay` with alpha blend, mask-weighted, legacy gradient, and Laplacian pyramid generated baselines. | docs/tables/fusion/fusion_result_summary_for_report.md | caveated because fusion outputs are generated offline, not runtime-captured |
| Missing image/fusion metrics | Added tiered IQA/fusion metrics, statistical summaries, failure mining, and per-claim thesis wording. | docs/tables/fusion/strict_paired_*_summary.md | Tier 2/3 for quality; no-reference/proxy caveats retained |
| No paired thermal/NIR evidence | Added `584` frame-strict paired NIR/thermal rows from timestamps.csv. | artifacts/paired_eval/strict_paired_manifest.csv | strong for pairing/capture synchronization |
| Runtime fusion validation | No captured runtime fusion output is present in paired data. | docs/fusion/FUSION_EVIDENCE_READINESS.md | not measured; captured_runtime_fusion_available=false |
