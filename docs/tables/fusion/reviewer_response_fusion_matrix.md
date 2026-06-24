# Reviewer Response Fusion Matrix

| reviewer_question | response | evidence |
| --- | --- | --- |
| Fusion hiện tại là gì? | foreground_mask_overlay is the current mode; baselines include alpha_blend_baseline and legacy_gradient_overlay. | fusion_algorithm_comparison.md |
| Có so với baseline không? | Yes: raw/NIR-only, thermal_heatmap_only, alpha blend, mask-weighted, legacy gradient, Laplacian pyramid where available. | fusion_algorithm_comparison.md |
| Metric nào chứng minh cải thiện? | Only Tier 1 strict paired/task metrics support strong claims; Tier 2/3 are proxy/no-reference. | metric_definitions.md |
| Limitation là gì? | Proxy/unpaired/synthetic rows are not proof of real fusion quality and require future paired capture validation. | FUSION_EVALUATION.md |
