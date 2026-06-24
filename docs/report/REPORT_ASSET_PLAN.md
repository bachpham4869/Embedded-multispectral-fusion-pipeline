# A4 Report Asset Plan

Status: A4 audit/preparation. Do not copy assets into `HK252-DATN-142/` until a later patch phase is approved.

## ML Tables

| Asset | Source artifact | Proposed LaTeX target | Caption draft | Label draft | Status | Size concern | Placement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Offline model comparison summary | `docs/tables/ml/model_comparison_cluster_aware.md` | `HK252-DATN-142/tables/ch6_evaluation/ml_classifier/ml_offline_model_comparison_table.tex` | Offline optical RGB-proxy model comparison on the duplicate-cluster-aware split. | `tab:ch6-ml-cluster-aware-models` | ready as prepared snippet | Compact summary only | Ch6 ML main |
| Model decision summary | `docs/tables/ml/model_decision_summary.md` | `HK252-DATN-142/tables/ch6_evaluation/ml_classifier/ml_model_decision_table.tex` | Model-selection decisions and migration gates. | `tab:ch6-ml-model-decisions` | ready as prepared snippet | Compact; avoid full prose columns if needed | Ch6 ML main |
| Class decision summary | `docs/tables/ml/class_decision_summary.md` | `HK252-DATN-142/tables/ch6_evaluation/ml_classifier/ml_class_decision_table.tex` | ENV class decisions and caveats after offline and sensor-proxy evidence. | `tab:ch6-ml-class-decisions` | ready as prepared snippet | May be wide; use `footnotesize` | Ch6 ML or appendix |
| Domain-shift summary | `docs/tables/ml/raw_sensor_domain_shift_summary.md`, `docs/tables/ml/paired_nir_domain_shift_summary.md` | `HK252-DATN-142/tables/ch6_evaluation/ml_classifier/ml_domain_shift_summary_table.tex` | Raw and paired sensor proxy inference summary. | `tab:ch6-ml-domain-shift-summary` | needs conversion | Compact two-row summary | Ch6 ML main |
| Agent-labeled diagnostic | `docs/tables/ml/agent_labeled_sensor_eval.md` | none for main report | Not for main result; optional limitation note only. | n/a | rejected_legacy_asset/not main | Avoid low-score table in main body | Ch7 limitation only |

## Fusion Tables

| Asset | Source artifact | Proposed LaTeX target | Caption draft | Label draft | Status | Size concern | Placement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fusion result summary | `docs/tables/fusion/fusion_result_summary_for_report.md` | `HK252-DATN-142/tables/ch6_evaluation/fusion/fusion_result_summary_table.tex` | Strict-paired generated fusion and image-processing results with evidence-tier caveats. | `tab:ch6-fusion-generated-summary` | ready as prepared snippet | Compact subset only | Ch6 fusion main |
| Per-bucket evidence table | `docs/tables/fusion/per_bucket_report_summary.md` | `HK252-DATN-142/tables/ch6_evaluation/fusion/per_bucket_evidence_table.tex` | Per-bucket evidence status for strict paired inputs. | `tab:ch6-fusion-bucket-evidence` | ready as prepared snippet | Use `footnotesize` | Ch6 fusion main |
| Alignment diagnostics | `docs/tables/fusion/paired_capture_alignment_diagnostics.md` | `HK252-DATN-142/tables/ch6_evaluation/fusion/alignment_diagnostics_table.tex` | Pairing skew and alignment evidence gates. | `tab:ch6-fusion-alignment-diagnostics` | needs conversion | Small | Ch6 fusion protocol |
| Failure/limitation summary | `docs/tables/fusion/fusion_failure_and_limitation_summary.md` | appendix or limitation table | Diagnostic failure categories from generated fusion. | `tab:ch7-fusion-limitations` | optional | Long; summarize only | Ch7 or appendix |

## Figures

| Asset | Source artifact | Proposed LaTeX target | Caption draft | Label draft | Status | Size concern | Placement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RF200 cluster-aware confusion matrix | `docs/figures/ml/cluster_aware/confusion_random_forest_200_current_config.png` | `HK252-DATN-142/figures/ch6_evaluation/ml_classifier/confusion_random_forest_200_cluster_aware.png` | Confusion matrix for the RF200 offline RGB-proxy baseline on the duplicate-cluster-aware split. | `fig:ch6-ml-rf200-cluster-confusion` | copied to LaTeX asset folder | Small PNG | Ch6 ML main |
| Raw sensor confidence histogram | `docs/figures/ml/raw_sensor_confidence_hist.png` | `HK252-DATN-142/figures/ch6_evaluation/ml_classifier/raw_sensor_confidence_hist.png` | Confidence distribution for unlabeled raw sensor proxy inference. | `fig:ch6-ml-raw-sensor-confidence` | copied to LaTeX asset folder | Small PNG | Ch6 ML caveated |
| Paired NIR feature PCA | `docs/figures/ml/paired_nir_feature_pca.png` | `HK252-DATN-142/figures/ch6_evaluation/ml_classifier/paired_nir_feature_pca.png` | Paired NIR feature-space drift visualization. | `fig:ch6-ml-paired-nir-pca` | copied to LaTeX asset folder | Check readability before use | Ch6 ML caveated |
| Strict paired fusion comparison grid | `docs/figures/fusion/strict_paired_fusion_comparison_grid.png` | `HK252-DATN-142/figures/ch6_evaluation/fusion/strict_paired_fusion_comparison_grid.png` | Generated fusion candidate comparison on strict paired inputs. | `fig:ch6-fusion-strict-paired-grid` | copied to LaTeX asset folder | Ensure not too large | Ch6 fusion main |
| Strict paired failure cases grid | `docs/figures/fusion/strict_paired_failure_cases_grid.png` | `HK252-DATN-142/figures/ch6_evaluation/fusion/strict_paired_failure_cases_grid.png` | Diagnostic failure examples from generated fusion candidates. | `fig:ch7-fusion-failure-grid` | copied to LaTeX asset folder | Use appendix/limitation only | Ch7 or appendix |

## Diagrams To Prepare Later

| Diagram | Source idea | Proposed target | Status | Placement |
| --- | --- | --- | --- | --- |
| ML evidence pipeline | A1 evidence flow: dataset -> leakage control -> duplicate-cluster split -> benchmark -> domain-shift audit | `figures/ch6_evaluation/ml_classifier/ml_evidence_pipeline.png` | needs drawing | Ch6 ML methodology |
| Fusion evidence pipeline | A2 flow: paired manifest -> NIR/thermal IQA -> generated fusion candidates -> metrics/failure mining | `figures/ch6_evaluation/fusion/fusion_evidence_pipeline.png` | needs drawing | Ch6 fusion methodology |
| Evidence tier diagram | Strong/caveated/preliminary/not measured mapping | `figures/ch6_evaluation/evidence_tier_diagram.png` | optional | Ch6 methodology |

## Advisory Legacy Asset Decisions

| Legacy asset | Decision | Reason |
| --- | --- | --- |
| `HK252-DATN-142/figures/ch6_evaluation/fusion/fusion_benefit_by_class.png` | rejected_legacy_asset | superseded by strict paired generated fusion evidence and not aligned with current positive-first caveats |
| `HK252-DATN-142/figures/ch6_evaluation/fusion/alpha_sweep_curve.png` | rejected_legacy_asset | stale/proxy relative to current generated-fusion summary; do not present as final optimum |
| `HK252-DATN-142/figures/ch6_evaluation/ml_classifier/reliability_night_classes.png` | optional legacy support | usable only if current ML section preserves calibration context; not required by A1 evidence |
| `HK252-DATN-142/figures/ch6_evaluation/ml_classifier/feature_importance.png` | optional legacy support | superseded in priority by cluster-aware model comparison and domain-shift evidence |
