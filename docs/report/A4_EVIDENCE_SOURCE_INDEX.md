# A4 Evidence Source Index

Status: A4 audit/preparation. New A1/A2 evidence is authoritative for this phase; legacy manifests are advisory.

| Source file | Evidence type | Status | Target report section | Recommended use |
| --- | --- | --- | --- | --- |
| `docs/ml/ML_EVIDENCE_READINESS.md` | ML evidence readiness summary | caveated | Ch6 ML, Ch7 limitations | Main framing and caveat source |
| `docs/tables/ml/ml_evidence_readiness_matrix.md` | ML evidence matrix | caveated/preliminary/not measured | Ch6 ML, Ch7 limitations | Main evidence index |
| `docs/ml/REPORT_SECTIONS_ML_DRAFT.md` | Draft report wording | caveated | Ch6 ML, Ch7 limitations | Draft text source, not direct proof |
| `docs/ml/MODEL_SELECTION_RATIONALE.md` | Model-selection rationale | caveated | Ch5 ML, Ch6 ML | Methodology and rationale |
| `docs/tables/ml/model_decision_summary.md` | Model decision table | caveated/preliminary | Ch6 ML | Main compact table |
| `docs/ml/CLASS_DECISION_RECORDS.md` | Class policy rationale | caveated/preliminary | Ch5 ML, Ch7 limitations | Methodology/limitation |
| `docs/tables/ml/class_decision_summary.md` | Class decision summary | caveated/preliminary | Ch6 ML or appendix | Compact table |
| `docs/ml/DOMAIN_SHIFT_EVALUATION.md` | Sensor-domain evaluation narrative | preliminary | Ch6 ML domain shift | Limitation and methodology |
| `docs/tables/ml/raw_sensor_domain_shift_summary.md` | Raw sensor drift metrics | preliminary | Ch6 ML domain shift | Main caveated domain-shift evidence, no accuracy |
| `docs/tables/ml/raw_sensor_prediction_summary.md` | Raw sensor prediction distribution | preliminary | Ch6 ML domain shift | Main caveated behavior evidence |
| `docs/tables/ml/paired_nir_domain_shift_summary.md` | Paired NIR drift metrics | preliminary | Ch6 ML domain shift | Main caveated paired sensor evidence |
| `docs/tables/ml/paired_nir_prediction_summary.md` | Paired NIR prediction distribution | preliminary | Ch6 ML domain shift | Main caveated behavior evidence |
| `docs/tables/ml/agent_labeled_sensor_eval.md` | Agent-labeled subset eval | preliminary | Ch7 limitations | Limitation/internal diagnostic only |
| `docs/tables/ml/feature_set_comparison_fair.md` | Feature v2/21 subset comparison | preliminary | Ch5 ML or Ch6 ML | Research-candidate note only |
| `docs/tables/ml/mlp_variant_comparison.md` | MLP family comparison | preliminary/caveated | Ch6 ML | Rationale for rejecting MLP |
| `docs/tables/ml/final_dataset_evidence_summary.md` | Dataset scope summary | caveated/preliminary | Ch6 methodology | Dataset inventory |
| `docs/fusion/FUSION_EVIDENCE_READINESS.md` | Fusion evidence readiness | strong/caveated/not measured | Ch6 fusion, Ch7 limitations | Main fusion evidence gate |
| `docs/tables/fusion/evidence_readiness_matrix.md` | Fusion evidence matrix | strong/caveated/preliminary/not measured | Ch6 fusion | Main evidence index |
| `docs/fusion/REPORT_SECTIONS_FUSION_DRAFT.md` | Fusion draft report sections | caveated | Ch6 fusion | Draft text source, not proof |
| `docs/tables/fusion/fusion_result_summary_for_report.md` | Fusion summary table | strong/caveated/preliminary/not measured | Ch6 fusion | Main compact fusion table |
| `docs/tables/fusion/per_bucket_report_summary.md` | Per-bucket evidence summary | caveated/not measured | Ch6 fusion, Ch7 limitations | Main caveated bucket evidence table |
| `docs/tables/fusion/fusion_failure_and_limitation_summary.md` | Failure mining | preliminary | Ch7 limitations or appendix | Limitation/diagnostic only |
| `docs/tables/fusion/paired_capture_alignment_diagnostics.md` | Capture skew and alignment diagnostics | strong/caveated/not measured | Ch6 fusion protocol | Main pairing quality and caveat source |
| `docs/fusion/FUSION_EVALUATION.md` | Fusion evaluation narrative | caveated/preliminary | Ch6 fusion | Methodology support |
| `docs/fusion/IMAGE_PROCESSING_EVALUATION.md` | Image processing evaluation narrative | caveated | Ch6 image/fusion | Methodology support |
| `docs/fusion/REVIEW_RESPONSE_FUSION_MATRIX.md` | Reviewer response mapping | caveated | Appendix if needed | Appendix/support only |
| `docs/paired/PAIRED_DATA_FUSION_EVALUATION_SUMMARY.md` | Paired fusion summary | strong/caveated/preliminary | Ch6 fusion | Main paired-data framing |
| `docs/paired/PAIRED_DATA_EVALUATION_SUMMARY.md` | Paired data summary | preliminary | Ch6 methodology | Use if present; otherwise note missing |
| `docs/tables/paired/paired_data_evidence_matrix.md` | Paired data matrix | missing | n/a | Not found in current checkout |
| `docs/ml/REPORT_PATCH_NOTES.md` | Patch notes | mixed | A4 patch plan | Planning only |
| `docs/ml/REVIEW_RESPONSE_MATRIX.md` | Reviewer response matrix | mixed | Appendix if needed | Planning/support only |

## Special Filtering Decisions

- `agent_labeled_sensor_eval` is not a main result because labels are not user-confirmed gold labels and the subset is small and biased toward review risk.
- Current paired fusion metrics can support generated-offline fusion comparison only. They cannot validate captured runtime fusion.
- Raw/paired sensor inference metrics can support domain-shift behavior only. They cannot support final sensor-real accuracy.
- Legacy figures/tables are not required. Include only if aligned with A1/A2 evidence and positive-first framing.
