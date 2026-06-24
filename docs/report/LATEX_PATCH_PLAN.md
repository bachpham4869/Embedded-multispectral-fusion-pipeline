# A4 LaTeX Patch Plan

Status: plan only. Do not patch `.tex` in A4.

## Source Priority For Later Patch

- Current LaTeX source and new A1/A2 evidence are authoritative.
- Legacy guides/manifests are advisory only.
- Do not let old `THESIS_EVIDENCE_INDEX.md` entries force stale assets into the patched report.

## Planned File Edits

| File | Section/subsection | Change type | Evidence input | Draft source | Risk | Dependencies | Recommended timing |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `HK252-DATN-142/chapters/main/ch6-evaluation.tex` | `Machine Learning Classifier Evaluation` | replace outdated text and add model comparison table | `docs/tables/ml/model_comparison_cluster_aware.md`, `docs/tables/ml/model_decision_summary.md` | `docs/ml/REPORT_SECTIONS_ML_DRAFT.md` | Medium | Prepared `ml_offline_model_comparison_table.tex` | Patch now after A4 approval |
| `HK252-DATN-142/chapters/main/ch6-evaluation.tex` | `Feature Ablation and Offline Domain Shift` | replace/reshape subsection | `raw_sensor_*`, `paired_nir_*`, `feature_set_comparison_fair.md` | `docs/ml/REPORT_SECTIONS_ML_DRAFT.md` | Medium | Domain-shift caveats must stay attached | Patch now after A4 approval |
| `HK252-DATN-142/chapters/main/ch6-evaluation.tex` | `Fusion Optimization and Utility` | replace outdated fusion result text, add protocol/result table and figure | `docs/fusion/FUSION_EVIDENCE_READINESS.md`, `fusion_result_summary_for_report.md`, `paired_capture_alignment_diagnostics.md` | `docs/fusion/REPORT_SECTIONS_FUSION_DRAFT.md` | Medium | Copy strict paired fusion grid only if approved | Patch now after A4 approval |
| `HK252-DATN-142/chapters/main/ch5-implementation.tex` | `Multi-Sensor Image Fusion` | add caveat paragraph | `docs/fusion/FUSION_EVIDENCE_READINESS.md` | A4 claim map | Low | Ensure no duplicate Ch6 result prose | Patch with Ch6 fusion |
| `HK252-DATN-142/chapters/main/ch5-implementation.tex` | `Machine Learning Inference Pipeline` | update model/feature rationale | `model_decision_summary.md`, `feature_set_comparison_fair.md` | A4 prepared snippets | Medium | Keep code file names out of body | Patch with Ch6 ML |
| `HK252-DATN-142/chapters/main/ch7-conclusion.tex` | `Evaluation Scope Limitations` | replace stale proxy/no-domain-shift wording | `ML_EVIDENCE_READINESS.md`, `FUSION_EVIDENCE_READINESS.md` | A4 limitations table | Medium | Must distinguish proxy evidence from final validation | Patch after Ch6 |
| `HK252-DATN-142/chapters/main/ch7-conclusion.tex` | `Machine Learning Limitations` | add user-confirmed label gate and RPi4 latency gate | `ml_evidence_readiness_matrix.md` | A4 limitations table | Low | None | Patch after Ch6 |
| `HK252-DATN-142/chapters/main/ch7-conclusion.tex` | `Future Work` | add focused validation next steps | `paired_capture_alignment_diagnostics.md`, `RPI4_MODEL_LATENCY_PROTOCOL.md` | A4 limitations table | Low | No cloud service names unless scoped | Patch later |

## Required Later Patch Rules

- Introduce every table/figure in prose before the float.
- Keep all caveats in the same paragraph or caption as the metric.
- Use `development workstation` or `offline evaluation platform`; do not write brand-specific workstation names.
- Do not add `\cite{}` unless the key exists in `refs/example.bib`.
- Keep artifact paths out of report body where possible; use logical evidence descriptions.
- Do not include `agent_labeled_sensor_eval` low metrics in the main result section.

## Recommended Patch Order

1. Patch Ch6 ML first with duplicate-cluster-aware benchmark, RF200/RF100/MLP decisions, and domain-shift caveats.
2. Patch Ch6 fusion with strict paired input protocol, generated-fusion results, and not-measured gates.
3. Patch Ch5 only where implementation/rationale would otherwise conflict with the new Ch6 evidence.
4. Patch Ch7 limitations/future work to align with the updated Ch6 evidence.
5. Run static LaTeX checks and compile only if the toolchain is available.
