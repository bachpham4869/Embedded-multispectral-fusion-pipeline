# Review Response Matrix

Status: Phase 2 response map for GV feedback.

| GV feedback | Phase 1 action | Evidence/file | Status |
| --- | --- | --- | --- |
| Class/bucket names are unclear | Separate ENV condition aliases from processing bucket names | `docs/ml/ENV_TAXONOMY.md`, `docs/ml/PROCESSING_BUCKETS.md` | done, no code rename |
| Missing data distribution | Added script and generated train/test/reference tables | `tools/analyze_dataset_distribution.py`, `docs/tables/ml/dataset_distribution_*.md` | done |
| Class imbalance and unreliable classes | Added support/risk table and provisional decisions | `docs/tables/ml/class_support_risk_summary.md` | done |
| `transition` may not be reliable | Added dedicated decision record | `docs/ml/DECISION_transition_class.md` | done; evidence weak |
| 12 features are basic | Added feature engineering guardrails and v2 plan | `docs/ml/REFINE_PLAN.md` | planned |
| RF hyperparameters unclear | Documented RF settings and benchmark script | `docs/ml/REFINE_PLAN.md`, `tools/compare_classifiers.py` | done |
| Need compare RF with lighter models | Ran quick same-feature benchmark with CI, size, latency, confusion artifacts | `tools/compare_classifiers.py`, `docs/tables/ml/model_comparison_12features.md`, `docs/ml/MODEL_SELECTION_RATIONALE.md` | quick/preliminary |
| Need specialized real-world models | Added dataset/model survey with license/modality/label/mapping/risk/recommendation | `docs/ml/DATASET_CANDIDATE_SURVEY.md`, `docs/ml/MODEL_RESEARCH_SURVEY.md`, `docs/tables/ml/specialized_model_survey.md` | metadata-only; no downloads |
| Domain shift to sensor not handled | Added domain-shift wording and protocol | `docs/ml/DOMAIN_SHIFT_EVALUATION.md` | planned; live data missing |
| Fusion is simple | Documented ML-to-bucket/fusion dependency without widening scope | `docs/ml/PROCESSING_BUCKETS.md` | done |
| Need concrete metrics | Added reproducibility standard, leakage protocol, and preliminary generated artifacts | `docs/ml/DATA_SPLIT_AND_LEAKAGE_PROTOCOL.md`, `docs/ml/METRIC_DISCREPANCY_INVESTIGATION.md` | partially done |

## Phase 2 Additions

| Reviewer concern | Phase 2 evidence | Status |
| --- | --- | --- |
| Could train/test leakage inflate frozen metric? | `docs/tables/ml/leakage_check_summary.md`, `docs/tables/ml/feature_vector_overlap_details.md`, `artifacts/ml/leakage/feature_vector_overlaps.csv` | not closed; no exact row overlap, but image-level leakage unverified |
| Does removing overlap records explain the metric discrepancy? | `docs/tables/ml/frozen_eval_overlap_sensitivity.md`, `artifacts/ml/eval/frozen_eval_*.json` | no; metrics barely change |
| Is RF still reasonable versus lighter models? | `docs/tables/ml/model_comparison_12features.md`, `docs/ml/MODEL_SELECTION_RATIONALE.md` | preliminary; RF100/RF200 strongest among same-feature models |
| Are real-world datasets/models considered? | `docs/ml/DATASET_CANDIDATE_SURVEY.md`, `docs/tables/ml/dataset_candidate_summary.md`, `docs/tables/ml/specialized_model_survey.md` | yes, metadata-only |

## Direct Answer Language

- "We renamed nothing in code yet; we added report-facing aliases and a migration
  gate so old production model bundles remain compatible."
- "We measured dataset distribution first. `glare`, `backlight`, and
  `transition` are low-support classes and are now marked provisional."
- "The current classifier is an optical RGB-proxy baseline, not an NIR-trained
  classifier."
- "The high frozen held-out score is not used as the main thesis number until
  the discrepancy against TimeSeriesSplit CV is explained."
- "Removing the 21 feature-vector-overlap test rows does not explain the
  discrepancy, so held-out metrics remain preliminary."
- "The model comparison is a quick/proxy benchmark on current split, not final
  thesis evidence."
- "The next phase is to run leakage-clean/source-aware benchmark and live NIR
  domain-shift evaluation."

## Phase 3 Additions

| Reviewer concern | Phase 3 evidence | Status |
| --- | --- | --- |
| Can JSONL records be mapped back to raw images? | `docs/ml/DATA_RECOVERABILITY_AUDIT.md`, `docs/tables/ml/raw_data_recoverability.md` | partially; 5,791 verified rows, 8,303 inferred-low-confidence rows |
| Does current split contain image-level leakage? | `docs/tables/ml/image_level_leakage_current_split_summary.md` | yes within verified coverage: 22 exact file SHA overlaps and 1,419 dHash screening pairs |
| Did Phase 3 split reduce exact duplicate leakage? | `docs/tables/ml/image_level_leakage_summary.md`, `docs/tables/ml/group_aware_split_distribution.md` | yes for exact file SHA and group overlap; dHash screening pairs remain |
| Is the new split source-held-out? | `docs/tables/ml/group_aware_split_distribution.md` | no; it is group-aware, with 8 overlapping source names |
| Which model is strongest on the group-aware split? | `docs/tables/ml/model_comparison_group_aware.md` | preliminary; RF100 leads balanced accuracy, HGB leads accuracy/macro-F1 |

Direct answer update:

- "We can now check exact file SHA leakage for 5,791 verified records. The
  current split had 22 exact file SHA overlaps; the Phase 3 group-aware split
  reduced exact file SHA overlap to zero."
- "We still do not claim visual-duplicate-free data because dHash screening
  found near-duplicate candidates and only covers records with verified image
  metadata."

## Phase 4 Additions

| Reviewer concern | Phase 4 evidence | Status |
| --- | --- | --- |
| Were dHash near-duplicate blockers resolved? | `docs/tables/ml/cluster_aware_leakage_summary.md`, `docs/tables/ml/duplicate_cluster_summary_conservative.md` | yes within hash-covered records at dHash <= 4; not an absolute visual-duplicate claim |
| Was the cluster graph built from the full dataset? | `tools/build_duplicate_clusters.py`, `data/training_v3/duplicate_cluster_manifest_conservative.json` | yes; Phase 3 cross-split CSV is diagnostic only |
| Does the benchmark split keep class support? | `docs/tables/ml/cluster_aware_split_distribution_conservative.md` | yes; all ENV classes have train/test support |
| Which RF setting is preferred after leakage cleanup? | `docs/tables/ml/model_comparison_cluster_aware.md`, `docs/ml/MODEL_SELECTION_RATIONALE.md` | RF200 leads balanced accuracy; RF100 remains lighter candidate |
| Is the evidence source-held-out or live validated? | `docs/tables/ml/cluster_aware_remaining_risks.md` | no; source overlap remains and live NIR/LWIR validation is still missing |

Direct answer update:

- "We rebuilt duplicate clusters on the full metadata-enriched merged dataset,
  not just the earlier cross-split pair list."
- "Under the conservative dHash<=4 cluster-aware group split, cross-split exact
  file SHA, split-group, duplicate-cluster, and dHash screening pairs are all
  zero within available hash coverage."
- "The Phase 4 metric table can be used as offline optical RGB-proxy evidence
  with caveats. It is not source-held-out and not live NIR/LWIR validated."

## Fusion Evidence Cross-Link

| Reviewer concern | Fusion evidence | Status |
| --- | --- | --- |
| Fusion/image-processing evaluation was weak | `docs/fusion/REVIEW_RESPONSE_FUSION_MATRIX.md`, `docs/fusion/FUSION_EVIDENCE_READINESS.md` | addressed with proxy/no-reference caveats; strict paired capture still future work |
## Phase M5-M9 Reviewer Response Addendum

| Reviewer concern | Phase M5-M9 response | Evidence | Status |
| --- | --- | --- | --- |
| Domain shift on real sensor | Raw sensor video was inventoried, sampled, featurized, and compared to Phase 4 train/test. | `docs/tables/ml/raw_sensor_domain_shift_summary.md`, `artifacts/ml/sensor_domain_shift/domain_shift_metrics.json` | partial; unlabeled |
| Claiming NIR/live accuracy | No accuracy is claimed. All raw sensor predictions are marked `RGB-scaler proxy inference`. | `docs/tables/ml/raw_sensor_prediction_summary.md` | done |
| Manual sensor validation | Created 120-frame manual label package and protocol. | `docs/ml/RAW_SENSOR_LABELING_PROTOCOL.md`, `artifacts/ml/sensor_domain_shift/manual_label_template.csv` | requires user labels |
| RF vs lighter model | MLP family and RF100/RF200 were compared over 3 seeds. | `docs/tables/ml/mlp_variant_comparison.md` | done offline |
| Feature weakness | Implemented non-production v2/21 candidate feature comparison. | `docs/tables/ml/feature_set_12_vs_v2_vs_21.md`, `docs/ml/FEATURE_SET_RATIONALE.md` | partial; subset-only |
| Weak classes | Added class decision records for `transition`, `glare`, `backlight`, and `nir_night` caveats. | `docs/ml/CLASS_DECISION_RECORDS.md` | done as documentation decision |

## Phase M10 Reviewer Response Addendum

| Reviewer concern | Phase M10 response | Evidence | Status |
| --- | --- | --- | --- |
| Sensor-domain accuracy | Checked for completed manual labels at runtime; none found, so sensor accuracy is not measured. | `docs/tables/ml/manual_label_status.md`, `docs/tables/ml/raw_sensor_labeled_eval.md` | requires user labels |
| Manual-label workflow | Refreshed timestamp-correct labeling template and contact sheet. | `docs/ml/RAW_SENSOR_LABELING_PROTOCOL.md`, `docs/figures/ml/raw_sensor_labeling_contact_sheet.png` | ready for user review |
| Feature set 21 ambiguity | Split still-compatible and temporal candidates; still candidate has 20 actual supervised features. | `docs/tables/ml/feature_21_definition.md`, `docs/ml/FEATURE_SET_RATIONALE.md` | addressed |
| Fair feature comparison | Compared 12/v2/21-still only on the same verified 4-class subset; no full 9-class overclaim. | `docs/tables/ml/feature_set_comparison_fair.md` | partial; subset-only |
| RPi4 20 FPS | Added target-hardware latency script/protocol but did not claim RPi4 performance. | `tools/rpi4_model_latency_benchmark.py`, `docs/ml/RPI4_MODEL_LATENCY_PROTOCOL.md` | requires RPi4 run |
| Final model choice | RF200 remains baseline, RF100 remains lightweight candidate, MLP remains rejected, v2/21 remain research-only. | `docs/ml/MODEL_SELECTION_RATIONALE.md` | done, no production migration |

## Phase M11 Reviewer Response Addendum

| Reviewer concern | Phase M11 response | Evidence | Status |
| --- | --- | --- | --- |
| Can raw sensor labeling be accelerated before full manual labels? | Added RF/heuristic `suggested_label` package and active-labeling priority. | `docs/tables/ml/suggested_label_summary.md`, `docs/tables/ml/active_labeling_priority.md` | done as review aid |
| Is there a stronger independent teacher model? | Local audit found no CLIP/OpenCLIP/timm/ONNX/TFLite/API teacher available; no download/API call was made. | `docs/ml/AUTO_LABELING_OPTIONS.md`, `docs/tables/ml/auto_labeling_model_candidates.md` | not available locally |
| Are suggestions used as ground truth? | No. `suggested_label` is separate from `manual_label`; sensor accuracy remains not measured. | `docs/ml/AUTO_LABELING_PROTOCOL.md`, `docs/ml/FINAL_DATASET_STRATEGY.md` | guarded |
| Is RF/heuristic agreement reliable evidence? | No. It is consistency analysis only because RF and heuristics share feature signals. | `docs/tables/ml/suggestion_consistency_summary.md` | preliminary/review support |
| Was a weak-labeled dataset created? | No. `auto_weak_label` is deferred until an independent teacher is approved. | `docs/ml/DATASET_GOVERNANCE.md` | deferred |

## Phase P12 Reviewer Response Addendum

| Reviewer concern | Phase P12 response | Evidence | Status |
| --- | --- | --- | --- |
| Need stronger sensor/domain evidence | Added paired IMX/thermal-display ML inventory, feature extraction, drift report, and prediction audit. | `docs/tables/ml/paired_data_ml_inventory.md`, `docs/tables/ml/paired_nir_domain_shift_summary.md` | partial; unlabeled |
| Need NIR validation | Inference is explicitly `RGB-scaler proxy inference, not validated NIR classifier accuracy`; no NIR accuracy is claimed. | `docs/tables/ml/paired_nir_prediction_summary.md` | guarded |
| Need trusted labels for sensor accuracy | Paired sidecar labels are absent/untrusted; paired labeled eval is `not measured`. | `docs/tables/ml/paired_sensor_labeled_eval.md` | requires manual labels |
| Need class decision evidence | Paired predictions are stratified by pairing/session/modality/source and added to class decisions without changing code taxonomy. | `docs/ml/CLASS_DECISION_RECORDS.md`, `docs/ml/DECISION_nir_night_class.md` | documentation-only |
| Need model selection from target data | Paired data has no trusted labels, so it is not used for model selection by accuracy. | `docs/ml/MODEL_SELECTION_RATIONALE.md` | guarded |

## Phase 1 ML Evidence Freeze Addendum

| Reviewer concern | Phase 1 response | Evidence | Status |
| --- | --- | --- | --- |
| Is ML evidence ready for report integration? | Added evidence readiness matrix separating caveated offline metrics, preliminary sensor evidence, and not-measured gates. | `docs/ml/ML_EVIDENCE_READINESS.md`, `docs/tables/ml/ml_evidence_readiness_matrix.md` | done |
| Can sensor labels be accelerated before user review? | Created a visually reviewed agent-label subset with explicit `label_source=agent_manual_label`. | `artifacts/ml/final_labeling_package/agent_manual_labels.csv`, `docs/tables/ml/agent_manual_label_summary.md` | preliminary |
| Does agent labeling create final sensor accuracy? | No. Agent-labeled eval is preliminary and not user-confirmed gold evidence. | `docs/tables/ml/agent_labeled_sensor_eval.md` | guarded |
| Which model should be reported? | RF200 baseline, RF100 lightweight candidate, MLP rejected, v2/21 research-only. | `docs/tables/ml/model_decision_summary.md` | done |
| Which classes are weak? | `transition` should be described as runtime transient candidate; `glare`, `backlight`, and `nir_night` remain provisional/caveated. | `docs/tables/ml/class_decision_summary.md` | done |
| Is report text ready for A4? | Draft ML sections were created for LaTeX integration with conservative wording. | `docs/ml/REPORT_SECTIONS_ML_DRAFT.md` | ready for A4 |
