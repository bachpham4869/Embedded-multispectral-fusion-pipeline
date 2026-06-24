# A3 Dirty Worktree Audit

- Branch: `refactor/ml-taxonomy-eval-plan`
- Date/time: `Wed May 27 23:48:44 +07 2026`
- Safety snapshot timestamp: `20260527_234638`
- Initial dirty status lines captured: `1327`
- Initial untracked files captured: `1305`
- Full pre-ignore inventories are stored locally in `.git_safety_snapshots/`; that directory is ignored and not committed.

## Tracked Modified/Deleted Files

```
M	.gitignore
M	AGENTS.md
M	CLAUDE.md
D	HK252-DATN-142.zip
M	docs/thesis_eval/ch5_thermal/figures/thermal_alg_3dnr.png
M	docs/thesis_eval/ch5_thermal/figures/thermal_alg_background.png
M	docs/thesis_eval/ch5_thermal/figures/thermal_alg_mad.png
M	docs/thesis_eval/ch5_thermal/figures/thermal_preset_comparison.png
M	docs/thesis_eval/ch5_thermal/figures/thermal_profile_comparison.png
M	docs/thesis_eval/thermal/figures/thermal_pipeline_stages.png
D	notebooklm_handoff/HANDOFF_00_MASTER_GUIDE.md
D	notebooklm_handoff/HANDOFF_01_PROJECT_OVERVIEW.md
D	notebooklm_handoff/HANDOFF_02_THEORY_ALGORITHMS.md
D	notebooklm_handoff/HANDOFF_03_SYSTEM_DESIGN.md
D	notebooklm_handoff/HANDOFF_04_IMPLEMENTATION.md
D	notebooklm_handoff/HANDOFF_05_EVALUATION.md
D	notebooklm_handoff/HANDOFF_06_LEARNING_PATH.md
D	notebooklm_handoff/HANDOFF_07_RESEARCH_PROMPT.md
M	src/smartbinocular/main.py
M	tools/gen_ch5_thermal_figures.py
M	tools/gen_thesis_thermal_figures.py
M	tools/offline_pipeline.py
```

## Current Untracked Files After Ignore Rules

```
data/_candidate_datasets/README.md
docs/figures/fusion/failure_cases_grid.png
docs/figures/fusion/sample_comparison_grid.png
docs/figures/fusion/strict_paired_failure_cases_grid.png
docs/figures/fusion/strict_paired_fusion_comparison_grid.png
docs/figures/ml/class_distribution_test.png
docs/figures/ml/class_distribution_train.png
docs/figures/ml/cluster_aware/confusion_extra_trees.png
docs/figures/ml/cluster_aware/confusion_hist_gradient_boosting.png
docs/figures/ml/cluster_aware/confusion_linear_svm.png
docs/figures/ml/cluster_aware/confusion_logistic_regression.png
docs/figures/ml/cluster_aware/confusion_mlp_small_32.png
docs/figures/ml/cluster_aware/confusion_random_forest_100.png
docs/figures/ml/cluster_aware/confusion_random_forest_200_current_config.png
docs/figures/ml/confusion_decision_tree.png
docs/figures/ml/confusion_extra_trees.png
docs/figures/ml/confusion_gaussian_nb.png
docs/figures/ml/confusion_gradient_boosting.png
docs/figures/ml/confusion_hist_gradient_boosting.png
docs/figures/ml/confusion_knn.png
docs/figures/ml/confusion_linear_svm.png
docs/figures/ml/confusion_logistic_regression.png
docs/figures/ml/confusion_mlp_small_32.png
docs/figures/ml/confusion_mlp_small_64_32.png
docs/figures/ml/confusion_random_forest_100.png
docs/figures/ml/confusion_random_forest_200_current_config.png
docs/figures/ml/confusion_random_forest_50.png
docs/figures/ml/confusion_random_forest_depth_12.png
docs/figures/ml/confusion_random_forest_depth_8.png
docs/figures/ml/confusion_random_forest_depth_none.png
docs/figures/ml/confusion_sgd_linear_svm.png
docs/figures/ml/dhash_pairs/dhash_pairs_01.png
docs/figures/ml/dhash_pairs/dhash_pairs_02.png
docs/figures/ml/dhash_pairs/dhash_pairs_03.png
docs/figures/ml/dhash_pairs/dhash_pairs_04.png
docs/figures/ml/group_aware/confusion_extra_trees.png
docs/figures/ml/group_aware/confusion_hist_gradient_boosting.png
docs/figures/ml/group_aware/confusion_linear_svm.png
docs/figures/ml/group_aware/confusion_logistic_regression.png
docs/figures/ml/group_aware/confusion_mlp_small_32.png
docs/figures/ml/group_aware/confusion_random_forest_100.png
docs/figures/ml/group_aware/confusion_random_forest_200_current_config.png
docs/figures/ml/paired_manual_label_contact_sheet.png
docs/figures/ml/paired_nir_confidence_hist.png
docs/figures/ml/paired_nir_feature_pca.png
docs/figures/ml/paired_nir_timelines/raw_sensor_prediction_timeline_unknown.png
docs/figures/ml/raw_sensor_confidence_hist.png
docs/figures/ml/raw_sensor_feature_pca.png
docs/figures/ml/raw_sensor_labeling_contact_sheet.png
docs/figures/ml/raw_sensor_modality_review.png
docs/figures/ml/raw_sensor_timelines/raw_sensor_prediction_timeline_test_30fps_morning.png
docs/figures/ml/suggested_label_timeline.png
docs/fusion/FUSION_ALGORITHM_SURVEY.md
docs/fusion/FUSION_EVALUATION.md
docs/fusion/FUSION_EVAL_DATA_AUDIT.md
docs/fusion/FUSION_EVIDENCE_READINESS.md
docs/fusion/IMAGE_FUSION_METRICS.md
docs/fusion/IMAGE_PROCESSING_EVALUATION.md
docs/fusion/IMAGE_PROCESSING_PIPELINE_AUDIT.md
docs/fusion/REVIEW_RESPONSE_FUSION_MATRIX.md
docs/fusion/RUNTIME_TIMING_EVIDENCE.md
docs/fusion/STRICT_PAIRED_FUSION_CAPTURE_PROTOCOL.md
docs/ml/AUTO_LABELING_OPTIONS.md
docs/ml/AUTO_LABELING_PROTOCOL.md
docs/ml/CLASS_DECISION_RECORDS.md
docs/ml/DATASET_CANDIDATE_SURVEY.md
docs/ml/DATASET_GOVERNANCE.md
docs/ml/DATA_RECOVERABILITY_AUDIT.md
docs/ml/DATA_SPLIT_AND_LEAKAGE_PROTOCOL.md
docs/ml/DECISION_glare_backlight_classes.md
docs/ml/DECISION_nir_night_class.md
docs/ml/DECISION_transition_class.md
docs/ml/DOMAIN_SHIFT_EVALUATION.md
docs/ml/ENV_TAXONOMY.md
docs/ml/FEATURE_SET_RATIONALE.md
docs/ml/FINAL_DATASET_STRATEGY.md
docs/ml/METRIC_DISCREPANCY_INVESTIGATION.md
docs/ml/MODEL_RESEARCH_SURVEY.md
docs/ml/MODEL_SELECTION_RATIONALE.md
docs/ml/PAIRED_DATA_LABELING_PROTOCOL.md
docs/ml/PAIRED_DATA_ML_AUDIT.md
docs/ml/PROCESSING_BUCKETS.md
docs/ml/RAW_SENSOR_CAPTURE_AUDIT.md
docs/ml/RAW_SENSOR_LABELING_PROTOCOL.md
docs/ml/REFINE_PLAN.md
docs/ml/REPORT_PATCH_NOTES.md
docs/ml/REVIEW_RESPONSE_MATRIX.md
docs/ml/RPI4_MODEL_LATENCY_PROTOCOL.md
docs/paired/PAIRED_DATA_AUDIT.md
docs/paired/PAIRED_DATA_FUSION_EVALUATION_SUMMARY.md
docs/tables/fusion/artifact_inventory.md
docs/tables/fusion/current_vs_legacy_algorithms.md
docs/tables/fusion/enhancement_algorithm_survey.md
docs/tables/fusion/evidence_readiness_matrix.md
docs/tables/fusion/failure_case_summary.md
docs/tables/fusion/fusion_algorithm_comparison.md
docs/tables/fusion/fusion_algorithm_survey.md
docs/tables/fusion/fusion_quality_summary.md
docs/tables/fusion/image_processing_algorithm_comparison.md
docs/tables/fusion/metric_definitions.md
docs/tables/fusion/nir_quality_summary.md
docs/tables/fusion/paired_runtime_timing_summary.md
docs/tables/fusion/pairing_manifest_summary.md
docs/tables/fusion/per_bucket_missing_evidence.md
docs/tables/fusion/per_bucket_processing_eval.md
docs/tables/fusion/processing_name_aliases.md
docs/tables/fusion/required_capture_metadata.md
docs/tables/fusion/reviewer_response_fusion_matrix.md
docs/tables/fusion/runtime_timing_summary.md
docs/tables/fusion/strict_paired_failure_case_summary.md
docs/tables/fusion/strict_paired_fusion_algorithm_comparison.md
docs/tables/fusion/strict_paired_fusion_quality_summary.md
docs/tables/fusion/strict_paired_nir_quality_summary.md
docs/tables/fusion/strict_paired_thermal_quality_summary.md
docs/tables/fusion/thermal_quality_summary.md
docs/tables/ml/active_labeling_priority.md
docs/tables/ml/auto_label_class_prompt_mapping.md
docs/tables/ml/auto_labeling_model_candidates.md
docs/tables/ml/class_imbalance_summary.md
docs/tables/ml/class_support_risk_summary.md
docs/tables/ml/cluster_aware_leakage_summary.md
docs/tables/ml/cluster_aware_near_duplicate_pairs.md
docs/tables/ml/cluster_aware_remaining_risks.md
docs/tables/ml/cluster_aware_split_distribution_conservative.md
docs/tables/ml/cluster_aware_split_distribution_strict.md
docs/tables/ml/dataset_candidate_summary.md
docs/tables/ml/dataset_distribution_reference.csv
docs/tables/ml/dataset_distribution_reference.md
docs/tables/ml/dataset_distribution_test.csv
docs/tables/ml/dataset_distribution_test.md
docs/tables/ml/dataset_distribution_train.csv
docs/tables/ml/dataset_distribution_train.md
docs/tables/ml/dhash_pair_analysis.md
docs/tables/ml/duplicate_cluster_summary_conservative.md
docs/tables/ml/duplicate_cluster_summary_strict.md
docs/tables/ml/feature_21_ablation.md
docs/tables/ml/feature_21_definition.md
docs/tables/ml/feature_candidate_build_summary.md
docs/tables/ml/feature_set_12_vs_v2_vs_21.csv
docs/tables/ml/feature_set_12_vs_v2_vs_21.md
docs/tables/ml/feature_set_comparison_fair.csv
docs/tables/ml/feature_set_comparison_fair.md
docs/tables/ml/feature_vector_overlap_details.md
docs/tables/ml/final_dataset_evidence_summary.md
docs/tables/ml/frozen_eval_overlap_sensitivity.md
docs/tables/ml/group_aware_split_distribution.md
docs/tables/ml/image_level_leakage_current_split_summary.md
docs/tables/ml/image_level_leakage_summary.md
docs/tables/ml/leakage_check_summary.md
docs/tables/ml/manual_label_status.md
docs/tables/ml/mlp_variant_comparison.csv
docs/tables/ml/mlp_variant_comparison.md
docs/tables/ml/model_comparison_12features.csv
docs/tables/ml/model_comparison_12features.md
docs/tables/ml/model_comparison_12features_ci.md
docs/tables/ml/model_comparison_12features_per_class.csv
docs/tables/ml/model_comparison_cluster_aware.csv
docs/tables/ml/model_comparison_cluster_aware.md
docs/tables/ml/model_comparison_cluster_aware_ci.md
docs/tables/ml/model_comparison_cluster_aware_per_class.csv
docs/tables/ml/model_comparison_group_aware.csv
docs/tables/ml/model_comparison_group_aware.md
docs/tables/ml/model_comparison_group_aware_ci.md
docs/tables/ml/model_comparison_group_aware_per_class.csv
docs/tables/ml/near_duplicate_current_split_pairs.md
docs/tables/ml/near_duplicate_pairs.md
docs/tables/ml/paired_data_ml_inventory.md
docs/tables/ml/paired_nir_domain_shift_summary.md
docs/tables/ml/paired_nir_feature_drift.md
docs/tables/ml/paired_nir_feature_summary.md
docs/tables/ml/paired_nir_prediction_distribution.md
docs/tables/ml/paired_nir_prediction_summary.md
docs/tables/ml/paired_sensor_labeled_eval.md
docs/tables/ml/raw_data_recoverability.md
docs/tables/ml/raw_sensor_capture_inventory.md
docs/tables/ml/raw_sensor_domain_shift_summary.md
docs/tables/ml/raw_sensor_feature_drift.md
docs/tables/ml/raw_sensor_feature_summary.md
docs/tables/ml/raw_sensor_frame_extraction_summary.md
docs/tables/ml/raw_sensor_labeled_eval.md
docs/tables/ml/raw_sensor_labeled_per_class.md
docs/tables/ml/raw_sensor_prediction_distribution.md
docs/tables/ml/raw_sensor_prediction_summary.md
docs/tables/ml/source_distribution.md
docs/tables/ml/specialized_model_survey.md
docs/tables/ml/split_benchmark_progression.md
docs/tables/ml/suggested_label_summary.md
docs/tables/ml/suggestion_consistency_summary.md
docs/tables/ml/suggestion_disagreement_cases.md
docs/tables/paired/paired_data_inventory.md
docs/tables/paired/paired_ml_evidence_summary.md
docs/tables/paired/strict_paired_manifest_summary.md
tests/test_capture_paired_data_standalone.py
tests/test_fusion_eval_metrics.py
tests/test_fusion_evidence_readiness.py
tests/test_fusion_failure_mining.py
tests/test_fusion_naming_aliases.py
tests/test_fusion_pairing_manifest.py
tests/test_fusion_runtime_timing_hardening.py
tests/test_ml_phase1_tools.py
tests/test_ml_phase3_tools.py
tests/test_ml_phase4_tools.py
tests/test_ml_sensor_phase_tools.py
tests/test_ml_suggested_label_tools.py
tests/test_paired_data_manifest.py
tests/test_paired_fusion_evaluation.py
tests/test_paired_ml_tools.py
tests/test_per_bucket_paired_eval.py
tests/test_strict_paired_capture_protocol.py
tools/analyze_dataset_distribution.py
tools/analyze_dhash_pairs.py
tools/audit_data_recoverability.py
tools/audit_labeling_support_options.py
tools/build_duplicate_clusters.py
tools/build_fusion_artifact_inventory.py
tools/build_optical_v2_feature_table.py
tools/build_paired_data_manifest.py
tools/build_paired_labeling_package.py
tools/build_sensor_labeling_package.py
tools/capture_paired_data.py
tools/check_dataset_leakage.py
tools/check_image_level_leakage.py
tools/compare_classifiers.py
tools/compare_feature_sets.py
tools/compare_fusion_algorithms.py
tools/compare_image_algorithms.py
tools/compare_mlp_variants.py
tools/domain_shift_report.py
tools/evaluate_frozen_classifier.py
tools/evaluate_fusion_quality.py
tools/evaluate_image_quality.py
tools/evaluate_paired_fusion.py
tools/evaluate_sensor_labeled_predictions.py
tools/evaluate_suggestion_consistency.py
tools/extract_paired_nir_features.py
tools/extract_sensor_frame_features.py
tools/extract_sensor_video_frames.py
tools/fusion_eval_manifest.py
tools/fusion_eval_metrics.py
tools/harden_fusion_evidence.py
tools/inventory_raw_sensor_captures.py
tools/ml_metadata_utils.py
tools/optical_candidate_features.py
tools/predict_sensor_frames.py
tools/q1_fusion_eval.py
tools/rebuild_training_jsonl_with_metadata.py
tools/rpi4_model_latency_benchmark.py
tools/split_duplicate_cluster_aware_jsonl.py
tools/split_group_aware_jsonl.py
tools/suggest_sensor_labels.py
tools/validate_sensor_manual_labels.py
```

## Large Files Summary (>5M)

```
./.gitnexus/lbug
./artifacts/paired_eval/fusion_quality_metrics.csv
./artifacts/paired_eval/nir_quality_metrics.csv
./data/paired_data/imx_paired.mp4
./data/paired_data/thermal_paired.mp4
./data/raw_sensor_captures/test_30fps_morning.mp4
./data/thermal/10393655/video_dataset10.zip
./data/thermal/10393655/video_dataset15.zip
./data/thermal/10393655/video_dataset5.zip
./data/thermal/ALL_IN_ONE_RGB_IMG_ANOT/Thumbs.db
./data/training/from_logs_train.jsonl
./data/training/from_logs_train_a.jsonl
./data/training/merged_logs_ml.jsonl
./data/training_v2/from_logs_train_metadata.jsonl
./data/training_v2/group_aware_train.jsonl
./data/training_v2/merged_logs_ml_metadata.jsonl
./data/training_v3/cluster_aware_conservative_train.jsonl
./data/training_v3/cluster_aware_strict_train.jsonl
./data/training_v3/merged_logs_ml_clustered_conservative.jsonl
./data/training_v3/merged_logs_ml_clustered_strict.jsonl
./data/training_v4/cluster_aware_conservative_train_optical_v2_verified.jsonl
./data/weather/ExDark/Bus/2015_02282.jpg
./data/weather/ExDark/People/2015_06615.JPG
./data/weather/ExDark/People/2015_06616.JPG
./data/weather/ExDark/People/2015_06618.JPG
./data/weather/ExDark/People/2015_06619.JPG
./data/weather/ExDark/People/2015_06620.JPG
./data/weather/ExDark/Table/2015_07143.JPG
./data/weather/ExDark/Table/2015_07145.JPG
./data/weather/ExDark/Table/2015_07146.JPG
./data/weather/indoor/indoorCVPR_09/Images/grocerystore/grocery265.jpg
./models/ablation/rf_optical_12.joblib
./models/ablation/rf_optical_9.joblib
./models/production/env_classifier.joblib
```

## Dataset/Artifact Directories Found

- `data/paired_data/` — 33M; local dataset/generated artifact; ignored, not staged.
- `data/raw_sensor_captures/` — 1.3G; local dataset/generated artifact; ignored, not staged.
- `data/sensor_eval/` — 86M; local dataset/generated artifact; ignored, not staged.
- `data/training_v2/` — 60M; local dataset/generated artifact; ignored, not staged.
- `data/training_v3/` — 90M; local dataset/generated artifact; ignored, not staged.
- `data/training_v4/` — 12M; local dataset/generated artifact; ignored, not staged.
- `artifacts/paired_eval/` — 74M; local dataset/generated artifact; ignored, not staged.
- `artifacts/fusion_eval/` — 2.6M; local dataset/generated artifact; ignored, not staged.
- `artifacts/ml/` — 29M; local dataset/generated artifact; ignored, not staged.
- `q1_results/` — 3.8M; local dataset/generated artifact; ignored, not staged.
- `noisy_output_test/` — 45M; local dataset/generated artifact; ignored, not staged.

## Ignored Dataset/Artifact Checks

```
.gitignore:26:data/paired_data/	data/paired_data/imx_paired.mp4
.gitignore:26:data/paired_data/	data/paired_data/thermal_paired.mp4
.gitignore:26:data/paired_data/	data/paired_data/timestamps.csv
.gitignore:36:artifacts/paired_eval/	artifacts/paired_eval/fusion_quality_metrics.csv
.gitignore:36:artifacts/paired_eval/	artifacts/paired_eval/manual_label_template_paired.csv
.gitignore:36:artifacts/paired_eval/	artifacts/paired_eval/nir_quality_metrics.csv
.gitignore:36:artifacts/paired_eval/	artifacts/paired_eval/paired_data_inventory.csv
.gitignore:36:artifacts/paired_eval/	artifacts/paired_eval/paired_data_manifest.jsonl
.gitignore:36:artifacts/paired_eval/	artifacts/paired_eval/paired_data_run_manifest.json
.gitignore:36:artifacts/paired_eval/	artifacts/paired_eval/paired_fusion_eval_run_manifest.json
.gitignore:36:artifacts/paired_eval/	artifacts/paired_eval/paired_ml_manifest.csv
.gitignore:36:artifacts/paired_eval/	artifacts/paired_eval/paired_nir_domain_shift_metrics.json
.gitignore:36:artifacts/paired_eval/	artifacts/paired_eval/paired_nir_predictions.csv
.gitignore:36:artifacts/paired_eval/	artifacts/paired_eval/strict_paired_failure_cases.csv
.gitignore:36:artifacts/paired_eval/	artifacts/paired_eval/strict_paired_manifest.csv
.gitignore:36:artifacts/paired_eval/	artifacts/paired_eval/thermal_quality_metrics.csv
.gitignore:27:data/raw_sensor_captures/	data/raw_sensor_captures/test_30fps_morning.mp4
.gitignore:39:artifacts/ml/	artifacts/ml/eval
.gitignore:39:artifacts/ml/	artifacts/ml/feature_set_12_vs_v2_vs_21
.gitignore:39:artifacts/ml/	artifacts/ml/feature_set_comparison
.gitignore:39:artifacts/ml/	artifacts/ml/leakage
.gitignore:39:artifacts/ml/	artifacts/ml/mlp_variant_comparison
.gitignore:39:artifacts/ml/	artifacts/ml/model_comparison_12features
.gitignore:39:artifacts/ml/	artifacts/ml/model_comparison_cluster_aware
.gitignore:39:artifacts/ml/	artifacts/ml/model_comparison_group_aware
.gitignore:39:artifacts/ml/	artifacts/ml/sensor_domain_shift
```

## Already-Tracked Forbidden Dataset/Artifact Check

```
(none)
```

No tracked paired dataset, paired artifact, runtime paired artifact, video, model binary, or listed forbidden file matched at audit time; no `git rm --cached` was required.

## Classification And Recommended Action

### A. Safe source code from paired/fusion/ML work
- `?? tests/test_paired_data_manifest.py`
- `?? tests/test_paired_fusion_evaluation.py`
- `?? tests/test_paired_ml_tools.py`
- `?? tests/test_per_bucket_paired_eval.py`
- `?? tools/build_paired_data_manifest.py`
- `?? tools/build_paired_labeling_package.py`
- `?? tools/evaluate_paired_fusion.py`
- `?? tools/extract_paired_nir_features.py`

Recommended action: stage explicitly for the paired/fusion or paired ML workflow commits after forbidden-file and test checks.

### B. Existing tool modifications that need review before commit
- `?? tools/domain_shift_report.py`
- `?? tools/predict_sensor_frames.py`

Recommended action: review diffs and CLI help first; commit separately only if backward-compatible.

### C. Safe docs/tables/small figures
- `?? docs/figures/fusion/failure_cases_grid.png`
- `?? docs/figures/fusion/sample_comparison_grid.png`
- `?? docs/figures/fusion/strict_paired_failure_cases_grid.png`
- `?? docs/figures/fusion/strict_paired_fusion_comparison_grid.png`
- `?? docs/figures/ml/class_distribution_test.png`
- `?? docs/figures/ml/class_distribution_train.png`
- `?? docs/figures/ml/cluster_aware/confusion_extra_trees.png`
- `?? docs/figures/ml/cluster_aware/confusion_hist_gradient_boosting.png`
- `?? docs/figures/ml/cluster_aware/confusion_linear_svm.png`
- `?? docs/figures/ml/cluster_aware/confusion_logistic_regression.png`
- `?? docs/figures/ml/cluster_aware/confusion_mlp_small_32.png`
- `?? docs/figures/ml/cluster_aware/confusion_random_forest_100.png`
- `?? docs/figures/ml/cluster_aware/confusion_random_forest_200_current_config.png`
- `?? docs/figures/ml/confusion_decision_tree.png`
- `?? docs/figures/ml/confusion_extra_trees.png`
- `?? docs/figures/ml/confusion_gaussian_nb.png`
- `?? docs/figures/ml/confusion_gradient_boosting.png`
- `?? docs/figures/ml/confusion_hist_gradient_boosting.png`
- `?? docs/figures/ml/confusion_knn.png`
- `?? docs/figures/ml/confusion_linear_svm.png`
- `?? docs/figures/ml/confusion_logistic_regression.png`
- `?? docs/figures/ml/confusion_mlp_small_32.png`
- `?? docs/figures/ml/confusion_mlp_small_64_32.png`
- `?? docs/figures/ml/confusion_random_forest_100.png`
- `?? docs/figures/ml/confusion_random_forest_200_current_config.png`
- `?? docs/figures/ml/confusion_random_forest_50.png`
- `?? docs/figures/ml/confusion_random_forest_depth_12.png`
- `?? docs/figures/ml/confusion_random_forest_depth_8.png`
- `?? docs/figures/ml/confusion_random_forest_depth_none.png`
- `?? docs/figures/ml/confusion_sgd_linear_svm.png`
- `?? docs/figures/ml/dhash_pairs/dhash_pairs_01.png`
- `?? docs/figures/ml/dhash_pairs/dhash_pairs_02.png`
- `?? docs/figures/ml/dhash_pairs/dhash_pairs_03.png`
- `?? docs/figures/ml/dhash_pairs/dhash_pairs_04.png`
- `?? docs/figures/ml/group_aware/confusion_extra_trees.png`
- `?? docs/figures/ml/group_aware/confusion_hist_gradient_boosting.png`
- `?? docs/figures/ml/group_aware/confusion_linear_svm.png`
- `?? docs/figures/ml/group_aware/confusion_logistic_regression.png`
- `?? docs/figures/ml/group_aware/confusion_mlp_small_32.png`
- `?? docs/figures/ml/group_aware/confusion_random_forest_100.png`
- `?? docs/figures/ml/group_aware/confusion_random_forest_200_current_config.png`
- `?? docs/figures/ml/paired_manual_label_contact_sheet.png`
- `?? docs/figures/ml/paired_nir_confidence_hist.png`
- `?? docs/figures/ml/paired_nir_feature_pca.png`
- `?? docs/figures/ml/paired_nir_timelines/raw_sensor_prediction_timeline_unknown.png`
- `?? docs/figures/ml/raw_sensor_confidence_hist.png`
- `?? docs/figures/ml/raw_sensor_feature_pca.png`
- `?? docs/figures/ml/raw_sensor_labeling_contact_sheet.png`
- `?? docs/figures/ml/raw_sensor_modality_review.png`
- `?? docs/figures/ml/raw_sensor_timelines/raw_sensor_prediction_timeline_test_30fps_morning.png`
- `?? docs/figures/ml/suggested_label_timeline.png`
- `?? docs/fusion/FUSION_ALGORITHM_SURVEY.md`
- `?? docs/fusion/FUSION_EVALUATION.md`
- `?? docs/fusion/FUSION_EVAL_DATA_AUDIT.md`
- `?? docs/fusion/FUSION_EVIDENCE_READINESS.md`
- `?? docs/fusion/IMAGE_FUSION_METRICS.md`
- `?? docs/fusion/IMAGE_PROCESSING_EVALUATION.md`
- `?? docs/fusion/IMAGE_PROCESSING_PIPELINE_AUDIT.md`
- `?? docs/fusion/REVIEW_RESPONSE_FUSION_MATRIX.md`
- `?? docs/fusion/RUNTIME_TIMING_EVIDENCE.md`
- `?? docs/fusion/STRICT_PAIRED_FUSION_CAPTURE_PROTOCOL.md`
- `?? docs/ml/AUTO_LABELING_OPTIONS.md`
- `?? docs/ml/AUTO_LABELING_PROTOCOL.md`
- `?? docs/ml/CLASS_DECISION_RECORDS.md`
- `?? docs/ml/DATASET_CANDIDATE_SURVEY.md`
- `?? docs/ml/DATASET_GOVERNANCE.md`
- `?? docs/ml/DATA_RECOVERABILITY_AUDIT.md`
- `?? docs/ml/DATA_SPLIT_AND_LEAKAGE_PROTOCOL.md`
- `?? docs/ml/DECISION_glare_backlight_classes.md`
- `?? docs/ml/DECISION_nir_night_class.md`
- `?? docs/ml/DECISION_transition_class.md`
- `?? docs/ml/DOMAIN_SHIFT_EVALUATION.md`
- `?? docs/ml/ENV_TAXONOMY.md`
- `?? docs/ml/FEATURE_SET_RATIONALE.md`
- `?? docs/ml/FINAL_DATASET_STRATEGY.md`
- `?? docs/ml/METRIC_DISCREPANCY_INVESTIGATION.md`
- `?? docs/ml/MODEL_RESEARCH_SURVEY.md`
- `?? docs/ml/MODEL_SELECTION_RATIONALE.md`
- `?? docs/ml/PAIRED_DATA_LABELING_PROTOCOL.md`
- `?? docs/ml/PAIRED_DATA_ML_AUDIT.md`
- `?? docs/ml/PROCESSING_BUCKETS.md`
- `?? docs/ml/RAW_SENSOR_CAPTURE_AUDIT.md`
- `?? docs/ml/RAW_SENSOR_LABELING_PROTOCOL.md`
- `?? docs/ml/REFINE_PLAN.md`
- `?? docs/ml/REPORT_PATCH_NOTES.md`
- `?? docs/ml/REVIEW_RESPONSE_MATRIX.md`
- `?? docs/ml/RPI4_MODEL_LATENCY_PROTOCOL.md`
- `?? docs/paired/PAIRED_DATA_AUDIT.md`
- `?? docs/paired/PAIRED_DATA_FUSION_EVALUATION_SUMMARY.md`
- `?? docs/tables/fusion/artifact_inventory.md`
- `?? docs/tables/fusion/current_vs_legacy_algorithms.md`
- `?? docs/tables/fusion/enhancement_algorithm_survey.md`
- `?? docs/tables/fusion/evidence_readiness_matrix.md`
- `?? docs/tables/fusion/failure_case_summary.md`
- `?? docs/tables/fusion/fusion_algorithm_comparison.md`
- `?? docs/tables/fusion/fusion_algorithm_survey.md`
- `?? docs/tables/fusion/fusion_quality_summary.md`
- `?? docs/tables/fusion/image_processing_algorithm_comparison.md`
- `?? docs/tables/fusion/metric_definitions.md`
- `?? docs/tables/fusion/nir_quality_summary.md`
- `?? docs/tables/fusion/paired_runtime_timing_summary.md`
- `?? docs/tables/fusion/pairing_manifest_summary.md`
- `?? docs/tables/fusion/per_bucket_missing_evidence.md`
- `?? docs/tables/fusion/per_bucket_processing_eval.md`
- `?? docs/tables/fusion/processing_name_aliases.md`
- `?? docs/tables/fusion/required_capture_metadata.md`
- `?? docs/tables/fusion/reviewer_response_fusion_matrix.md`
- `?? docs/tables/fusion/runtime_timing_summary.md`
- `?? docs/tables/fusion/strict_paired_failure_case_summary.md`
- `?? docs/tables/fusion/strict_paired_fusion_algorithm_comparison.md`
- `?? docs/tables/fusion/strict_paired_fusion_quality_summary.md`
- `?? docs/tables/fusion/strict_paired_nir_quality_summary.md`
- `?? docs/tables/fusion/strict_paired_thermal_quality_summary.md`
- `?? docs/tables/fusion/thermal_quality_summary.md`
- `?? docs/tables/ml/active_labeling_priority.md`
- `?? docs/tables/ml/auto_label_class_prompt_mapping.md`
- `?? docs/tables/ml/auto_labeling_model_candidates.md`
- `?? docs/tables/ml/class_imbalance_summary.md`
- `?? docs/tables/ml/class_support_risk_summary.md`
- `?? docs/tables/ml/cluster_aware_leakage_summary.md`
- `?? docs/tables/ml/cluster_aware_near_duplicate_pairs.md`
- `?? docs/tables/ml/cluster_aware_remaining_risks.md`
- `?? docs/tables/ml/cluster_aware_split_distribution_conservative.md`
- `?? docs/tables/ml/cluster_aware_split_distribution_strict.md`
- `?? docs/tables/ml/dataset_candidate_summary.md`
- `?? docs/tables/ml/dataset_distribution_reference.csv`
- `?? docs/tables/ml/dataset_distribution_reference.md`
- `?? docs/tables/ml/dataset_distribution_test.csv`
- `?? docs/tables/ml/dataset_distribution_test.md`
- `?? docs/tables/ml/dataset_distribution_train.csv`
- `?? docs/tables/ml/dataset_distribution_train.md`
- `?? docs/tables/ml/dhash_pair_analysis.md`
- `?? docs/tables/ml/duplicate_cluster_summary_conservative.md`
- `?? docs/tables/ml/duplicate_cluster_summary_strict.md`
- `?? docs/tables/ml/feature_21_ablation.md`
- `?? docs/tables/ml/feature_21_definition.md`
- `?? docs/tables/ml/feature_candidate_build_summary.md`
- `?? docs/tables/ml/feature_set_12_vs_v2_vs_21.csv`
- `?? docs/tables/ml/feature_set_12_vs_v2_vs_21.md`
- `?? docs/tables/ml/feature_set_comparison_fair.csv`
- `?? docs/tables/ml/feature_set_comparison_fair.md`
- `?? docs/tables/ml/feature_vector_overlap_details.md`
- `?? docs/tables/ml/final_dataset_evidence_summary.md`
- `?? docs/tables/ml/frozen_eval_overlap_sensitivity.md`
- `?? docs/tables/ml/group_aware_split_distribution.md`
- `?? docs/tables/ml/image_level_leakage_current_split_summary.md`
- `?? docs/tables/ml/image_level_leakage_summary.md`
- `?? docs/tables/ml/leakage_check_summary.md`
- `?? docs/tables/ml/manual_label_status.md`
- `?? docs/tables/ml/mlp_variant_comparison.csv`
- `?? docs/tables/ml/mlp_variant_comparison.md`
- `?? docs/tables/ml/model_comparison_12features.csv`
- `?? docs/tables/ml/model_comparison_12features.md`
- `?? docs/tables/ml/model_comparison_12features_ci.md`
- `?? docs/tables/ml/model_comparison_12features_per_class.csv`
- `?? docs/tables/ml/model_comparison_cluster_aware.csv`
- `?? docs/tables/ml/model_comparison_cluster_aware.md`
- `?? docs/tables/ml/model_comparison_cluster_aware_ci.md`
- `?? docs/tables/ml/model_comparison_cluster_aware_per_class.csv`
- `?? docs/tables/ml/model_comparison_group_aware.csv`
- `?? docs/tables/ml/model_comparison_group_aware.md`
- `?? docs/tables/ml/model_comparison_group_aware_ci.md`
- `?? docs/tables/ml/model_comparison_group_aware_per_class.csv`
- `?? docs/tables/ml/near_duplicate_current_split_pairs.md`
- `?? docs/tables/ml/near_duplicate_pairs.md`
- `?? docs/tables/ml/paired_data_ml_inventory.md`
- `?? docs/tables/ml/paired_nir_domain_shift_summary.md`
- `?? docs/tables/ml/paired_nir_feature_drift.md`
- `?? docs/tables/ml/paired_nir_feature_summary.md`
- `?? docs/tables/ml/paired_nir_prediction_distribution.md`
- `?? docs/tables/ml/paired_nir_prediction_summary.md`
- `?? docs/tables/ml/paired_sensor_labeled_eval.md`
- `?? docs/tables/ml/raw_data_recoverability.md`
- `?? docs/tables/ml/raw_sensor_capture_inventory.md`
- `?? docs/tables/ml/raw_sensor_domain_shift_summary.md`
- `?? docs/tables/ml/raw_sensor_feature_drift.md`
- `?? docs/tables/ml/raw_sensor_feature_summary.md`
- `?? docs/tables/ml/raw_sensor_frame_extraction_summary.md`
- `?? docs/tables/ml/raw_sensor_labeled_eval.md`
- `?? docs/tables/ml/raw_sensor_labeled_per_class.md`
- `?? docs/tables/ml/raw_sensor_prediction_distribution.md`
- `?? docs/tables/ml/raw_sensor_prediction_summary.md`
- `?? docs/tables/ml/source_distribution.md`
- `?? docs/tables/ml/specialized_model_survey.md`
- `?? docs/tables/ml/split_benchmark_progression.md`
- `?? docs/tables/ml/suggested_label_summary.md`
- `?? docs/tables/ml/suggestion_consistency_summary.md`
- `?? docs/tables/ml/suggestion_disagreement_cases.md`
- `?? docs/tables/paired/paired_data_inventory.md`
- `?? docs/tables/paired/paired_ml_evidence_summary.md`
- `?? docs/tables/paired/strict_paired_manifest_summary.md`

Recommended action: stage only requested paired/fusion/ML docs, tables, and small figures explicitly; leave broad prior docs/figures for separate review if outside A1/A2 scope.

### D. Large generated artifacts to ignore, not commit by default
- None currently visible after ignore rules.

Recommended action: ignore; do not commit generated artifact directories by default.

### E. Raw datasets to ignore, not commit
- `?? data/_candidate_datasets/README.md`

Recommended action: ignore; do not delete local raw data.

### F. Model binaries to ignore, not commit
- None currently visible after ignore rules.

Recommended action: ignore; do not commit model binaries.

### G. Pre-existing/out-of-scope runtime or thesis edits
- ` M AGENTS.md`
- ` M CLAUDE.md`
- ` D HK252-DATN-142.zip`
- ` M docs/thesis_eval/ch5_thermal/figures/thermal_alg_3dnr.png`
- ` M docs/thesis_eval/ch5_thermal/figures/thermal_alg_background.png`
- ` M docs/thesis_eval/ch5_thermal/figures/thermal_alg_mad.png`
- ` M docs/thesis_eval/ch5_thermal/figures/thermal_preset_comparison.png`
- ` M docs/thesis_eval/ch5_thermal/figures/thermal_profile_comparison.png`
- ` M docs/thesis_eval/thermal/figures/thermal_pipeline_stages.png`
- ` D notebooklm_handoff/HANDOFF_00_MASTER_GUIDE.md`
- ` D notebooklm_handoff/HANDOFF_01_PROJECT_OVERVIEW.md`
- ` D notebooklm_handoff/HANDOFF_02_THEORY_ALGORITHMS.md`
- ` D notebooklm_handoff/HANDOFF_03_SYSTEM_DESIGN.md`
- ` D notebooklm_handoff/HANDOFF_04_IMPLEMENTATION.md`
- ` D notebooklm_handoff/HANDOFF_05_EVALUATION.md`
- ` D notebooklm_handoff/HANDOFF_06_LEARNING_PATH.md`
- ` D notebooklm_handoff/HANDOFF_07_RESEARCH_PROMPT.md`
- ` M src/smartbinocular/main.py`
- ` M tools/gen_ch5_thermal_figures.py`
- ` M tools/gen_thesis_thermal_figures.py`
- ` M tools/offline_pipeline.py`

Recommended action: leave unstaged; user confirmation required before commit or revert.

### Other / needs explicit review
- `M .gitignore`
- `?? tests/test_capture_paired_data_standalone.py — source/test work outside requested A1/A2 stage list`
- `?? tests/test_fusion_eval_metrics.py — source/test work outside requested A1/A2 stage list`
- `?? tests/test_fusion_evidence_readiness.py — source/test work outside requested A1/A2 stage list`
- `?? tests/test_fusion_failure_mining.py — source/test work outside requested A1/A2 stage list`
- `?? tests/test_fusion_naming_aliases.py — source/test work outside requested A1/A2 stage list`
- `?? tests/test_fusion_pairing_manifest.py — source/test work outside requested A1/A2 stage list`
- `?? tests/test_fusion_runtime_timing_hardening.py — source/test work outside requested A1/A2 stage list`
- `?? tests/test_ml_phase1_tools.py — source/test work outside requested A1/A2 stage list`
- `?? tests/test_ml_phase3_tools.py — source/test work outside requested A1/A2 stage list`
- `?? tests/test_ml_phase4_tools.py — source/test work outside requested A1/A2 stage list`
- `?? tests/test_ml_sensor_phase_tools.py — source/test work outside requested A1/A2 stage list`
- `?? tests/test_ml_suggested_label_tools.py — source/test work outside requested A1/A2 stage list`
- `?? tests/test_strict_paired_capture_protocol.py — source/test work outside requested A1/A2 stage list`
- `?? tools/analyze_dataset_distribution.py — source/test work outside requested A1/A2 stage list`
- `?? tools/analyze_dhash_pairs.py — source/test work outside requested A1/A2 stage list`
- `?? tools/audit_data_recoverability.py — source/test work outside requested A1/A2 stage list`
- `?? tools/audit_labeling_support_options.py — source/test work outside requested A1/A2 stage list`
- `?? tools/build_duplicate_clusters.py — source/test work outside requested A1/A2 stage list`
- `?? tools/build_fusion_artifact_inventory.py — source/test work outside requested A1/A2 stage list`
- `?? tools/build_optical_v2_feature_table.py — source/test work outside requested A1/A2 stage list`
- `?? tools/build_sensor_labeling_package.py — source/test work outside requested A1/A2 stage list`
- `?? tools/capture_paired_data.py — source/test work outside requested A1/A2 stage list`
- `?? tools/check_dataset_leakage.py — source/test work outside requested A1/A2 stage list`
- `?? tools/check_image_level_leakage.py — source/test work outside requested A1/A2 stage list`
- `?? tools/compare_classifiers.py — source/test work outside requested A1/A2 stage list`
- `?? tools/compare_feature_sets.py — source/test work outside requested A1/A2 stage list`
- `?? tools/compare_fusion_algorithms.py — source/test work outside requested A1/A2 stage list`
- `?? tools/compare_image_algorithms.py — source/test work outside requested A1/A2 stage list`
- `?? tools/compare_mlp_variants.py — source/test work outside requested A1/A2 stage list`
- `?? tools/evaluate_frozen_classifier.py — source/test work outside requested A1/A2 stage list`
- `?? tools/evaluate_fusion_quality.py — source/test work outside requested A1/A2 stage list`
- `?? tools/evaluate_image_quality.py — source/test work outside requested A1/A2 stage list`
- `?? tools/evaluate_sensor_labeled_predictions.py — source/test work outside requested A1/A2 stage list`
- `?? tools/evaluate_suggestion_consistency.py — source/test work outside requested A1/A2 stage list`
- `?? tools/extract_sensor_frame_features.py — source/test work outside requested A1/A2 stage list`
- `?? tools/extract_sensor_video_frames.py — source/test work outside requested A1/A2 stage list`
- `?? tools/fusion_eval_manifest.py — source/test work outside requested A1/A2 stage list`
- `?? tools/fusion_eval_metrics.py — source/test work outside requested A1/A2 stage list`
- `?? tools/harden_fusion_evidence.py — source/test work outside requested A1/A2 stage list`
- `?? tools/inventory_raw_sensor_captures.py — source/test work outside requested A1/A2 stage list`
- `?? tools/ml_metadata_utils.py — source/test work outside requested A1/A2 stage list`
- `?? tools/optical_candidate_features.py — source/test work outside requested A1/A2 stage list`
- `?? tools/q1_fusion_eval.py — source/test work outside requested A1/A2 stage list`
- `?? tools/rebuild_training_jsonl_with_metadata.py — source/test work outside requested A1/A2 stage list`
- `?? tools/rpi4_model_latency_benchmark.py — source/test work outside requested A1/A2 stage list`
- `?? tools/split_duplicate_cluster_aware_jsonl.py — source/test work outside requested A1/A2 stage list`
- `?? tools/split_group_aware_jsonl.py — source/test work outside requested A1/A2 stage list`
- `?? tools/suggest_sensor_labels.py — source/test work outside requested A1/A2 stage list`
- `?? tools/validate_sensor_manual_labels.py — source/test work outside requested A1/A2 stage list`

Recommended action: review individually; do not stage via broad pathspec.
