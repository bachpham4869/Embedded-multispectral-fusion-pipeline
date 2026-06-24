# EVAL DOMAIN SHIFT MAD WORK REPORT

Date: 2026-06-05

## 1. Executive summary

This pass created a review-only evidence package for two newly accepted label sets and an external BU-TIV MAD benchmark. The two label sets are treated as `user-accepted agent-reviewed labels`, not independent multi-annotator gold labels. No thesis LaTeX files were edited.

The classifier evaluation now covers 203 retained accepted-label frames across day/backlight/glare and IMX paired night/low-light subsets. The strongest valid classifier view is `retained_nonambiguous`: top-1 accuracy 0.3716, top-2 hit rate 0.6120, balanced accuracy 0.2981, macro-F1 0.2678, weighted-F1 0.3407. The IMX paired night subset is the clearest positive result: high-confidence top-1 accuracy 0.5904 and top-2 hit rate 0.8072 on 83 frames.

The BU-TIV benchmark used extracted 16-bit NUC frames resized to 80x62. The selected MAD configuration was chosen on a 160-frame tune subset only; primary metrics are reported on the 3,322-frame full-minus-tune heldout set. Heldout frame precision is 0.9905, frame recall is 0.3772, frame F1 is 0.5464, object recall at 3 px is 0.1523, and mean runtime is 0.1527 ms/frame on the local workstation.

## 2. Files read

| Group | Paths |
| --- | --- |
| Accepted day labels | `review_artifacts/sensor_domain_labeling/sensor_top2_labels_agent_review.csv`, report and ambiguity summary in the same folder |
| Accepted IMX night labels | `review_artifacts/imx_paired_night_labeling/imx_paired_night_top2_labels_agent_review.csv`, `diagnostic_model_comparison.csv`, report and ambiguity summary |
| Predictions/features | `artifacts/ml/sensor_domain_shift/raw_sensor_predictions.csv`, `data/sensor_eval/raw_sensor_features.jsonl`, `data/sensor_eval/paired_nir_features.jsonl` |
| Offline reference features | `data/training_v4/cluster_aware_conservative_train_optical_v2_verified.jsonl`, `data/training_v4/cluster_aware_conservative_test_optical_v2_verified.jsonl` |
| BU-TIV local data | `/Users/phongpham/Downloads/atrium-test-orange-nuc.tar`, `/Users/phongpham/Downloads/orange 2/nuc/`, `/Users/phongpham/Downloads/atrium_orange_2d.xml`, `/Users/phongpham/Downloads/atrium-test-orange-raw.tar`, `/Users/phongpham/Downloads/atrium_orange.mp4`, `/Users/phongpham/Downloads/atriumHomographyExample.m` |

## 3. Files created

| Output group | Key files |
| --- | --- |
| Temporary eval scripts | `review_artifacts/eval_scripts/eval_common.py`, `merge_accepted_top2_labels.py`, `evaluate_accepted_top2_env_labels.py`, `accepted_label_domain_shift.py`, `butiv_mad_benchmark.py` |
| Accepted-label merge | `review_artifacts/domain_shift_eval/accepted_sensor_top2_labels_combined.csv`, `accepted_label_merge_report.md`, `accepted_label_class_distribution.csv`, `accepted_label_ambiguity_exclusion_summary.csv` |
| Classifier metrics | `env_classifier_eval_retained_all.*`, `env_classifier_eval_retained_nonambiguous.*`, `env_classifier_eval_high_confidence_labels.*`, `env_classifier_eval_by_subset.csv`, `env_classifier_per_class_metrics.csv`, `env_classifier_confusion_matrix.csv`, `env_classifier_error_cases.csv`, `env_classifier_top2_near_miss_cases.csv` |
| Domain shift | `domain_shift_summary.md`, `domain_shift_feature_summary.csv`, `domain_shift_top_features.csv`, `domain_shift_linked_features.csv` |
| Thesis-ready ML | `review_artifacts/domain_shift_eval/thesis_tables/thesis_ready_ml_table.md`, `review_artifacts/domain_shift_eval/thesis_ready_interpretation.md` |
| BU-TIV preprocess | `butiv_inventory.md`, `butiv_decode_resize_report.md`, `butiv_80x62_manifest.csv`, `frames_80x62/`, `gt_masks_80x62/` |
| BU-TIV MAD | `tune_subset_manifest.csv`, `full_minus_tune_manifest.csv`, `mad_sweep_results.csv`, `mad_selected_config.json`, `mad_selected_config_summary.md`, `mad_benchmark_metrics_selected.json`, `mad_benchmark_metrics_selected.md`, `mad_failure_cases.csv`, `qualitative_examples/` |
| Thesis-ready MAD | `thesis_ready_mad_benchmark_table.md`, `thesis_ready_mad_benchmark_caption.txt`, `thesis_ready_mad_interpretation.md` |

No files under `tools/` were added or intentionally edited in this pass. The worktree already contains dirty `tools/` paths unrelated to this pass: `tools/gen_ch5_thermal_figures.py`, `tools/gen_thesis_thermal_figures.py`, and `tools/offline_pipeline.py`.

## 4. Label acceptance and merge summary

| Split | Total | Retained | Excluded | Ambiguous | High-confidence retained |
| --- | ---: | ---: | ---: | ---: | ---: |
| day_sensor_template | 120 | 104 | 16 | 4 | 104 |
| imx_paired_night | 120 | 99 | 21 | 16 | 83 |
| all | 240 | 203 | 37 | 20 | 187 |

Validation checks passed: labels are from the current nine-class production list, top1/top2 differ, confidence values are in range, top1 confidence is not below top2 confidence, excluded rows carry reasons, ambiguous rows carry notes/cues, and all rows carry accepted provenance.

## 5. Environment classifier evaluation summary

| View | n | Top-1 accuracy | Top-2 hit rate | Balanced accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| retained_all | 203 | 0.3547 | 0.5764 | 0.2441 | 0.2137 | 0.3122 |
| retained_nonambiguous | 183 | 0.3716 | 0.6120 | 0.2981 | 0.2678 | 0.3407 |
| high_confidence_labels | 187 | 0.3636 | 0.6043 | 0.2930 | 0.2645 | 0.3350 |

Subset highlight: `imx_paired_night:high_confidence_labels` has n=83, top-1 accuracy 0.5904, top-2 hit rate 0.8072, balanced accuracy 0.2500, macro-F1 0.1870, weighted-F1 0.4416. The day sensor subset is substantially weaker: `day_sensor_template:high_confidence_labels` has n=104, top-1 accuracy 0.1827 and top-2 hit rate 0.4423.

## 6. Domain-shift findings

The accepted labels now make two sensor-domain slices measurable: day/backlight/glare frames and IMX paired night/low-light frames. Feature linking succeeded for all 203 retained labels.

Highest drift features versus the offline train reference:

| Subset | Feature | KS vs train | Mean delta | Out-of-range | n |
| --- | --- | ---: | ---: | ---: | ---: |
| imx_paired_night | nir_dark_fraction | 0.9799 | 0.6897 | 0.9899 | 99 |
| imx_paired_night | nir_mean_brightness | 0.9725 | -98.8901 | 0.9899 | 99 |
| imx_paired_night | nir_p95 | 0.9451 | -141.4369 | 0.9798 | 99 |
| imx_paired_night | nir_entropy | 0.9171 | -2.1596 | 0.9596 | 99 |
| day_sensor_template | nir_mean_brightness | 0.7732 | -36.0507 | 0.1346 | 104 |

Interpretation: the night subset is strongly shifted toward darkness and lower entropy relative to the offline reference. The classifier's stronger top-2 behavior on IMX night is useful, but the low balanced accuracy and macro-F1 still indicate class imbalance and class-overlap risk.

## 7. BU-TIV inventory and preprocessing

| Item | Result |
| --- | --- |
| Extracted NUC frames | 3,482 PNG frames under `/Users/phongpham/Downloads/orange 2/nuc/` |
| Frame format | 16-bit 512x512 PNG, observed uint16 range example 3082-7139 |
| Annotation XML | `/Users/phongpham/Downloads/atrium_orange_2d.xml`, 3,482 frame entries |
| Object annotations | XML object attrs include `x1`, `x2`, `y1`, `y2` |
| Exact filename/XML matching | 357 exact filename-number matches only |
| Mapping used | `FRAME_MAPPING_ASSUMPTION_order_mapping` |
| Resized artifacts | 13,928 normalized/resized frames: 2 normalization policies x 2 resize variants x 3,482 frames |
| GT masks | 6,964 masks: 2 resize variants x 3,482 frames |
| Qualitative alignment overlays | 20 random overlays generated |

Exact filename-number matching was attempted first. Because the extracted NUC filenames use sampled original frame numbers while XML entries are sequential 1..3482, exact matching was incomplete; order mapping was used and explicitly marked as `FRAME_MAPPING_ASSUMPTION`.

## 8. MAD benchmark design

The benchmark evaluates MAD as a visual-warning/foreground indicator on an external thermal-like dataset resized to SmartBinocular's 80x62 target thermal display resolution.

Design details:
- normalization policies compared: global percentile and per-frame percentile;
- resize variants compared: direct 512x512 -> 80x62 and aspect-preserving horizontal pad -> 80x62;
- mask metrics compared with GT dilation 0 and 1 in the sweep table;
- object/centroid recall reported at 2, 3, and 5 pixel tolerances;
- selected config chosen using tune subset only;
- primary selected-config metrics computed on full-minus-tune heldout frames;
- all-frame metrics are reported separately as including tuning frames.

The first full run with tune-size 420 was stopped because the full sweep was too slow. The final run used a 160-frame tune subset and a 3,322-frame heldout/full-minus-tune set. This still preserves the tune/heldout split and the primary heldout reporting rule.

## 9. MAD selected configuration and metrics

Selected on tune subset only:

| Parameter | Value |
| --- | --- |
| normalization_policy | per_frame_percentile |
| resize_variant | direct |
| gt_dilation | 0 |
| z_threshold | 2.5 |
| min_blob_area | 1 |
| temporal_window | 1 |
| morph_kernel | 0 |
| persistence_frames | 1 |

Primary heldout/full-minus-tune metrics:

| Metric | Value |
| --- | ---: |
| n_frames | 3,322 |
| frame_precision | 0.9905 |
| frame_recall | 0.3772 |
| frame_f1 | 0.5464 |
| object_recall_tol2 | 0.0715 |
| object_recall_tol3 | 0.1523 |
| object_recall_tol5 | 0.2789 |
| pixel_f1 | 0.0829 |
| mean_iou | 0.0294 |
| false_blobs_per_frame | 1.0837 |
| mean_runtime_ms | 0.1527 |
| p95_runtime_ms | 0.2150 |

All-frame selected-config metrics are available in `mad_benchmark_metrics_selected.json` and are explicitly labeled as including tuning frames.

## 10. Positive thesis-ready results

ML/domain shift:
- The accepted labels convert the previous unlabeled sensor-domain audit into a preliminary accepted-label evaluation.
- Top-2 hit rate is meaningfully higher than top-1 accuracy, supporting a robustness/ambiguity interpretation for visually overlapping classes.
- The IMX paired night subset provides the strongest valid positive classifier result: top-2 hit rate 0.8072 on 83 high-confidence retained frames.

MAD:
- BU-TIV adds a labeled external surrogate benchmark at the same 80x62 target resolution used for SmartBinocular thermal display processing.
- Heldout MAD frame precision is high at 0.9905, supporting conservative visual-warning behavior when detections occur.
- Local workstation MAD processing cost is low at 0.1527 ms/frame mean for 80x62 frames.

Use the generated thesis-ready files:
- `review_artifacts/domain_shift_eval/thesis_ready_interpretation.md`
- `review_artifacts/butiv_mad_benchmark/thesis_ready_mad_interpretation.md`

## 11. Weak/negative results retained

The report retains the following limitations rather than hiding them:
- retained_all classifier top-1 accuracy is only 0.3596 and macro-F1 is 0.2201;
- day sensor subset performance is weak, especially compared with the IMX night subset;
- balanced accuracy remains low, indicating poor or unsupported class coverage;
- BU-TIV MAD frame recall is 0.3772 and object recall at 3 px is 0.1523;
- pixel F1 is 0.0829 and mean IoU is 0.0294, so mask-quality claims must remain weak;
- exact BU-TIV XML-to-NUC frame-id matching was incomplete, so annotation alignment depends on `FRAME_MAPPING_ASSUMPTION_order_mapping`;
- BU-TIV is external surrogate data, not MI48 raw/radiometric validation.

## 12. Thesis files updated, if any

No thesis LaTeX files were edited. No files under `HK252-DATN-142/` were intentionally changed in this pass.

Thesis-ready snippets were generated as standalone artifacts only:
- `review_artifacts/domain_shift_eval/thesis_ready_interpretation.md`
- `review_artifacts/domain_shift_eval/thesis_tables/thesis_ready_ml_table.md`
- `review_artifacts/butiv_mad_benchmark/thesis_ready_mad_interpretation.md`
- `review_artifacts/butiv_mad_benchmark/thesis_ready_mad_benchmark_table.md`
- `review_artifacts/butiv_mad_benchmark/thesis_ready_mad_benchmark_caption.txt`

## 13. Remaining risks and recommended next steps

| Risk | Status | Next step |
| --- | --- | --- |
| Label provenance | User-accepted agent-reviewed labels, not gold labels | Add independent human review or multi-annotator adjudication before stronger wording |
| Day sensor classifier behavior | Weak top-1/top-2 metrics | Collect more day/backlight/glare/fog/rain labeled sensor frames and consider retraining/evaluation split |
| Low-support classes | Fog/rain/transition remain weak or absent | Keep support counts visible; avoid across-all-environment claims |
| BU-TIV mapping assumption | Order mapping used | Verify BU-TIV documentation or frame extraction convention; keep overlays in appendix |
| MAD object localization | Low object recall and low mask IoU | Tune with a more detector-like foreground model or annotate MI48-like sequences directly |
| MI48 validation | Not measured | Collect labeled MI48 raw/display thermal sequences for direct validation |

## 14. Exact commands run

Core implementation and evaluation commands:

```bash
python3 -m py_compile review_artifacts/eval_scripts/eval_common.py review_artifacts/eval_scripts/merge_accepted_top2_labels.py review_artifacts/eval_scripts/evaluate_accepted_top2_env_labels.py review_artifacts/eval_scripts/accepted_label_domain_shift.py review_artifacts/eval_scripts/butiv_mad_benchmark.py
python3 review_artifacts/eval_scripts/merge_accepted_top2_labels.py
python3 review_artifacts/eval_scripts/evaluate_accepted_top2_env_labels.py
python3 review_artifacts/eval_scripts/accepted_label_domain_shift.py
python3 review_artifacts/eval_scripts/butiv_mad_benchmark.py
pkill -f 'review_artifacts/eval_scripts/butiv_mad_benchmark.py' || true
python3 -m py_compile review_artifacts/eval_scripts/butiv_mad_benchmark.py
python3 review_artifacts/eval_scripts/butiv_mad_benchmark.py --reuse-preprocessed --tune-size 160
rm -rf review_artifacts/eval_scripts/__pycache__
```

Important validation commands included:

```bash
find review_artifacts/butiv_mad_benchmark/qualitative_examples -type f | wc -l
find review_artifacts/butiv_mad_benchmark/frames_80x62 -type f | wc -l
find review_artifacts/butiv_mad_benchmark/gt_masks_80x62 -type f | wc -l
wc -l review_artifacts/butiv_mad_benchmark/mad_sweep_results.csv review_artifacts/butiv_mad_benchmark/tune_subset_manifest.csv review_artifacts/butiv_mad_benchmark/full_minus_tune_manifest.csv review_artifacts/butiv_mad_benchmark/butiv_80x62_manifest.csv
git status --porcelain=v1 -uall -- review_artifacts/domain_shift_eval review_artifacts/butiv_mad_benchmark review_artifacts/eval_scripts EVAL_DOMAIN_SHIFT_MAD_WORK_REPORT.md HK252-DATN-142 tools src docs data artifacts models tests fusion_captures
```

## 15. Commands skipped and why

| Command/task | Reason |
| --- | --- |
| Thesis LaTeX edits/build | Out of scope for this pass; user requested artifact-first evidence pass and no LaTeX edits |
| RPi4 latency benchmark | Explicitly out of scope |
| Full mode-matrix timing | Explicitly out of scope |
| Hardware field capture | Explicitly out of scope |
| Q1 runtime fusion triples | Explicitly out of scope |
| Production model retraining | Would change model/evaluation scope and risk tuning on accepted labels |
| Writing under `tools/` | Avoided by using `review_artifacts/eval_scripts/` |
| BU-TIV raw tar decode | NUC PNG frames were available and preferred; raw tar kept as inventory/fallback |

## Evidence wording boundary

Allowed:
- preliminary sensor-domain evaluation using user-accepted agent-reviewed labels;
- top-2 hit rate helps interpret visually overlapping classes;
- external BU-TIV thermal benchmark resized to 80x62 provides a surrogate labeled benchmark for MAD indicator behavior;
- parameter-sweep-selected operating point.

Forbidden:
- ground-truth gold labels;
- deployment accuracy proven;
- classifier validated across all environments;
- MAD validated on MI48 raw radiometric data;
- production detector performance proven.
