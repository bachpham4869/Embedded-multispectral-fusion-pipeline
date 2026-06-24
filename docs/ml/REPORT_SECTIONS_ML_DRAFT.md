# Draft Report Sections: ML Evidence Freeze

Status: draft text for Agent A4. The wording is intentionally conservative and
should be copied into LaTeX only with artifact references preserved.

## 1. Dataset Distribution and Leakage Control

The ML evaluation uses an offline optical RGB-proxy dataset with the current
production-compatible `optical_12_baseline` feature set. The final offline
evaluation split is a duplicate-cluster-aware group split, not a random split.
The split contains 11,981 training rows and 2,113 test rows, with nonzero
support for all nine ENV classes. Leakage checks report zero exact JSON-row,
file-SHA, split-group, duplicate-cluster, and dHash near-duplicate overlap
within available metadata/hash coverage. Source-name overlap remains across
train and test, so this split should not be described as source-held-out.

Use artifact: `docs/tables/ml/cluster_aware_leakage_summary.md`.

## 2. Duplicate-Cluster-Aware Split

The duplicate-cluster-aware split addresses the earlier concern that
feature-vector or visually near-duplicate records could inflate held-out
metrics. The split groups records by duplicate cluster and split group before
assignment. This reduces the strongest leakage risks, but the evidence remains
offline and proxy-domain because the underlying training set is not live
NIR/LWIR capture.

Recommended report claim: "offline optical RGB-proxy baseline with leakage
controls," not "validated NIR classifier."

## 3. Model Comparison and Selection

Random Forest remains justified as the current production-compatible baseline.
On the duplicate-cluster-aware split, RF200 reaches accuracy 0.8263, balanced
accuracy 0.7463, and macro-F1 0.7362. RF100 is very close, with accuracy
0.8230, balanced accuracy 0.7415, and macro-F1 0.7325, while using a smaller
model artifact. Therefore, RF200 should be presented as the accuracy baseline,
and RF100 should be presented as the lightweight tree candidate pending a real
Raspberry Pi 4 latency run.

The MLP family is not recommended for current production because balanced
accuracy and macro-F1 remain more than 3 percentage points below RF200, even
though proxy latency is lower. Feature v2 and the still-compatible 21-derived
candidate remain research-only because their fair comparison is limited to a
4-class verified subset.

Use artifacts: `docs/tables/ml/model_comparison_cluster_aware.md`,
`docs/tables/ml/mlp_variant_comparison.md`, and
`docs/tables/ml/feature_set_comparison_fair.md`.

## 4. Raw and Paired Sensor Domain Shift

Raw sensor evaluation uses sampled frames from `test_30fps_morning.mp4`.
Predictions are explicitly `RGB-scaler proxy inference`, and no sensor accuracy
is claimed because user-confirmed labels are absent. The raw sensor prediction
distribution is broad, with high abstention, which indicates a meaningful
domain shift from the offline training data.

Paired-data evaluation uses 584 strict paired IMX/thermal-display rows. This
evidence is also unlabeled. The exact inference wording is `RGB-scaler proxy
inference, not validated NIR classifier accuracy`. The production RF predicts
mostly `normal_night`, which is useful as a domain-shift signal but not as an
accuracy metric. The IMX stream remains `unknown_optical` until user modality
confirmation.

Use artifacts: `docs/tables/ml/raw_sensor_prediction_summary.md`,
`docs/tables/ml/paired_nir_prediction_summary.md`, and
`docs/tables/paired/paired_ml_evidence_summary.md`.

## 5. Agent/Manual-Reviewed Subset

A small agent-reviewed subset was created from visually inspectable raw and
paired contact sheets. The subset has 24 rows, with 22 rows at
`label_confidence >= 0.8`. The label source is `agent_manual_label`, not
user-confirmed gold labels. This subset is useful for accelerated review and
for demonstrating that RGB-proxy inference may misclassify backlit sensor
frames, but it must not be used as a final sensor-real performance claim.

The preliminary agent-labeled evaluation shows very low agreement with the
production RF on this deliberately high-risk subset. This should be reported as
a limitation and motivation for user-confirmed labels, not as final model
accuracy.

Use artifacts: `artifacts/ml/final_labeling_package/agent_manual_labels.csv`,
`docs/tables/ml/agent_labeled_sensor_eval.md`, and
`docs/figures/ml/agent_manual_label_contact_sheet.png`.

## 6. Class Decisions

The class taxonomy remains unchanged in code. The report-facing decision is:

- Keep `normal_day`, `normal_night`, `night_clear`, `fog`, and `rain` from
  offline evidence, with sensor-validation caveats.
- Keep `nir_night` because it is deployment-relevant, but do not claim NIR
  validation until modality and night labels are confirmed.
- Keep `glare` and `backlight` provisional because they correspond to useful
  processing policies and appear in high-risk sensor review, but require more
  targeted labels.
- Treat `transition` as a weak/provisional classifier class and a stronger
  candidate for a runtime transient state such as `dawn_dusk_blend`.

Use artifact: `docs/tables/ml/class_decision_summary.md`.

## 7. Limitations and Future Work

The thesis should explicitly state that offline metrics are for an optical
RGB-proxy baseline, not a live NIR/LWIR validated classifier. Final sensor-real
accuracy requires user-confirmed labels. Final deployment speed requires
Raspberry Pi 4 feature+predict latency. Feature v2/21 production migration
requires schema versioning, full-class coverage, and explicit user approval.
