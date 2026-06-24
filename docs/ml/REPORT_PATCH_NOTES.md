# Report Patch Notes

Status: text guidance for thesis/report updates after Phase 2.

## Replace Overclaims

Use:

- "optical RGB-proxy baseline"
- "optical-only proxy baseline"
- "12 handcrafted optical proxy features"
- "metadata-level leakage check"
- "preliminary frozen held-out evaluation"

Avoid until evidence exists:

- "NIR-trained classifier"
- "NIR sensor validated"
- "production-ready ML accuracy"
- "glare/backlight/transition are robustly solved"
- "RPi4 deployment latency proven" from macOS runs

## Sections To Add

1. Environment Taxonomy and Processing Buckets
2. Dataset Composition and Class Distribution
3. Data Split and Leakage Protocol
4. Feature Set Versioning and Guardrails
5. Classifier Selection and Quantitative Comparison
6. Metric Discrepancy Investigation
7. Domain Shift Evaluation for Real Sensor Deployment
8. Limitations and Future Work

## Metrics Caveat

Any metric table in the thesis must include command, git commit, model SHA,
dataset manifest/hash, train/test row counts, feature-set version, split method,
random seed, Python/sklearn/numpy/scipy versions, per-class support, and raw +
normalized confusion matrix for classification metrics.

If this provenance is missing, add `preliminary / not thesis-ready` beside the
number.

## Current Phase 1 Findings To Mention

- Train/test/reference are RGB proxy and no thermal.
- Train support is imbalanced: max/min class ratio about 7.55.
- Low-support classes: `glare`, `backlight`, `transition`.
- `glare` and `backlight` each come from one dominant source family.
- `transition` comes from Weather-Time Dawn/Dusk and MWD sunrise and should be
  treated as a weak/provisional class.
- Metadata-level leakage check found no exact JSON-row overlap but found 21
  optical_12 feature-vector hash overlaps and lacks image path/session metadata.
- Frozen held-out metrics are not primary thesis evidence until discrepancy is
  explained.

## Current Phase 2 Findings To Mention

- Leakage wording must be cautious: no exact JSON-row overlap found, but 21
  unique test records / 22 train-test pairs have exact `optical_12_baseline`
  feature-vector overlap.
- Image-level and near-duplicate leakage remain unverified because current JSONL
  lacks original image path/hash metadata.
- Excluding feature-vector-overlap test records barely changes frozen held-out
  metrics: balanced accuracy moves from 0.9281 to 0.9265 and macro-F1 from
  0.9164 to 0.9149. This does not explain the discrepancy against sidecar CV.
- The same-feature quick benchmark found RF variants strongest under the
  current split, with RF100 and RF200 close. This is engineering evidence, not
  thesis-ready proof.
- All benchmark latency is macOS proxy latency, not Raspberry Pi 4 CPU
  deployment latency.

## Future JSONL Metadata Requirement

Future JSONL records should include `source_dataset`, `original_image_path` or
`relative_image_id`, `file_sha256`, optional `perceptual_hash`,
`split_group_id`, `session_id` for live/video logs, `capture_device`,
`nir_channel`, `thermal_channel`, `label_source`, and `label_confidence`.

Do not claim "NIR sensor validated" until live IMX290/NIR labeled logs and a
domain-shift report exist.

## Phase 3 Notes

Add:

- "Phase 3 rebuilt metadata-enriched non-production JSONL under
  `data/training_v2/`; original `data/training/*.jsonl` files were not
  overwritten."
- "Metadata identity has three statuses: `verified`,
  `inferred_low_confidence`, and `unresolved`."
- "Feature-vector/hash matches are weak hints only, not original-image identity
  evidence."
- "The Phase 3 split is a group-aware split, not a source-held-out split;
  source-name overlap remains and is reported."
- "Exact file SHA256 overlap in verified coverage was reduced from 22 pairs on
  the enriched current split to 0 pairs on the group-aware split."
- "dHash near-duplicate screening still finds 1,313 cross-split pairs at
  Hamming threshold <= 4, so group-aware benchmark metrics remain preliminary."
- "Focused group-aware benchmark keeps RF100 as the strongest balanced-accuracy
  lightweight candidate, while RF200 remains close."

Avoid:

- "No visual duplicates exist."
- "Source-held-out evaluation."
- "Thesis-ready final classifier metric."
- "RPi4 deployment latency proven."

## Phase 4 Notes

Add:

- "Phase 4 built duplicate clusters over the full metadata-enriched merged
  dataset, using strict dHash <= 2 and conservative dHash <= 4 modes."
- "The conservative duplicate-cluster-aware group split preserves all ENV
  classes in train/test and has 0 exact row, file SHA, split-group,
  duplicate-cluster, and dHash cross-split pairs within hash-covered records."
- "This is still not source-held-out; source-name overlap remains across 8
  source families."
- "The Phase 4 benchmark is now usable as offline optical RGB-proxy evidence
  with caveats, but it is not live NIR/LWIR validation."
- "RF200 has the strongest balanced accuracy in the Phase 4 focused benchmark;
  RF100 remains close and is the lighter candidate for a future migration
  experiment."

Avoid:

- "No visual duplicates exist in the dataset."
- "The dataset is source-held-out."
- "The classifier is NIR-trained or NIR-validated."
- "MacOS latency proves Raspberry Pi 4 CPU 20 FPS deployment."

## Fusion/Image Processing Evidence Hardening

- Fusion/image-processing tables now carry evidence-tier and caveat columns.
- Paired input evidence now includes `584` frame-strict NIR/thermal rows.
- Current fusion-quality evidence is still caveated/preliminary for runtime validation because `captured_runtime_fusion_available=false`.
- Fusion comparisons from paired data are `paired_generated_fusion`, not runtime-captured fusion output.
- Runtime timing from the paired dataset covers capture cadence/skew; per-stage NIR/thermal/fusion latency is not measured in this paired source.
- Failure-case mining is diagnostic; generated/proxy failures are not proof of runtime fusion failure.

## Phase M5-M9 Patch Notes

- Added raw sensor inventory for `test_30fps_morning.mp4`: 1280x720 H.264,
  about 30 FPS, 35,364 frames, 1,178.79 seconds, 1.386 GB.
- Extracted 590 sampled frames at 0.5 FPS with file SHA256/dHash manifest and a
  modality review contact sheet. Modality remains `unknown optical` until user
  confirmation.
- Extracted 590 unlabeled `optical_12_baseline` feature rows and ran domain
  shift against Phase 4 train/test.
- Raw sensor predictions are explicitly `RGB-scaler proxy inference`; no sensor
  accuracy is claimed.
- Raw sensor abstention at `tau1=0.62` is 0.5593, mean confidence is 0.6065,
  and mean posterior entropy is 1.4628.
- Added manual labeling package with 120 candidate frames and a CSV template.
- Benchmarked MLP family over seeds 42, 43, and 44. MLP is rejected for current
  production because balanced accuracy and macro-F1 trail RF by more than 3
  percentage points despite lower latency.
- Added non-production `optical_v2_candidate` and `optical_21_candidate`
  rationale. `optical_v2_candidate` shows a small four-class subset gain;
  `optical_21_candidate` is deferred because temporal labels are unavailable in
  still-image train/test rows.

## Phase M10 Patch Notes

- Added runtime manual-label validation. Current result: no completed manual
  labels found; raw sensor labeled evaluation is `not measured`.
- Fixed raw-sensor prediction/template timestamp propagation to use video
  `timestamp_sec` instead of internal feature-extractor `ts=0.0`.
- Regenerated the 120-frame manual labeling package and contact sheet. Model
  predictions remain `RGB-scaler proxy inference`, not ground truth.
- Split the 21-feature discussion into `optical_21_candidate_still` and
  `optical_21_candidate_temporal`. The still-compatible candidate has 20 actual
  supervised features after excluding `temporal_brightness_std`.
- Reran fair subset feature-set comparison on the same verified 4-class subset.
  Results are subset/preliminary and are not same-condition evidence against the
  full 9-class baseline.
- Added RPi4 latency protocol/script covering feature extraction, model
  inference, feature+predict latency, load time, model size, and memory.
- Production recommendation remains unchanged: RF200 baseline, RF100 lightweight
  tree candidate, MLP rejected for current production, v2/21 research-only, no
  migration without manual labels, RPi4 timing, schema versioning, and user
  confirmation.

## Phase M11 Patch Notes

- Audited local labeling-support options. No CLIP/OpenCLIP/timm/ONNX/TFLite/API
  independent teacher is available in the current environment.
- Added `suggested_label` support from RF/heuristic rules for manual review and
  active-labeling priority. This is explicitly not `auto_weak_label`, not an
  independent teacher, and not ground truth.
- Generated `artifacts/ml/sensor_domain_shift/suggested_labels.csv` for 590 raw
  sensor frames and `manual_label_template_autofilled.csv` for 120 review
  frames.
- Added consistency analysis between RF and heuristics. This is not accuracy and
  not independent reliability because the two sources share feature signals.
- No weak-labeled sensor subset was created. Sensor-labeled accuracy remains
  `not measured` until human-confirmed labels exist.

- Strict paired rows: `584`
- Pairing distribution: `{'frame_strict': 584}`
- Thesis claim level: `strict paired offline fusion evaluation; fusion candidates are generated offline, not runtime-captured fusion`
- Thermal caveat: `thermal_modality=display_heatmap_like` means the thermal video is not raw radiometric thermal.
- Generated fusion caveat: `paired_generated_fusion` is evidence for offline fusion algorithms on real paired inputs, not proof of runtime-captured fusion output.
- NIR summary: `docs/tables/fusion/strict_paired_nir_quality_summary.md`
- Thermal summary: `docs/tables/fusion/strict_paired_thermal_quality_summary.md`
- Fusion summary: `docs/tables/fusion/strict_paired_fusion_quality_summary.md`
- Failure summary rows: `17`

## Phase P12 Paired ML Patch Notes

- Added paired ML inventory and manifest for `data/paired_data/`: 584
  `frame_strict` rows.
- Extracted 584 paired IMX `optical_12_baseline` feature rows into
  `data/sensor_eval/paired_nir_features.jsonl`.
- Paired thermal video is documented as `display_heatmap_like`, not raw
  radiometric thermal.
- Paired RF inference is explicitly `RGB-scaler proxy inference, not validated
  NIR classifier accuracy`.
- Paired prediction distribution is `normal_night`: 579 and `transition`: 5,
  with abstention 0.0086 at tau1=0.62.
- No trusted paired labels were found, so paired sensor labeled evaluation is
  `not measured`.
- Created paired manual-label template and side-by-side contact sheet for human
  review.

## Phase 1 ML Evidence Freeze Notes

- Added `docs/ml/ML_EVIDENCE_READINESS.md` and
  `docs/tables/ml/ml_evidence_readiness_matrix.md` so Agent A4 can separate
  thesis-ready, caveated, preliminary, and not-measured ML evidence.
- Created a small agent-reviewed label subset from visually inspectable raw and
  paired contact sheets. Label source is `agent_manual_label`, not user gold
  labels.
- Ran `agent-labeled sensor subset evaluation` as preliminary evidence only.
  The low agreement with production RF is treated as a domain-shift warning,
  not final sensor-real performance.
- Froze current model decision: RF200 baseline, RF100 lightweight candidate,
  MLP rejected for current production, v2/21 research-only.
- Froze current class decision: keep code taxonomy unchanged; treat
  `transition` as runtime-transient candidate and `glare`/`backlight`/
  `nir_night` as caveated/provisional until user-confirmed labels exist.
- Added `docs/ml/REPORT_SECTIONS_ML_DRAFT.md` for A4 LaTeX integration.

- Strict paired rows: `584`
- Pairing distribution: `{'frame_strict': 584}`
- Thesis claim level: `strict paired offline fusion evaluation; fusion candidates are generated offline, not runtime-captured fusion`
- Thermal caveat: `thermal_modality=display_heatmap_like` means the thermal video is not raw radiometric thermal.
- Generated fusion caveat: `paired_generated_fusion` is evidence for offline fusion algorithms on real paired inputs, not proof of runtime-captured fusion output.
- NIR summary: `docs/tables/fusion/strict_paired_nir_quality_summary.md`
- Thermal summary: `docs/tables/fusion/strict_paired_thermal_quality_summary.md`
- Fusion summary: `docs/tables/fusion/strict_paired_fusion_quality_summary.md`
- Failure summary rows: `17`
## Phase P13 Paired Fusion/Image Final Evidence

- Finalized paired fusion/image evidence with `584` frame-strict NIR/thermal rows.
- `captured_runtime_fusion_available=false`; fusion comparisons are `paired_generated_fusion`, not runtime-captured fusion validation.
- Thermal video remains `display_heatmap_like`, not raw radiometric thermal.
- Added report-ready summary, evidence readiness matrix, per-bucket report summary, failure/limitation table, and fusion report-section draft.
- Entropy decreases are described as ambiguous and are not used as standalone quality proof.
- Rain temporal median and dawn/dusk blend remain `not measured` without explicit supporting evidence.
