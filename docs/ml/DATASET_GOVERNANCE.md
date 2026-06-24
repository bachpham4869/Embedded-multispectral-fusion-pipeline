# Dataset Governance

Status: Phase 3 governance note for the optical RGB-proxy baseline.

## Metadata Status Rules

Feature-vector or feature-hash matching is not image identity evidence. It is
only a weak hint for investigation.

`metadata_status` values:

- `verified`: the record maps to raw image identity through deterministic
  path/filename/frame replay or a file SHA256 identity.
- `inferred_low_confidence`: raw images exist, but the current JSONL lacks the
  original path/hash and the source order was shuffled or subsampled.
- `unresolved`: raw image identity cannot be recovered.

For `inferred_low_confidence` rows, candidate paths are kept separate from
verified `file_sha256` and `dhash` fields so later leakage scripts do not treat
them as original-image proof.

## Phase 3 Dataset Version

Generated non-production files:

- `data/training_v2/merged_logs_ml_metadata.jsonl`
- `data/training_v2/from_logs_train_metadata.jsonl`
- `data/training_v2/from_logs_test_metadata.jsonl`
- `data/training_v2/group_aware_train.jsonl`
- `data/training_v2/group_aware_test.jsonl`

Manifest files:

- `data/training_v2/manifest.json`
- `data/training_v2/group_aware_split_manifest.json`

These files do not replace `data/training/*.jsonl`.

## Current Coverage

`data/training_v2/manifest.json` reports 14,094 rows in the enriched merged
dataset:

- `verified`: 5,791 rows
- `inferred_low_confidence`: 8,303 rows
- `file_sha256` coverage: 5,791 rows
- `dhash` coverage: 5,791 rows

The verified rows are from `offline_mwd`, `offline_weather11`, and
`offline_weather_time`. Shuffled/subsampled sources remain
`inferred_low_confidence`.

## Split Policy

Phase 3 uses a group-aware split, not a source-held-out split. Verified records
use `file_sha256::<sha>` as `split_group_id`, so exact duplicate files cannot
cross train/test when the group-aware split is used.

Remaining source-name overlap is reported separately and is not hidden.

## Artifact Policy

Large benchmark `model.joblib` artifacts are not staged or committed without
explicit confirmation. Phase 3 benchmark metrics store in-memory serialized
model SHA256 values when model files are not persisted.

## Phase 4 Dataset Version

Generated non-production files:

- `data/training_v3/merged_logs_ml_clustered_strict.jsonl`
- `data/training_v3/merged_logs_ml_clustered_conservative.jsonl`
- `data/training_v3/cluster_aware_strict_train.jsonl`
- `data/training_v3/cluster_aware_strict_test.jsonl`
- `data/training_v3/cluster_aware_conservative_train.jsonl`
- `data/training_v3/cluster_aware_conservative_test.jsonl`

Manifest files:

- `data/training_v3/duplicate_cluster_manifest_strict.json`
- `data/training_v3/duplicate_cluster_manifest_conservative.json`
- `data/training_v3/cluster_aware_split_manifest_strict.json`
- `data/training_v3/cluster_aware_split_manifest_conservative.json`

These files do not replace `data/training/*.jsonl`.

## Phase 4 Split Policy

Duplicate clusters are built over the full metadata-enriched merged dataset.
The Phase 3 cross-split dHash CSV is diagnostic only and is not used as the
complete cluster graph.

Two cluster modes are recorded:

- `strict`: dHash Hamming distance <= 2
- `conservative`: dHash Hamming distance <= 4

The conservative split is the Phase 4 benchmark split because it preserves all
ENV classes in train/test and keeps the test ratio inside 10-25%.

The Phase 4 split is a duplicate-cluster-aware group split, not source-held-out.
It has 0 duplicate-cluster overlap and 0 split-group overlap, but 8 overlapping
source names.
## Phase M5 Raw Sensor Governance

Raw sensor capture `data/raw_sensor_captures/test_30fps_morning.mp4` is treated
as immutable raw input. Sampled frames and manifests are non-production
artifacts under `data/sensor_eval/` and `artifacts/ml/sensor_domain_shift/`.

Rules added in Phase M5:

- Raw sensor frames are unlabeled until the user fills a manual label template.
- Raw sensor predictions are `RGB-scaler proxy inference`; they are not accuracy
  labels.
- `data/sensor_eval/raw_sensor_features.jsonl` may be used for domain-shift
  analysis only.
- `data/training_v4/*.jsonl` is non-production feature-ablation data and must
  not be staged if large without explicit confirmation.
- Manual-labeled sensor frames should become a separate eval set first, not
  training data.

## Phase M10 Label And Feature Governance

- Manual-label files are re-scanned at runtime. The newest completed CSV with
  valid taxonomy labels is selected; templates with empty `manual_label` values
  do not produce accuracy metrics.
- Current status: no completed manual labels found, so raw sensor accuracy is
  `not measured`.
- `optical_21_candidate_still` excludes temporal-only fields and currently has
  20 supervised features. It must not be described as the full temporal
  21-feature set.
- Feature-v2/21 benchmark evidence is limited to a verified-image 4-class
  subset and must not be compared as same-condition evidence against full
  9-class `optical_12_baseline` metrics.
- RPi4 latency remains protocol-only until `tools/rpi4_model_latency_benchmark.py`
  is run on target hardware.

## Phase M11 Suggested-Label Governance

- `suggested_label` means RF/heuristic review support only. It is not a
  stronger model, not an independent teacher, and not ground truth.
- `auto_weak_label` is reserved for a future independent teacher path such as
  CLIP/VLM/local pretrained model after explicit user approval.
- `manual_label` remains the only raw-sensor ground truth for accuracy/F1
  claims. `accept_suggested_label=yes` is treated as ground truth only when the
  user explicitly fills it in the review CSV.
- No weak-labeled sensor subset is created in M11 because no independent teacher
  is locally available.

## Phase P12 Paired Data Governance

- `data/paired_data/` is immutable raw paired input.
- Pairing tiers use the Agent 2 vocabulary: `frame_strict`,
  `time_strict_100ms`, `protocol_strict_1s`, `near_paired`, `weak_paired`,
  and `unpaired`.
- Paired ML rows must carry `pairing_tier`, `session_id`, `frame_idx`,
  `nir_modality`, `thermal_modality`, and `label_source`.
- `data/sensor_eval/paired_nir_features.jsonl` is non-production domain-shift
  evidence and must not be mixed into training without approval.
- Sidecar `env_label` values are ground truth only when taxonomy-valid,
  `label_source` is trusted, and `label_confidence >= 0.8`.
- Current paired capture has 584 strict rows and 0 trusted labels, so paired
  labeled evaluation is `not measured`.

## Phase 1 Agent-Reviewed Label Governance

- Agent-reviewed labels use `label_source=agent_manual_label` and require
  `label_confidence`, visual evidence notes, and contact-sheet/frame
  provenance.
- Agent labels are accelerated review labels, not user-confirmed gold labels.
- Agent labels may support preliminary sensor-evaluation and limitation
  discussion, but must not be mixed into official train/test splits without
  explicit approval and a leakage-safe split design.
- Final sensor-real accuracy still requires user-confirmed manual labels.
