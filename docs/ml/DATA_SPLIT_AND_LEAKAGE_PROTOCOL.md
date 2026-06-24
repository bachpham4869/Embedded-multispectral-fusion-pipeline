# Data Split and Leakage Protocol

Status: Phase 2 protocol. Current JSONL files allow metadata-level checks only.

## Goal

Prove that reported train/test metrics are not inflated by duplicate frames,
near-duplicate images, shared sessions, or source-specific shortcuts. If the
required evidence is missing, the metric must be marked `preliminary / not
thesis-ready`.

## Required Metric Provenance

Each classification metric must include command, git commit, model path and
SHA256, dataset path and hash/manifest, train/test rows, feature set version,
split method, random seed, Python/sklearn/numpy/scipy versions, per-class
support, raw confusion matrix, and normalized confusion matrix.

## Split Rules

- Do not tune hyperparameters on `from_logs_test.jsonl`.
- Prefer source/session-grouped validation when source images or sessions are
  available.
- Keep a frozen held-out test file for final reporting, but only after duplicate
  and near-duplicate checks pass.
- For live RPi logs, split by session rather than individual frame to avoid
  adjacent-frame leakage.
- Record the split command and random seed. If split files were produced earlier
  and the command is unknown, mark metrics as preliminary.

## Leakage Checks

Implemented command:

```bash
.venv/bin/python tools/check_dataset_leakage.py \
  --train data/training/from_logs_train.jsonl \
  --test data/training/from_logs_test.jsonl \
  --out docs/tables/ml/leakage_check_summary.md \
  --feature-overlap-details-md docs/tables/ml/feature_vector_overlap_details.md \
  --feature-overlap-details-csv artifacts/ml/leakage/feature_vector_overlaps.csv
```

Checks:

- source overlap
- path overlap when path-like metadata exists
- filename overlap when path-like metadata exists
- session/frame overlap when session/frame metadata exists
- exact JSON-row hash overlap
- complete optical_12 feature-vector hash overlap as a proxy warning
- optional average-hash near-duplicate check when resolvable image paths exist

## Current Limitation

The current JSONL does not contain source image path metadata or session IDs.
Therefore:

- exact image SHA256 duplicate check cannot be completed
- perceptual-hash near-duplicate check cannot be completed
- session/frame leakage cannot be completed
- path overlap cannot be interpreted as a real image-level guarantee

Current leakage result is documented in
`docs/tables/ml/leakage_check_summary.md`. Correct wording:

- no exact JSON-row overlap found
- 21 unique test records with optical_12 feature-vector overlap found
- 22 train/test overlap pairs because one test vector matches two train rows
- image-level and near-duplicate leakage remain unverified because JSONL lacks
  original image path/hash metadata

These overlaps may be benign repeated feature values, but they require
source-image inspection before frozen held-out metrics are thesis-ready.

## Required Metadata For Future JSONL

Add these fields where possible:

- `source_dataset`
- `original_image_path` or `relative_image_id`
- `file_sha256`
- optional `perceptual_hash`
- `split_group_id`
- `session_id` if live log or video-derived
- `frame_idx`
- `capture_ts`
- `capture_device`
- `sensor_id`
- `labeler_id` when manual labels are used
- `label_source`
- `label_confidence`
- `nir_channel`
- `thermal_channel`

For live RPi logs, `split_group_id` should group frames by session/capture run
so adjacent frames cannot cross train/test.

## Acceptance Criteria

A metric can be called thesis-ready only if:

- no exact image-hash overlap exists across train/test
- no unresolved near-duplicate pHash cluster crosses train/test
- no session/frame overlap crosses train/test
- source overlap is either absent or justified by a source-aware validation run
- low-support classes have confidence intervals
- domain-shift status is explicit

## Phase 2 Sensitivity Result

`tools/evaluate_frozen_classifier.py` evaluates the frozen production bundle
read-only. It accepts `--train` only to identify feature-vector overlaps and
metadata limitations; it never refits or recalibrates the model.

Phase 2 result:

- original held-out: 2113 rows, balanced accuracy 0.9281, macro-F1 0.9164
- excluding 21 feature-overlap test rows: 2092 rows, balanced accuracy 0.9265,
  macro-F1 0.9149

This small change does not explain the gap versus sidecar CV balanced accuracy
around 0.7408. The likely remaining issues are source overlap, split difficulty,
and unverified image-level duplicate status.

## Phase 3 Group-Aware Split And Image-Level Screening

Phase 3 added metadata-enriched non-production JSONL under `data/training_v2/`.
The metadata status rules are:

- `verified`: path/filename/frame identity can be replayed, or file SHA256 is
  available.
- `inferred_low_confidence`: raw images exist but source order is shuffled or
  subsampled and original JSONL lacks path/hash identity.
- `unresolved`: raw identity cannot be recovered.

Feature-vector/hash matching remains a weak hint only and is not treated as
image identity evidence.

Current split after enrichment:

- file SHA256 coverage: 4,923 train rows and 868 test rows
- exact JSON-row overlap: 0
- exact file SHA256 overlap: 22 pairs
- dHash near-duplicate screening pairs at Hamming threshold <= 4: 1,419
- source-name overlap: 8 sources

Group-aware split:

- train rows: 11,981
- test rows: 2,113
- group field: `split_group_id`
- group overlap count: 0
- exact file SHA256 overlap: 0
- feature-vector weak-hint overlap: 0
- dHash near-duplicate screening pairs at Hamming threshold <= 4: 1,313
- source-name overlap remains: 8 sources

This is not a source-held-out split. The correct wording is `group-aware split`;
source overlap must be reported separately.

dHash is a screening method only. If a report says no pair was found, that claim
is limited to records with dHash coverage, the dHash method, and the configured
threshold. Phase 3 did find dHash screening pairs, so metrics remain
`preliminary / not thesis-ready`.

## Phase 4 Duplicate-Cluster-Aware Split

Phase 4 builds duplicate clusters over the full
`data/training_v2/merged_logs_ml_metadata.jsonl` dataset. The Phase 3
cross-split `near_duplicate_pairs.csv` is diagnostic only and is not the full
cluster source.

Cluster modes:

- `strict`: dHash Hamming distance <= 2
- `conservative`: dHash Hamming distance <= 4

Cluster edges are recorded separately as exact file SHA duplicates or dHash
screening edges. dHash edges are not automatically treated as certain leakage;
they are used to form conservative split groups.

The Phase 4 benchmark uses the conservative duplicate-cluster-aware group split
because it preserves every ENV class in train/test and keeps the test ratio
inside 10-25%.

Conservative split check:

- duplicate-cluster overlap: 0
- split-group overlap: 0
- exact file SHA overlap: 0 within file-hash coverage
- dHash cross-split pairs at threshold <= 4: 0 within dHash coverage
- source-name overlap: 8 sources

This can support offline optical RGB-proxy classifier metrics with caveats. It
is still not a source-held-out split and not live NIR/LWIR validation.
