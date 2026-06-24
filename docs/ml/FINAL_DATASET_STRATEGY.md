# Final Dataset Strategy

Status: current best strategy after raw sensor ingestion. No dataset was mixed
into production training.

| Dataset role | Path | Label status | Use |
| --- | --- | --- | --- |
| Offline train | `data/training_v3/cluster_aware_conservative_train.jsonl` | labeled offline proxy | current best offline train split |
| Offline test | `data/training_v3/cluster_aware_conservative_test.jsonl` | labeled offline proxy | current best offline evaluation split |
| Raw sensor frames | `data/sensor_eval/raw_sensor_features.jsonl` | unlabeled | domain-shift and prediction audit only |
| Raw sensor manual candidates | `artifacts/ml/sensor_domain_shift/manual_label_template.csv` | pending manual labels | future sensor eval subset |
| Feature v2 subset | `data/training_v4/*_optical_v2_verified.jsonl` | labeled subset only | non-production feature ablation |

Rules:

- Raw sensor unlabeled frames must not be used as supervised training labels.
- Pseudo-labeling is not allowed for final metrics unless explicitly marked as
  weak/semi-supervised.
- Manual-labeled raw sensor frames should first be used as a separate sensor
  evaluation set, not mixed into training.
- Production model migration requires explicit confirmation.

## Phase M10 Status

Runtime label validation found no completed manual-label CSV. The current
sensor evaluation set remains:

| Dataset role | Path | Status | Allowed use |
| --- | --- | --- | --- |
| Raw sensor unlabeled features | `data/sensor_eval/raw_sensor_features.jsonl` | unlabeled / preliminary | domain-shift and RGB-scaler proxy prediction audit |
| Manual label template | `artifacts/ml/sensor_domain_shift/manual_label_template.csv` | 120 rows, 0 completed labels | user/manual labeling only |
| Sensor-labeled eval | `docs/tables/ml/raw_sensor_labeled_eval.md` | not measured | no accuracy claim |
| Feature v2/21 verified subset | `data/training_v4/*_optical_v2_verified.jsonl` | 4-class verified-image subset | research ablation only |

No raw sensor frame, pseudo-label, feature-v2 row, or manual-label candidate is
mixed into supervised training in Phase M10.

## Phase M11 Suggested-Label Dataset Rules

Phase M11 creates a review accelerator, not a new supervised dataset.

| Artifact | Label layer | Allowed use | Forbidden use |
| --- | --- | --- | --- |
| `artifacts/ml/sensor_domain_shift/suggested_labels.csv` | `suggested_label` | review prioritization and template prefill | ground truth, thesis accuracy, train/test mixing |
| `artifacts/ml/sensor_domain_shift/manual_label_template_autofilled.csv` | pending human confirmation | user review and correction | automatic evaluation unless user fills `manual_label` or confirms `accept_suggested_label=yes` |
| `data/sensor_eval/weak_labeled_sensor_subset.jsonl` | not created | none in M11 | pseudo-ground-truth use |

`auto_weak_label` remains reserved for a future independent teacher plan. If the
user later approves CLIP/VLM/local pretrained labeling, that plan must document
model provenance, mapping, confidence thresholds, and leakage-safe dataset use
before any weak-labeled dataset is generated.

## Phase P12 Paired Dataset Rules

Paired capture data is a separate target-domain evidence set:

| Dataset role | Path | Label status | Allowed use |
| --- | --- | --- | --- |
| Paired ML manifest | `artifacts/paired_eval/paired_ml_manifest.csv` | unlabeled | pairing/session/modality audit |
| Paired NIR features | `data/sensor_eval/paired_nir_features.jsonl` | unlabeled | domain-shift and prediction audit only |
| Paired manual template | `artifacts/paired_eval/manual_label_template_paired.csv` | pending manual labels | user/manual labeling only |
| Paired labeled eval | `docs/tables/ml/paired_sensor_labeled_eval.md` | not measured | no metric until trusted labels exist |

Paired data must not be mixed into official training or test sets until a
leakage-safe split strategy, trusted labels, and explicit user confirmation
exist. Sidecar labels count as ground truth only if the label is taxonomy-valid,
`label_source` is trusted, and `label_confidence >= 0.8`.

## Phase 1 Evidence Freeze Dataset Boundaries

| Dataset / artifact | Role | Training use | Evaluation use | Status |
| --- | --- | --- | --- | --- |
| `data/training_v3/cluster_aware_conservative_{train,test}.jsonl` | current best offline split | yes, offline benchmark only | offline optical RGB-proxy | caveated |
| `data/sensor_eval/raw_sensor_features.jsonl` | raw sensor domain-shift set | no | drift/prediction only | unlabeled |
| `data/sensor_eval/paired_nir_features.jsonl` | paired sensor domain-shift set | no | drift/prediction only | unlabeled |
| `artifacts/ml/final_labeling_package/agent_manual_labels.csv` | agent-reviewed label subset | no | preliminary review/eval only | not user-confirmed |
| `artifacts/ml/sensor_domain_shift/manual_label_template_autofilled.csv` | user labeling template | no | future gold-label eval if completed | pending user |
| `artifacts/paired_eval/manual_label_template_paired.csv` | paired user labeling template | no | future paired gold-label eval if completed | pending user |

Pseudo-labels, suggested labels, and agent labels must not be mixed into
official train/test splits without explicit approval and a leakage-safe split
design. User-confirmed labels remain the gold standard for final sensor-real
accuracy.
