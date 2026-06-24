# Paired Data ML Audit

Status: paired data is used for ML domain-shift, confidence, abstention, prediction distribution, and manual labeling evidence.

No paired-data accuracy, model-selection-by-accuracy, training integration, production migration, schema migration, or class migration is allowed without trusted labels and user confirmation.

| Field | Value |
| --- | --- |
| pair_rows | 584 |
| trusted_label_rows | 0 |
| skipped_feature_rows | 0 |
| label_policy | taxonomy-valid + trusted source + label_confidence >= 0.8 |
| inference_scope | RGB-scaler proxy inference, not validated NIR classifier accuracy |
