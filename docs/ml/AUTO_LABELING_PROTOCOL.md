# Auto-Labeling Protocol

Label layers are intentionally separated:

- `suggested_label`: RF/heuristic review aid. It is not ground truth and not an independent teacher.
- `auto_weak_label`: reserved for a future independent teacher such as CLIP/VLM/local pretrained model after user approval.
- `manual_label`: the only ground truth for raw sensor accuracy.

Current phase output is limited to `suggested_label`, review priority, and consistency analysis. Agreement between RF and heuristic is not independent confidence because both use related optical feature signals.

Before any `auto_weak_label` dataset is generated, create a separate plan documenting teacher model provenance, prompt mapping, confidence thresholds, cost, and human verification workflow.
