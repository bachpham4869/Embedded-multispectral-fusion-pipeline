# Auto-Labeling Options

Current decision: do not generate `auto_weak_label` records in this phase because no local independent teacher is available and no external API/model download has been approved.

| Field | Value |
| --- | --- |
| status | no_independent_teacher_available |
| selected_method | rf_heuristic_suggested_label |
| independent_teacher_available | False |
| weak_label_dataset_allowed | False |
| git_commit | 15fbe64d5c70873cf200eac688f37d03e8f103da |

RF/heuristic suggestions are not a stronger model and not an independent teacher. They only pre-fill a review template and prioritize frames for human labeling.
