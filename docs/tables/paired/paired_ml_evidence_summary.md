# Paired ML Evidence Summary

| Evidence item | Value | Thesis use | Caveat |
| --- | --- | --- | --- |
| paired capture rows | 584 | domain-shift and uncertainty evidence | not labeled |
| pairing tier | `frame_strict`: 584 | stronger pairing provenance | still not accuracy evidence |
| NIR/optical modality | `unknown_optical` | modality review needed | user confirmation required |
| thermal modality | `display_heatmap_like` | paired context / qualitative thermal review | not raw radiometric thermal |
| trusted labels | 0 | none for labeled metrics | paired eval is `not measured` |
| inference scope | `RGB-scaler proxy inference, not validated NIR classifier accuracy` | prediction distribution only | not validated NIR classifier accuracy |
| top-1 distribution | `normal_night`: 579, `transition`: 5 | model behavior under paired domain shift | not ground truth |
| tau1=0.62 abstention | 0.0086 | confidence/uncertainty evidence | no correctness claim |

Decision: paired data improves domain-shift and labeling evidence, but does not
justify production model/schema/class migration until trusted labels and user
confirmation exist.
