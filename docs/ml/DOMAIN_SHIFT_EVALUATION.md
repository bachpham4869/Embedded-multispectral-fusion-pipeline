# Domain Shift Evaluation

Status: measured on unlabeled paired nir sampled frames. No sensor-real accuracy is claimed.

Current prediction scope is `RGB-scaler proxy inference, not validated NIR classifier accuracy` because the production bundle uses the selected scaler. This is not live NIR/LWIR validation.

| Metric | Value |
| --- | --- |
| sensor_rows | 584 |
| abstention_rate_tau1_0.62 | 0.0086 |
| mean_confidence | 0.9479 |
| mean_entropy | 0.3084 |
| label_status | unlabeled / preliminary |

Manual labels are required before accuracy, balanced accuracy, F1, confusion matrix, or calibration-on-sensor claims can be made.

## Phase P12 Paired NIR/Thermal Capture

Paired capture evidence was added from `data/paired_data/`:

| Item | Value |
| --- | --- |
| paired rows | 584 |
| pairing tier | `frame_strict`: 584 |
| NIR/optical modality | `unknown_optical` until user confirms |
| thermal modality | `display_heatmap_like`, not raw radiometric thermal |
| trusted labels | 0 |
| inference scope | `RGB-scaler proxy inference, not validated NIR classifier accuracy` |

The paired run is useful domain-shift and uncertainty evidence, but it is not
sensor accuracy evidence. The production RF predicted `normal_night` for 579 /
584 paired IMX frames and `transition` for 5 / 584 frames. Because `label_source`
is `none` for all rows, these predictions are not used to choose a model by
accuracy.
