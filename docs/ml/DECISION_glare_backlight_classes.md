# Decision: `glare` and `backlight` Classes

Status: provisional keep. No code migration was applied.

## Evidence

| Class | Phase 4 test support | Raw sensor top-1 count | Raw sensor label status | Current decision |
| --- | ---: | ---: | --- | --- |
| `glare` | 60 | 65 / 590 | unlabeled, RGB-scaler proxy inference | keep provisional |
| `backlight` | 55 | 52 / 590 | unlabeled, RGB-scaler proxy inference | keep provisional |

The raw sensor video provides useful domain-shift evidence, but it does not
provide accuracy evidence because no manual labels exist. Predictions into
`glare` and `backlight` may indicate real bright-source behavior or model
miscalibration under sensor shift.

## Decision

Keep both classes provisionally because they can route meaningful processing
policies, but do not overclaim classifier reliability. A thesis defense should
state that these classes need targeted source-diverse data, confidence
intervals, and manual sensor validation.

## Phase P12 Paired Capture Update

The paired IMX/thermal-display capture has 584 `frame_strict` rows and 0 trusted
ENV labels. Under `RGB-scaler proxy inference, not validated NIR classifier
accuracy`, the production RF predicted 0 `glare` and 0 `backlight` rows.

Decision stays unchanged: keep both classes provisional. The paired capture does
not add positive support for glare/backlight, and it does not justify dropping
them because the capture may not include those conditions and has no manual
labels.

## Phase 1 Evidence Freeze

The agent-reviewed subset adds preliminary visual evidence for bright-source
and backlit scenes:

- `glare`: one borderline raw-sensor row is visually consistent with direct
  bright-light glare.
- `backlight`: several raw/paired rows show dark foregrounds against bright
  backgrounds. The paired rows are especially useful for review because the
  thermal side is visible alongside the optical stream.

These labels are `agent_manual_label`, not user-confirmed gold labels, so they
do not make a final sensor-real accuracy claim. They do justify keeping both
classes provisional and prioritizing targeted manual labels.
