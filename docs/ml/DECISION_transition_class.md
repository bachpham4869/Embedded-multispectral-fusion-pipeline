# Decision Record: `transition` Class

Status: provisional. Decision is not to claim `transition` as a strong ENV class
yet.

## Current Definition

`transition` currently means dawn/dusk/sunrise-like lighting. In code it routes
to Bucket F, which blends low-light enhancement and highlight tone mapping.

## Support

| Split | Rows | Percent | Mean label confidence | Sources |
| --- | --- | --- | --- | --- |
| train | 438 | 3.66% | 0.784 | `offline_mwd`:301, `offline_weather_time`:137 |
| test | 77 | 3.64% | 0.786 | `offline_mwd`:56, `offline_weather_time`:21 |
| reference | 515 | 3.65% | 0.785 | `offline_mwd`:357, `offline_weather_time`:158 |

Source labels:

- Weather-Time `Dawn` and `Dusk` map to `transition` with confidence 0.75.
- MWD `sunrise` maps to `transition` with confidence 0.80.

## Frozen Evaluation Snapshot

Metric status: `preliminary / not thesis-ready`.

The frozen eval command reported `transition` precision 0.868, recall 0.857,
F1 0.863, support 77. Confusions:

| True `transition` predicted as | Count |
| --- | --- |
| `transition` | 66 |
| `normal_day` | 5 |
| `normal_night` | 4 |
| `fog` | 1 |
| `glare` | 1 |

Related confusions into `transition`:

| True class | Predicted `transition` count |
| --- | --- |
| `normal_day` | 4 |
| `fog` | 4 |
| `normal_night` | 2 |

## Evidence Gaps

- Test support is only 77 rows.
- No live NIR dawn/dusk sessions are labeled.
- JSONL lacks source image paths, so near-duplicate checks cannot be completed.
- No evidence yet that Bucket F improves image quality versus Bucket A or C on
  transition frames.
- Current features include cyclic hour fields, but sidecar feature importance
  reports `hour_of_day_sin`, `hour_of_day_cos`, and `prev_env_class` at 0.0,
  suggesting the present RF is not using time context to identify transition.

## Decision

Do not present `transition` as a strong independent ENV classifier class in the
thesis yet. Present it as:

- a weak/provisional class in the current optical RGB-proxy baseline, and
- a candidate runtime state named `dawn_dusk_blend`.

Keep Bucket F available because the processing policy is plausible and low risk,
but do not require `transition` to remain in the main classifier if Phase 2
evidence remains weak.

## Migration Plan If Dropped From Classifier

1. Keep old model compatibility and old JSONL labels.
2. Add a schema version and sidecar compatibility gate.
3. Map `transition` labels to `normal_day` or `normal_night` only after a data
   audit decides the boundary rule.
4. Implement runtime `dawn_dusk_blend` using brightness/time/hysteresis rather
   than classifier output.
5. Re-run same-feature benchmark and bucket IQA before making thesis claims.

## Evidence Needed To Keep It As A Class

- At least source-diverse dawn/dusk data, not just MWD sunrise and Weather-Time
  Dawn/Dusk.
- Per-class F1 with bootstrap CI under a leakage-clean split.
- Confusion analysis versus `normal_day`, `normal_night`, and `glare`.
- Bucket F IQA/latency evidence showing a real processing benefit.
- Live IMX290/NIR validation across dawn/dusk sessions.
## Phase M5 Raw Sensor Update

Raw sensor video `test_30fps_morning.mp4` produced 590 sampled unlabeled frames.
The production model predicted `transition` for only 3 / 590 frames under
`RGB-scaler proxy inference`, with no manual labels. This does not validate
`transition` as an environment class.

Updated recommendation: do not defend `transition` as a strong classifier class
unless manually labeled dawn/dusk sensor data shows stable support,
source-diverse examples, and meaningful processing benefit. Prefer documenting
it as a runtime transient state such as `dawn_dusk_blend` with hysteresis or
brightness/time policy. Keep existing code unchanged until migration is
explicitly approved.

## Phase P12 Paired Capture Update

The paired IMX/thermal-display capture contributes 584 strict paired rows, but
all rows are unlabeled (`label_source=none`). Under `RGB-scaler proxy inference,
not validated NIR classifier accuracy`, the production RF predicted
`transition` for 5 / 584 rows.

This does not validate `transition` as an ENV class. It slightly reinforces the
current recommendation: keep Bucket F/report wording available, but treat
`transition` as a runtime transient-state candidate until trusted dawn/dusk
paired labels and bucket-quality evidence exist.

## Phase 1 Evidence Freeze

The agent-reviewed subset did not identify reliable dawn/dusk transient
lighting. The class remains useful as a report discussion point, but the
defensible implementation direction is still to convert `transition` into a
runtime transient/blend state after a documented migration plan.

No code migration is applied in this phase.
