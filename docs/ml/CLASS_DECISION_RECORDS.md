# Class Decision Records

Status: documentation decision only. No ENV class was renamed, dropped, or
merged in code.

## Evidence Inputs

| Evidence | Scope | Status |
| --- | --- | --- |
| Phase 4 duplicate-cluster-aware split | offline optical RGB-proxy baseline | usable with caveats |
| Raw sensor video `test_30fps_morning.mp4` | 590 sampled unlabeled frames | preliminary domain-shift/prediction only |
| Raw sensor predictions | RGB-scaler proxy inference | no accuracy claim |
| Manual sensor labels | not provided | not measured |

## Recommendations

| Class | Phase 4 evidence | Raw sensor signal | Processing-policy need | Recommendation |
| --- | --- | --- | --- | --- |
| `normal_day` | strong support and frequent raw prediction | 241 / 590 top-1 frames | baseline daytime policy | keep |
| `fog` | supported in verified-image subset | 73 / 590 top-1 frames | important visibility policy | keep, seek manual sensor labels |
| `rain` | supported in verified-image subset | 139 / 590 top-1 frames | weather policy useful | keep, validate sensor labels |
| `night_clear` | offline support exists | 0 / 590 top-1 frames in morning raw video | deployment relevant | keep; raw video does not cover night |
| `normal_night` | offline support exists | 17 / 590 top-1 frames in morning raw video | deployment relevant | keep; needs night sensor data |
| `nir_night` | offline class exists but raw video does not prove NIR | 0 / 590 top-1 frames | deployment critical | keep provisional until manual NIR labels |
| `glare` | low-support class, Phase 4 F1 remains moderate | 65 / 590 top-1 frames | separate glare policy can matter | keep provisional; targeted labels required |
| `backlight` | low-support class, Phase 4 F1 remains moderate | 52 / 590 top-1 frames | separate backlight policy can matter | keep provisional; targeted labels required |
| `transition` | weak conceptual class | 3 / 590 top-1 frames | bucket F may be runtime blend | convert to runtime transient state unless labeled dawn/dusk data proves class |

Final recommendation for this phase: keep code taxonomy unchanged, but defend
`transition`, `glare`, and `backlight` cautiously in the thesis. Do not claim
sensor-real class accuracy until manual labels are joined and evaluated.

## Phase M11 Suggested-Label Caveat

The RF/heuristic suggestion package may surface frames for `glare`,
`backlight`, `transition`, and `nir_night` review, but it is not an independent
teacher and cannot strengthen class decisions by itself. These rows should be
used to decide which frames the user labels first.

Class decisions remain unchanged:

- `transition`: keep as a cautious/report-facing class for now, but prefer a
  runtime transient-state interpretation unless manual dawn/dusk evidence is
  added.
- `glare` and `backlight`: keep provisional; targeted manual labels are still
  required.
- `nir_night`: keep as deployment-relevant; modality and manual labels are
  still required before sensor-real claims.

## Phase P12 Paired NIR/Thermal Update

Paired capture `data/paired_data/` adds 584 strict paired IMX/thermal-display
rows. Label status is still unlabeled: `label_source=none`, 0 trusted labels.
Inference is `RGB-scaler proxy inference, not validated NIR classifier
accuracy`.

| Class | Paired proxy prediction signal | Decision impact |
| --- | ---: | --- |
| `normal_night` | 579 / 584 top-1 predictions | strong domain-shift signal, but no accuracy claim |
| `transition` | 5 / 584 top-1 predictions | still best treated as runtime transient unless manual dawn/dusk labels exist |
| `nir_night` | 0 / 584 top-1 predictions | still deployment-relevant; paired modality/user labels needed |
| `glare` / `backlight` | 0 / 584 top-1 predictions | no paired evidence yet; keep provisional and collect targeted labels |
| `fog` / `rain` | 0 / 584 top-1 predictions | no paired evidence yet |

Recommendation remains documentation-only: keep code taxonomy unchanged. Paired
data should be manually labeled before it changes class keep/drop/merge
decisions.

## Phase 1 Evidence Freeze

The agent-reviewed subset adds preliminary visual evidence but does not apply a
code migration. It contains 24 visually reviewed rows, 22 of which have
`label_confidence >= 0.8`; the labels are `agent_manual_label`, not
user-confirmed gold labels.

| Class | Phase 1 update | Decision |
| --- | --- | --- |
| `normal_day` | Agent-reviewed raw rows confirm several ordinary daylight frames. | keep |
| `backlight` | Agent-reviewed paired/raw rows suggest backlit sensor scenes are common and may be misread by RGB-proxy RF as night-like. | keep provisional; collect user-confirmed labels |
| `glare` | One borderline raw row is visually consistent with direct bright-light glare. | keep provisional; needs targeted labels |
| `transition` | No inspected row shows reliable dawn/dusk transient lighting. | prefer runtime transient-state interpretation |
| `nir_night` | Paired modality remains `unknown_optical`; no trusted NIR-night label exists. | keep deployment-relevant but caveated |
| `fog` / `rain` | Raw proxy predictions exist, but inspected agent rows did not confirm fog/rain. | keep from offline evidence; target sensor collection |

Final Phase 1 recommendation: keep the current code taxonomy unchanged, defend
`normal_day`, `fog`, `rain`, and night classes from offline evidence, and treat
`transition`, `glare`, `backlight`, and `nir_night` with explicit caveats until
user-confirmed sensor labels are available.

See `docs/tables/ml/class_decision_summary.md`.
