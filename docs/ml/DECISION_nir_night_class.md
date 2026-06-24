# Decision: `nir_night` Class

Status: keep provisional. No code migration was applied.

## Phase P12 Paired Evidence

The paired capture under `data/paired_data/` provides 584 strict paired
IMX/thermal-display rows. The IMX stream is treated as `unknown_optical` until
the user confirms modality. The thermal stream is `display_heatmap_like`, not
raw radiometric thermal.

| Evidence | Value |
| --- | --- |
| paired rows | 584 |
| trusted labels | 0 |
| inference scope | `RGB-scaler proxy inference, not validated NIR classifier accuracy` |
| `nir_night` top-1 predictions | 0 / 584 |

## Decision

Keep `nir_night` as a deployment-relevant class, but do not claim paired-sensor
validation yet. The current paired capture does not provide trusted `nir_night`
support because labels and modality confirmation are missing.

Required next evidence:

- user-confirmed modality for the IMX stream,
- manually labeled paired night/NIR frames,
- domain-shift and labeled-eval tables stratified by pairing tier/session,
- RPi4 latency evidence before production migration.

## Phase 1 Evidence Freeze

The final evidence freeze does not add trusted `nir_night` support. The current
agent-reviewed subset focuses on morning raw sensor frames and paired backlit
frames; modality remains `unknown_optical`.

Decision remains: keep `nir_night` because it is deployment-relevant, but mark
accuracy as unvalidated until confirmed NIR modality and user-confirmed night
labels exist.
