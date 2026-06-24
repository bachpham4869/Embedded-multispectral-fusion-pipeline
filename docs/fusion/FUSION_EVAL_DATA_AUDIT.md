# Fusion Evaluation Data Audit

## Inputs

- `fusion_captures`
- `data/eval`
- `data/thermal_sequences`
- `docs/thesis_eval`
- `docs/tables/iqa`
- `docs/tables/timing`
- `q1_results`
- `legacy`

## Pairing Policy

- Strict paired: same-session/time nearest with gap <= 1.0s.
- Qualitative weak: nearest gap <= 20.0s or incomplete thermal/NIR evidence.
- Proxy/unpaired/synthetic: usable for stress testing only, not proof of real fusion quality.

## Inventory Summary

| kind | modality | evidence_label | extension | count | size_mb |
| --- | --- | --- | --- | --- | --- |
| fusion_output | fusion | real_paired | .png | 13 | 5.816 |
| nir_raw | nir | real_paired | .png | 3 | 1.841 |
| numeric_sequence | thermal | proxy | .npy | 33 | 18.736 |
| numeric_sequence | thermal | synthetic | .npy | 1 | 3.784 |
| run_manifest | metrics | real_paired | .csv | 1 | 0.039 |
| run_manifest | metrics | real_paired | .json | 32 | 0.055 |
| session_metrics | metrics | real_paired | .csv | 2 | 0.008 |
| session_metrics | metrics | real_paired | .json | 40 | 0.153 |
| sidecar_or_table | metadata | unknown | .csv | 42 | 5.913 |
| sidecar_or_table | metadata | unknown | .json | 22 | 0.038 |
| thermal_output | thermal | real_paired | .png | 3 | 1.516 |
| unknown_image | unknown | unknown | .jpeg | 1 | 0.101 |
| unknown_image | unknown | unknown | .jpg | 56 | 7.199 |
| unknown_image | unknown | unknown | .png | 70 | 37.331 |

## Pairing Summary

| pair_status | evidence_label | count |
| --- | --- | --- |
| qualitative_weak | proxy | 2 |
| reject_unpaired | unpaired | 11 |
