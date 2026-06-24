# Feature Set Rationale

Status: non-production feature engineering evidence. Production schema and
model loader are unchanged.

## Feature Set Versions

| Feature set | Scope | Production status | Evidence status |
| --- | --- | --- | --- |
| `optical_12_baseline` | Current 12 handcrafted optical features | Current production-compatible schema | Offline optical RGB-proxy evidence with caveats |
| `optical_v2_candidate` | 12 baseline + 8 still-image candidate features | Non-production candidate only | Subset ablation only |
| `optical_21_candidate_still` | 12 baseline + 8 still-image candidate features | Non-production research candidate only | Benchmarked on verified 4-class subset; actual count is 20, not the full temporal 21 |
| `optical_21_candidate_temporal` | 12 baseline + 8 still-image candidate features + `temporal_brightness_std` | Non-production video/analysis candidate only | Not supervised-benchmarked; requires labeled sequence data |

## `optical_21_candidate` Features

| # | Feature | Group | Cost | Still/video | Leakage risk | Raw sensor compatible |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | `nir_mean_brightness` | brightness | existing | still | low | yes |
| 2 | `nir_std` | brightness | existing | still | low | yes |
| 3 | `nir_entropy` | texture | existing | still | low | yes |
| 4 | `nir_p95` | brightness | existing | still | low | yes |
| 5 | `nir_glare_score` | glare | existing | still | low | yes |
| 6 | `nir_sharpness` | texture | existing | still | low | yes |
| 7 | `nir_dark_fraction` | brightness | existing | still | low | yes |
| 8 | `nir_saturation_mean` | color | existing | still | low | yes |
| 9 | `hour_of_day_sin` | temporal/context | existing | still if absolute timestamp exists | medium if timestamp proxy is wrong | yes, but raw video has no absolute timestamp |
| 10 | `hour_of_day_cos` | temporal/context | existing | still if absolute timestamp exists | medium if timestamp proxy is wrong | yes, but raw video has no absolute timestamp |
| 11 | `prev_env_class` | metadata/context | existing | sequence/runtime | medium if derived from test labels | yes for runtime, not offline stills |
| 12 | `nir_blue_mean_ema` | color | existing | sequence/still extractor state | low | yes |
| 13 | `local_contrast_mean` | contrast | low, blur + abs diff | still | low | yes |
| 14 | `local_contrast_std` | contrast | low, blur + abs diff | still | low | yes |
| 15 | `edge_density` | texture | low, Canny | still | low | yes |
| 16 | `dark_channel_mean` | haze | low, channel min | still | low | yes |
| 17 | `dark_channel_p95` | haze | low, channel min + percentile | still | low | yes |
| 18 | `highlight_connected_component_area` | glare | medium, connected components | still | low | yes |
| 19 | `saturated_component_count` | glare | medium, connected components | still | low | yes |
| 20 | `p99_p1_dynamic_range` | brightness/contrast | low, percentiles | still | low | yes |
| 21 | `temporal_brightness_std` | brightness/temporal | low rolling std | video only | medium if computed across split boundary | yes for sampled video frames |

## Measured Ablation

The verified-image subset has only four classes because many Phase 4 JSONL rows
still lack original image paths. Therefore these results are not directly
comparable to the full 9-class `optical_12_baseline`.

| Feature set | Best measured model | Classes | Train/test rows | Balanced accuracy | Macro-F1 | Decision |
| --- | --- | --- | --- | ---: | ---: | --- |
| `optical_12_baseline` | RF200 | fog, normal_day, rain, transition | 4923 / 868 | 0.7482 | 0.7590 | baseline for subset |
| `optical_v2_candidate` | RF200 | fog, normal_day, rain, transition | 4923 / 868 | 0.7584 | 0.7719 | candidate, but evidence is subset-only |
| `optical_21_candidate` | not run | none | 0 / 0 compatible labeled rows | n/a | n/a | defer; temporal labels unavailable in still-image split |

## Phase M10 Fair Feature-Set Comparison

The M10 comparison uses the same verified-image subset for all compared feature
sets. The subset contains only four classes (`fog`, `normal_day`, `rain`,
`transition`) with 4,923 train rows and 868 test rows. These results are
therefore subset/preliminary and must not be compared as same-condition evidence
against the full 9-class `optical_12_baseline` benchmark.

| Feature set | Actual features | Best measured model | Coverage | Balanced accuracy | Macro-F1 | Decision |
| --- | ---: | --- | --- | ---: | ---: | --- |
| `optical_12_baseline` | 12 | RF200 | 4/9 | 0.7482 | 0.7590 | subset baseline |
| `optical_v2_candidate` | 20 | RF200 | 4/9 | 0.7584 | 0.7719 | research candidate only |
| `optical_21_candidate_still` | 20 | RF200 | 4/9 | 0.7584 | 0.7719 | same still-compatible vector as v2; research candidate only |

Decision: do not migrate production schema. `optical_v2_candidate` and
`optical_21_candidate_still` show a small subset gain, but coverage is limited
to four classes and RPi4 feature-extraction timing is not measured. The temporal
21-feature candidate remains deferred until a labeled sequential dataset exists.

## Phase P12 Paired Feature Evidence

Paired IMX/thermal-display data produced 584 `optical_12_baseline` rows. These
rows are useful for domain-shift analysis only because trusted labels are absent.

| Feature set | Paired status | Decision |
| --- | --- | --- |
| `optical_12_baseline` | extracted for 584 paired rows | use for drift/confidence/abstention |
| `optical_v2_candidate` | not promoted by paired run | remains research-only |
| `optical_21_candidate_still` | not promoted by paired run | remains research-only |
| `optical_21_candidate_temporal` | requires labeled sequence data | deferred |

No production feature schema change is justified by paired data without manual
labels and a fair labeled benchmark.
