# Feature 21 Definition and Applicability

Status: non-production feature-set documentation. Production schema and model
loader are unchanged.

`optical_21_candidate_still` is a still-compatible 21-derived candidate, not
the full temporal 21-feature set. It excludes `temporal_brightness_std`, so the
actual supervised feature count is 20. The full `optical_21_candidate_temporal`
has 21 features and is valid only for sequential video/runtime analysis or
future labeled sequence datasets.

| # | Feature | Group | Applicability | Cost | Expected class benefit | Leakage risk | Sensor compatible |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | `nir_mean_brightness` | brightness | still/runtime | existing | day/night/fog separation | low | yes |
| 2 | `nir_std` | brightness/contrast | still/runtime | existing | fog/night contrast | low | yes |
| 3 | `nir_entropy` | texture | still/runtime | existing | low-light/fog texture | low | yes |
| 4 | `nir_p95` | brightness | still/runtime | existing | glare/backlight high tail | low | yes |
| 5 | `nir_glare_score` | glare | still/runtime | existing | glare/backlight | low | yes |
| 6 | `nir_sharpness` | texture | still/runtime | existing | fog/rain blur | low | yes |
| 7 | `nir_dark_fraction` | brightness | still/runtime | existing | night/underexposure | low | yes |
| 8 | `nir_saturation_mean` | color | still/runtime | existing | color/NIR proxy | low | yes |
| 9 | `hour_of_day_sin` | metadata/context | still if absolute timestamp exists | existing | day/night prior | medium if timestamp is wrong | raw video lacks absolute timestamp |
| 10 | `hour_of_day_cos` | metadata/context | still if absolute timestamp exists | existing | day/night prior | medium if timestamp is wrong | raw video lacks absolute timestamp |
| 11 | `prev_env_class` | temporal/context | runtime sequence | existing | runtime hysteresis | medium if derived from labels | runtime only |
| 12 | `nir_blue_mean_ema` | color/temporal | runtime extractor state | existing | NIR/RGB proxy | low | yes |
| 13 | `local_contrast_mean` | contrast | still/runtime | low | fog/rain/local haze | low | yes |
| 14 | `local_contrast_std` | contrast | still/runtime | low | fog/rain/local haze | low | yes |
| 15 | `edge_density` | texture | still/runtime | low | rain/fog blur | low | yes |
| 16 | `dark_channel_mean` | haze | still/runtime | low | fog/haze | low | yes |
| 17 | `dark_channel_p95` | haze | still/runtime | low | fog/haze brightness tail | low | yes |
| 18 | `highlight_connected_component_area` | glare | still/runtime | medium | glare/backlight | low | yes |
| 19 | `saturated_component_count` | glare | still/runtime | medium | glare/headlight clusters | low | yes |
| 20 | `p99_p1_dynamic_range` | brightness/contrast | still/runtime | low | glare/backlight/low contrast | low | yes |
| 21 | `temporal_brightness_std` | temporal/brightness | video only | low rolling window | transition/flicker/stability | medium if computed across split boundary | yes for sequential raw frames |

Fair benchmark status: `optical_21_candidate_still` has 20 supervised
features and covers only four verified-image classes in the current dataset
(`fog`, `normal_day`, `rain`, `transition`). It must not be compared as
same-condition evidence against the full 9-class `optical_12_baseline`.
