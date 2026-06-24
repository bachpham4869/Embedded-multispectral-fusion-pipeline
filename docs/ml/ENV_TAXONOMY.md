# Environment Taxonomy

Status: report-facing taxonomy proposal. Phase 1 does not rename code symbols.

ENV classes describe environmental conditions. Processing buckets describe image
processing policies. Do not mix the two in report language.

| Current code name | Report-facing name | Short definition | Data/feature cues | Processing policy | Keep in classifier? | Merge/drop candidate | Rename risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `night_clear` | `dark_clear_optical` | Dark outdoor scene with little ambient light | low mean brightness, high dark fraction, low saturation | Bucket A `night_hybrid_enhance` | yes | none | high: production model sidecar and label mapping use current name |
| `normal_night` | `urban_ambient_night` | Night scene with ambient/street lighting | moderate brightness, street-light texture, ExDark source | Bucket A `night_hybrid_enhance` | yes | could merge with `night_clear` only if confusion high | high |
| `normal_day` | `clear_daylight` | Normal daytime condition | higher brightness/p95, lower dark fraction | Bucket C `highlight_tone_map` | yes | none | high |
| `fog` | `reduced_visibility_fog_haze` | Fog, smog, sandstorm-like reduced visibility | lower contrast/sharpness, Weather11 fogsmog/sandstorm | Bucket D `fog_dehaze_lite` | yes | haze can remain policy-level, not separate class | high |
| `rain` | `rain_wet_scene` | Rain or wet road condition | Weather-Time/Weather11/MWD rain labels; temporal evidence not yet used | Bucket E `rain_temporal_median` | yes | none yet | high |
| `glare` | `direct_glare_highlight` | Direct bright source/highlight saturation | high p95/glare score, glare_street source | Bucket C `highlight_tone_map` | provisional | may remain safety/runtime override if support stays weak | high |
| `backlight` | `backlit_high_dynamic_range` | Bright background with darker subject/foreground | high dynamic range/backlight source; current features are weak for foreground relation | Bucket C `highlight_tone_map` | provisional | may merge with glare/HDR policy if not separable | high |
| `transition` | `dawn_dusk_transition` | Dawn/dusk lighting boundary | Weather-Time Dawn/Dusk, MWD sunrise; low support | Bucket F `dawn_dusk_blend` | weak/provisional | runtime state `dawn_dusk_blend` | high |
| `nir_night` | `nir_assisted_mono_night` | NIR-like monochrome night proxy | gray_nir source, low saturation, distinct blue-channel EMA | Bucket B `nir_mono_clahe` | yes as proxy | requires live NIR validation | high |

## Phase 1 Decision

Keep all current code names for backward compatibility. Use report-facing names
as aliases in thesis text and tables. Any official rename requires:

- schema version
- model sidecar validation update
- backward-compatible loader
- migration note for old JSONL/model bundles
- full benchmark evidence after rename
- user confirmation

## Provisional Classes

`glare` and `backlight` are kept temporarily because their processing policies
matter operationally, but they have low support and narrow source diversity:

- train support: glare 340, backlight 313
- test support: glare 60, backlight 55
- reference support: glare 400, backlight 368
- sources: glare only `offline_glare_street`; backlight only `offline_backlight`

Report language must say these are provisional classes requiring confidence
intervals, source-diversity review, and live validation.

## Transition

`transition` should not be claimed as a strong independent ENV class in the
current evidence state. It is better framed as a candidate runtime state for
`dawn_dusk_blend` until more data, confusion analysis, and Bucket F image-quality
evidence exist. See `docs/ml/DECISION_transition_class.md`.
