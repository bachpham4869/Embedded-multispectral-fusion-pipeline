# Class Decision Summary

Status: Phase 1 freeze. No ENV class was renamed, dropped, or merged in code.

| Class | Offline evidence | Raw / paired sensor evidence | Processing-policy need | Decision | Caveat |
| --- | --- | --- | --- | --- | --- |
| `normal_day` | Strong offline support; appears in raw sensor predictions | Agent-reviewed raw subset includes normal daylight frames | Baseline daytime policy | keep | Raw labels are agent-reviewed only |
| `normal_night` | Strong offline support | Paired proxy predictions collapse mostly to `normal_night` | Night policy useful | keep | Paired agent review suggests many rows may be backlit rather than night |
| `night_clear` | Offline support exists | Not represented in current morning raw/paired evidence | Deployment relevant | keep | Needs night sensor capture |
| `nir_night` | Offline support exists | Paired modality remains `unknown_optical`; no trusted `nir_night` label yet | Deployment relevant | keep provisional | Requires confirmed NIR modality and manual labels |
| `fog` | Offline support and feature subset evidence | Raw proxy predicts fog on some frames, but agent-reviewed subset did not confirm fog | Visibility policy important | keep | Needs targeted sensor labels |
| `rain` | Offline support and feature subset evidence | Raw proxy predicts rain on some frames, but agent-reviewed subset did not confirm rain | Weather policy useful | keep | Needs targeted sensor labels |
| `glare` | Low-support/risk class | Agent-reviewed raw subset has one borderline glare row | Glare policy meaningful | keep provisional | Needs targeted data and user-confirmed labels |
| `backlight` | Low-support/risk class | Agent-reviewed raw/paired subset suggests backlit scenes are common and RF proxy may miss them | Backlight policy meaningful | keep provisional | Needs user review; do not overclaim |
| `transition` | Offline support but concept remains ambiguous | Raw/paired proxy transition frequency is low; no dawn/dusk evidence in inspected frames | Better as smoothing/blend state | convert to runtime transient candidate | Do not claim as strong ENV classifier class without dawn/dusk labels |
