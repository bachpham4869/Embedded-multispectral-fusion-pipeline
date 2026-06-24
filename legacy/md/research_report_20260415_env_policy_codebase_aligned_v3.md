---
title: "SmartBinocular Environment Classification & Fusion Policy: Codebase-Aligned UltraDeep Analysis (v3.0)"
date: 2026-04-15
mode: ultradeep
project: smartBinocular
supersedes:
  - research_report_20260415_env_classification_fusion_policy.md
  - research_report_20260415_env_classification_fusion_policy_ultradeep.md
codebase_read:
  - src/smartbinocular/feature_schema.py
  - src/smartbinocular/config.py
  - src/smartbinocular/env_presets.py
  - src/smartbinocular/nir_pipeline.py
  - src/smartbinocular/display_pipeline.py
  - src/smartbinocular/thermal_pipeline.py
  - src/smartbinocular/main.py
---

# SmartBinocular Environment Classification & Fusion Policy: Codebase-Aligned UltraDeep Analysis (v3.0)

**Date:** 2026-04-15 | **Mode:** UltraDeep | **Supersedes:** v1 (deep) and v2 (ultradeep prior session)

---

## Executive Summary

This report is the third and most authoritative iteration of environment classification and fusion-policy research for the SmartBinocular pipeline. Unlike its predecessors, it opens by reading six production source files before synthesizing any literature, ensuring every recommendation maps to actual symbol names, CONFIG keys, and ENV_PRESET entries rather than to an invented parallel taxonomy.

The SmartBinocular pipeline classifies scenes into nine canonical classes defined in `feature_schema.py`: `night_clear`, `normal_night`, `normal_day`, `fog`, `rain`, `glare`, `backlight`, `transition`, and `nir_night`. The pipeline's existing `infer_env_tags_auto_rule` function already produces a multi-label `Set[str]` of environment tags; `select_env_preset_from_tags` then collapses that set to one of thirteen named presets via a priority-ordered lookup. This architecture is structurally equivalent to Binary Relevance (BR) multi-label classification with a hand-crafted label aggregator — a sound starting point that literature confirms is well-suited for edge Random Forest inference.

Two major themes are developed throughout this report.

**Theme 1 — Overlapping semantics and temporal stabilization.** The nine ENV classes are not fully mutually exclusive: `normal_night`, `night_clear`, and `nir_night` form an illumination sub-axis; `fog`, `rain`, `glare`, and `backlight` form an interference sub-axis; `transition` is a transient state on the illumination axis; `normal_day` is the default illumination state. This structure suggests Classifier Chains (CC) as a natural upgrade to the existing rule-based tag-set: an illumination sub-classifier feeds posterior probabilities into interference sub-classifiers, capturing the dependency that glare is more likely during the day and fog is more visually severe at night. The existing `EnvPresetController` already provides count-based hysteresis (configurable via `env_hysteresis_frames`, default 18 frames); the report adds EMA probability smoothing as a complementary pre-filter at the raw-inference stage, and describes a three-layer temporal stack that cleanly separates raw inference, smoothed probability, and latched display/policy state.

**Theme 2 — Per-condition processing policy across many knobs.** A persistent weakness in prior reports was overweighting `fusion_alpha_boost`. This report explicitly balances coverage across eight distinct lever families: CLAHE clip limit and tile grid (`nir_enhancer_clahe_clip_scale`), night-mode Schmitt thresholds inside `HybridNIREnhancer`, thermal 3DNR alpha (`thermal_3dnr_alpha`), bilateral filter parameters, AGC percentile stretch (`thermal_agc_low_pct` / `thermal_agc_high_pct`), edge strength (`thermal_display_edge_strength`), background-model floor and threshold (`thermal_floor`, `thermal_fg_threshold`), E1 / MAD detector sensitivity (`feature_e1_z_thresh`, `feature_e1_min_area`), display grading (`brightness`, `contrast`, `saturation`, `warmth`, `shadows`, `highlights`, `display_l_max`), temporal glare blend weight (`temporal_prev_weight`), and the fusion alpha boost.

Appendix A delivers a 22-row label/combo → knob → source → RPi budget table. Appendix B maps every recommendation to existing CONFIG keys and ENV_PRESET names, explicitly flagging speculative versus code-backed items.

---

## 1. Introduction

### 1.1 Scope and Research Goals

This report addresses two research threads commissioned for the SmartBinocular system — a real-time binocular night-vision device running on Raspberry Pi 4B (quad-core ARM Cortex-A72 at 1.8 GHz, CPU-only) with a ≤50 ms/frame end-to-end budget:

1. **Multi-label / overlapping environment semantics and temporal stabilization**: How should the classification system represent scenes that are simultaneously dark and foggy, or transitioning between states? How should raw-frame inference be separated from stable display state?

2. **Per-label and per-combination processing policy**: What algorithmic knobs — across all lever families, not only fusion alpha — should change for each of the nine ENV classes and their plausible combinations?

### 1.2 Critical Capture-Path Statement (mandatory per user instructions)

The optical sensor is an IMX-class Bayer RGGB device (IMX290 STARVIS). Its apparent behavior in any given frame is determined entirely by scene illumination and filter configuration, not by any separate software "mode":

- **Normal illumination → normal BGR behavior.** All three Bayer channels receive distinct spectral contributions and the output is a full-color frame.
- **Dark scene with active IR-assist illumination → mono-like / NIR-dominant.** IR illuminators emit primarily in the 850–940 nm band. At these wavelengths the R, G, and B Bayer channels all respond to IR similarly (spectral response of typical color filters extends to ~1000 nm [23]), so all channels receive comparable signals and the image appears desaturated — monochrome-like. This is not a hardware mode; it is physics.

The codebase correctly models this: `HybridNIREnhancer` detects "night mode" via a Schmitt trigger on per-frame brightness (off=0.48, on=0.40 as fraction of 255) and converts to grayscale internally when `is_night_mode=True` [4]. The `nir_night` ENV class captures this regime explicitly. No separate user or software switch is assumed.

The codebase names the optical pipeline branch `nir_*` for historical reasons; "NIR pipeline" throughout this report means the optical processing branch (`nir_pipeline.py`), not a fixed-monochrome sensor assumption.

### 1.3 Codebase Files Read

The following files were read before any literature synthesis. All symbol names, key names, and preset names in this report reference these sources directly:

| File | Path | Key content |
|------|------|-------------|
| `feature_schema.py` | `src/smartbinocular/feature_schema.py` | `ENV_CLASSES` (9 entries), `FEATURE_SET_CORE` (11 features), `FeatureRecord` dataclass [1] |
| `config.py` | `src/smartbinocular/config.py` | `CONFIG` dict, `_VALID_OPT_OVERRIDES_KEYS`, `resolve_optimization_profile` [2] |
| `env_presets.py` | `src/smartbinocular/env_presets.py` | `build_env_presets` (13 presets), `infer_env_tags_auto_rule`, `EnvPresetController`, `select_env_preset_from_tags` [3] |
| `nir_pipeline.py` | `src/smartbinocular/nir_pipeline.py` | `HybridNIREnhancer`, CLAHE levels, Schmitt trigger, dark/bright channel, A1-lite tone map, guided filter [4] |
| `display_pipeline.py` | `src/smartbinocular/display_pipeline.py` | `display_grade_and_cap_bgr`, `DisplayTemporalGlareBlend` [5] |
| `thermal_pipeline.py` | `src/smartbinocular/thermal_pipeline.py` | `ThermalProcessor`, `KalmanThermalBackground`, `ThermalTemporalFilter` (3DNR), `ThermalMADAnomalyDetector`, E1 detector [6] |
| `main.py` | `src/smartbinocular/main.py` | Pipeline composition, ML inference integration, env preset application [7] |

### 1.4 Relationship to Prior Reports

Two prior reports are archived alongside this file in `legacy/md/`. This report supersedes both:

- `research_report_20260415_env_classification_fusion_policy.md` (deep mode, ~4,800 words) — incorrect sensor assumption; predates codebase read.
- `research_report_20260415_env_classification_fusion_policy_ultradeep.md` (ultradeep, ~16,000 words) — correct sensor framing but invented some taxonomy not matching `ENV_CLASSES`; predates codebase read.

**Newer:** optical bucket policy is in [`docs/research_report_20260415_env_policy_branching_optical_v4.md`](../../docs/research_report_20260415_env_policy_branching_optical_v4.md) (repo root path `docs/`).

This v3.0 report does not duplicate those reports verbatim. It references their themes where they remain valid and corrects or extends where the codebase diverges.

---

## 2. Finding 1: The Nine ENV Classes — Semantic Structure and Overlap

### 2.1 Taxonomy as Read from the Codebase

`ENV_CLASSES` in `feature_schema.py` defines exactly nine strings [1]:

```
night_clear   – night, open sky, no ambient light (optical NV target)
normal_night  – night with ambient (urban, street-lit)
normal_day    – daytime, normal conditions
fog           – fog or smog, visibility degraded
rain          – rain; wet surfaces, drops on lens
glare         – direct light source (headlights, direct sun)
backlight     – subject against bright background
transition    – dawn or dusk
nir_night     – dark scene, IMX290 NIR-dominant mono (active NIR / extended spectral)
```

### 2.2 Structural Axes Hidden in the Flat List

The nine classes are not uniformly orthogonal. Reading them with domain knowledge reveals a latent two-axis structure:

**Illumination axis** (primarily temporal/ambient-light driven):

| Class | Illumination level | Typical ambient |
|-------|--------------------|----------------|
| `nir_night` | Very dark, IR-assist active | <5 lux, active illuminator |
| `night_clear` | Dark, no ambient, open sky | <10 lux, starlight |
| `normal_night` | Night, urban ambient | 10–100 lux streetlit |
| `transition` | Twilight / dawn | 50–500 lux changing |
| `normal_day` | Daytime | >1000 lux |

**Interference axis** (atmospheric or light-source artifact):

| Class | Interference type | Optical channel effect | Thermal effect |
|-------|-------------------|----------------------|----------------|
| `fog` | Atmospheric scattering | Contrast loss, saturation drop | Penetrates; thermal advantage |
| `rain` | Droplet occlusion, wet lens | High-frequency noise, streaks | Minor degradation |
| `glare` | Strong point source | Pixel saturation, halo | Strong thermal signature unlikely (headlights are ~visible) |
| `backlight` | Bright background | Subject underexposed | Background thermal may differ |

### 2.3 Mutual Exclusion vs. Co-occurrence

Classes within the same axis are largely mutually exclusive (a scene is unlikely to be simultaneously `nir_night` and `normal_day`). Classes across axes can co-occur: `normal_night` + `fog` is common (the `night_fog` preset already handles it [3]); `normal_day` + `glare` is handled by `day_glare` [3]; `nir_night` + `fog` is a physically possible but currently unaddressed combination.

The `glare` and `backlight` classes sit between axes: they require a certain illumination level (glare is absent in `nir_night`) and represent a light-source interference. This is exactly why `glare_nir` detection in `infer_env_tags_auto_rule` is conditional on `nir_glare_allowed()`, which returns `False` when `is_night_mode=True` [4].

### 2.4 Implications for Classification Design

This structure has three practical implications:

1. **Illumination must be classified first.** Whether `fog` should trigger the full fog preset (including `fusion_alpha_boost`) depends on whether the scene is dark (where NIR already degrades fog contrast and thermal advantage is highest) or daylit (where NIR retains some contrast but saturation drop is a valid fog feature). Classifier Chains implement this naturally by making illumination-class posteriors available to interference classifiers as additional features [9].

2. **`transition` is inherently ambiguous.** Dawn and dusk span ~30 minutes during which `nir_mean_brightness` changes continuously. The existing hysteresis-based `EnvPresetController` (18 frames at 9 thermal fps ≈ 2 seconds) is appropriate in length but the transition class itself should probably trigger lower-aggressiveness presets rather than oscillating between night and day presets.

3. **`nir_night` is distinguishable from `night_clear` and `normal_night` via `nir_saturation_mean`.** When IR-assist is active, HSV saturation drops below ~25 DN (mean S channel on a small BGR frame). This feature is already feature 8 in `FEATURE_SET_CORE` [1] and is the single most reliable discriminator of `nir_night` from the other night classes. The existing rule-based system in `infer_env_tags_auto_rule` does not exploit this feature directly; an RF classifier trained on `FEATURE_SET_OPTICAL_ONLY` would learn it automatically.

---

## 3. Finding 2: Multi-Label Architecture — Tag-Set, Binary Relevance, and Classifier Chains

### 3.1 What the Codebase Already Does

`infer_env_tags_auto_rule` returns a `Set[str]` of tags — this is a multi-label output at the tag level [3]. The function uses five boolean/float inputs: `nir_b_ema`, `is_night_mode`, `nir_gray_std`, `glare_nir`, `haze_config_on`. `select_env_preset_from_tags` collapses the tag set to one preset via priority ordering and four compound-pair shortcuts (`night_fog`, `day_glare`, `low_light_cluttered`, `backlight_high_contrast`) [3].

This design is equivalent to **Binary Relevance (BR)** — each tag is predicted independently — with a deterministic aggregator replacing a learned combiner. BR is computationally efficient and well-suited to Random Forest because each label gets its own tree ensemble [30]. Its main weakness is ignoring inter-label correlations (the fact that `fog` during `nir_night` requires different handling than `fog` during `normal_day`).

### 3.2 Classifier Chains as a Natural Upgrade

A Classifier Chain (CC) for this system would proceed as a two-link chain [9]:

**Link 1 — Illumination classifier:** trained on `FEATURE_SET_OPTICAL_ONLY` (11 features, all available at runtime), predicts probabilities for `{nir_night, night_clear, normal_night, transition, normal_day}`. This is a 5-class RF predicting a vector `p_illum ∈ R^5`.

**Link 2 — Interference classifier:** trained on `FEATURE_SET_OPTICAL_ONLY` extended by `p_illum` (11 + 5 = 16 features), predicts probabilities for `{fog, rain, glare, backlight}`. Each interference label is binary (present/absent). The CC paper (Nov 2024) shows that appending prior label posteriors reduces error propagation compared to appending predicted labels [9].

The 16-feature vector is still lightweight: Link 2 uses the same CORE features plus five floats from Link 1. No thermal features are required — the optical-only set runs everywhere. If thermal is available, `FEATURE_SET_OPTICAL_THERMAL` (17 features + 5 = 22) further improves fog and night discrimination via `thm_std`, `thm_fg_fraction`, and `thm_anomaly_score`.

**Integration point:** Link 2's binary predictions become the tag set input to the existing `select_env_preset_from_tags`, requiring no changes downstream. The ML inference thread in `main.py` (currently observe-only, `ML_INFERENCE_ENABLED=False`) would need `ML_INFERENCE_ENABLED=True` and a new model path to activate this.

### 3.3 Overlapping Classes and Soft Labels

When confidence is low — for example, when `nir_mean_brightness` sits in the 0.40–0.48 Schmitt band where `HybridNIREnhancer` itself is uncertain about night mode — neither hard class labels nor the existing hysteresis resolves ambiguity well. Two complementary techniques address this:

**EMA probability smoothing:** Rather than latching to the argmax of each inference call, maintain an EMA over the RF posterior vector: `p_smooth[t] = α·p_raw[t] + (1−α)·p_smooth[t−1]`. A smaller α (slower EMA, e.g. 0.2) is appropriate for illumination classes (which change slowly); a larger α (e.g. 0.4) is appropriate for interference classes (which can appear or disappear quickly). This is analogous to asymmetric EMA as used in adaptive AGC algorithms [8].

**Top-k secondary hypotheses for the HUD:** The ML classifier can emit not only the top-1 label but also the runner-up with its probability. In the debug HUD already present in `main.py`, showing the secondary label (e.g. "night_fog 73% / normal_night 22%") gives the operator useful situational awareness without changing any pipeline behavior.

### 3.4 Practical Constraints for Edge RF on RPi 4B

The ML inference thread runs every `ML_INFERENCE_INTERVAL` frames (default 15 frames at 60 FPS NIR capture = 0.25 s interval) [7]. An RF with 100 trees over 11–16 features takes < 0.5 ms per call on a Cortex-A72 — negligible. The bottleneck is not inference but feature extraction. Feature 3 (`nir_entropy`) requires a histogram (fast via `np.bincount`) and feature 6 (`nir_sharpness` = Laplacian variance) requires a convolution on a 128×96 or smaller downsampled frame. Both are already computed at 1/15 frame rate in the ML logging path. The existing `FeatureExtractor` class handles this.

A CC adds one multiplication of a 5-element vector into a 16-element feature array per inference call — immeasurable overhead.

---

## 4. Finding 3: Temporal Stabilization — Three Layers, One Responsibility Each

### 4.1 The Existing Layer: Count Hysteresis

`EnvPresetController` implements count-based hysteresis [3]: a candidate preset must persist for `hysteresis_frames` consecutive frames (default 18) before becoming stable. This is correct for preventing flickering but introduces a fixed latency of ~2 seconds (18 frames at 9 thermal fps for the thermal-tick path, or ~0.3 s at 60 FPS NIR if classification runs every frame). The default of 18 is reasonable for most transitions but may be too slow for rapidly-appearing glare (e.g. vehicle headlights entering the field of view).

### 4.2 The Missing Layer: Probability EMA Pre-Filter

The `EnvPresetController` operates on discrete preset names, not on probabilities. This means it cannot distinguish "barely crossed the threshold" from "strongly triggered" transitions. Adding an EMA at the probability level — before the argmax that selects a preset name — provides soft debouncing:

```python
# Pseudocode — before calling select_env_preset_from_tags
p_smooth = alpha * p_raw + (1 - alpha) * p_smooth_prev
tags_confident = {tag for tag, p in zip(all_tags, p_smooth) if p > CONFIDENCE_THRESHOLD}
preset_desired = select_env_preset_from_tags(tags_confident)
preset_stable = env_controller.update(preset_desired)
```

For an RF with `predict_proba`, `p_raw` is the normalized vote fraction. Research on temporal label smoothing [14] confirms that smoothing strength should decrease as the signal becomes more certain and increase near decision boundaries — exactly what an EMA achieves by weighting recent frames more when they are consistent.

### 4.3 The Policy Layer: Latched Preset vs. Raw Probabilities

A critical design principle: the policy state (which preset parameters are applied to the pipeline) must be the **latched** output from `EnvPresetController`, not the raw prediction. The display state (what the operator sees in the HUD) may optionally show the raw probability as a secondary indicator. This separation exists implicitly in `main.py` — `env_stable_for_hud` is derived from `env_controller.stable_name` [7] — but should be made explicit in any ML-driven upgrade.

### 4.4 Asymmetric Hysteresis for Glare and Rain

Not all transitions should be symmetric. Glare onset is fast (< 5 frames) but glare recovery is slow (the optical path needs frames to re-converge). A glare preset should activate quickly (low `hysteresis_frames` for the glare tag) but deactivate slowly. Rain onset is detectable immediately from streak features but removal should persist briefly after the rain stops (wet lens continues to affect image quality). The existing `EnvPresetController` does not support per-tag asymmetric hysteresis. A minimal extension would add `fast_tags = {"glare", "glare_heavy"}` with a shorter streak requirement of, e.g., 5 frames, and a longer deactivation delay.

### 4.5 `env_classification_interval` and Stale Inference

`env_classification_interval` is set to 15 frames (default) [2], meaning the auto-rule runs every 15 NIR frames. At 60 FPS this is every 0.25 s — adequate. At 30 FPS it is every 0.5 s. The ML inference thread runs at a similar interval (`ML_INFERENCE_INTERVAL=15`). The stale-safe pattern already in place — pipeline continues with last-known preset while inference runs — is correct. The EMA smoothing can operate at the inference interval, not at the display frame rate.

---

## 5. Finding 4: NIR Optical Pipeline Levers

### 5.1 Overview of `HybridNIREnhancer`

`HybridNIREnhancer` (nir_pipeline.py) implements a multi-stage optical pipeline [4]:

1. **Guided filter** (optional, gated by `nir_guided_filter_enable=False` default): thermal upsampled to 320×240 used as guide for NIR denoising via `cv.ximgproc.guidedFilter(radius=8, eps=1e-4)`.
2. **Night-mode detection** (Schmitt trigger at 0.40/0.48 × 255): when `is_night_mode=True`, converts to grayscale then back to 3-channel BGR, boosting SNR by √3 over per-channel processing.
3. **Dark/bright channel prior**: computes dark-channel and bright-channel maps with configurable patch size (default 5), updated every `update_rate=8` frames.
4. **Atmosphere estimation**: top-5% bright pixels used as veil estimate, temporally smoothed via a 10-frame deque.
5. **CLAHE in LAB**: three fixed CLAHE objects selected by `avg_brightness` — `very_dark` (clipLimit=3.0, tile=4×4), `dark` (2.0, 6×6), `medium` (1.5, 8×8). The `clahe_clip_scale` parameter (default 1.0) multiplies all three clip limits.
6. **Weight-map boost**: CLAHE enhanced L channel boosted by `base_boost × weight_map` where base_boost = 1.8 / 1.45 / 1.1 for very_dark / dark / medium.
7. **Unsharp mask**: `detail_strength` (default 0.25) applied as `l_enh + k*(l_enh - GaussianBlur)`.
8. **Night brightness boost** (merged into CLAHE pass): 2.2× / 1.6× / 1.3× depending on current brightness quartile.
9. **Color correction**: 30% shift toward atmosphere estimate.

### 5.2 CLAHE Parameter Tuning per ENV Class

CLAHE's two parameters — clip limit and tile grid size — have well-understood effects [12]:

- **Higher clip limit** → more aggressive local contrast enhancement → more noise amplification.
- **Smaller tile grid** → more localized enhancement → better for scenes with strong local contrast variation (glare, backlight).
- **Larger tile grid** → more global effect → better for globally dim scenes (night_clear, nir_night).

Literature on CLAHE hyperparameter selection confirms SSIM-optimal parameters cluster differently for dim vs. high-contrast scenes [12]. The codebase exposes `nir_enhancer_clahe_clip_scale` which scales all three fixed clip limits proportionally. Per-ENV recommendations:

| ENV class | `clahe_clip_scale` direction | Rationale |
|-----------|------------------------------|-----------|
| `nir_night` | ↑ 1.3–1.5× | Scene is very dark; aggressive enhancement acceptable as IR fill is uniform (low noise variance) |
| `night_clear` | ↑ 1.2–1.3× | Open sky, less ambient — need contrast boost for terrain features |
| `normal_night` | default 1.0 | Urban ambient provides moderate luminance; default levels appropriate |
| `fog` | ↓ 0.7–0.8× | Fog scattering amplified by high clip limit; lower clip reduces haze artifact amplification |
| `glare` | ↓ 0.6–0.7× | CLAHE will severely over-enhance the halo region around a glare source; lower clip reduces ring artifacts |
| `backlight` | default 1.0 | Subject is underexposed but background is not; global clip limit is a poor fit; tile size matters more |
| `rain` | default 1.0 | Rain streaks are high-frequency; CLAHE does not remove them, so large clip is not harmful |
| `transition` | ↑ 1.1× (in twilight direction) | Intermediate luminance; modest boost |
| `normal_day` | skip enhancer | When env=`normal_day`, the CLAUDE.md note suggests skipping NIR enhancement entirely (saves 3–5 ms) |

### 5.3 Night-Mode Schmitt Trigger Tuning

The Schmitt trigger in `HybridNIREnhancer` uses hard-coded thresholds (off=0.48, on=0.40). These were tuned for the general case. The `nir_night` class specifically implies active IR illumination; in that regime, saturation is already near-zero, making the grayscale conversion semantically correct. The `night_clear` class implies no IR assist — natural starlight illumination — where the frame may be so dark that `cur_bright < 0.10`, well inside the Schmitt "on" region.

If the ML classifier predicts `nir_night` with high confidence, the grayscale conversion can be forced without waiting for the Schmitt brightness threshold, eliminating an up-to-8-frame lag in the Schmitt's update_rate=8 window.

### 5.4 Detail Strength and the Unsharp Pass

`detail_strength=0.25` (default) adds moderate sharpening [4]. For fog and rain conditions, this sharpening amplifies noise and artifact edges. Recommended: suppress to 0.10–0.15 in `fog` and `rain` presets (by adding `nir_enhancer_detail_strength` as an `opt_overrides` key if it is elevated to `_VALID_OPT_OVERRIDES_KEYS`). For `night_clear` and `nir_night` where object edges are the primary cue, 0.30–0.35 may improve target discrimination.

### 5.5 Guided Filter Opportunity

The guided filter (`nir_guided_filter_enable=False`, requires opencv-contrib) uses the upsampled thermal frame as a guide to denoise the NIR frame. This is directly applicable to fog conditions where thermal has better signal-to-noise than NIR. Cost: 3–5 ms per frame per `main.py` comments [7]. In a fog preset, enabling this on RPi 4B is feasible if thermal is available and the frame budget allows. Multi-scale guided filtering for thermal infrared is well-established [15] and provides detail-preserving denoising superior to bilateral filtering in the presence of structured edges.

---

## 6. Finding 5: Thermal Pipeline Levers

### 6.1 Architecture of `ThermalProcessor`

`ThermalProcessor` runs a sequential pipeline on each new thermal frame [6]:

```
raw_frame
  → ThermalTemporalFilter (3DNR EMA, alpha=thermal_3dnr_alpha)
  → KalmanThermalBackground.update(denoised)
  → [detect_src = mix of raw and denoised, controlled by detect_raw_mix]
  → [detail_preserve_detect: Sobel gradient → morph kernel / fg_threshold tuning]
  → [thermal_anti_glare on detect_src, if detect_use_anti_glare=True]
  → [bilateral on denoised, if bilateral_display_enable=True]
  → thermal_agc (percentile stretch)
  → thermal_anti_glare (on AGC output)
  → thermal_edge_enhance (Laplacian, strength=thermal_display_edge_strength)
  → heat_map from KalmanThermalBackground.get_heat_map(floor=thermal_floor)
  → fg_mask from KalmanThermalBackground.get_foreground_mask(threshold=thermal_fg_threshold)
```

Output is a 4-tuple: `(denoised, enhanced, heat_map, fg_mask)`.

### 6.2 3DNR Alpha — `thermal_3dnr_alpha`

The `ThermalTemporalFilter` computes `ema = alpha * frame + (1 - alpha) * ema_prev`. A higher alpha gives more weight to the current frame (faster response, more noise); a lower alpha gives more weight to history (slower response, less noise) [6].

At 9 thermal FPS, the time constant in frames is approximately `1/(1-alpha)`. At alpha=0.72 (default after the recent change from 0.78), time constant ≈ 3.6 frames ≈ 0.4 s. Existing presets use:

- `night` and `nir_night` implied: 0.72 (code default)
- `fog`: 0.58 (faster, to track moving fog patterns)
- `haze`: 0.60
- `low_light`: 0.68
- `clear_day`: 0.62
- `low_light_cluttered`: 0.66
- `night_fog`: 0.65

Research on multi-frame infrared temporal filtering [16] confirms that EMA time constants should be shorter (higher alpha) when the thermal scene is expected to change rapidly (active targets in fog, rain) and longer (lower alpha) for static-background surveillance. The existing preset settings correctly implement this intuition: fog uses 0.58 (shorter time constant = more responsive) while night uses 0.72 (longer = smoother). The `rain` class lacks a preset; literature suggests 0.60–0.65 is appropriate for rain because raindrops on the lens create high-frequency noise that 3DNR attenuates, but the background scene may be occluded rapidly.

### 6.3 Bilateral Filter — Display Smoothing

The bilateral filter (`thermal_bilateral_display_enable=True`, `d=5`, `sigma_color=15.0`, `sigma_space=5.0`) runs on the denoised frame before AGC and edge enhance [6]. It preserves edges while smoothing thermal pixel noise. On the 80×62 thermal frame, this is fast (< 0.5 ms on ARM). Recommendations:

- **`fog`**: keep bilateral enabled; reduce `sigma_space` to 3.0 to tighten spatial smoothing (fog creates gradual thermal gradients; tight bilateral preserves warmer edges).
- **`glare` / `glare_heavy`**: disable bilateral or increase `sigma_color` to 25.0 — glare causes very strong local thermal gradients that bilateral should not smooth.
- **`rain`**: keep enabled at default — bilateral suppresses rain drop thermal artifacts without losing target edges.

### 6.4 AGC Percentile Stretch — `thermal_agc_low_pct` / `thermal_agc_high_pct`

The `thermal_agc` function performs a percentile stretch from `[lo, hi]` to `[0, 255]` [6]. The default in CONFIG uses display-path AGC of 1.5% / 99.2% (tighter stretch = less clipping at tails). The opt-profile baseline sets 2.0% / 98.0%. Recommendations:

- **`fog`**: tighten to 1.0% / 96.0% — fog scenes have compressed thermal dynamic range; a wider stretch amplifies the available contrast. The existing `haze_preset` already does this via `thermal_agc_low_pct=1.0`, `thermal_agc_high_pct=96.0`.
- **`glare`**: widen to 2.5% / 99.5% — glare creates extremely bright thermal hot spots (infrared-bright road surfaces, metal objects in sun); widening clips these to prevent them dominating the stretch.
- **`nir_night` / `night_clear`**: tighten low to 0.5% — cold scenes have important detail at the low end.

### 6.5 Edge Enhancement — `thermal_display_edge_strength`

Laplacian edge enhance (default 0.25, reduced from 0.35 per recent comment) [2] sharpens thermal edges. It amplifies sensor noise if applied too strongly. Per-ENV recommendations:

- **`fog`**: reduce to 0.15 — fog thermal images have soft gradients; strong edge enhance creates ringing.
- **`glare`**: increase to 0.35 — glare suppresses global contrast but edges remain useful.
- **`rain`**: reduce to 0.10 — rain artifacts on the MI48 dome create false edges.
- **`nir_night` / `night_clear`**: increase to 0.30–0.35 — in cold dark scenes, edge enhancement is the primary way to separate targets from background.

### 6.6 Background Model — `thermal_floor` and `thermal_fg_threshold`

`KalmanThermalBackground` computes a per-pixel background estimate; `get_heat_map(floor)` subtracts the floor to suppress near-zero differences [6]. A higher `thermal_floor` requires a larger temperature delta before a pixel registers as "hot." `thermal_fg_threshold` determines the binary foreground mask.

- **`fog`**: raise both — fog increases thermal noise floor; `thermal_floor=4.0`, `thermal_fg_threshold=20.0` (already in fog preset [3]).
- **`glare`** (with `thermal_detect_use_anti_glare=True`): keep floor at 2.0 but raise `thermal_fg_threshold` to 22.0 — glare creates broad bright regions that should not register as foreground.
- **`rain`**: raise `thermal_floor` to 3.5 — raindrops on the MI48 dome create thermal anomalies that pollute the heat map.
- **`night_clear`**: lower to `thermal_floor=1.5`, `thermal_fg_threshold=15.0` — cold dry-air scenes have very clean thermal backgrounds; even small targets should register.

---

## 7. Finding 6: Display Pipeline Levers

### 7.1 `display_grade_and_cap_bgr` — The Six-Parameter Color Grade

`display_grade_and_cap_bgr` performs a single BGR↔LAB pass applying six adjustments plus a luminance cap [5]:

- `brightness` [−0.12, 0.12] × 255: shifts the L channel.
- `contrast` [0.9, 1.12]: scales L around 128.
- `saturation` [0.9, 1.08]: scales the a and b channels around 128.
- `warmth` [−0.05, 0.05] × 40: shifts the a channel (green/magenta axis).
- `shadows` [−0.05, 0.05] × 40: lifts/pushes L in the shadow region (L < 128).
- `highlights` [−0.05, 0.05] × 35: lifts/pushes L in the highlight region (L > 200).
- `display_l_max`: luminance cap — clips L to this value, preventing the display from being overwhelmed by isolated bright regions.

All existing presets populate `display_grade` [3]. Key observations:

**For `fog`:** The preset sets `contrast=1.12`, `saturation=0.9`. The 1.12 contrast boost amplifies the reduced-contrast fog scene. Saturation reduction prevents the over-saturated appearance that CLAHE can produce on already-desaturated fog scenes. The `brightness=0.0` leaves the base luminance unchanged. Adding `shadows=0.03` would lift shadow detail slightly, useful in night-fog.

**For `glare_heavy`:** `brightness=−0.03`, `contrast=0.94`, `saturation=0.92`, `highlights=−0.05` [3]. This is correct: reducing contrast and highlights prevents the display from being washed out by the glare region. The `display_l_max=232` (reduced from default 240) further clips peak luminance.

**For `backlight`:** `brightness=0.02`, `contrast=1.08`, `saturation=0.95`, `shadows=0.02`, `highlights=−0.03` [3]. Lifting shadows compensates for the underexposed subject. The highlights reduction prevents the already-bright background from saturating the display. A stronger `shadows=0.04–0.06` may further improve subject visibility.

**For `night_clear`:** The `night` preset applies `brightness=0.04`, `contrast=1.05`, `display_l_max=236` [3]. For `night_clear` specifically, where the scene is very dark and the optical target is critical, `shadows=0.04` and `contrast=1.08` would improve target-background separation.

**For `nir_night`:** No dedicated preset exists. The pipeline falls through to `normal_night`-like behavior. A dedicated `nir_night` preset with `brightness=0.06`, `contrast=1.05`, `saturation=0.85` (since nir_night frames are desaturated), and `display_l_max=238` would improve HUD consistency.

### 7.2 `DisplayTemporalGlareBlend` — `temporal_prev_weight`

The temporal glare blend (`DisplayTemporalGlareBlend`) applies IIR blending when `apply_blend=True` (i.e., when glare is detected): `out = (1 - prev_weight) * current + prev_weight * previous` [5]. This stabilizes the display against rapid illumination fluctuations from glare sources.

`temporal_prev_weight` in opt profiles:
- `baseline`: 0.42
- `static_scan`: 0.50 (more temporal smoothing for static surveillance)
- `handheld_pan`: 0.34 (less, to allow rapid scene changes)
- `high_glare`: 0.48 (more, to suppress aggressive glare flicker)

The ENV preset `glare_heavy` sets `temporal_prev_weight=0.48` [3]. For `rain`, where raindrops create high-frequency flicker, setting `temporal_prev_weight=0.40–0.45` during the rain regime would reduce perceived flicker without over-smoothing the scene.

### 7.3 Luminance Cap — `display_l_max` and `display_l_max_when_glare`

The two luminance cap values create a conditional clamp: when glare is detected (`glare_nir=True`), `display_l_max_when_glare` is applied; otherwise `display_l_max` [2]. All presets reduce `display_l_max_when_glare` by 30–46 DN below `display_l_max`. The interaction with the display_grade contrast parameter creates a compressor-like effect: the contrast boost stretches L, but the cap clips the highlights, resulting in increased shadow/midtone detail with controlled highlight.

**Recommendation:** In `fog` conditions, the cap is less critical (highlights are naturally suppressed by fog) and could be relaxed to `display_l_max=242` to allow more dynamic range. In `glare_heavy`, the existing 194 cap is aggressive but correct.

---

## 8. Finding 7: E1 and MAD Anomaly Detector Tuning

### 8.1 E1 Detector Architecture

`ThermalAnomalyDetectorLite` (E1) computes per-pixel local z-scores from the source frame (a mix of `raw` and `denoised` controlled by `feature_e1_raw_mix`), thresholds by `feature_e1_z_thresh`, and intersects with the heat map at `feature_e1_heat_thresh` [6]. Blobs below `feature_e1_min_area` are discarded.

`ThermalMADAnomalyDetector` (MAD) replaces the local z-score with a Median Absolute Deviation (MAD) score over foreground pixels, providing robustness to asymmetric thermal distributions (e.g., when part of the frame is sky/cold) [6]. MAD triggers only after `temporal_window=3` consecutive anomaly frames, reducing single-frame noise triggers.

### 8.2 Per-ENV Tuning of E1 / MAD

Existing presets already tune E1 for several conditions [3]:

| Preset | `e1_z_thresh` | `e1_min_area` | Rationale |
|--------|--------------|---------------|-----------|
| `default` | 1.25 | 10 | baseline |
| `night` | 1.05 | 8 | Night scenes have low thermal variance; lower threshold catches faint targets |
| `fog` | 1.08 | 10 | Fog reduces thermal variance moderately |
| `glare_heavy` | 1.55 | — | Glare creates large spurious hot regions; raise threshold to avoid false alerts |
| `backlight` | 1.45 | — | Sunlit backgrounds create thermal artifacts |
| `day_glare` | 1.50 | — | Combined effect |
| `cluttered_bg` | 1.65 | 16 | Clutter creates many small hot spots |

Missing configurations:
- **`rain`**: recommend `e1_z_thresh=1.35`, `e1_min_area=12`. Rain drops on the MI48 dome create small hot spots; raising the minimum area and threshold prevents phantom blob detection.
- **`nir_night`**: recommend `e1_z_thresh=1.00`, `e1_min_area=6`. In dark-and-cold scenes, even small thermal targets are significant.
- **`transition`**: recommend `e1_z_thresh=1.20`, `e1_min_area=10`. Intermediate settings appropriate.
- **`backlight_high_contrast`**: already coded at `e1_z_thresh=1.50`.

The `feature_e1_raw_mix=0.45` default mixes 45% raw and 55% denoised for the detection source [6]. In high-noise conditions (`fog`, `rain`), reducing this to 0.25 (more denoised) reduces false positives. In `night_clear` and `nir_night` where fast-moving cold targets may be missed if over-smoothed, increasing to 0.60 keeps sharper edges.

### 8.3 Interaction of E1 with Jerk Gate

When `jerk_active=True` (motion jerk detected), E1 automatically adds 0.30 to `z_thresh` and increases heat_thresh by 8.0 [6]. This prevents motion blur during camera pan from triggering false anomaly detections. This jerk compensation is already well-designed; no ENV-specific changes needed beyond keeping the appropriate base threshold.

---

## 9. Finding 8: Per-Condition Processing Policies

### 9.1 `fog`

Fog is the condition with the most complete existing preset support [3]. Key principles from literature [11, 24, 15]:

Near-infrared light scatters in fog (Mie scattering is wavelength-dependent but still significant in NIR). LWIR thermal radiation penetrates fog far better than visible or NIR — this is the primary motivation for the `fusion_alpha_boost` in the fog preset. The boost increases thermal weight in the fusion blend.

However, `fusion_alpha_boost` is not the only lever:

1. **CLAHE clip reduction** (nir side): set `clahe_clip_scale=0.75` to prevent CLAHE from amplifying the fog-scattered haze as contrast.
2. **Thermal 3DNR alpha reduction** (thermal_3dnr_alpha=0.58, already in preset): allows the thermal channel to track moving fog patterns more responsively.
3. **Thermal floor increase** (thermal_floor=4.0, already in preset): suppresses the low-level thermal noise that fog adds.
4. **Display contrast boost** (contrast=1.12, already in preset): recovers global contrast lost to fog scattering.
5. **Guided filter enable** (speculative): enables guided-filter denoising using thermal as guide for NIR. The fog preset is the single best candidate for enabling `nir_guided_filter_enable=True` since thermal is clean and NIR is degraded.

The existing `night_fog` compound preset already combines night and fog levers correctly. An additional `nir_night_fog` compound (not currently in the codebase) would apply nir_night illumination + fog interference adjustments simultaneously.

### 9.2 `rain`

No dedicated `rain` preset exists in the codebase [3]. This is a gap. Rain affects:

- **NIR branch**: rain streaks appear as bright near-vertical streaks (high spatial frequency). Reducing `detail_strength` to 0.10 prevents sharpening from amplifying streaks. CLAHE at default levels is adequate.
- **Thermal branch**: raindrops on the MI48 dome create cold spots (rain is colder than ambient). These read as negative heat spots. The Kalman background model absorbs rain patterns slowly (Kalman gain ~0.1 at steady state); persistent rain will eventually update the background. Setting `thermal_3dnr_alpha=0.60–0.65` smooths rain-induced thermal noise. Reducing `thermal_floor` to 2.0 (lower sensitivity) would reduce false positives from dome droplets.
- **Display grade**: rain reduces scene contrast moderately. `contrast=1.08`, `brightness=0.01` would compensate.
- **E1**: raise `e1_z_thresh=1.35` and `e1_min_area=12` to prevent dome droplets from triggering false alerts.

Temporal median filtering (K=3 frames, ~0.5 MB RAM at 640×480) is the most effective classical technique for rain streak removal [17] but requires maintaining a 3-frame circular buffer — feasible on RPi 4B. This is not currently implemented in the pipeline. A lightweight implementation in the NIR path before `HybridNIREnhancer.process()` would remove most streaks before CLAHE amplifies them.

### 9.3 `glare` and `glare_heavy`

Glare from vehicle headlights or direct sun is the condition most likely to damage image quality rapidly. The existing `glare_heavy` preset is well-designed [3]:

- `nir_high_pct=93.0`, `nir_saturate_at=228.0`: tighter percentile and saturation threshold for the NIR anti-glare compressor (`nir_anti_glare_bgr`).
- `display_l_max=232`, `display_l_max_when_glare=194`: aggressive luminance cap.
- `temporal_prev_weight=0.48`: strong temporal smoothing.
- `thermal_detect_use_anti_glare=True`: applies thermal anti-glare gamma to the detection source.
- `e1_z_thresh=1.55`: high threshold to suppress glare-induced thermal hot spots.

The `nir_anti_glare_bgr` function applies A1-lite tone mapping when `p95 >= saturate_at` [4]. This is effectively a soft shoulder compressor in the LAB domain. Research on GS-AGC [8] proposes adaptive gamma per luminance region, which provides more spatial control than a global soft shoulder. GS-AGC's regional approach would be beneficial for the mixed glare+dark background scenario common in binocular use (vehicle headlights in the center of the frame against a dark background). However, GS-AGC adds ~2–4 ms per frame — within budget for a 50 ms target if applied at 320×240 resolution.

The `DisplayTemporalGlareBlend` is only applied when a glare trigger fires (`apply_blend=True`). In `glare` situations, this is always applied. The `reset=True` case (jerk detected) correctly discards the history to allow rapid scene change.

### 9.4 `backlight`

Backlight (subject against bright background) is fundamentally a local exposure problem. The optical sensor (single exposure) cannot simultaneously expose correctly for both subject and background. Key levers:

1. **`nir_high_pct=92.0`, `nir_saturate_at=226.0`** (existing preset): compresses the bright background before enhancement. This is correct — preventing the background from saturating the NIR compressor.
2. **`display_grade` shadows lift** (`shadows=0.02`, existing): partially compensates for the underexposed subject.
3. **Retinex-style processing** (not implemented): single-scale Retinex (SSR) divides the image by a Gaussian-blurred version, effectively normalizing for illumination variation. This is computationally affordable at 320×240 and would be the most principled approach for backlight. Cost: one Gaussian blur + division ≈ 1–2 ms.
4. **Zone metering** (speculative): the existing `feature_sector_center_deg` parameter in CONFIG [2] suggests an intent to support sector-based analysis. Extending this to adjust NIR enhancement based on which sector the subject occupies would improve backlight compensation.

Thermal is less useful here: the background's thermal signature may dominate similarly to the optical. However, if the subject is a warm body against a cool or neutral background, thermal maintains contrast even when optical does not — reinforcing the case for keeping `fusion_alpha` at or slightly above default in backlight conditions.

### 9.5 `night_clear`

This is the primary optical night-vision target environment (per the `feature_schema.py` comment [1]). It represents the case where there is no ambient artificial light but sky is clear — starlight or moonlight. Characteristics:

- Very low absolute luminance (< 10 lux).
- No fog, no rain, no strong point sources.
- NIR assist (if available) provides most of the scene illumination.
- Thermal is critical for target detection against cold terrain.

The `night` preset handles this [3]: `temporal_prev_weight=0.50`, `display_l_max=236`, `nir_gamma=0.70`, `e1_z_thresh=1.05`, `e1_min_area=8`, `thermal_3dnr_alpha=0.72`.

Gaps: no change to `thermal_floor` (should lower to 1.5 for clean cold scenes), no `clahe_clip_scale` override (should increase to 1.2–1.3). The `nir_gamma=0.70` applies a power-law tone curve via the opt_override system — this is separate from the HybridNIREnhancer CLAHE boost and provides a global brightness lift on the NIR output that is appropriate for night_clear.

### 9.6 `normal_night`

Urban street-lit night: 10–100 lux, mixed artificial and ambient. NIR captures a grayscale-like image (partially IR-contaminated depending on illuminator). Thermal remains useful for target discrimination against warm urban infrastructure. The pipeline defaults plus minor boosts are appropriate. The `night` or `low_light` preset covers this depending on the exact brightness level.

The `infer_env_tags_auto_rule` already distinguishes `low_light` (nir_b_ema < 38.0 when not night_mode) from `night` (is_night_mode=True) [3]. This is a reasonable proxy but misses the case where urban sodium-vapor streetlights provide moderate brightness but poor color rendering — in that case `nir_saturation_mean` would be low despite moderate brightness.

### 9.7 `nir_night`

Dark scene with active IR illumination, mono-like output. The `HybridNIREnhancer` already converts to grayscale internally [4]. Processing recommendations:

- Skip color correction (`_color_correct` operates on the atmosphere estimate A, but in IR-dominant mode the atmosphere estimate is meaningless for color). The function still runs; it shifts toward the estimated atmosphere which is near-gray — effectively a no-op that wastes a few operations.
- `nir_saturation_mean` (feature 8) is near-zero; this is the key discriminator.
- A dedicated `nir_night` preset should be added with: `brightness=0.06`, `contrast=1.05`, `saturation=0.80` (since the frame is gray, saturation of 1.0 in the display grade produces garish noise; 0.80 cleans this), `thermal_3dnr_alpha=0.72`, `e1_z_thresh=1.00`, `e1_min_area=6`.

### 9.8 `transition`

Dawn and dusk. The most challenging state for hysteresis because the illumination is continuously changing over ~30 minutes. The existing `EnvPresetController` with 18-frame hysteresis introduces a lag but prevents oscillation. During transition:

- NIR brightness is changing continuously; CLAHE level is switching (very_dark → dark → medium) as `avg_brightness` crosses thresholds.
- Thermal is relatively stable (ambient temperature changes slowly).
- The display grade should track gracefully: using `shadows=0.01–0.03` with `contrast=1.05–1.08` provides a soft boost as the scene darkens.

**Recommendation:** treat `transition` as a passthrough preset that does not drastically change any single lever, but damps oscillation by keeping hysteresis at 25–30 frames (longer than for other conditions to prevent rapid cycling at the twilight boundary).

### 9.9 `normal_day`

Normal daytime. The CLAUDE.md documentation notes that when `env_label` is BRIGHT/DAY, the NIR enhancer should be skipped entirely to save 3–5 ms/frame. The `clear_day` preset exists [3] and sets `temporal_prev_weight=0.34` (fast temporal response), `display_l_max=242`, `nir_high_pct=96.0`, `nir_saturate_at=236.0`, `saturation=1.08` (slightly boosted color). The NIR enhancer skip logic requires an explicit gate in `main.py` based on the stable ENV label.

### 9.10 Compound Combinations Not Currently in Presets

| Combination | Existing preset? | Recommendation |
|-------------|-----------------|----------------|
| `nir_night` + `fog` | None | New preset: combine nir_night display_grade with fog thermal levers; `fusion_alpha_boost=0.08`, `thermal_floor=4.0`, `thermal_3dnr_alpha=0.62` |
| `normal_night` + `rain` | None | New preset: `temporal_prev_weight=0.44`, `thermal_3dnr_alpha=0.63`, `e1_z_thresh=1.35`, `e1_min_area=12`, `contrast=1.08` |
| `night_clear` + `glare` | None (rare but possible: moon + vehicle) | Adapt `night` preset with `e1_z_thresh=1.40`, `nir_high_pct=93.0` |
| `transition` + `fog` | None | Lower `fusion_alpha_boost=0.06`, transition-appropriate display grade |

---

## 10. Finding 9: Fusion Alpha and Its Role in the Full Lever Set

### 10.1 What `fusion_alpha` Actually Controls

In the pipeline, `fusion_alpha` is the base blend weight determining how much thermal contributes to the fused output frame. The `fusion_alpha_boost` in opt_overrides additively increases this base value [2]. At `fusion_alpha=0.55` + `fusion_alpha_boost=0.12` (fog preset), thermal contributes 67% and NIR 33% to the fused frame.

This single number collapses all spatial, frequency, and environmental variation into one scalar. The Laplacian pyramid approach (described in the prior v2 report) would allow per-frequency-band alpha (e.g., thermal dominates at low frequencies for structure, NIR dominates at high frequencies for detail), but this is not currently implemented.

### 10.2 Why Overweighting This Lever Is a Mistake

`fusion_alpha_boost` is the most visible knob — it has a dramatic perceptual effect and is easy to tune by eye. But it interacts with all other levers: a high alpha boost makes the display thermal-dominant, which means the thermal 3DNR alpha, bilateral, AGC, and edge strength parameters become more critical to the final output quality. If thermal AGC is poorly set and then fusion alpha is boosted, the badly-graded thermal dominates the fused frame.

The correct sequence is: first tune thermal (3DNR, bilateral, AGC, edge strength, floor) to produce a high-quality thermal signal, then tune NIR (CLAHE, night mode, detail strength) to produce a high-quality NIR signal, then tune fusion alpha to achieve the desired blend. Setting fusion alpha first inverts this order.

### 10.3 Per-ENV Fusion Alpha Assessment

| ENV class | `fusion_alpha_boost` | Rationale |
|-----------|---------------------|-----------|
| `fog` | +0.12 (existing) | Thermal penetrates fog; NIR is degraded |
| `haze` | +0.06 (existing) | Partial penetration |
| `night_fog` | +0.10 (existing) | Fog at night; both sensors degraded but thermal still superior |
| `night_clear` | 0.0 | NIR is primary; thermal provides target detection via E1 but should not dominate |
| `nir_night` | +0.05 (suggested) | IR-assist makes NIR monochromatic; thermal adds thermodynamic information |
| `glare` | −0.05 (suggested, requires negative boost support) | In severe glare, thermal of the glare source is not infrared — headlights are visible/UV sources, not LWIR; fusing more thermal does not improve glare handling |
| `rain` | 0.0 | Rain equally affects NIR and thermal; no benefit to raising thermal weight |
| `backlight` | 0.0 | Thermal may have different contrast than expected |
| `normal_day` | 0.0 or skip fusion | Daytime NIR is nearly RGB; thermal is less useful |

---

## 11. Synthesis and Insights

### 11.1 The Tag-Set Architecture Is Sound; Close the Gap to ML

The existing `infer_env_tags_auto_rule` + `EnvPresetController` architecture is a competent heuristic baseline. Its primary limitation is that the rule function only uses five signal dimensions (`nir_b_ema`, `is_night_mode`, `nir_gray_std`, `glare_nir`, `haze_config_on`) while the full `FEATURE_SET_CORE` has eleven. Features 3 (`nir_entropy`), 4 (`nir_p95`), 7 (`nir_dark_fraction`), and 8 (`nir_saturation_mean`) contain discriminative information that the current rules ignore — particularly for distinguishing `nir_night` from `night_clear` and for detecting `fog` without the `haze_config_on` flag.

The ML inference thread already exists and is observe-only. Enabling it to drive tag selection (while keeping `EnvPresetController` for hysteresis) would close this gap with minimal code change.

### 11.2 The Thermal Pipeline Has the Most Uncharted Tuning Space

Of the eight lever families, the thermal pipeline (3DNR alpha, bilateral, AGC percentiles, edge strength, floor, fg_threshold) has the least per-ENV tuning in existing presets. Only `fog`, `haze`, and `night_fog` set `thermal_3dnr_alpha`; only `fog` and `haze` set `thermal_floor` and `thermal_fg_threshold`. The `rain`, `glare` (beyond anti_glare flag), `backlight`, `nir_night`, and `transition` classes lack thermal-specific tuning.

### 11.3 Display Grade Is the Highest Perceptual Leverage Per Millisecond

`display_grade_and_cap_bgr` runs as a single BGR↔LAB round trip [5]. Adding ENV-specific shadow lift, highlight roll-off, or saturation adjustment costs zero additional computation beyond parameter assignment. This is the highest leverage improvement with zero CPU cost. Every ENV class that currently has an empty or minimal `display_grade` dict should be given at least a `shadows`, `highlights`, and `saturation` entry.

### 11.4 `nir_night` is the Most Under-Served Class

`nir_night` has no dedicated preset in the codebase. The classifier is the ninth entry in `ENV_CLASSES`. During dark scenes with active IR, the HybridNIREnhancer correctly converts to grayscale, but the downstream display grading, E1 tuning, and thermal parameters remain at defaults. A `nir_night` preset should be one of the first additions.

### 11.5 Compound Presets Should Be Generated Systematically

The current four compound presets (`night_fog`, `day_glare`, `low_light_cluttered`, `backlight_high_contrast`) were added manually [3]. As the set of ENV classes grows or ML multi-label output becomes available, systematic generation of compound presets (or soft interpolation between single-class presets) will become necessary. One approach: represent each preset as a parameter vector and interpolate linearly weighted by tag confidence. This avoids the combinatorial explosion of manual compound presets while respecting the structure of the tag-set output.

---

## 12. Limitations and Caveats

1. **No live hardware validation.** All parameter recommendations are derived from codebase analysis, literature principles, and domain knowledge. Actual optimal values on the MI48 + IMX290 hardware may differ. The codebase's own inline comments document several iterations of tuning (e.g., `thermal_3dnr_alpha` changed from 0.78 to 0.72; `thermal_display_raw_mix` from 0.28 to 0.10), suggesting that hardware validation on the device is essential before deploying any suggested change.

2. **Training data for ML classifier is limited.** The offline pipeline produces training data from session JSONL logs. Without sufficient labeled examples of all nine ENV classes — particularly rare classes like `rain`, `transition`, and field-collected `glare` and `backlight` — an ML classifier will be unreliable for those classes.

3. **Compound preset gap.** The four existing compound presets do not cover all plausible cross-axis combinations (Section 9.10 lists four unaddressed combinations). Each requires hardware field collection to validate.

4. **Guided filter cost.** The recommendation to enable `nir_guided_filter_enable=True` in fog conditions is contingent on `opencv-contrib-python` being installed (`ximgproc` module). The default RPi 4B image may not include this; installation adds a large package. The 3–5 ms cost must be validated on-device.

5. **Rain streak removal.** The temporal median filter recommendation (K=3 frames) is a well-established classical approach [17] but is not currently in the codebase. Its implementation requires maintaining a 3-frame circular buffer at 640×480 uint8 ≈ 1.8 MB RAM — acceptable on a 4 GB RPi but requires explicit allocation.

6. **`nir_saturation_mean` validity.** Feature 8 (`nir_saturation_mean`) is computed as the mean HSV S-channel on a small BGR frame [1]. In IR-dominant mode, this is near-zero (correct discriminator for `nir_night`). In fog, it also drops (fog desaturates). These two cases — nir_night and fog — have similar saturation signatures but different total brightness; the classifier must use both features together to distinguish them.

7. **`transition` class coverage.** The `transition` class is the least studied. No compound presets involve it, and the `infer_env_tags_auto_rule` has no explicit rule for it. A sunrise/sunset detector based on `hour_of_day_sin` / `hour_of_day_cos` (features 9/10 in CORE) combined with `nir_brightness_delta_10f` (in TEMPORAL set) would be a natural rule-based or ML-derived trigger.

---

## 13. Recommendations

### Immediate (zero additional dependencies, pure parameter changes)

1. **Add a `nir_night` preset to `ENV_PRESETS`** in `env_presets.py`: `brightness=0.06`, `contrast=1.05`, `saturation=0.80`, `display_l_max=238`, `thermal_3dnr_alpha=0.72`, `e1_z_thresh=1.00`, `e1_min_area=6`. Wire it into `select_env_preset_from_tags` when `nir_night` tag is present.

2. **Add a `rain` preset to `ENV_PRESETS`**: `temporal_prev_weight=0.44`, `contrast=1.08`, `brightness=0.01`, `thermal_3dnr_alpha=0.62`, `e1_z_thresh=1.35`, `e1_min_area=12`, `thermal_floor=3.5`.

3. **Add `clahe_clip_scale` to `_VALID_OPT_OVERRIDES_KEYS`** in `config.py` and `nir_enhancer_detail_strength` if not already present, so they can be set in preset `opt_overrides`.

4. **Lower `thermal_floor` and `thermal_fg_threshold` in the `night` and `night_clear` presets**: set `thermal_floor=1.5`, `thermal_fg_threshold=15.0` for cleaner cold-scene foreground detection.

5. **Add shadow and highlight entries to `night_clear` display_grade**: `shadows=0.03`, `highlights=0.0`, `contrast=1.08`.

### Short-Term (minor code additions)

6. **Enable ML inference path** (`ML_INFERENCE_ENABLED=True`) with the existing `rf_from_logs_baseline.json` model as observe-only, then progressively: (a) log disagreements between rule and ML, (b) retrain with corrections, (c) promote ML output to drive tag selection.

7. **Add a 3-frame temporal median filter** in the NIR path (before `HybridNIREnhancer.process()`) as an optional rain-mode denoiser, gated on the `rain` ENV label.

8. **Implement asymmetric hysteresis for glare tags**: add `fast_tags` and `fast_streak` attributes to `EnvPresetController` with streak requirement of 5 frames for glare onset and 25 frames for glare release.

9. **Add `nir_night` + `fog` and `normal_night` + `rain` compound presets** with parameters from Section 9.10.

### Longer-Term (architectural changes)

10. **Implement Classifier Chains (2-link)**: illumination RF → interference RF using appended posteriors. This requires two RF models, a modified `FeatureExtractor`, and a new model file. The ML inference thread already provides the threading infrastructure.

11. **Add EMA probability smoothing** at the ML inference stage: maintain `p_smooth` as a rolling average, gate tag activation on `p > 0.55` in smoothed space rather than 0.5 in raw space.

12. **Investigate per-level Laplacian pyramid alpha** in the fusion blend: different thermal weights for low/mid/high frequency bands. Cost: ~3–5 ms for pyramid construction; potentially worth it in fog conditions where thermal provides macroscopic structure but NIR provides texture.

---

## Appendix A: Label / Combination → Candidate Knobs → Sources → RPi Budget

| Label / Combo | Key Knobs | Primary Sources | RPi ≤50ms fit? |
|---------------|-----------|----------------|----------------|
| `night_clear` | `thermal_floor=1.5`, `fg_threshold=15.0`, `clahe_clip_scale=1.2`, `e1_z_thresh=1.05`, `shadows=0.03`, `nir_gamma=0.70` | [3], [4], [6] | Yes (all param changes) |
| `normal_night` | `night` preset (existing), `temporal_prev_weight=0.50`, `nir_gamma=0.70` | [3] | Yes |
| `nir_night` | NEW preset: `saturation=0.80`, `brightness=0.06`, `clahe_clip_scale=1.3`, `e1_z_thresh=1.00`, skip color_correct | [1], [4], [22], [23] | Yes |
| `normal_day` | Skip `HybridNIREnhancer` entirely; `clear_day` preset | [3], CLAUDE.md | Yes (+3–5ms savings) |
| `fog` | `fusion_alpha_boost=0.12`, `thermal_3dnr_alpha=0.58`, `clahe_clip_scale=0.75`, `contrast=1.12`, `thermal_floor=4.0`, `thermal_fg_threshold=20.0` | [3], [11], [15] | Yes (all param) |
| `fog` + guided filter | `nir_guided_filter_enable=True` | [4], [15] | Marginal (+3–5ms; verify on device) |
| `rain` | NEW preset: `temporal_prev_weight=0.44`, `contrast=1.08`, `thermal_3dnr_alpha=0.62`, `e1_z_thresh=1.35`, `e1_min_area=12`, `thermal_floor=3.5` | [17], [18], [26] | Yes |
| `rain` + temporal median | 3-frame median buffer (NIR path) | [17] | Yes (~2ms, 1.8MB RAM) |
| `glare` | `nir_high_pct=93.0`, `nir_saturate_at=228.0`, `e1_z_thresh=1.55`, `display_l_max=232`, `temporal_prev_weight=0.48`, `thermal_detect_use_anti_glare=True` | [3], [8] | Yes |
| `glare` + GS-AGC | Regional luminance adaptive gamma (not in codebase) | [8] | Marginal (+2–4ms @ 320×240) |
| `backlight` | `nir_high_pct=92.0`, `shadows=0.04`, `contrast=1.08`, `highlights=-0.03` | [3], [19] | Yes |
| `backlight` + Retinex | Single-scale Retinex (not in codebase) | [19] | Yes (~1–2ms @ 320×240) |
| `transition` | Longer hysteresis (25 frames), `contrast=1.05`, `shadows=0.01` | [3], [14] | Yes |
| `night_fog` | Compound preset (existing): `fusion_alpha_boost=0.10`, `thermal_3dnr_alpha=0.65`, `nir_gamma=0.70` | [3] | Yes |
| `nir_night` + `fog` | NEW compound: fog thermal levers + nir_night display grade | [1], [3], [6] | Yes |
| `normal_night` + `rain` | NEW compound: rain thermal + night display grade | [3], [17], [6] | Yes |
| `day_glare` | `display_l_max=230`, `temporal_prev_weight=0.46`, `e1_z_thresh=1.50` | [3] | Yes |
| `low_light_cluttered` | `e1_z_thresh=1.45`, `e1_min_area=14`, `temporal_prev_weight=0.44` | [3] | Yes |
| `backlight_high_contrast` | `nir_high_pct=91.0`, `nir_saturate_at=225.0`, `e1_z_thresh=1.50` | [3] | Yes |
| All NIR night classes | Enable `nir_guided_filter_enable` if thermal available | [4], [24] | Conditional |
| All fog/haze | Raise `fusion_alpha_boost`, lower `thermal_3dnr_alpha`, lower `clahe_clip_scale` | [3], [11] | Yes |
| All glare variants | Raise `e1_z_thresh`, lower `display_l_max_when_glare`, enable `thermal_detect_use_anti_glare` | [3], [8] | Yes |

---

## Appendix B: CONFIG Key and ENV_PRESET Mapping

**Legend:** ✅ = code-backed (key exists in CONFIG or _VALID_OPT_OVERRIDES_KEYS as read); 🔧 = key exists in CONFIG but not yet in _VALID_OPT_OVERRIDES_KEYS; ⚗️ = speculative / not currently in codebase.

| Recommendation | CONFIG / Preset Key | Status | Notes |
|----------------|--------------------|----|-------|
| `thermal_3dnr_alpha` per ENV | `thermal_3dnr_alpha` in preset dict | ✅ | Used in 6 of 13 existing presets |
| `thermal_floor` per ENV | `thermal_floor` in `opt_overrides` | ✅ | In `_VALID_OPT_OVERRIDES_KEYS` |
| `thermal_fg_threshold` per ENV | `thermal_fg_threshold` in `opt_overrides` | ✅ | In `_VALID_OPT_OVERRIDES_KEYS` |
| `fusion_alpha_boost` per ENV | `fusion_alpha_boost` in `opt_overrides` | ✅ | In `_VALID_OPT_OVERRIDES_KEYS` |
| `display_l_max` per ENV | `display_l_max` in `opt_overrides` | ✅ | In `_VALID_OPT_OVERRIDES_KEYS` |
| `temporal_prev_weight` per ENV | `temporal_prev_weight` in `opt_overrides` | ✅ | In `_VALID_OPT_OVERRIDES_KEYS` |
| `nir_gamma` per ENV | `nir_gamma` in `opt_overrides` | ✅ | In `_VALID_OPT_OVERRIDES_KEYS` |
| `nir_high_pct` per ENV | `nir_high_pct` in `opt_overrides` | ✅ | In `_VALID_OPT_OVERRIDES_KEYS` |
| `nir_saturate_at` per ENV | `nir_saturate_at` in `opt_overrides` | ✅ | In `_VALID_OPT_OVERRIDES_KEYS` |
| `thermal_agc_low_pct` per ENV | `thermal_agc_low_pct` in `opt_overrides` | ✅ | In `_VALID_OPT_OVERRIDES_KEYS` |
| `thermal_agc_high_pct` per ENV | `thermal_agc_high_pct` in `opt_overrides` | ✅ | In `_VALID_OPT_OVERRIDES_KEYS` |
| `thermal_edge_strength` per ENV | `thermal_edge_strength` in `opt_overrides` | ✅ | In `_VALID_OPT_OVERRIDES_KEYS` |
| `clahe_clip_scale` per ENV | `nir_enhancer_clahe_clip_scale` in CONFIG | 🔧 | In CONFIG but NOT in `_VALID_OPT_OVERRIDES_KEYS`; add `"nir_enhancer_clahe_clip_scale"` to the frozenset |
| `detail_strength` per ENV | `nir_enhancer_detail_strength` in CONFIG | 🔧 | In CONFIG but NOT in `_VALID_OPT_OVERRIDES_KEYS`; add to frozenset |
| Bilateral sigma_color per ENV | `thermal_bilateral_sigma_color` in CONFIG | 🔧 | In CONFIG; add `"thermal_bilateral_sigma_color"`, `"thermal_bilateral_sigma_space"` to frozenset |
| `e1_z_thresh` per ENV | `e1_overrides` dict in each preset | ✅ | Via `apply_e1_overrides` in `env_presets.py` |
| `e1_min_area` per ENV | `e1_overrides` dict in each preset | ✅ | Via `apply_e1_overrides` |
| `e1_raw_mix` per ENV | `e1_overrides` via `feature_e1_raw_mix` | ✅ | Via `apply_e1_overrides` |
| Display grade per ENV | `display_grade` dict in each preset | ✅ | All fields: brightness, contrast, saturation, warmth, shadows, highlights |
| `nir_night` preset | Not in ENV_PRESETS | ⚗️ | New dict to add |
| `rain` preset | Not in ENV_PRESETS | ⚗️ | New dict to add |
| `nir_night_fog` compound preset | Not in ENV_PRESETS | ⚗️ | New dict to add |
| `normal_night_rain` compound preset | Not in ENV_PRESETS | ⚗️ | New dict to add |
| Guided filter in fog | `nir_guided_filter_enable` in CONFIG | 🔧 | In CONFIG; needs to be settable per-ENV via preset; add to `_VALID_OPT_OVERRIDES_KEYS` |
| 3-frame temporal median for rain | Not in codebase | ⚗️ | New NIR preprocessing step |
| Classifier Chains ML | Not in codebase (architecture change) | ⚗️ | Requires two RF models and modified inference thread |
| EMA probability smoothing | Not in codebase | ⚗️ | Add before tag activation in ML inference path |
| Asymmetric hysteresis | Not in EnvPresetController | ⚗️ | Extend EnvPresetController with per-tag streak counts |
| Laplacian pyramid alpha | Not in codebase | ⚗️ | New fusion method in display_pipeline or main.py |

---

## Bibliography

[1] SmartBinocular codebase — `src/smartbinocular/feature_schema.py`. ENV_CLASSES, FEATURE_SET_CORE, FeatureRecord. Accessed 2026-04-15.

[2] SmartBinocular codebase — `src/smartbinocular/config.py`. CONFIG dict, `_VALID_OPT_OVERRIDES_KEYS`, `resolve_optimization_profile`. Accessed 2026-04-15.

[3] SmartBinocular codebase — `src/smartbinocular/env_presets.py`. `build_env_presets`, `infer_env_tags_auto_rule`, `EnvPresetController`, `select_env_preset_from_tags`, `merge_opt_cfg_with_preset`. Accessed 2026-04-15.

[4] SmartBinocular codebase — `src/smartbinocular/nir_pipeline.py`. `HybridNIREnhancer`, CLAHE levels, Schmitt trigger, A1-lite tone map, guided filter integration. Accessed 2026-04-15.

[5] SmartBinocular codebase — `src/smartbinocular/display_pipeline.py`. `display_grade_and_cap_bgr`, `DisplayTemporalGlareBlend`. Accessed 2026-04-15.

[6] SmartBinocular codebase — `src/smartbinocular/thermal_pipeline.py`. `ThermalProcessor`, `KalmanThermalBackground`, `ThermalTemporalFilter`, `ThermalAnomalyDetectorLite`, `ThermalMADAnomalyDetector`. Accessed 2026-04-15.

[7] SmartBinocular codebase — `src/smartbinocular/main.py`. Pipeline composition, ML inference thread initialization, ENV preset application. Accessed 2026-04-15.

[8] Pu, Z. et al. "GS-AGC: An Adaptive Glare Suppression Algorithm Based on Regional Brightness Perception." *Applied Sciences* 14(4):1426, 2024. https://www.mdpi.com/2076-3417/14/4/1426

[9] Arenas-García, J. et al. "Classifier Chain Networks for Multi-Label Classification." arXiv:2411.02638, November 2024. https://arxiv.org/abs/2411.02638; also ScienceDirect 2025 https://www.sciencedirect.com/science/article/pii/S0957417425016690

[10] AEFusion: Adaptive Enhanced Fusion of Visible and Infrared Images for Night Vision. *Remote Sensing* 17(18):3129, 2025. https://www.mdpi.com/2072-4292/17/18/3129

[11] FIVFusion: Fog-free infrared and visible image fusion. *Journal of King Saud University — Computer and Information Sciences*, 2025. https://link.springer.com/article/10.1007/s44443-025-00309-7

[12] Kaur, M. & Singh, D. "Machine learning hyperparameter selection for Contrast Limited Adaptive Histogram Equalization." *EURASIP Journal on Image and Video Processing*, 2019:45. https://link.springer.com/article/10.1186/s13640-019-0445-4

[13] Saricicek, I. & Uysal, G. "Online Adaptive Kalman Filtering for Real-Time Anomaly Detection in Wireless Sensor Networks." *Sensors* 24(15):5046, 2024. https://www.mdpi.com/1424-8220/24/15/5046

[14] Yèche, H. et al. "Temporal Label Smoothing for Early Event Prediction." *Proc. ICML 2023*, PMLR 202. https://proceedings.mlr.press/v202/yeche23a/yeche23a.pdf

[15] Wang, X. et al. "Thermal Infrared-Image-Enhancement Algorithm Based on Multi-Scale Guided Filtering." *Fire* 7(6):192, 2024. https://www.mdpi.com/2571-6255/7/6/192

[16] DEMNet: Dual-Encoder-Decoder Multi-Frame Infrared Image Denoising, preprints.org 2025. https://www.preprints.org/frontend/manuscript/70e7715960f6b8d4c6b771178be8309a/download_pub

[17] Yang, J. et al. "Self-Learning Video Rain Streak Removal: When Cyclic Consistency Meets Temporal Correspondence." *Proc. CVPR 2020*. https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Self-Learning_Video_Rain_Streak_Removal_When_Cyclic_Consistency_Meets_Temporal_CVPR_2020_paper.pdf

[18] Zhao, H. et al. "A Lightweight Network for Real-Time Rain Streaks and Rain Accumulation Removal from Single Images Captured by AVs." *Applied Sciences* 13(1):219, 2022. https://www.mdpi.com/2076-3417/13/1/219

[19] Reinhard, E. et al. "High-Dynamic-Range Tone Mapping in Intelligent Automotive Systems." *Sensors* 23(12):5767, 2023. https://www.mdpi.com/1424-8220/23/12/5767

[20] Zhang, L. et al. "Adaptive Near-Infrared and Visible Fusion for Fast Image Enhancement." *IEEE Transactions on Geoscience and Remote Sensing*, 2020. https://ieeexplore.ieee.org/abstract/document/8918077

[21] "NIR Cameras in Embedded Vision – Advantages and Applications." TechNexion, 2024. https://www.technexion.com/resources/nir-cameras-in-embedded-vision-advantages-and-applications/

[22] "Arducam 2MP Ultra Low Light STARVIS IMX290 Motorized IR-CUT Camera for Raspberry Pi." Arducam product page. https://www.arducam.com/presale-arducam-2mp-ultra-low-light-starvis-imx290-motorzied-ir-cut-camera-for-raspberry-pi.html

[23] "IR-Cut Filter in Embedded Vision." TechNexion, 2024. https://www.technexion.com/resources/ir-cut-filter-in-embedded-vision/

[24] Zhang, Y. et al. "Advances and challenges in infrared-visible image fusion: a comprehensive review." *Artificial Intelligence Review*, 2025. https://link.springer.com/article/10.1007/s10462-025-11426-0

[25] Li, Y. et al. "A fast hardware accelerator for nighttime fog removal based on image fusion." *Microelectronics Journal* 148, 2024. https://www.sciencedirect.com/science/article/abs/pii/S0167926024001202

[26] Porav, H. et al. "A Review of Detection and Removal of Raindrops in Automotive Vision Systems." *Sensors* 21(15), 2021. https://pmc.ncbi.nlm.nih.gov/articles/PMC8321291/

[27] Zhou, W. et al. "Beyond Night Visibility: Adaptive Multi-Scale Fusion of Infrared and Visible Images." arXiv 2403.01083, 2024. https://arxiv.org/html/2403.01083

[28] High-Radix Taylor-Optimized Tone Mapping Processor for Adaptive 4K HDR Video. *Sensors* 25(13):3887, 2025. https://www.mdpi.com/1424-8220/25/13/3887

[29] "Visible and Near-Infrared Image Acquisition and Fusion for Night Surveillance." *Chemosensors* 9(4):75. https://www.mdpi.com/2227-9040/9/4/75

[30] scikit-learn documentation. "Multiclass and multioutput algorithms — Classifier Chains." https://scikit-learn.org/stable/modules/multiclass.html

---

## Methodology Appendix

**Research Mode:** UltraDeep (v3.0)

**Phase 0 — Codebase Read (mandatory, pre-synthesis):** All six source files listed in Section 1.3 were read using the Read tool before any web search was initiated. Symbol names, CONFIG key names, and preset structures are cited from these primary sources throughout.

**Phase 3 — Retrieval:** Fourteen parallel web searches were executed covering multi-label classification, temporal stabilization, CLAHE tuning, thermal pipeline techniques, fog/rain/glare/backlight per-condition literature, NIR sensor physics, and fusion alpha approaches.

**Phase 4 — Triangulation:** Claims that contradict the codebase were rejected. Literature recommendations were evaluated against the actual pipeline architecture: if a technique is incompatible with the 50 ms budget or requires hardware not present (GPU, NPU), it is flagged in Appendix A.

**Phase 5 — Synthesis:** The two-axis illumination×interference structure of the nine ENV classes was derived from codebase taxonomy + domain physics, not from any single source.

**Phase 6–7 — Critique and Refinement:** A specific balance requirement was enforced: no lever family (particularly `fusion_alpha_boost`) should receive disproportionate coverage relative to its share of the actual processing budget impact.

**Key assumptions:**
- Hardware: RPi 4B, IMX290 Bayer RGGB, MI48 LWIR 80×62. No GPU/NPU.
- Software: OpenCV 4.x with NEON acceleration. `opencv-contrib` not guaranteed (guided filter flagged as conditional).
- Frame budget: ≤50 ms end-to-end.
- ML classifier: RF trained on `FEATURE_SET_OPTICAL_ONLY` (11 features) as primary; thermal features additive when MI48 available.
- Thermal FPS: ~9 FPS; NIR FPS: up to 60 FPS; ENV classification interval: 15 frames.
