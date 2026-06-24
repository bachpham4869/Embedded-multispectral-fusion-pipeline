# SmartBinocular Pipeline Evidence Register

**Version:** 1.0 — 2026-04-25  
**Purpose:** Thesis-defense traceability document. For every pipeline stage, sub-block,
algorithm, and tunable number: where the code lives, where the justification lives,
and where the literature was simplified for the RPi4 ≤50 ms/frame budget.  
**Regen script:** Deferred (V1). Part D tables are hand-authored against the codebase
at the commit captured in `models/model_registry.json`. No `docs/_evidence_sources.yaml`
or `tools/regen_evidence_register.py` exists in this release.  
**Evidence types used throughout this document:**

| Tag | Meaning |
|-----|---------|
| **CIT** | IEEE citation in `LINK.md` §II (index number given) |
| **SWEEP** | Measured sweep CSV in `docs/tables/timing/`, `docs/tables/iqa/`, or `docs/tables/ml/` (see [`tables/README.md`](tables/README.md)) |
| **SIDECAR** | Training metric in `models/production/env_classifier.json` |
| **TEST** | Pinned / asserted by a test file (test ID or function name given) |
| **DOC** | Argued in a research report or rationale doc in `docs/` |
| **UNVERIFIED** | No measurement exists; documented as hand-tuned; see proposed experiment |

---

## Table of Contents

- [Part A — Pipeline Map](#part-a--pipeline-map)
  - [A.1 Frame data flow](#a1-frame-data-flow)
  - [A.2 Thread model](#a2-thread-model)
  - [A.3 Frame budget](#a3-frame-budget)
- [Part B — Stages and Blocks](#part-b--stages-and-blocks)
  - [B.1 Capture](#b1-capture)
  - [B.2 FrameCache](#b2-framecache)
  - [B.3 NIR Enhancement — Buckets A–F](#b3-nir-enhancement--buckets-af)
  - [B.4 Thermal Preprocessing](#b4-thermal-preprocessing)
  - [B.5 Motion](#b5-motion)
  - [B.6 ENV Classification](#b6-env-classification)
  - [B.7 ML Inference](#b7-ml-inference)
  - [B.8 Fusion Blend](#b8-fusion-blend)
  - [B.9 Display Grade / L-cap](#b9-display-grade--l-cap)
  - [B.10 HUD](#b10-hud)
  - [B.11 Metrics](#b11-metrics)
- [Part C — Algorithms (Non-trivial)](#part-c--algorithms-non-trivial)
  - [C.1 NIR Night Enhancement (HybridNIREnhancer / Bucket A)](#c1-nir-night-enhancement-hybridnirenhancer--bucket-a)
  - [C.2 CLAHE per bucket](#c2-clahe-per-bucket)
  - [C.3 Multi-scale fusion (explicit gap)](#c3-multi-scale-fusion-explicit-gap)
  - [C.4 DCP dehaze (Bucket D)](#c4-dcp-dehaze-bucket-d)
  - [C.5 Rain temporal median (Bucket E)](#c5-rain-temporal-median-bucket-e)
  - [C.6 Thermal background — EMA and Kalman](#c6-thermal-background--ema-and-kalman)
  - [C.7 MAD anomaly detector](#c7-mad-anomaly-detector)
  - [C.8 Sparse optical flow (Lucas–Kanade)](#c8-sparse-optical-flow-lucaskanade)
  - [C.9 Random Forest classifier with isotonic calibration](#c9-random-forest-classifier-with-isotonic-calibration)
  - [C.10 Posterior EMA on ML probabilities](#c10-posterior-ema-on-ml-probabilities)
  - [C.11 Top-1/top-2 compositor](#c11-top-1top-2-compositor)
  - [C.12 ENV preset hysteresis (Schmitt-trigger analogy)](#c12-env-preset-hysteresis-schmitt-trigger-analogy)
- [Part D — Parameters & Thresholds](#part-d--parameters--thresholds)
  - [D.1 ML gates](#d1-ml-gates)
  - [D.2 EMA / smoothing](#d2-ema--smoothing)
  - [D.3 Fusion blending](#d3-fusion-blending)
  - [D.4 Display grade caps](#d4-display-grade-caps)
  - [D.5 CLAHE per bucket](#d5-clahe-per-bucket)
  - [D.6 Guided filter](#d6-guided-filter)
  - [D.7 Thermal Kalman](#d7-thermal-kalman)
  - [D.8 MAD anomaly](#d8-mad-anomaly)
  - [D.9 Motion / LK](#d9-motion--lk)
  - [D.10 ENV-FSM hysteresis](#d10-env-fsm-hysteresis)
  - [D.11 Capture timing](#d11-capture-timing)
  - [D.12 ML model meta](#d12-ml-model-meta)
  - [D.X Proposed minimal experiments (UNVERIFIED rows)](#dx-proposed-minimal-experiments-unverified-rows)
- [Thesis open questions (frozen scope)](#thesis-open-questions-frozen-scope)

---

## Thesis open questions (frozen scope)

**Version:** 1.0 — 2026-04-27 (merged from archived `docs/THESIS_OPEN_QUESTIONS.md`)

These three questions define measurable thesis claims. Do not add questions after Phase 1 begins without updating this register and re-aligning the evaluation protocol.

**Source context:** [`legacy/md/THESIS_MENTOR_REVIEW_SYNTHESIS.md`](../legacy/md/THESIS_MENTOR_REVIEW_SYNTHESIS.md) §A (executive summary gaps).

### Q1 — Does fused display improve a defined task over NIR-only?

**Claim under investigation:** The NIR + thermal fusion composite (`mode = "fusion"`) produces better task performance than NIR-alone (`mode = "nir"`) on a specified detection or recognition task.

**Why it matters:** `fusion_alpha = 0` is **not** a valid NIR-only baseline — fusion arithmetic still runs and `fusion_alpha_boost` presets can shift the alpha at runtime. The correct ablation arm is `--mode nir` (`main.py:1283–1315`), which bypasses the entire fusion composite block.

**Measurement protocol (Phase 5.1):**

- Pick one defined task on held-out scenes (e.g. E1 anomaly detections per minute, or a 50-frame hand-labeled detection set).
- Run the same recorded sequence twice: `--mode fusion` and `--mode nir`, with the same `pipeline_config_sha256` to ensure identical configuration.
- Report per-task metric + `stage_timing_ms` deltas.

**Defensible null result:** “Fusion adds thermal overlay at a latency cost of X ms with no measurable detection improvement on the test set” is a valid thesis contribution, provided the test protocol is defined and the measurement is honest.

### Q2 — Does A–F bucket dispatch improve quality over a fixed single path?

**Claim under investigation:** ENV-class-driven optical bucket selection (A–F dispatch via `OPTICAL_BUCKET_DISPATCH`) produces better NIR image quality or pipeline efficiency than routing all frames through a single fixed bucket (e.g. Bucket A for all ENV classes).

**Measurement protocol (Phase 5.2):**

- Add `BUCKET_OVERRIDE` env var honored at `nir_pipeline.py:resolve_optical_bucket` (or equivalent config key in `RPI_THROUGHPUT_MAX_DEFAULTS`).
- Lock all frames to Bucket A; re-run the same recording.
- Compare: offline IQA metrics from `tools/batch_nir_enhancer.py` (stratified by ENV class) + ML confidence stability + `nir_bucket` stage timing mean and σ.

**Non-claim:** This does not claim optimal bucket parameters per ENV class — only that dispatch outperforms or matches a fixed Bucket A path on the current thresholds.

### Q3 — Is E1 a validated detector, or an anomaly indicator?

**Claim under investigation:** The MAD-based thermal anomaly detector (`ThermalMADAnomalyDetector`) produces E1 alerts with measurable precision/recall on a ground-truth labeled set.

**Decision gate (Phase 5.3):**

- If ≥50 labeled frames of `fusion_captures/` are hand-annotated: compute precision/recall vs MAD detector output → defensible detection claim.
- If labeling is not feasible before defense: demote E1 to “anomaly indicator, not validated detector” in the thesis.

**Minimum defensible framing (fallback):** “E1 outputs a thermal anomaly score; the MAD 0.6745 constant and z-threshold 3.5 are from [67] Leys 2013; no precision/recall evaluation was performed on this dataset.”

### What this thesis does NOT claim

*(Pre-registered non-claims — see also [`docs/eval/manifest_schema.md`](eval/manifest_schema.md))*

1. **≤50 ms/frame steady-state.** Measured `fusion_composite` p50 ≈ 72 ms (Mac host) and mean ~6.3 FPS (RPi4, `session_20260425-211623.json`). The thesis frames this as “optimization target not yet met” and presents Phase 2 post-optimization measurements.

2. **Optimal CLAHE clip per bucket.** Bucket B sweep (Phase 3.5) gives evidence for the range [0.5, 8.0]; it does not claim the chosen value minimizes a ground-truth perceptual metric.

3. **Multi-scale fusion parity.** The pipeline uses `addWeighted` scalar blend — a known simplification of [65] Raskar pyramid fusion (explicitly documented in §C.3 of this register).

4. **BRISQUE as a primary IQA metric.** BRISQUE assumes natural image statistics; NIR night imagery violates those assumptions (Risk R1 in [`legacy/md/OFFLINE_IQA_EVALUATION_AGENT_SYNTHESIS.md`](../legacy/md/OFFLINE_IQA_EVALUATION_AGENT_SYNTHESIS.md)). BRISQUE is logged as an exploratory column only.

5. **HybridNIREnhancer quality on still images equals in-pipeline behavior.** The batch tool uses `update_rate=1` and `reset()` per image — a “still_image_cold_start_mode” that does not replicate temporal EMA state from live video.

---

## Part A — Pipeline Map

### A.1 Frame data flow

```
[MI48 SPI 80×62 ~9 FPS]              [IMX290 Picamera2 640×480 up to 60 FPS]
        │                                            │
  ThermalCapture daemon                        NIRCapture daemon
  (hardware.py:ThermalCapture)           (hardware.py:NIRCapture)
        │                                            │
  ThermalProcessor.process()              FrameCache  (utils.py:build_frame_cache)
   • KalmanThermalBackground              • nir_gray_small cached once per frame
   • ThermalTemporalFilter (3DNR)         • brightness stats reused across blocks
   • ThermalMADAnomalyDetector
        │                                            │
        │                         OPTICAL_BUCKET_DISPATCH
        │                         (nir_pipeline.py:OPTICAL_BUCKET_DISPATCH)
        │                          ENV_CLASS → Bucket A/B/C/D/E/F
        │                                            │
        │              ┌─────────────────────────────┘
        │              │  Bucket A: HybridNIREnhancer (night_clear, normal_night)
        │              │  Bucket B: nir_nir_night_clahe  (nir_night)
        │              │  Bucket C: nir_anti_glare_bgr   (glare, backlight, normal_day)
        │              │  Bucket D: nir_dehaze_lite       (fog)
        │              │  Bucket E: RainTemporalMedian    (rain)
        │              │  Bucket F: nir_transition_blend  (transition)
        │              │
        └──────────────┤
                       │  ENV controller (env_presets.py:EnvPresetController)
                       │   • rule-based: infer_env_tags_auto_rule
                       │   • ML compositor: compose_env_from_ml_top2
                       │   • preset hysteresis: _TRANSITION_HYSTERESIS
                       │
                       │  Fusion composite
                       │   • cv2.addWeighted  (fusion_alpha)
                       │   • fg heat overlay  (thermal mask broadcast)
                       │
                       │  display_grade_and_cap_bgr
                       │   • BGR↔LAB roundtrip, L-cap
                       │   • DisplayTemporalGlareBlend IIR
                       │
                       │  DisplayShakeReducerLite (blend or shift mode)
                       │
                       │  HUD overlay (bearing, sector text)
                       │
                       ▼
                  cv2.imshow() + ThesisRunMetrics → JSONL log
```

### A.2 Thread model

| Thread | Class / function | Owns | Communication |
|--------|-----------------|------|---------------|
| Main loop | `main.py:main` | Frame loop, ENV FSM, fusion, display | Reads from all daemon queues |
| ThermalCapture daemon | `hardware.py:ThermalCapture` | MI48 SPI read, thermal frames | Puts `(frame, ts)` into `thermal_queue` |
| NIRCapture daemon | `hardware.py:NIRCapture` | Picamera2 capture | Puts `(frame, ts)` into `nir_queue` |
| MLInferenceThread daemon | `ml_inference.py:MLInferenceThread` | RF inference + EMA | Writes `MLTop2` to `MLSharedResult` (lock-protected) |
| MLLogger worker | `utils.py:MLLogger` | JSONL session log writes | Reads from `log_queue`; non-blocking |

`MLSharedResult` is the only shared mutable state between the ML thread and the main loop; it is
protected by a `threading.Lock`. The main loop reads the last-posted `MLTop2` without waiting
(stale-safe by design — documented in `CLAUDE.md` §ML Thread Concurrency Model).

### A.3 Frame budget

**Status: PARTIAL — session data available from Mac host; RPi4 cross-session statistics
generated by `tools/measure_stage_timing.py` → `docs/tables/timing/stage_timing_summary.csv`.**

14 session manifests were processed from `fusion_captures/metrics/session_*.json`.
The table below shows cross-session mean/p50/p95/p99 (each data point is one session's
pre-aggregated per-stage mean). "n_sessions" is the count of sessions that reported
each stage.

> **Important:** these figures come from the Mac development host, not a deployed RPi4.
> Wall-clock latency on RPi4 Cortex-A72 (no NEON JIT warm-up, thermal throttling) will
> differ. Full RPi4 per-frame profiling remains a TODO in
> `legacy/md/THESIS_IMPROVEMENT_PLAN.md` §Phase 3b.

| Stage | n_sessions | mean_ms | p50_ms | p95_ms | p99_ms | Evidence |
|-------|-----------|---------|--------|--------|--------|----------|
| framecache | 14 | 10.3 | 9.9 | 12.4 | 12.4 | SWEEP (`stage_timing_summary.csv`) |
| nir_bucket (Bucket A/B/C/D/E/F) | 14 | 29.3 | 24.5 | 47.1 | 48.2 | SWEEP |
| jerk gate | 14 | 3.6 | 3.5 | 5.0 | 5.4 | SWEEP |
| thermal_proc (Kalman + 3DNR + MAD) | 12 | 20.0 | 19.6 | 25.4 | 25.5 | SWEEP |
| blend (NIR + display grade) | 14 | 40.6 | 29.8 | 81.8 | 83.7 | SWEEP |
| fusion_composite | 10 | 67.7 | 72.3 | 89.3 | 92.6 | SWEEP |
| HUD overlay | 14 | 3.7 | 3.9 | 5.1 | 5.1 | SWEEP |
| display | 14 | 2.3 | 2.1 | 3.0 | 3.0 | SWEEP |
| ml_infer (background thread) | 14 | 7.2 | 6.5 | 10.7 | 11.9 | SWEEP |

**Finding:** `fusion_composite` (NIR processing + blend + thermal overlay) averages ~67 ms
and exceeds 89 ms at p95. The individual `nir_bucket` stage alone averages ~29 ms with
p95 ~47 ms. The ≤50 ms/frame end-to-end budget is **not consistently met** in measured
sessions. Optimization opportunities: guided filter is already disabled; Bucket A
HybridNIREnhancer at 320×240 is the dominant cost in night-class sessions.

> **Documentation TODO (`README.md`):** Lines asserting "≤50 ms/frame" and "up to 60 FPS"
> should be updated to note the measured `fusion_composite` p50 of ~72 ms and that the
> 50 ms target is not consistently met. This is an honest thesis contribution.

See also: `legacy/md/THESIS_IMPROVEMENT_PLAN.md` §Phase 3b for the optimization roadmap.

---

## Part B — Stages and Blocks

### B.1 Capture

| Sub-block | File:symbol | Role | Primary citation | In-repo doc | Test |
|-----------|-------------|------|-----------------|-------------|------|
| Thermal frame acquisition | `hardware.py:ThermalCapture` | MI48 SPI read; 80×62 float array ~9 FPS | [19] vendor | `CLAUDE.md` §Hardware | no test |
| NIR frame acquisition | `hardware.py:NIRCapture` | IMX290 via Picamera2; 640×480 BGR up to 60 FPS | [22] Picamera2 | `CLAUDE.md` §Hardware | no test |
| Homography alignment | `hardware.py:load_homography` | Loads pre-computed 3×3 H; scales to work size | Standard pinhole geometry | `CLAUDE.md` §Pipeline | no test |

### B.2 FrameCache

| Sub-block | File:symbol | Role | Primary citation | In-repo doc | Test |
|-----------|-------------|------|-----------------|-------------|------|
| Gray cache builder | `utils.py:build_frame_cache` | One shared small gray per frame; reuses brightness stats | Engineering decision | — | `test_utils.py` |
| Stream skew gate | `utils.py:StreamSkewQualityGate` | Drops frame pairs with excessive timestamp skew | Engineering decision | — | `test_utils.py` |
| NIR gray (cached) | `nir_pipeline.py:nir_compute_gray_cached` | Single green-channel proxy; reused by JerkGate and DisplayShakeReducer | Engineering decision | — | no test |

### B.3 NIR Enhancement — Buckets A–F

| Bucket | ENV classes | File:symbol | Role | Primary citation | Test |
|--------|------------|-------------|------|-----------------|------|
| A | night_clear, normal_night (default) | `nir_pipeline.py:HybridNIREnhancer` | Dark/bright channel + adaptive CLAHE + optional guided filter; proc at 320×240 | [12][13][14][18] | `test_nir_pipeline.py` |
| B | nir_night | `nir_pipeline.py:nir_nir_night_clahe` | Single CLAHE pass on LAB-L; clip clamped [0.5, 8.0] | [15] | `test_nir_pipeline.py` |
| C | glare, backlight, normal_day | `nir_pipeline.py:nir_anti_glare_bgr` | Highlight suppression; percentile-based L normalization | DOC [research_report_v4] | T040–T050 |
| D | fog | `nir_pipeline.py:nir_dehaze_lite` | DCP at 160×120, ω=0.85; no soft matting | [13][36][37] | `test_nir_pipeline.py` |
| E | rain | `nir_pipeline.py:RainTemporalMedian` | N-frame (N=3) median; no streak segmentation | [48] (simplified) | `test_nir_pipeline.py` |
| F | transition | `nir_pipeline.py:nir_transition_blend` | Alpha-blended passthrough during ENV transitions | DOC [research_report_v4] | T040–T050 |

Dispatch table: `nir_pipeline.py:OPTICAL_BUCKET_DISPATCH`  
Resolver function: `nir_pipeline.py:resolve_optical_bucket`

### B.4 Thermal Preprocessing

| Sub-block | File:symbol | Role | Primary citation | Test |
|-----------|-------------|------|-----------------|------|
| Temporal 3DNR (EMA) | `thermal_pipeline.py:ThermalTemporalFilter` | Per-pixel exponential moving average; α=0.65 | [19] vendor (GST-IR 3DNR pattern) | `test_thermal_pipeline.py` |
| Kalman background model | `thermal_pipeline.py:KalmanThermalBackground` | Per-pixel 1-D Kalman; vectorised over 80×62; Q=0.5, R=4.0, P₀=100.0 | [11] Kalman 1960; [73] Welch & Bishop | `test_thermal_pipeline.py` |
| Soft Gaussian fg mask | `thermal_pipeline.py:ThermalProcessor` | Gaussian-weighted foreground probability from Kalman residual | Engineering decision on top of [11] | `test_thermal_pipeline.py` |
| MAD anomaly detector | `thermal_pipeline.py:ThermalMADAnomalyDetector` | Modified-z (0.6745 constant), temporal-window confirmation | [67] Leys 2013 | `test_thermal_pipeline.py` |
| Thermal AGC | `thermal_pipeline.py:thermal_agc` | Adaptive gain-control for display overlay | [19] vendor | no test |
| Edge enhance | `thermal_pipeline.py:thermal_edge_enhance` | Laplacian sharpening for HUD emphasis | Engineering decision | no test |
| Colormap — thermal-only path | `main.py:1345–1346` | `gray_to_thermal_bgr(thermal_vis)` on native ~80×62 frame, then resize; LUT runs on small grid. **VERIFIED — no further optimization.** | Engineering decision | no test |
| Colormap — fusion path | `main.py:1388–1394` | `cv.resize(heat_map, thermal_size)` runs first, then `gray_to_thermal_bgr` on the upscaled grid. **Optimization candidate (Phase 1.4b):** swap order — LUT on native 80×62, then resize. `thermal_size` loaded from `load_homography` (`main.py:320`). | DOC (Phase 0 code trace) | no test |

### B.5 Motion

| Sub-block | File:symbol | Role | Primary citation | Test |
|-----------|-------------|------|-----------------|------|
| JerkGate | `motion.py:JerkGate` | Frame-diff + phase-correlation jerk detection; holds display shake-reducer reset for `hold_frames` | Engineering decision (no direct IEEE ref) | `test_utils.py` |
| SparseOpticalFlowMotion | `motion.py:SparseOpticalFlowMotion` | Lucas–Kanade sparse flow on `nir_160`; max_points=20, refresh_interval=15 | [59] Lucas–Kanade | no test |
| DisplayShakeReducerLite | `motion.py:DisplayShakeReducerLite` | Temporal blend (mode=blend) or sub-pixel phase-correlation shift (mode=shift) | Engineering decision | no test |
| OneEuroFilter1D | `motion.py:OneEuroFilter1D` | Low-pass filter for shift-mode sub-pixel stabilisation | Engineering decision (1€ filter; no direct IEEE ref) | no test |

### B.6 ENV Classification

| Sub-block | File:symbol | Role | Primary citation | Test |
|-----------|-------------|------|-----------------|------|
| Rule-based tag inference | `env_presets.py:infer_env_tags_auto_rule` | Heuristic brightness / contrast / glare tags; ML fallback when confidence < τ₁ | DOC [research_report_v4] | `test_env_compositor.py` (TC01–TC13) |
| ML top-2 compositor | `env_presets.py:compose_env_from_ml_top2` | 10-rule policy mapping (class_1, proba_1, class_2, proba_2) → (env_class, hint) | DOC [research_report_v4]; multi-label background [45][46][47] | TC01–TC13 |
| ENV preset FSM | `env_presets.py:EnvPresetController` | Hysteresis-guarded preset switching; merges opt_cfg slice | Schmitt-trigger analogy [61] | `test_env_preset_hysteresis.py` |
| Preset table | `env_presets.py:build_env_presets` | 20 named presets; ENV_CLASS-aligned ones map directly | DOC [research_report_v4] | `test_env_preset_hysteresis.py` |
| NIR Schmitt hysteresis | `main.py:NIR_NON_NIGHT_RAW_ENTER` / `NIR_NON_NIGHT_RAW_EXIT` | Raw-brightness Schmitt trigger (center=25, hyst=3) for night/non-night switch | [61] (analogy) | no test (config constants) |

### B.7 ML Inference

| Sub-block | File:symbol | Role | Primary citation | Test |
|-----------|-------------|------|-----------------|------|
| ENV classifier (RF) | `ml_inference.py:EnvClassifier` | Random Forest, 12 optical features, isotonic calibration | [63] Breiman; [64] Pedregosa | T026–T036 |
| Posterior EMA | `ml_inference.py:MLPosteriorEMA` | Full-vector EMA on class probabilities; α=0.55 | Engineering decision (no direct IEEE ref) | T031–T036 |
| Inference thread | `ml_inference.py:MLInferenceThread` | Daemon; runs every `ML_INFERENCE_INTERVAL` frames; posts `MLTop2` | Engineering decision | T026–T036 |
| Shared result | `ml_inference.py:MLSharedResult` | Lock-protected last-known ML result | Engineering decision | T026–T036 |
| SHA-256 integrity | `ml_inference.py:EnvClassifier` (load path) | Verifies joblib SHA-256 against `models/model_registry.json` on load | Engineering decision | `test_model_registry_integrity.py` |

### B.8 Fusion Blend

| Sub-block | File:symbol | Role | Primary citation | Test |
|-----------|-------------|------|-----------------|------|
| Additive weighted blend | `main.py` (`_stage_profiler("fusion_composite")`) | `cv2.addWeighted(nir, fusion_alpha, thermal_overlay, 1-fusion_alpha, 0)` | [65] Raskar (simplified — see Part C §C.3) | no test |
| Fg heat overlay | `main.py` (`_stage_profiler("fusion_composite")`) | Thermal soft-mask broadcast over NIR; mode=fg | Engineering decision | no test |
| Secondary hint delta | `env_presets.py:apply_secondary_hint` | Light opt_cfg tweak when top-2 hint present | DOC [research_report_v4] | TC01–TC13 |

### B.9 Display Grade / L-cap

| Sub-block | File:symbol | Role | Primary citation | Test |
|-----------|-------------|------|-----------------|------|
| LAB grade + L-cap | `display_pipeline.py:display_grade_and_cap_bgr` | Single BGR↔LAB roundtrip; brightness/contrast/saturation micro-adjustments; L clamped to `l_max` | Engineering decision | no test |
| Glare L-cap | `config.py:TIER_S_DISPLAY_L_MAX_WHEN_GLARE` | L-cap=208 when glare detected; tighter than default 240 | DOC `../legacy/md/DEPLOY_HARDENING.md` §8 | no test |
| Temporal glare IIR | `display_pipeline.py:DisplayTemporalGlareBlend` | IIR blend prev-frame when glare active; prev_weight=0.42 | Engineering decision | no test |
| Glare gate skip | `config.py:RPI_THROUGHPUT_MAX_DEFAULTS["display_luma_cap_glare_gate"]` | Skip LAB roundtrip ~63% dark-night frames (RPi throughput max) | DOC `../legacy/md/DEPLOY_HARDENING.md` §8 | no test |

### B.10 HUD

| Sub-block | File:symbol | Role | Primary citation | Test |
|-----------|-------------|------|-----------------|------|
| Bearing computation | `hardware.py:bearing_hv_deg_from_uv` | Pinhole bearing estimate H/V degrees | Standard pinhole geometry | no test |
| Sector text | `hardware.py:sector_from_bearing_deg` | 8-sector compass label | Engineering decision | no test |
| E1 blob overlay | `main.py` (phase1_blobs path) | MAD-confirmed thermal anomaly markers on HUD | [67] MAD-based detection | no test |

### B.11 Metrics

| Sub-block | File:symbol | Role | Primary citation | Test |
|-----------|-------------|------|-----------------|------|
| FPS / latency counter | `utils.py:FPSCounter` | Rolling FPS and per-stage latency | Engineering decision | `test_utils.py` |
| Session metrics | `metrics.py:ThesisRunMetrics` | Stage timing, env-class distribution, ML hit rate, ECE tracking | Engineering decision | no test |
| JSON export | `metrics.py:metrics_write_run_manifest` | Writes `session_*.json` to `fusion_captures/metrics/` | Engineering decision | no test |
| BRISQUE score | `nir_pipeline.py:compute_brisque_score` | No-reference IQA on NIR frame (sampled); optional | [71] BRISQUE | no test |
| CONFIG fingerprint | `metrics.py:metrics_write_run_manifest` *(planned Phase 1.2)* | `pipeline_config_sha256`: SHA-256 over allowlist of pipeline-relevant CONFIG keys (`_VALID_OPT_OVERRIDES_KEYS` + extended set); `pipeline_config_keys` lists the allowlist for auditability | DOC (Phase 0 plan; not yet implemented) | no test |
| Throttle snapshot | `metrics.py:ThesisRunMetrics.finalize` *(planned Phase 1.1)* | `experiment_context["throttle_snapshot"]`: vcgencmd throttle flag, temp, CPU freq glob — populated on RPi4, all-None on Mac (soft-fail) | DOC (Phase 0 plan; not yet implemented) | no test |

---

## Part C — Algorithms (Non-trivial)

### C.1 NIR Night Enhancement (HybridNIREnhancer / Bucket A)

| | |
|---|---|
| **Full literature method** | Shi 2018 bright/dark channel prior [12] + He DCP [13][36][37] + He guided filter [14] + LearnOpenCV tutorial [18]: per-frame dark-channel atmosphere estimation, soft-matte transmission, guided-filter refinement at full resolution. |
| **What is implemented** | `nir_pipeline.py:HybridNIREnhancer`: (1) downsample to 320×240 proc canvas; (2) estimate dark-channel atmosphere over small patches; (3) three pre-built CLAHE tiers selected by `avg_brightness` thresholds (very_dark/dark/medium); (4) optional guided-filter denoise on the same 320×240 canvas (disabled by default); (5) dark-scene L-channel boost (2.2/1.6/1.3× by brightness tier); (6) upsample result to display size. |
| **What was cut / approximated** | No per-frame entropy computation. No per-pixel transmission map (replaced by CLAHE tier selection). Single-channel guide (green channel as NIR luminance proxy). Processing resolution 320×240 vs 640×480 native. No soft matting. |
| **Cost saved** | ~6–10 ms/frame estimated vs full-resolution guided-filter transmission map. |
| **Citations** | [12][13][14][18] — see `LINK.md` §II |

### C.2 CLAHE per bucket

| | |
|---|---|
| **Full literature method** | Zuiderveld [15] CLAHE: contrast-limited tile histogram equalization; clip limit prevents noise amplification. Reza [40], Bai [39], Pisano [57], Stark [58], Kim [41], Pizer [68]: medical/satellite-derived guidance on optimal clip. |
| **What is implemented** | Three separate clip regimes: Bucket A uses three pre-built CLAHE instances at `clip_limit × clahe_clip_scale`; Bucket B uses a single clamp `np.clip(3.0 × scale, 0.5, 8.0)`. |
| **What was cut** | No per-frame adaptive clip computation. No per-class sweep to determine optimal clip. Clip values are hand-tuned starting points. |
| **Per-bucket evidence** | See Part D §D.5 for per-row evidence; **Bucket B clamp is PARTIAL — proxy sweep pending** (see `tools/sweep_clahe_clip.py`). |
| **Citations** | [15][39][40][57][58][41][68] |

### C.3 Multi-scale fusion (explicit gap)

| | |
|---|---|
| **Full literature method** | Raskar et al. [65] and Zou et al. [33]: Laplacian pyramid decomposition, per-level fusion by weight maps (focus/saliency), pyramid reconstruction. |
| **What is implemented** | `cv2.addWeighted(nir, fusion_alpha, thermal_overlay, 1−fusion_alpha, 0)` — a single-level linear blend with a scalar `fusion_alpha`. No pyramid decomposition. |
| **Explicit gap** | No multi-scale frequency decomposition. Per-ENV `fusion_alpha_boost` (from presets) partially compensates, but the frequency-band separation benefit of a true Laplacian pyramid is absent. |
| **Rationale** | `addWeighted` is ~1–2 ms on RPi4; full pyramid fusion at 640×480 would add ~5–10 ms, consuming the remaining frame budget. |
| **Citations** | [65][33] — acknowledged gap; not claimed as equivalent |

### C.4 DCP dehaze (Bucket D)

| | |
|---|---|
| **Full literature method** | He et al. [13][36][37]: dark-channel prior, per-pixel transmission map, guided-filter soft matting, atmospheric scattering model inversion at full resolution. Tarel [38] fast alternative; Zhu [54], Berman [55], Li [56] for benchmarks. |
| **What is implemented** | `nir_pipeline.py:nir_dehaze_lite`: DCP at fixed 160×120 downsample (4× spatial reduction); fixed ω=0.85 (retain 15% haze for depth); nearest-neighbour upsample; no soft matting; no guided-filter refinement. |
| **What was cut** | Soft matting (He 2010 guided filter) — saves ~3–5 ms. Per-pixel transmission map refinement (replaced by fixed ω). Full resolution processing (→ 4× fewer pixels). |
| **Cost saved** | ~4–6 ms vs full-resolution DCP with guided filter. |
| **Citations** | [13][36][37][38][54][55][56] |

### C.5 Rain temporal median (Bucket E)

| | |
|---|---|
| **Full literature method** | Kang [48]: rain-streak segmentation, mixture-of-Gaussians model, MAP estimation. Tripathi [49], Gu [50], Li [51]: optical-flow + decomposition alternatives. |
| **What is implemented** | `nir_pipeline.py:RainTemporalMedian`: N-frame (N=3) median over the LAB-L channel. Per-pixel robust estimator; no streak segmentation; no GAN de-raining. |
| **What was cut** | No rain-streak detection or segmentation. No frequency-domain analysis. No GAN-based de-raining (would require GPU). N=3 is a minimal buffer (any increase trades latency vs smoothing quality). |
| **Citations** | [48][49][50][51] — implemented method is the lightweight median-only baseline |

### C.6 Thermal background — EMA and Kalman

| | |
|---|---|
| **Full literature method** | Kalman 1960 [11]; Welch & Bishop [73] implementation guide; Barnich ViBe [9]: sample-based, multi-model background with non-parametric persistence. |
| **What is implemented** | `thermal_pipeline.py:KalmanThermalBackground`: per-pixel 1-D Kalman with vectorised NumPy over 80×62 pixels. Process noise Q=0.5, measurement noise R=4.0 (≈ sensor ±2°C), initial covariance P₀=100.0. Fallback EMA: `thermal_pipeline.py:ThermalTemporalFilter`, α=0.65. |
| **What was cut** | ViBe's multi-sample-per-pixel background model (much richer, 3–5× more memory). Per-pixel Q/R adaptation (replaced by global scalar). Full covariance matrix (replaced by scalar P, valid for 1-D state). |
| **Cost saved** | ~0.5 ms vs a full ViBe implementation; ~2–3× less RAM. |
| **Unverified params** | Q=0.5, R=4.0, P₀=100.0 — hand-tuned; no thermal-sequence sweep committed. See §D.7 and proposed experiment in §D.X. |
| **Citations** | [11][73][9] |

### C.7 MAD anomaly detector

| | |
|---|---|
| **Full literature method** | Leys et al. 2013 [67]: modified z-score using median absolute deviation; 0.6745 constant from normal-distribution MAD consistency factor. |
| **What is implemented** | `thermal_pipeline.py:ThermalMADAnomalyDetector`: `modified_z = 0.6745 × (pixel − median) / (MAD + ε)`; threshold mad_z_thresh=3.5; temporal-window=3 frame confirmation before alert. |
| **What was cut** | Leys recommends 3.5 as the threshold — this implementation uses exactly that value (**CIT**). Temporal-window confirmation (3 frames) is an engineering addition not in the paper. |
| **Citations** | [67] Leys 2013 (0.6745 constant and 3.5 threshold) |

### C.8 Sparse optical flow (Lucas–Kanade)

| | |
|---|---|
| **Full literature method** | Lucas & Kanade 1981 [59]: iterative image registration using local gradient constraints; dense or sparse over detected corners. |
| **What is implemented** | `motion.py:SparseOpticalFlowMotion`: `cv2.calcOpticalFlowPyrLK` on `nir_160` thumbnail; max_points=20 Shi-Tomasi corners; corner refresh every `refresh_interval=15` frames. |
| **What was cut** | Dense flow (Farneback) — too expensive on RPi4 at full resolution. Pyramid levels limited to OpenCV defaults. Only 20 feature points (vs hundreds in full implementations). |
| **Citations** | [59] Lucas–Kanade; OpenCV [21] for Shi-Tomasi + LK implementation |

### C.9 Random Forest classifier with isotonic calibration

| | |
|---|---|
| **Full literature method** | Breiman [63]: ensemble of decision trees, majority vote, out-of-bag error. Pedregosa [64] scikit-learn implementation. Niculescu-Mizil [66]: post-hoc calibration (Platt / isotonic); scikit-learn calibration [74]. |
| **What is implemented** | `models/train_classifier.py`: RF with isotonic calibration wrapper; 12-feature `FEATURE_SET_OPTICAL_ONLY`; `TimeSeriesSplit` CV; macro balanced accuracy scoring over 9 ENV_CLASSES. Production model: `models/production/env_classifier.joblib` / `env_classifier.json`. |
| **What was cut** | Neural classifiers (SwinIR [6], MLP) — too slow for RPi4 inference at 15-frame intervals. Only optical features (no thermal features, no temporal features beyond rolling EMAs) — see `feature_schema.py:FEATURE_SET_OPTICAL_ONLY`. |
| **Measured performance** | cv_bal_acc=0.7408±0.0331 (TimeSeriesSplit); train_bal_acc=0.9509; ECE all classes <0.05 — see sidecar and §D.12. |
| **Citations** | [63][64][66][74] |

### C.10 Posterior EMA on ML probabilities

| | |
|---|---|
| **Implementation** | `ml_inference.py:MLPosteriorEMA`: full 9-class probability vector smoothed with EMA; α=0.55 general; asymmetric glare: up-α=0.85 (fast response to entering glare), down-α=0.45 (slow exit from glare). |
| **Literature status** | **No direct IEEE citation.** This is an engineering decision motivated by: (a) RF posterior spikiness between frame batches at the 15-frame inference interval; (b) asymmetric glare α mimics hysteresis without requiring a second threshold sweep; (c) full-vector smoothing preserves relative class probabilities better than smoothing only the top-1 label. |
| **Evidence** | Tests T031–T036 in `test_ml_inference.py` verify EMA update arithmetic and asymmetric behaviour. α=0.55 and [0.85, 0.45] are **UNVERIFIED — hand-tuned**. |
| **Proposed experiment** | §D.X #1 — EMA alpha stability sweep. |

### C.11 Top-1/top-2 compositor

| | |
|---|---|
| **Implementation** | `env_presets.py:compose_env_from_ml_top2`: 10 ordered rules mapping (class_1, proba_1, class_2, proba_2, τ₁, τ₂) → (env_class, secondary_hint). Core invariant: night-axis and day-axis classes as primary are never displaced by weather hints as env_class — only as hints. |
| **Literature background** | Multi-label classification concepts: Boutell [45], Zhang [46], Tsoumakas [47]. The *specific* 10 rules are an engineering policy. |
| **Evidence** | TC01–TC13 in `test_env_compositor.py` pin each compositor rule against corner cases. Full rationale: `docs/research_report_20260415_env_policy_branching_optical_v4.md`. |

### C.12 ENV preset hysteresis (Schmitt-trigger analogy)

| | |
|---|---|
| **Implementation** | `env_presets.py:_TRANSITION_HYSTERESIS`: per-preset `(onset_frames, decay_frames)` tuples enforced by `env_presets.py:EnvPresetController`. |
| **Literature analogy** | Schmitt trigger / hysteresis switching [61]. |
| **Evidence** | `test_env_preset_hysteresis.py:test_hysteresis_table_values`, `test_glare_onset_2`, `test_glare_decay_20` pin the table entries. Specific onset/decay values are **UNVERIFIED — hand-tuned** (see §D.10). |

---

## Part D — Parameters & Thresholds

> **Pointer policy:** `file:symbol` is preferred over `file:line` throughout Part D.
> Line numbers appear only for module-level constants with no containing function;
> in that case they are suffixed with a short-SHA note.
>
> **Evidence gating:** a parameter can be edited in `src/smartbinocular/` *only* when
> evidence type is SWEEP / SIDECAR / TEST / CIT / DOC **and** the change is recorded
> here with before/after symbols. UNVERIFIED rows = "do not touch yet".
>
> **Regen script:** deferred. These tables are the single source of truth; update
> them when re-running sweeps or retraining.

---

### D.1 ML gates

| Name | Value | File:symbol | Evidence type | Evidence pointer |
|------|-------|------------|---------------|-----------------|
| Primary confidence threshold τ₁ | 0.62 | `config.py:CONFIG["ml_confidence_threshold"]` | SWEEP + SIDECAR | `docs/tables/ml/threshold_sweep.csv` row τ=0.62; `models/production/env_classifier.json:ml_gate_reference.top1_tau_0.62`; `docs/tables/ml/ML_GATE_RATIONALE.md` |
| Secondary confidence threshold τ₂ | 0.20 | `config.py:CONFIG["ml_secondary_confidence_threshold"]` | SWEEP + SIDECAR | `docs/tables/ml/secondary_threshold_sweep.csv` row secondary_tau=0.2; `models/production/env_classifier.json:ml_gate_reference.top2_tau_0.20`; `docs/tables/ml/ML_GATE_RATIONALE.md` |
| Alternative τ₁ (higher F1, more abstention) | 0.65 | Sidecar field only | SIDECAR | `models/production/env_classifier.json:ml_gate_reference.top1_tau_0.65_alternative` — macro_f1_night=0.9866, abstention_rate=0.1491 |
| Glare asymmetric EMA alpha (up) | 0.85 | `config.py:CONFIG["ml_posterior_ema_asym"]` key "glare"[0] | UNVERIFIED | No sweep. See §D.X #1 |
| Glare asymmetric EMA alpha (down) | 0.45 | `config.py:CONFIG["ml_posterior_ema_asym"]` key "glare"[1] | UNVERIFIED | No sweep. See §D.X #1 |

**Sweep format version:** `threshold_sweep.csv` columns are `threshold, abstention_rate, n_accepted, f1_night_clear, f1_normal_night, f1_nir_night, macro_f1_night` (version 1; column-name keyed in `tests/test_threshold_sweep_consistency.py`).

**Consistency test:** `tests/test_threshold_sweep_consistency.py` asserts that the τ₁=0.62 row's
`macro_f1_night` and `abstention_rate` equal the values in `ml_gate_reference.top1_tau_0.62`
(tolerance ±1e-4). Updating only one of the two artifacts will fail the test.

**Secondary consistency test:** `tests/test_secondary_sweep_consistency.py` asserts that the
τ₂=0.20 row's `hint_rate_of_ml` equals the sidecar `ml_gate_reference.top2_tau_0.20.hint_rate_of_ml_active` (±1e-4).

---

### D.2 EMA / smoothing

| Name | Value | File:symbol | Evidence type | Evidence pointer |
|------|-------|------------|---------------|-----------------|
| ML posterior EMA alpha (general) | 0.55 | `config.py:CONFIG["ml_posterior_ema_alpha"]` | UNVERIFIED | Hand-tuned. See §D.X #1 |
| Glare EMA up-alpha | 0.85 | `config.py:CONFIG["ml_posterior_ema_asym"]["glare"][0]` | UNVERIFIED | Hand-tuned. See §D.X #1 |
| Glare EMA down-alpha | 0.45 | `config.py:CONFIG["ml_posterior_ema_asym"]["glare"][1]` | UNVERIFIED | Hand-tuned. See §D.X #1 |
| NIR nir_b_ema alpha (Bucket B brightness EMA) | 1.0 (no EMA; instantaneous) | `nir_pipeline.py:nir_nir_night_clahe` | DOC | `CLAHE_CLIP_SCALE` param passed from preset `nir_enhancer_clahe_clip_scale=1.2` for nir_night |
| 3DNR EMA alpha (ThermalTemporalFilter) | 0.65 | `thermal_pipeline.py:ThermalTemporalFilter.__init__` param `alpha` | UNVERIFIED | Originated from `legacy/py/final_fusion.py`; no sweep. See §D.X #4 |
| Glare temporal IIR prev_weight (DisplayTemporalGlareBlend) | 0.42 | `display_pipeline.py:DisplayTemporalGlareBlend.__init__` param `prev_weight` (= `config.py:TIER_S_GLARE_TEMPORAL_PREV_WEIGHT`) | UNVERIFIED | Hand-tuned. See §D.X #2 |
| OneEuroFilter1D min_cutoff | 1.15 | `motion.py:DisplayShakeReducerLite.__init__` param `one_euro_min_cutoff` | UNVERIFIED | Hand-tuned. See §D.X #3 |
| OneEuroFilter1D beta | 0.018 | `motion.py:DisplayShakeReducerLite.__init__` param `one_euro_beta` | UNVERIFIED | Hand-tuned. See §D.X #3 |
| OneEuroFilter1D d_cutoff | 1.0 | `motion.py:DisplayShakeReducerLite.__init__` param `one_euro_d_cutoff` | UNVERIFIED | Hand-tuned. See §D.X #3 |
| Shake-reducer shift EMA | 0.42 | `motion.py:DisplayShakeReducerLite.__init__` param `shift_ema` | UNVERIFIED | Hand-tuned. See §D.X #3 |
| Adaptive blend base weight | 0.50 | `motion.py:DisplayShakeReducerLite.__init__` param `blend_current_weight` | UNVERIFIED | Hand-tuned |
| Adaptive blend tanh gain | 0.14 | `motion.py:DisplayShakeReducerLite.process` (inline 0.14) | UNVERIFIED | Hand-tuned |
| Adaptive blend offset | −0.07 | `motion.py:DisplayShakeReducerLite.process` (inline −0.07) | UNVERIFIED | Hand-tuned |

---

### D.3 Fusion blending

| Name | Value | File:symbol | Evidence type | Evidence pointer |
|------|-------|------------|---------------|-----------------|
| Base fusion alpha (NIR weight) | 0.55 | `config.py:CONFIG["fusion_alpha"]` | UNVERIFIED | Hand-tuned global default |
| Fog preset fusion_alpha_boost | +0.12 | `env_presets.py:build_env_presets` key "fog" → `fusion_alpha_boost` | DOC | `docs/research_report_20260415_env_policy_branching_optical_v4.md` |
| Haze preset fusion_alpha_boost | +0.06 | `env_presets.py:build_env_presets` key "haze" → `fusion_alpha_boost` | DOC | `docs/research_report_20260415_env_policy_branching_optical_v4.md` |
| Night-fog preset fusion_alpha_boost | +0.10 | `env_presets.py:build_env_presets` key "night_fog" → `fusion_alpha_boost` | DOC | `docs/research_report_20260415_env_policy_branching_optical_v4.md` |
| ML-compositor secondary fog hint boost | +0.05 | `env_presets.py:_SECONDARY_HINT_OPT_DELTA` key "fog" → `fusion_alpha_boost` | DOC | `docs/research_report_20260415_env_policy_branching_optical_v4.md` |
| Fg/gradient mode flag | fg (default) | `config.py:RPI_THROUGHPUT_MAX_DEFAULTS` | DOC | `CLAUDE.md` integration note #2 |
| Apply secondary hint policy | applies delta when hint in `_VALID_SECONDARY_HINTS` | `env_presets.py:apply_secondary_hint` | TEST | TC01–TC13 |

---

### D.4 Display grade caps

| Name | Value | File:symbol | Evidence type | Evidence pointer |
|------|-------|------------|---------------|-----------------|
| Default L-cap | 240 | `config.py:TIER_S_DISPLAY_L_MAX` | DOC | `../legacy/md/DEPLOY_HARDENING.md` §8 |
| Glare L-cap | 208 | `config.py:TIER_S_DISPLAY_L_MAX_WHEN_GLARE` | DOC | `../legacy/md/DEPLOY_HARDENING.md` §8 |
| Glare temporal IIR prev_weight | 0.42 | `config.py:TIER_S_GLARE_TEMPORAL_PREV_WEIGHT` | UNVERIFIED | Hand-tuned. Same as `display_pipeline.py:DisplayTemporalGlareBlend` default |
| night_clear preset display_l_max | 236 | `env_presets.py:build_env_presets` key "night_clear" | DOC | `docs/research_report_20260415_env_policy_branching_optical_v4.md` |
| normal_night preset display_l_max | 238 | `env_presets.py:build_env_presets` key "normal_night" | DOC | `docs/research_report_20260415_env_policy_branching_optical_v4.md` |
| glare preset display_l_max | 232 | `env_presets.py:build_env_presets` key "glare" | DOC | `docs/research_report_20260415_env_policy_branching_optical_v4.md` |
| normal_day preset display_l_max | 242 | `env_presets.py:build_env_presets` key "normal_day" | DOC | `docs/research_report_20260415_env_policy_branching_optical_v4.md` |
| Glare gate (skip LAB roundtrip) | True (RPi max) | `config.py:RPI_THROUGHPUT_MAX_DEFAULTS["display_luma_cap_glare_gate"]` | DOC | `../legacy/md/DEPLOY_HARDENING.md` §8 |

---

### D.5 CLAHE per bucket

Each CLAHE clamp below has its own evidence row. A sweep on one bucket does not justify the
same clip value on another bucket without a separate per-bucket sweep or citation.

#### Bucket A — HybridNIREnhancer CLAHE tiers (night_clear, normal_night)

| Tier | Clip value | Brightness trigger | File:symbol | Evidence type | Evidence pointer |
|------|-----------|-------------------|------------|---------------|-----------------|
| very_dark | `3.0 × clahe_clip_scale` | `avg_brightness < 0.25` | `nir_pipeline.py:HybridNIREnhancer.__init__` → `clahe_levels["very_dark"]` | UNVERIFIED | Hand-tuned. Proxy sweep pending via `tools/sweep_clahe_clip.py` |
| dark | `2.0 × clahe_clip_scale` | `avg_brightness < 0.45` | `nir_pipeline.py:HybridNIREnhancer.__init__` → `clahe_levels["dark"]` | UNVERIFIED | Hand-tuned. Proxy sweep pending |
| medium | `1.5 × clahe_clip_scale` | else | `nir_pipeline.py:HybridNIREnhancer.__init__` → `clahe_levels["medium"]` | UNVERIFIED | Hand-tuned. Proxy sweep pending |
| brightness thresholds (0.25, 0.45) | fixed | — | `nir_pipeline.py:HybridNIREnhancer._apply_clahe_boost` | UNVERIFIED | Hand-tuned breakpoints |
| `clahe_clip_scale` default | 1.0 | — | `config.py:CONFIG["nir_enhancer_clahe_clip_scale"]` | UNVERIFIED | Preset `nir_night` overrides to 1.2 (see D.5 Bucket B note) |
| tileGridSize very_dark | (4,4) | — | `nir_pipeline.py:HybridNIREnhancer.__init__` | CIT | [15] Zuiderveld: smaller tiles = finer local adaptation |
| tileGridSize dark | (6,6) | — | `nir_pipeline.py:HybridNIREnhancer.__init__` | CIT | [15] |
| tileGridSize medium | (8,8) | — | `nir_pipeline.py:HybridNIREnhancer.__init__` | CIT | [15] |

Dark-scene L-channel boost multipliers:

| Brightness range | Boost multiplier | File:symbol | Evidence type |
|-----------------|-----------------|------------|---------------|
| `cur_bright < 0.15` | 2.2× | `nir_pipeline.py:HybridNIREnhancer._apply_clahe_boost` | UNVERIFIED |
| `cur_bright < 0.25` | 1.6× | `nir_pipeline.py:HybridNIREnhancer._apply_clahe_boost` | UNVERIFIED |
| `cur_bright < 0.45` | 1.3× | `nir_pipeline.py:HybridNIREnhancer._apply_clahe_boost` | UNVERIFIED |

#### Bucket B — nir_nir_night_clahe (nir_night)

| Name | Value | File:symbol | Evidence type | Evidence pointer |
|------|-------|------------|---------------|-----------------|
| Clip formula | `np.clip(3.0 × clahe_clip_scale, 0.5, 8.0)` | `nir_pipeline.py:nir_nir_night_clahe` | SWEEP | Algorithm: [15] Zuiderveld. Clamp [0.5, 8.0] validated by `tools/sweep_clahe_clip.py` on manifest_v2 nir_night class (40 images, 7 clip values). Optimal: `clahe_clip_scale=0.16` → clip=0.5 (stopping rule: max log_rms s.t. pct_sat < 0.02). Sweep CSV: `docs/tables/iqa/clahe_clip_sweep.csv`. nir_night preset updated: scale 1.2→0.16. |
| tileGridSize | (4,4) | `nir_pipeline.py:nir_nir_night_clahe` | CIT | [15] |

#### Bucket C — nir_anti_glare_bgr (glare, backlight, normal_day)

| Name | Value | File:symbol | Evidence type | Evidence pointer |
|------|-------|------------|---------------|-----------------|
| nir_high_pct (glare preset) | 93.0 | `env_presets.py:build_env_presets` key "glare" | DOC | `docs/research_report_20260415_env_policy_branching_optical_v4.md` |
| nir_saturate_at (glare preset) | 228.0 | `env_presets.py:build_env_presets` key "glare" | DOC | `docs/research_report_20260415_env_policy_branching_optical_v4.md` |
| nir_high_pct (normal_day preset) | 96.0 | `env_presets.py:build_env_presets` key "normal_day" | DOC | `docs/research_report_20260415_env_policy_branching_optical_v4.md` |
| nir_saturate_at (normal_day preset) | 236.0 | `env_presets.py:build_env_presets` key "normal_day" | DOC | `docs/research_report_20260415_env_policy_branching_optical_v4.md` |

#### Bucket D — nir_dehaze_lite (fog)

| Name | Value | File:symbol | Evidence type | Evidence pointer |
|------|-------|------------|---------------|-----------------|
| DCP omega | 0.85 | `config.py:CONFIG["fog_dehaze_omega"]` | CIT | [13][36][37] He: ω<1 retains residual haze for perceptual depth; 0.85 is the value suggested in He TPAMI [37] |
| Proc resolution | 160×120 | `nir_pipeline.py:nir_dehaze_lite` (hardcoded) | DOC | `docs/research_report_20260415_env_policy_branching_optical_v4.md` §DCP cost reduction |

#### Bucket E — RainTemporalMedian (rain)

| Name | Value | File:symbol | Evidence type | Evidence pointer |
|------|-------|------------|---------------|-----------------|
| N median frames | 3 | `config.py:CONFIG["rain_median_frames"]` / `env_presets.py:build_env_presets` key "rain" | DOC | `docs/research_report_20260415_env_policy_branching_optical_v4.md` §rain; [48] Kang (median baseline) |

#### Bucket F — nir_transition_blend (transition)

No CLAHE; passthrough alpha blend during ENV transitions. No clip parameters.

---

### D.6 Guided filter

| Name | Value | File:symbol | Evidence type | Evidence pointer |
|------|-------|------------|---------------|-----------------|
| Radius | 8 | `nir_pipeline.py:HybridNIREnhancer` (inline, ximgproc call) | CIT | [14] He guided filter; radius 8 is default from He's implementation |
| eps | 1e-4 | `nir_pipeline.py:HybridNIREnhancer` (inline, ximgproc call) | CIT | [14] He guided filter; eps controls smoothing vs detail trade-off |
| Enabled by default | False | `config.py:CONFIG["nir_guided_filter_enable"]` | DOC | Disabled in `RPI_THROUGHPUT_MAX_DEFAULTS`; adds ~3–5 ms — `CLAUDE.md` §NIR path and cost |

---

### D.7 Thermal Kalman

| Name | Value | File:symbol | Evidence type | Evidence pointer |
|------|-------|------------|---------------|-----------------|
| Process noise Q | 0.5 | `thermal_pipeline.py:KalmanThermalBackground.__init__` param `process_noise` | UNVERIFIED | Represents expected background drift per frame in °C²; no sweep committed. See §D.X #4 |
| Measurement noise R | 4.0 | `thermal_pipeline.py:KalmanThermalBackground.__init__` param `measurement_noise` | UNVERIFIED | Approximates MI48 sensor noise (±2°C → variance ≈ 4°C²); no sweep. See §D.X #4 |
| Initial covariance P₀ | 100.0 | `thermal_pipeline.py:KalmanThermalBackground.__init__` param `init_uncertainty` | UNVERIFIED | Large initial uncertainty → fast convergence in first frames. No sweep. See §D.X #4 |
| Warmup frames | 1 | `thermal_pipeline.py:KalmanThermalBackground.__init__` param `warmup_frames` | UNVERIFIED | Minimum frames before background considered stable. Hand-tuned |
| 3DNR EMA alpha | 0.65 | `thermal_pipeline.py:ThermalTemporalFilter.__init__` param `alpha` | UNVERIFIED | Originates from `legacy/py/final_fusion.py`; no sweep. See §D.X #4 |
| Night preset 3DNR alpha override | 0.72 | `env_presets.py:build_env_presets` key "night_clear" `thermal_3dnr_alpha` | UNVERIFIED | Higher smoothing for night. Hand-tuned |
| nir_night preset 3DNR alpha override | 0.70 | `env_presets.py:build_env_presets` key "nir_night" `thermal_3dnr_alpha` | UNVERIFIED | Hand-tuned |
| fog preset 3DNR alpha override | 0.58 | `env_presets.py:build_env_presets` key "fog" `thermal_3dnr_alpha` | UNVERIFIED | Lower smoothing for fog (quicker BG update). Hand-tuned |
| Kalman adaptive_rate | 0.005 | `thermal_pipeline.py:KalmanThermalBackground.__init__` param `adaptive_rate` | UNVERIFIED | Rate of Q/R adaptation if adaptive mode enabled. Hand-tuned |

---

### D.8 MAD anomaly

| Name | Value | File:symbol | Evidence type | Evidence pointer |
|------|-------|------------|---------------|-----------------|
| 0.6745 consistency constant | 0.6745 | `thermal_pipeline.py:ThermalMADAnomalyDetector.update` (inline computation) | CIT | [67] Leys 2013 — normal-distribution MAD consistency factor |
| Modified-z threshold | 3.5 | `thermal_pipeline.py:ThermalMADAnomalyDetector.__init__` param `mad_z_thresh` | CIT | [67] Leys 2013 recommends ±3.5 as outlier threshold |
| Temporal confirmation window | 3 | `thermal_pipeline.py:ThermalMADAnomalyDetector.__init__` param `temporal_window` | UNVERIFIED | Engineering addition (not in Leys); no sweep |
| Heat weight | 0.35 | `thermal_pipeline.py:ThermalMADAnomalyDetector.__init__` param `heat_weight` | UNVERIFIED | Blend weight for heat-map contribution to anomaly score. Hand-tuned |
| E1 z-thresh override (night preset) | 1.05 | `env_presets.py:build_env_presets` key "night_clear" `e1_overrides` | UNVERIFIED | Tighter threshold for night (darker → more FP anomalies). Hand-tuned |
| E1 z-thresh override (glare preset) | 1.55 | `env_presets.py:build_env_presets` key "glare" `e1_overrides` | UNVERIFIED | Looser threshold for glare (avoid FP from hot reflections). Hand-tuned |
| E1 z-thresh global default | 1.25 | `config.py:CONFIG["feature_e1_z_thresh"]` | UNVERIFIED | Hand-tuned |
| E1 heat threshold | 46.0 | `config.py:CONFIG["feature_e1_heat_thresh"]` | UNVERIFIED | Hand-tuned (°C or raw DN) |

---

### D.9 Motion / LK

| Name | Value | File:symbol | Evidence type | Evidence pointer |
|------|-------|------------|---------------|-----------------|
| JerkGate diff_threshold | 8.5 | `motion.py:JerkGate.__init__` param `diff_threshold` | UNVERIFIED | No sweep. See §D.X #3 |
| JerkGate hold_frames | 12 | `motion.py:JerkGate.__init__` param `hold_frames` | UNVERIFIED | No sweep |
| JerkGate percentile | 94.0 | `motion.py:JerkGate.__init__` param `percentile` | UNVERIFIED | No sweep |
| JerkGate max_side | 128 | `motion.py:JerkGate.__init__` param `max_side` | UNVERIFIED | Processing thumbnail size. Hand-tuned for speed |
| JerkGate consecutive_frames | 2 | `motion.py:JerkGate.__init__` param `consecutive_frames` | UNVERIFIED | 2 consecutive detections required. Hand-tuned |
| JerkGate near_active_ratio | 0.62 | `motion.py:JerkGate.__init__` param `near_active_ratio` | UNVERIFIED | Soft-motion threshold (fraction of diff_threshold). Hand-tuned |
| LK max_points | 20 | `motion.py:SparseOpticalFlowMotion.__init__` param `max_points` | UNVERIFIED | Reduced from typical 200–500; balances speed vs coverage on RPi4 |
| LK refresh_interval | 15 | `motion.py:SparseOpticalFlowMotion.__init__` param `refresh_interval` | UNVERIFIED | Corner refresh cadence; matches ML_INFERENCE_INTERVAL |
| Shake-reducer max_shift_px | 14.0 | `motion.py:DisplayShakeReducerLite.__init__` param `max_shift_px` | UNVERIFIED | Maximum allowed correction shift. Hand-tuned |
| Phase-correlation gray EMA | 0.88/0.12 | `motion.py:DisplayShakeReducerLite.process` (inline `0.88 * prev + 0.12 * small`) | UNVERIFIED | Reference-frame EMA for phase-correlation stability. Hand-tuned |
| JerkGate blend ratio (score weighting) | 0.48 m + 0.52 p | `motion.py:JerkGate.update` (inline) | UNVERIFIED | Frame-diff (m) / phase-correlation (p) blend for jerk score. Hand-tuned |

---

### D.10 ENV-FSM hysteresis

Tested by: `test_env_preset_hysteresis.py:test_hysteresis_table_values`,
`test_glare_onset_2`, `test_glare_decay_20`.

| Preset | onset_frames | decay_frames | File:symbol | Evidence type | Evidence pointer |
|--------|------------|------------|------------|---------------|-----------------|
| glare | 2 | 20 | `env_presets.py:_TRANSITION_HYSTERESIS["glare"]` | TEST | `test_env_preset_hysteresis.py:test_glare_onset_2`, `test_glare_decay_20`; specific values are UNVERIFIED — hand-tuned. Tests pin the table, not the rationale. |
| glare_heavy | 2 | 20 | `env_presets.py:_TRANSITION_HYSTERESIS["glare_heavy"]` | TEST | Same tests via parametrize |
| rain | 10 | 25 | `env_presets.py:_TRANSITION_HYSTERESIS["rain"]` | TEST + DOC | Docstring in `env_presets.py` explains debounce rationale; values are UNVERIFIED |
| transition | 12 | 30 | `env_presets.py:_TRANSITION_HYSTERESIS["transition"]` | TEST + DOC | "Catch-all" slowest FSM; values are UNVERIFIED |
| fog | 6 | 18 | `env_presets.py:_TRANSITION_HYSTERESIS["fog"]` | TEST + DOC | "Fog develops gradually" — docstring; values are UNVERIFIED |
| all other presets | uses `env_hysteresis_frames=18` default | `config.py:CONFIG["env_hysteresis_frames"]` | UNVERIFIED | Global fallback hysteresis. Hand-tuned |
| NIR raw Schmitt enter | 25.0 + 3.0 = 28.0 | `main.py:NIR_NON_NIGHT_RAW_ENTER` | UNVERIFIED | Schmitt center=25, hysteresis=3. Hand-tuned |
| NIR raw Schmitt exit | 25.0 − 3.0 = 22.0 | `main.py:NIR_NON_NIGHT_RAW_EXIT` | UNVERIFIED | See `README.md` hysteresis note |

---

### D.11 Capture timing

| Name | Value | File:symbol | Evidence type | Evidence pointer |
|------|-------|------------|---------------|-----------------|
| NIR capture target FPS | 60.0 | `config.py:CONFIG["nir_capture_fps"]` | DOC | IMX290 sensor spec via Picamera2 [22]; actual FPS budget-limited |
| Thermal MI48 FPS | ~9 FPS | `hardware.py:ThermalCapture` (MI48 SPI readout rate) | DOC | MI48 datasheet [19]; SPI bandwidth-limited |
| ML inference interval | 15 frames | `config.py:CONFIG["ML_INFERENCE_INTERVAL"]` | UNVERIFIED | Matches LK corner refresh. Balances CPU time vs classification freshness. Hand-tuned |
| Thermal hybrid update rate | 10 | `config.py:CONFIG["nir_hybrid_update_rate"]` | UNVERIFIED | NIR stats update cadence. Hand-tuned |

---

### D.12 ML model meta

Source: `models/production/env_classifier.json` (also aliased in `models/model_registry.json`).
Consistency test: `tests/test_sidecar_ece.py` enforces policy ECE bounds (global ≤0.15; night classes ≤0.10).
Integrity test: `tests/test_model_registry_integrity.py` verifies SHA-256 against `models/model_registry.json`.

| Metric | Value | Source field | Evidence type |
|--------|-------|-------------|---------------|
| n_estimators (RF) | from sidecar `model_params` | `env_classifier.json:model_params.n_estimators` | SIDECAR |
| Training balanced accuracy | 0.9509 | `env_classifier.json:train_balanced_accuracy` | SIDECAR |
| CV balanced accuracy (mean) | 0.7408 | `env_classifier.json:cv_balanced_accuracy_mean` | SIDECAR |
| CV balanced accuracy (std) | ±0.0331 | `env_classifier.json:cv_balanced_accuracy_std` | SIDECAR |
| CV strategy | TimeSeriesSplit | `env_classifier.json:cv_strategy` | SIDECAR |
| Calibration method | isotonic | `env_classifier.json:calibration_method` | SIDECAR + CIT [66][74] |
| n_features | 12 | `env_classifier.json:n_features` | SIDECAR + TEST (FS01–FS07) |
| n_samples (training) | 14094 | `env_classifier.json:n_samples` | SIDECAR |
| τ₁=0.62 macro_f1_night | 0.9835 | `env_classifier.json:ml_gate_reference.top1_tau_0.62.macro_f1_night_ovr` | SIDECAR + SWEEP |
| τ₁=0.62 abstention_rate | 0.1278 | `env_classifier.json:ml_gate_reference.top1_tau_0.62.abstention_rate` | SIDECAR + SWEEP |
| τ₂=0.20 hint_rate_of_ml | 0.0472 | `env_classifier.json:ml_gate_reference.top2_tau_0.20.hint_rate_of_ml_active` | SIDECAR + SWEEP |
| ECE night_clear | 0.0176 | `env_classifier.json:ece_by_class.night_clear` | SIDECAR (policy bound ≤0.10) |
| ECE normal_night | 0.0278 | `env_classifier.json:ece_by_class.normal_night` | SIDECAR (policy bound ≤0.10) |
| ECE nir_night | 0.0145 | `env_classifier.json:ece_by_class.nir_night` | SIDECAR (policy bound ≤0.10) |
| ECE normal_day | 0.0427 | `env_classifier.json:ece_by_class.normal_day` | SIDECAR (policy bound ≤0.15; worst class) |
| ECE fog | 0.0163 | `env_classifier.json:ece_by_class.fog` | SIDECAR (policy bound ≤0.15) |
| ECE rain | 0.0330 | `env_classifier.json:ece_by_class.rain` | SIDECAR (policy bound ≤0.15) |
| ECE glare | 0.0185 | `env_classifier.json:ece_by_class.glare` | SIDECAR (policy bound ≤0.15) |
| ECE backlight | 0.0119 | `env_classifier.json:ece_by_class.backlight` | SIDECAR (policy bound ≤0.15) |
| ECE transition | 0.0210 | `env_classifier.json:ece_by_class.transition` | SIDECAR (policy bound ≤0.15) |
| SHA-256 (production joblib) | `00c4f0eecb6e7cc56b77cfea911c447e1f0e9448d0ec984ab9bc799c637d6b48` | `models/model_registry.json` | TEST (`test_model_registry_integrity.py`) |

**ECE policy bounds rationale.** All observed ECE values are well below the policy bounds
(global ≤0.15; night classes ≤0.10). The bounds are intentionally conservative to accommodate:
(a) class-imbalance drift if the training set is augmented; (b) future retrains where newly added
samples may shift calibration. Setting the bound at the current observed value would be a snapshot
pin, not a policy — a single additional misclassified night frame would fail the test.

**Feature ablation (9 vs 12 features) — Phase 4:**

| Metric | Optical-12 (prod) | Optical-9 (ablation) | Source |
|--------|:-----------------:|:--------------------:|--------|
| Test accuracy | 0.8348 | **0.8448** | SWEEP — `docs/tables/ablation_9_vs_12.md`, commit `4df19d7` |
| Test balanced accuracy | 0.7418 | **0.7554** | SWEEP |
| Test macro F1 | 0.744 | **0.760** | SWEEP |
| Night F1 night_clear | 0.968 | 0.967 | SWEEP |
| Night F1 normal_night | 0.851 | **0.858** | SWEEP |
| Night F1 nir_night | 0.977 | **0.978** | SWEEP |
| Night ECE night_clear | 0.0177 | **0.0153** | SIDECAR `models/ablation/rf_optical_9.json` |
| Night ECE normal_night | 0.0281 | **0.0256** | SIDECAR |
| Night ECE nir_night | 0.0178 | **0.0100** | SIDECAR |

Test set: `data/training/from_logs_test.jsonl` SHA-256 `583ce85f…`, n=2113 (single touch, 2026-04-28).
Dropped features (`hour_of_day_sin/cos`, `prev_env_class`) all show 0.000 importance in Optical-12 — zero-imputed offline.
Optical-9 wins on all held-out metrics but **is not deployed** (fails `EnvClassifier.feature_set` validation).
Interpretation: zero-imputed temporal features add noise variance in offline training; live deployment retains them for real-time context signal. See `docs/tables/ablation_9_vs_12.md` for full narrative.

---

### D.X Proposed minimal experiments (UNVERIFIED rows)

Each experiment below corresponds to one or more UNVERIFIED rows in Part D.
Running the experiment replaces the **UNVERIFIED** tag with **SWEEP** and commits the CSV.

**#1 — EMA alpha stability sweep (D.1 glare asymmetric α, D.2 ml_posterior_ema_alpha)**

Goal: find α that minimises ENV-class oscillation rate (transitions per minute) across a 10-minute
session log while preserving sub-2-second response to genuine ENV changes.

Proposed script: `tools/ema_alpha_sweep.py` (not yet written).
Inputs: a committed session JSONL from RPi4 with `env_class` and `ml_top1_proba` fields.
Output: `docs/tables/ml/ema_alpha_sweep.csv` — one row per (α, session, transitions_per_min,
median_response_frames).

**#2 — Temporal glare IIR weight sweep (D.2 glare temporal prev_weight = 0.42)**

Goal: determine prev_weight that eliminates glare flicker (>3 Hz L-channel oscillation)
without introducing >1-frame visual lag.

Proposed script: `tools/sweep_glare_iir.py` (not yet written).
Metric: FFT peak frequency of L-channel signal during known-glare frames from session JSONL.

**#3 — JerkGate and shake-reducer parameter sweep (D.9)**

Goal: characterise FP/FN jerk-detection rate as a function of `diff_threshold` across
a captured motion sequence.

Proposed script: `tools/sweep_jerkgate.py` (not yet written).
Inputs: committed motion-test session JSONL.
Output: `docs/tables/timing/jerkgate_sweep.csv` — precision/recall vs threshold.

**#4 — Thermal Kalman Q/R sweep (D.7)**

Goal: characterise RMS background residual in foreground-free thermal regions as a
function of Q ∈ {0.1, 0.3, 0.5, 1.0, 2.0} × R ∈ {2.0, 4.0, 8.0}.

Script: `tools/sweep_kalman_qr.py` (implemented; requires thermal sequence under `data/`).
Output: `docs/tables/timing/kalman_qr_sweep.csv`.
Current status: stub exits non-zero if no thermal sequence is staged.

**#5 — Bucket B CLAHE clip proxy sweep (D.5 Bucket B)**

Goal: quantify `log_rms_contrast`, `pct_saturated`, `pct_crushed` across clip ∈ {0.5, 1.0,
2.0, 3.0, 4.0, 6.0, 8.0} on representative nir_night frames.

Script: `tools/sweep_clahe_clip.py` (implemented; requires committed NIR sample frames).
Output: `docs/tables/iqa/clahe_clip_sweep.csv`.
Current status: exits non-zero if no frames are staged under `data/training/`.

**#6 — Stage timing on RPi4 (Part A §A.3) — Phase 2 target**

Procedure: deploy optimized build (Phase 1 complete), run `python -m smartbinocular` for ≥5 minutes on RPi4, then:
```bash
python tools/measure_stage_timing.py
```
Output: `docs/tables/timing/stage_timing_summary.csv` and `docs/tables/timing/stage_timing_summary.md`.
This is the only path to replace the **UNVERIFIED** budget claim in Part A §A.3 and in `README.md`.

**Acceptance:** Report mean ± σ deltas vs baseline session `session_20260425-211623.json`. A null or
negative result is acceptable evidence provided: (a) throttle snapshot is attached for interpretation,
(b) `pipeline_config_sha256` between baseline and post-opt is recorded for reproducibility, and
(c) thesis text frames the finding neutrally. Do not gate Phase 2 on a fixed improvement percentage.

**Post-Phase 2 edit:** Update §A.3 table with new session means; record both session IDs and their
`pipeline_config_sha256` values. Update status from `SWEEP (Mac host)` → `SWEEP (RPi4 post-opt)`.
