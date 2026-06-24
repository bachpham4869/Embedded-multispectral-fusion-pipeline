# Thesis mentor synthesis — pipeline analysis, bottlenecks, Edge Impulse, open questions

**Source:** Main-agent handoff (mentor role: evidence-based code review + session `session_20260425-211623.json` + `src/`).  
**Session reference:** 225 s field run, 1310 frames, 1070 fusion, mean FPS ~6.3.

**Status:** Ghi chú nội bộ / **ý tưởng** — chưa duyệt, chưa thực thi thay thế bằng chứng chính thức; đối chiếu [`PIPELINE_EVIDENCE_REGISTER.md`](../../docs/PIPELINE_EVIDENCE_REGISTER.md) khi trích số.

This document is a **consolidated write-up** of that analysis (not a second opinion). Use it in the thesis as internal notes; cite original papers and re-run any measurements you rely on in defense.

---

## Code architecture insight (summary)

The codebase separates **what to run** (ENV + bucket dispatch) from **how to process** (HybridNIR enhancer, thermal Kalman). The `RPI_THROUGHPUT_MAX_DEFAULTS` block acts as a **field-mode** feature layer: a single configuration merge can ablate a whole optimization tier. That pattern is convenient for a thesis: one “knob” for tiered ablations.

---

## A. Executive summary

- **Measured throughput:** ~6.3 FPS mean over a ~225 s session (1310 frames, 1070 in fusion) — on the order of **3× below** a notional “≤ 50 ms per frame” system target, if that target is read as steady-state.
- **Dominant costs (session means):** `nir_bucket` (~40 ms, high σ), `fuse_blend_math` (~37 ms, large float buffers), and `framecache` (~12 ms, far above a nominal ~0.4 ms resize path — interpreted in the source analysis as plausibly **CPU/thermal throttling** rather than a single obvious software bug in FrameCache).
- **Strongest demonstrated ML result (documented in project artifacts):** ENV classifier with **macro F1 (night) ≈ 0.984** at τ = 0.62 on held-out test, isotonic calibration, TimeSeriesSplit CV (see project tables / `ml_gate` docs).
- **Gaps to close for a defensible thesis:** (1) whether **fused display** actually improves a defined task over NIR-only; (2) whether **A–F bucket dispatch** is better than a fixed path for the same quality budget; (3) whether **E1** has any **validated** detection performance.

---

## B. Code / architecture map

### B.1 Dataflow (simplified)

- **MI48 SPI** ~80×62 @ ~9 FPS → `thermal_pipeline.py` (3DNR, Kalman background, `get_heat_map` / `get_foreground_mask`).
- **IMX290 Picamera2** 640×480 @ ≤60 FPS → `main.py` `FrameCache` / `nir_compute_gray_cached` → small NIR (e.g. 96×72 for ML) and 320×240 (or as configured) for NIR enhancer.
- **Environment:** `env_presets.py` (rule + ML top-2) → `env_class`.
- **NIR path:** `OPTICAL_BUCKET_DISPATCH_LITE` when `nir_optical_lite=True` (e.g. A full enhancer, B CLAHE-only, E rain median).
- **Fusion:** `main.py` — resize NIR, thermal prep, colormap, BGRA warp, blur FG, **blend math** → 800×480 BGR.
- **Display:** `display_pipeline.py` (glare blend, luma cap in `luma_only`), `cv.imshow`.
- **Metrics:** `metrics.py` / `ThesisRunMetrics` → `session_*.json` with `stage_timing_ms`, `fuse_stage_timing_ms`, env/bucket counts.

### B.2 Key symbols (reference table)

| Symbol / area | File (approx.) | Role |
|---------------|----------------|------|
| `main()` | `main.py` | Full frame loop |
| Pre-alloc fusion float buffers | `main.py` init | 600×360×3 `float32` for warp/blend work |
| `HybridNIREnhancer.process` | `nir_pipeline.py` | Bucket A heavy path + CLAHE / detail / color correct / resize |
| `_compute_channels` | `nir_pipeline.py` | Min/max filters, atmosphere estimate |
| `_cc_buf` | `nir_pipeline.py` | Pre-alloc color-correct buffer |
| `OPTICAL_BUCKET_DISPATCH_LITE` | `nir_pipeline.py` | Lite routing A/B/E when `nir_optical_lite` |
| `KalmanThermalBackground` | `thermal_pipeline.py` | Per-pixel 1D Kalman, vectorized |
| `EnvClassifier` / `MLInferenceThread` | `ml_inference.py` | RF + background inference |
| `FEATURE_SET_OPTICAL_ONLY` | `feature_schema.py` | 12-feature production set |
| `RPI_THROUGHPUT_MAX_DEFAULTS` | `config.py` | e.g. `update_rate=20`, `fusion_warp_work_scale=0.75`, `colormap_levels=64` in that session’s merge |

Line numbers in the source analysis are **as cited by the agent** and may shift — verify in-tree before using in the thesis.

---

## C. Measured bottlenecks and ranked cut candidates

### C.1 Session table (illustrative, from one session)

| Stage | Mean ms | σ (session) | n | Session field |
|-------|---------|---------------|---|----------------|
| framecache | 12.1 | 7.9 | 1310 | `stage_timing_ms.framecache` |
| nir_bucket | 40.2 | 42.5 | 1214 | `stage_timing_ms.nir_bucket` |
| thermal_proc | 16.9 | 20.9 | 1031 | `stage_timing_ms.thermal_proc` |
| fuse_nir_resize | 6.0 | 6.7 | 1070 | `fuse_stage_timing_ms` |
| fuse_thermal_prep | 3.1 | 5.7 | 1047 | idem |
| fuse_colormap | 9.0 | 9.3 | 1047 | idem |
| fuse_warp_prep | 8.3 | 10.8 | 652 | idem (FG path subset) |
| fuse_warp_perspective | 6.7 | 7.0 | 652 | idem |
| fuse_blur_fg | 5.4 | 8.0 | 652 | idem |
| **fuse_blend_math** | **36.5** | 18.6 | 652 | idem (largest single sub-cost in listed fuse stages) |
| blend / hud / display | ~3–5 | (see session) | 1310 | `stage_timing_ms` |

**Do not** sum rows naïvely for “total frame time” — sub-stages overlap, sample counts differ (FG path ~652 vs fusion frames 1070), and the session reports aggregate stage timers, not a serial sum.

A **rough** “heavy fusion frame” budget order-of-magnitude (~155 ms) was argued in the source analysis to be **consistent** with ~6.3 FPS for mixed modes — re-verify with your own session after any code change.

### C.2 #1 — `fuse_blend_math` — reduce `fusion_warp_work_scale` (e.g. 0.75 → 0.5)

**Idea:** Smaller 600×360-style work buffers (when scale=0.75) → less memory traffic for NumPy + `convertScaleAbs` + resize. Agent cited **L2 size vs buffer size** and **broadcast** traffic as hypotheses for the gap between “back-of-envelope” and measured ~36.5 ms.

**Risk (mitigation):** Thermal sensor is low-res; the agent argues limiting factor is **MI48** spatial resolution, not warp grid — but **validate visually** and with user study / metrics if the thesis claims quality.

**Test:** Set `RPI_THROUGHPUT_MAX_DEFAULTS["fusion_warp_work_scale"]` to **0.5** (or equivalent config path), 3+ minute field session, compare `fuse_blend_math` means and save side-by-side stills for an appendix.

### C.3 #2 — `nir_bucket` — SciPy vs OpenCV morphology; optional proc size cut

**Idea (morph):** Bimodal cost (high σ) linked to `nir_hybrid_update_rate=20` (heavy path 1/20 frames) and `scipy.ndimage` min/max filters vs **OpenCV** `erode`/`dilate` on **aarch64** (NEON). **Micro-benchmark on device first** (agent provided a one-liner script idea).

**Idea (size):** Lower `nir_enhancer_proc_w/h` (e.g. 240×180 vs 320×240) to cut work in dark-channel and CLAHE; validate with **BRISQUE** / existing IQA hooks if available.

**Test:** Before code change, benchmark SciPy vs OpenCV on representative sizes; then A/B field sessions; report mean and tail latency.

### C.4 #3 — `framecache` — thermal throttling diagnosis

**Idea:** ~12 ms mean is inconsistent with a “tiny resize only” story; **hypothesis: CPU throttling** (e.g. ~80°C → lower clocks). **Not** a code “cut” until confirmed.

**Next step (instrumentation):** Log `vcgencmd get_throttled` and per-CPU `scaling_cur_freq` (or your preferred Pi thermal API) into `experiment_context` at session end / periodic snapshots — as proposed in the agent text.

**Thesis:** If throttled, either improve cooling, reduce threads (`opencv_num_threads`), or **explicitly** separate **design target** (50 ms) from **measured operating point** in writing.

### C.5 #4 — `fuse_colormap` — colormap at native 80×62 then resize

**Idea:** If colormap currently runs on a **large** intermediate grid, run LUT on **native** 80×62 and **then** resize colorized BGR — may reduce per-pixel work.

**Test:** Micro-bench `gray_to_thermal_bgr` at 80×62 vs current working size; profile in isolation.

### C.6 #6 — `thermal_proc` — decompose scope of timer

**Idea:** Documented sub-parts (Kalman, morph) should be sub-ms–low-ms class; 16.9 ms mean with high σ may include **other work** in the same timer (AGC, E1, capture, etc.).

**Test:** **Sub-profiler** inside `ThermalProcessor.process()` to split (3DNR+Kalman) vs (fg mask + heat map + E1) — “small instrumentation change” in the original note.

### C.7 Composite estimate (illustrative only)

The agent’s **illustrative** stack of savings (scale 0.5, smaller proc, morphology, colormap order) was on the order of **~20–27 ms** mean improvement **in their estimate** — not a guarantee. Re-measure. Their conclusion: not necessarily **20 FPS** on Pi4 without **Pi5 class hardware**, lower display resolution, or more aggressive quality tradeoffs.

---

## D. Edge Impulse + thermal person detection — design study

### D.1 Physical / thesis insight (MI48 resolution)

~80×62 thermal: at typical FoV, a person at **long range** may **occupy only a few pixels** — any “detection + thermal confirm” story needs **stated range limits** and **field validation** (the agent’s point — not a substitute for your measurements).

### D.2 What it could add

Multimodal agreement: NIR/edge “person-like” + **thermal plausibility** in the **mapped** region → fewer false triggers from **hot non-person** objects if shape and thermal are jointly consistent.

### D.3 Risks (thesis + engineering)

| Risk | Severity (agent) | Note |
|------|------------------|------|
| Domain shift: RGB-trained detectors on **night NIR** | High | Fine-tune / collect NIR-night data; report measured precision. |
| Thermal **spatial** resolution (few px at range) | High | Cite limits in Discussion; KAIST / multispectral lit for “performance cliff” at small thermal detail. |
| **Absolute** temperature (not Kelvin) from MI48 | Medium | Use **relative** / z-score style gating (aligned with how anomaly work already reasoned in-tree). |
| **Homography** error under **motion** | Medium | `rate_jerk` in session; map error vs motion — open measurement. |
| **Latency** budget | High | Amortize with **daemon** thread (mirror `MLInferenceThread`); avoid blocking the frame loop. |
| **Privacy / ethics** | Medium | Scope deployment, consent, VN personal-data rules / IEEE EAD as applicable to your context. |
| **Dataset** bias (urban lit vs your field) | Medium | Build or cite a **domain** eval set. |

### D.4 Feasible integration (design only)

- **Option A (agent):** After `framecache` — NIR small **96×72**; tiny person model (e.g. FOMO int8) at low input size on a **thread**; project bbox to thermal via `_H_fuse`; combine with `fg_mask` / overlap score.
- **Option B:** After NIR enhancer (better image, **+latency** on inference frames).

**Thermal confirmation (concept):** project bbox corners; check overlap with **warm FG** or blob statistics — calibrate as **relative** features, not absolute °C (unless you calibrate the sensor — Future work).

### D.5 Validation matrix (abbrev.)

Scenes: person near/far, hot non-person, fog (thermal complement), small animal (confusable), etc. Report **per-condition** precision/recall; compare to **E1** baseline on the **same** labeled frames (agent suggestion).

**Minimum experiment (agent):** Edge Impulse export → bench on Pi → ~3 sessions → 50 hand-labeled frames with `write_capture_meta` if used → per-ENV class metrics; compare to thermal-only E1 on same labels.

### D.6 Comparable work (bibliography seeds — verify full citations in thesis)

- Hwang et al., multispectral pedestrian / KAIST (CVPR 2015).
- Liu et al., multispectral detection + calibration ideas (BMVC 2018).
- Guo et al., **temperature scaling** / calibration (ICML 2017) — for combining scores.
- Choi et al., KAIST day/night multispectral, ITS (2018) — supports “small thermal = hard” narrative (verify which claim you need from the **actual** paper).

---

## E. Open questions and thesis narrative hooks

1. **Fusion justification (E1 in agent text):** Show a **defined task** where fusion beats NIR-only (human study or **metric/annotation** on the same scenes).
2. **9 vs 12 features:** If `env_classifier` importances for time features are 0, run **9-feature** ablation on the **same** test protocol and report F1 + ECE — defensible “simpler model” or “surprising discrepancy” story.
3. **50 ms vs ~155 ms:** **Design intent** vs **measured** field session must be **named explicitly**; list mitigations (cooling, scale cuts, next HW).
4. **Homography + motion (E4):** Log **motion magnitude**; quantify alignment error vs jerk — one figure for “handheld fusion limits.”
5. **ENV reporting:** For night focus, lead with **relevant** night classes + **abstain** behavior, not only global balanced acc.
6. **E1 ground truth:** If you claim “detector,” you need **labels** (even a small n) — range sweep at fixed geometry as agent sketched.
7. **Future work bullets:** absolute thermal calibration, RPi5, 17-feature thermal+optical if calibrated, **online** homography / flow, Edge Impulse person track as a **separate** timeline with the validation matrix above.

### Closing (agent’s closing thesis point)

**Defensible now:** session logging, **RF + calibration** path, **bucketed NIR** design, and infrastructure for RPi field evidence. **Committee preparation:** is mainly **rigorous evaluation** and **honest** performance framing, not an endless list of new features.

---

*End of synthesis.*
