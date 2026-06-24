# Thesis Scope and Limitations

**Project:** smartBinocular — Real-Time NIR + Thermal Fusion on Raspberry Pi 4B  
**Draft:** 2026-04-27  
**Status:** Pre-defense draft — update with Phase 2 RPi timing results when available

---

## 1. Scope

This thesis makes three measurable claims:

**Q1 — Fusion utility:** The fused NIR + thermal composite output (via the `mode="fusion"` pipeline path) provides a detectable improvement in at least one objective metric (IQA or detection rate) over the NIR-only path (`mode="nir"`, `main.py:1283–1315`) on low-light field sequences, under comparable pipeline configuration (`pipeline_config_sha256` matched).

**Q2 — A–F dispatch utility:** The environment-adaptive optical bucket dispatch (`OPTICAL_BUCKET_DISPATCH`, `nir_pipeline.py`) produces measurably better offline IQA metrics on the NIR evaluation set than a fixed-path baseline (bucket A for all classes), across at least the `fog`, `glare`, and `nir_night` env classes.

**Q3 — E1 anomaly indicator:** The MAD-based E1 anomaly score from `ThermalProcessor` flags thermal outliers in session recordings. This claim is limited to indicator-level: no precision/recall vs. a ground-truth label set is available unless 50 frames are manually annotated (deferred).

---

## 2. What the thesis does NOT claim

| Non-claim | Reason | Evidence type available |
|-----------|--------|------------------------|
| ≤50 ms steady-state frame latency on RPi4B | One session result; throttle and thermal conditions vary | Session `stage_timing_ms` (pending Phase 2) |
| BRISQUE as a primary IQA metric | BRISQUE assumes natural-image statistics; NIR night imagery violates this (Risk R1) | Proxy metrics only: log-RMS, pct\_saturated, hist\_entropy |
| Still-image offline metrics replicate live EMA behavior | `HybridNIREnhancer` EMA state resets per image in batch mode; temporal smoothing is absent | Labelled `still_image_cold_start_mode` in all CSVs |
| 9-feature model generalizes across all deployment contexts | Ablation tested on same-distribution test split; no domain-shift evaluation | `docs/tables/ml/ablation_9_vs_12.md` |
| Fusion alpha=0 is a valid NIR-only baseline | `fusion_alpha=0` still runs fusion-path arithmetic (`fuse_blend_math`, `fuse_warp_perspective`); thermal stage cost and `fusion_alpha_boost` presets remain active | Use `mode="nir"` for clean baseline |
| E1 is a validated anomaly detector | No ground-truth labels available without manual annotation | Indicator only |

---

## 3. Staged evidence table

Evidence is presented in three levels of reproducibility. Claims depending on RPi hardware are marked PENDING until Phase 2 session is captured.

| Claim | Stage | Evidence artifact | Status |
|-------|-------|------------------|--------|
| Q2 — Bucket B outperforms A on fog IQA | 1 (offline, dev) | `docs/tables/iqa/batch_nir_iqa.csv`, `docs/figures/iqa/iqa_ab_appendix.md` | **DONE** |
| Q2 — Bucket D unsuitable for night classes | 1 (offline, dev) | Same; pct\_crush\_after >0.34 on night\_clear, normal\_night | **DONE** |
| Q2 — CLAHE clip=0.5 is proxy-optimal for Bucket B night classes | 1 (offline, dev) | `docs/tables/iqa/clahe_clip_sweep_summary.md` | **DONE** |
| ML — 9-feat model ≥ 12-feat model on calibration and macro F1 | 1 (offline, dev) | `docs/tables/ml/ablation_9_vs_12.md` | **DONE** |
| Q1 — Pipeline meets ≤50 ms/frame budget | 3 (RPi field session) | `docs/tables/timing/stage_timing_summary.csv` | **PENDING Phase 2** |
| Phase 1 optimizations (warp scale, colormap order) reduce timing | 3 (RPi field session) | Phase 2 before/after `stage_timing_ms` delta table | **PENDING Phase 2** |
| Q1 — Fusion vs NIR-only A/B timing + IQA | 3 (RPi field session) | Phase 5 driver script output | **DEFERRED** |
| Q3 — E1 precision/recall vs labeled anomalies | Manual labels | 50-frame annotation | **DEFERRED** |

---

## 4. Honest framing of the 50 ms budget

The ≤50 ms/frame target is a design goal, not a verified steady-state result. The following caveats apply:

- **Thermal throttling:** RPi4B throttles under sustained load >~80°C. The `throttle_snapshot` field (vcgencmd + CPU freq) will be recorded in Phase 2 session JSON to separate algorithmic cost from thermal cost.
- **Stage attribution:** All per-stage costs (`fuse_blend_math`, `nir_bucket`, `thermal_proc`, etc.) will be compared against the baseline session `session_20260425-211623.json` using `pipeline_config_sha256` to confirm both runs used equivalent config.
- **Null result is acceptable:** If the Phase 1 optimizations (warp scale 0.75→0.5, colormap order swap) do not produce a measurable reduction, the thesis will report this with the throttle snapshot as interpretive context. A null result with intact methodology is valid evidence.

---

## 5. Offline IQA limitations

1. **Dataset domain shift:** Darkface, ExDark, and weather11 datasets are visible-spectrum images, not IMX290 NIR captures. The IMX290 sensor has extended near-infrared response and suppressed color. Proxy metrics provide directional evidence, not field-performance prediction.
2. **Still-image cold-start:** All batch IQA results use `still_image_cold_start_mode`. In live video, `HybridNIREnhancer` EMA state accumulates across frames; the temporal smoothing prevents overcorrection on isolated bright frames. The cold-start mode likely overstates saturation risk for Bucket A on bright nir\_night inputs.
3. **No perceptual ground truth:** Proxy metrics (Laplacian RMS contrast, histogram entropy, pixel-level saturation/crush fractions) correlate with IQA in low-light NIR contexts but are not substitutes for a human observer study.
4. **Bucket E excluded:** `RainTemporalMedian` requires a contiguous frame sequence; batch still-image evaluation cannot replicate its temporal denoising behavior.
5. **Bucket C and F excluded:** Bucket C (`nir_anti_glare_bgr`) targets daytime / indirect-glare scenes outside the core night NIR thesis claim. Bucket F (`nir_transition_blend`) is a passthrough and trivially unchanged.

---

## 6. ML limitations

1. **Same-distribution test split:** `from_logs_test.jsonl` was split from the same source distribution as training data. No out-of-distribution generalization test was performed.
2. **Offline feature extraction:** `FeatureRecord` features are computed from frame statistics, not from the live inference pipeline. Pipeline-specific factors (camera AGC, real-time frame timing, EMA state) are not captured in offline features.
3. **Temporal features dropped in 9-feat ablation:** `hour_of_day_*` and `prev_env_class` may carry useful signal in long outdoor sessions with strong time-of-day variation. The ablation result (9-feat ≥ 12-feat) is specific to the current dataset composition.
4. **One test-set touch:** `from_logs_test.jsonl` was evaluated exactly once per model. No hyperparameter tuning or threshold selection was performed against the test set.

---

## 7. RPi follow-up checklist (Phase 2 — deferred)

When the Raspberry Pi 4B is available:

1. **rsync optimized build:** `rsync -avz --exclude='.venv' --exclude='*.joblib' . pi@raspberrypi.local:~/smartBinocular/`
2. **rsync model:** `rsync -avz models/production/env_classifier.joblib pi@raspberrypi.local:~/smartBinocular/models/production/`
3. **Run ≥5 min session:** `ML_INFERENCE_ENABLED=1 ML_MODEL_PATH=models/production/env_classifier.joblib python -m smartbinocular` — verify `throttle_snapshot` and `pipeline_config_sha256` appear in output JSON
4. **Pull session back:** `rsync -avz pi@raspberrypi.local:~/smartBinocular/fusion_captures/metrics/ fusion_captures/metrics/`
5. **Update timing table:** `python tools/measure_stage_timing.py` → compare `docs/tables/timing/stage_timing_summary.csv` against baseline session `session_20260425-211623.json`; produce `docs/tables/timing/phase2_before_after.md` (optional)

**Files to rsync back from Pi:**
- `fusion_captures/metrics/session_*.json` (the new post-optimization session)
- Any `fusion_captures/*.png` if visual captures were enabled

---

## 8. Phase 7 note (Edge Impulse — deferred)

Edge Impulse integration is a **future-work item only**. No production commits to `src/` are planned until Phases 0–6 are stable. The design memo is in `docs/EDGE_IMPULSE_FUTURE_WORK.md`.
