# Offline NIR / IQA batch evaluation — agent synthesis (methodology, risks, repo map)

**Source:** Main-agent handoff (codebase + established IQA knowledge; no ad-hoc web search in the original run).  
**Date archived:** 2026-04-26.

This document records **one consolidated analysis** for the idea: run NIR bucket paths and ML feature extraction on **still images** on the dev machine, score before/after with **appropriate** metrics, and optionally sweep parameters. Use it in the thesis and implementation planning; re-verify all paths and line references against the current tree.

---

## Key orientation (insight)

The repo **already** partially implements this idea:

- `tools/sweep_clahe_clip.py` — proxy quality metrics on NIR across CLAHE parameters.
- `tools/offline_pipeline.py` — full production **feature** set from still images.

The student’s proposal is to **bridge** them: add image processing paths to the offline runner and attach IQA scores. The question is not *whether it can be built* but *what it will prove* and *what it will silently mislead* on.

---

## (A) Summary insight box

**Idea, precisely:** Run NIR enhancement buckets (A–F in `nir_pipeline.py`) and optionally the display grading path on a **labeled** still-image dataset on the **dev** machine, extract `FeatureRecord` features before and after, score both with image-quality metrics, and optionally **sweep** pipeline parameters to maximize those scores. The thesis would cite this as evidence that the processing paths improve measurable optical statistics in a **controlled, repeatable** setting.

**What this legitimately provides (if framed as a written protocol, not a performance proof):** Evidence that enhancement paths improve **measurable optical statistics** (contrast, sharpness, saturation balance) on a **label-stratified** set — with per-`ENV` breakdowns and reproducible sweep results.

**What this does *not* provide, no matter how well executed:**

- Evidence that **fusion (NIR+thermal)** helps — offline path **cannot** run the fusion composite without real or simulated thermal.
- Evidence of **real-time** performance — RPi `session_*.json` remains the only ground truth for **latency**.
- **Task-utility** evidence — no IQA score shows that an operator detects a target **faster or more reliably** in the enhanced view. State this gap explicitly in the thesis.

---

## (B) Risks (numbered)

| ID | Risk | Notes |
|----|------|--------|
| **R1** | **BRISQUE/NIQE invalid for night NIR** (committee will ask) | Both use MSCN / NSS from **daytime color** natural scenes. Night NIR violates assumptions (grayscale, readout noise, low mean + hot spots). Optimizing BRISQUE may push images toward “daytime natural” — wrong for thesis purpose. `sweep_clahe_clip.py` **does not** use BRISQUE; it uses `log_rms_contrast`, `pct_saturated_new`, `pct_crushed_new` — **domain-specific proxies**. **Calibration test:** BRISQUE on 10 dark NIR raw vs `nir_anti_glare_bgr()` — if “clearer” looks worse, report as calibration check, not a bug. |
| **R2** | **Degenerate solutions** from maximizing a no-ref metric | Unsharp / stretch can maximize `log_rms_contrast` while crushing blacks. `sweep_clahe_clip` already uses saturation penalties. Any auto-tune needs: **bounded** search, **≥2** independent proxies + penalties, **held-out** val set, **mandatory** A/B human panels. |
| **R3** | **Feature distribution shift: stills vs video** | `offline_pipeline.py` uses `has_temporal=False`, `sequence_reset=True` — temporal features (e.g. `nir_brightness_delta_10f`, `nir_blue_mean_ema`) are not like production. Label still-image ML confidence as e.g. **“still-image inference (EMA cold-start)”** in the thesis. |
| **R4** | **Missing thermal → fusion composite unreachable** | `_dummy_thermal()` / zeros — no realistic thermal stream. Offline batch can evaluate **raw → bucket-processed NIR quality only**. State as **scope boundary**; use a **staged evaluation** table (see below). |
| **R5** | **ML leakage** if tuning display using classifier confidence on same images | Use **separate** train/val/test; do **not** tune CLAHE from RF confidence on the **same** images used to train the RF. `from_logs_test.jsonl` is eval-only; respect splits. |
| **R6** | **“After is better” ≠ detection benefit** | Frame as: necessary but not sufficient for situational awareness; add detection task or human study if that claim is needed. |
| **R7** | **Dev vs RPi not interchangeable** | **Timing** on Mac vs aarch64; numerical CLAHE/morph may match, **latency** tables must not mix. |

---

## (C) Repo map

### Directly usable (present in tree)

| Path | Role |
|------|------|
| `tools/offline_pipeline.py` | Reads datasets → `FeatureExtractor` → JSONL. Extend to call bucket fns and write before/after pairs. |
| `tools/sweep_clahe_clip.py` | CLAHE sweep; `log_rms_contrast`, saturation/crush metrics → CSV. Template for bucket A: call `HybridNIREnhancer.process()`. |
| `tools/sweep_kalman_qr.py` | Thermal Q/R sweep pattern — copy loop style for NIR sweeps. |
| `src/smartbinocular/feature_extractor.py` | `FeatureExtractor.extract()` — same offline as RPi. |
| `src/smartbinocular/nir_pipeline.py` | Standalone: `HybridNIREnhancer.process`, `nir_nir_night_clahe`, `nir_dehaze_lite`, `RainTemporalMedian.process`, `nir_anti_glare_bgr` — **no** camera loop. |
| `data/training/from_logs_test.jsonl` | **Eval-only** for ML — do not tune against it. |
| `data/training/from_logs_train.jsonl` | Training — not ideal as **sole** val for IQA parameter tuning. |
| `tools/validate_schema.py` | Validate JSONL after batch. |
| `tools/check_features.py` | Per-class feature distributions. |
| `tools/audit_night_classes.py` | Class counts. |
| `tools/label_mapping.yaml` | Label → `ENV_CLASSES`. |

### Small additions (suggested, not in original tree)

1. **`tools/batch_nir_enhancer.py` (new, ~order 150 LoC):** manifest `(path, env_class)` → correct bucket per class → before/after PNGs + CSV (`image_name`, `env`, `bucket`, metrics…). Use `sweep_clahe_clip.py` as template.
2. **Extra IQA columns:** `laplacian_variance`, `histogram_entropy`, `local_contrast_std` — **not** BRISQUE as primary (see R1). BRISQUE only **exploratory** with explicit caveat.

### Impossible offline (needs RPi or future work)

- Full **fusion composite** (needs `ThermalProcessor` + `heat_map` / `fg_mask`).
- **EMA-in-context** feature evaluation.
- **Wall-clock** latency.

### Staged evaluation (proposed)

| Stage | What runs | Machine | What is evaluated |
|-------|------------|---------|-------------------|
| **1** | NIR buckets (e.g. A/B/D) on labeled stills | Dev | Before/after NIR metrics, per ENV class |
| **2** | Stage 1 + **synthetic** thermal stub (zeros or RPi-captured `.npy`) | Dev | Feature / ML coherence vs real thermal (careful) |
| **3** | Full pipeline + `session_*.json` | RPi | End-to-end timing, ML confidence, qualitative capture |

Thesis can chain: *Stage 1 optical stats* → *Stage 2 feature coherence* → *Stage 3 field behavior* (with honest limits).

---

## (D) Recommended evaluation protocol (abbreviated)

### Setup

1. **Freeze** the image set before any sweep. Consider **70/15/15** resplit with `--seed 42` if only train/test exist; **test** never touched until final report.
2. **Document manifest** fields: `dataset_name`, `version`, `nir_channel`, `thermal`, `label_source`, `n_per_class`, `seed`. Lock dataset commit; cross-reference `docs/PIPELINE_EVIDENCE_REGISTER.md` as appropriate.
3. **“Before”:** resized input at enhancer working resolution (e.g. 320×240) — same as `HybridNIREnhancer` work size; do not compare at full-res raw vs small-res enhanced.
4. **“After”:** output of the same bucket at **same** resolution and ROI.

### Metrics — four cells (2×2), not a single score

| Type | Metric | Rationale / repo |
|------|--------|-------------------|
| Global contrast | `log_rms_contrast` | No NSS; dynamic range. **Yes** — `sweep_clahe_clip.py` |
| Saturation | `pct_saturated_new`, `pct_crushed_new` | Direct penalty. **Yes** |
| Local sharpness | Laplacian variance | Detail; add ~3 lines if missing |
| ML coherence | Per-class RF confidence raw vs enhanced | Training distribution; needs careful split / cold-start caveats |

Stratify by **label**; do not report one aggregate IQA. **Red flag:** mean confidence **drops** after enhancement for a class.

### Splits (conceptual)

- `nir_eval_val.jsonl` / `nir_eval_test.jsonl` should come from **image-backed** still datasets — **not** from `from_logs_*.jsonl` alone, if those rows lack linked images for IQA.

### Tuning rules (if any sweep)

1. **Bounds** before running (e.g. CLAHE clip [0.5, 8.0], `detail_strength` [0, 0.5]).
2. **Multi-criterion** objective: e.g. `log_rms_contrast` ↑ with `pct_saturated` / `pct_crushed` under thresholds; never a single scalar.
3. **Tune on val only**; **one** frozen run on test; **no** retune after seeing test.
4. **Qualitative** A/B appendix (e.g. 6 images: 2 per major night class, 1 fog, 1 transition).

### What the thesis will *not* claim (template)

> The offline IQA evaluation shows that NIR enhancement buckets improve **stated optical statistics** on a labeled still-image set. This does **not** demonstrate: (a) better **detection**, (b) benefit of **thermal fusion**, (c) **real-time** RPi performance, or (d) generalization outside the evaluation set. **Field** `session_*.json` remains authoritative for **timing** and real-time behavior.

---

## (E) Open questions

- **E1 — Dataset domain:** DarkFace / ExDark are **visible** low-light, not true IMX290 NIR. Will any images come from **real** `fusion_captures/` RPi NIR? If yes, add a **source adapter** in `offline_pipeline` and state spectral limits.
- **E2 — Fog / D bucket:** DCP dehazing — CLAHE-style metrics may be wrong; transmission / paired haze–clear if available; else **proxy-only** and state limitation.
- **E3 — RGB vs NIR in weather datasets:** `HybridNIREnhancer` on daytime RGB may look worse to humans even if contrast ↑ — separate **by `nir_channel` / source** in tables.
- **E4 — Stopping rule for sweep:** Pre-register (e.g. “max `log_rms_contrast` with `pct_saturated` &lt; 0.02”) **before** running — avoid post-hoc parameter picking.
- **E5 — EMA cold-start vs production:** `nir_blue_mean_ema` = 0.0 on every still may bias `normal_night` vs `nir_night` — compare to production EMA distribution from RPi JSONL; document limitation.

---

*End of synthesis. Implement `tools/batch_nir_enhancer.py` only when the manifest and split protocol above are fixed.*
