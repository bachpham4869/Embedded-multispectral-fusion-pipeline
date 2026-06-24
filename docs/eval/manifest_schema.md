# Evaluation Manifest Schema & Stopping Rules

**Version:** 1.0 — 2026-04-27  
**Status:** Pre-registered before any Phase 3 sweeps are run. Do not change stopping rules after
seeing sweep results — that is HARKing (Hypothesizing After Results are Known).

---

## Batch Run Manifest Schema

Every batch evaluation run (Phase 3 offline IQA, Phase 4 ML ablation) must record these fields
in a sidecar JSON or CSV header row. This enables future reproducibility and evidence-register
traceability.

| Field | Type | Description |
|-------|------|-------------|
| `dataset_name` | string | Human-readable name (e.g. `"nir_val_2026-04-27"`) |
| `version` | string | Git commit SHA at run time (`git rev-parse HEAD`) |
| `nir_channel` | string | NIR source channel (e.g. `"green"`, `"gray"`) |
| `thermal_source` | string | `"real"` or `"dummy_zeros"` (offline-only) |
| `label_source` | string | `"by_label_dir"` or `"jsonl"` + path |
| `n_per_class` | dict[str, int] | Count of samples per ENV class |
| `seed` | int | RNG seed used for any shuffling (use `42` by default) |
| `pipeline_config_sha256` | string | SHA-256 of CONFIG allowlist (added Phase 1.2) |
| `run_mode` | string | `"still_image_cold_start"` for batch stills; `"live"` for session data |

---

## Phase 3 Sweep Protocol (Pre-registered)

### CLAHE clip sweep (Bucket B — `nir_nir_night_clahe`)

- **Script:** `tools/sweep_clahe_clip.py`
- **Input:** `data/eval/nir_val/` (val image set, NOT `from_logs_test.jsonl`)
- **Clip values:** `{0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0}`
- **Output:** `docs/tables/iqa/clahe_clip_sweep.csv`
- **Primary metric:** `log_rms_contrast` — log of RMS local contrast (higher = more texture)
- **Secondary metrics:** `pct_saturated_new` (fraction of pixels > 250/255), `pct_crushed_new`
  (fraction < 5/255)

**Stopping rule (pre-registered):**  
Select the highest clip value `c` such that `pct_saturated_new(c) < 0.02` (2% saturation cap).
If no clip value meets this constraint, flag as "all clips over-saturate" and report the
clip minimizing `pct_saturated_new`. Do not select by `log_rms_contrast` alone.

### Batch NIR enhancer sweep (Buckets A, B, D)

- **Script:** `tools/batch_nir_enhancer.py` (Phase 3 deliverable)
- **MVP buckets:** A (`HybridNIREnhancer`), B (`nir_nir_night_clahe`), D (`nir_dehaze_lite`)
- **Excluded buckets:** C (day-class, not core thesis), E (requires frame sequence — excluded from
  stills), F (passthrough, trivially unchanged)
- **Output:** `docs/tables/iqa/batch_nir_iqa.csv` — stratified by `env_class`
- **Metrics per row:** `log_rms_contrast`, `pct_saturated_new`, `pct_crushed_new`, `hist_entropy`,
  `local_contrast_std`

**HybridNIREnhancer statefulness note:**  
`HybridNIREnhancer` (`nir_pipeline.py:406`) increments `frame_count` per call and gates the heavy
path at `frame_count % update_rate == 0`. For still-image batch processing, the tool MUST:
1. Call `enhancer.reset()` (`nir_pipeline.py:324`) before each image.
2. Construct with `update_rate=1` to always run the heavy path.
3. Label all output with `run_mode = "still_image_cold_start_mode"` — this mode does NOT replicate
   live-video temporal EMA behavior, which is a documented scope limitation (R3 in synthesis doc).

### Detail strength sweep (HybridNIREnhancer, Bucket A)

- **Range:** `detail_strength` ∈ {0.0, 0.1, 0.2, 0.3, 0.5}
- **Same stopping rule:** max `detail_strength` s.t. `pct_saturated_new < 0.02`.

---

## Phase 4 ML Ablation Protocol (Pre-registered)

### 9-feature vs 12-feature model

- **Training sets:** Both models trained from `data/training/from_logs_train_a.jsonl` (same merged
  dataset, same seed=42).
- **Feature sets:** `FEATURE_SET_OPTICAL_ONLY` (12) and `FEATURE_SET_OPTICAL_9` (9, drops
  `hour_of_day_*` and `prev_env_class` — see `feature_schema.py`).
- **Evaluation:** One-shot against `data/training/from_logs_test.jsonl` — touched ONCE per model,
  after all training decisions are frozen.
- **Primary comparison metrics:** `macro_f1_night`, `abstention_rate`, `ece_night_clear`,
  `ece_normal_night`.
- **Defensible narrative:** If 9-feature model is within 0.01 F1 of 12-feature: "simpler model
  sufficient, temporal features carry no significant weight." If 12 wins: "temporal features
  (`hour_of_day_*`, `prev_env_class`) contribute X% to night-class F1."

---

## What this thesis does NOT claim (pre-registered)

*(Aligned with [`docs/PIPELINE_EVIDENCE_REGISTER.md`](../PIPELINE_EVIDENCE_REGISTER.md) § “What this thesis does NOT claim” — included here for evaluator reference.)*

1. The NIR enhancement parameters are globally optimal — they are hand-tuned starting points
   with a CLAHE sweep providing evidence for the Bucket B clip range.
2. Still-image IQA results generalize to live-video temporal behavior — the EMA and temporal
   median paths produce different outputs when operating on a frame sequence.
3. BRISQUE is a primary metric — it is an exploratory column only (Risk R1: BRISQUE assumes
   natural image statistics not satisfied by NIR night imagery).
4. The 9-vs-12 ablation is exhaustive — it tests one specific feature subset defined in
   `feature_schema.py:FEATURE_SET_OPTICAL_9`; other ablations are out of scope.
5. Phase 2 RPi timing results are attributable to individual optimizations — changes in Phase 1
   are deployed as a bundle; per-knob attribution requires separate microbench
   (`tools/microbench_morph.py` covers morphology only).
