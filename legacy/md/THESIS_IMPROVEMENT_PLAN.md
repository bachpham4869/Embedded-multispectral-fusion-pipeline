# SmartBinocular Thesis Improvement — Full Execution Plan (Night-Vision Scope)

This document is the **canonical** roadmap: it merges the original multi-phase research plan (Phases 1–3) with **Phase 3b** (real-time pipeline optimization on RPi4). **Anything listed after 3b is not a fixed commitment**—see [Post-3b review and optional follow-on](#post-3b-review-and-optional-follow-on).

**Agent synthesis (methodology, not a substitute for this roadmap):** [`OFFLINE_IQA_EVALUATION_AGENT_SYNTHESIS.md`](OFFLINE_IQA_EVALUATION_AGENT_SYNTHESIS.md) — still-image NIR IQA, BRISQUE caveats, splits, and what *not* to claim; aligns with **P2-A** (BRISQUE) and **P3-B** (night-bucket / quality tables) *when* you implement the offline batch. Related mentor notes (ideas, not approved): [`THESIS_MENTOR_REVIEW_SYNTHESIS.md`](THESIS_MENTOR_REVIEW_SYNTHESIS.md).

---

## Instructions for the implementing agent (read first)

1. **Continue** from the current tree: re-read `src/smartbinocular/main.py` (fusion + blend), `display_pipeline.py`, `metrics.py`, `config.py` (`RPI_THROUGHPUT_MAX_DEFAULTS`), and the latest `fusion_captures/metrics/session_*.json` on hand.
2. **Think** through further options: reduce `fusion_composite` cost (~70+ ms on RPi remains the main bottleneck), manage **S4 / luma** semantics when `display_luma_cap_glare_gate` skips full LAB on non-glare frames, bimodal `nir_bucket` cost, thermal dedup (keep `get_last_mono()`), homography quality from manifests.
3. **Clean up** code: remove dead comments, align config keys and comments, keep schema/version bumps consistent when session JSON changes.
4. **Review** your own diffs: fusion math invariant `NIR + (H − NIR) × m3`, tests (`pytest` targeted + full as appropriate).
5. **After pipeline cuts and FPS work are stable**: re-read [Post-3b review](#post-3b-review-and-optional-follow-on). **You decide** whether later phases (thesis chapters, extra perf, QA) still fit; if a track is obsolete, **revise this doc, clean up, and close the plan** rather than executing placeholder phases by rote.

**Do not** add unrelated refactors. Prefer measured wins on RPi for performance claims.

---

## Context

Two ULTRADEEP research reports were produced for the SmartBinocular thesis:

1. `docs/research_report_20260421_thesis_improvement.md` — codebase-grounded scientific gap analysis.
2. `research_report_20260417_edge_device_security.md` — security architecture analysis (mostly hardware; only software-side controls overlap with the thesis).

### Scope decision (user directive, 2026-04-21)

The thesis is a **night-vision / low-light / NIR** real-time binocular. Rare ENV classes — glare, backlight, transition — are **out of scope**: training data is too scarce for a defensible scientific claim. Dropped: recommendations targeting those classes as primary research. **In-scope evaluation surface:**

- **Primary:** `night_clear`, `normal_night`, `nir_night` (night-vision use case).
- **Secondary** (only when they affect night use): fog, rain at night.
- **Reference baseline only:** `normal_day` (control, not a research focus).

**Assessment:** The weakness is not core architecture (FrameCache pyramid, Kalman thermal background, six-bucket optical dispatch, `MLPosteriorEMA`, ENV compositor are defensible). The weakness is **evaluation rigor** on night classes: RF train vs CV gap, uncalibrated thresholds, feature importance holes, no quality metric on some optical buckets, no runtime homography validation, and gaps in test coverage. **Intended outcome:** a thesis that defends night-vision design with **quantitative** evidence—calibrated probabilities, per-bucket quality where scoped, per-stage RPi timing, Kalman characterisation, reproducible protocol—**without** new hardware or expanding rare auxiliary classes as claims.

---

## Selected findings (night-vision–aligned subset)

| # | Source finding | Kept? | Reason |
|---|----------------|-------|--------|
| 1 | RF overfitting + uncalibrated thresholds | ✅ | Calibrated metrics on night classes |
| 2 | Minority-class starvation (glare/backlight/transition) | ❌ | Out of scope; scarce data |
| 3 | No image-quality measurement on optical buckets | ✅ (scoped) | A, D, E, F (night/low-vis); not B/C as thesis focus when out of scope |
| 4 | Homography has no runtime drift metric | ✅ | First-order for sensor-fusion thesis |
| 5 | ML has no UQ | ✅ (scoped) | Night classes only for calibration + variance |
| 6 | Evaluation not reproducible | ✅ | Cheap logging, large gain |
| 7 | No per-stage latency vs ≤50 ms | ✅ | Embedded-systems claim |
| 8 | Test gaps in safety-critical modules | ✅ (scoped) | Thermal, NIR night paths, utils |
| Sec-1 | Model integrity (minisign / SHA-256) | ✅ | Reproducibility |
| Sec-2 | `uv sync --frozen` in deploy | ✅ | Doc/script |
| Sec-3 | JSONL field minimisation (GDPR Art. 5(1)(c)) | ✅ | Logging hygiene |
| Sec-* | LUKS, HSM, mTLS, ITAR, fail2ban, UFW, AIDE | ❌ | Hardware/ops; outside thesis |

---

## Phase 1 — Immediate, no-hardware actions (Week 1, ~5 working days)

Offline training (Mac) and code instrumentation.

### P1-A. Constrained RF + probability calibration

- **File:** `models/train_classifier.py` (both `RandomForestClassifier(...)` constructors).
- **Change:** `max_depth=20`, `min_samples_leaf=4`, `max_features="sqrt"`. After final fit, wrap with `CalibratedClassifierCV(..., method="isotonic", cv=5)`; save calibrated model in `save_model()`.
- **Why:** Tighten train vs CV gap; probabilities match reality for `ml_confidence_threshold` on night classes.

### P1-B. StratifiedKFold → session-aware temporal split

- **File:** `models/train_classifier.py` (~L374, L379).
- **Change:** `TimeSeriesSplit` by default; if `merged_logs_ml.jsonl` has `session_id`, use `GroupKFold` on `session_id`. Persist `cv_strategy` in model sidecar.
- **Risk:** CV accuracy may **drop**—that is the honest outcome.

### P1-C. Drop three zero-importance features (9-feature ablation)

- **Files:** `src/smartbinocular/feature_schema.py` — e.g. `FEATURE_SET_OPTICAL_9 = [f for f in FEATURE_SET_CORE if f not in {"hour_of_day_sin","hour_of_day_cos","prev_env_class"}]`; `models/train_classifier.py` new mode `optical_only_9`.
- **Action:** train both; ablation table (CV mean±std, train acc, ECE) for **night** metrics.

### P1-D. Reliability diagram + ECE (night classes)

- **File:** `models/train_classifier.py` (helper near `save_model`).
- **Change:** `calibration_curve` (10 bins), ECE; persist; PNGs for `night_clear`, `normal_night`, `nir_night` only.
- **Deps:** `matplotlib` in `[ml-tools]` if missing.

### P1-E. Session-level reproducibility in `metrics.py`

- `ThesisRunMetrics`: `frames_by_env`, `frames_by_bucket`; `record_frame(env_class=..., optical_bucket=...)`, surfaced in `finalize()`.
- `metrics_write_run_manifest()`: optional `model_path` → `ml_model_sha256_short`; optional H + quality → `homography_matrix`, `homography_reprojection_error_px`.
- `main.py`: pass env and bucket to `record_frame()`.

### P1-F. Homography quality metric in manifest

- **File:** `src/smartbinocular/utils.py` — `compute_homography_quality(H, thermal_shape, nir_shape) -> dict` (corner projection, `max_corner_drift_px`, `all_corners_within_nir`).
- **Wire:** `main.py` after load; `metrics_write_run_manifest(extra=...)`.

### P1-G. Tests (night-relevant)

- `tests/test_thermal_pipeline.py` — Kalman / fg-mask / shapes / `ThermalProcessor.process` contract.
- `tests/test_nir_pipeline.py` — night: HybridNIREnhancer, `nir_dehaze_lite`, RainTemporalMedian; skip B/C as per scope.
- `tests/test_utils.py` — `StreamSkewQualityGate` (if still present), `CaptureIntegrityChain`.

---

## Phase 2 — Medium-term software / algorithm (Weeks 2–3, ~5 working days)

### P2-A. BRISQUE on night-relevant buckets (A, D, E, F) — gated

- `nir_pipeline` / metrics; `iqa_logging_enabled` (default False); subsample 1/30; skip B/C for thesis table focus.
- **Caveat:** BRISQUE is natural-image biased—comparative only.
- **Deps:** `opencv-contrib-python` for `QualityBRISQUE_compute` if used.

### P2-B. Per-stage latency profiler

- `utils.py` — `StageProfiler`; wrap stages in `main.py`; `finalize()` → `stage_timing_ms`.
- **Action:** ~300 frames per `optimization_profile` on RPi; skip high_glare as primary if out of scope.

### P2-C. Night-class data audit (no rare-class aug)

- Inventory `data/training/from_logs_train.jsonl`; optional LLVIP / Dark Face / ExDark for `nir_night` balance; **document** spectral mismatch vs IMX290 (P3-C).

### P2-D. Epistemic variance (night)

- `ml_inference.py` — after `predict_proba`, variance across trees; path through `CalibratedClassifierCV` as in plan.

### P2-E. Model integrity

- `tools/sign_model.py`, `tools/verify_model.py`; `EnvClassifier._load` verifies before `joblib.load`.
- **Deps:** `minisign` on dev + device.

### P2-F. Deployment hardening (docs)

- `DEPLOY_HARDENING.md` — `uv sync --frozen`, JSONL whitelist, hashes; mark HW items out of scope.

---

## Phase 3b — Real-time pipeline optimization & metrics (RPi4) — *inserted roadmap*

**Status:** Largely **landed** in `main` (fusion hot path, blend, metrics). **Ongoing** until `fusion_composite` mean is acceptable and S4/UX trade-offs are signed off.

**Problem:** Per-stage RPi session JSON shows **`fusion_composite` ~70–75 ms** as the **dominant** cost; **`blend`** was ~40 ms before a **luma cap glare gate** under `luma_only`, then **~10–11 ms** in recent sessions (field evidence). `schema_version` advanced (e.g. `1.4-opt`) with skew stream metrics removed from live summary where applicable.

**Delivered / targeted in code (verify tree):**

- Pre-allocated float buffers for `NIR + (H−NIR)×m3` blend; `cv.convertScaleAbs`; uint8 `GaussianBlur` on fg mask where applicable; single BGRA `warpPerspective` for heat+mask; resize guard; `_did_fusion` fallback: one `cv.resize` to `display_size`.
- `display_luma_cap_glare_gate` in `RPI_THROUGHPUT_MAX_DEFAULTS` — skips full `display_luminance_cap_bgr` (BGR↔LAB) when no NIR/thermal **glare** flags; **review** S4 “always L-cap” product intent.
- `ThesisRunMetrics` without skew deques/keys in session summary; optional `StreamSkewQualityGate` kept in `utils` for future use.
- **Keep:** `get_last_mono()` and thermal reprocess **deduplication** — not “skew S6”.

**Open (post-3b):**

- Attack **`fusion_composite`** (work scale, warps, colormap path, allocations).
- Optional **cheaper S4** if gate skips full LAB (document trade-off).
- **Homography** manifest drift; **“pink” fusion** vs thermal-only = blend + NIR BGR, not colormap bug per se.
- Align session **`software`** field name with `smartbinocular` if low-risk.

**Reference session (field, schema 1.4-opt, example):** `fusion_captures/metrics/session_20260424-144637.json` — `fps_mean` ~6.97, `fusion_composite` ~72.4 ms, `blend` ~10.9 ms, `nir_bucket` bimodal, `rate_glare_nir` ~0.4.

**Verification (pipeline):** run on RPi, capture `session_*.json`, compare `stage_timing_ms` before/after changes; `pytest` for touched modules.

---

## Phase 3 — Thesis-level research enhancements (Weeks 4–6, ~7 working days)

Chapters / tables, **night-vision** only.

### P3-A. Calibration chapter (night classes)

- Reliability + ECE (P1-D); sweep `ml_confidence_threshold` ∈ {0.50, …, 0.75}; P/R on **night** predictions only.
- **Output:** Section *Probabilistic Calibration of the Night-Vision Environment Classifier*.

### P3-B. Night-bucket evaluation chapter

- BRISQUE × bucket (P2-A) × night ENV; cross with P2-B latency.
- **Output:** Table 2 (Quality × Night Bucket × Night ENV) + Figure 3 (Latency × Bucket × Profile on RPi).

### P3-C. Night domain-shift study

- Field night train ↔ public night test (LLVIP / Dark Face / ExDark) and reverse; **document** spectral / domain gap.

### P3-D. Kalman thermal background characterisation

- Synthetic step changes; frames-to-convergence vs old EMA warmup; night-typical contrast.
- **Output:** Convergence figure.

### P3-E. Reproducibility appendix

- Manifest schema (P1-E), CV strategy (P1-B), model registry (P2-E), `tests/` coverage.
- **Output:** Appendix *Experimental Reproducibility Protocol*.

---

## Mapping: actions → codebase (abbreviated)

| Action | Main files | Reuses |
|--------|------------|--------|
| P1-A RF + isotonic | `models/train_classifier.py` | sklearn |
| P1-B temporal CV | `models/train_classifier.py` | TimeSeriesSplit / GroupKFold |
| P1-C 9-feature | `feature_schema.py`, `train_classifier.py` | `FeatureRecord` |
| P1-D reliability | `train_classifier.py` | `calibration_curve`, matplotlib |
| P1-E session meta | `metrics.py`, `main.py` | `ThesisRunMetrics` |
| P1-F homography Q | `utils.py`, `main.py`, `metrics` manifest | NumPy |
| P1-G tests | `tests/test_thermal_pipeline.py`, `test_nir_pipeline.py`, `test_utils.py` | … |
| P2-A BRISQUE | `nir_pipeline.py`, `metrics` | opencv-contrib quality |
| P2-B profiler | `utils.py`, `main.py`, `metrics` | `time.perf_counter` |
| P2-C data | `tools/…`, `label_mapping.yaml` | existing tooling |
| P2-D epistemic | `ml_inference.py` | calibrated trees path |
| P2-E signing | `tools/sign_model.py`, `verify_model.py`, `ml_inference.py` | minisign |
| P2-F docs | `DEPLOY_HARDENING.md` | — |
| **3b pipeline** | `main.py`, `display_pipeline.py`, `config.py`, `metrics.py`, `hardware.py` | OpenCV, NumPy |
| P3-A–E | thesis text + scripts | P1/P2/3b outputs |

---

## Risks, assumptions, dependencies

**Risks:** R1 accuracy drops after P1-A/B (document as corrected baseline). R2 public NIR ≠ IMX290 (P3-C caveat). R3 `CalibratedClassifierCV` tree access path. R4 BRISQUE on NIR is comparative. R5 `session_id` in JSONL for GroupKFold—fallback in plan.

**Assumptions:** A1 Mac dev / training; A2 Master’s or regional conf; A3 fixed HW; A4 ~14k JSONL primary; A5 rare classes out of scope.

**Dependencies:** scikit-learn ≥1.0; opencv-contrib (BRISQUE); matplotlib; minisign (P2-E).

---

## Roadmap (weeks) — *adjust after Phase 3b sign-off*

| Week | Phase | Deliverables | Hardware? |
|------|--------|--------------|-----------|
| 1 | Phase 1 | Constrained calibrated RF; 9-feat ablation; reliability/ECE night; session manifest; homography Q; 3 test files | None |
| 2 | 2A–2B | BRISQUE A/D/E/F; stage profiler; RPi 3-profile run | RPi |
| 3 | 2C–2F | Night data audit; epistemic log; model signing; `DEPLOY_HARDENING.md` | None |
| **—** | **3b** | **Pipeline: fusion/blend/metrics; session JSON; field timing** | **RPi** |
| 4 | 3A | Calibration chapter draft + threshold sweep | None |
| 5 | 3B–3C | Night-bucket chapter; domain-shift study | None |
| 6 | 3D–3E | Kalman characterisation; reproducibility appendix | None |

**Total (original):** ~17 working days over 6 weeks, **zero new hardware**; **3b** adds **variable** RPi time until fusion budget and UX are accepted.

---

## Verification checklist (from original plan)

**Phase 1**

1. `python models/train_classifier.py --mode optical_only --dataset data/training/from_logs_train.jsonl --output models/baseline/rf_constrained.joblib` → expect train acc **&lt; 0.95** (post-constraint).
2. `optical_only_9` / `--eval-only` → ablation row.
3. Open `models/baseline/rf_constrained_reliability_night_clear.png` (and `_normal_night`, `_nir_night`) → ~diagonal curves.
4. `pytest tests/test_thermal_pipeline.py tests/test_nir_pipeline.py tests/test_utils.py -v` → green.
5. Run `python -m smartbinocular` (or Mac dry-run) → manifest has `frames_by_env`, `frames_by_bucket`, `ml_model_sha256_short`, homography extras if wired.

**Phase 2**

6. `iqa_logging_enabled=True`, 2 min night → `brisque_by_bucket` for A/D/E/F, not B/C.
7. StageProfiler, 300 frames RPi → `stage_timing_ms` for instrumented stages; target budget per thesis ≤50 ms **where claimed**.
8. After P2-C retrain — improved `nir_night` F1 (example expectation; validate on your split).
9. Tamper model file → `EnvClassifier` rejects load (when signing wired).

**Phase 3**

10. Each thesis chapter cites ≥1 figure/table from P1/2/3b instrumentation; **numbers restricted to night-vision class set** where the thesis claim applies.

---

## Post-3b review and optional follow-on

**Intent:** Once **algorithm reduction** and **pipeline optimization** have delivered acceptable **speed / FPS** on RPi, **do not** treat any “Phase 4+” list below as mandatory. The next owner (or agent) should **re-evaluate** what still matters for the thesis vs product.

### Mandatory review (lightweight)

1. **Re-baseline** using fresh RPi `fusion_captures/metrics/session_*.json` and, if useful, **update this doc** with current **mean ms** for `fusion_composite`, `blend`, `nir_bucket` (and note date / build).
2. **Decide** whether **S4 / luma** behaviour is acceptable (`display_luma_cap_glare_gate` vs always-on cap); if you change behaviour, update `display_pipeline.py` and any user-facing notes (README / deploy doc).
3. **Reconcile** P1–P3 / 3b **exit criteria** with reality: which thesis deliverables (calibration chapter, BRISQUE tables, signing, etc.) are **still in scope** after pipeline work.

### If further integration is **not** needed

If thesis or product direction **no longer requires** the full original Phases 1–3 stack (e.g. evaluation focus moved, or FPS work subsumed the research window):

- **Review** remaining items; **remove or archive** obsolete bullets in this file.
- **Clean up** code and docs (stale flags, duplicate plans, dead experiment hooks) with minimal diffs.
- **Close the plan**: add a short **“Closure”** subsection (date, reason, link to final session JSON or tag) or move narrative to `CHANGELOG.md` / thesis repo—so future agents do not re-open completed work.

### Optional ideas (non-binding — pick none, some, or rename)

*Only consider these if they still align with the thesis and product after the review above. They were previously labeled “Phase 4–6 (examples).”*

- Deeper **fusion_composite** analysis (sub-step timing, `fusion_warp_work_scale` trade-study, NEON only with measured proof—**do not overclaim** without device evidence).
- **Field QA:** homography `max_corner_drift_px`, fusion colour vs thermal-only, readability.
- **Thesis assets:** refresh figures for P3-A–E using **post-3b** latency numbers; model registry / signed bundles in CI **if** ML signing remains a deliverable.

### Progress note (2026-04-24)

**`fusion_composite` sub-profiling landed.** Seven `_fuse_sp` sub-stages are now instrumented inside the `fg_mask` branch (`fuse_nir_resize`, `fuse_thermal_prep`, `fuse_colormap`, `fuse_warp_prep`, `fuse_warp_perspective`, `fuse_blur_fg`, `fuse_blend_math`). `ThesisRunMetrics.fuse_stage_profiler` is wired; the next field session on RPi will write `fuse_stage_timing_ms` to the session JSON, providing per-sub-stage mean ± std. The `fusion_warp_work_scale` trade-study and any NEON-level changes remain **pending until RPi sub-stage baseline is in hand** — do not optimise before measuring.

S4 / luma gate trade-off is documented in `DEPLOY_HARDENING.md` § 8. No code change made; `display_luma_cap_glare_gate: True` remains the operational default.

*Rename, merge, or delete this subsection when P1–3 and 3b are either **done** or **explicitly descoped**.*

---

*Document generated for SmartBinocular; merge conflicts with other roadmaps should be resolved in favour of this file’s Phase **3b** section and the **Instructions for the implementing agent** block.*
