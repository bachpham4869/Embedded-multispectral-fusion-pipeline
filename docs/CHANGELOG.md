# Changelog

## [Unreleased] — Docs layout: tables/figures + legacy/md (2026-04-27)

### Changed

- **`docs/tables/`** grouped into **`timing/`**, **`iqa/`**, **`ml/`** with **[`tables/README.md`](tables/README.md)** index; tool defaults (`measure_stage_timing`, `sweep_clahe_clip`, `sweep_kalman_qr`, threshold sweeps) updated.
- **`docs/figures/`** grouped into **`iqa/`**, **`ml/reliability/`**, **`ml/ablation/`** with **[`figures/README.md`](figures/README.md)**; composed reliability PNGs live under `ml/reliability/`.
- **Thesis syntheses / improvement plan** moved to **`legacy/md/`** (root); **`docs/THESIS_OPEN_QUESTIONS.md`** removed — content merged into **`PIPELINE_EVIDENCE_REGISTER.md`** § *Thesis open questions (frozen scope)*.
- **Ablation reliability PNGs** moved from **`models/ablation/*.png`** → **`docs/figures/ml/ablation/`** (joblib/json stay in `models/ablation/`).
- **`legacy/py/`** added at repo root (placeholder README for future archived scripts).
- **Root-level `legacy/*.md`** consolidated under **`legacy/md/`**; **`legacy/*.py`** under **`legacy/py/`** (monolith + historical scripts). Index: **`legacy/README.md`**, **`legacy/md/README.md`**, **`legacy/py/README.md`**.

### Updated references

- Root **`README.md`**, **`docs/README.md`**, **`legacy/*`**, **`tools/README.md`**, **`models/README.md`**, tests reading sweep CSVs, **`config.py`** comment, **`nir_pipeline.py`** microbench path.

---

## [Unreleased] — Docs cleanup + baseline folder trim (2026-04-25)

### Added

- **`docs/OFFLINE_IQA_EVALUATION_AGENT_SYNTHESIS.md`** — Agent synthesis: offline NIR / IQA batch evaluation (risks R1–R7, repo map, staged eval, protocol, open questions E1–E5); supports thesis sections tied to P2-A / P3-B.

### Moved to `legacy/`

- **`docs/DEPLOY_ML_MODEL.md`**, **`docs/DEPLOY_HARDENING.md`**, **`docs/RPi_FOLLOWUP_WORK.md`** → **`legacy/`** (field RPi test campaign done; runbook tích cực: **`docs/RPi_RUNBOOK.md`**). Cập nhật **`README.md`**, **`docs/README.md`**, **`docs/PIPELINE_EVIDENCE_REGISTER.md`**, **`docs/THESIS_IMPROVEMENT_PLAN.md`**, **`docs/tables/ML_GATE_RATIONALE.md`**, **`legacy/README.md`**, **`legacy/ML_PER_CLASS_CONFIDENCE_PLAN.md`**, **`docs/RPi_RUNBOOK.md`**.

### Changed

- **`docs/README.md`:** Index row for `OFFLINE_IQA…`; mentor row describes full file in `docs/` (draft, not only redirect); **`THESIS_IMPROVEMENT_PLAN.md`:** mentor link points to `docs/THESIS_MENTOR_REVIEW_SYNTHESIS.md`.
- **`THESIS_MENTOR_REVIEW_SYNTHESIS.md`:** Stays in **`docs/`** (undo 2026-04-26 move to `legacy/`) — internal agent notes, not approved; **`legacy/README.md`:** removed errant table row for that file.
- **`README.md`:** Links to `PIPELINE_EVIDENCE_REGISTER.md`, RPi runbook + archived follow-up; **design vs measured** latency called out (see `docs/tables/stage_timing_summary.md`).
- **`docs/README.md`:** `ML_PER_CLASS_CONFIDENCE_PLAN` moved to `legacy/`; index rows for evidence register and RPi follow-up.
- **`legacy/README.md`:** Row for archived `ML_PER_CLASS_CONFIDENCE_PLAN.md`.
- **`tools/threshold_sweep.py` / `tools/secondary_threshold_sweep.py`:** default bundle → `models/production/env_classifier.joblib`.
- **`tools/compose_reliability_figure.py`:** default PNG source → `docs/figures/ml_reliability_sources/`.
- **`models/model_registry.json`:** only `production/env_classifier.joblib` entry (duplicate baseline key removed).
- **`docs/tables/ece_all_env_classes.md`:** links to legacy plan + `PIPELINE_EVIDENCE_REGISTER.md`.
- **`legacy/DEPLOY_HARDENING.md`** (lúc đó ở `docs/`), **`CLAUDE.md`**, **`docs/research_report_20260421_thesis_improvement.md`**, **`models/README.md`**, **`tools/sign_model.py`:** paths updated away from removed `rf_from_logs_baseline` artifacts.

### Added

- **`docs/RPi_FOLLOWUP_WORK.md`** (sau này → **`legacy/RPi_FOLLOWUP_WORK.md`**) — checklist of work to perform on the physical RPi (timing, deploy, sweeps).
- **`docs/figures/ml_reliability_sources/README.md`** + copied `rf_phase1_retrain_optical12_reliability_*.png` (per-class reliability sources for `compose_reliability_figure`).

### Removed (local artifacts)

- From **`models/baseline/`:** `rf_from_logs_baseline.*`, `rf_phase1_retrain_optical9_ablation.*`, `rf_phase1_retrain_optical12.joblib`, and all `*_reliability_*.png` (superseded by `docs/figures/ml_reliability_sources/`). **Kept:** `rf_phase1_retrain_optical12.json` (ECE source for `docs/tables/ece_*.md`).

### Renamed

- **`docs/ML_PER_CLASS_CONFIDENCE_PLAN.md`** → **`legacy/ML_PER_CLASS_CONFIDENCE_PLAN.md`** (plan §7 complete; thesis traceability lives in `PIPELINE_EVIDENCE_REGISTER.md`).

---

## [Unreleased] — Pipeline Evidence Register + Thesis Audit (2026-04-25)

### Added

- **`docs/PIPELINE_EVIDENCE_REGISTER.md`** — consolidated thesis-defense traceability document.
  Parts A–D: pipeline map (data flow, thread model, frame budget), per-stage block tables,
  algorithm literature-vs-implementation gaps, and 12 grouped parameter tables. Every numeric
  threshold is tagged CIT / SWEEP / SIDECAR / TEST / DOC / UNVERIFIED. Explicit gap documented
  for fusion (addWeighted vs Laplacian pyramid [65][33]).
- **`tools/measure_stage_timing.py`** — reads `fusion_captures/metrics/session_*.json`, emits
  `docs/tables/stage_timing_summary.csv` + `.md` with per-stage mean/p50/p95/p99 ms. Exits
  non-zero with a clear message if no session files are found.
- **`tools/sweep_clahe_clip.py`** — proxy metrics (log_rms_contrast, pct_saturated, pct_crushed)
  over committed NIR sample frames; emits `docs/tables/clahe_clip_sweep.csv`.
- **`tools/sweep_kalman_qr.py`** — documented stub; sweeps Q/R on a committed thermal sequence;
  emits `docs/tables/kalman_qr_sweep.csv`. Exits non-zero if no sequence is staged.
- **`tests/test_threshold_sweep_consistency.py`** — consistency check: τ₁=0.62 CSV row vs
  `ml_gate_reference` sidecar field (tolerance ±1e-4, column-name keyed).
- **`tests/test_secondary_sweep_consistency.py`** — same shape for secondary sweep τ₂=0.20.
- **`tests/test_model_registry_integrity.py`** — raw-bytes SHA-256 of production joblib vs
  `models/model_registry.json`; skips cleanly if joblib is gitignored.
- **`tests/test_sidecar_ece.py`** — asserts ECE present for all 9 ENV_CLASSES; policy bounds:
  global ≤0.15, three night classes ≤0.10 (conservative for future retrain drift).

### Changed

- **`LINK.md` §IV/§V** — cross-index entries retargeted from the non-existent
  `docs/ALGORITHMS_DETAILED.md` and `docs/ALGORITHM_INTEGRATION_TIERS.md` to
  `docs/PIPELINE_EVIDENCE_REGISTER.md` Parts B–D. A consolidation note explains
  the change above each table.

---

## [Unreleased] — ENV Optical Refactor v4 + ML-Driven Routing

### Documentation (week of 2026-04-21)

- **`LINK.md`:** Rev. 5 — IEEE references **[66]–[77]** from `docs/research_report_20260421_thesis_improvement.md`; new **§IX** cross-index (calibration, BRISQUE/NIQE, SMOTE, KAIST, Kalman tutorial, fusion survey, RPi OpenCV notes). Weekly index row points to `weekly_report/week_3.md`.
- **`docs/README.md`:** Rows for `DEPLOY_HARDENING.md`, `research_report_20260421_thesis_improvement.md`, `weekly_report/week_3.md`; THESIS blurb mentions `fuse_stage_timing_ms`.
- **`README.md` (root):** Topic table — thesis plan, `LINK.md`, deploy hardening, `week_3` report; ML section links thesis improvement report; **S4** — `display_luma_cap_glare_gate` + `DEPLOY_HARDENING` §8; session JSON — `fuse_stage_timing_ms`.
- **`weekly_report/week_3.md`:** New — sprint summary (fusion sub-profiling, docs, next steps on RPi).

### Summary

Replaces the Schmitt-trigger-based NIR bypass with a purpose-built per-environment optical
dispatch table (Buckets A–F). ML inference labels now drive optical routing in `auto_rule`
mode, with the rule-based heuristic as fallback. `HybridNIREnhancer` is called only for
the two night classes; all other ENV classes get cheaper, domain-tuned processors.

---

### Removed

- **Schmitt trigger** (`NIR_BRIGHT_RAW_ON`, `NIR_DIM_ENHANCE_ON`, `nir_hyst_initialized`,
  `nir_use_raw_auto` flip logic) removed from `main.py`. The trigger was a dual-threshold
  brightness switch that bypassed NIR enhancement for bright frames; its role is now fully
  handled by `OPTICAL_BUCKET_DISPATCH` based on the stabilized ENV class.

  > **Behavioral note:** In `env_mode="off"` (the default), `_stable_env_class` stays at
  > `"normal_night"` → Bucket A → `HybridNIREnhancer` runs every frame. The old Schmitt
  > trigger would have skipped enhancement when NIR brightness exceeded 30.0. To restore
  > adaptive routing, enable `env_mode: auto_rule`.

- BGR→grayscale "night mono" conversion path inside `HybridNIREnhancer.process()` removed.
  Processing is now entirely on BGR via LAB L-channel. No `COLOR_BGR2GRAY / GRAY2BGR` round
  trip.

- Duplicate tag→preset heuristic branches in `main.py` ENV block replaced by a single call
  to `auto_rule_preset_to_env_class()`.

---

### Added

#### `src/smartbinocular/nir_pipeline.py`

- **`OPTICAL_BUCKET_DISPATCH: Dict[str, str]`** — maps all 9 `ENV_CLASSES` + `"default"` to
  bucket letters A–F:

  | ENV class | Bucket | Function |
  |-----------|--------|----------|
  | `night_clear`, `normal_night` | A | `HybridNIREnhancer.process()` |
  | `nir_night` | B | `nir_nir_night_clahe()` |
  | `glare`, `backlight`, `normal_day` | C | `nir_anti_glare_bgr()` |
  | `fog` | D | `nir_dehaze_lite()` |
  | `rain` | E | `RainTemporalMedian.process()` |
  | `transition` | F | `nir_transition_blend()` |

- **`nir_nir_night_clahe(frame, clahe_clip_scale)`** — Bucket B: single CLAHE on LAB L
  channel at clip=3×scale with 4×4 tiles. ~1–2 ms on RPi4.

- **`nir_dehaze_lite(frame, omega)`** — Bucket D: Dark Channel Prior dehazing at 160×120
  downsample (DCP transmission at `omega=0.85`). ~4–6 ms on RPi4.

- **`RainTemporalMedian(n_frames)`** — Bucket E: N-frame ring-buffer pixel-wise median.
  Stateful processor with `reset()` called on mode switches. ~4–6 ms on RPi4.

- **`nir_transition_blend(frame, nir_enhancer, nir_b_ema_norm)`** — Bucket F: weighted blend
  of Bucket A and Bucket C by brightness EMA (lo=0.15, hi=0.45). Zero-cost when brightness
  is in the night range; smoothly cross-fades toward glare processing as scene brightens.

#### `src/smartbinocular/env_presets.py`

- **7 ENV_CLASS-aligned presets** added to `ENV_PRESETS`: `night_clear`, `normal_night`,
  `normal_day`, `glare`, `nir_night`, `rain`, `transition`. These keys can be passed
  directly to `EnvPresetController.update()` and `ENV_PRESETS` lookup without remapping.

- **`_TRANSITION_HYSTERESIS: Dict[str, Tuple[int, int]]`** — asymmetric hysteresis table.
  `"glare": (2, 20)` means 2 frames to enter glare, 20 frames to leave it (prevents flicker
  from brief specular reflections). Default hysteresis (`hysteresis_frames=18`) applies to
  all other transitions.

- **`auto_rule_preset_to_env_class(preset_name) -> str`** — maps legacy auto_rule preset
  names (e.g., `"glare_heavy"`, `"night"`, `"haze"`) to canonical `ENV_CLASS` strings.
  Unknown names fall back to `"normal_night"`.

- **Fixed import bug**: `from config import _VALID_OPT_OVERRIDES_KEYS` →
  `from smartbinocular.config import _VALID_OPT_OVERRIDES_KEYS` (src-layout requires
  absolute imports).

#### `src/smartbinocular/main.py`

- **`rain_processor = RainTemporalMedian(...)`** initialized alongside `nir_enhancer`;
  `rain_processor.reset()` called on mode switches.

- **`_stable_env_class: str`** — new frame-level tracking variable. Updated after each
  `env_controller.update()` call; normalized through `auto_rule_preset_to_env_class()` for
  manual-mode preset names that predate the ENV_CLASS taxonomy.

- **Bucket dispatch block** at the NIR processing site: single `OPTICAL_BUCKET_DISPATCH`
  lookup replaces the former `if nir_b_ema > NIR_BRIGHT_RAW_ON` / else chain.

- **ML-primary routing in `auto_rule` mode**: if `ML_INFERENCE_ENABLED` and
  `ml_confidence ≥ ml_confidence_threshold` (default 0.62), the ML label drives desired
  ENV class directly. Rule-based heuristic runs only when ML confidence is below threshold
  or ML is disabled.

#### `src/smartbinocular/config.py`

- Added to `_VALID_OPT_OVERRIDES_KEYS`: `"nir_enhancer_clahe_clip_scale"`,
  `"fog_dehaze_omega"`, `"rain_median_frames"`.
- Added to `CONFIG`: `"ml_confidence_threshold": 0.62`, `"fog_dehaze_omega": 0.85`,
  `"rain_median_frames": 3`.

#### `tests/test_optical_bucket_dispatch.py` (new file)

47 tests covering T040–T050:
- Dispatch table completeness and valid bucket keys
- Bucket A exclusivity for night classes; non-night classes excluded
- Bucket B CLAHE shape/dtype and dark-frame brightening
- Bucket C passthrough when scene not saturated
- Bucket D shape/dtype preservation
- Bucket E ring-buffer behavior (before-full passthrough, median after N frames, reset)
- Asymmetric hysteresis boundaries (glare onset=2, decay=20, default=18)
- `auto_rule_preset_to_env_class()` full old-preset coverage
- ENV_CLASS names present in `ENV_PRESETS`

---

### Fixed

- `ThesisRunMetrics` construction in `main.py` referenced `NIR_BRIGHT_RAW_ON` and
  `NIR_DIM_ENHANCE_ON` after those constants were deleted by the Schmitt trigger removal.
  Fixed by inlining `cfg.get("nir_schmitt_raw_on", 30.0)` / `cfg.get("nir_schmitt_dim_on", 18.0)`
  directly (values preserved in CONFIG for JSONL backward compat).

- `_stable_env_class` normalization: when `env_mode="manual"` uses old preset names
  (e.g., `"night"`), `_stable_env_class` is normalized through
  `auto_rule_preset_to_env_class()` before bucket lookup to avoid silent Bucket A fallback.

---

### Known / Out-of-Scope

- `test_ml_inference.py` — 3 pre-existing failures due to `scipy`/`sklearn` binary
  incompatibility with Python 3.14.0a7 (`_PyUnstable_Object_IsUniquelyReferenced`). Not
  introduced by this refactor; will resolve when scipy ships a 3.14-compatible wheel.

- `nir_use_raw_auto` parameter is retained in `write_capture_meta()` and JSONL exports
  (always `False`) for backward compatibility with capture metadata consumers.

---

## [Unreleased] — v4 Research Backlog (Tasks 1–5) — 2026-04-17

Closes Findings 9–12 of `docs/research_report_20260415_env_policy_branching_optical_v4.md`.
See the plan at `.claude/plans/sequential-stargazing-cloud.md` for the full gap table.

### Task 1 — Finding 11 override key gaps closed

- **`src/smartbinocular/config.py`** — `_VALID_OPT_OVERRIDES_KEYS` extended with
  `nir_enhancer_detail_strength`, `thermal_bilateral_d`, `thermal_bilateral_sigma_color`,
  `thermal_bilateral_sigma_space`. Preset `opt_overrides` can now tune NIR detail strength
  and thermal bilateral σ without a `WARN: unknown opt_overrides keys` log.

- **`src/smartbinocular/thermal_pipeline.py`** — Added `ThermalProcessor.update_runtime_params(opt_cfg)`
  to apply bilateral filter params at runtime (construction-only before). Called by `main.py`
  on preset-stable change only — not per frame.

- **`src/smartbinocular/nir_pipeline.py`** — Added `HybridNIREnhancer.update_runtime_params(opt_cfg)`
  to apply `nir_enhancer_detail_strength` and `nir_enhancer_clahe_clip_scale` at runtime.
  `clahe_levels` is rebuilt when `clahe_clip_scale` changes.

- **`src/smartbinocular/main.py`** — Both `update_runtime_params()` calls wired in the
  `if stable != prev_env_stable:` block (at most once per stable-change event; never per frame).

- **`tests/test_runtime_param_updates.py`** (new, 13 tests) — covers all 4 new override keys,
  odd/even `d` enforcement, CLAHE rebuild, unknown keys ignored.

### Task 2 — Extended `_TRANSITION_HYSTERESIS` for rain / transition / fog

- **`src/smartbinocular/env_presets.py`** — `_TRANSITION_HYSTERESIS` extended:
  ```python
  "rain":       (10, 25),   # 10 frames onset, 25 frames decay
  "transition": (12, 30),   # slow onset for dawn/dusk ambiguity
  "fog":        (6, 18),    # moderate onset, default decay
  ```
  `"glare"` / `"glare_heavy"` (2, 20) unchanged.

- **`tests/test_env_preset_hysteresis.py`** (new, 20 tests) — parametrized table and
  onset/decay frame-count proofs for all 5 entries; key-match preset name tests.

### Task 3 — Compositor illumination-primary + weather-hint-overlay policy

**Product policy (codified):** When illumination (night axis or `normal_day`) is top-1,
`env_class` stays with illumination. Interference (fog / rain / glare) as top-2 is applied
as a small `apply_secondary_hint` overlay only — never replacing `env_class`. Interference
buckets are used ONLY when interference is actually top-1.

- **`src/smartbinocular/env_presets.py`** — `compose_env_from_ml_top2` updated:
  - Docstring lists all 10 rules in priority order plus the invariant.
  - **Rule 6 (new):** night top-1 + rain top-2 → `(class_1, "rain")`
  - **Rule 7 (new):** day top-1 + {fog, rain, glare} top-2 → `(class_1, class_2)` (day stays env)
  - **Rule 9 (new):** rain top-1 + night top-2 → `(class_1, class_2)` (rain stays env)
  - Existing Rule 5 (night+fog → (night, "fog")) and Rule 8 (fog+night) unchanged.

- **`tests/test_env_compositor.py`** — expanded to 40 tests:
  - TC14–TC16: new rules 6/7/9
  - Full compound rule table (parametrized)
  - `test_night_top1_never_becomes_fog_or_rain_env_class` — invariant guard
  - `test_day_top1_never_becomes_interference_env_class` — symmetric day guard

### Task 4 — Layer 1: EMA on RF full posterior vector

- **`src/smartbinocular/ml_inference.py`** — Added:
  - **`_resolve_asym(asym_str, classes)`** — resolves string class-name keys to ints,
    filters to `rf.classes_` intersection; drops unknown keys with DEBUG log.
  - **`MLPosteriorEMA`** — per-class EMA on the full RF probability vector. General
    α=0.55; asymmetric α for glare (α_up, α_down from CONFIG); renormalized to Σ=1 after
    each update. Thread-owned; not thread-safe by design.
  - `MLInferenceThread` now constructs `MLPosteriorEMA` from `rf.classes_` on `run()`,
    calls `ema.update(proba_dict)` before building `MLTop2` from the smoothed distribution.

- **`src/smartbinocular/config.py`** — added:
  ```python
  "ml_posterior_ema_alpha": 0.55,
  "ml_posterior_ema_asym": {"glare": [0.85, 0.45]},
  ```

- **`src/smartbinocular/main.py`** — `MLInferenceThread` construction passes `ema_alpha`
  and `ema_asym` from CONFIG.

- **`tests/test_ml_inference.py`** — 6 new EMA tests (T031–T036):
  symmetric smoothing with renormalization, asymmetric glare rise/decay (with slow-decay
  invariant vs symmetric α), top-1/top-2 consistency from smoothed dict, first-call seed,
  α=1.0 passthrough, and glare-absent-from-bundle graceful fallback.

### Task 5 — `nir_blue_mean_ema` as feature #12

> **Naming note:** `nir_blue_mean_ema` (B-channel mean EMA over `nir_small_bgr`) is
> **distinct** from `main.py`'s existing `nir_b_ema` (brightness scalar EMA, coefficient
> `NIR_B_EMA`). Both coexist; different fields, different semantics.

- **`src/smartbinocular/feature_schema.py`**:
  - `FEATURE_SET_CORE` extended with `"nir_blue_mean_ema"` at index 11 (0-based).
  - `FeatureRecord` field `nir_blue_mean_ema: float = 0.0` added in the CORE block.
  - `FEATURE_SET_OPTICAL_ONLY` is now 12 features.

- **`src/smartbinocular/feature_extractor.py`**:
  - `FeatureExtractor` computes `nir_blue_mean_ema` per frame (B-channel mean of
    `nir_small_bgr`, EMA α=0.55; seeded on first frame).
  - `reset_sequence()` resets `_nir_blue_mean_ema` so EMA does not bleed across sessions.

- **`src/smartbinocular/ml_inference.py`** — mismatch log updated to name expected length
  (12) and `nir_blue_mean_ema` for easier diagnosis.

- **`models/train_classifier.py`** — comment updated to 12 features; assertion added:
  `assert len(feature_set) == 12` with actionable message pointing to
  `tools/offline_pipeline.py` for JSONL regeneration.

- **`tests/test_feature_schema.py`** (new, 10 tests, FS01–FS07):
  feature count, last-feature name, naming collision guard (`nir_b_ema` NOT in set),
  array shape, round-trip, None raises ValueError (C8), old-record default 0.0,
  11-feature bundle rejection regression.

**Deploy order (RPi):**
1. Ship Task 5 code (`nir_blue_mean_ema` schema bump).
2. Regenerate training JSONL: `python tools/offline_pipeline.py ...`.
3. Retrain: `python models/train_classifier.py ...`.
4. `rsync` new 12-feature bundle to RPi; set `ML_MODEL_PATH`.
5. Until step 4 lands, Pi runs rule-based (safe degrade — not broken).

**Breaking change:** Existing 11-feature RF bundles are rejected by `EnvClassifier._load`
(feature_set mismatch) → `available=False` → compositor returns `(None, None)` → `auto_rule`
fallback. See `tests/test_feature_schema.py::test_feature_set_mismatch_disables_classifier`.
